//! Opt-in wire-message logging. Observe body frames without buffering delivery.
use axum::{
    body::{Body, Bytes},
    extract::Request,
    middleware::Next,
    response::Response,
};
use http_body::{Body as HttpBody, Frame, SizeHint};
use std::{
    pin::Pin,
    sync::atomic::{AtomicU64, Ordering},
    task::{Context, Poll},
};

const TARGET: &str = "catena_models::server::protocol";
const MAX_CAPTURE: usize = 8 * 1024 * 1024;
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

pub async fn log_messages(mut request: Request, next: Next) -> Response {
    if !tracing::enabled!(target: TARGET, tracing::Level::DEBUG) {
        return next.run(request).await;
    }
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    tracing::debug!(target: TARGET, request_id=id, direction="incoming", method=%request.method(), path=%request.uri().path(), "HTTP request");
    let body = std::mem::replace(request.body_mut(), Body::empty());
    *request.body_mut() = Body::new(LoggedBody::new(body, id, "incoming", false));
    let mut response = next.run(request).await;
    tracing::debug!(target: TARGET, request_id=id, direction="outgoing", status=response.status().as_u16(), "HTTP response");
    let sse = response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.starts_with("text/event-stream"));
    let body = std::mem::replace(response.body_mut(), Body::empty());
    *response.body_mut() = Body::new(LoggedBody::new(body, id, "outgoing", sse));
    response
}

struct LoggedBody {
    inner: Body,
    id: u64,
    direction: &'static str,
    sse: bool,
    line_has_data: bool,
    pending: Vec<u8>,
    truncated: bool,
    finished: bool,
}
impl LoggedBody {
    fn new(inner: Body, id: u64, direction: &'static str, sse: bool) -> Self {
        Self {
            inner,
            id,
            direction,
            sse,
            line_has_data: false,
            pending: Vec::new(),
            truncated: false,
            finished: false,
        }
    }
    fn record(&mut self, bytes: &[u8]) {
        if self.sse {
            // An SSE event can cross HTTP frames, or share a frame with others.
            for chunk in bytes.split_inclusive(|&b| b == b'\n') {
                self.append(chunk);
                // Track separators independently of the bounded capture, so
                // even a clipped event cannot swallow the following events.
                self.line_has_data |= chunk.iter().any(|&b| b != b'\n' && b != b'\r');
                if chunk.ends_with(b"\n") {
                    if !self.line_has_data {
                        self.emit(false);
                    }
                    self.line_has_data = false;
                }
            }
        } else {
            self.append(bytes);
        }
    }
    fn append(&mut self, bytes: &[u8]) {
        let count = bytes.len().min(MAX_CAPTURE - self.pending.len());
        self.pending.extend_from_slice(&bytes[..count]);
        self.truncated |= count != bytes.len();
    }
    fn emit(&mut self, partial: bool) {
        tracing::debug!(target: TARGET, request_id=self.id, direction=self.direction,
            format=if self.sse {"sse"} else {"body"}, partial,
            truncated=self.truncated, body=%String::from_utf8_lossy(&self.pending), "protocol message");
        self.pending.clear();
        self.truncated = false;
    }
    fn finish(&mut self, partial: bool) {
        if !self.finished {
            if !self.pending.is_empty() || !self.sse {
                self.emit(partial);
            }
            self.finished = true;
        }
    }
}
impl HttpBody for LoggedBody {
    type Data = Bytes;
    type Error = axum::Error;
    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, axum::Error>>> {
        let this = self.get_mut();
        match Pin::new(&mut this.inner).poll_frame(cx) {
            Poll::Ready(Some(Ok(frame))) => {
                if let Some(data) = frame.data_ref() {
                    this.record(data);
                }
                if this.inner.is_end_stream() {
                    this.finish(false);
                }
                Poll::Ready(Some(Ok(frame)))
            }
            Poll::Ready(None) => {
                this.finish(false);
                Poll::Ready(None)
            }
            Poll::Ready(Some(Err(error))) => {
                this.finish(true);
                tracing::debug!(target: TARGET, request_id=this.id,direction=this.direction,%error,"protocol body error");
                Poll::Ready(Some(Err(error)))
            }
            Poll::Pending => Poll::Pending,
        }
    }
    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }
    fn size_hint(&self) -> SizeHint {
        self.inner.size_hint()
    }
}
impl Drop for LoggedBody {
    fn drop(&mut self) {
        self.finish(!self.inner.is_end_stream());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::tests::LogWriter;
    use std::{
        convert::Infallible,
        sync::{Arc, Mutex},
    };
    use tracing::instrument::WithSubscriber;
    #[tokio::test]
    async fn split_sse_frames_are_forwarded_and_logged_as_complete_messages() {
        let captured = Arc::new(Mutex::new(Vec::new()));
        let writer = LogWriter(captured.clone());
        let subscriber = tracing_subscriber::fmt()
            .without_time()
            .with_ansi(false)
            .with_max_level(tracing::Level::DEBUG)
            .with_writer(move || writer.clone())
            .finish();
        let chunks = [
            "data: {",
            "\"text\":\"hi\"}",
            "\n",
            "\ndata: [DO",
            "NE]\n\n",
        ];
        let expected = chunks.concat();
        let inner = Body::from_stream(tokio_stream::iter(
            chunks.map(|s| Ok::<_, Infallible>(Bytes::from(s))),
        ));
        let body = Body::new(LoggedBody::new(inner, 42, "outgoing", true));
        let actual = axum::body::to_bytes(body, 1024)
            .with_subscriber(subscriber)
            .await
            .unwrap();
        assert_eq!(actual.as_ref(), expected.as_bytes());
        let logs = String::from_utf8(captured.lock().unwrap().clone()).unwrap();
        assert_eq!(logs.matches("protocol message").count(), 2);
        assert!(logs.contains("data: {\"text\":\"hi\"}"));
        assert!(logs.contains("data: [DONE]"));
    }
}
