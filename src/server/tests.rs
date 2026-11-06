use super::{
    api::{ServerState, router},
    engine::{self, Backend, Delta, Event, Failure, Generation, Job, Sink},
    tools::Call,
};
use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use serde_json::{Value, json};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use tower::ServiceExt;

struct Fake {
    calls: Arc<AtomicUsize>,
}
impl Backend for Fake {
    fn generate(&mut self, request: Generation, sink: &Sink) -> Result<(), Failure> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        if request.max_tokens == Some(999) {
            return Err(Failure::invalid("context overflow"));
        }
        sink.send(Event::Started);
        if request.max_tokens == Some(998) {
            return Err(Failure::internal("GPU failed"));
        }
        if request.thinking {
            sink.send(Event::Delta(Delta::Reasoning("reason".into())));
        }
        if !request.tools.is_empty() {
            sink.send(Event::Delta(Delta::Tool {
                index: 0,
                call: Call {
                    name: request.tools[0].function.name.clone(),
                    arguments: "{}".into(),
                },
            }));
        } else {
            sink.send(Event::Delta(Delta::Content("Hello 世界".into())));
        }
        sink.send(Event::Done {
            reason: if request.tools.is_empty() {
                "stop"
            } else {
                "tool_calls"
            },
            prompt_tokens: 4,
            completion_tokens: 2,
            reasoning_tokens: usize::from(request.thinking),
        });
        Ok(())
    }
}
fn post(value: Value, key: Option<&str>) -> Request<Body> {
    let mut b = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json");
    if let Some(key) = key {
        b = b.header("authorization", format!("Bearer {key}"));
    }
    b.body(Body::from(value.to_string())).unwrap()
}
fn base() -> Value {
    json!({"model":"test","messages":[{"role":"user","content":"Hi"}]})
}
async fn body(response: axum::response::Response) -> String {
    String::from_utf8(
        to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap()
            .to_vec(),
    )
    .unwrap()
}
fn events(body: &str) -> Vec<Value> {
    body.lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|line| *line != "[DONE]")
        .map(|s| serde_json::from_str(s).unwrap())
        .collect()
}
#[tokio::test]
async fn json_stream_tools_auth_and_errors() {
    let calls = Arc::new(AtomicUsize::new(0));
    let clone = calls.clone();
    let worker = engine::start(
        move || Ok(Fake { calls: clone }),
        Arc::new(AtomicBool::new(false)),
    )
    .unwrap();
    let app = router(ServerState {
        model: Arc::from("test"),
        key: Some(Arc::from("secret")),
        jobs: worker.sender,
    });
    assert_eq!(
        app.clone()
            .oneshot(post(base(), None))
            .await
            .unwrap()
            .status(),
        StatusCode::UNAUTHORIZED
    );
    for path in ["/health", "/v1/models"] {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri(path)
                    .header("authorization", "Bearer secret")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }
    let response = app
        .clone()
        .oneshot(post(base(), Some("secret")))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let json: Value = serde_json::from_str(&body(response).await).unwrap();
    assert_eq!(json["choices"][0]["message"]["content"], "Hello 世界");
    assert_eq!(json["usage"]["total_tokens"], 6);
    let mut streaming = base();
    streaming["stream"] = json!(true);
    streaming["stream_options"] = json!({"include_usage":true});
    let response = app
        .clone()
        .oneshot(post(streaming, Some("secret")))
        .await
        .unwrap();
    assert_eq!(response.headers()["content-type"], "text/event-stream");
    let raw = body(response).await;
    assert!(raw.ends_with("data: [DONE]\n\n"));
    let stream = events(&raw);
    assert_eq!(
        stream[1]["choices"][0]["delta"]["content"],
        json["choices"][0]["message"]["content"]
    );
    assert_eq!(stream[2]["choices"][0]["finish_reason"], "stop");
    assert_eq!(stream[3]["usage"], json["usage"]);
    let mut tools = base();
    tools["tools"] = json!([{"type":"function","function":{"name":"read"}}]);
    tools["chat_template_kwargs"] = json!({"enable_thinking":true});
    let response = app
        .clone()
        .oneshot(post(tools.clone(), Some("secret")))
        .await
        .unwrap();
    let result: Value = serde_json::from_str(&body(response).await).unwrap();
    assert_eq!(result["choices"][0]["finish_reason"], "tool_calls");
    assert_eq!(
        result["choices"][0]["message"]["reasoning_content"],
        "reason"
    );
    assert!(result["choices"][0]["message"]["content"].is_null());
    tools["stream"] = json!(true);
    let stream = events(
        &body(
            app.clone()
                .oneshot(post(tools, Some("secret")))
                .await
                .unwrap(),
        )
        .await,
    );
    assert_eq!(
        stream[2]["choices"][0]["delta"]["tool_calls"][0]["index"],
        0
    );
    assert_eq!(
        stream[2]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"],
        "{}"
    );
    for (limit, status) in [
        (999, StatusCode::BAD_REQUEST),
        (998, StatusCode::INTERNAL_SERVER_ERROR),
    ] {
        let mut req = base();
        req["max_tokens"] = json!(limit);
        assert_eq!(
            app.clone()
                .oneshot(post(req, Some("secret")))
                .await
                .unwrap()
                .status(),
            status
        );
    }
    let mut req = base();
    req["stream"] = json!(true);
    req["max_tokens"] = json!(998);
    let stream = events(
        &body(
            app.clone()
                .oneshot(post(req, Some("secret")))
                .await
                .unwrap(),
        )
        .await,
    );
    assert_eq!(stream[1]["error"]["message"], "GPU failed");
    assert!(
        stream
            .iter()
            .all(|v| v["choices"][0]["finish_reason"].is_null())
    );
    drop(app);
    worker.thread.join().unwrap();
}

#[tokio::test]
async fn queue_saturation_and_worker_unavailability() {
    let (tx, rx) = tokio::sync::mpsc::channel::<Job>(1);
    let app = router(ServerState {
        model: Arc::from("test"),
        key: None,
        jobs: tx.clone(),
    });
    let (out, _) = tokio::sync::mpsc::channel(1);
    tx.try_send(Job {
        request: Generation {
            messages: vec![],
            tools: vec![],
            thinking: false,
            parallel: true,
            max_tokens: None,
            stops: vec![],
            structured: None,
        },
        output: out,
    })
    .unwrap();
    assert_eq!(
        app.clone()
            .oneshot(post(base(), None))
            .await
            .unwrap()
            .status(),
        StatusCode::TOO_MANY_REQUESTS
    );
    drop(rx);
    assert_eq!(
        app.oneshot(post(base(), None)).await.unwrap().status(),
        StatusCode::SERVICE_UNAVAILABLE
    );
}

#[test]
fn cancellation_releases_backpressure() {
    let (tx, rx) = tokio::sync::mpsc::channel(1);
    let shutdown = Arc::new(AtomicBool::new(false));
    let clone = shutdown.clone();
    let thread = std::thread::spawn(move || {
        let sink = Sink {
            output: tx,
            shutdown: clone,
        };
        assert!(sink.send(Event::Started));
        assert!(!sink.send(Event::Delta(Delta::Content("blocked".into()))));
    });
    while rx.is_empty() {
        std::thread::yield_now();
    }
    // The channel stays full until cancellation releases the blocked sender.
    shutdown.store(true, Ordering::Relaxed);
    drop(rx);
    thread.join().unwrap();
}

#[tokio::test]
async fn dropped_http_stream_cancels_generation_and_next_request_succeeds() {
    struct Cancellable {
        cancelled: Arc<AtomicBool>,
    }
    impl Backend for Cancellable {
        fn generate(&mut self, req: Generation, sink: &Sink) -> Result<(), Failure> {
            sink.send(Event::Started);
            if req.messages[0]["content"] == "slow" {
                while sink.send(Event::Delta(Delta::Content("x".into()))) {}
                self.cancelled.store(true, Ordering::Relaxed);
            } else {
                sink.send(Event::Delta(Delta::Content("next request".into())));
                sink.send(Event::Done {
                    reason: "stop",
                    prompt_tokens: 1,
                    completion_tokens: 1,
                    reasoning_tokens: 0,
                });
            }
            Ok(())
        }
    }
    let cancelled = Arc::new(AtomicBool::new(false));
    let clone = cancelled.clone();
    let worker = engine::start(
        move || Ok(Cancellable { cancelled: clone }),
        Arc::new(AtomicBool::new(false)),
    )
    .unwrap();
    let app = router(ServerState {
        model: Arc::from("test"),
        key: None,
        jobs: worker.sender,
    });
    let mut req = base();
    req["stream"] = json!(true);
    req["messages"][0]["content"] = json!("slow");
    let response = app.clone().oneshot(post(req, None)).await.unwrap();
    drop(response);
    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        while !cancelled.load(Ordering::Relaxed) {
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
    let response = app.clone().oneshot(post(base(), None)).await.unwrap();
    assert!(body(response).await.contains("next request"));
    drop(app);
    worker.thread.join().unwrap();
}

#[tokio::test]
async fn malformed_oversized_and_unknown_requests_keep_error_envelopes() {
    let (tx, _rx) = tokio::sync::mpsc::channel::<Job>(1);
    let app = router(ServerState {
        model: Arc::from("test"),
        key: None,
        jobs: tx,
    });
    for (path, method, payload, status) in [
        (
            "/v1/chat/completions",
            "POST",
            "{".to_owned(),
            StatusCode::BAD_REQUEST,
        ),
        (
            "/v1/chat/completions",
            "POST",
            "x".repeat(8 * 1024 * 1024 + 1),
            StatusCode::PAYLOAD_TOO_LARGE,
        ),
        (
            "/v1/chat/completions",
            "GET",
            String::new(),
            StatusCode::METHOD_NOT_ALLOWED,
        ),
        (
            "/v1/not-an-endpoint",
            "POST",
            "{}".to_owned(),
            StatusCode::NOT_FOUND,
        ),
    ] {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(method)
                    .uri(path)
                    .header("content-type", "application/json")
                    .body(Body::from(payload))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), status);
        let value: Value = serde_json::from_str(&body(response).await).unwrap();
        assert!(value["error"]["message"].is_string());
    }
    let mut unknown = base();
    unknown["model"] = json!("unknown");
    assert_eq!(
        app.oneshot(post(unknown, None)).await.unwrap().status(),
        StatusCode::NOT_FOUND
    );
}

#[derive(Clone)]
pub(super) struct LogWriter(pub(super) Arc<std::sync::Mutex<Vec<u8>>>);
impl std::io::Write for LogWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
#[tokio::test]
async fn protocol_logs_capture_requests_responses_and_sse_without_credentials() {
    use tracing::instrument::WithSubscriber;
    let captured = Arc::new(std::sync::Mutex::new(Vec::new()));
    let writer = LogWriter(captured.clone());
    let subscriber = tracing_subscriber::fmt()
        .without_time()
        .with_ansi(false)
        .with_env_filter("catena_models::server::protocol=debug")
        .with_writer(move || writer.clone())
        .finish();
    let worker = engine::start(
        || {
            Ok(Fake {
                calls: Arc::new(AtomicUsize::new(0)),
            })
        },
        Arc::new(AtomicBool::new(false)),
    )
    .unwrap();
    let app = router(ServerState {
        model: Arc::from("test"),
        key: Some(Arc::from("credential-not-for-logs")),
        jobs: worker.sender,
    });
    async {
        let response = app
            .clone()
            .oneshot(post(base(), Some("credential-not-for-logs")))
            .await
            .unwrap();
        assert!(body(response).await.contains("Hello 世界"));
        let mut streaming = base();
        streaming["stream"] = json!(true);
        let response = app
            .clone()
            .oneshot(post(streaming, Some("credential-not-for-logs")))
            .await
            .unwrap();
        let output = body(response).await;
        assert!(output.contains("[DONE]"));
        let mut invalid = base();
        invalid["model"] = json!("missing");
        let response = app
            .clone()
            .oneshot(post(invalid, Some("credential-not-for-logs")))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        body(response).await;
    }
    .with_subscriber(subscriber)
    .await;
    drop(app);
    worker.thread.join().unwrap();
    let logs = String::from_utf8(captured.lock().unwrap().clone()).unwrap();
    for expected in [
        "incoming",
        "outgoing",
        "request_id=",
        "/v1/chat/completions",
        "Hi",
        "Hello 世界",
        "data:",
        "[DONE]",
        "unknown model",
        "status=404",
    ] {
        assert!(logs.contains(expected), "missing {expected}: {logs}");
    }
    assert!(!logs.contains("credential-not-for-logs"));
    assert!(!logs.contains("authorization"));
}
