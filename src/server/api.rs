//! Chat Completions adapter. The inference worker only sees normalized turns.
use super::{
    engine::{Delta, Event, Failure, Generation, Job},
    tools::{Tool, validate_tools},
};
use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, State, rejection::JsonRejection},
    http::{HeaderMap, StatusCode},
    response::{
        IntoResponse, Response,
        sse::{Event as SseEvent, KeepAlive, Sse},
    },
    routing::{get, post},
};
use serde::Deserialize;
use serde_json::{Map, Value, json};
use std::{
    collections::BTreeSet,
    convert::Infallible,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::{SystemTime, UNIX_EPOCH},
};
use tokio::sync::mpsc;

#[derive(Clone)]
pub struct ServerState {
    pub model: Arc<str>,
    pub key: Option<Arc<str>>,
    pub jobs: mpsc::Sender<Job>,
}

pub fn router(state: ServerState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat))
        .fallback(|| async { error(StatusCode::NOT_FOUND, "not_found", "unknown endpoint") })
        .method_not_allowed_fallback(|| async {
            error(
                StatusCode::METHOD_NOT_ALLOWED,
                "invalid_request_error",
                "method not allowed",
            )
        })
        .layer(DefaultBodyLimit::max(8 * 1024 * 1024))
        .layer(axum::middleware::from_fn(super::protocol::log_messages))
        .with_state(state)
}
fn authorize(state: &ServerState, headers: &HeaderMap) -> Result<(), StatusCode> {
    if let Some(key) = &state.key
        && headers.get("authorization").and_then(|v| v.to_str().ok())
            != Some(&format!("Bearer {key}"))
    {
        return Err(StatusCode::UNAUTHORIZED);
    }
    Ok(())
}

async fn health(State(state): State<ServerState>, headers: HeaderMap) -> Response {
    if let Err(status) = authorize(&state, &headers) {
        return error(
            status,
            "authentication_error",
            "invalid or missing bearer token",
        );
    }
    if state.jobs.is_closed() {
        return error(
            StatusCode::SERVICE_UNAVAILABLE,
            "server_error",
            "inference worker is unavailable",
        );
    }
    Json(json!({"status":"ok"})).into_response()
}
async fn models(State(state): State<ServerState>, headers: HeaderMap) -> Response {
    if let Err(status) = authorize(&state, &headers) {
        return error(
            status,
            "authentication_error",
            "invalid or missing bearer token",
        );
    }
    Json(json!({"object":"list","data":[{"id":&*state.model,"object":"model","created":0,"owned_by":"catena"}]})).into_response()
}
fn error_json(kind: &str, message: &str) -> Value {
    json!({"error":{"message":message,"type":kind,"param":null,"code":null}})
}
fn error(status: StatusCode, kind: &str, message: &str) -> Response {
    (status, Json(error_json(kind, message))).into_response()
}
fn failure(f: Failure) -> Response {
    if f.invalid_request {
        error(StatusCode::BAD_REQUEST, "invalid_request_error", &f.message)
    } else {
        error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "server_error",
            &f.message,
        )
    }
}

#[derive(Debug, Deserialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Value>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    stream_options: Option<StreamOptions>,
    max_tokens: Option<usize>,
    max_completion_tokens: Option<usize>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    n: Option<usize>,
    frequency_penalty: Option<f64>,
    presence_penalty: Option<f64>,
    seed: Option<i64>,
    #[serde(default)]
    tools: Vec<Tool>,
    tool_choice: Option<Value>,
    parallel_tool_calls: Option<bool>,
    stop: Option<Value>,
    #[serde(default)]
    chat_template_kwargs: Map<String, Value>,
    #[serde(flatten)]
    extra: Map<String, Value>,
}
#[derive(Debug, Deserialize)]
struct StreamOptions {
    #[serde(default)]
    include_usage: bool,
}
impl ChatRequest {
    fn normalize(&self, model: &str) -> Result<Generation, String> {
        if self.model != model {
            return Err("unknown model".into());
        }
        if self.temperature.is_some_and(|t| t != 0.)
            || self.top_p.is_some_and(|p| p != 1.)
            || self.n.is_some_and(|n| n != 1)
            || self.frequency_penalty.is_some_and(|p| p != 0.)
            || self.presence_penalty.is_some_and(|p| p != 0.)
        {
            return Err(
                "only greedy decoding is supported: temperature=0, top_p=1, n=1, penalties=0"
                    .into(),
            );
        }
        let _ = self.seed; // Greedy generation has no random state.
        for (key, value) in &self.extra {
            let neutral = match key.as_str() {
                "user" | "metadata" => true,
                "store" | "logprobs" => value == false || value.is_null(),
                "logit_bias" => value.as_object().is_some_and(Map::is_empty) || value.is_null(),
                "response_format" => value.is_null() || value == &json!({"type":"text"}),
                _ => value.is_null(),
            };
            if !neutral {
                return Err(format!("unsupported request parameter: {key}"));
            }
        }
        if self
            .max_tokens
            .zip(self.max_completion_tokens)
            .is_some_and(|(a, b)| a != b)
        {
            return Err("max_tokens and max_completion_tokens conflict".into());
        }
        let max_tokens = self.max_completion_tokens.or(self.max_tokens);
        if max_tokens == Some(0) {
            return Err("output token limit must be positive".into());
        }
        let stops = match &self.stop {
            None | Some(Value::Null) => vec![],
            Some(Value::String(s)) => vec![s.clone()],
            Some(Value::Array(a)) => a
                .iter()
                .map(|s| {
                    s.as_str()
                        .map(str::to_owned)
                        .ok_or("stop must contain strings")
                })
                .collect::<Result<Vec<_>, _>>()?,
            _ => return Err("stop must be a string or string array".into()),
        };
        if stops.len() > 4 || stops.iter().any(String::is_empty) {
            return Err("stop accepts up to four nonempty strings".into());
        }
        let mut thinking = false;
        for (key, value) in &self.chat_template_kwargs {
            if key != "enable_thinking" {
                return Err(format!("unsupported chat template option: {key}"));
            }
            thinking = value.as_bool().ok_or("enable_thinking must be boolean")?;
        }
        validate_tools(&self.tools)?;
        let tools = match self.tool_choice.as_ref().and_then(Value::as_str) {
            Some("none") => vec![],
            None if self.tool_choice.as_ref().is_none_or(Value::is_null) => self.tools.clone(),
            Some("auto") => self.tools.clone(),
            _ => return Err("tool_choice supports only auto or none".into()),
        };
        let messages = normalize_messages(&self.messages)?;
        Ok(Generation {
            messages,
            tools,
            thinking,
            parallel: self.parallel_tool_calls.unwrap_or(true),
            max_tokens,
            stops,
        })
    }
}

fn normalize_messages(messages: &[Value]) -> Result<Vec<Value>, String> {
    if messages.is_empty() {
        return Err("messages must not be empty".into());
    }
    let mut normalized = Vec::new();
    let mut instructions = Vec::new();
    let mut pending = BTreeSet::new();
    let mut used = BTreeSet::new();
    let mut has_user = false;
    for message in messages {
        let role = message["role"].as_str().ok_or("message has no role")?;
        if role != "assistant" && message.get("tool_calls").is_some_and(|v| !v.is_null()) {
            return Err("tool_calls are only valid on assistant messages".into());
        }
        if role != "assistant" && message["content"].is_null() {
            return Err("message content must be text".into());
        }
        let content = match &message["content"] {
            Value::Null => String::new(),
            Value::String(s) => s.clone(),
            Value::Array(parts) => {
                let mut s = String::new();
                for part in parts {
                    if part["type"] != "text" {
                        return Err("only text message content is supported".into());
                    }
                    s.push_str(part["text"].as_str().ok_or("text part has no text")?);
                }
                s
            }
            _ => return Err("message content must be text".into()),
        };
        if matches!(role, "system" | "developer") {
            instructions.push(content);
            continue;
        }
        if role != "tool" && !pending.is_empty() {
            return Err(
                "assistant tool calls require matching tool results before the next turn".into(),
            );
        }
        let mut out = json!({"role":role,"content":content});
        match role {
            "user" => has_user = true,
            "assistant" => {
                if let Some(reasoning) = message.get("reasoning_content").filter(|v| !v.is_null()) {
                    if !reasoning.is_string() {
                        return Err("reasoning_content must be a string".into());
                    }
                    out["reasoning_content"] = reasoning.clone();
                }
                if let Some(calls) = message.get("tool_calls").filter(|v| !v.is_null()) {
                    let calls = calls.as_array().ok_or("tool_calls must be an array")?;
                    let mut normalized_calls = Vec::new();
                    for call in calls {
                        let id = call["id"]
                            .as_str()
                            .filter(|s| !s.is_empty())
                            .ok_or("tool call has no id")?;
                        if !used.insert(id.to_owned()) {
                            return Err("duplicate tool call id".into());
                        }
                        if call["type"] != "function" {
                            return Err("only function tool calls are supported".into());
                        }
                        let name = call["function"]["name"]
                            .as_str()
                            .filter(|s| !s.is_empty())
                            .ok_or("tool call has no function name")?;
                        let args: Value = serde_json::from_str(
                            call["function"]["arguments"]
                                .as_str()
                                .ok_or("function arguments must be a JSON string")?,
                        )
                        .map_err(|e| format!("invalid historical tool arguments: {e}"))?;
                        if !args.is_object() {
                            return Err("function arguments must encode an object".into());
                        }
                        normalized_calls.push(json!({"id":id,"type":"function","function":{"name":name,"arguments":args}}));
                        pending.insert(id.to_owned());
                    }
                    out["tool_calls"] = json!(normalized_calls);
                }
            }
            "tool" => {
                let id = message["tool_call_id"]
                    .as_str()
                    .ok_or("tool message has no tool_call_id")?;
                if !pending.remove(id) {
                    return Err("tool result does not match a pending call".into());
                }
                out["tool_call_id"] = json!(id);
            }
            _ => return Err(format!("unsupported message role: {role}")),
        }
        normalized.push(out);
    }
    if !pending.is_empty() {
        return Err("missing tool results".into());
    }
    if !has_user {
        return Err("messages must contain a user turn".into());
    }
    if !instructions.is_empty() {
        normalized.insert(
            0,
            json!({"role":"system","content":instructions.join("\n\n")}),
        );
    }
    Ok(normalized)
}

static NEXT_ID: AtomicU64 = AtomicU64::new(1);
struct Envelope {
    id: String,
    created: u64,
    model: Arc<str>,
}
impl Envelope {
    fn new(model: Arc<str>) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();
        Self {
            id: format!(
                "chatcmpl-{:x}-{:x}",
                now.as_nanos(),
                NEXT_ID.fetch_add(1, Ordering::Relaxed)
            ),
            created: now.as_secs(),
            model,
        }
    }
    fn chunk(&self, delta: Value, reason: Option<&str>, usage: bool) -> Value {
        let mut v = json!({"id":self.id,"object":"chat.completion.chunk","created":self.created,"model":&*self.model,"choices":[{"index":0,"delta":delta,"finish_reason":reason}]});
        if usage {
            v["usage"] = Value::Null;
        }
        v
    }
    fn delta(&self, delta: Delta) -> Value {
        match delta {
            Delta::Content(s) => json!({"content":s}),
            Delta::Reasoning(s) => json!({"reasoning_content":s}),
            Delta::Tool { index, call } => {
                json!({"tool_calls":[{"index":index,"id":format!("call_{}_{}",self.id,index),"type":"function","function":{"name":call.name,"arguments":call.arguments}}]})
            }
        }
    }
}
fn usage(prompt: usize, completion: usize) -> Value {
    json!({"prompt_tokens":prompt,"completion_tokens":completion,"total_tokens":prompt+completion})
}
fn sse(value: Value) -> Result<SseEvent, Infallible> {
    Ok(SseEvent::default().data(value.to_string()))
}

async fn chat(
    State(state): State<ServerState>,
    headers: HeaderMap,
    body: Result<Json<ChatRequest>, JsonRejection>,
) -> Response {
    if let Err(status) = authorize(&state, &headers) {
        return error(
            status,
            "authentication_error",
            "invalid or missing bearer token",
        );
    }
    let request = match body {
        Ok(Json(r)) => r,
        Err(e) => {
            return error(
                if e.status() == StatusCode::PAYLOAD_TOO_LARGE {
                    e.status()
                } else {
                    StatusCode::BAD_REQUEST
                },
                "invalid_request_error",
                &e.body_text(),
            );
        }
    };
    if request.model != state.model.as_ref() {
        return error(StatusCode::NOT_FOUND, "model_not_found", "unknown model");
    }
    let generation = match request.normalize(&state.model) {
        Ok(r) => r,
        Err(e) => return error(StatusCode::BAD_REQUEST, "invalid_request_error", &e),
    };
    let (tx, mut rx) = mpsc::channel(32);
    match state.jobs.try_send(Job {
        request: generation,
        output: tx,
    }) {
        Ok(()) => {}
        Err(mpsc::error::TrySendError::Full(_)) => {
            return error(
                StatusCode::TOO_MANY_REQUESTS,
                "rate_limit_error",
                "inference queue is full",
            );
        }
        Err(_) => {
            return error(
                StatusCode::SERVICE_UNAVAILABLE,
                "server_error",
                "inference worker is unavailable",
            );
        }
    }
    match rx.recv().await {
        Some(Event::Started) => {}
        Some(Event::Failed(f)) => return failure(f),
        _ => {
            return error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
                "inference worker stopped unexpectedly",
            );
        }
    }
    let envelope = Envelope::new(state.model);
    let include_usage = request
        .stream_options
        .as_ref()
        .is_some_and(|o| o.include_usage);
    if request.stream {
        return Sse::new(ChatStream {
            receiver: rx,
            pending: std::collections::VecDeque::from([SseEvent::default().data(
                envelope
                    .chunk(
                        json!({"role":"assistant","content":""}),
                        None,
                        include_usage,
                    )
                    .to_string(),
            )]),
            envelope,
            include_usage,
            finished: false,
        })
        .keep_alive(KeepAlive::default())
        .into_response();
    }
    let mut content = String::new();
    let mut reasoning = String::new();
    let mut calls = Vec::new();
    while let Some(event) = rx.recv().await {
        match event {
            Event::Delta(Delta::Content(s)) => content.push_str(&s),
            Event::Delta(Delta::Reasoning(s)) => reasoning.push_str(&s),
            Event::Delta(d @ Delta::Tool { .. }) => {
                let delta = envelope.delta(d);
                let mut call = delta["tool_calls"][0].clone();
                call.as_object_mut().unwrap().remove("index");
                calls.push(call);
            }
            Event::Failed(f) => return failure(f),
            Event::Done {
                reason,
                prompt_tokens,
                completion_tokens,
            } => {
                let mut message = json!({"role":"assistant","content":if content.is_empty() && !calls.is_empty() {Value::Null} else {json!(content)}});
                if !reasoning.is_empty() {
                    message["reasoning_content"] = json!(reasoning);
                }
                if !calls.is_empty() {
                    message["tool_calls"] = json!(calls);
                }
                return Json(json!({"id":envelope.id,"object":"chat.completion","created":envelope.created,"model":&*envelope.model,"choices":[{"index":0,"message":message,"finish_reason":reason}],"usage":usage(prompt_tokens,completion_tokens)})).into_response();
            }
            _ => {
                return error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "server_error",
                    "unexpected generation event",
                );
            }
        }
    }
    error(
        StatusCode::INTERNAL_SERVER_ERROR,
        "server_error",
        "inference worker stopped unexpectedly",
    )
}

// The HTTP body owns the receiver. Disconnecting drops it, including while the
// request is queued, so no detached forwarding task can keep generation alive.
struct ChatStream {
    receiver: mpsc::Receiver<Event>,
    envelope: Envelope,
    include_usage: bool,
    pending: std::collections::VecDeque<SseEvent>,
    finished: bool,
}
impl tokio_stream::Stream for ChatStream {
    type Item = Result<SseEvent, Infallible>;
    fn poll_next(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        use std::task::Poll;
        let this = self.get_mut();
        if let Some(event) = this.pending.pop_front() {
            return Poll::Ready(Some(Ok(event)));
        }
        if this.finished {
            return Poll::Ready(None);
        }
        let event = match this.receiver.poll_recv(cx) {
            Poll::Pending => return Poll::Pending,
            Poll::Ready(event) => event,
        };
        match event {
            Some(Event::Delta(delta)) => {
                return Poll::Ready(Some(sse(this.envelope.chunk(
                    this.envelope.delta(delta),
                    None,
                    this.include_usage,
                ))));
            }
            Some(Event::Done {
                reason,
                prompt_tokens,
                completion_tokens,
            }) => {
                this.pending.push_back(
                    SseEvent::default().data(
                        this.envelope
                            .chunk(json!({}), Some(reason), this.include_usage)
                            .to_string(),
                    ),
                );
                if this.include_usage {
                    this.pending.push_back(SseEvent::default().data(json!({"id":this.envelope.id,"object":"chat.completion.chunk","created":this.envelope.created,"model":&*this.envelope.model,"choices":[],"usage":usage(prompt_tokens,completion_tokens)}).to_string()));
                }
            }
            other => {
                let f = if let Some(Event::Failed(f)) = other {
                    f
                } else {
                    Failure::internal("inference worker stopped unexpectedly")
                };
                this.pending.push_back(
                    SseEvent::default().data(
                        error_json(
                            if f.invalid_request {
                                "invalid_request_error"
                            } else {
                                "server_error"
                            },
                            &f.message,
                        )
                        .to_string(),
                    ),
                );
            }
        }
        this.finished = true;
        this.pending.push_back(SseEvent::default().data("[DONE]"));
        Poll::Ready(this.pending.pop_front().map(Ok))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn request(extra: Value) -> ChatRequest {
        let mut value = json!({"model":"test","messages":[{"role":"user","content":"hello"}]});
        value
            .as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        serde_json::from_value(value).unwrap()
    }
    #[test]
    fn limits_and_unsupported_features_are_explicit() {
        assert!(request(json!({})).normalize("test").is_ok());
        assert!(
            request(
                json!({"temperature":0,"top_p":1,"seed":7,"max_tokens":5,"max_completion_tokens":5})
            )
            .normalize("test")
            .is_ok()
        );
        for extra in [
            json!({"temperature":1}),
            json!({"n":2}),
            json!({"max_tokens":0}),
            json!({"max_tokens":5,"max_completion_tokens":6}),
            json!({"stop":[""]}),
            json!({"tool_choice":"required"}),
            json!({"response_format":{"type":"json_object"}}),
            json!({"chat_template_kwargs":{"enable_thinking":"yes"}}),
        ] {
            assert!(request(extra.clone()).normalize("test").is_err(), "{extra}");
        }
        assert!(request(json!({})).normalize("other").is_err());
    }
    #[test]
    fn instructions_and_function_result_history() {
        let history = json!([
            {"role":"system","content":"first"},{"role":"developer","content":[{"type":"text","text":"second"}]},
            {"role":"user","content":"question"},
            {"role":"assistant","content":null,"tool_calls":[{"id":"a","type":"function","function":{"name":"read","arguments":"{\"path\":\"a\"}"}}]},
            {"role":"tool","tool_call_id":"a","content":"result"}
        ]);
        let normalized = normalize_messages(history.as_array().unwrap()).unwrap();
        assert_eq!(normalized[0]["content"], "first\n\nsecond");
        assert_eq!(
            normalized[2]["tool_calls"][0]["function"]["arguments"]["path"],
            "a"
        );
        let mut bad = history.as_array().unwrap().clone();
        bad.pop();
        assert!(normalize_messages(&bad).is_err());
        assert!(
            normalize_messages(&[json!({"role":"tool","tool_call_id":"bad","content":"x"})])
                .is_err()
        );
        assert!(
            normalize_messages(&[
                json!({"role":"user","content":[{"type":"image_url","image_url":{"url":"x"}}]})
            ])
            .is_err()
        );
    }
}
