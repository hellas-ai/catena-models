//! Stateless Responses HTTP adapter; conversation replay becomes normalized turns.
use super::{
    api::{self, ServerState},
    engine::{Event, Generation},
    responses_output::Output,
};
use axum::{
    Json,
    extract::{State, rejection::JsonRejection},
    http::{HeaderMap, StatusCode},
    response::{
        IntoResponse, Response,
        sse::{Event as SseEvent, KeepAlive, Sse},
    },
};
use serde::Deserialize;
use serde_json::{Map, Value, json};
use std::{
    collections::BTreeSet,
    convert::Infallible,
    pin::Pin,
    task::{Context, Poll},
};
use tokio::sync::mpsc;

#[derive(Debug, Deserialize)]
pub(super) struct Request {
    pub model: String,
    input: Value,
    instructions: Option<String>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    tools: Vec<Value>,
    max_output_tokens: Option<usize>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    tool_choice: Option<Value>,
    parallel_tool_calls: Option<bool>,
    reasoning: Option<Value>,
    text: Option<Value>,
    #[serde(default)]
    chat_template_kwargs: Map<String, Value>,
    #[serde(default)]
    metadata: Map<String, Value>,
    #[serde(default)]
    include: Vec<String>,
    store: Option<bool>,
    previous_response_id: Option<String>,
    background: Option<bool>,
    truncation: Option<String>,
    #[serde(flatten)]
    extra: Map<String, Value>,
}
pub(super) struct Prepared {
    pub generation: Generation,
    pub custom: BTreeSet<String>,
    pub settings: Value,
    pub structured: Option<std::sync::Arc<super::structured::StructuredOutput>>,
}
impl Request {
    pub(super) fn prepare(&self, model: &str) -> Result<Prepared, String> {
        if self.store == Some(true) || self.previous_response_id.is_some() {
            return Err("Responses is stateless: use store=false and replay input items instead of previous_response_id".into());
        }
        if self.background == Some(true)
            || self.truncation.as_deref().is_some_and(|s| s != "disabled")
        {
            return Err("background generation and automatic truncation are not supported".into());
        }
        if self
            .include
            .iter()
            .any(|s| s != "reasoning.encrypted_content")
        {
            return Err("only reasoning.encrypted_content is recognized in include; reasoning is returned as plaintext summary items".into());
        }
        for (key, value) in &self.extra {
            let accepted = match key.as_str() {
                // Codex supplies diagnostic metadata; it is not model input.
                "client_metadata" => value.is_object() || value.is_null(),
                "user" | "safety_identifier" | "prompt_cache_key" => {
                    value.is_string() || value.is_null()
                }
                "service_tier" => {
                    value.is_null() || matches!(value.as_str(), Some("auto" | "default"))
                }
                "prompt_cache_retention" => value.is_null() || value == "in_memory",
                "stream_options" => {
                    value.is_null()
                        || value.as_object().is_some_and(|m| {
                            m.iter()
                                .all(|(k, v)| k == "include_obfuscation" && v.is_boolean())
                        })
                }
                _ => value.is_null(),
            };
            if !accepted {
                return Err(format!("unsupported Responses parameter: {key}"));
            }
        }
        let mut messages = normalize_input(&self.input)?;
        if let Some(instructions) = &self.instructions {
            messages.insert(0, json!({"role":"system","content":instructions}));
        }
        let mut kwargs = self.chat_template_kwargs.clone();
        let mut reasoning = json!({"effort":"none","summary":null});
        if let Some(options) = self.reasoning.as_ref().filter(|v| !v.is_null()) {
            let options = options.as_object().ok_or("reasoning must be an object")?;
            for key in options.keys() {
                if !matches!(key.as_str(), "effort" | "summary") {
                    return Err(format!("unsupported reasoning option: {key}"));
                }
            }
            if let Some(effort) = options.get("effort").filter(|v| !v.is_null()) {
                let effort = effort.as_str().ok_or("reasoning effort must be a string")?;
                if !matches!(
                    effort,
                    "none" | "minimal" | "low" | "medium" | "high" | "xhigh"
                ) {
                    return Err("unsupported reasoning effort".into());
                }
                let thinking = effort != "none";
                if kwargs
                    .get("enable_thinking")
                    .is_some_and(|v| v != &json!(thinking))
                {
                    return Err("reasoning effort conflicts with enable_thinking".into());
                }
                kwargs.insert("enable_thinking".into(), json!(thinking));
                reasoning["effort"] = json!(effort);
            }
            if let Some(summary) = options.get("summary").filter(|v| !v.is_null()) {
                if !matches!(summary.as_str(), Some("auto" | "concise" | "detailed")) {
                    return Err("unsupported reasoning summary option".into());
                }
                reasoning["summary"] = summary.clone();
            }
        }
        let mut text = json!({"format":{"type":"text"},"verbosity":"medium"});
        let mut structured = None;
        if let Some(options) = self.text.as_ref().filter(|v| !v.is_null()) {
            let options = options.as_object().ok_or("text must be an object")?;
            for (key, value) in options {
                match key.as_str() {
                    // Optional controls may be explicitly null in Codex requests.
                    "format"
                        if value.is_null()
                            || value.as_object().is_some_and(|format| {
                                format.get("type").is_some_and(|kind| kind == "text")
                                    && format
                                        .iter()
                                        .all(|(key, value)| key == "type" || value.is_null())
                            }) => {}
                    "format" if value["type"] == "json_schema" => {
                        structured = Some(std::sync::Arc::new(
                            super::structured::StructuredOutput::new(value)?,
                        ));
                        text["format"] = value.clone();
                    }
                    "format" => {
                        return Err(
                            "unsupported text.format: expected plain text or json_schema".into(),
                        );
                    }
                    "verbosity" if value.is_null() => {}
                    "verbosity" if matches!(value.as_str(), Some("low" | "medium" | "high")) => {
                        text["verbosity"] = value.clone();
                        let instruction = match value.as_str().unwrap() {
                            "low" => Some("Keep the final answer concise."),
                            "high" => Some("Provide a detailed final answer."),
                            _ => None,
                        };
                        if let Some(instruction) = instruction {
                            messages.insert(0, json!({"role":"developer","content":instruction}));
                        }
                    }
                    _ => return Err(format!("unsupported text option: {key}")),
                }
            }
        }
        // Namespace groups are intentionally ignored, including nested tools.
        let declared_tools: Vec<_> = self
            .tools
            .iter()
            .filter(|tool| tool["type"] != "namespace")
            .cloned()
            .collect();
        let (tools, custom) = normalize_tools(&declared_tools)?;
        let mut generation = api::normalize_chat(
            json!({
                "model":self.model,"messages":messages,"tools":tools,
                "max_completion_tokens":self.max_output_tokens,"temperature":self.temperature,
                "top_p":self.top_p,"tool_choice":self.tool_choice,
                "parallel_tool_calls":self.parallel_tool_calls,"chat_template_kwargs":kwargs,
            }),
            model,
        )?;
        generation.structured = structured.clone();
        if let Some(format) = &structured {
            let instruction = format.instruction();
            if generation.messages[0]["role"] == "system" {
                let content = generation.messages[0]["content"].as_str().unwrap();
                generation.messages[0]["content"] = json!(format!("{content}\n\n{instruction}"));
            } else {
                generation
                    .messages
                    .insert(0, json!({"role":"system","content":instruction}));
            }
        }
        if generation.thinking && reasoning["effort"] == "none" {
            reasoning["effort"] = json!("medium");
        }
        let settings = json!({"instructions":self.instructions,"max_output_tokens":self.max_output_tokens,
            "tools":declared_tools,"tool_choice":self.tool_choice.clone().unwrap_or(json!("auto")),
            "parallel_tool_calls":generation.parallel,"reasoning":reasoning,"text":text,"metadata":self.metadata});
        Ok(Prepared {
            generation,
            custom,
            settings,
            structured,
        })
    }
}

fn normalize_tools(tools: &[Value]) -> Result<(Vec<Value>, BTreeSet<String>), String> {
    let mut normalized = Vec::new();
    let mut custom = BTreeSet::new();
    for tool in tools {
        let name = tool["name"]
            .as_str()
            .ok_or("Responses tools require a top-level name")?;
        let definition = match tool["type"].as_str() {
            Some("function") => json!({"type":"function","function":{
                "name":name,"description":tool["description"],
                "parameters":tool.get("parameters").cloned().unwrap_or(json!({"type":"object","properties":{}})),
                "strict":tool["strict"]}}),
            Some("custom") => {
                custom.insert(name.to_owned());
                let mut description = tool
                    .get("description")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_owned();
                if let Some(format) = tool.get("format").filter(|v| !v.is_null()) {
                    match format["type"].as_str() {
                        Some("text") => {}
                        Some("grammar") => {
                            if !matches!(format["syntax"].as_str(), Some("lark" | "regex"))
                                || !format["definition"].is_string()
                            {
                                return Err("custom grammar requires lark/regex syntax and a string definition".into());
                            }
                            description.push_str("\nReturn the input string in this format (grammar guidance, without constrained decoding):\n");
                            description.push_str(format["definition"].as_str().unwrap());
                        }
                        _ => return Err("unsupported custom tool format".into()),
                    }
                }
                json!({"type":"function","function":{"name":name,"description":description,"parameters":{"type":"object","properties":{"input":{"type":"string","description":"The complete free-form tool input"}},"required":["input"],"additionalProperties":false}}})
            }
            _ => {
                return Err(
                    "Responses supports function and custom client-executed tools only".into(),
                );
            }
        };
        normalized.push(definition);
    }
    Ok((normalized, custom))
}

fn text_content(content: &Value, input: bool) -> Result<String, String> {
    if let Some(text) = content.as_str() {
        return Ok(text.into());
    }
    let parts = content
        .as_array()
        .ok_or("content must be a string or text-part array")?;
    let mut text = String::new();
    for part in parts {
        if !matches!(
            part["type"].as_str(),
            Some("input_text" | "output_text" | "text")
        ) || (input && part["type"] == "output_text")
        {
            return Err("only text content parts are supported".into());
        }
        text.push_str(
            part["text"]
                .as_str()
                .ok_or("text content part has no text")?,
        );
    }
    Ok(text)
}
fn append_assistant(messages: &mut Vec<Value>) -> &mut Value {
    if messages.last().is_none_or(|m| m["role"] != "assistant") {
        messages.push(json!({"role":"assistant","content":""}));
    }
    messages.last_mut().unwrap()
}
fn append_field(message: &mut Value, key: &str, text: &str) {
    let mut joined = message[key].as_str().unwrap_or("").to_owned();
    if !joined.is_empty() && !text.is_empty() {
        joined.push('\n');
    }
    joined.push_str(text);
    message[key] = json!(joined);
}
fn normalize_input(input: &Value) -> Result<Vec<Value>, String> {
    if let Some(text) = input.as_str() {
        return Ok(vec![json!({"role":"user","content":text})]);
    }
    let items = input
        .as_array()
        .ok_or("input must be a string or an array of input items")?;
    let mut messages = Vec::new();
    for item in items {
        let kind = item["type"].as_str().unwrap_or("message");
        match kind {
            "message" => {
                let role = item["role"].as_str().ok_or("message has no role")?;
                if !matches!(role, "system" | "developer" | "user" | "assistant") {
                    return Err("unsupported Responses message role".into());
                }
                let text = text_content(&item["content"], role != "assistant")?;
                if role == "assistant" {
                    append_field(append_assistant(&mut messages), "content", &text);
                } else {
                    messages.push(json!({"role":role,"content":text}));
                }
            }
            "function_call" | "custom_tool_call" => {
                let call_id = item["call_id"]
                    .as_str()
                    .filter(|s| !s.is_empty())
                    .ok_or("tool call has no call_id")?;
                let name = item["name"]
                    .as_str()
                    .filter(|s| !s.is_empty())
                    .ok_or("tool call has no name")?;
                let arguments = if kind == "custom_tool_call" {
                    json!({"input":item["input"].as_str().ok_or("custom tool input must be a string")?}).to_string()
                } else {
                    item["arguments"]
                        .as_str()
                        .ok_or("function arguments must be a JSON string")?
                        .to_owned()
                };
                let message = append_assistant(&mut messages);
                if message.get("tool_calls").is_none() {
                    message["tool_calls"] = json!([]);
                }
                message["tool_calls"].as_array_mut().unwrap().push(json!({"id":call_id,"type":"function","function":{"name":name,"arguments":arguments}}));
            }
            "function_call_output" | "custom_tool_call_output" => {
                let id = item["call_id"]
                    .as_str()
                    .ok_or("tool output has no call_id")?;
                messages.push(json!({"role":"tool","tool_call_id":id,"content":text_content(&item["output"],false)?}));
            }
            "reasoning" => {
                if item
                    .get("encrypted_content")
                    .is_some_and(|v| !v.is_null() && v != "")
                {
                    return Err("opaque encrypted reasoning cannot be replayed; send plaintext reasoning items returned by this server".into());
                }
                let mut text = String::new();
                let parts = item
                    .get("content")
                    .filter(|v| v.as_array().is_some_and(|a| !a.is_empty()))
                    .or_else(|| item.get("summary"));
                if let Some(parts) = parts {
                    for part in parts.as_array().ok_or("reasoning parts must be an array")? {
                        if !matches!(
                            part["type"].as_str(),
                            Some("summary_text" | "reasoning_text")
                        ) {
                            return Err("unsupported reasoning part".into());
                        }
                        text.push_str(part["text"].as_str().ok_or("reasoning part has no text")?);
                    }
                }
                if !text.is_empty() {
                    append_field(append_assistant(&mut messages), "reasoning_content", &text);
                }
            }
            _ => return Err(format!("unsupported Responses input item: {kind}")),
        }
    }
    Ok(messages)
}

pub(super) async fn create(
    State(state): State<ServerState>,
    headers: HeaderMap,
    body: Result<Json<Value>, JsonRejection>,
) -> Response {
    if let Err(status) = api::authorize(&state, &headers) {
        return api::error(
            status,
            "authentication_error",
            "invalid or missing bearer token",
        );
    }
    let incoming = match body {
        Ok(Json(r)) => r,
        Err(e) => {
            return api::error(
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
    let request: Request = match serde_json::from_value(incoming.clone()) {
        Ok(request) => request,
        Err(error) => {
            return rejected_request(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                &error.to_string(),
                &incoming,
            );
        }
    };
    if request.model != state.model.as_ref() {
        return rejected_request(
            StatusCode::NOT_FOUND,
            "model_not_found",
            "unknown model",
            &incoming,
        );
    }
    let prepared = match request.prepare(&state.model) {
        Ok(p) => p,
        Err(e) => {
            return rejected_request(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                &e,
                &incoming,
            );
        }
    };
    drop(incoming);
    let mut output = Output::new(
        state.model.clone(),
        prepared.custom,
        prepared.settings,
        request.stream,
        prepared.structured,
    );
    let mut receiver = match api::enqueue(&state, prepared.generation).await {
        Ok(rx) => rx,
        Err(e) => return *e,
    };
    if request.stream {
        output.start();
        return Sse::new(ResponseStream {
            receiver,
            output,
            finished: false,
        })
        .keep_alive(KeepAlive::default())
        .into_response();
    }
    while let Some(event) = receiver.recv().await {
        match event {
            Event::Delta(delta) => {
                if let Err(e) = output.delta(delta) {
                    return api::failure(super::engine::Failure::internal(e));
                }
            }
            Event::Done {
                reason,
                prompt_tokens,
                completion_tokens,
                reasoning_tokens,
                cached_tokens,
                cache_write_tokens,
            } => {
                if let Err(e) = output.finish(
                    reason,
                    prompt_tokens,
                    completion_tokens,
                    reasoning_tokens,
                    cached_tokens,
                    cache_write_tokens,
                ) {
                    return api::failure(super::engine::Failure::internal(e));
                }
                return Json(output.response()).into_response();
            }
            Event::Failed(f) => return api::failure(f),
            Event::Started => break,
        }
    }
    api::error(
        StatusCode::INTERNAL_SERVER_ERROR,
        "server_error",
        "inference worker stopped unexpectedly",
    )
}

fn rejected_request(status: StatusCode, kind: &str, message: &str, incoming: &Value) -> Response {
    // Log the original JSON (including client extension fields), not a
    // normalized request. Default warn-level logging makes this easy to copy.
    tracing::warn!(target: "catena_models::server::protocol", status=status.as_u16(), error=message,
        request_json=%incoming, "Rejected Responses request");
    api::error(status, kind, message)
}
struct ResponseStream {
    receiver: mpsc::Receiver<Event>,
    output: Output,
    finished: bool,
}
impl tokio_stream::Stream for ResponseStream {
    type Item = Result<SseEvent, Infallible>;
    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        if let Some(event) = this.output.events.pop_front() {
            return Poll::Ready(Some(Ok(wire_event(event))));
        }
        if this.finished {
            return Poll::Ready(None);
        }
        match this.receiver.poll_recv(cx) {
            Poll::Pending => return Poll::Pending,
            Poll::Ready(Some(Event::Delta(delta))) => {
                if let Err(e) = this.output.delta(delta) {
                    this.output.fail(e);
                    this.finished = true;
                    this.receiver.close();
                }
            }
            Poll::Ready(Some(Event::Done {
                reason,
                prompt_tokens,
                completion_tokens,
                reasoning_tokens,
                cached_tokens,
                cache_write_tokens,
            })) => {
                if let Err(e) = this.output.finish(
                    reason,
                    prompt_tokens,
                    completion_tokens,
                    reasoning_tokens,
                    cached_tokens,
                    cache_write_tokens,
                ) {
                    this.output.fail(e);
                }
                this.finished = true;
            }
            Poll::Ready(other) => {
                let message = if let Some(Event::Failed(f)) = other {
                    f.message
                } else {
                    "inference worker stopped unexpectedly".into()
                };
                this.output.fail(message);
                this.finished = true;
            }
        }
        // Structured text is held until validated; it may not emit an event yet.
        match this.output.events.pop_front() {
            Some(event) => Poll::Ready(Some(Ok(wire_event(event)))),
            None => {
                cx.waker().wake_by_ref();
                Poll::Pending
            }
        }
    }
}
fn wire_event(event: Value) -> SseEvent {
    SseEvent::default()
        .event(event["type"].as_str().unwrap())
        .data(event.to_string())
}
