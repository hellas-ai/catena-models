//! Responses output state shared by JSON collection and incremental SSE delivery.
use super::engine::Delta;
use serde_json::{Value, json};
use std::{
    collections::{BTreeSet, VecDeque},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::{SystemTime, UNIX_EPOCH},
};

static NEXT_ID: AtomicU64 = AtomicU64::new(1);
#[derive(PartialEq)]
enum Kind {
    Message,
    Reasoning,
    Function { name: String, call_id: String },
    Custom { name: String, call_id: String },
}
struct Item {
    id: String,
    kind: Kind,
    text: String,
    status: &'static str,
}
impl Item {
    fn part(&self) -> Value {
        match self.kind {
            Kind::Reasoning => json!({"type":"summary_text","text":self.text}),
            _ => json!({"type":"output_text","text":self.text,"annotations":[],"logprobs":[]}),
        }
    }
    fn value(&self, empty: bool) -> Value {
        match &self.kind {
            Kind::Message => {
                json!({"type":"message","id":self.id,"role":"assistant","status":self.status,"content":if empty {vec![]} else {vec![self.part()]}})
            }
            Kind::Reasoning => {
                json!({"type":"reasoning","id":self.id,"status":self.status,"summary":if empty {vec![]} else {vec![self.part()]},"encrypted_content":null})
            }
            Kind::Function { name, call_id } => {
                json!({"type":"function_call","id":self.id,"call_id":call_id,"name":name,"arguments":self.text,"status":self.status})
            }
            Kind::Custom { name, call_id } => {
                json!({"type":"custom_tool_call","id":self.id,"call_id":call_id,"name":name,"input":self.text,"status":self.status})
            }
        }
    }
}
pub(super) struct Output {
    id: String,
    created_at: u64,
    completed_at: Option<u64>,
    model: Arc<str>,
    custom: BTreeSet<String>,
    settings: Value,
    items: Vec<Item>,
    active: Option<usize>,
    status: &'static str,
    usage: Value,
    error: Value,
    incomplete_details: Value,
    sequence: u64,
    stream: bool,
    pub events: VecDeque<Value>,
    structured: Option<Arc<super::structured::StructuredOutput>>,
    pending_text: String,
}
impl Output {
    pub fn new(
        model: Arc<str>,
        custom: BTreeSet<String>,
        settings: Value,
        stream: bool,
        structured: Option<Arc<super::structured::StructuredOutput>>,
    ) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();
        Self {
            id: format!(
                "resp_{:x}_{:x}",
                now.as_nanos(),
                NEXT_ID.fetch_add(1, Ordering::Relaxed)
            ),
            created_at: now.as_secs(),
            completed_at: None,
            model,
            custom,
            settings,
            items: vec![],
            active: None,
            status: "in_progress",
            usage: Value::Null,
            error: Value::Null,
            incomplete_details: Value::Null,
            sequence: 0,
            stream,
            events: VecDeque::new(),
            structured,
            pending_text: String::new(),
        }
    }
    pub fn response(&self) -> Value {
        let mut value = json!({"id":self.id,"object":"response","created_at":self.created_at,
            "status":self.status,"completed_at":self.completed_at,
            "model":&*self.model,"output":self.items.iter().map(|i|i.value(false)).collect::<Vec<_>>(),
            "error":self.error,"incomplete_details":self.incomplete_details,"usage":self.usage,
            "store":false,"background":false,"previous_response_id":null,
            "temperature":0.0,"top_p":1.0,"truncation":"disabled"});
        value
            .as_object_mut()
            .unwrap()
            .extend(self.settings.as_object().unwrap().clone());
        value
    }
    fn emit(&mut self, kind: &str, mut value: Value) {
        if !self.stream {
            return;
        }
        value["type"] = json!(kind);
        value["sequence_number"] = json!(self.sequence);
        self.sequence += 1;
        self.events.push_back(value);
    }
    pub fn start(&mut self) {
        self.emit("response.created", json!({"response":self.response()}));
        self.emit("response.in_progress", json!({"response":self.response()}));
    }
    fn open(&mut self, kind: Kind) {
        self.close("completed");
        let prefix = match kind {
            Kind::Message => "msg",
            Kind::Reasoning => "rs",
            Kind::Function { .. } => "fc",
            Kind::Custom { .. } => "ctc",
        };
        let index = self.items.len();
        let item = Item {
            id: format!("{prefix}_{}_{index}", self.id),
            kind,
            text: String::new(),
            status: "in_progress",
        };
        let value = item.value(true);
        let id = item.id.clone();
        self.items.push(item);
        self.active = Some(index);
        self.emit(
            "response.output_item.added",
            json!({"output_index":index,"item":value}),
        );
        match self.items[index].kind {
            Kind::Message=>self.emit("response.content_part.added",json!({"item_id":id,"output_index":index,"content_index":0,"part":{"type":"output_text","text":"","annotations":[],"logprobs":[]}})),
            Kind::Reasoning=>self.emit("response.reasoning_summary_part.added",json!({"item_id":id,"output_index":index,"summary_index":0,"part":{"type":"summary_text","text":""}})),
            _=>{},
        }
    }
    pub fn delta(&mut self, delta: Delta) -> Result<(), String> {
        match delta {
            Delta::Content(text) | Delta::Reasoning(text) if text.is_empty() => {}
            Delta::Content(text) if self.structured.is_some() => {
                self.pending_text.push_str(&text);
            }
            Delta::Content(text) => {
                if self
                    .active
                    .is_none_or(|i| self.items[i].kind != Kind::Message)
                {
                    self.open(Kind::Message);
                }
                let index = self.active.unwrap();
                let id = self.items[index].id.clone();
                self.items[index].text.push_str(&text);
                self.emit("response.output_text.delta",json!({"item_id":id,"output_index":index,"content_index":0,"delta":text,"logprobs":[]}));
            }
            Delta::Reasoning(text) => {
                if self
                    .active
                    .is_none_or(|i| self.items[i].kind != Kind::Reasoning)
                {
                    self.open(Kind::Reasoning);
                }
                let index = self.active.unwrap();
                let id = self.items[index].id.clone();
                self.items[index].text.push_str(&text);
                self.emit(
                    "response.reasoning_summary_text.delta",
                    json!({"item_id":id,"output_index":index,"summary_index":0,"delta":text}),
                );
            }
            Delta::Tool { index, call } => {
                // Schema constraints apply to the final answer, not commentary
                // on a turn that hands control to client-executed tools.
                let format = self.structured.take();
                let pending = std::mem::take(&mut self.pending_text);
                self.delta(Delta::Content(pending))?;
                self.structured = format;

                let call_id = format!("call_{}_{index}", self.id);
                let custom = self.custom.contains(&call.name);
                let text = if custom {
                    let args: Value = serde_json::from_str(&call.arguments)
                        .map_err(|e| format!("invalid custom tool arguments: {e}"))?;
                    args["input"]
                        .as_str()
                        .ok_or("custom tool call requires a string input")?
                        .to_owned()
                } else {
                    call.arguments
                };
                self.open(if custom {
                    Kind::Custom {
                        name: call.name,
                        call_id,
                    }
                } else {
                    Kind::Function {
                        name: call.name,
                        call_id,
                    }
                });
                let index = self.active.unwrap();
                let id = self.items[index].id.clone();
                self.items[index].text = text.clone();
                self.emit(
                    if custom {
                        "response.custom_tool_call_input.delta"
                    } else {
                        "response.function_call_arguments.delta"
                    },
                    json!({"item_id":id,"output_index":index,"delta":text}),
                );
                self.close("completed");
            }
        }
        Ok(())
    }
    fn close(&mut self, status: &'static str) {
        let Some(index) = self.active.take() else {
            return;
        };
        let item = &mut self.items[index];
        item.status = status;
        let id = item.id.clone();
        let text = item.text.clone();
        let part = item.part();
        let value = item.value(false);
        match item.kind {
            Kind::Message => {
                self.emit("response.output_text.done",json!({"item_id":id,"output_index":index,"content_index":0,"text":text,"logprobs":[]}));
                self.emit(
                    "response.content_part.done",
                    json!({"item_id":id,"output_index":index,"content_index":0,"part":part}),
                );
            }
            Kind::Reasoning => {
                self.emit(
                    "response.reasoning_summary_text.done",
                    json!({"item_id":id,"output_index":index,"summary_index":0,"text":text}),
                );
                self.emit(
                    "response.reasoning_summary_part.done",
                    json!({"item_id":id,"output_index":index,"summary_index":0,"part":part}),
                );
            }
            Kind::Function { .. } => self.emit(
                "response.function_call_arguments.done",
                json!({"item_id":id,"output_index":index,"arguments":text}),
            ),
            Kind::Custom { .. } => self.emit(
                "response.custom_tool_call_input.done",
                json!({"item_id":id,"output_index":index,"input":text}),
            ),
        }
        self.emit(
            "response.output_item.done",
            json!({"output_index":index,"item":value}),
        );
    }
    pub fn finish(
        &mut self,
        reason: &str,
        prompt: usize,
        completion: usize,
        reasoning: usize,
        cached: usize,
        cache_write: usize,
    ) -> Result<(), String> {
        let length = reason == "length";
        if let Some(format) = self.structured.take() {
            let tool_turn = self
                .items
                .iter()
                .any(|item| matches!(item.kind, Kind::Function { .. } | Kind::Custom { .. }));
            if !length && !tool_turn {
                format.validate(&self.pending_text)?;
            }
            let text = std::mem::take(&mut self.pending_text);
            if !text.is_empty() {
                self.delta(Delta::Content(text))?;
            }
        }
        self.close(if length { "incomplete" } else { "completed" });
        self.status = if length { "incomplete" } else { "completed" };
        if !length {
            self.completed_at = Some(
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
            );
        }
        if length {
            self.incomplete_details = json!({"reason":"max_output_tokens"});
        }
        self.usage = json!({"input_tokens":prompt,"input_tokens_details":{"cached_tokens":cached,"cache_write_tokens":cache_write},"output_tokens":completion,"output_tokens_details":{"reasoning_tokens":reasoning},"total_tokens":prompt+completion});
        self.emit(
            if length {
                "response.incomplete"
            } else {
                "response.completed"
            },
            json!({"response":self.response()}),
        );
        Ok(())
    }
    pub fn fail(&mut self, message: String) {
        if let Some(index) = self.active.take() {
            self.items[index].status = "incomplete";
        }
        self.status = "failed";
        self.error = json!({"code":"server_error","message":message});
        self.emit(
            "error",
            json!({"code":"server_error","message":message,"param":null}),
        );
        self.emit("response.failed", json!({"response":self.response()}));
    }
}
