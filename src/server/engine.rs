//! One owning GPU thread, receiving protocol-independent conversations.
use super::{
    output::{OutputParser, StopFilter},
    prompt::Prompt,
    tools::{Call, Tool},
};
use catena_models::{catena::InferenceOptions, model::Model, tokenizer::QwenTokenizer};
use serde_json::Value;
use std::{
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::Duration,
};
use tokio::sync::mpsc;

#[derive(Clone, Debug)]
pub struct Generation {
    pub messages: Vec<Value>,
    pub tools: Vec<Tool>,
    pub thinking: bool,
    pub parallel: bool,
    pub max_tokens: Option<usize>,
    pub stops: Vec<String>,
}
#[derive(Clone, Debug)]
pub enum Delta {
    Content(String),
    Reasoning(String),
    Tool { index: usize, call: Call },
}
#[derive(Clone, Debug)]
pub enum Event {
    Started,
    Delta(Delta),
    Done {
        reason: &'static str,
        prompt_tokens: usize,
        completion_tokens: usize,
    },
    Failed(Failure),
}
#[derive(Clone, Debug)]
pub struct Failure {
    pub message: String,
    pub invalid_request: bool,
}
impl Failure {
    pub fn invalid(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            invalid_request: true,
        }
    }
    pub fn internal(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            invalid_request: false,
        }
    }
}

pub struct Job {
    pub request: Generation,
    pub output: mpsc::Sender<Event>,
}
pub struct Sink {
    pub output: mpsc::Sender<Event>,
    pub shutdown: Arc<AtomicBool>,
}
impl Sink {
    pub fn cancelled(&self) -> bool {
        self.output.is_closed() || self.shutdown.load(Ordering::Relaxed)
    }
    pub fn send(&self, mut event: Event) -> bool {
        loop {
            if self.cancelled() {
                return false;
            }
            match self.output.try_send(event) {
                Ok(()) => return true,
                Err(mpsc::error::TrySendError::Closed(_)) => return false,
                Err(mpsc::error::TrySendError::Full(e)) => {
                    event = e;
                    thread::sleep(Duration::from_millis(10));
                }
            }
        }
    }
}
pub trait Backend {
    fn generate(&mut self, request: Generation, sink: &Sink) -> Result<(), Failure>;
}
pub struct Worker {
    pub sender: mpsc::Sender<Job>,
    pub thread: thread::JoinHandle<()>,
}
pub fn start<B: Backend + 'static>(
    factory: impl FnOnce() -> anyhow::Result<B> + Send + 'static,
    shutdown: Arc<AtomicBool>,
) -> anyhow::Result<Worker> {
    let (sender, mut receiver) = mpsc::channel::<Job>(8);
    let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel(1);
    let worker = thread::Builder::new()
        .name("qwen-inference".into())
        .spawn(move || {
            let mut backend = match factory() {
                Ok(b) => {
                    let _ = ready_tx.send(Ok(()));
                    b
                }
                Err(e) => {
                    let _ = ready_tx.send(Err(e));
                    return;
                }
            };
            while let Some(job) = receiver.blocking_recv() {
                let sink = Sink {
                    output: job.output,
                    shutdown: shutdown.clone(),
                };
                if sink.cancelled() {
                    continue;
                }
                if let Err(error) = backend.generate(job.request, &sink) {
                    sink.send(Event::Failed(error));
                }
            }
        })?;
    match ready_rx.recv()? {
        Ok(()) => Ok(Worker {
            sender,
            thread: worker,
        }),
        Err(error) => {
            let _ = worker.join();
            Err(error)
        }
    }
}

pub struct QwenBackend {
    model: Model,
    tokenizer: QwenTokenizer,
    prompt: Prompt,
    ctx_size: usize,
    stop_ids: Vec<u64>,
}
impl QwenBackend {
    pub fn load(path: PathBuf, options: InferenceOptions, ctx_size: usize) -> anyhow::Result<Self> {
        let tokenizer = QwenTokenizer::from_gguf(&path)?
            .ok_or_else(|| anyhow::anyhow!("checkpoint has no tokenizer"))?;
        let template = tokenizer
            .chat_template()
            .ok_or_else(|| anyhow::anyhow!("checkpoint has no tokenizer.chat_template"))?;
        let prompt = Prompt::new(template.into())?;
        // Exercise the template before expensive compilation and before listening.
        prompt
            .render(
                &[serde_json::json!({"role":"user","content":"Hello"})],
                &[],
                false,
            )
            .map_err(anyhow::Error::msg)?;
        let model = Model::load_gguf_with_options(path, options)?;
        anyhow::ensure!(
            ctx_size > 0 && ctx_size <= model.config().max_position_embeddings,
            "--ctx-size must be between 1 and {}",
            model.config().max_position_embeddings
        );
        let mut stop_ids: Vec<u64> = model.config().eos_token_id.into_iter().collect();
        for token in ["<|im_end|>", "<|endoftext|>"] {
            if let Ok(id) = tokenizer.special_token_id(token) {
                stop_ids.push(id);
            }
        }
        Ok(Self {
            model,
            tokenizer,
            prompt,
            ctx_size,
            stop_ids,
        })
    }
}
impl Backend for QwenBackend {
    fn generate(&mut self, request: Generation, sink: &Sink) -> Result<(), Failure> {
        let text = self
            .prompt
            .render(&request.messages, &request.tools, request.thinking)
            .map_err(Failure::invalid)?;
        let tokens = self
            .tokenizer
            .encode_without_special_tokens(&text)
            .map_err(|e| Failure::invalid(e.to_string()))?;
        if tokens.is_empty() || tokens.len() >= self.ctx_size {
            return Err(Failure::invalid(format!(
                "prompt length {} exceeds the configured context size ({})or is empty",
                tokens.len(),
                self.ctx_size
            )));
        }
        let max_tokens = request
            .max_tokens
            .unwrap_or_else(|| 512.min(self.ctx_size - tokens.len()));
        let capacity = tokens
            .len()
            .checked_add(max_tokens)
            .filter(|&n| n <= self.ctx_size)
            .ok_or_else(|| Failure::invalid("prompt plus output limit exceeds --ctx-size"))?;
        if sink.cancelled() {
            return Ok(());
        }
        let internal = |e: anyhow::Error| Failure::internal(format!("inference failed: {e:#}"));
        let mut session = self.model.session(capacity).map_err(internal)?;
        if !sink.send(Event::Started) {
            return Ok(());
        }
        let mut next = session.prefill_next_token(&tokens).map_err(internal)?;
        let mut decoder = self.tokenizer.decode_stream(false);
        let mut stop = StopFilter::new(request.stops);
        let mut parser = OutputParser::new(request.thinking, request.tools, request.parallel);
        let mut generated = 0;
        let mut length = true;
        for index in 0..max_tokens {
            if sink.cancelled() {
                return Ok(());
            }
            if index > 0 {
                next = session.decode_next_token(next).map_err(internal)?;
            }
            if sink.cancelled() {
                return Ok(());
            }
            generated += 1;
            if self.stop_ids.contains(&next) {
                length = false;
                break;
            }
            let chunk = decoder
                .step(u32::try_from(next).map_err(|e| Failure::internal(e.to_string()))?)
                .map_err(|e| Failure::internal(e.to_string()))?
                .unwrap_or_default();
            let text = stop.push(&chunk);
            for delta in parser.push(&text).map_err(Failure::internal)? {
                if !sink.send(Event::Delta(delta)) {
                    return Ok(());
                }
            }
            if stop.stopped {
                length = false;
                break;
            }
        }
        for delta in parser.push(&stop.finish()).map_err(Failure::internal)? {
            if !sink.send(Event::Delta(delta)) {
                return Ok(());
            }
        }
        for delta in parser.finish(length).map_err(Failure::internal)? {
            if !sink.send(Event::Delta(delta)) {
                return Ok(());
            }
        }
        let reason = if length {
            "length"
        } else if parser.calls > 0 {
            "tool_calls"
        } else {
            "stop"
        };
        sink.send(Event::Done {
            reason,
            prompt_tokens: tokens.len(),
            completion_tokens: generated,
        });
        Ok(())
    }
}
