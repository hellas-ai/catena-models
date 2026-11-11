//! One owning GPU thread, receiving protocol-independent conversations.
use super::{
    output::{OutputParser, StopFilter},
    prefix_cache::{self, Entry, PREFILL_BLOCK, PrefixCache},
    prompt::Prompt,
    tools::{Call, Tool},
};
use catena_models::{
    catena::InferenceOptions,
    model::{Model, Session, SessionCheckpoint},
    tokenizer::QwenTokenizer,
};
use serde_json::Value;
use std::{
    path::PathBuf,
    rc::Rc,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};
use tokio::sync::mpsc;

const DEFAULT_MAX_OUTPUT_TOKENS: usize = 32 * 1024;

#[derive(Clone, Debug)]
pub struct Generation {
    pub messages: Vec<Value>,
    pub tools: Vec<Tool>,
    pub thinking: bool,
    pub parallel: bool,
    pub max_tokens: Option<usize>,
    pub stops: Vec<String>,
    pub structured: Option<Arc<super::structured::StructuredOutput>>,
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
        reasoning_tokens: usize,
        cached_tokens: usize,
        cache_write_tokens: usize,
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
    prefix_cache: PrefixCache<SessionCheckpoint>,
    prefix_cache_enabled: bool,
}
impl QwenBackend {
    pub fn load(
        path: PathBuf,
        options: InferenceOptions,
        ctx_size: usize,
        prefix_cache_enabled: bool,
    ) -> anyhow::Result<Self> {
        let tokenizer = QwenTokenizer::from_gguf(&path)?
            .ok_or_else(|| anyhow::anyhow!("checkpoint has no tokenizer"))?;
        let template = tokenizer
            .chat_template()
            .ok_or_else(|| anyhow::anyhow!("checkpoint has no tokenizer.chat_template"))?;
        let prompt = Prompt::new(template.into())?
            .with_special_tokens(tokenizer.bos_token(), tokenizer.eos_token());
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
            prefix_cache: PrefixCache::default(),
            prefix_cache_enabled,
        })
    }
}
struct Attempt {
    deltas: Vec<Delta>,
    reason: &'static str,
    prompt_tokens: usize,
    completion_tokens: usize,
    reasoning_tokens: usize,
    cached_tokens: usize,
    cache_write_tokens: usize,
}

/// Cold and warm requests execute identical absolute token blocks. Never rewind
/// a session by changing its position: DeltaNet state cannot be rewound that way.
fn prefill<'model>(
    model: &'model Model,
    cache: &mut PrefixCache<SessionCheckpoint>,
    enabled: bool,
    tokens: &[u64],
    capacity: usize,
    stable: usize,
    sink: &Sink,
) -> anyhow::Result<Option<(Session<'model>, u64, usize, usize)>> {
    if sink.cancelled() {
        return Ok(None);
    }
    let recent = prefix_cache::block_boundary(tokens.len());
    // If we need a new stable checkpoint, don't jump past its boundary.
    let limit = if stable > 0 && cache.exact(&tokens[..stable], capacity).is_none() {
        stable
    } else {
        tokens.len()
    };
    let hit = enabled
        .then(|| cache.matching(tokens, capacity, limit).cloned())
        .flatten();
    let mut copy_time = Duration::ZERO;
    let mut prefill_time = Duration::ZERO;
    let (mut session, mut next, cached) = if let Some(entry) = hit {
        let start = Instant::now();
        let restored = model.resume_checkpoint(&entry.state);
        copy_time += start.elapsed();
        match restored {
            Ok(session) => {
                let position = entry.tokens.len();
                if position == stable {
                    cache.stable = Some(entry.clone());
                }
                if position == recent {
                    cache.recent = Some(entry.clone());
                }
                (session, entry.next, position)
            }
            Err(error) => {
                tracing::warn!(target: "catena_models::server::prefix_cache", %error,
                    "checkpoint restore failed; falling back to cold prefill");
                cache.clear();
                (model.session(capacity)?, 0, 0)
            }
        }
    } else {
        (model.session(capacity)?, 0, 0)
    };
    let mut written_through = cached;
    while session.position() < tokens.len() {
        if sink.cancelled() {
            return Ok(None);
        }
        let position = session.position();
        let end = ((position / PREFILL_BLOCK + 1) * PREFILL_BLOCK).min(tokens.len());
        let start = Instant::now();
        next = session.append_next_token(&tokens[position..end])?;
        prefill_time += start.elapsed();
        if sink.cancelled() {
            return Ok(None);
        }
        if enabled && end > 0 && (end == stable || end == recent) {
            let entry = if let Some(entry) = cache.exact(&tokens[..end], capacity) {
                entry.clone()
            } else {
                // Move the working buffers into the slot before making a copy.
                // Eviction first bounds GPU memory to two snapshots + one session.
                if end == stable {
                    cache.stable = None;
                }
                if end == recent {
                    cache.recent = None;
                }
                let entry = Rc::new(Entry {
                    tokens: tokens[..end].to_vec(),
                    capacity,
                    state: session.into_checkpoint()?,
                    next,
                });
                if end == stable {
                    cache.stable = Some(entry.clone());
                }
                if end == recent {
                    cache.recent = Some(entry.clone());
                }
                let start = Instant::now();
                session = model.resume_checkpoint(&entry.state)?;
                copy_time += start.elapsed();
                written_through = end;
                entry
            };
            if end == stable {
                cache.stable = Some(entry.clone());
            }
            if end == recent {
                cache.recent = Some(entry);
            }
        }
    }
    tracing::debug!(target: "catena_models::server::prefix_cache", hit = cached > 0,
        cached_tokens = cached, prompt_tokens = tokens.len(), capacity,
        cache_write_tokens = written_through - cached,
        copy_ms = copy_time.as_secs_f64() * 1000.,
        prefill_ms = prefill_time.as_secs_f64() * 1000., "prompt prefill");
    Ok(Some((session, next, cached, written_through - cached)))
}

impl QwenBackend {
    fn generate_once(
        &mut self,
        request: &Generation,
        sink: &Sink,
        started: bool,
    ) -> Result<Option<Attempt>, Failure> {
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
            .unwrap_or_else(|| DEFAULT_MAX_OUTPUT_TOKENS.min(self.ctx_size - tokens.len()));
        let required = tokens
            .len()
            .checked_add(max_tokens)
            .filter(|&n| n <= self.ctx_size)
            .ok_or_else(|| Failure::invalid("prompt plus output limit exceeds --ctx-size"))?;
        let capacity = prefix_cache::capacity_bucket(required, self.ctx_size);
        let stable = if self.prefix_cache_enabled {
            if let Some(user) = request.messages.iter().rposition(|m| m["role"] == "user") {
                let text = self
                    .prompt
                    .render_with_generation_prompt(
                        &request.messages[..=user],
                        &request.tools,
                        request.thinking,
                        false,
                    )
                    .map_err(Failure::invalid)?;
                let stable_tokens = self
                    .tokenizer
                    .encode_without_special_tokens(&text)
                    .map_err(|e| Failure::invalid(e.to_string()))?;
                prefix_cache::block_boundary(prefix_cache::common_prefix(&stable_tokens, &tokens))
            } else {
                0
            }
        } else {
            0
        };
        if sink.cancelled() {
            return Ok(None);
        }
        let internal = |e: anyhow::Error| Failure::internal(format!("inference failed: {e:#}"));
        if !started && !sink.send(Event::Started) {
            return Ok(None);
        }
        let Some((mut session, mut next, cached_tokens, cache_write_tokens)) = prefill(
            &self.model,
            &mut self.prefix_cache,
            self.prefix_cache_enabled,
            &tokens,
            capacity,
            stable,
            sink,
        )
        .map_err(internal)?
        else {
            return Ok(None);
        };
        let mut decoder = self.tokenizer.decode_stream(false);
        let mut stop = StopFilter::new(request.stops.clone());
        let mut parser =
            OutputParser::new(request.thinking, request.tools.clone(), request.parallel);
        let mut deltas = Vec::new();
        let mut emit = |delta: Delta| {
            if request.structured.is_some() {
                deltas.push(delta);
                !sink.cancelled()
            } else {
                sink.send(Event::Delta(delta))
            }
        };
        let mut generated = 0;
        let mut reasoning_tokens = 0;
        let mut length = true;
        for index in 0..max_tokens {
            if sink.cancelled() {
                return Ok(None);
            }
            if index > 0 {
                next = session.decode_next_token(next).map_err(internal)?;
            }
            if sink.cancelled() {
                return Ok(None);
            }
            generated += 1;
            if self.stop_ids.contains(&next) {
                length = false;
                break;
            }
            let was_reasoning = parser.is_reasoning();
            let chunk = decoder
                .step(u32::try_from(next).map_err(|e| Failure::internal(e.to_string()))?)
                .map_err(|e| Failure::internal(e.to_string()))?
                .unwrap_or_default();
            let text = stop.push(&chunk);
            for delta in parser.push(&text).map_err(Failure::internal)? {
                if !emit(delta) {
                    return Ok(None);
                }
            }
            if was_reasoning || parser.is_reasoning() {
                reasoning_tokens += 1;
            }
            if stop.stopped {
                length = false;
                break;
            }
        }
        for delta in parser.push(&stop.finish()).map_err(Failure::internal)? {
            if !emit(delta) {
                return Ok(None);
            }
        }
        for delta in parser.finish(length).map_err(Failure::internal)? {
            if !emit(delta) {
                return Ok(None);
            }
        }
        let reason = if length {
            "length"
        } else if parser.calls > 0 {
            "tool_calls"
        } else {
            "stop"
        };
        Ok(Some(Attempt {
            deltas,
            reason,
            prompt_tokens: tokens.len(),
            completion_tokens: generated,
            reasoning_tokens,
            cached_tokens,
            cache_write_tokens,
        }))
    }
}

impl Backend for QwenBackend {
    fn generate(&mut self, mut request: Generation, sink: &Sink) -> Result<(), Failure> {
        let budget = request.max_tokens.unwrap_or(DEFAULT_MAX_OUTPUT_TOKENS);
        let mut prompt_tokens = 0;
        let mut completion_tokens = 0;
        let mut reasoning_tokens = 0;
        let mut cached_tokens = 0;
        let mut cache_write_tokens = 0;
        for attempt in 0..3 {
            let result = self
                .generate_once(&request, sink, attempt > 0)
                .map_err(|error| {
                    if !error.invalid_request {
                        self.prefix_cache.clear();
                    }
                    if attempt == 0 {
                        error
                    } else {
                        Failure::internal(format!(
                            "structured output correction failed: {}",
                            error.message
                        ))
                    }
                })?;
            let Some(result) = result else {
                return Ok(());
            };
            prompt_tokens += result.prompt_tokens;
            completion_tokens += result.completion_tokens;
            reasoning_tokens += result.reasoning_tokens;
            cached_tokens += result.cached_tokens;
            cache_write_tokens += result.cache_write_tokens;
            if let Some(format) = &request.structured {
                let tool_turn = result
                    .deltas
                    .iter()
                    .any(|delta| matches!(delta, Delta::Tool { .. }));
                if result.reason != "length" && !tool_turn {
                    let answer: String = result
                        .deltas
                        .iter()
                        .filter_map(|delta| {
                            if let Delta::Content(text) = delta {
                                Some(text.as_str())
                            } else {
                                None
                            }
                        })
                        .collect();
                    if let Err(error) = format.validate(&answer) {
                        let remaining = budget.saturating_sub(completion_tokens);
                        if attempt == 2 || remaining == 0 {
                            return Err(Failure::internal(error));
                        }
                        tracing::debug!(target: "catena_models::server::structured", attempt=attempt+1, %error, "correcting structured output");
                        // Keep failed candidates out of the replayed prompt: small
                        // models otherwise tend to repeat the same invalid value.
                        let correction = format.correction_instruction();
                        request
                            .messages
                            .push(serde_json::json!({"role":"user","content":correction}));
                        request.max_tokens = Some(remaining);
                        continue;
                    }
                }
            }
            for delta in result.deltas {
                if !sink.send(Event::Delta(delta)) {
                    return Ok(());
                }
            }
            sink.send(Event::Done {
                reason: result.reason,
                prompt_tokens,
                completion_tokens,
                reasoning_tokens,
                cached_tokens,
                cache_write_tokens,
            });
            return Ok(());
        }
        unreachable!("structured correction loop always returns")
    }
}

#[cfg(test)]
mod prefix_tests {
    use super::*;

    fn load() -> anyhow::Result<Model> {
        let _ = tracing_subscriber::fmt()
            .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
            .with_test_writer()
            .try_init();
        let selector = std::env::var("CATENA_PREFIX_MODEL")?;
        let options = InferenceOptions::default()
            .with_wmma_prefill(std::env::var("CATENA_PREFIX_WMMA").as_deref() != Ok("0"));
        if std::path::Path::new(&selector).exists() {
            Model::load_gguf_with_options(selector, options)
        } else {
            Model::load_hf_with_options(&selector, options)
        }
    }

    fn bits(logits: Vec<f32>) -> Vec<u32> {
        logits.into_iter().map(f32::to_bits).collect()
    }

    #[test]
    #[ignore = "requires GPU and CATENA_PREFIX_MODEL (dense or MoE BF16)"]
    fn serving_prefix_cache_bitwise_parity() -> anyhow::Result<()> {
        let model = load()?;
        let (output, _receiver) = mpsc::channel(8);
        let sink = Sink {
            output,
            shutdown: Arc::new(AtomicBool::new(false)),
        };
        let mut cache = PrefixCache::default();
        let mut tokens: Vec<_> = (0..2051).map(|i| 100 + i % 97).collect();
        // Both slots, including the exact-checkpoint/no-suffix path.
        for (enabled, expected) in [(true, 0), (true, 2048), (false, 0)] {
            let (mut session, next, cached, _) = prefill(
                &model,
                &mut cache,
                enabled,
                &tokens[..2048],
                4096,
                1024,
                &sink,
            )?
            .unwrap();
            assert_eq!(cached, expected);
            let logits = bits(session.decode_logits(next)?);
            let (mut cold, cold_next, _, _) = prefill(
                &model,
                &mut PrefixCache::default(),
                false,
                &tokens[..2048],
                4096,
                1024,
                &sink,
            )?
            .unwrap();
            assert_eq!(next, cold_next);
            assert_eq!(logits, bits(cold.decode_logits(cold_next)?));
        }
        // Mutating a restored decode session above must leave the prompt frozen.
        for (end, changed, capacity, expected) in [
            (2049, false, 4096, 2048),
            (2051, false, 4096, 2048),
            (2051, true, 4096, 1024),
            (2051, false, 8192, 0),
        ] {
            if changed {
                tokens[1400] += 1;
            } // rewritten earlier reasoning
            let (mut warm, next, cached, _) = prefill(
                &model,
                &mut cache,
                true,
                &tokens[..end],
                capacity,
                1024,
                &sink,
            )?
            .unwrap();
            assert_eq!(cached, expected);
            let logits = bits(warm.decode_logits(next)?);
            drop(warm);
            let (mut cold, cold_next, _, _) = prefill(
                &model,
                &mut PrefixCache::default(),
                false,
                &tokens[..end],
                capacity,
                1024,
                &sink,
            )?
            .unwrap();
            assert_eq!(next, cold_next);
            assert_eq!(logits, bits(cold.decode_logits(cold_next)?));
        }
        // Unrelated short prompts and cancelled requests leave both slots intact.
        let stable = cache.stable.as_ref().unwrap().clone();
        let recent = cache.recent.as_ref().unwrap().clone();
        drop(prefill(&model, &mut cache, true, &[1, 2, 3], 16, 0, &sink)?);
        assert!(Rc::ptr_eq(cache.stable.as_ref().unwrap(), &stable));
        assert!(Rc::ptr_eq(cache.recent.as_ref().unwrap(), &recent));
        sink.shutdown.store(true, Ordering::Relaxed);
        assert!(prefill(&model, &mut cache, true, &tokens, 8192, 1024, &sink)?.is_none());
        sink.shutdown.store(false, Ordering::Relaxed);
        cache.clear();
        let (_, _, cached, _) =
            prefill(&model, &mut cache, true, &tokens, 8192, 1024, &sink)?.unwrap();
        assert_eq!(cached, 0);
        Ok(())
    }

    #[test]
    #[ignore = "requires GPU and CATENA_PREFIX_MODEL; measures 4k/16k/32k prefix reuse"]
    fn serving_prefix_cache_benchmark() -> anyhow::Result<()> {
        let model = load()?;
        let (output, _receiver) = mpsc::channel(8);
        let sink = Sink {
            output,
            shutdown: Arc::new(AtomicBool::new(false)),
        };
        let lengths =
            std::env::var("CATENA_PREFIX_LENGTHS").unwrap_or_else(|_| "4096,16384,32768".into());
        for length in lengths.split(',') {
            let length: usize = length.parse()?;
            anyhow::ensure!(
                length > 0 && length.is_multiple_of(PREFILL_BLOCK),
                "benchmark lengths must be full prefill blocks"
            );
            let tokens: Vec<_> = (0..length + 32).map(|i| 100 + i as u64 % 97).collect();
            let capacity = prefix_cache::capacity_bucket(
                tokens.len() + 512,
                model.config().max_position_embeddings,
            );
            let mut cache = PrefixCache::default();
            drop(prefill(
                &model,
                &mut cache,
                true,
                &tokens[..length],
                capacity,
                length,
                &sink,
            )?);
            let start = Instant::now();
            let (session, next, cached, _) =
                prefill(&model, &mut cache, true, &tokens, capacity, length, &sink)?.unwrap();
            let warm = start.elapsed();
            drop(session);
            cache.clear();
            let start = Instant::now();
            let (_, cold_next, _, _) =
                prefill(&model, &mut cache, false, &tokens, capacity, length, &sink)?.unwrap();
            let cold = start.elapsed();
            assert_eq!(next, cold_next);
            assert_eq!(cached, length);
            eprintln!(
                "prefix={length} suffix=32 cached={cached} warm={warm:?} cold={cold:?} speedup={:.2}",
                cold.as_secs_f64() / warm.as_secs_f64()
            );
        }
        Ok(())
    }
}
