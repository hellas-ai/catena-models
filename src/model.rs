use std::path::Path;

use anyhow::{Context, Result, ensure};
use catena_lang::{
    codegen::GpuDialect,
    runtime::{MemOwn, Runtime, Value},
    stdlib,
};

use crate::{
    catena::model_source,
    checkpoint::Checkpoint,
    rope,
    spec::{
        KV_CACHE_ELEMENTS_PER_TOKEN, MAX_POSITION_EMBEDDINGS, MODEL_INPUT_COUNT, VOCAB_SIZE,
        WEIGHT_COUNT, weight_specs,
    },
};

/// Loaded Qwen3 runtime and persistent GPU weight allocations.
pub struct Model {
    runtime: Runtime,
    weights: Vec<MemOwn>,
}

impl Model {
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let model_source = model_source();
        let mut sources = stdlib::sources().collect::<Vec<&str>>();
        sources.push(model_source.as_str());
        let runtime = Runtime::from_sources(sources, GpuDialect::Hip)
            .context("failed to compile the Qwen3 Catena GPU program")?;
        let checkpoint = Checkpoint::open(model_dir)?;
        let mut weights = Vec::with_capacity(WEIGHT_COUNT);
        for spec in weight_specs() {
            let values = checkpoint
                .load_bf16(&spec)
                .with_context(|| format!("failed to load {}", spec.name))?;
            weights.push(
                runtime
                    .mem_u16(&values)
                    .with_context(|| format!("failed to upload {}", spec.name))?,
            );
        }
        ensure!(
            weights.len() == WEIGHT_COUNT,
            "internal weight ABI mismatch"
        );
        Ok(Self { runtime, weights })
    }

    /// Run a full causal prefill and return final-position vocabulary logits.
    pub fn logits(&self, tokens: &[u64]) -> Result<Vec<f32>> {
        let cache = self.allocate_cache(tokens.len())?;
        let (_cache, logits, _next_token) = self.cached_outputs(tokens, 0, tokens.len(), cache)?;
        read_logits(logits)
    }

    pub fn session(&self, capacity: usize) -> Result<Session<'_>> {
        ensure!(capacity > 0, "cache capacity must be non-zero");
        ensure!(
            capacity <= MAX_POSITION_EMBEDDINGS,
            "cache capacity {capacity} exceeds the model maximum {MAX_POSITION_EMBEDDINGS}"
        );
        Ok(Session {
            model: self,
            cache: Some(self.allocate_cache(capacity)?),
            position: 0,
            capacity,
        })
    }

    fn allocate_cache(&self, capacity: usize) -> Result<MemOwn> {
        let elements = capacity
            .checked_mul(KV_CACHE_ELEMENTS_PER_TOKEN)
            .context("KV cache element count overflowed")?;
        self.runtime
            .mem_f32_zeroed(elements)
            .context("failed to allocate the KV cache")
    }

    fn cached_outputs(
        &self,
        tokens: &[u64],
        start: usize,
        capacity: usize,
        cache: MemOwn,
    ) -> Result<(MemOwn, MemOwn, u64)> {
        ensure!(!tokens.is_empty(), "the token prefix must not be empty");
        ensure!(
            start
                .checked_add(tokens.len())
                .is_some_and(|end| end <= capacity),
            "token range {start}..{} exceeds cache capacity {capacity}",
            start.saturating_add(tokens.len())
        );
        ensure!(
            tokens.iter().all(|&token| token < VOCAB_SIZE as u64),
            "token prefix contains an ID outside the {VOCAB_SIZE}-token vocabulary"
        );

        let token_buffer = self.runtime.mem_u64(tokens)?;
        let rope_values = rope::table_range(start, tokens.len());
        let rope_buffer = self.runtime.mem_f32(&rope_values)?;
        let inputs = [token_buffer.as_ref().into(), rope_buffer.as_ref().into()]
            .into_iter()
            .chain([
                cache.into(),
                (capacity as u64).into(),
                (start as u64).into(),
            ])
            .chain(self.weights.iter().map(|memory| memory.as_ref().into()))
            .collect::<Vec<_>>();
        ensure!(
            inputs.len() == MODEL_INPUT_COUNT,
            "internal model ABI mismatch"
        );
        let mut outputs = self.runtime.exec_values("qwen3", inputs)?;
        ensure!(outputs.len() == 3, "internal model output ABI mismatch");
        let next_token_output = outputs.pop().expect("output count was validated");
        let logits_output = outputs.pop().expect("output count was validated");
        let cache_output = outputs.pop().expect("output count was validated");
        let Value::MemOwn(cache) = cache_output else {
            anyhow::bail!("qwen3 returned a non-memory cache: {cache_output:?}");
        };
        let Value::MemOwn(logits_output) = logits_output else {
            anyhow::bail!("qwen3 returned non-memory logits: {logits_output:?}");
        };
        let Value::MemOwn(next_token_output) = next_token_output else {
            anyhow::bail!("qwen3 returned a non-memory token: {next_token_output:?}");
        };
        let next_tokens = next_token_output.try_to_u64_vec()?;
        let [next_token] = next_tokens.as_slice() else {
            anyhow::bail!(
                "qwen3 returned {} next-token values, expected 1",
                next_tokens.len()
            );
        };
        ensure!(
            *next_token < VOCAB_SIZE as u64,
            "qwen3 returned next-token ID {next_token} outside the vocabulary"
        );
        Ok((cache, logits_output, *next_token))
    }

    pub fn next_token(&self, tokens: &[u64]) -> Result<u64> {
        let cache = self.allocate_cache(tokens.len())?;
        let (_cache, _logits, next_token) = self.cached_outputs(tokens, 0, tokens.len(), cache)?;
        Ok(next_token)
    }
}

/// Persistent, append-only KV-cache state for one generation sequence.
pub struct Session<'model> {
    model: &'model Model,
    cache: Option<MemOwn>,
    position: usize,
    capacity: usize,
}

impl Session<'_> {
    pub fn position(&self) -> usize {
        self.position
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn prefill_logits(&mut self, tokens: &[u64]) -> Result<Vec<f32>> {
        ensure!(self.position == 0, "a session can only be prefilled once");
        let (logits, _next_token) = self.forward(tokens)?;
        read_logits(logits)
    }

    pub fn decode_logits(&mut self, token: u64) -> Result<Vec<f32>> {
        ensure!(self.position > 0, "decode requires a completed prefill");
        let (logits, _next_token) = self.forward(std::slice::from_ref(&token))?;
        read_logits(logits)
    }

    pub fn prefill_next_token(&mut self, tokens: &[u64]) -> Result<u64> {
        ensure!(self.position == 0, "a session can only be prefilled once");
        let (_logits, next_token) = self.forward(tokens)?;
        Ok(next_token)
    }

    pub fn decode_next_token(&mut self, token: u64) -> Result<u64> {
        ensure!(self.position > 0, "decode requires a completed prefill");
        let (_logits, next_token) = self.forward(std::slice::from_ref(&token))?;
        Ok(next_token)
    }

    fn forward(&mut self, tokens: &[u64]) -> Result<(MemOwn, u64)> {
        let cache = self
            .cache
            .take()
            .context("the session cache is unavailable after a failed execution")?;
        let (cache, logits, next_token) =
            self.model
                .cached_outputs(tokens, self.position, self.capacity, cache)?;
        self.cache = Some(cache);
        self.position += tokens.len();
        Ok((logits, next_token))
    }
}

fn read_logits(logits_output: MemOwn) -> Result<Vec<f32>> {
    let logits = logits_output.try_to_f32_vec()?;
    ensure!(
        logits.len() == VOCAB_SIZE,
        "qwen3 returned {} logits, expected {VOCAB_SIZE}",
        logits.len()
    );
    ensure!(
        logits.iter().all(|value| value.is_finite()),
        "qwen3 returned non-finite logits"
    );
    Ok(logits)
}
