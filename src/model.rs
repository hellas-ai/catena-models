use std::path::Path;

use anyhow::{Context, Result, ensure};
use catena_lang::{
    codegen::GpuDialect,
    runtime::{MemOwn, Runtime, Value},
    stdlib,
};

use crate::{
    catena::MODEL_SOURCE,
    checkpoint::{Checkpoint, GgufDtype, TensorView, read_gguf_model_config},
    spec::{ModelArchitecture, ModelConfig, weight_specs},
};

/// Loaded Qwen3 runtime and registered native-dtype GGUF mapping.
pub struct Model {
    runtime: Runtime,
    weights: Vec<TensorView>,
    checkpoint: Checkpoint,
    config: ModelConfig,
    parameter_count: usize,
    weight_byte_len: usize,
}

impl Model {
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self> {
        Self::load_gguf(model_dir.as_ref().join("model-bf16.gguf"))
    }

    pub fn load_gguf(gguf_path: impl AsRef<Path>) -> Result<Self> {
        let gguf_path = gguf_path.as_ref();
        let config = read_gguf_model_config(gguf_path)?;
        let specs = weight_specs(&config)?;
        let parameter_count = specs
            .iter()
            .try_fold(0_usize, |total, spec| {
                total.checked_add(spec.element_count())
            })
            .context("model parameter count overflowed")?;
        let runtime =
            Runtime::from_sources(stdlib::sources().chain([MODEL_SOURCE]), GpuDialect::Hip)
                .context("failed to compile the Qwen3 Catena GPU program")?;
        let checkpoint = Checkpoint::open(gguf_path, &config)?;
        let mut weights = Vec::with_capacity(specs.len());
        let mut mapped_bf16_bytes = 0_usize;
        let mut mapped_f32_bytes = 0_usize;
        for spec in specs {
            let view = checkpoint
                .tensor_view(&spec)
                .with_context(|| format!("failed to locate {}", spec.name))?;
            match view.dtype {
                GgufDtype::Bf16 => mapped_bf16_bytes += view.byte_len,
                GgufDtype::F32 => mapped_f32_bytes += view.byte_len,
            }
            weights.push(view);
        }
        ensure!(
            weights.len() + 4 == config_model_input_count(&config),
            "internal model weight ABI mismatch"
        );
        eprintln!(
            "registered {} GGUF file(s), {:.2} GiB total; {:.2} GiB BF16 and {:.2} MiB F32 mapped zero-copy in native storage",
            checkpoint.file_count(),
            checkpoint.registered_byte_len() as f64 / (1_u64 << 30) as f64,
            mapped_bf16_bytes as f64 / (1_u64 << 30) as f64,
            mapped_f32_bytes as f64 / (1_u64 << 20) as f64,
        );
        Ok(Self {
            runtime,
            weights,
            checkpoint,
            config,
            parameter_count,
            weight_byte_len: mapped_bf16_bytes + mapped_f32_bytes,
        })
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    pub fn display_name(&self) -> &str {
        self.config
            .model_name
            .as_deref()
            .unwrap_or(match self.config.architecture {
                ModelArchitecture::Qwen3Moe => "Qwen3 MoE",
                ModelArchitecture::Qwen35Moe => "Qwen3.5/3.6 MoE",
            })
    }

    pub fn parameter_count(&self) -> usize {
        self.parameter_count
    }

    pub fn weight_byte_len(&self) -> usize {
        self.weight_byte_len
    }

    /// Run a full causal prefill and return final-position vocabulary logits.
    pub fn logits(&self, tokens: &[u64]) -> Result<Vec<f32>> {
        self.ensure_execution_supported()?;
        let cache = self.allocate_cache(tokens.len())?;
        let (_cache, logits, _next_token) = self.cached_outputs(tokens, 0, tokens.len(), cache)?;
        read_logits(logits, self.config.vocabulary_size)
    }

    pub fn session(&self, capacity: usize) -> Result<Session<'_>> {
        ensure!(capacity > 0, "cache capacity must be non-zero");
        ensure!(
            capacity <= self.config.max_position_embeddings,
            "cache capacity {capacity} exceeds the model maximum {}",
            self.config.max_position_embeddings
        );
        self.ensure_execution_supported()?;
        Ok(Session {
            model: self,
            cache: Some(self.allocate_cache(capacity)?),
            position: 0,
            capacity,
        })
    }

    fn allocate_cache(&self, capacity: usize) -> Result<MemOwn> {
        let elements = capacity
            .checked_mul(
                self.config.num_layers * 2 * self.config.num_key_value_heads * self.config.head_dim,
            )
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
        self.ensure_execution_supported()?;
        ensure!(!tokens.is_empty(), "the token prefix must not be empty");
        ensure!(
            start
                .checked_add(tokens.len())
                .is_some_and(|end| end <= capacity),
            "token range {start}..{} exceeds cache capacity {capacity}",
            start.saturating_add(tokens.len())
        );
        ensure!(
            tokens
                .iter()
                .all(|&token| token < self.config.vocabulary_size as u64),
            "token prefix contains an ID outside the {}-token vocabulary",
            self.config.vocabulary_size
        );

        let token_buffer = self.runtime.mem_u64(tokens)?;
        let mut inputs = Vec::with_capacity(config_model_input_count(&self.config));
        inputs.extend([
            token_buffer.as_ref().into(),
            cache.into(),
            (capacity as u64).into(),
            (start as u64).into(),
        ]);
        for weight in &self.weights {
            inputs.push(self.checkpoint.mapped_ref(weight)?.into());
        }
        ensure!(
            inputs.len() == config_model_input_count(&self.config),
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
            *next_token < self.config.vocabulary_size as u64,
            "qwen3 returned next-token ID {next_token} outside the vocabulary"
        );
        Ok((cache, logits_output, *next_token))
    }

    pub fn next_token(&self, tokens: &[u64]) -> Result<u64> {
        self.ensure_execution_supported()?;
        let cache = self.allocate_cache(tokens.len())?;
        let (_cache, _logits, next_token) = self.cached_outputs(tokens, 0, tokens.len(), cache)?;
        Ok(next_token)
    }

    fn ensure_execution_supported(&self) -> Result<()> {
        ensure!(
            self.config.architecture == ModelArchitecture::Qwen3Moe,
            "Qwen3.5/3.6 GGUF metadata and all {} tensors were validated, but the hybrid \
             linear-attention Catena program is not implemented yet",
            self.weights.len()
        );
        Ok(())
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
        read_logits(logits, self.model.config.vocabulary_size)
    }

    pub fn decode_logits(&mut self, token: u64) -> Result<Vec<f32>> {
        ensure!(self.position > 0, "decode requires a completed prefill");
        let (logits, _next_token) = self.forward(std::slice::from_ref(&token))?;
        read_logits(logits, self.model.config.vocabulary_size)
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

fn config_model_input_count(config: &ModelConfig) -> usize {
    // Header validation has already built the same inventory.
    4 + weight_specs(config).expect("validated model config").len()
}

fn read_logits(logits_output: MemOwn, vocabulary_size: usize) -> Result<Vec<f32>> {
    let logits = logits_output.try_to_f32_vec()?;
    ensure!(
        logits.len() == vocabulary_size,
        "qwen3 returned {} logits, expected {vocabulary_size}",
        logits.len(),
    );
    ensure!(
        logits.iter().all(|value| value.is_finite()),
        "qwen3 returned non-finite logits"
    );
    Ok(logits)
}
