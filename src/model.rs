use std::{collections::HashSet, env, path::Path};

use anyhow::{Context, Result, ensure};
use catena_lang::{
    codegen::GpuDialect,
    runtime::{Artifact, MemOwn, Runtime, Value},
};

use crate::{
    catena::{
        InferenceOptions, InferencePrecision, generated_model_source_for_options,
        generated_multimodal_source, runtime_sources_for_options,
    },
    checkpoint::{Checkpoint, GgufDtype, TensorView, read_gguf_model_config},
    spec::{ModelArchitecture, ModelConfig, weight_specs},
};

/// Loaded Qwen3 runtime and registered native-dtype GGUF mapping.
pub struct Model {
    runtime: Runtime,
    artifact: Artifact,
    weights: Vec<TensorView>,
    checkpoint: Checkpoint,
    config: ModelConfig,
    parameter_count: usize,
    weight_byte_len: usize,
    precision: InferencePrecision,
    dialect: GpuDialect,
    multimodal: bool,
    experimental_workspace_bytes: usize,
}

impl Model {
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self> {
        Self::load_gguf(model_dir.as_ref().join("model-bf16.gguf"))
    }

    pub fn load_gguf(gguf_path: impl AsRef<Path>) -> Result<Self> {
        Self::load_gguf_with_precision(gguf_path, InferencePrecision::F32)
    }

    pub fn load_gguf_with_precision(
        gguf_path: impl AsRef<Path>,
        precision: InferencePrecision,
    ) -> Result<Self> {
        Self::load_gguf_with_options(gguf_path, InferenceOptions::new(precision))
    }

    pub fn load_gguf_with_options(
        gguf_path: impl AsRef<Path>,
        options: InferenceOptions,
    ) -> Result<Self> {
        Self::load_gguf_with_options_and_dialect(gguf_path, options, configured_gpu_dialect()?)
    }

    pub fn load_gguf_with_options_and_dialect(
        gguf_path: impl AsRef<Path>,
        options: InferenceOptions,
        dialect: GpuDialect,
    ) -> Result<Self> {
        Self::load_impl(gguf_path, options, dialect, false)
    }

    /// Compile the additional embedding/MRoPE entrypoint for image-conditioned sessions.
    pub fn load_gguf_with_multimodal_options(
        gguf_path: impl AsRef<Path>,
        options: InferenceOptions,
    ) -> Result<Self> {
        Self::load_gguf_with_multimodal_options_and_dialect(
            gguf_path,
            options,
            configured_gpu_dialect()?,
        )
    }

    pub fn load_gguf_with_multimodal_options_and_dialect(
        gguf_path: impl AsRef<Path>,
        options: InferenceOptions,
        dialect: GpuDialect,
    ) -> Result<Self> {
        Self::load_impl(gguf_path, options, dialect, true)
    }

    fn load_impl(
        gguf_path: impl AsRef<Path>,
        options: InferenceOptions,
        dialect: GpuDialect,
        multimodal: bool,
    ) -> Result<Self> {
        ensure!(
            !options.fused_attention || dialect == GpuDialect::Hip,
            "experimental fused attention currently supports HIP only"
        );
        let precision = options.precision;
        let gguf_path = gguf_path.as_ref();
        let config = read_gguf_model_config(gguf_path)?;
        if options.codegen == catena_lang::codegen::CodegenKind::Experimental {
            crate::experimental::validate(&config, options, Some(dialect))?;
            ensure!(
                !multimodal,
                "experimental codegen does not yet support image inputs"
            );
        }
        let experimental_workspace_bytes = crate::experimental::workspace_bytes(&config, options)?;
        let specs = weight_specs(&config)?;
        let weight_count = specs.len();
        let mut physical_tensors = HashSet::new();
        let parameter_count = specs
            .iter()
            .filter(|spec| physical_tensors.insert(spec.gguf_name.as_str()))
            .try_fold(0_usize, |total, spec| {
                total.checked_add(spec.element_count())
            })
            .context("model parameter count overflowed")?;
        let model_source = if multimodal {
            generated_multimodal_source(&config, options)?
        } else {
            generated_model_source_for_options(&config, options)?
        };
        let (runtime, artifact) = crate::catena::load_runtime_with_codegen(
            runtime_sources_for_options(&model_source, options),
            dialect,
            options.codegen,
        )
        .context("failed to compile the Qwen3 Catena GPU program")?;
        let checkpoint = Checkpoint::open_with_dialect(gguf_path, &config, dialect)?;
        let mut weights = Vec::with_capacity(specs.len());
        let mut mapped_bf16_bytes = 0_usize;
        let mut mapped_f32_bytes = 0_usize;
        let mut counted_tensors = HashSet::new();
        for spec in specs {
            let view = checkpoint
                .tensor_view(&spec)
                .with_context(|| format!("failed to locate {}", spec.name))?;
            if counted_tensors.insert(spec.gguf_name) {
                match view.dtype {
                    GgufDtype::Bf16 => mapped_bf16_bytes += view.byte_len,
                    GgufDtype::F32 => mapped_f32_bytes += view.byte_len,
                }
            }
            weights.push(view);
        }
        ensure!(
            weights.len() == weight_count,
            "internal model weight ABI mismatch"
        );
        let weight_location = match dialect {
            GpuDialect::Hip => "mapped zero-copy",
            GpuDialect::Cuda => "copied to device memory",
        };
        eprintln!(
            "loaded {} GGUF file(s), {:.2} GiB total; {:.2} GiB BF16 and {:.2} MiB F32 {weight_location} in native storage",
            checkpoint.file_count(),
            checkpoint.storage_byte_len() as f64 / (1_u64 << 30) as f64,
            mapped_bf16_bytes as f64 / (1_u64 << 30) as f64,
            mapped_f32_bytes as f64 / (1_u64 << 20) as f64,
        );
        Ok(Self {
            runtime,
            artifact,
            weights,
            checkpoint,
            config,
            parameter_count,
            weight_byte_len: mapped_bf16_bytes + mapped_f32_bytes,
            precision,
            dialect,
            multimodal,
            experimental_workspace_bytes,
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
                ModelArchitecture::Qwen3Dense => "Qwen3 dense",
                ModelArchitecture::Qwen3Moe => "Qwen3 MoE",
                ModelArchitecture::Qwen35Dense => "Qwen3.5 dense",
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
        let mut session = self.session(tokens.len())?;
        session.prefill_logits(tokens)
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
            workspace: if self.experimental_workspace_bytes == 0 {
                None
            } else {
                Some(
                    self.runtime
                        .mem_f32(&vec![0.0; self.experimental_workspace_bytes / 4])?,
                )
            },
            position: 0,
            capacity,
            next_rope_position: None,
        })
    }

    fn allocate_cache(&self, capacity: usize) -> Result<MemOwn> {
        let elements = self.config.cache_elements(capacity)?;
        match self.precision {
            InferencePrecision::F32 => self.runtime.mem_f32(&vec![0.0; elements]),
            InferencePrecision::Bf16 => self.runtime.mem_u16(&vec![0; elements]),
        }
        .context("failed to allocate the model cache")
    }

    fn cached_outputs(
        &self,
        tokens: &[u64],
        start: usize,
        capacity: usize,
        cache: MemOwn,
        workspace: Option<MemOwn>,
    ) -> Result<(MemOwn, Option<MemOwn>, MemOwn, u64)> {
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
        let has_workspace = workspace.is_some();
        let model_input_count = 4 + usize::from(has_workspace) + self.weights.len();
        let mut inputs = Vec::with_capacity(model_input_count);
        inputs.extend([token_buffer.as_ref().into(), cache.into()]);
        if let Some(workspace) = workspace {
            inputs.push(workspace.into());
        }
        inputs.extend([(capacity as u64).into(), (start as u64).into()]);
        for weight in &self.weights {
            inputs.push(self.checkpoint.gpu_ref(weight)?.into());
        }
        ensure!(
            inputs.len() == model_input_count,
            "internal model ABI mismatch"
        );
        let mut outputs = self.artifact.exec_values("qwen3", inputs)?;
        let workspace = if has_workspace {
            ensure!(
                outputs.len() == 4,
                "internal experimental output ABI mismatch"
            );
            let Value::MemOwn(workspace) = outputs.remove(1) else {
                anyhow::bail!("experimental artifact returned non-memory workspace");
            };
            Some(workspace)
        } else {
            None
        };
        let (cache, logits, token) = Self::cached_result(outputs, self.config.vocabulary_size)?;
        Ok((cache, workspace, logits, token))
    }

    fn cached_result(
        mut outputs: Vec<Value<'_>>,
        vocabulary_size: usize,
    ) -> Result<(MemOwn, MemOwn, u64)> {
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
            *next_token < vocabulary_size as u64,
            "qwen3 returned next-token ID {next_token} outside the vocabulary"
        );
        Ok((cache, logits_output, *next_token))
    }

    pub fn gpu_dialect(&self) -> GpuDialect {
        self.dialect
    }

    fn token_embedding(&self, token: u64) -> Result<MemOwn> {
        let tokens = self.runtime.mem_u64(&[token])?;
        let mut output = self.artifact.exec_values(
            "qwen3.embedding",
            vec![
                tokens.as_ref().into(),
                self.checkpoint.gpu_ref(&self.weights[0])?.into(),
            ],
        )?;
        ensure!(output.len() == 1, "embedding ABI mismatch");
        match output.pop().unwrap() {
            Value::MemOwn(m) => Ok(m),
            _ => anyhow::bail!("expected embedding memory"),
        }
    }

    fn cached_embedding_outputs(
        &self,
        hidden: MemOwn,
        positions: [u64; 3],
        start: usize,
        capacity: usize,
        cache: MemOwn,
    ) -> Result<(MemOwn, MemOwn, u64)> {
        let positions = self.runtime.mem_u64(&positions)?;
        let mut inputs = vec![
            hidden.into(),
            positions.as_ref().into(),
            cache.into(),
            (capacity as u64).into(),
            (start as u64).into(),
        ];
        for weight in &self.weights {
            inputs.push(self.checkpoint.gpu_ref(weight)?.into());
        }
        Self::cached_result(
            self.artifact.exec_values("qwen3.multimodal", inputs)?,
            self.config.vocabulary_size,
        )
    }

    pub fn next_token(&self, tokens: &[u64]) -> Result<u64> {
        let mut session = self.session(tokens.len())?;
        session.prefill_next_token(tokens)
    }

    fn ensure_execution_supported(&self) -> Result<()> {
        ensure!(
            matches!(
                self.config.architecture,
                ModelArchitecture::Qwen3Dense
                    | ModelArchitecture::Qwen3Moe
                    | ModelArchitecture::Qwen35Dense
                    | ModelArchitecture::Qwen35Moe
            ),
            "unsupported model architecture with {} validated tensors",
            self.weights.len()
        );
        Ok(())
    }
}

pub fn configured_gpu_dialect() -> Result<GpuDialect> {
    const ENVIRONMENT_VARIABLE: &str = "CATENA_GPU_DIALECT";
    match env::var(ENVIRONMENT_VARIABLE).as_deref() {
        Ok("hip") | Err(env::VarError::NotPresent) => Ok(GpuDialect::Hip),
        Ok("cuda") => Ok(GpuDialect::Cuda),
        Ok(value) => anyhow::bail!(
            "invalid GPU dialect `{value}` in {ENVIRONMENT_VARIABLE}; expected `hip` or `cuda`"
        ),
        Err(env::VarError::NotUnicode(value)) => anyhow::bail!(
            "invalid GPU dialect in {ENVIRONMENT_VARIABLE}: non-Unicode value {value:?}"
        ),
    }
}

/// Persistent, append-only KV-cache state for one generation sequence.
pub struct Session<'model> {
    model: &'model Model,
    cache: Option<MemOwn>,
    workspace: Option<MemOwn>,
    position: usize,
    capacity: usize,
    next_rope_position: Option<u64>,
}

impl Session<'_> {
    #[cfg(test)]
    pub(crate) fn cache_values_for_test(&self) -> Result<Vec<u16>> {
        Ok(self
            .cache
            .as_ref()
            .context("missing test cache")?
            .try_to_u16_vec()?)
    }

    pub fn position(&self) -> usize {
        self.position
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn prefill_logits(&mut self, tokens: &[u64]) -> Result<Vec<f32>> {
        ensure!(self.position == 0, "a session can only be prefilled once");
        let (logits, _next_token) = self.forward(tokens)?;
        read_logits(
            logits,
            self.model.config.vocabulary_size,
            self.model.precision,
        )
    }

    pub fn decode_logits(&mut self, token: u64) -> Result<Vec<f32>> {
        ensure!(self.position > 0, "decode requires a completed prefill");
        let (logits, _next_token) = self.forward(std::slice::from_ref(&token))?;
        read_logits(
            logits,
            self.model.config.vocabulary_size,
            self.model.precision,
        )
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

    pub fn prefill_multimodal_logits(
        &mut self,
        prompt: &crate::vision::PreparedMultimodalPrompt,
    ) -> Result<Vec<f32>> {
        let (logits, _) = self.prefill_multimodal(prompt)?;
        read_logits(
            logits,
            self.model.config.vocabulary_size,
            self.model.precision,
        )
    }

    pub fn prefill_multimodal_next_token(
        &mut self,
        prompt: &crate::vision::PreparedMultimodalPrompt,
    ) -> Result<u64> {
        let (_, next) = self.prefill_multimodal(prompt)?;
        Ok(next)
    }

    fn prefill_multimodal(
        &mut self,
        prompt: &crate::vision::PreparedMultimodalPrompt,
    ) -> Result<(MemOwn, u64)> {
        ensure!(self.position == 0, "a session can only be prefilled once");
        ensure!(
            self.model.config.is_qwen35(),
            "image inputs require Qwen3.5 dense or MoE"
        );
        ensure!(
            self.model.multimodal,
            "load the model with load_gguf_with_multimodal_options for image inputs"
        );
        ensure!(
            !prompt.tokens.is_empty()
                && prompt.tokens.len() <= self.capacity
                && prompt.positions.len() == prompt.tokens.len(),
            "multimodal prompt exceeds capacity or has invalid positions"
        );
        ensure!(
            prompt
                .tokens
                .iter()
                .all(|&t| t < self.model.config.vocabulary_size as u64),
            "prompt token outside vocabulary"
        );
        ensure!(
            prompt.images.len() == prompt.spans.len(),
            "image span count mismatch"
        );
        for (image, span) in prompt.images.iter().zip(&prompt.spans) {
            ensure!(
                image.width == self.model.config.hidden_size && image.dialect == self.model.dialect,
                "image embedding width or GPU dialect mismatch"
            );
            ensure!(
                span.len == image.token_count()
                    && image.values.byte_len() == (span.len * image.width * 4) as u64,
                "image embedding length mismatch"
            );
        }
        let mut result = None;
        let mut image_index = 0;
        for (index, &token) in prompt.tokens.iter().enumerate() {
            while image_index < prompt.spans.len()
                && index >= prompt.spans[image_index].start + prompt.spans[image_index].len
            {
                image_index += 1;
            }
            let hidden =
                if image_index < prompt.spans.len() && index >= prompt.spans[image_index].start {
                    let image = &prompt.images[image_index];
                    let mut out = self.model.artifact.exec_values(
                        "qwen3.mm-image-row",
                        vec![
                            image.values.as_ref().into(),
                            ((index - prompt.spans[image_index].start) as u64).into(),
                        ],
                    )?;
                    match out.pop() {
                        Some(Value::MemOwn(m)) => m,
                        _ => anyhow::bail!("image row ABI mismatch"),
                    }
                } else {
                    self.model.token_embedding(token)?
                };
            result = Some(self.forward_embedding(hidden, prompt.positions[index])?);
        }
        self.next_rope_position = Some(prompt.next_rope_position);
        result.context("empty multimodal prefill")
    }

    fn forward_embedding(
        &mut self,
        hidden: MemOwn,
        coordinates: [u64; 3],
    ) -> Result<(MemOwn, u64)> {
        ensure!(
            self.position < self.capacity,
            "token exceeds cache capacity"
        );
        let cache = self
            .cache
            .take()
            .context("session cache unavailable after failed execution")?;
        let (cache, logits, token) = self.model.cached_embedding_outputs(
            hidden,
            coordinates,
            self.position,
            self.capacity,
            cache,
        )?;
        self.cache = Some(cache);
        self.position += 1;
        Ok((logits, token))
    }

    fn forward(&mut self, tokens: &[u64]) -> Result<(MemOwn, u64)> {
        ensure!(
            !tokens.is_empty()
                && self
                    .position
                    .checked_add(tokens.len())
                    .is_some_and(|n| n <= self.capacity),
            "token range exceeds cache capacity"
        );
        ensure!(
            tokens
                .iter()
                .all(|&t| t < self.model.config.vocabulary_size as u64),
            "token ID outside vocabulary"
        );
        if let Some(mut rope_position) = self.next_rope_position {
            let mut result = None;
            for &token in tokens {
                let hidden = self.model.token_embedding(token)?;
                result = Some(self.forward_embedding(hidden, [rope_position; 3])?);
                rope_position += 1;
            }
            self.next_rope_position = Some(rope_position);
            return result.context("empty decode");
        }
        let cache = self
            .cache
            .take()
            .context("the session cache is unavailable after a failed execution")?;
        let workspace = if self.model.experimental_workspace_bytes == 0 {
            None
        } else {
            Some(
                self.workspace
                    .take()
                    .context("the session workspace is unavailable after a failed execution")?,
            )
        };
        let (cache, workspace, logits, next_token) =
            self.model
                .cached_outputs(tokens, self.position, self.capacity, cache, workspace)?;
        self.cache = Some(cache);
        self.workspace = workspace;
        self.position += tokens.len();
        Ok((logits, next_token))
    }
}

fn read_logits(
    logits_output: MemOwn,
    vocabulary_size: usize,
    precision: InferencePrecision,
) -> Result<Vec<f32>> {
    let logits = match precision {
        InferencePrecision::F32 => logits_output.try_to_f32_vec()?,
        InferencePrecision::Bf16 => logits_output
            .try_to_u16_vec()?
            .into_iter()
            .map(|bits| f32::from_bits(u32::from(bits) << 16))
            .collect(),
    };
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
