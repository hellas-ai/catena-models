//! Klein Qwen3 conditioning; reuses the dense Qwen .hex decoder through layer 27.
use super::{
    config::{TextConfig, read},
    ops::Ops,
    weights::Weights,
};
use crate::catena::{
    InferenceOptions, InferencePrecision, MatrixCoreMode, generated_model_source_for_options,
};
use anyhow::{Context, Result, ensure};
use catena_lang::runtime::{MemOwn, Value};
use std::path::Path;
use tokenizers::Tokenizer;
pub const SEQUENCE: usize = 512;
pub const CAPTURES: [usize; 3] = [9, 18, 27];
pub struct PromptTokens {
    pub ids: Vec<u64>,
    pub valid: usize,
}
pub fn tokenize(snapshot: &Path, prompt: &str) -> Result<PromptTokens> {
    let mut tokenizer = Tokenizer::from_file(snapshot.join("tokenizer/tokenizer.json"))
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    tokenizer.with_padding(None);
    tokenizer
        .with_truncation(None)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let text = format!(
        "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    );
    let e = tokenizer
        .encode(text, false)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let mut ids: Vec<u64> = e
        .get_ids()
        .iter()
        .take(SEQUENCE)
        .map(|&v| v as u64)
        .collect();
    let valid = ids.len();
    ids.resize(SEQUENCE, 151643);
    Ok(PromptTokens { ids, valid })
}
pub struct TextEncoder {
    pub ops: Ops,
    weights: Weights,
    names: Vec<String>,
    config: crate::spec::ModelConfig,
}
impl TextEncoder {
    pub fn load(snapshot: &Path, valid: usize) -> Result<Self> {
        Self::load_with_backend(snapshot, valid, super::ops::Backend::Fast)
    }
    pub fn load_with_backend(
        snapshot: &Path,
        valid: usize,
        backend: super::ops::Backend,
    ) -> Result<Self> {
        let tc: TextConfig = read(snapshot.join("text_encoder/config.json"))?;
        let config = tc.qwen()?;
        let mut weights = Weights::open(&snapshot.join("text_encoder"))?;
        let mut names = vec!["model.embed_tokens.weight".to_owned()];
        weights.validate(&names[0], &[config.vocabulary_size, config.hidden_size])?;
        let layer_names = [
            "input_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "post_attention_layernorm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ];
        let h = config.hidden_size;
        let q = config.num_attention_heads * config.head_dim;
        let kv = config.num_key_value_heads * config.head_dim;
        let ff = config.intermediate_size;
        let shapes = vec![
            vec![h],
            vec![q, h],
            vec![kv, h],
            vec![kv, h],
            vec![h, q],
            vec![config.head_dim],
            vec![config.head_dim],
            vec![h],
            vec![ff, h],
            vec![ff, h],
            vec![h, ff],
        ];
        for layer in 0..27 {
            for (suffix, shape) in layer_names.iter().zip(&shapes) {
                let name = format!("model.layers.{layer}.{suffix}");
                weights.validate(&name, shape)?;
                names.push(name);
            }
        }
        let source_start = std::time::Instant::now();
        let mut source = generated_model_source_for_options(
            &config,
            InferenceOptions::new(InferencePrecision::Bf16).with_matrix_cores(MatrixCoreMode::Wmma),
        )?;
        let end = source
            .find("(def program qwen3 :\n")
            .context("Qwen entrypoint marker missing")?;
        source.truncate(end);
        source = source.replace(
            "qwen3.cached-attention [cache-after-attention context.]",
            "flux.qwen-attention [cache-after-attention context.]",
        );
        // HF Qwen rounds RoPE cos/sin and each product to BF16 before adding.
        source = source
            .replace(
                "qwen3.f32-buffer-at [cosine.]",
                "qwen3.f32-buffer-at f32.to-bf16 bf16.to-f32 [cosine.]",
            )
            .replace(
                "qwen3.f32-buffer-at [sine.]",
                "qwen3.f32-buffer-at f32.to-bf16 bf16.to-f32 [sine.]",
            )
            .replace(
                "([.projection-value cosine] f32.mul [cosine-term.])",
                "([.projection-value cosine] f32.mul f32.to-bf16 bf16.to-f32 [cosine-term.])",
            )
            .replace(
                "([.partner-value sine] f32.mul [sine-term.])",
                "([.partner-value sine] f32.mul f32.to-bf16 bf16.to-f32 [sine-term.])",
            );
        source.push_str(
            &include_str!("../catena/flux_text_attention.hex")
                .replace("{{VALID_KEYS}}", &format!("{valid:016X}")),
        );
        super::ops::startup_timing("text source assembly", source_start);
        let ops = Ops::with_backend(Some(&source), backend)?;
        weights.prepare(&ops.runtime, &names, false)?;
        let norms = names
            .iter()
            .filter(|n| n.ends_with("norm.weight"))
            .cloned()
            .collect::<Vec<_>>();
        weights.prepare(&ops.runtime, &norms, true)?;
        Ok(Self {
            ops,
            weights,
            names,
            config,
        })
    }
    pub fn encode(&self, tokens: &PromptTokens, dump: Option<&Path>) -> Result<MemOwn> {
        ensure!(tokens.ids.len() == SEQUENCE, "expected 512 tokens");
        let r = &self.ops.runtime;
        let ids = r.mem_u64(&tokens.ids)?;
        let mut hidden = self.ops.run(
            "qwen3.embedding",
            vec![
                ids.as_ref().into(),
                self.weights.gpu_ref(&self.names[0])?.into(),
            ],
        )?;
        let rope = self.ops.run(
            "qwen3.rope-table",
            vec![0u64.into(), (SEQUENCE as u64).into()],
        )?;
        let mut cache = r.mem_bf16_zeroed(
            2 * SEQUENCE * self.config.num_key_value_heads * self.config.head_dim,
        )?;
        let mut captures = Vec::new();
        for layer in 0..27 {
            let mut args = vec![
                hidden.into(),
                rope.as_ref().into(),
                (SEQUENCE as u64).into(),
                cache.into(),
                (SEQUENCE as u64).into(),
                0u64.into(),
                0u64.into(),
            ];
            for name in &self.names[1 + layer * 11..1 + (layer + 1) * 11] {
                args.push(self.weights.gpu_ref(name)?.into());
            }
            let mut outputs = r.exec_values("flux.text-layer", args)?;
            cache = take_memory(outputs.pop().context("missing cache")?)?;
            hidden = take_memory(outputs.pop().context("missing hidden")?)?;
            if CAPTURES.contains(&(layer + 1)) {
                super::debug::dump(
                    dump,
                    &format!("text_{}", layer + 1),
                    &hidden,
                    &[1, 512, 2560],
                    true,
                )?;
                captures.push(self.ops.copy(hidden.as_ref(), true)?);
            }
            eprintln!("text layer {}/27", layer + 1);
        }
        let first = self.ops.run(
            "flux.concat-bf16",
            vec![
                captures[0].as_ref().into(),
                captures[1].as_ref().into(),
                (512 * 5120u64).into(),
                2560u64.into(),
                2560u64.into(),
            ],
        )?;
        let result = self.ops.run(
            "flux.concat-bf16",
            vec![
                first.as_ref().into(),
                captures[2].as_ref().into(),
                (512 * 7680u64).into(),
                5120u64.into(),
                2560u64.into(),
            ],
        )?;
        super::debug::dump(dump, "conditioning", &result, &[1, 512, 7680], true)?;
        Ok(result)
    }
}
fn take_memory(v: Value<'_>) -> Result<MemOwn> {
    match v {
        Value::MemOwn(m) => Ok(m),
        _ => anyhow::bail!("unexpected Qwen output"),
    }
}
