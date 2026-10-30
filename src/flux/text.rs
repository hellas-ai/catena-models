//! Klein Qwen3 conditioning; reuses the dense Qwen .hex decoder through layer 27.
use super::{
    config::{TextConfig, read},
    ops::Ops,
};
use crate::weights::Weights;
use anyhow::{Result, ensure};
use catena_lang::runtime::MemOwn;
use std::path::Path;
use tokenizers::Tokenizer;
pub(super) const LAYER_NAMES: [&str; 11] = [
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
    debug: bool,
}
impl TextEncoder {
    pub fn load(snapshot: &Path, valid: usize) -> Result<Self> {
        Self::load_with_debug(snapshot, valid, false)
    }
    pub fn load_with_debug(snapshot: &Path, _valid: usize, debug: bool) -> Result<Self> {
        super::validate_backend()?;
        let tc: TextConfig = read(snapshot.join("text_encoder/config.json"))?;
        let config = tc.qwen()?;
        let mut weights =
            Weights::open(&snapshot.join("text_encoder/model.safetensors.index.json"))?;
        let mut names = vec!["model.embed_tokens.weight".to_owned()];
        weights.validate(&names[0], &[config.vocabulary_size, config.hidden_size])?;
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
            for (suffix, shape) in LAYER_NAMES.iter().zip(&shapes) {
                let name = format!("model.layers.{layer}.{suffix}");
                weights.validate(&name, shape)?;
                names.push(name);
            }
        }
        let source_start = std::time::Instant::now();
        let source = generated_source_with_debug(&config, debug)?;
        super::ops::startup_timing("text source assembly", source_start);
        let ops = Ops::new(Some(&source))?;
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
            debug,
        })
    }
    pub fn encode(&self, tokens: &PromptTokens, dump: Option<&Path>) -> Result<MemOwn> {
        ensure!(
            tokens.ids.len() == SEQUENCE && tokens.valid > 0 && tokens.valid <= SEQUENCE,
            "invalid prompt tokens"
        );
        let ids = self.ops.runtime.mem_u64(&tokens.ids)?;
        let mut args = vec![ids.as_ref().into(), (tokens.valid as u64).into()];
        for name in &self.names {
            args.push(self.weights.gpu_ref(name)?.into());
        }
        let program = super::graph::Program {
            source: String::new(),
            weights: vec![],
            captures: if self.debug {
                CAPTURES
                    .iter()
                    .map(|n| (format!("text_{n}"), vec![1, 512, 2560], true))
                    .collect()
            } else {
                vec![]
            },
        };
        let result =
            super::graph::execute(&self.ops.artifact, "flux.encode-text", args, &program, dump)?;
        super::debug::dump(dump, "conditioning", &result, &[1, 512, 7680], true)?;
        Ok(result)
    }
}

/// Generates the complete encoder once; valid length remains a runtime scalar.
pub fn generated_source(config: &crate::spec::ModelConfig) -> Result<String> {
    generated_source_with_debug(config, false)
}
pub fn generated_source_with_debug(
    config: &crate::spec::ModelConfig,
    debug: bool,
) -> Result<String> {
    Ok(super::kernels::with_wmma(super::kernels::text(
        config, debug,
    )?))
}

/// Shared checkpoint argument order for standalone Hex assets.
pub(super) fn weight_comments() -> String {
    let mut source =
        "\n# Safetensors arguments for flux.encode-text:\n# w0: model.embed_tokens.weight (BF16)\n"
            .to_owned();
    for l in 0..27 {
        for (i, suffix) in LAYER_NAMES.iter().enumerate() {
            let dtype = if suffix.ends_with("norm.weight") {
                "F32"
            } else {
                "BF16"
            };
            source.push_str(&format!(
                "# w{}: model.layers.{l}.{suffix} ({dtype})\n",
                1 + l * 11 + i
            ));
        }
    }
    source
}
