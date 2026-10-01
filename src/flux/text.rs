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
use catena_lang::runtime::MemOwn;
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
    debug: bool,
}
impl TextEncoder {
    pub fn load(snapshot: &Path, valid: usize) -> Result<Self> {
        Self::load_with_backend(snapshot, valid, false, super::ops::Backend::Fast)
    }
    pub fn load_with_backend(
        snapshot: &Path,
        _valid: usize,
        debug: bool,
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
        let source = generated_source_with_debug(&config, debug)?;
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
            super::graph::execute(&self.ops.runtime, "flux.encode-text", args, &program, dump)?;
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
    let mut source = generated_model_source_for_options(
        config,
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
    // Preserve the existing HF BF16 RoPE operation boundaries.
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
    // Extend only the dense layer. Cache addressing still receives start=layer=0.
    let begin = source
        .find("(def program qwen3.cached-dense-decoder-layer :")
        .unwrap();
    let end = source[begin + 1..]
        .find("(def program ")
        .map_or(source.len(), |i| begin + 1 + i);
    let layer = source[begin..end].to_string();
    let layer = layer
        .replacen("\n  }\n  ->", "\n    (u64 val)\n  }\n  ->", 1)
        .replacen("down-weight.]", "down-weight valid.]", 1)
        .replace(
            "[.start] [.layer]}\n    flux.qwen-attention",
            "[.start] [.layer] [.valid]}\n    flux.qwen-attention",
        );
    source.replace_range(begin..end, &layer);
    source += include_str!("../catena/flux_text_attention.hex");
    let c = |n: usize| format!("(const.u64.0x{n:016X})");
    let mut body = format!(
        " ({{[.ids] [.w0]}} qwen3.embedding [h0.])\n ({{{} {}}} qwen3.rope-table [rope.])\n ({} inference.zeros-bf16 [cache0.])\n",
        c(0),
        c(512),
        c(2 * 512 * config.num_key_value_heads * config.head_dim)
    );
    let mut hidden = "h0".to_string();
    let mut cache = "cache0".to_string();
    let mut captures = Vec::new();
    for l in 0..27 {
        let args = (1 + l * 11..1 + (l + 1) * 11)
            .map(|i| format!("[.w{i}]"))
            .collect::<Vec<_>>()
            .join(" ");
        body += &format!(
            " ({{[.{hidden}] [.rope] {} [.{cache}] {} {} {} {args} [.valid]}} flux.text-layer [h{} cache{}.])\n",
            c(512),
            c(512),
            c(0),
            c(0),
            l + 1,
            l + 1
        );
        hidden = format!("h{}", l + 1);
        cache = format!("cache{}", l + 1);
        if CAPTURES.contains(&(l + 1)) {
            let name = format!("capture{}", l + 1);
            body += &format!(
                " ({{[.{hidden}] {}}} flux.copy-bf16-o [{name}.])\n",
                c(512 * 2560)
            );
            captures.push(name);
        }
    }
    body += &format!(
        " ({{[.{}] [.{}] {} {} {}}} flux.concat-bf16-oo [first.])\n ({{[.first] [.{}] {} {} {}}} flux.concat-bf16-oo [result.])\n",
        captures[0],
        captures[1],
        c(512 * 5120),
        c(2560),
        c(2560),
        captures[2],
        c(512 * 7680),
        c(5120),
        c(2560)
    );
    let mut result = "result".to_string();
    for (i, m) in captures
        .iter()
        .filter(|_| !debug)
        .chain([&hidden, &cache, &"first".to_string()])
        .enumerate()
    {
        body += &format!(" ({{[.{m}] [.{result}]}} qwen3.free-bf16-before [done{i}.])\n");
        result = format!("done{i}");
    }
    body += &format!(" ({{[.rope] [.{result}]}} qwen3.free-f32-before [done-rope.])\n");
    if debug {
        body += &format!(
            " {{[.done-rope] {}}}\n",
            captures
                .iter()
                .map(|n| format!("[.{n}]"))
                .collect::<Vec<_>>()
                .join(" ")
        );
    } else {
        body += " [.done-rope]\n";
    }
    let output_types = if debug {
        "{(cap.own mem) (cap.own mem) (cap.own mem) (cap.own mem)}"
    } else {
        "(cap.own mem)"
    };
    source.push_str(
        "\n# Safetensors arguments for flux.encode-text:\n# w0: model.embed_tokens.weight (BF16)\n",
    );
    for l in 0..27 {
        for (i, suffix) in [
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
        ]
        .iter()
        .enumerate()
        {
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
    let weights = (0..1 + 27 * 11)
        .map(|i| format!("w{i}"))
        .collect::<Vec<_>>();
    source += &format!(
        "\n(def program flux.encode-text : {{(cap.ref mem) (u64 val) {}}} -> {output_types}\n= ([ids valid {}.]\n{body}))\n",
        vec!["(cap.ref mem)"; weights.len()].join(" "),
        weights.join(" ")
    );
    Ok(source)
}
