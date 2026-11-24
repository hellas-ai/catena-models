//! Hugging Face dense Qwen weights in the existing text-model ABI.
use std::{collections::HashSet, path::Path};

use anyhow::{Context, Result, ensure};
use catena_lang::runtime::{MemRef, Runtime};
use serde::Deserialize;
use serde_json::Value;

use crate::{
    hfhub::read_json,
    spec::{ModelArchitecture, ModelConfig, TensorSpec, WeightDtype},
    weights::{Weights, startup_timing},
};

#[derive(Deserialize)]
struct TextConfig {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: Option<usize>,
    head_dim: Option<usize>,
    max_position_embeddings: usize,
    rms_norm_eps: f32,
    tie_word_embeddings: Option<bool>,
    eos_token_id: Option<Value>,
    hidden_act: Option<String>,
    #[serde(default)]
    attention_bias: bool,
    #[serde(default)]
    use_sliding_window: bool,
    attn_output_gate: Option<bool>,
    layer_types: Option<Vec<String>>,
    full_attention_interval: Option<usize>,
    linear_conv_kernel_dim: Option<usize>,
    linear_key_head_dim: Option<usize>,
    linear_value_head_dim: Option<usize>,
    linear_num_key_heads: Option<usize>,
    linear_num_value_heads: Option<usize>,
    rope_theta: Option<f32>,
    rope_parameters: Option<RopeConfig>,
    rope_scaling: Option<RopeConfig>,
    partial_rotary_factor: Option<f32>,
    quantization_config: Option<Value>,
}

#[derive(Default, Deserialize)]
struct RopeConfig {
    #[serde(alias = "type")]
    rope_type: Option<String>,
    rope_theta: Option<f32>,
    partial_rotary_factor: Option<f32>,
    mrope_section: Option<[usize; 3]>,
    mrope_interleaved: Option<bool>,
}

pub(crate) fn read_config(path: &Path, repo: &str) -> Result<ModelConfig> {
    let root: Value = read_json(path)?;
    parse_config(root, repo)
        .with_context(|| format!("reading dense Qwen config from {}", path.display()))
}

fn parse_config(mut root: Value, repo: &str) -> Result<ModelConfig> {
    let architecture = match root["model_type"].as_str() {
        Some("qwen3") => ModelArchitecture::Qwen3Dense,
        Some("qwen3_5" | "qwen3_5_text") => ModelArchitecture::Qwen35Dense,
        other => anyhow::bail!(
            "unsupported safetensors model type {other:?}; expected dense Qwen3 or Qwen3.5"
        ),
    };
    let qwen35 = architecture == ModelArchitecture::Qwen35Dense;
    let tied = root["tie_word_embeddings"].as_bool();
    ensure!(
        root["quantization_config"].is_null(),
        "quantized Qwen safetensors are unsupported"
    );
    let text = if qwen35 && !root["text_config"].is_null() {
        let text = root["text_config"].take();
        ensure!(
            text["model_type"].is_null() || text["model_type"] == "qwen3_5_text",
            "unsupported Qwen3.5 text model type"
        );
        text
    } else {
        root
    };
    let hf: TextConfig = serde_json::from_value(text)?;
    ensure!(
        hf.quantization_config.is_none(),
        "quantized Qwen safetensors are unsupported"
    );
    ensure!(
        !hf.attention_bias && !hf.use_sliding_window,
        "Qwen safetensors require bias-free full attention"
    );
    ensure!(
        hf.hidden_act.as_deref().is_none_or(|act| act == "silu"),
        "Qwen safetensors require SiLU"
    );
    ensure!(
        !qwen35 || hf.attn_output_gate != Some(false),
        "Qwen3.5 requires a gated attention output"
    );
    let rope = hf.rope_parameters.or(hf.rope_scaling).unwrap_or_default();
    ensure!(
        rope.rope_type
            .as_deref()
            .is_none_or(|kind| kind == "default"),
        "scaled RoPE is unsupported for Qwen safetensors"
    );
    let head_dim = hf.head_dim.unwrap_or(if qwen35 { 256 } else { 128 });
    let rotary_fraction = rope
        .partial_rotary_factor
        .or(hf.partial_rotary_factor)
        .unwrap_or(if qwen35 { 0.25 } else { 1.0 });
    ensure!(
        rotary_fraction > 0.0 && rotary_fraction <= 1.0,
        "invalid partial rotary factor"
    );
    let rope_theta = rope.rope_theta.or(hf.rope_theta).unwrap_or(10_000.0);
    ensure!(
        rope_theta.is_finite() && rope_theta > 0.0,
        "invalid RoPE theta"
    );
    ensure!(
        hf.rms_norm_eps.is_finite() && hf.rms_norm_eps > 0.0,
        "invalid RMSNorm epsilon"
    );

    let mut config = ModelConfig {
        model_name: Some(repo.into()),
        lfm3: None,
        architecture,
        vocabulary_size: hf.vocab_size,
        hidden_size: hf.hidden_size,
        intermediate_size: hf.intermediate_size,
        num_layers: hf.num_hidden_layers,
        num_nextn_layers: 0,
        num_attention_heads: hf.num_attention_heads,
        num_key_value_heads: hf.num_key_value_heads.unwrap_or(hf.num_attention_heads),
        head_dim,
        max_position_embeddings: hf.max_position_embeddings,
        rms_norm_eps: hf.rms_norm_eps,
        rope_theta,
        rope_dimension_count: (head_dim as f32 * rotary_fraction) as usize,
        mrope_sections: [0; 3],
        num_experts: 0,
        experts_per_token: 0,
        shared_expert_intermediate_size: None,
        full_attention_interval: None,
        linear_conv_kernel: None,
        linear_state_size: None,
        linear_group_count: None,
        linear_time_step_rank: None,
        linear_inner_size: None,
        output_weight_tied: hf.tie_word_embeddings.or(tied).unwrap_or(false),
        eos_token_id: hf
            .eos_token_id
            .as_ref()
            .and_then(|id| id.as_u64().or_else(|| id[0].as_u64())),
    };
    if qwen35 {
        ensure!(
            rope.mrope_interleaved != Some(false),
            "Qwen3.5 requires interleaved MRoPE"
        );
        config.mrope_sections = rope.mrope_section.unwrap_or([11, 11, 10]);
        let interval = hf
            .full_attention_interval
            .or_else(|| {
                hf.layer_types
                    .as_ref()?
                    .iter()
                    .position(|layer| layer == "full_attention")
                    .map(|i| i + 1)
            })
            .unwrap_or(4);
        ensure!(interval > 0, "full-attention interval must be positive");
        config.full_attention_interval = Some(interval);
        let value_heads = hf.linear_num_value_heads.unwrap_or(32);
        config.linear_conv_kernel = Some(hf.linear_conv_kernel_dim.unwrap_or(4));
        config.linear_state_size = Some(hf.linear_key_head_dim.unwrap_or(128));
        config.linear_group_count = Some(hf.linear_num_key_heads.unwrap_or(16));
        config.linear_time_step_rank = Some(value_heads);
        config.linear_inner_size = Some(
            value_heads
                .checked_mul(hf.linear_value_head_dim.unwrap_or(128))
                .context("linear value width overflowed")?,
        );
    }
    if let Some(layers) = hf.layer_types {
        ensure!(
            layers.len() == config.num_layers
                && layers.iter().enumerate().all(|(i, layer)| {
                    let full = config
                        .full_attention_interval
                        .is_none_or(|n| (i + 1).is_multiple_of(n));
                    layer
                        == if full {
                            "full_attention"
                        } else {
                            "linear_attention"
                        }
                }),
            "Qwen checkpoint has an unsupported attention layer schedule"
        );
    }
    config.validate()?;
    Ok(config)
}

pub(crate) struct QwenWeights {
    weights: Weights,
    bindings: Vec<Binding>,
}

struct Binding {
    name: String,
    dtype: WeightDtype,
    transform: Transform,
}

#[derive(Clone, Copy, Default)]
struct Transform {
    add_one: bool,
    negative_exp: bool,
    permutation: Option<Permutation>,
}

/// Value heads are grouped by key in HF and tiled by replica in the text ABI.
#[derive(Clone, Copy)]
struct Permutation {
    keys: usize,
    replicas: usize,
    head_width: usize,
    prefix: usize,
    columns: bool,
}

impl Permutation {
    fn reorder<T: Copy>(self, values: &[T], columns: usize) -> Vec<T> {
        let mut out = Vec::with_capacity(values.len());
        let (stride, block) = if self.columns {
            (columns, self.head_width)
        } else {
            (values.len(), self.head_width * columns)
        };
        let prefix = self.prefix * columns;
        for row in values.chunks_exact(stride) {
            out.extend_from_slice(&row[..prefix]);
            for replica in 0..self.replicas {
                for key in 0..self.keys {
                    let start = prefix + (key * self.replicas + replica) * block;
                    out.extend_from_slice(&row[start..start + block]);
                }
            }
        }
        out
    }
}

impl Transform {
    fn for_tensor(config: &ModelConfig, name: &str) -> Self {
        if !config.is_qwen35() {
            return Self::default();
        }
        let mut transform = Self {
            add_one: name.ends_with("norm.weight") && !name.ends_with("linear_attn.norm.weight"),
            negative_exp: name.ends_with(".linear_attn.A_log"),
            permutation: None,
        };
        let Some((_, tensor)) = name.split_once(".linear_attn.") else {
            return transform;
        };
        let keys = config.linear_group_count.unwrap();
        let values = config.linear_time_step_rank.unwrap();
        if keys == values {
            return transform;
        }
        let mut order = Permutation {
            keys,
            replicas: values / keys,
            head_width: config.linear_value_head_size().unwrap(),
            prefix: 0,
            columns: false,
        };
        match tensor {
            "in_proj_qkv.weight" | "conv1d.weight" => {
                order.prefix = 2 * config.linear_key_size().unwrap()
            }
            "in_proj_z.weight" => {}
            "in_proj_a.weight" | "in_proj_b.weight" | "A_log" | "dt_bias" => order.head_width = 1,
            "out_proj.weight" => order.columns = true,
            _ => return transform,
        }
        transform.permutation = Some(order);
        transform
    }
}

impl QwenWeights {
    pub(crate) fn open(path: &Path, config: &ModelConfig, specs: &[TensorSpec]) -> Result<Self> {
        let weights = Weights::open(path)?;
        let wrapped = config.is_qwen35()
            && weights
                .tensors
                .contains_key("model.language_model.embed_tokens.weight");
        let mut bindings = Vec::with_capacity(specs.len());
        for spec in specs {
            let name = if spec.name == "lm_head.weight" && config.output_weight_tied {
                "model.embed_tokens.weight"
            } else {
                spec.name.as_str()
            };
            let name = match name.strip_prefix("model.") {
                Some(suffix) if wrapped => format!("model.language_model.{suffix}"),
                _ => name.into(),
            };
            let mut shape = spec.shape.clone();
            if spec.name.ends_with(".linear_attn.conv1d.weight")
                && weights
                    .tensors
                    .get(&name)
                    .is_some_and(|info| info.shape.len() == 3)
            {
                shape.insert(1, 1);
            }
            weights.validate(&name, &shape)?;
            bindings.push(Binding {
                name,
                dtype: spec.dtype,
                transform: Transform::for_tensor(config, &spec.name),
            });
        }
        Ok(Self { weights, bindings })
    }

    pub(crate) fn prepare(&mut self, runtime: &Runtime) -> Result<usize> {
        let start = std::time::Instant::now();
        let mut prepared = HashSet::new();
        let mut bytes = 0;
        for binding in &self.bindings {
            if !prepared.insert(binding.name.as_str()) {
                continue;
            }
            let info = &self.weights.tensors[&binding.name];
            let f32_storage = binding.dtype == WeightDtype::F32;
            bytes += info.shape.iter().product::<usize>() * if f32_storage { 4 } else { 2 };
            binding
                .prepare(runtime, &mut self.weights)
                .with_context(|| format!("preparing {}", binding.name))?;
        }
        startup_timing("Qwen safetensors preparation", start);
        eprintln!(
            "loaded safetensors, {:.2} GiB on disk; {:.2} GiB in the text weight ABI",
            self.weights.byte_len() as f64 / (1_u64 << 30) as f64,
            bytes as f64 / (1_u64 << 30) as f64,
        );
        Ok(bytes)
    }

    pub(crate) fn gpu_ref(&self, index: usize) -> Result<MemRef<'_>> {
        self.weights.gpu_ref(&self.bindings[index].name)
    }

    pub(crate) fn len(&self) -> usize {
        self.bindings.len()
    }
}

impl Binding {
    fn prepare(&self, runtime: &Runtime, weights: &mut Weights) -> Result<()> {
        let name = &self.name;
        let transform = self.transform;
        let f32_storage = self.dtype == WeightDtype::F32;
        if transform.permutation.is_none() && !transform.add_one && !transform.negative_exp {
            return weights.prepare_tensor(runtime, name, f32_storage);
        }
        let columns = weights.tensors[name]
            .shape
            .get(1..)
            .unwrap_or_default()
            .iter()
            .product();
        let mem = if f32_storage {
            let values = self.transformed_f32(weights.host_f32(name)?, columns);
            runtime.mem_f32(&values)?
        } else {
            let values = weights.host_bf16(name)?;
            let order = transform.permutation.context("missing BF16 permutation")?;
            runtime.mem_u16(&order.reorder(&values, columns))?
        };
        weights.insert_prepared(name, mem)
    }

    fn transformed_f32(&self, mut values: Vec<f32>, columns: usize) -> Vec<f32> {
        if self.transform.add_one {
            values.iter_mut().for_each(|value| *value += 1.0);
        } else if self.transform.negative_exp {
            values.iter_mut().for_each(|value| *value = -value.exp());
        }
        match self.transform.permutation {
            Some(order) => order.reorder(&values, columns),
            None => values,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        hfhub::{HubCheckpoint, download_model},
        spec::weight_specs,
    };
    use safetensors::{
        Dtype,
        tensor::{TensorView, serialize_to_file},
    };
    use std::fs;

    #[test]
    fn binds_wrapped_weights_across_shards_in_model_order() -> Result<()> {
        let directory =
            std::env::temp_dir().join(format!("catena-qwen-weights-{}", std::process::id()));
        fs::create_dir_all(&directory)?;
        let result = (|| {
            let mut config = ModelConfig::qwen35_0_8b();
            config.hidden_size = 2;
            config.vocabulary_size = 2;
            let embedding = "model.language_model.embed_tokens.weight";
            let norm = "model.language_model.norm.weight";
            let head = "lm_head.weight";
            serialize_to_file(
                [
                    (
                        embedding,
                        TensorView::new(Dtype::BF16, vec![2, 2], &[0; 8])?,
                    ),
                    (
                        norm,
                        TensorView::new(Dtype::F16, vec![2], &[0, 0x38, 0, 0xb8])?,
                    ),
                ],
                None,
                &directory.join("one.safetensors"),
            )?;
            serialize_to_file(
                [(head, TensorView::new(Dtype::BF16, vec![2, 2], &[0; 8])?)],
                None,
                &directory.join("two.safetensors"),
            )?;
            let index = directory.join("model.safetensors.index.json");
            fs::write(
                &index,
                serde_json::to_vec(&serde_json::json!({
                    "weight_map": {
                        (head): "two.safetensors",
                        (norm): "one.safetensors",
                        (embedding): "one.safetensors"
                    }
                }))?,
            )?;
            let globals = |config: &ModelConfig| -> Result<Vec<TensorSpec>> {
                Ok(weight_specs(config)?
                    .into_iter()
                    .filter(|spec| !spec.name.starts_with("model.layers."))
                    .collect())
            };
            for tied in [true, false] {
                config.output_weight_tied = tied;
                let weights = QwenWeights::open(&index, &config, &globals(&config)?)?;
                assert_eq!(
                    weights
                        .bindings
                        .iter()
                        .map(|binding| binding.name.as_str())
                        .collect::<Vec<_>>(),
                    [embedding, norm, if tied { embedding } else { head }]
                );
                assert_eq!(
                    weights.bindings[1].transformed_f32(weights.weights.host_f32(norm)?, 1),
                    [1.5, 0.5]
                );
            }
            let mut specs = globals(&config)?;
            specs[1].shape = vec![3];
            assert!(QwenWeights::open(&index, &config, &specs).is_err());
            fs::remove_file(directory.join("two.safetensors"))?;
            assert!(QwenWeights::open(&index, &config, &globals(&config)?).is_err());
            Ok(())
        })();
        fs::remove_dir_all(directory)?;
        result
    }

    #[test]
    fn qwen35_reorders_only_the_value_heads() {
        let mut config = ModelConfig::qwen35_0_8b();
        config.linear_group_count = Some(2);
        config.linear_time_step_rank = Some(4);
        config.linear_state_size = Some(1);
        config.linear_inner_size = Some(8);
        let order = |tensor: &str| {
            Transform::for_tensor(&config, &format!("model.layers.0.linear_attn.{tensor}"))
                .permutation
                .unwrap()
        };
        for tensor in ["in_proj_qkv.weight", "conv1d.weight"] {
            assert_eq!(
                order(tensor).reorder(&(0..12).collect::<Vec<_>>(), 1),
                [0, 1, 2, 3, 4, 5, 8, 9, 6, 7, 10, 11]
            );
        }
        assert_eq!(
            order("out_proj.weight").reorder(&(0..16).collect::<Vec<_>>(), 8),
            [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15]
        );
        let transform = Transform::for_tensor(&config, "model.layers.0.linear_attn.A_log");
        let binding = Binding {
            name: "A_log".into(),
            dtype: WeightDtype::F32,
            transform,
        };
        let values = [0.0_f32, 1.0, 2.0, 3.0];
        assert_eq!(
            binding.transformed_f32(values.to_vec(), 1),
            [0, 2, 1, 3].map(|i| -values[i].exp())
        );
        assert!(!Transform::for_tensor(&config, "model.layers.0.linear_attn.norm.weight").add_one);
    }

    #[test]
    #[ignore = "requires Hugging Face models or their cached files"]
    fn resolves_and_validates_qwen_safetensors() -> Result<()> {
        use crate::tokenizer::QwenTokenizer;
        for (repo, gguf_repo, indexed) in [
            ("unsloth/Qwen3-0.6B", "unsloth/Qwen3-0.6B-GGUF:BF16", false),
            (
                "unsloth/Qwen3.5-0.8B",
                "unsloth/Qwen3.5-0.8B-GGUF:BF16",
                true,
            ),
            ("Qwen/Qwen3.5-4B", "unsloth/Qwen3.5-4B-GGUF:BF16", true),
        ] {
            let checkpoint = download_model(repo)?;
            let HubCheckpoint::Safetensors { path, config } = &checkpoint else {
                anyhow::bail!("{repo} did not resolve to safetensors");
            };
            let tokenizer = checkpoint.tokenizer()?;
            assert!(tokenizer.chat_template().is_some());
            assert_eq!(path.ends_with("model.safetensors.index.json"), indexed);
            let specs = weight_specs(config)?;
            let weights = QwenWeights::open(path, config, &specs)?;
            assert_eq!(weights.len(), specs.len());
            let gguf = crate::hfhub::download_gguf(gguf_repo)?;
            let gguf_config = crate::checkpoint::read_gguf_model_config(&gguf.primary)?;
            assert_eq!(specs, weight_specs(&gguf_config)?);
            let gguf_tokenizer =
                QwenTokenizer::from_gguf(&gguf.primary)?.context("missing GGUF tokenizer")?;
            let prompt = "The capital of France is";
            assert_eq!(tokenizer.encode(prompt)?, gguf_tokenizer.encode(prompt)?);
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires a GPU and Hugging Face models"]
    fn qwen35_safetensors_match_bf16_gguf_with_wmma() -> Result<()> {
        use crate::{catena::InferenceOptions, model::Model};

        fn trajectory(model: &Model, tokens: &[u64]) -> Result<Vec<(usize, Vec<f32>)>> {
            let mut session = model.session(tokens.len() + 3)?;
            let mut logits = session.prefill_logits(tokens)?;
            let mut steps = Vec::new();
            for step in 0..4 {
                let next = logits
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .context("missing logits")?
                    .0;
                steps.push((next, std::mem::take(&mut logits)));
                if step < 3 {
                    logits = session.decode_logits(next as u64)?;
                }
            }
            Ok(steps)
        }

        let repo = "unsloth/Qwen3.5-0.8B";
        let checkpoint = download_model(repo)?;
        assert!(matches!(checkpoint, HubCheckpoint::Safetensors { .. }));
        let tokens = checkpoint.tokenizer()?.encode("The capital of France is")?;
        let options = InferenceOptions::default().with_wmma_prefill(true);
        let actual = trajectory(
            &Model::load_checkpoint_with_options(&checkpoint, options)?,
            &tokens,
        )?;
        let gguf = crate::hfhub::download_gguf(&format!("{repo}-GGUF:BF16"))?;
        let expected = trajectory(
            &Model::load_gguf_with_options(gguf.primary, options)?,
            &tokens,
        )?;
        for (step, ((token, actual), (gguf_token, expected))) in
            actual.into_iter().zip(expected).enumerate()
        {
            assert_eq!(token, gguf_token, "{repo} greedy token at step {step}");
            assert_eq!(actual.len(), expected.len());
            let max_delta = actual
                .into_iter()
                .zip(expected)
                .map(|(a, b)| if a == b { 0.0 } else { (a - b).abs() })
                .fold(0.0_f32, |max, delta| {
                    if delta.is_finite() {
                        max.max(delta)
                    } else {
                        f32::INFINITY
                    }
                });
            eprintln!("{repo} step {step}: greedy token {token}, max logit delta {max_delta}");
            assert!(max_delta <= 0.0625, "{repo} logits at step {step}");
        }
        Ok(())
    }
}
