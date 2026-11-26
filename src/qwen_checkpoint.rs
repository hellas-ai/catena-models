//! Hugging Face Qwen weights in the existing text-model ABI.
use std::{collections::HashSet, path::Path};

use anyhow::{Context, Result, ensure};
use catena_lang::{
    codegen::GpuDialect,
    runtime::{MemOwn, MemRef, Runtime},
};
use memmap2::MmapMut;
use safetensors::Dtype;
use serde::Deserialize;
use serde_json::Value;

use crate::{
    hfhub::read_json,
    spec::{ModelArchitecture, ModelConfig, TensorSpec, WeightDtype},
    weights::{MappedWeight, Weights, f32_to_bf16, startup_timing},
};

#[derive(Deserialize)]
struct TextConfig {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: Option<usize>,
    moe_intermediate_size: Option<usize>,
    shared_expert_intermediate_size: Option<usize>,
    num_experts: Option<usize>,
    num_experts_per_tok: Option<usize>,
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
    parse_config(root, repo).with_context(|| format!("reading Qwen config from {}", path.display()))
}

fn parse_config(mut root: Value, repo: &str) -> Result<ModelConfig> {
    let (architecture, text_type) = match root["model_type"].as_str() {
        Some("qwen3") => (ModelArchitecture::Qwen3Dense, "qwen3"),
        Some("qwen3_5" | "qwen3_5_text") => (ModelArchitecture::Qwen35Dense, "qwen3_5_text"),
        Some("qwen3_5_moe" | "qwen3_5_moe_text") => {
            (ModelArchitecture::Qwen35Moe, "qwen3_5_moe_text")
        }
        other => {
            anyhow::bail!("unsupported safetensors model type {other:?}; expected Qwen3 or Qwen3.5")
        }
    };
    let qwen35 = architecture != ModelArchitecture::Qwen3Dense;
    let moe = architecture == ModelArchitecture::Qwen35Moe;
    let tied = root["tie_word_embeddings"].as_bool();
    fp8_block_size(&root["quantization_config"], architecture)?;
    let text = if qwen35 && !root["text_config"].is_null() {
        let text = root["text_config"].take();
        ensure!(
            text["model_type"].is_null() || text["model_type"] == text_type,
            "unsupported Qwen3.5 text model type"
        );
        text
    } else {
        root
    };
    let hf: TextConfig = serde_json::from_value(text)?;
    if qwen35 && let Some(quantization) = &hf.quantization_config {
        fp8_block_size(quantization, architecture)?;
    }
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
        intermediate_size: if moe {
            hf.moe_intermediate_size
                .context("missing moe_intermediate_size")?
        } else {
            hf.intermediate_size.context("missing intermediate_size")?
        },
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
        num_experts: if moe {
            hf.num_experts.context("missing num_experts")?
        } else {
            0
        },
        experts_per_token: if moe {
            hf.num_experts_per_tok
                .context("missing num_experts_per_tok")?
        } else {
            0
        },
        shared_expert_intermediate_size: if moe {
            hf.shared_expert_intermediate_size
        } else {
            None
        },
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

fn fp8_block_size(
    quantization: &Value,
    architecture: ModelArchitecture,
) -> Result<Option<[usize; 2]>> {
    if quantization.is_null() {
        return Ok(None);
    }
    ensure!(
        matches!(
            architecture,
            ModelArchitecture::Qwen3Dense
                | ModelArchitecture::Qwen35Dense
                | ModelArchitecture::Qwen35Moe
        ) && quantization["quant_method"] == "fp8"
            && quantization["activation_scheme"] == "dynamic",
        "quantized Qwen safetensors require dynamic FP8 in Qwen3 dense or Qwen3.5 dense/MoE"
    );
    ensure!(
        quantization["quant_type"].is_null() || quantization["quant_type"] == "float",
        "FP8 quant_type must be float"
    );
    let block: [usize; 2] = Deserialize::deserialize(&quantization["weight_block_size"])
        .context("FP8 weight_block_size must contain two positive integers")?;
    ensure!(
        block.iter().all(|&size| size > 0),
        "FP8 weight_block_size must contain two positive integers"
    );
    Ok(Some(block))
}

pub(crate) struct QwenWeights {
    weights: Weights,
    bindings: Vec<Binding>,
    moe: bool,
    prefill: Vec<usize>,
    prefill_buffers: Vec<MemOwn>,
}

struct Binding {
    name: String,
    dtype: WeightDtype,
    transform: Transform,
    fp8: Option<Fp8Weight>,
    experts: Option<ExpertSource>,
    prepared_expert: Option<ExpertMemory>,
    elements: usize,
}

enum ExpertSource {
    FusedHalf(usize),
    Separate(Vec<ExpertMatrix>),
}

struct ExpertMatrix {
    name: String,
    fp8: Option<Fp8Weight>,
}

impl ExpertMatrix {
    fn to_bf16(&self, weights: &Weights) -> Result<Vec<u16>> {
        match &self.fp8 {
            Some(fp8) => fp8.to_bf16(weights, &self.name),
            None => weights.host_bf16(&self.name),
        }
    }
}

enum ExpertMemory {
    Device(MemOwn),
    Mapped(MappedWeight),
}

struct Fp8Weight {
    scale: String,
    block: [usize; 2],
}

impl Fp8Weight {
    fn new(weights: &Weights, name: &str, shape: &[usize], block: [usize; 2]) -> Result<Self> {
        let [rows, cols] = *shape else {
            anyhow::bail!("{name}: expected a matrix for FP8 weights, got {shape:?}");
        };
        let scale = format!("{name}_scale_inv");
        weights
            .validate(&scale, &[rows.div_ceil(block[0]), cols.div_ceil(block[1])])
            .with_context(|| format!("{name}: invalid FP8 scales"))?;
        Ok(Self { scale, block })
    }

    /// Permute bytes and block scales together without requantizing checkpoint weights.
    fn native_operands(
        &self,
        weights: &Weights,
        name: &str,
        order: Option<Permutation>,
    ) -> Result<(Vec<u8>, Vec<f32>)> {
        let raw = weights.raw_bytes(name)?;
        let scales = weights.host_f32(&self.scale)?;
        let columns = weights.tensors[name].shape[1];
        let rows = weights.tensors[name].shape[0];
        let Some(order) = order else {
            return Ok((pad_fp8_matrix(raw, rows, columns), scales));
        };
        let [br, bc] = self.block;
        // Whole-head permutations preserve 128x128 quantization blocks.
        ensure!(
            order.prefix.is_multiple_of(br)
                && order
                    .head_width
                    .is_multiple_of(if order.columns { bc } else { br }),
            "{name}: FP8 head permutation must preserve weight blocks"
        );
        let scale_order = Permutation {
            head_width: order.head_width / if order.columns { bc } else { br },
            prefix: order.prefix / br,
            ..order
        };
        Ok((
            pad_fp8_matrix(&order.reorder(raw, columns), rows, columns),
            scale_order.reorder(&scales, columns.div_ceil(bc)),
        ))
    }

    fn to_bf16(&self, weights: &Weights, name: &str) -> Result<Vec<u16>> {
        let cols = weights.tensors[name].shape[1];
        let bytes = weights.raw_bytes(name)?;
        let scales = weights.host_f32(&self.scale)?;
        ensure!(
            scales
                .iter()
                .all(|&scale| scale.is_finite() && scale >= 0.0),
            "{}: FP8 scales must be finite and nonnegative",
            self.scale
        );
        let [block_rows, block_cols] = self.block;
        let scale_cols = cols.div_ceil(block_cols);
        let decoded: [f32; 256] = std::array::from_fn(|bits| e4m3fn_to_f32(bits as u8));
        let mut bf16 = Vec::with_capacity(bytes.len());
        for (row, values) in bytes.chunks_exact(cols).enumerate() {
            let start = row / block_rows * scale_cols;
            for (block, &scale) in values
                .chunks(block_cols)
                .zip(&scales[start..start + scale_cols])
            {
                bf16.extend(
                    block
                        .iter()
                        .map(|&bits| f32_to_bf16(decoded[bits as usize] * scale)),
                );
            }
        }
        Ok(bf16)
    }
}

fn pad_fp8_matrix(raw: &[u8], rows: usize, cols: usize) -> Vec<u8> {
    let padded_cols = cols.next_multiple_of(128);
    if rows.is_multiple_of(128) && cols == padded_cols {
        return raw.to_vec();
    }
    let mut out = vec![0; rows.next_multiple_of(128) * padded_cols];
    for (dst, src) in out
        .chunks_exact_mut(padded_cols)
        .zip(raw.chunks_exact(cols))
    {
        dst[..cols].copy_from_slice(src);
    }
    out
}

fn encode_e4m3fn(value: f32) -> u8 {
    let sign = ((value.to_bits() >> 24) & 0x80) as u8;
    let x = value.abs().min(448.0);
    let bits = if x < 1.0 / 64.0 {
        (x * 512.0).round_ties_even() as u8
    } else {
        let bits = x.to_bits();
        ((bits - (120_u32 << 23) + 0x0007ffff + ((bits >> 20) & 1)) >> 20) as u8
    };
    sign | bits
}

fn quantize_bf16_matrix(values: &[u16], rows: usize, cols: usize) -> Result<(Vec<u8>, Vec<f32>)> {
    let pc = cols.next_multiple_of(128);
    let pr = rows.next_multiple_of(128);
    let mut scales = vec![0.0_f32; pr / 128 * (pc / 128)];
    for (i, &v) in values.iter().enumerate() {
        let x = crate::weights::bf16_to_f32(v);
        ensure!(x.is_finite(), "cannot quantize non-finite FP8 weight");
        let j = (i / cols) / 128 * (pc / 128) + (i % cols) / 128;
        scales[j] = scales[j].max(x.abs());
    }
    for scale in &mut scales {
        *scale = if *scale == 0.0 { 1.0 } else { *scale / 448.0 };
    }
    let mut raw = vec![0; pr * pc];
    for (i, &v) in values.iter().enumerate() {
        let row = i / cols;
        let col = i % cols;
        raw[row * pc + col] = encode_e4m3fn(
            crate::weights::bf16_to_f32(v) / scales[row / 128 * (pc / 128) + col / 128],
        );
    }
    Ok((raw, scales))
}

fn e4m3fn_to_f32(bits: u8) -> f32 {
    let exponent = u32::from((bits >> 3) & 0x0f);
    let fraction = u32::from(bits & 0x07);
    let value = match (exponent, fraction) {
        (0, fraction) => fraction as f32 / 512.0,
        (15, 7) => f32::NAN,
        _ => f32::from_bits(((exponent + 120) << 23) | (fraction << 20)),
    };
    if bits & 0x80 == 0 { value } else { -value }
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
        // Some exported checkpoints repeat the language_model wrapper.
        let mut text_prefix = "model.".to_owned();
        if config.is_qwen35() {
            while !weights
                .tensors
                .contains_key(&format!("{text_prefix}embed_tokens.weight"))
            {
                let nested = format!("{text_prefix}language_model.");
                if !weights.tensors.keys().any(|name| name.starts_with(&nested)) {
                    break;
                }
                text_prefix = nested;
            }
        }
        let mut bindings = Vec::with_capacity(specs.len());
        let block_size = if weights.tensors.values().any(|t| t.dtype == Dtype::F8_E4M3) {
            let directory = path.parent().context("missing checkpoint directory")?;
            let root: Value = read_json(directory.join("config.json"))?;
            let quantization = root
                .get("quantization_config")
                .filter(|q| !q.is_null())
                .unwrap_or(&root["text_config"]["quantization_config"]);
            fp8_block_size(quantization, config.architecture)?
        } else {
            None
        };
        let validate = |name: &str, shape: &[usize], dtype| -> Result<Option<Fp8Weight>> {
            let tensor = weights.validate_shape(name, shape)?;
            if tensor.dtype == Dtype::F8_E4M3 {
                ensure!(
                    dtype == WeightDtype::Bf16 && name.ends_with(".weight"),
                    "{name}: FP8 requires a BF16 weight matrix"
                );
                let block = block_size
                    .with_context(|| format!("{name}: missing FP8 quantization_config"))?;
                Ok(Some(Fp8Weight::new(&weights, name, shape, block)?))
            } else {
                weights.validate(name, shape)?;
                Ok(None)
            }
        };
        let checkpoint_name = |name: &str| -> String {
            match name.strip_prefix("model.") {
                Some(suffix) => format!("{text_prefix}{suffix}"),
                _ => name.into(),
            }
        };
        for spec in specs {
            let name = if spec.name == "lm_head.weight" && config.output_weight_tied {
                "model.embed_tokens.weight"
            } else {
                spec.name.as_str()
            };
            let mut name = checkpoint_name(name);
            let mut shape = spec.shape.clone();
            let mut experts = None;
            if config.architecture == ModelArchitecture::Qwen35Moe
                && let Some((prefix, projection)) = name.rsplit_once(".mlp.experts.")
            {
                let first = checkpoint_name(&spec.sources[0].name);
                if weights.tensors.contains_key(&first) {
                    let matrices = spec
                        .sources
                        .iter()
                        .map(|source| {
                            let name = checkpoint_name(&source.name);
                            let fp8 = validate(&name, &source.shape, spec.dtype)?;
                            Ok(ExpertMatrix { name, fp8 })
                        })
                        .collect::<Result<_>>()?;
                    experts = Some(ExpertSource::Separate(matrices));
                } else {
                    match projection {
                        "gate_proj.weight" | "up_proj.weight" => {
                            experts = Some(ExpertSource::FusedHalf(usize::from(
                                projection == "up_proj.weight",
                            )));
                            shape[1] *= 2;
                            name = format!("{prefix}.mlp.experts.gate_up_proj");
                        }
                        "down_proj.weight" => name.truncate(name.len() - ".weight".len()),
                        _ => {}
                    }
                }
            }
            if spec.name.ends_with(".linear_attn.conv1d.weight")
                && weights
                    .tensors
                    .get(&name)
                    .is_some_and(|info| info.shape.len() == 3)
            {
                shape.insert(1, 1);
            }
            let fp8 = match experts {
                Some(ExpertSource::Separate(_)) => None,
                Some(ExpertSource::FusedHalf(_)) => {
                    weights.validate(&name, &shape)?;
                    None
                }
                None => validate(&name, &shape, spec.dtype)?,
            };
            bindings.push(Binding {
                name,
                dtype: spec.dtype,
                transform: Transform::for_tensor(config, &spec.name),
                fp8,
                experts,
                prepared_expert: None,
                elements: spec.element_count(),
            });
        }
        Ok(Self {
            weights,
            bindings,
            moe: config.architecture == ModelArchitecture::Qwen35Moe,
            prefill: Vec::new(),
            prefill_buffers: Vec::new(),
        })
    }

    pub(crate) fn enable_fp8_prefill(&mut self, config: &ModelConfig) -> Result<()> {
        let mut native_count = 0;
        let mut cursor = 1;
        for layer in 0..config.num_layers {
            let linear = config.layer_type(layer) == crate::spec::LayerType::LinearAttention;
            for &local in crate::qwen::fp8_projection_indices(self.moe, linear) {
                let index = cursor + local;
                let binding = self
                    .bindings
                    .get(index)
                    .context("missing Qwen projection")?;
                if let Some(fp8) = &binding.fp8 {
                    ensure!(
                        fp8.block == [128, 128],
                        "FP8 prefill requires 128x128 weight blocks: {}",
                        binding.name
                    );
                    native_count += 1;
                }
                self.prefill.push(index);
            }
            cursor += match (self.moe, linear) {
                (false, false) => 11,
                (false, true) => 14,
                (true, false) => 16,
                (true, true) => 19,
            };
        }
        ensure!(
            self.moe || native_count > 0,
            "FP8 prefill requires an FP8 dense Qwen safetensors checkpoint"
        );
        Ok(())
    }

    pub(crate) fn prepare(&mut self, runtime: &Runtime) -> Result<usize> {
        let start = std::time::Instant::now();
        let mut prepared = HashSet::new();
        let mut bytes = 0;
        let map_tensors = self.moe && runtime.dialect() == GpuDialect::Hip;
        for binding in &mut self.bindings {
            if binding.experts.is_none() && !prepared.insert(binding.name.clone()) {
                continue;
            }
            let f32_storage = binding.dtype == WeightDtype::F32;
            bytes += binding.elements * if f32_storage { 4 } else { 2 };
            binding
                .prepare(runtime, &mut self.weights, map_tensors)
                .with_context(|| format!("preparing {}", binding.name))?;
        }
        for &index in &self.prefill {
            let binding = &self.bindings[index];
            if let Some((raw, scales)) = binding.native_prefill(&self.weights, self.moe)? {
                ensure!(
                    scales.iter().all(|s| s.is_finite() && *s >= 0.0),
                    "invalid scales for {}",
                    binding.name
                );
                bytes += raw.len() + scales.len() * 4;
                self.prefill_buffers.push(runtime.mem_u8(&raw)?);
                self.prefill_buffers.push(runtime.mem_f32(&scales)?);
            } else {
                self.prefill_buffers.push(runtime.mem_u8(&[])?);
                self.prefill_buffers.push(runtime.mem_f32(&[])?);
            }
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
        if index >= self.bindings.len() {
            return Ok(self.prefill_buffers[index - self.bindings.len()].as_ref());
        }
        let binding = &self.bindings[index];
        if binding.experts.is_some() {
            return match binding
                .prepared_expert
                .as_ref()
                .context("expert weights not prepared")?
            {
                ExpertMemory::Device(mem) => Ok(mem.as_ref()),
                ExpertMemory::Mapped(mem) => mem.gpu_ref(),
            };
        }
        self.weights.gpu_ref(&binding.name)
    }

    pub(crate) fn len(&self) -> usize {
        self.bindings.len() + self.prefill.len() * 2
    }
}

impl Binding {
    fn native_prefill(
        &self,
        weights: &Weights,
        quantize_unconverted: bool,
    ) -> Result<Option<(Vec<u8>, Vec<f32>)>> {
        if let Some(fp8) = &self.fp8 {
            ensure!(
                fp8.block == [128, 128],
                "FP8 prefill requires 128x128 blocks: {}",
                self.name
            );
            return Ok(Some(fp8.native_operands(
                weights,
                &self.name,
                self.transform.permutation,
            )?));
        }
        if !quantize_unconverted {
            return Ok(None);
        }
        let mut raw = Vec::new();
        let mut scales = Vec::new();
        match &self.experts {
            Some(ExpertSource::Separate(matrices)) => {
                for matrix in matrices {
                    let (bytes, blocks) = if let Some(fp8) = &matrix.fp8 {
                        ensure!(
                            fp8.block == [128, 128],
                            "expert FP8 prefill requires 128x128 blocks"
                        );
                        fp8.native_operands(weights, &matrix.name, None)?
                    } else {
                        let shape = &weights.tensors[&matrix.name].shape;
                        quantize_bf16_matrix(&weights.host_bf16(&matrix.name)?, shape[0], shape[1])?
                    };
                    raw.extend(bytes);
                    scales.extend(blocks);
                }
            }
            Some(ExpertSource::FusedHalf(half)) => {
                let shape = &weights.tensors[&self.name].shape;
                let rows = shape[1] / 2;
                let cols = shape[2];
                for values in expert_projection(&weights.host_bf16(&self.name)?, rows * cols, *half)
                {
                    let (bytes, blocks) = quantize_bf16_matrix(values, rows, cols)?;
                    raw.extend(bytes);
                    scales.extend(blocks);
                }
            }
            None => {
                let shape = &weights.tensors[&self.name].shape;
                let rows = shape[shape.len() - 2];
                let cols = shape[shape.len() - 1];
                let mut values = weights.host_bf16(&self.name)?;
                if let Some(order) = self.transform.permutation {
                    values = order.reorder(&values, cols);
                }
                for matrix in values.chunks_exact(rows * cols) {
                    let (bytes, blocks) = quantize_bf16_matrix(matrix, rows, cols)?;
                    raw.extend(bytes);
                    scales.extend(blocks);
                }
            }
        }
        Ok(Some((raw, scales)))
    }

    fn prepare(
        &mut self,
        runtime: &Runtime,
        weights: &mut Weights,
        map_tensors: bool,
    ) -> Result<()> {
        let name = &self.name;
        if let Some(ExpertSource::Separate(matrices)) = &self.experts {
            if map_tensors {
                let len = self.elements * 2;
                let mut bytes = MmapMut::map_anon(len)?;
                let expert_bytes = len / matrices.len();
                for (dst, matrix) in bytes.chunks_exact_mut(expert_bytes).zip(matrices) {
                    let values = matrix
                        .to_bf16(weights)
                        .with_context(|| format!("converting {}", matrix.name))?;
                    for (dst, value) in dst.chunks_exact_mut(2).zip(values) {
                        dst.copy_from_slice(&value.to_le_bytes());
                    }
                }
                self.prepared_expert = Some(ExpertMemory::Mapped(MappedWeight::new(
                    bytes.make_read_only()?,
                    0,
                    len,
                )?));
            } else {
                let mut values = Vec::with_capacity(self.elements);
                for matrix in matrices {
                    values.extend(
                        matrix
                            .to_bf16(weights)
                            .with_context(|| format!("converting {}", matrix.name))?,
                    );
                }
                self.prepared_expert = Some(ExpertMemory::Device(runtime.mem_u16(&values)?));
            }
            return Ok(());
        }
        if let Some(ExpertSource::FusedHalf(half)) = self.experts {
            let shape = &weights.tensors[name].shape;
            let half_len = shape[1] / 2 * shape[2];
            // The checkpoint stores gate and up together for each expert.
            let bf16 = weights.tensors[name].dtype == Dtype::BF16;
            if map_tensors {
                let len = shape[0] * half_len * 2;
                let mut bytes = MmapMut::map_anon(len)?;
                if bf16 {
                    for (dst, src) in bytes.chunks_exact_mut(half_len * 2).zip(expert_projection(
                        weights.raw_bytes(name)?,
                        half_len * 2,
                        half,
                    )) {
                        dst.copy_from_slice(src);
                    }
                } else {
                    for (dst, value) in bytes
                        .chunks_exact_mut(2)
                        .zip(expert_projection(&weights.host_bf16(name)?, half_len, half).flatten())
                    {
                        dst.copy_from_slice(&value.to_le_bytes());
                    }
                }
                self.prepared_expert = Some(ExpertMemory::Mapped(MappedWeight::new(
                    bytes.make_read_only()?,
                    0,
                    len,
                )?));
                return Ok(());
            }
            let values = if bf16 {
                expert_projection(weights.raw_bytes(name)?, half_len * 2, half)
                    .flat_map(|expert| expert.chunks_exact(2))
                    .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
                    .collect::<Vec<_>>()
            } else {
                expert_projection(&weights.host_bf16(name)?, half_len, half)
                    .flatten()
                    .copied()
                    .collect()
            };
            self.prepared_expert = Some(ExpertMemory::Device(runtime.mem_u16(&values)?));
            return Ok(());
        }
        if let Some(fp8) = &self.fp8 {
            let values = fp8.to_bf16(weights, name)?;
            let values = match self.transform.permutation {
                Some(order) => order.reorder(&values, weights.tensors[name].shape[1]),
                None => values,
            };
            return weights.insert_prepared(name, runtime.mem_u16(&values)?);
        }
        let transform = self.transform;
        let f32_storage = self.dtype == WeightDtype::F32;
        if transform.permutation.is_none() && !transform.add_one && !transform.negative_exp {
            if map_tensors && !f32_storage && weights.tensors[name].dtype == Dtype::BF16 {
                return weights.prepare_bf16_mapping(name);
            }
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

fn expert_projection<T>(packed: &[T], half_len: usize, half: usize) -> impl Iterator<Item = &[T]> {
    let start = half * half_len;
    packed
        .chunks_exact(2 * half_len)
        .map(move |expert| &expert[start..start + half_len])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        hfhub::{HubCheckpoint, download_model},
        spec::weight_specs,
    };

    #[test]
    fn splits_fused_expert_projections_in_expert_order() {
        let packed = [10, 11, 20, 21, 30, 31, 40, 41];
        let projection = |half| {
            expert_projection(&packed, 2, half)
                .flatten()
                .copied()
                .collect::<Vec<_>>()
        };
        assert_eq!(projection(0), [10, 11, 30, 31]);
        assert_eq!(projection(1), [20, 21, 40, 41]);
    }
    use safetensors::{
        Dtype,
        tensor::{TensorView, serialize_to_file},
    };
    use std::fs;

    #[test]
    fn fp8_encoder_rounds_ties_even_and_pads_partial_blocks() -> Result<()> {
        for bits in 0_u8..127 {
            let value = e4m3fn_to_f32(bits);
            assert_eq!(encode_e4m3fn(value), bits);
            assert_eq!(encode_e4m3fn(-value), bits | 0x80);
            if bits < 126 {
                let midpoint = (value + e4m3fn_to_f32(bits + 1)) * 0.5;
                let expected = if bits & 1 == 0 { bits } else { bits + 1 };
                assert_eq!(encode_e4m3fn(midpoint), expected);
                assert_eq!(encode_e4m3fn(-midpoint), expected | 0x80);
            }
        }
        assert_eq!(encode_e4m3fn(1.0e30), 0x7e);
        let mut values = vec![0; 136 * 136];
        for (row, col, x) in [(0, 0, 1.0), (0, 128, 2.0), (128, 0, 3.0), (128, 128, 4.0)] {
            values[row * 136 + col] = f32_to_bf16(x);
        }
        let (raw, scales) = quantize_bf16_matrix(&values, 136, 136)?;
        assert_eq!(raw.len(), 256 * 256);
        assert_eq!(
            scales,
            vec![1.0 / 448.0, 2.0 / 448.0, 3.0 / 448.0, 4.0 / 448.0]
        );
        for row in 0..256 {
            for col in 0..256 {
                if row >= 136 || col >= 136 {
                    assert_eq!(raw[row * 256 + col], 0);
                }
            }
        }
        let (raw, scales) = quantize_bf16_matrix(&[0; 8 * 32], 8, 32)?;
        assert!(raw.iter().all(|&x| x == 0));
        assert_eq!(scales, [1.0]);
        Ok(())
    }

    #[test]
    fn moe_native_expert_operands_preserve_scales_and_split_fused_halves() -> Result<()> {
        let directory =
            std::env::temp_dir().join(format!("catena-moe-native-{}", std::process::id()));
        fs::create_dir_all(&directory)?;
        let result = (|| {
            let rows = 136;
            let cols = 8;
            let raw: Vec<u8> = (0..rows * cols).map(|i| (i % 120) as u8).collect();
            let scales: Vec<u8> = [0.125_f32, 0.75]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect();
            let fused: Vec<u8> = [1.0, 2.0, 3.0, 4.0]
                .into_iter()
                .flat_map(|v| vec![f32_to_bf16(v); rows * cols])
                .flat_map(u16::to_le_bytes)
                .collect();
            let path = directory.join("weights.safetensors");
            serialize_to_file(
                [
                    (
                        "expert",
                        TensorView::new(Dtype::F8_E4M3, vec![rows, cols], &raw)?,
                    ),
                    (
                        "expert_scale_inv",
                        TensorView::new(Dtype::F32, vec![2, 1], &scales)?,
                    ),
                    (
                        "fused",
                        TensorView::new(Dtype::BF16, vec![2, rows * 2, cols], &fused)?,
                    ),
                ],
                None,
                &path,
            )?;
            let weights = Weights::open(&path)?;
            let binding = |name: &str, experts| Binding {
                name: name.into(),
                dtype: WeightDtype::Bf16,
                transform: Transform::default(),
                fp8: None,
                experts,
                prepared_expert: None,
                elements: 2 * rows * cols,
            };
            let expert = ExpertMatrix {
                name: "expert".into(),
                fp8: Some(Fp8Weight::new(
                    &weights,
                    "expert",
                    &[rows, cols],
                    [128, 128],
                )?),
            };
            let b = binding("expert", Some(ExpertSource::Separate(vec![expert])));
            let (bytes, scales) = b.native_prefill(&weights, true)?.unwrap();
            assert_eq!(scales, [0.125, 0.75]);
            assert_eq!(bytes, pad_fp8_matrix(&raw, rows, cols));
            for half in [0, 1] {
                let b = binding("fused", Some(ExpertSource::FusedHalf(half)));
                let (bytes, scales) = b.native_prefill(&weights, true)?.unwrap();
                assert_eq!(bytes.len(), 2 * 256 * 128);
                for e in 0..2 {
                    let value = (e * 2 + half + 1) as f32;
                    assert_eq!(&scales[e * 2..e * 2 + 2], &[value / 448.0; 2]);
                    for row in 0..256 {
                        for col in 0..128 {
                            assert_eq!(
                                bytes[e * 256 * 128 + row * 128 + col],
                                if row < rows && col < cols { 0x7e } else { 0 }
                            );
                        }
                    }
                }
            }
            Ok(())
        })();
        fs::remove_dir_all(directory)?;
        result
    }

    #[test]
    fn dynamic_fp8_metadata_accepts_dense_and_moe_checkpoints() -> Result<()> {
        let mut quantization = serde_json::json!({
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "quant_type": "float",
            "weight_block_size": [128, 128],
        });
        for architecture in [
            ModelArchitecture::Qwen3Dense,
            ModelArchitecture::Qwen35Dense,
            ModelArchitecture::Qwen35Moe,
        ] {
            assert_eq!(
                fp8_block_size(&quantization, architecture)?,
                Some([128, 128])
            );
            quantization["quant_type"] = serde_json::json!("int");
            assert!(fp8_block_size(&quantization, architecture).is_err());
            quantization["quant_type"] = serde_json::json!("float");
        }
        assert!(fp8_block_size(&quantization, ModelArchitecture::Lfm3).is_err());
        Ok(())
    }

    #[test]
    fn fp8_head_permutations_preserve_bytes_scales_and_decoded_values() -> Result<()> {
        let directory =
            std::env::temp_dir().join(format!("catena-qwen35-fp8-permute-{}", std::process::id()));
        fs::create_dir_all(&directory)?;
        let result = (|| {
            for columns_order in [false, true] {
                let (rows, cols, prefix) = if columns_order {
                    (256, 512, 0)
                } else {
                    (768, 256, 256)
                };
                let raw: Vec<u8> = (0..rows * cols).map(|i| (i % 120) as u8).collect();
                let scales: Vec<f32> = (0..rows / 128 * (cols / 128))
                    .map(|i| (i + 1) as f32 * 0.125)
                    .collect();
                let scale_bytes: Vec<u8> = scales.iter().flat_map(|s| s.to_le_bytes()).collect();
                let path = directory.join("weights.safetensors");
                serialize_to_file(
                    [
                        (
                            "w",
                            TensorView::new(Dtype::F8_E4M3, vec![rows, cols], &raw)?,
                        ),
                        (
                            "w_scale_inv",
                            TensorView::new(
                                Dtype::F32,
                                vec![rows / 128, cols / 128],
                                &scale_bytes,
                            )?,
                        ),
                    ],
                    None,
                    &path,
                )?;
                let weights = Weights::open(&path)?;
                let fp8 = Fp8Weight::new(&weights, "w", &[rows, cols], [128, 128])?;
                let order = Permutation {
                    keys: 2,
                    replicas: 2,
                    head_width: 128,
                    prefix,
                    columns: columns_order,
                };
                let (bytes, native_scales) = fp8.native_operands(&weights, "w", Some(order))?;
                assert_eq!(bytes, order.reorder(&raw, cols));
                let expected = order.reorder(&fp8.to_bf16(&weights, "w")?, cols);
                let actual: Vec<u16> = bytes
                    .iter()
                    .enumerate()
                    .map(|(i, &b)| {
                        let scale =
                            native_scales[(i / cols) / 128 * (cols / 128) + (i % cols) / 128];
                        f32_to_bf16(e4m3fn_to_f32(b) * scale)
                    })
                    .collect();
                assert_eq!(actual, expected);
                let invalid = Permutation {
                    head_width: 64,
                    ..order
                };
                assert!(fp8.native_operands(&weights, "w", Some(invalid)).is_err());
            }
            Ok(())
        })();
        fs::remove_dir_all(directory)?;
        result
    }

    #[test]
    fn dynamic_fp8_uses_block_scales_across_shards() -> Result<()> {
        let directory =
            std::env::temp_dir().join(format!("catena-qwen-fp8-{}", std::process::id()));
        fs::create_dir_all(&directory)?;
        let result = (|| {
            let config = ModelConfig::qwen3_0_6b();
            let weight = "model.layers.0.self_attn.q_proj.weight";
            let scale = "model.layers.0.self_attn.q_proj.weight_scale_inv";
            let embedding = "model.embed_tokens.weight";
            let mut specs: Vec<_> = weight_specs(&config)?
                .into_iter()
                .filter(|spec| spec.name == embedding || spec.name == weight)
                .collect();
            specs[0].shape = vec![2, 2];
            specs[1].shape = vec![3, 5];
            let mut quantization = serde_json::json!({
                "activation_scheme": "dynamic",
                "modules_to_not_convert": null,
                "quant_method": "fp8",
                "weight_block_size": [2, 3],
            });
            assert_eq!(
                fp8_block_size(&quantization, config.architecture)?,
                Some([2, 3])
            );
            quantization["quant_type"] = serde_json::json!("float");
            assert!(fp8_block_size(&quantization, config.architecture)?.is_some());
            quantization["quant_type"] = serde_json::json!("int");
            assert!(fp8_block_size(&quantization, config.architecture).is_err());
            quantization["quant_type"] = serde_json::json!("float");
            quantization["weight_block_size"] = serde_json::json!([0, 3]);
            assert!(fp8_block_size(&quantization, config.architecture).is_err());
            quantization["weight_block_size"] = serde_json::json!([2, 3]);
            fs::write(
                directory.join("config.json"),
                serde_json::to_vec(&serde_json::json!({"quantization_config": quantization}))?,
            )?;
            serialize_to_file(
                [
                    (
                        embedding,
                        TensorView::new(Dtype::BF16, vec![2, 2], &[0; 8])?,
                    ),
                    (
                        weight,
                        TensorView::new(
                            Dtype::F8_E4M3,
                            vec![3, 5],
                            &[
                                0x80, 0x01, 0x15, 0x38, 0xfe, 0xb8, 0x7e, 0x00, 0x7a, 0x85, 0x0f,
                                0x42, 0xc8, 0x01, 0x7f,
                            ],
                        )?,
                    ),
                ],
                None,
                &directory.join("weights.safetensors"),
            )?;
            let scale_shard = directory.join("scales.safetensors");
            let index = directory.join("model.safetensors.index.json");
            fs::write(
                &index,
                serde_json::to_vec(&serde_json::json!({
                    "weight_map": {
                        (embedding): "weights.safetensors",
                        (weight): "weights.safetensors",
                        (scale): "scales.safetensors",
                    }
                }))?,
            )?;
            let scale_bits = [0x3ec1_u16, 0x4000, 0x4048, 0x3e80];
            let scale_bytes: Vec<_> = scale_bits.into_iter().flat_map(u16::to_le_bytes).collect();
            for shape in [vec![1, 4], vec![2, 2]] {
                serialize_to_file(
                    [(
                        scale,
                        TensorView::new(Dtype::BF16, shape.clone(), &scale_bytes)?,
                    )],
                    None,
                    &scale_shard,
                )?;
                let opened = QwenWeights::open(&index, &config, &specs);
                if shape == [1, 4] {
                    let error = opened.err().context("accepted malformed scale shape")?;
                    assert!(format!("{error:#}").contains(scale));
                    continue;
                }
                let weights = opened?;
                assert!(weights.bindings[0].fp8.is_none());
                let fp8 = weights.bindings[1]
                    .fp8
                    .as_ref()
                    .context("missing FP8 binding")?;
                let actual = fp8.to_bf16(&weights.weights, weight)?;
                // PyTorch float8_e4m3fn -> float32, block multiplication, then bfloat16.
                assert_eq!(
                    actual[..14],
                    [
                        0x8000, 0x3a41, 0x3c9d, 0x4000, 0xc460, 0xbec1, 0x4329, 0x0000, 0x4420,
                        0xbca0, 0x3dbc, 0x40fa, 0xc148, 0x3a00,
                    ]
                );
                assert!(crate::weights::bf16_to_f32(actual[14]).is_nan());
            }
            let error = QwenWeights::open(&directory.join("weights.safetensors"), &config, &specs)
                .err()
                .context("accepted missing scales")?;
            assert!(format!("{error:#}").contains(scale));
            Ok(())
        })();
        fs::remove_dir_all(directory)?;
        result
    }

    #[test]
    #[ignore = "requires a GPU and the Unsloth Qwen3 FP8 checkpoint"]
    fn qwen3_dynamic_fp8_runs_with_wmma() -> Result<()> {
        use crate::{catena::InferenceOptions, model::Model};

        let checkpoint = download_model("unsloth/Qwen3-0.6B-FP8")?;
        let HubCheckpoint::Safetensors { path, config } = &checkpoint else {
            anyhow::bail!("expected Unsloth safetensors");
        };
        let weights = QwenWeights::open(path, config, &weight_specs(config)?)?;
        assert_eq!(
            weights
                .bindings
                .iter()
                .filter(|binding| binding.fp8.is_some())
                .count(),
            196
        );
        // PyTorch reference values sampled across several block rows and columns.
        for (name, expected) in [
            (
                "model.layers.0.self_attn.q_proj.weight",
                [0x3a2a, 0x3aa6, 0xbd33, 0xbc82, 0x3ca2, 0x3d60, 0xbd47],
            ),
            (
                "model.layers.0.mlp.down_proj.weight",
                [0xbc4e, 0x3c87, 0x3cad, 0x3c3c, 0xbc9e, 0xbcbc, 0xbb2f],
            ),
        ] {
            let binding = weights
                .bindings
                .iter()
                .find(|binding| binding.name == name)
                .context("missing reference tensor")?;
            let values = binding
                .fp8
                .as_ref()
                .context("reference tensor is not FP8")?
                .to_bf16(&weights.weights, name)?;
            let columns = weights.weights.tensors[name].shape[1];
            let indices = [
                (1, 2),
                (12, 251),
                (129, 6),
                (129, 251),
                (513, 400),
                (767, 687),
                (1023, 1023),
            ];
            assert_eq!(
                indices.map(|(row, col)| values[row * columns + col]),
                expected,
                "{name}"
            );
        }
        drop(weights);
        let prompt = checkpoint.tokenizer()?.encode("The capital of France is")?;
        let options = InferenceOptions::default().with_wmma_prefill(true);
        let model = Model::load_checkpoint_with_options(&checkpoint, options)?;
        let mut session = model.session(prompt.len() + 1)?;
        let logits = session.prefill_logits(&prompt)?;
        assert!(logits.iter().all(|value| value.is_finite()));
        let next = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .context("missing prefill logits")?
            .0;
        assert!(
            session
                .decode_logits(next as u64)?
                .iter()
                .all(|value| value.is_finite())
        );
        Ok(())
    }

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
            fp8: None,
            experts: None,
            prepared_expert: None,
            elements: 4,
        };
        let values = [0.0_f32, 1.0, 2.0, 3.0];
        assert_eq!(
            binding.transformed_f32(values.to_vec(), 1),
            [0, 2, 1, 3].map(|i| -values[i].exp())
        );
        assert!(!Transform::for_tensor(&config, "model.layers.0.linear_attn.norm.weight").add_one);
    }

    #[test]
    #[ignore = "requires CUDA and downloads tiny-random/qwen3.6-moe"]
    fn native_fp8_tiny_moe_prefill_matches_dequantized_weights() -> Result<()> {
        use crate::{
            catena::{InferenceOptions, PrefillDtype},
            model::Model,
        };
        use std::collections::BTreeMap;
        let checkpoint = download_model("tiny-random/qwen3.6-moe")?;
        let HubCheckpoint::Safetensors { path, config } = checkpoint else {
            anyhow::bail!("expected safetensors");
        };
        let specs = weight_specs(&config)?;
        let mut qwen = QwenWeights::open(&path, &config, &specs)?;
        qwen.enable_fp8_prefill(&config)?;
        // Build a native, separately stored FP8 expert fixture from the tiny
        // BF16 export, so the BF16 reference dequantizes exactly the same weights.
        let mut tensors: BTreeMap<String, (Dtype, Vec<usize>, Vec<u8>)> = qwen
            .weights
            .tensors
            .iter()
            .map(|(name, t)| {
                Ok((
                    name.clone(),
                    (
                        t.dtype,
                        t.shape.clone(),
                        qwen.weights.raw_bytes(name)?.to_vec(),
                    ),
                ))
            })
            .collect::<Result<_>>()?;
        for &index in &qwen.prefill {
            let binding = &qwen.bindings[index];
            let names: Vec<&str> = match &binding.experts {
                Some(ExpertSource::Separate(matrices)) => {
                    matrices.iter().map(|m| m.name.as_str()).collect()
                }
                _ => vec![binding.name.as_str()],
            };
            for name in names {
                let shape = &qwen.weights.tensors[name].shape;
                let [matrix_rows, cols] = *shape.as_slice() else {
                    anyhow::bail!("tiny fixture expects separate 2D matrices");
                };
                let matrix = qwen.weights.host_bf16(name)?;
                let name = name.to_owned();
                let (padded, scales) = quantize_bf16_matrix(&matrix, matrix_rows, cols)?;
                let raw = padded
                    .chunks_exact(cols.next_multiple_of(128))
                    .take(matrix_rows)
                    .flat_map(|row| row[..cols].iter().copied())
                    .collect();
                tensors.insert(name.clone(), (Dtype::F8_E4M3, vec![matrix_rows, cols], raw));
                tensors.insert(
                    format!("{name}_scale_inv"),
                    (
                        Dtype::F32,
                        vec![matrix_rows.div_ceil(128), cols.div_ceil(128)],
                        scales.into_iter().flat_map(f32::to_le_bytes).collect(),
                    ),
                );
            }
        }
        let directory =
            std::env::temp_dir().join(format!("catena-native-fp8-moe-{}", std::process::id()));
        fs::create_dir_all(&directory)?;
        let result = (|| {
            let path = directory.join("model.safetensors");
            let views = tensors
                .iter()
                .map(|(name, (dtype, shape, raw))| {
                    Ok((name.as_str(), TensorView::new(*dtype, shape.clone(), raw)?))
                })
                .collect::<Result<Vec<_>>>()?;
            serialize_to_file(views, None, &path)?;
            fs::write(
                directory.join("config.json"),
                serde_json::to_vec(&serde_json::json!({"quantization_config": {
                    "quant_method": "fp8", "activation_scheme": "dynamic", "quant_type": "float", "weight_block_size": [128,128]
                }}))?,
            )?;
            let checkpoint = HubCheckpoint::Safetensors { path, config };
            let reference =
                Model::load_checkpoint_with_options(&checkpoint, InferenceOptions::default())?;
            let fp8 = Model::load_checkpoint_with_options(
                &checkpoint,
                InferenceOptions::default()
                    .with_wmma_prefill(true)
                    .with_prefill_dtype(PrefillDtype::Fp8),
            )?;
            for n in [2, 17, 65, 128] {
                let tokens: Vec<_> = (0..n).map(|i| 100 + i * 17 % 5000).collect();
                let expected = reference.logits(&tokens)?;
                let actual = fp8.logits(&tokens)?;
                let dot: f64 = expected
                    .iter()
                    .zip(&actual)
                    .map(|(&a, &b)| a as f64 * b as f64)
                    .sum();
                let norm = |v: &[f32]| v.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
                let cosine = dot / (norm(&expected) * norm(&actual));
                eprintln!("native FP8 MoE pp{n}, BF16 dequantized reference cosine={cosine:.8}");
                assert!(actual.iter().all(|x| x.is_finite()));
                assert!(cosine > 0.98);
            }
            Ok(())
        })();
        fs::remove_dir_all(directory)?;
        result
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
    #[ignore = "requires the Qwen3.6-35B-A3B checkpoint"]
    fn resolves_and_validates_qwen36_moe_safetensors() -> Result<()> {
        let checkpoint = download_model("Qwen/Qwen3.6-35B-A3B")?;
        let HubCheckpoint::Safetensors { path, config } = &checkpoint else {
            anyhow::bail!("expected Qwen3.6 MoE safetensors");
        };
        assert!(checkpoint.tokenizer()?.chat_template().is_some());
        let specs = weight_specs(config)?;
        assert_eq!(specs, weight_specs(&ModelConfig::qwen36_35b_a3b())?);
        let weights = QwenWeights::open(path, config, &specs)?;
        assert_eq!(
            weights
                .bindings
                .iter()
                .filter(|b| matches!(b.experts, Some(ExpertSource::FusedHalf(_))))
                .count(),
            2 * config.num_layers,
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires the Qwen3.6-35B-A3B-FP8 checkpoint"]
    fn resolves_and_converts_qwen36_moe_fp8_safetensors() -> Result<()> {
        let checkpoint = download_model("Qwen/Qwen3.6-35B-A3B-FP8")?;
        let HubCheckpoint::Safetensors { path, config } = &checkpoint else {
            anyhow::bail!("expected Qwen3.6 MoE FP8 safetensors");
        };
        let weights = QwenWeights::open(path, config, &weight_specs(config)?)?;
        assert_eq!(
            weights
                .bindings
                .iter()
                .filter(|b| matches!(b.experts, Some(ExpertSource::Separate(_))))
                .count(),
            3 * config.num_layers,
        );

        let layer = "model.language_model.layers.0";
        // Independently decoded samples spanning several scale blocks.
        for (projection, expert, expected) in [
            ("gate_proj", 0, [0x3b7e, 0x3c6e, 0x39aa, 0x8000, 0x3c68]),
            ("down_proj", 255, [0x3bc7, 0x8000, 0x8000, 0x8000, 0xbba3]),
        ] {
            let name = format!("{layer}.mlp.experts.{projection}.weight");
            let binding = weights
                .bindings
                .iter()
                .find(|b| b.name == name)
                .context("missing expert binding")?;
            let Some(ExpertSource::Separate(matrices)) = &binding.experts else {
                anyhow::bail!("expected separate expert matrices");
            };
            assert_eq!(matrices.len(), config.num_experts);
            let matrix = &matrices[expert];
            assert_eq!(
                matrix.name,
                format!("{layer}.mlp.experts.{expert}.{projection}.weight")
            );
            let values = matrix.to_bf16(&weights.weights)?;
            let cols = weights.weights.tensors[&matrix.name].shape[1];
            assert_eq!(
                [(0, 0), (0, 128), (127, 127), (128, 128), (255, 257)]
                    .map(|(row, col)| values[row * cols + col]),
                expected,
            );
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
