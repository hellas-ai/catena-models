//! Runtime model configuration and GGUF tensor ABI.

use anyhow::{Result, ensure};

// Kept for the checked-in legacy Catena core and its tests.
pub const VOCAB_SIZE: usize = 151_936;
pub const HIDDEN_SIZE: usize = 2_048;
pub const INTERMEDIATE_SIZE: usize = 768;
pub const NUM_LAYERS: usize = 48;
pub const NUM_ATTENTION_HEADS: usize = 32;
pub const NUM_KEY_VALUE_HEADS: usize = 4;
pub const HEAD_DIM: usize = 128;
pub const ROPE_HALF_DIM: usize = HEAD_DIM / 2;
pub const MAX_POSITION_EMBEDDINGS: usize = 32_768;
pub const RMS_NORM_EPS: f32 = 1.0e-6;
pub const ROPE_THETA: f32 = 1_000_000.0;
pub const NUM_EXPERTS: usize = 128;
pub const EXPERTS_PER_TOKEN: usize = 8;
pub const EOS_TOKEN_ID: u64 = 151_643;
pub const TENSORS_PER_LAYER: usize = 12;
pub const WEIGHT_COUNT: usize = 579;
pub const MODEL_INPUT_COUNT: usize = 583;
pub const KV_CACHE_ELEMENTS_PER_TOKEN: usize = 49_152;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelArchitecture {
    Qwen3Moe,
    Qwen35Moe,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    FullAttention,
    LinearAttention,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModelConfig {
    pub architecture: ModelArchitecture,
    pub vocabulary_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub rope_dimension_count: usize,
    pub num_experts: usize,
    pub experts_per_token: usize,
    pub shared_expert_intermediate_size: Option<usize>,
    pub full_attention_interval: Option<usize>,
    pub linear_conv_kernel: Option<usize>,
    pub linear_state_size: Option<usize>,
    pub linear_group_count: Option<usize>,
    pub linear_time_step_rank: Option<usize>,
    pub linear_inner_size: Option<usize>,
    pub eos_token_id: Option<u64>,
}

impl ModelConfig {
    pub fn qwen3_30b() -> Self {
        Self {
            architecture: ModelArchitecture::Qwen3Moe,
            vocabulary_size: VOCAB_SIZE,
            hidden_size: HIDDEN_SIZE,
            intermediate_size: INTERMEDIATE_SIZE,
            num_layers: NUM_LAYERS,
            num_attention_heads: NUM_ATTENTION_HEADS,
            num_key_value_heads: NUM_KEY_VALUE_HEADS,
            head_dim: HEAD_DIM,
            max_position_embeddings: MAX_POSITION_EMBEDDINGS,
            rms_norm_eps: RMS_NORM_EPS,
            rope_theta: ROPE_THETA,
            rope_dimension_count: HEAD_DIM,
            num_experts: NUM_EXPERTS,
            experts_per_token: EXPERTS_PER_TOKEN,
            shared_expert_intermediate_size: None,
            full_attention_interval: None,
            linear_conv_kernel: None,
            linear_state_size: None,
            linear_group_count: None,
            linear_time_step_rank: None,
            linear_inner_size: None,
            eos_token_id: Some(EOS_TOKEN_ID),
        }
    }

    pub fn validate(&self) -> Result<()> {
        for (name, value) in [
            ("vocabulary size", self.vocabulary_size),
            ("hidden size", self.hidden_size),
            ("intermediate size", self.intermediate_size),
            ("layer count", self.num_layers),
            ("attention heads", self.num_attention_heads),
            ("KV heads", self.num_key_value_heads),
            ("head dimension", self.head_dim),
            ("context length", self.max_position_embeddings),
            ("expert count", self.num_experts),
            ("experts used", self.experts_per_token),
        ] {
            ensure!(value > 0, "GGUF {name} must be non-zero");
        }
        ensure!(
            self.num_attention_heads
                .is_multiple_of(self.num_key_value_heads),
            "attention head count must be divisible by KV head count"
        );
        ensure!(
            self.experts_per_token <= self.num_experts,
            "experts used exceeds expert count"
        );
        ensure!(
            self.rope_dimension_count > 0 && self.rope_dimension_count <= self.head_dim,
            "rotary dimension must be within the attention head"
        );
        if self.architecture == ModelArchitecture::Qwen35Moe {
            for (name, value) in [
                ("shared expert size", self.shared_expert_intermediate_size),
                ("full-attention interval", self.full_attention_interval),
                ("linear convolution kernel", self.linear_conv_kernel),
                ("linear state size", self.linear_state_size),
                ("linear group count", self.linear_group_count),
                ("linear time-step rank", self.linear_time_step_rank),
                ("linear inner size", self.linear_inner_size),
            ] {
                ensure!(value.is_some_and(|v| v > 0), "GGUF is missing {name}");
            }
        }
        Ok(())
    }

    pub fn layer_type(&self, layer: usize) -> LayerType {
        match self.full_attention_interval {
            Some(n) if (layer + 1).is_multiple_of(n) => LayerType::FullAttention,
            Some(_) => LayerType::LinearAttention,
            None => LayerType::FullAttention,
        }
    }
    pub fn query_size(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }
    pub fn key_value_size(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightDtype {
    F32,
    Bf16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorSource {
    pub name: String,
    pub shape: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorSpec {
    pub name: String,
    pub gguf_name: String,
    pub dtype: WeightDtype,
    pub shape: Vec<usize>,
    pub sources: Vec<TensorSource>,
}

impl TensorSpec {
    pub fn element_count(&self) -> usize {
        self.shape.iter().product()
    }
    fn one(name: String, gguf_name: String, dtype: WeightDtype, shape: Vec<usize>) -> Self {
        Self {
            sources: vec![TensorSource {
                name: name.clone(),
                shape: shape.clone(),
            }],
            name,
            gguf_name,
            dtype,
            shape,
        }
    }
    fn experts(
        c: &ModelConfig,
        layer: usize,
        projection: &str,
        gguf: &str,
        source_shape: Vec<usize>,
    ) -> Self {
        let sources = (0..c.num_experts)
            .map(|expert| TensorSource {
                name: format!("model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"),
                shape: source_shape.clone(),
            })
            .collect();
        let mut shape = vec![c.num_experts];
        shape.extend(source_shape);
        Self {
            name: format!("model.layers.{layer}.mlp.experts.{projection}.weight"),
            gguf_name: format!("blk.{layer}.{gguf}.weight"),
            dtype: WeightDtype::Bf16,
            shape,
            sources,
        }
    }
}

fn layer_tensor(
    layer: usize,
    name: &str,
    gguf: &str,
    dtype: WeightDtype,
    shape: Vec<usize>,
) -> TensorSpec {
    TensorSpec::one(
        format!("model.layers.{layer}.{name}"),
        format!("blk.{layer}.{gguf}"),
        dtype,
        shape,
    )
}

pub fn weight_specs(c: &ModelConfig) -> Result<Vec<TensorSpec>> {
    c.validate()?;
    let mut out = vec![TensorSpec::one(
        "model.embed_tokens.weight".into(),
        "token_embd.weight".into(),
        WeightDtype::Bf16,
        vec![c.vocabulary_size, c.hidden_size],
    )];
    for l in 0..c.num_layers {
        out.push(layer_tensor(
            l,
            "input_layernorm.weight",
            "attn_norm.weight",
            WeightDtype::F32,
            vec![c.hidden_size],
        ));
        match (c.architecture, c.layer_type(l)) {
            (ModelArchitecture::Qwen3Moe, _)
            | (ModelArchitecture::Qwen35Moe, LayerType::FullAttention) => {
                let q = c.query_size()
                    * if c.architecture == ModelArchitecture::Qwen35Moe {
                        2
                    } else {
                        1
                    };
                out.extend([
                    layer_tensor(
                        l,
                        "self_attn.q_proj.weight",
                        "attn_q.weight",
                        WeightDtype::Bf16,
                        vec![q, c.hidden_size],
                    ),
                    layer_tensor(
                        l,
                        "self_attn.k_proj.weight",
                        "attn_k.weight",
                        WeightDtype::Bf16,
                        vec![c.key_value_size(), c.hidden_size],
                    ),
                    layer_tensor(
                        l,
                        "self_attn.v_proj.weight",
                        "attn_v.weight",
                        WeightDtype::Bf16,
                        vec![c.key_value_size(), c.hidden_size],
                    ),
                    layer_tensor(
                        l,
                        "self_attn.o_proj.weight",
                        "attn_output.weight",
                        WeightDtype::Bf16,
                        vec![c.hidden_size, c.query_size()],
                    ),
                    layer_tensor(
                        l,
                        "self_attn.q_norm.weight",
                        "attn_q_norm.weight",
                        WeightDtype::F32,
                        vec![c.head_dim],
                    ),
                    layer_tensor(
                        l,
                        "self_attn.k_norm.weight",
                        "attn_k_norm.weight",
                        WeightDtype::F32,
                        vec![c.head_dim],
                    ),
                ]);
            }
            (ModelArchitecture::Qwen35Moe, LayerType::LinearAttention) => {
                let inner = c.linear_inner_size.unwrap();
                let rank = c.linear_time_step_rank.unwrap();
                out.extend([
                    layer_tensor(
                        l,
                        "linear_attn.in_proj_qkv.weight",
                        "attn_qkv.weight",
                        WeightDtype::Bf16,
                        vec![2 * inner, c.hidden_size],
                    ),
                    layer_tensor(
                        l,
                        "linear_attn.in_proj_z.weight",
                        "attn_gate.weight",
                        WeightDtype::Bf16,
                        vec![inner, c.hidden_size],
                    ),
                    layer_tensor(
                        l,
                        "linear_attn.in_proj_a.weight",
                        "ssm_alpha.weight",
                        WeightDtype::Bf16,
                        vec![rank, c.hidden_size],
                    ),
                    layer_tensor(
                        l,
                        "linear_attn.in_proj_b.weight",
                        "ssm_beta.weight",
                        WeightDtype::Bf16,
                        vec![rank, c.hidden_size],
                    ),
                    layer_tensor(
                        l,
                        "linear_attn.dt_bias",
                        "ssm_dt.bias",
                        WeightDtype::F32,
                        vec![rank],
                    ),
                    layer_tensor(
                        l,
                        "linear_attn.A_log",
                        "ssm_a",
                        WeightDtype::F32,
                        vec![rank],
                    ),
                    layer_tensor(
                        l,
                        "linear_attn.norm.weight",
                        "ssm_norm.weight",
                        WeightDtype::F32,
                        vec![c.linear_state_size.unwrap()],
                    ),
                    layer_tensor(
                        l,
                        "linear_attn.out_proj.weight",
                        "ssm_out.weight",
                        WeightDtype::Bf16,
                        vec![c.hidden_size, inner],
                    ),
                    layer_tensor(
                        l,
                        "linear_attn.conv1d.weight",
                        "ssm_conv1d.weight",
                        WeightDtype::F32,
                        vec![2 * inner, c.linear_conv_kernel.unwrap()],
                    ),
                ]);
            }
        }
        let post_norm = if c.architecture == ModelArchitecture::Qwen35Moe {
            "post_attention_norm.weight"
        } else {
            "ffn_norm.weight"
        };
        out.push(layer_tensor(
            l,
            "post_attention_layernorm.weight",
            post_norm,
            WeightDtype::F32,
            vec![c.hidden_size],
        ));
        out.push(layer_tensor(
            l,
            "mlp.gate.weight",
            "ffn_gate_inp.weight",
            WeightDtype::F32,
            vec![c.num_experts, c.hidden_size],
        ));
        out.extend([
            TensorSpec::experts(
                c,
                l,
                "gate_proj",
                "ffn_gate_exps",
                vec![c.intermediate_size, c.hidden_size],
            ),
            TensorSpec::experts(
                c,
                l,
                "up_proj",
                "ffn_up_exps",
                vec![c.intermediate_size, c.hidden_size],
            ),
            TensorSpec::experts(
                c,
                l,
                "down_proj",
                "ffn_down_exps",
                vec![c.hidden_size, c.intermediate_size],
            ),
        ]);
        if let Some(shared) = c.shared_expert_intermediate_size {
            out.extend([
                layer_tensor(
                    l,
                    "mlp.shared_expert_gate.weight",
                    "ffn_gate_inp_shexp.weight",
                    WeightDtype::F32,
                    vec![1, c.hidden_size],
                ),
                layer_tensor(
                    l,
                    "mlp.shared_expert.gate_proj.weight",
                    "ffn_gate_shexp.weight",
                    WeightDtype::Bf16,
                    vec![shared, c.hidden_size],
                ),
                layer_tensor(
                    l,
                    "mlp.shared_expert.up_proj.weight",
                    "ffn_up_shexp.weight",
                    WeightDtype::Bf16,
                    vec![shared, c.hidden_size],
                ),
                layer_tensor(
                    l,
                    "mlp.shared_expert.down_proj.weight",
                    "ffn_down_shexp.weight",
                    WeightDtype::Bf16,
                    vec![c.hidden_size, shared],
                ),
            ]);
        }
    }
    out.extend([
        TensorSpec::one(
            "model.norm.weight".into(),
            "output_norm.weight".into(),
            WeightDtype::F32,
            vec![c.hidden_size],
        ),
        TensorSpec::one(
            "lm_head.weight".into(),
            "output.weight".into(),
            WeightDtype::Bf16,
            vec![c.vocabulary_size, c.hidden_size],
        ),
    ]);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn qwen35() -> ModelConfig {
        let mut c = ModelConfig::qwen3_30b();
        c.architecture = ModelArchitecture::Qwen35Moe;
        c.vocabulary_size = 248_320;
        c.intermediate_size = 512;
        c.num_layers = 40;
        c.num_attention_heads = 16;
        c.num_key_value_heads = 2;
        c.head_dim = 256;
        c.max_position_embeddings = 262_144;
        c.rope_theta = 10_000_000.;
        c.rope_dimension_count = 64;
        c.num_experts = 256;
        c.shared_expert_intermediate_size = Some(512);
        c.full_attention_interval = Some(4);
        c.linear_conv_kernel = Some(4);
        c.linear_state_size = Some(128);
        c.linear_group_count = Some(16);
        c.linear_time_step_rank = Some(32);
        c.linear_inner_size = Some(4096);
        c
    }
    #[test]
    fn qwen3_inventory() -> Result<()> {
        let s = weight_specs(&ModelConfig::qwen3_30b())?;
        assert_eq!(s.len(), 579);
        assert_eq!(
            s.iter().map(TensorSpec::element_count).sum::<usize>(),
            30_532_122_624
        );
        Ok(())
    }
    #[test]
    fn qwen35_inventory() -> Result<()> {
        let c = qwen35();
        let s = weight_specs(&c)?;
        assert_eq!(s.len(), 733);
        assert_eq!(
            s.iter()
                .filter(|s| s.gguf_name.ends_with("attn_q.weight"))
                .count(),
            10
        );
        assert_eq!(
            s.iter()
                .filter(|s| s.gguf_name.ends_with("attn_qkv.weight"))
                .count(),
            30
        );
        Ok(())
    }
}
