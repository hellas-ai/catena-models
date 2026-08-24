//! Runtime model configuration and GGUF tensor ABI.

use anyhow::{Result, ensure};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelArchitecture {
    Qwen3Dense,
    Qwen3Moe,
    Qwen35Dense,
    Qwen35Moe,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    FullAttention,
    LinearAttention,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModelConfig {
    pub model_name: Option<String>,
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
    pub fn is_qwen35(&self) -> bool {
        matches!(
            self.architecture,
            ModelArchitecture::Qwen35Dense | ModelArchitecture::Qwen35Moe
        )
    }

    pub fn is_dense(&self) -> bool {
        matches!(
            self.architecture,
            ModelArchitecture::Qwen3Dense | ModelArchitecture::Qwen35Dense
        )
    }

    pub fn qwen3_0_6b() -> Self {
        let mut config = Self::qwen3_30b();
        config.model_name = Some("Qwen3-0.6B".into());
        config.architecture = ModelArchitecture::Qwen3Dense;
        config.hidden_size = 1_024;
        config.intermediate_size = 3_072;
        config.num_layers = 28;
        config.num_attention_heads = 16;
        config.num_key_value_heads = 8;
        config.max_position_embeddings = 40_960;
        config.num_experts = 0;
        config.experts_per_token = 0;
        config
    }

    pub fn qwen3_1_7b() -> Self {
        let mut config = Self::qwen3_0_6b();
        config.model_name = Some("Qwen3-1.7B".into());
        config.hidden_size = 2_048;
        config.intermediate_size = 6_144;
        config
    }

    pub fn qwen3_30b() -> Self {
        Self {
            model_name: Some("Qwen3-30B-A3B-Base".into()),
            architecture: ModelArchitecture::Qwen3Moe,
            vocabulary_size: 151_936,
            hidden_size: 2_048,
            intermediate_size: 768,
            num_layers: 48,
            num_attention_heads: 32,
            num_key_value_heads: 4,
            head_dim: 128,
            max_position_embeddings: 32_768,
            rms_norm_eps: 1.0e-6,
            rope_theta: 1_000_000.0,
            rope_dimension_count: 128,
            num_experts: 128,
            experts_per_token: 8,
            shared_expert_intermediate_size: None,
            full_attention_interval: None,
            linear_conv_kernel: None,
            linear_state_size: None,
            linear_group_count: None,
            linear_time_step_rank: None,
            linear_inner_size: None,
            eos_token_id: Some(151_643),
        }
    }

    pub fn qwen35_0_8b() -> Self {
        Self {
            model_name: Some("Qwen3.5-0.8B".into()),
            architecture: ModelArchitecture::Qwen35Dense,
            vocabulary_size: 248_320,
            hidden_size: 1_024,
            intermediate_size: 3_584,
            num_layers: 24,
            num_attention_heads: 8,
            num_key_value_heads: 2,
            head_dim: 256,
            max_position_embeddings: 262_144,
            rms_norm_eps: 1.0e-6,
            rope_theta: 10_000_000.0,
            rope_dimension_count: 64,
            num_experts: 0,
            experts_per_token: 0,
            shared_expert_intermediate_size: None,
            full_attention_interval: Some(4),
            linear_conv_kernel: Some(4),
            linear_state_size: Some(128),
            linear_group_count: Some(16),
            linear_time_step_rank: Some(16),
            linear_inner_size: Some(2_048),
            eos_token_id: Some(248_044),
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
        ] {
            ensure!(value > 0, "GGUF {name} must be non-zero");
        }
        ensure!(
            self.num_attention_heads
                .is_multiple_of(self.num_key_value_heads),
            "attention head count must be divisible by KV head count"
        );
        match self.architecture {
            ModelArchitecture::Qwen3Dense | ModelArchitecture::Qwen35Dense => ensure!(
                self.num_experts == 0 && self.experts_per_token == 0,
                "dense Qwen metadata must not define routed experts"
            ),
            ModelArchitecture::Qwen3Moe | ModelArchitecture::Qwen35Moe => {
                ensure!(self.num_experts > 0, "GGUF expert count must be non-zero");
                ensure!(
                    self.experts_per_token > 0,
                    "GGUF experts used must be non-zero"
                );
                ensure!(
                    self.experts_per_token <= self.num_experts,
                    "experts used exceeds expert count"
                );
            }
        }
        ensure!(
            self.rope_dimension_count > 0 && self.rope_dimension_count <= self.head_dim,
            "rotary dimension must be within the attention head"
        );
        if self.is_qwen35() {
            for (name, value) in [
                ("full-attention interval", self.full_attention_interval),
                ("linear convolution kernel", self.linear_conv_kernel),
                ("linear state size", self.linear_state_size),
                ("linear group count", self.linear_group_count),
                ("linear time-step rank", self.linear_time_step_rank),
                ("linear inner size", self.linear_inner_size),
            ] {
                ensure!(value.is_some_and(|v| v > 0), "GGUF is missing {name}");
            }
            if self.architecture == ModelArchitecture::Qwen35Moe {
                ensure!(
                    self.shared_expert_intermediate_size.is_some_and(|v| v > 0),
                    "GGUF is missing shared expert size"
                );
            }
            ensure!(
                self.linear_inner_size
                    .unwrap()
                    .is_multiple_of(self.linear_time_step_rank.unwrap()),
                "linear inner size must be divisible by the value-head count"
            );
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
    pub fn linear_key_size(&self) -> Option<usize> {
        self.linear_group_count?
            .checked_mul(self.linear_state_size?)
    }
    pub fn linear_value_head_size(&self) -> Option<usize> {
        Some(self.linear_inner_size? / self.linear_time_step_rank?)
    }
    pub fn linear_qkv_size(&self) -> Option<usize> {
        self.linear_key_size()?
            .checked_mul(2)?
            .checked_add(self.linear_inner_size?)
    }
    pub fn kv_cache_elements_per_token(&self) -> Result<usize> {
        self.num_layers
            .checked_mul(2)
            .and_then(|value| value.checked_mul(self.num_key_value_heads))
            .and_then(|value| value.checked_mul(self.head_dim))
            .ok_or_else(|| anyhow::anyhow!("KV cache stride overflowed"))
    }
    pub fn cache_elements(&self, capacity: usize) -> Result<usize> {
        let kv = capacity
            .checked_mul(self.kv_cache_elements_per_token()?)
            .ok_or_else(|| anyhow::anyhow!("KV cache element count overflowed"))?;
        if !self.is_qwen35() {
            return Ok(kv);
        }
        let conv_kernel = self
            .linear_conv_kernel
            .ok_or_else(|| anyhow::anyhow!("linear convolution kernel is missing"))?;
        let value_heads = self
            .linear_time_step_rank
            .ok_or_else(|| anyhow::anyhow!("linear value-head count is missing"))?;
        let key_dim = self
            .linear_state_size
            .ok_or_else(|| anyhow::anyhow!("linear key-head dimension is missing"))?;
        let conv = self
            .linear_qkv_size()
            .and_then(|value| value.checked_mul(conv_kernel - 1))
            .ok_or_else(|| anyhow::anyhow!("linear convolution cache size overflowed"))?;
        let recurrent = value_heads
            .checked_mul(key_dim)
            .and_then(|value| value.checked_mul(self.linear_value_head_size()?))
            .ok_or_else(|| anyhow::anyhow!("linear recurrent cache size overflowed"))?;
        let per_layer = conv
            .checked_add(recurrent)
            .ok_or_else(|| anyhow::anyhow!("linear layer cache size overflowed"))?;
        let linear = self
            .num_layers
            .checked_mul(per_layer)
            .ok_or_else(|| anyhow::anyhow!("linear model cache size overflowed"))?;
        kv.checked_add(linear)
            .ok_or_else(|| anyhow::anyhow!("hybrid cache size overflowed"))
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
            (ModelArchitecture::Qwen3Dense, _)
            | (ModelArchitecture::Qwen3Moe, _)
            | (ModelArchitecture::Qwen35Dense, LayerType::FullAttention)
            | (ModelArchitecture::Qwen35Moe, LayerType::FullAttention) => {
                let q = c.query_size() * if c.is_qwen35() { 2 } else { 1 };
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
            (
                ModelArchitecture::Qwen35Dense | ModelArchitecture::Qwen35Moe,
                LayerType::LinearAttention,
            ) => {
                let inner = c.linear_inner_size.unwrap();
                let rank = c.linear_time_step_rank.unwrap();
                let qkv = c.linear_qkv_size().unwrap();
                out.extend([
                    layer_tensor(
                        l,
                        "linear_attn.in_proj_qkv.weight",
                        "attn_qkv.weight",
                        WeightDtype::Bf16,
                        vec![qkv, c.hidden_size],
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
                        vec![c.linear_value_head_size().unwrap()],
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
                        vec![qkv, c.linear_conv_kernel.unwrap()],
                    ),
                ]);
            }
        }
        let post_norm = if c.is_qwen35() {
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
        if c.is_dense() {
            out.extend([
                layer_tensor(
                    l,
                    "mlp.gate_proj.weight",
                    "ffn_gate.weight",
                    WeightDtype::Bf16,
                    vec![c.intermediate_size, c.hidden_size],
                ),
                layer_tensor(
                    l,
                    "mlp.up_proj.weight",
                    "ffn_up.weight",
                    WeightDtype::Bf16,
                    vec![c.intermediate_size, c.hidden_size],
                ),
                layer_tensor(
                    l,
                    "mlp.down_proj.weight",
                    "ffn_down.weight",
                    WeightDtype::Bf16,
                    vec![c.hidden_size, c.intermediate_size],
                ),
            ]);
        } else {
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
        }
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
            if c.is_dense() {
                "token_embd.weight"
            } else {
                "output.weight"
            }
            .into(),
            WeightDtype::Bf16,
            vec![c.vocabulary_size, c.hidden_size],
        ),
    ]);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn physical_parameter_count(specs: &[TensorSpec]) -> usize {
        let mut names = HashSet::new();
        specs
            .iter()
            .filter(|spec| names.insert(spec.gguf_name.as_str()))
            .map(TensorSpec::element_count)
            .sum()
    }

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
        let config = ModelConfig::qwen3_30b();
        let s = weight_specs(&config)?;
        assert_eq!(s.len(), 579);
        assert_eq!(
            s.iter().map(TensorSpec::element_count).sum::<usize>(),
            30_532_122_624
        );
        assert_eq!(config.kv_cache_elements_per_token()?, 49_152);
        Ok(())
    }

    #[test]
    fn dense_qwen3_inventories_use_tied_embeddings() -> Result<()> {
        for (config, parameters) in [
            (ModelConfig::qwen3_0_6b(), 596_049_920),
            (ModelConfig::qwen3_1_7b(), 1_720_574_976),
        ] {
            let specs = weight_specs(&config)?;
            assert_eq!(specs.len(), 311);
            assert_eq!(
                specs
                    .iter()
                    .map(|spec| spec.gguf_name.as_str())
                    .collect::<HashSet<_>>()
                    .len(),
                310
            );
            assert_eq!(specs.first().unwrap().gguf_name, "token_embd.weight");
            assert_eq!(specs.last().unwrap().gguf_name, "token_embd.weight");
            assert_eq!(physical_parameter_count(&specs), parameters);
            assert!(
                specs
                    .iter()
                    .any(|spec| spec.gguf_name == "blk.0.ffn_gate.weight")
            );
            assert!(
                !specs
                    .iter()
                    .any(|spec| spec.gguf_name.contains("ffn_gate_exps"))
            );
        }
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

    #[test]
    fn qwen35_dense_inventory_uses_dense_ffn_and_tied_embeddings() -> Result<()> {
        let mut c = qwen35();
        c.architecture = ModelArchitecture::Qwen35Dense;
        c.hidden_size = 1_024;
        c.intermediate_size = 3_584;
        c.num_layers = 24;
        c.num_attention_heads = 8;
        c.num_key_value_heads = 2;
        c.num_experts = 0;
        c.experts_per_token = 0;
        c.shared_expert_intermediate_size = None;
        c.linear_time_step_rank = Some(16);
        c.linear_inner_size = Some(2_048);
        let specs = weight_specs(&c)?;
        assert_eq!(specs.len(), 321);
        assert_eq!(specs.first().unwrap().gguf_name, "token_embd.weight");
        assert_eq!(specs.last().unwrap().gguf_name, "token_embd.weight");
        assert!(specs.iter().any(|spec| {
            spec.gguf_name == "blk.0.attn_qkv.weight" && spec.shape == [6_144, 1_024]
        }));
        assert!(
            specs
                .iter()
                .any(|spec| spec.gguf_name == "blk.0.ffn_gate.weight")
        );
        assert!(
            !specs
                .iter()
                .any(|spec| spec.gguf_name.contains("ffn_gate_exps"))
        );
        assert_eq!(c.cache_elements(1)?, 6_758_400);
        Ok(())
    }
}
