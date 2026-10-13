use crate::spec::{ModelArchitecture, ModelConfig};
use anyhow::{Result, ensure};
use serde::Deserialize;
use std::path::PathBuf;
const FILES: &[&str] = &[
    "model_index.json",
    "scheduler/scheduler_config.json",
    "tokenizer/tokenizer.json",
    "text_encoder/config.json",
    "text_encoder/model.safetensors.index.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
];
pub fn snapshot(model: &str) -> Result<PathBuf> {
    crate::hfhub::resolve_model_files(model, FILES)
}
pub use crate::hfhub::read_json as read;
#[derive(Deserialize, Debug)]
pub struct TextConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
}
impl TextConfig {
    pub fn qwen(&self) -> Result<ModelConfig> {
        ensure!(
            self.hidden_size == 2560
                && self.num_hidden_layers == 36
                && self.intermediate_size == 9728
                && self.num_attention_heads == 32
                && self.num_key_value_heads == 8
                && self.head_dim == 128
                && self.vocab_size == 151936
                && self.rms_norm_eps == 1e-6
                && self.rope_theta == 1000000.,
            "only Klein 4B Qwen3 supported"
        );
        let mut c = ModelConfig::qwen3_0_6b();
        c.architecture = ModelArchitecture::Qwen3Dense;
        c.hidden_size = self.hidden_size;
        c.intermediate_size = self.intermediate_size;
        c.num_layers = 27;
        c.num_attention_heads = self.num_attention_heads;
        c.num_key_value_heads = self.num_key_value_heads;
        c.head_dim = self.head_dim;
        c.rope_dimension_count = self.head_dim;
        c.vocabulary_size = self.vocab_size;
        c.rms_norm_eps = self.rms_norm_eps;
        c.rope_theta = self.rope_theta;
        c.max_position_embeddings = self.max_position_embeddings;
        Ok(c)
    }
}
#[derive(Deserialize, Debug)]
pub struct TransformerConfig {
    pub attention_head_dim: usize,
    pub axes_dims_rope: Vec<usize>,
    pub eps: f32,
    pub guidance_embeds: bool,
    pub in_channels: usize,
    pub joint_attention_dim: usize,
    pub mlp_ratio: f32,
    pub num_attention_heads: usize,
    pub num_layers: usize,
    pub num_single_layers: usize,
    pub rope_theta: f32,
    pub timestep_guidance_channels: usize,
}
impl TransformerConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.attention_head_dim == 128
                && self.axes_dims_rope == [32, 32, 32, 32]
                && !self.guidance_embeds
                && self.in_channels == 128
                && self.joint_attention_dim == 7680
                && self.mlp_ratio == 3.0
                && self.num_attention_heads == 24
                && self.num_layers == 5
                && self.num_single_layers == 20
                && self.timestep_guidance_channels == 256,
            "unsupported Flux2 transformer configuration"
        );
        Ok(())
    }
}
#[derive(Deserialize, Debug)]
pub struct VaeConfig {
    pub block_out_channels: Vec<usize>,
    pub latent_channels: usize,
    pub norm_num_groups: usize,
    pub batch_norm_eps: f32,
    pub patch_size: Vec<usize>,
    pub force_upcast: bool,
    pub layers_per_block: usize,
}
impl VaeConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.block_out_channels == [128, 256, 512, 512]
                && self.latent_channels == 32
                && self.norm_num_groups == 32
                && self.patch_size == [2, 2]
                && self.force_upcast
                && self.layers_per_block == 2,
            "unsupported Flux2 VAE configuration"
        );
        Ok(())
    }
}
