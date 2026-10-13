use super::{
    config::{TransformerConfig, read},
    ops::Ops,
};
use crate::weights::Weights;
use anyhow::Result;
use catena_lang::runtime::{MemOwn, MemRef};
use std::path::Path;
pub struct Transformer {
    pub ops: Ops,
    weights: Weights,
    program: super::graph::Program,
}
impl Transformer {
    pub fn load(snapshot: &Path, height: usize, width: usize, steps: usize) -> Result<Self> {
        Self::load_with_backend(
            snapshot,
            height,
            width,
            steps,
            false,
            super::ops::Backend::Fast,
        )
    }
    pub fn load_with_backend(
        snapshot: &Path,
        height: usize,
        width: usize,
        steps: usize,
        debug: bool,
        backend: super::ops::Backend,
    ) -> Result<Self> {
        let config: TransformerConfig = read(snapshot.join("transformer/config.json"))?;
        config.validate()?;
        let mut weights =
            Weights::open(&snapshot.join("transformer/diffusion_pytorch_model.safetensors"))?;
        for (name, shape) in [
            ("x_embedder.weight", vec![3072, 128]),
            ("context_embedder.weight", vec![3072, 7680]),
            (
                "time_guidance_embed.timestep_embedder.linear_1.weight",
                vec![3072, 256],
            ),
            (
                "time_guidance_embed.timestep_embedder.linear_2.weight",
                vec![3072, 3072],
            ),
            (
                "double_stream_modulation_img.linear.weight",
                vec![18432, 3072],
            ),
            (
                "double_stream_modulation_txt.linear.weight",
                vec![18432, 3072],
            ),
            ("single_stream_modulation.linear.weight", vec![9216, 3072]),
            ("norm_out.linear.weight", vec![6144, 3072]),
            ("proj_out.weight", vec![128, 3072]),
        ] {
            weights.validate(name, &shape)?;
        }
        for layer in 0..config.num_layers {
            for (i, suffix) in super::graph::DUAL_NAMES.iter().enumerate() {
                let shape = match i {
                    4 | 5 | 10 | 11 => vec![128],
                    12 | 14 => vec![18432, 3072],
                    13 | 15 => vec![3072, 9216],
                    _ => vec![3072, 3072],
                };
                weights.validate(&format!("transformer_blocks.{layer}.{suffix}"), &shape)?;
            }
        }
        for layer in 0..config.num_single_layers {
            for (suffix, shape) in [
                ("attn.to_qkv_mlp_proj.weight", vec![27648, 3072]),
                ("attn.to_out.weight", vec![3072, 12288]),
                ("attn.norm_q.weight", vec![128]),
                ("attn.norm_k.weight", vec![128]),
            ] {
                weights.validate(
                    &format!("single_transformer_blocks.{layer}.{suffix}"),
                    &shape,
                )?;
            }
        }
        let program = super::graph::transformer(&config, height, width, steps, debug)?;
        let ops = Ops::with_backend(Some(&program.source), backend)?;
        weights.prepare(&ops.runtime, &program.weights, false)?;
        Ok(Self {
            ops,
            weights,
            program,
        })
    }
    pub fn denoise(
        &self,
        noise: MemRef<'_>,
        conditioning: MemRef<'_>,
        dump: Option<&Path>,
    ) -> Result<MemOwn> {
        let mut args = vec![noise.into(), conditioning.into()];
        for name in &self.program.weights {
            args.push(self.weights.gpu_ref(name)?.into());
        }
        super::graph::execute(
            &self.ops.artifact,
            "flux.denoise",
            args,
            &self.program,
            dump,
        )
    }
}
