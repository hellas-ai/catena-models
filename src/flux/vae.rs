use super::{
    config::{VaeConfig, read},
    ops::Ops,
};
use crate::weights::Weights;
use anyhow::Result;
use catena_lang::runtime::MemOwn;
use std::path::Path;
pub struct VaeDecoder {
    pub ops: Ops,
    weights: Weights,
    program: super::graph::Program,
}
impl VaeDecoder {
    pub fn load(snapshot: &Path, height: usize, width: usize) -> Result<Self> {
        Self::load_with_debug(snapshot, height, width, false)
    }
    pub fn load_with_debug(
        snapshot: &Path,
        height: usize,
        width: usize,
        debug: bool,
    ) -> Result<Self> {
        let config: VaeConfig = read(snapshot.join("vae/config.json"))?;
        config.validate()?;
        let mut weights = Weights::open(&snapshot.join("vae/diffusion_pytorch_model.safetensors"))?;
        weights.validate("bn.running_mean", &[128])?;
        weights.validate("bn.running_var", &[128])?;
        for (name, ci, co, ks) in [
            ("post_quant_conv", 32, 32, 1),
            ("decoder.conv_in", 32, 512, 3),
            ("decoder.conv_out", 128, 3, 3),
        ] {
            validate_conv(&weights, name, ci, co, ks)?;
        }
        for layer in 0..2 {
            validate_res(
                &weights,
                &format!("decoder.mid_block.resnets.{layer}"),
                512,
                512,
            )?;
        }
        let mut ci = 512;
        for (block, co) in [512, 512, 256, 128].into_iter().enumerate() {
            for layer in 0..3 {
                validate_res(
                    &weights,
                    &format!("decoder.up_blocks.{block}.resnets.{layer}"),
                    ci,
                    co,
                )?;
                ci = co;
            }
            if block < 3 {
                validate_conv(
                    &weights,
                    &format!("decoder.up_blocks.{block}.upsamplers.0.conv"),
                    co,
                    co,
                    3,
                )?;
            }
        }
        weights.validate("decoder.conv_norm_out.weight", &[128])?;
        weights.validate("decoder.conv_norm_out.bias", &[128])?;
        for suffix in ["group_norm.weight", "group_norm.bias"] {
            weights.validate(&format!("decoder.mid_block.attentions.0.{suffix}"), &[512])?;
        }
        for suffix in ["to_q", "to_k", "to_v", "to_out.0"] {
            weights.validate(
                &format!("decoder.mid_block.attentions.0.{suffix}.weight"),
                &[512, 512],
            )?;
            weights.validate(
                &format!("decoder.mid_block.attentions.0.{suffix}.bias"),
                &[512],
            )?;
        }
        let program = super::graph::vae(&config, height, width, debug)?;
        let ops = Ops::new(Some(&program.source))?;
        weights.prepare(&ops.runtime, &program.weights, true)?;
        Ok(Self {
            ops,
            weights,
            program,
        })
    }
    pub fn decode(&self, latents: &MemOwn, dump: Option<&Path>) -> Result<MemOwn> {
        let mut args = vec![latents.as_ref().into()];
        for name in &self.program.weights {
            args.push(self.weights.gpu_ref(name)?.into());
        }
        super::graph::execute(
            &self.ops.artifact,
            "flux.decode-image",
            args,
            &self.program,
            dump,
        )
    }
}

fn validate_conv(w: &Weights, name: &str, ci: usize, co: usize, ks: usize) -> Result<()> {
    w.validate(&format!("{name}.weight"), &[co, ci, ks, ks])?;
    w.validate(&format!("{name}.bias"), &[co])
}
fn validate_res(w: &Weights, name: &str, ci: usize, co: usize) -> Result<()> {
    for (n, c) in [("norm1", ci), ("norm2", co)] {
        for suffix in ["weight", "bias"] {
            w.validate(&format!("{name}.{n}.{suffix}"), &[c])?;
        }
    }
    validate_conv(w, &format!("{name}.conv1"), ci, co, 3)?;
    validate_conv(w, &format!("{name}.conv2"), co, co, 3)?;
    if ci != co {
        validate_conv(w, &format!("{name}.conv_shortcut"), ci, co, 1)?;
    }
    Ok(())
}
