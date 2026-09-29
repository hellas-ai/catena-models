use super::{
    config::{VaeConfig, read},
    ops::Ops,
    weights::Weights,
};
use anyhow::Result;
use catena_lang::runtime::{MemOwn, Value};
use std::path::Path;
const SOURCES: &str = concat!(
    include_str!("../catena/flux_vae_resblock.hex"),
    "\n",
    include_str!("../catena/flux_vae_resblock_shortcut.hex"),
    "\n",
    include_str!("../catena/flux_vae_attention.hex")
);
pub struct VaeDecoder {
    pub ops: Ops,
    weights: Weights,
    config: VaeConfig,
}
const RES_NAMES: [&str; 8] = [
    "norm1.weight",
    "norm1.bias",
    "conv1.weight",
    "conv1.bias",
    "norm2.weight",
    "norm2.bias",
    "conv2.weight",
    "conv2.bias",
];
impl VaeDecoder {
    pub fn load(snapshot: &Path) -> Result<Self> {
        Self::load_with_backend(snapshot, super::ops::Backend::Fast)
    }
    pub fn load_with_backend(snapshot: &Path, backend: super::ops::Backend) -> Result<Self> {
        let config: VaeConfig = read(snapshot.join("vae/config.json"))?;
        config.validate()?;
        let mut weights = Weights::open(&snapshot.join("vae"))?;
        let names: Vec<_> = weights
            .tensors
            .keys()
            .filter(|n| {
                n.starts_with("decoder.")
                    || n.starts_with("post_quant_conv.")
                    || n.starts_with("bn.running_")
            })
            .cloned()
            .collect();
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
        let ops = Ops::with_backend(Some(SOURCES), backend)?;
        weights.prepare(&ops.runtime, &names, true)?;
        Ok(Self {
            ops,
            weights,
            config,
        })
    }
    fn conv(
        &self,
        x: &MemOwn,
        name: &str,
        ci: usize,
        co: usize,
        h: usize,
        w: usize,
        ks: usize,
    ) -> Result<MemOwn> {
        self.ops.run(
            "flux.conv2d-f32",
            vec![
                x.as_ref().into(),
                self.weights.gpu_ref(&format!("{name}.weight"))?.into(),
                self.weights.gpu_ref(&format!("{name}.bias"))?.into(),
                (ci as u64).into(),
                (co as u64).into(),
                (h as u64).into(),
                (w as u64).into(),
                (ks as u64).into(),
                ((ks / 2) as u64).into(),
            ],
        )
    }
    fn resblock(
        &self,
        x: &MemOwn,
        name: &str,
        ci: usize,
        co: usize,
        h: usize,
        w: usize,
    ) -> Result<MemOwn> {
        let mut args: Vec<Value<'_>> = vec![x.as_ref().into()];
        for suffix in RES_NAMES {
            args.push(self.weights.gpu_ref(&format!("{name}.{suffix}"))?.into());
        }
        if ci != co {
            for suffix in ["conv_shortcut.weight", "conv_shortcut.bias"] {
                args.push(self.weights.gpu_ref(&format!("{name}.{suffix}"))?.into());
            }
        }
        args.extend([
            (ci as u64).into(),
            (co as u64).into(),
            (h as u64).into(),
            (w as u64).into(),
            1e-6f32.into(),
        ]);
        self.ops.run(
            if ci == co {
                "flux.vae-resblock"
            } else {
                "flux.vae-resblock-shortcut"
            },
            args,
        )
    }
    pub fn decode(
        &self,
        latents: &MemOwn,
        height: usize,
        width: usize,
        dump: Option<&Path>,
    ) -> Result<MemOwn> {
        let mut h = height / 8;
        let mut w = width / 8;
        let mut x = self.ops.run(
            "flux.prepare-vae",
            vec![
                latents.as_ref().into(),
                self.weights.gpu_ref("bn.running_mean")?.into(),
                self.weights.gpu_ref("bn.running_var")?.into(),
                ((32 * h * w) as u64).into(),
                ((height / 16) as u64).into(),
                ((width / 16) as u64).into(),
                self.config.batch_norm_eps.into(),
            ],
        )?;
        super::debug::dump(dump, "vae_latents", &x, &[1, 32, h, w], false)?;
        x = self.conv(&x, "post_quant_conv", 32, 32, h, w, 1)?;
        super::debug::dump(dump, "vae_post_quant", &x, &[1, 32, h, w], false)?;
        x = self.conv(&x, "decoder.conv_in", 32, 512, h, w, 3)?;
        super::debug::dump(dump, "vae_conv_in", &x, &[1, 512, h, w], false)?;
        x = self.resblock(&x, "decoder.mid_block.resnets.0", 512, 512, h, w)?;
        super::debug::dump(dump, "vae_res0", &x, &[1, 512, h, w], false)?;
        let base = "decoder.mid_block.attentions.0";
        let mut args = vec![x.as_ref().into()];
        for suffix in [
            "group_norm.weight",
            "group_norm.bias",
            "to_q.weight",
            "to_q.bias",
            "to_k.weight",
            "to_k.bias",
            "to_v.weight",
            "to_v.bias",
            "to_out.0.weight",
            "to_out.0.bias",
        ] {
            args.push(self.weights.gpu_ref(&format!("{base}.{suffix}"))?.into());
        }
        args.extend([
            (h as u64).into(),
            (w as u64).into(),
            1e-6f32.into(),
            (1. / 512f32.sqrt()).into(),
        ]);
        x = self.ops.run("flux.vae-attention", args)?;
        super::debug::dump(dump, "vae_attention", &x, &[1, 512, h, w], false)?;
        x = self.resblock(&x, "decoder.mid_block.resnets.1", 512, 512, h, w)?;
        super::debug::dump(dump, "vae_mid", &x, &[1, 512, h, w], false)?;
        let mut ci = 512;
        for (block, co) in [512, 512, 256, 128].into_iter().enumerate() {
            for layer in 0..3 {
                x = self.resblock(
                    &x,
                    &format!("decoder.up_blocks.{block}.resnets.{layer}"),
                    ci,
                    co,
                    h,
                    w,
                )?;
                super::debug::dump(
                    dump,
                    &format!("vae_up_{block}_res_{layer}"),
                    &x,
                    &[1, co, h, w],
                    false,
                )?;
                ci = co;
            }
            if block < 3 {
                x = self.ops.run(
                    "flux.upsample",
                    vec![
                        x.as_ref().into(),
                        ((co * h * w * 4) as u64).into(),
                        (h as u64).into(),
                        (w as u64).into(),
                    ],
                )?;
                h *= 2;
                w *= 2;
                x = self.conv(
                    &x,
                    &format!("decoder.up_blocks.{block}.upsamplers.0.conv"),
                    co,
                    co,
                    h,
                    w,
                    3,
                )?;
            }
            super::debug::dump(dump, &format!("vae_up_{block}"), &x, &[1, co, h, w], false)?;
            eprintln!("VAE up block {}/4", block + 1);
        }
        let norm = self.ops.norm(
            x.as_ref(),
            self.weights.gpu_ref("decoder.conv_norm_out.weight")?,
            self.weights.gpu_ref("decoder.conv_norm_out.bias")?,
            32,
            4 * h * w,
            h * w,
            2,
            2,
            1e-6,
            false,
        )?;
        let act = self.ops.run(
            "flux.silu-f32",
            vec![norm.as_ref().into(), ((128 * h * w) as u64).into()],
        )?;
        let rgb = self.conv(&act, "decoder.conv_out", 128, 3, h, w, 3)?;
        super::debug::dump(dump, "rgb", &rgb, &[1, 3, h, w], false)?;
        Ok(rgb)
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
