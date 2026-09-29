use super::{
    config::{TransformerConfig, read},
    ops::Ops,
    weights::Weights,
};
use anyhow::Result;
use catena_lang::runtime::{MemOwn, MemRef, Value};
use std::path::Path;
const SOURCES: &str = concat!(
    include_str!("../catena/flux_transformer_dual.hex"),
    "\n",
    include_str!("../catena/flux_transformer_single.hex")
);
pub struct Transformer {
    pub ops: Ops,
    weights: Weights,
    config: TransformerConfig,
    time_freq: MemOwn,
    rope: MemOwn,
}
const DUAL_NAMES: [&str; 16] = [
    "attn.to_q.weight",
    "attn.to_k.weight",
    "attn.to_v.weight",
    "attn.to_out.0.weight",
    "attn.norm_q.weight",
    "attn.norm_k.weight",
    "attn.add_q_proj.weight",
    "attn.add_k_proj.weight",
    "attn.add_v_proj.weight",
    "attn.to_add_out.weight",
    "attn.norm_added_q.weight",
    "attn.norm_added_k.weight",
    "ff.linear_in.weight",
    "ff.linear_out.weight",
    "ff_context.linear_in.weight",
    "ff_context.linear_out.weight",
];
impl Transformer {
    pub fn load(snapshot: &Path, height: usize, width: usize) -> Result<Self> {
        Self::load_with_backend(snapshot, height, width, super::ops::Backend::Fast)
    }
    pub fn load_with_backend(
        snapshot: &Path,
        height: usize,
        width: usize,
        backend: super::ops::Backend,
    ) -> Result<Self> {
        let config: TransformerConfig = read(snapshot.join("transformer/config.json"))?;
        config.validate()?;
        let mut weights = Weights::open(&snapshot.join("transformer"))?;
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
            for (i, suffix) in DUAL_NAMES.iter().enumerate() {
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
        let ops = Ops::with_backend(Some(SOURCES), backend)?;
        let names = weights.tensors.keys().cloned().collect::<Vec<_>>();
        weights.prepare(&ops.runtime, &names, false)?;
        let time_freq = ops.runtime.mem_f32(
            &(0..128)
                .map(|i| 10000f32.powf(-(i as f32) / 128.))
                .collect::<Vec<_>>(),
        )?;
        let freq = ops.runtime.mem_f32(
            &(0..16)
                .map(|i| config.rope_theta.powf(-(i as f32) / 16.))
                .collect::<Vec<_>>(),
        )?;
        let rows = height / 16 * (width / 16) + 512;
        let rope = ops.run(
            "flux.rope-table",
            vec![
                freq.as_ref().into(),
                ((rows * 128) as u64).into(),
                512u64.into(),
                ((width / 16) as u64).into(),
            ],
        )?;
        Ok(Self {
            ops,
            weights,
            config,
            time_freq,
            rope,
        })
    }
    fn projection(
        &self,
        x: MemRef<'_>,
        name: &str,
        rows: usize,
        ci: usize,
        co: usize,
    ) -> Result<MemOwn> {
        self.ops
            .linear(x, self.weights.gpu_ref(name)?, rows, ci, co)
    }
    pub fn predict_velocity(
        &self,
        latents: MemRef<'_>,
        conditioning: MemRef<'_>,
        sigma: f32,
        step: usize,
        dump: Option<&Path>,
    ) -> Result<MemOwn> {
        let rows = latents.byte_len() as usize / 2 / 128;
        let all_rows = rows + 512;
        let ops = &self.ops;
        // Diffusers casts scheduler timestep to BF16, divides by 1000, then the model multiplies by 1000.
        let quant =
            |v: f32| crate::flux::pipeline::bf16_to_f32(crate::flux::pipeline::f32_to_bf16(v));
        let time = quant(quant(quant(sigma * 1000.) / 1000.) * 1000.);
        let embedding = ops.run(
            "flux.time-embedding",
            vec![self.time_freq.as_ref().into(), 256u64.into(), time.into()],
        )?;
        let t1 = self.projection(
            embedding.as_ref(),
            "time_guidance_embed.timestep_embedder.linear_1.weight",
            1,
            256,
            3072,
        )?;
        let t1a = ops.run("flux.silu-bf16", vec![t1.as_ref().into(), 3072u64.into()])?;
        let temb = self.projection(
            t1a.as_ref(),
            "time_guidance_embed.timestep_embedder.linear_2.weight",
            1,
            3072,
            3072,
        )?;
        let ta = ops.run("flux.silu-bf16", vec![temb.as_ref().into(), 3072u64.into()])?;
        let mi = self.projection(
            ta.as_ref(),
            "double_stream_modulation_img.linear.weight",
            1,
            3072,
            18432,
        )?;
        let mt = self.projection(
            ta.as_ref(),
            "double_stream_modulation_txt.linear.weight",
            1,
            3072,
            18432,
        )?;
        let ms = self.projection(
            ta.as_ref(),
            "single_stream_modulation.linear.weight",
            1,
            3072,
            9216,
        )?;
        for (name, tensor, cols) in [
            ("temb", &temb, 3072),
            ("mod_image", &mi, 18432),
            ("mod_text", &mt, 18432),
            ("mod_single", &ms, 9216),
        ] {
            super::debug::dump(
                dump,
                &format!("step_{step}_{name}"),
                tensor,
                &[1, cols],
                true,
            )?;
        }
        let mut image = self.projection(latents, "x_embedder.weight", rows, 128, 3072)?;
        let mut text = self.projection(conditioning, "context_embedder.weight", 512, 7680, 3072)?;
        let scale = 1. / 128f32.sqrt();
        for layer in 0..self.config.num_layers {
            let mut args: Vec<Value<'_>> = vec![
                image.as_ref().into(),
                text.as_ref().into(),
                mi.as_ref().into(),
                mt.as_ref().into(),
                self.rope.as_ref().into(),
            ];
            for suffix in DUAL_NAMES {
                args.push(
                    self.weights
                        .gpu_ref(&format!("transformer_blocks.{layer}.{suffix}"))?
                        .into(),
                );
            }
            args.extend([(rows as u64).into(), self.config.eps.into(), scale.into()]);
            let joint = ops.run("flux.dual-block", args)?;
            if layer == 0 || layer == 4 {
                super::debug::dump(
                    dump,
                    &format!("step_{step}_dual_{layer}"),
                    &joint,
                    &[1, all_rows, 3072],
                    true,
                )?;
            }
            image = ops.slice(joint.as_ref(), rows, 3072, 3072, 512 * 3072, true)?;
            text = ops.slice(joint.as_ref(), 512, 3072, 3072, 0, true)?;
            eprintln!("step {step}: dual block {}/5", layer + 1);
        }
        let mut joint = ops.run(
            "flux.rows-bf16",
            vec![
                text.as_ref().into(),
                image.as_ref().into(),
                ((all_rows * 3072) as u64).into(),
                (512 * 3072u64).into(),
            ],
        )?;
        for layer in 0..self.config.num_single_layers {
            let mut args = vec![
                joint.as_ref().into(),
                ms.as_ref().into(),
                self.rope.as_ref().into(),
            ];
            for suffix in [
                "attn.to_qkv_mlp_proj.weight",
                "attn.to_out.weight",
                "attn.norm_q.weight",
                "attn.norm_k.weight",
            ] {
                args.push(
                    self.weights
                        .gpu_ref(&format!("single_transformer_blocks.{layer}.{suffix}"))?
                        .into(),
                );
            }
            args.extend([
                (all_rows as u64).into(),
                self.config.eps.into(),
                scale.into(),
            ]);
            joint = ops.run("flux.single-block", args)?;
            if dump.is_some() {
                super::debug::dump(
                    dump,
                    &format!("step_{step}_single_{layer}"),
                    &joint,
                    &[1, all_rows, 3072],
                    true,
                )?;
            }
            eprintln!("step {step}: single block {}/20", layer + 1);
        }
        image = ops.slice(joint.as_ref(), rows, 3072, 3072, 512 * 3072, true)?;
        let norm = ops.norm(
            image.as_ref(),
            ops.zero.as_ref(),
            ops.zero.as_ref(),
            rows,
            3072,
            1,
            0,
            0,
            self.config.eps,
            true,
        )?;
        let out_mod = self.projection(ta.as_ref(), "norm_out.linear.weight", 1, 3072, 6144)?;
        // AdaLayerNormContinuous projection order is scale, shift (op uses shift, scale).
        let shift = ops.slice(out_mod.as_ref(), 1, 6144, 3072, 3072, true)?;
        let scales = ops.slice(out_mod.as_ref(), 1, 6144, 3072, 0, true)?;
        let swapped = ops.run(
            "flux.concat-bf16",
            vec![
                shift.as_ref().into(),
                scales.as_ref().into(),
                6144u64.into(),
                3072u64.into(),
                3072u64.into(),
            ],
        )?;
        let modulated = ops.run(
            "flux.modulate",
            vec![
                norm.as_ref().into(),
                swapped.as_ref().into(),
                ((rows * 3072) as u64).into(),
                3072u64.into(),
                0u64.into(),
            ],
        )?;
        let velocity = self.projection(modulated.as_ref(), "proj_out.weight", rows, 3072, 128)?;
        super::debug::dump(
            dump,
            &format!("step_{step}_velocity"),
            &velocity,
            &[1, rows, 128],
            true,
        )?;
        Ok(velocity)
    }
}
