//! Complete FLUX programs. Rust here emits Hex before compilation; it never executes layers.
use super::config::{TransformerConfig, VaeConfig};
use anyhow::{Result, ensure};
use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet},
    path::Path,
};
macro_rules! discard_log {
    ($($a:tt)*) => {};
}
#[derive(Clone)]
struct Memory {
    name: String,
    bytes: usize,
    owned: bool,
}
impl Memory {
    fn as_ref(&self) -> Self {
        self.clone()
    }
    fn byte_len(&self) -> u64 {
        self.bytes as u64
    }
}
type MemOwn = Memory;
#[derive(Clone)]
enum Argument {
    Memory(Memory),
    U64(u64),
    F32(f32),
    Scalar(String),
}
impl From<Memory> for Argument {
    fn from(v: Memory) -> Self {
        Self::Memory(v)
    }
}
impl From<u64> for Argument {
    fn from(v: u64) -> Self {
        Self::U64(v)
    }
}
impl From<f32> for Argument {
    fn from(v: f32) -> Self {
        Self::F32(v)
    }
}
fn c(n: u64) -> String {
    format!("(const.u64.0x{n:016X})")
}
fn f(n: f32) -> String {
    format!("(const.u32.0x{:08X} u32.bitcast-f32)", n.to_bits())
}
fn wire(n: &str) -> String {
    format!("[.{n}]")
}
impl Argument {
    fn hex(&self) -> String {
        match self {
            Self::Memory(m) => wire(&m.name),
            Self::U64(n) => c(*n),
            Self::F32(n) => f(*n),
            Self::Scalar(n) => wire(n),
        }
    }
}
struct Operation {
    op: String,
    args: Vec<Argument>,
    out: Memory,
    dtype: &'static str,
}
#[derive(Default)]
struct Builder {
    operations: Vec<Operation>,
    weights: Vec<String>,
    captures: Vec<(String, Memory, Vec<usize>, bool)>,
    extra: String,
}
struct GraphWeights<'a>(&'a RefCell<Builder>);
impl GraphWeights<'_> {
    fn gpu_ref(&self, name: &str) -> Result<Memory> {
        let mut b = self.0.borrow_mut();
        let index = if let Some(i) = b.weights.iter().position(|n| n == name) {
            i
        } else {
            b.weights.push(name.into());
            b.weights.len() - 1
        };
        Ok(Memory {
            name: format!("w{index}"),
            bytes: 0,
            owned: false,
        })
    }
}
struct GraphRuntime<'a>(&'a RefCell<Builder>);
impl GraphRuntime<'_> {
    fn mem_f32(&self, values: &[f32]) -> Result<Memory> {
        let mut b = self.0.borrow_mut();
        let id = b.operations.len();
        let op = format!("flux.constant-{id}");
        let mut body = String::new();
        body.push_str(&format!("{} [value0.]\n", f(values[0])));
        for (i, value) in values.iter().enumerate().skip(1) {
            body.push_str(&format!("({{{} [.i]}} u64.lte [p{i}.])\n ({{[.p{i}] {} [.value{}]}} f32.select [value{i}.])\n", c(i as u64), f(*value),i-1));
        }
        b.extra.push_str(&format!("\n(def program {op}-at : ([n.] {{(u64 val) ([.n] ix val)}}) -> ([n.] (f32 val))\n= ([ignored index.] ([.index] ix.to-u64 [i.])\n{body} [.value{}]))\n(def program {op} : {{}} -> (cap.own mem)\n= ({} u64.name [n.] {{[.n] ({{[.n] (const.u64.0x0000000000000000) (([.n] :.param) name.{op}-at)}} materializec)}} buf.to-mem))\n", values.len()-1,c(values.len() as u64)));
        let out = Memory {
            name: format!("m{id}"),
            bytes: values.len() * 4,
            owned: true,
        };
        b.operations.push(Operation {
            op,
            args: vec![],
            out: out.clone(),
            dtype: "f32",
        });
        Ok(out)
    }
}
struct GraphOps<'a> {
    builder: &'a RefCell<Builder>,
    runtime: GraphRuntime<'a>,
    zero: Memory,
}
fn buffers(op: &str) -> usize {
    match op {
        "linear-bf16" => 2,
        "gated-residual" | "norm-bf16" | "norm-f32" | "conv2d-f32" | "linear-f32"
        | "attention-bf16" | "attention-f32" | "prepare-vae" => 3,
        "apply-rope" | "concat-bf16" | "rows-bf16" | "euler" | "modulate" | "add-f32" => 2,
        "noise-to-bf16" | "image-rgb" | "swiglu" | "transpose-f32" | "upsample" | "silu-bf16"
        | "silu-f32" | "slice-bf16" | "slice-f32" | "copy-bf16" | "copy-f32" | "rope-table"
        | "time-embedding" => 1,
        _ => 0,
    }
}
impl<'a> GraphOps<'a> {
    fn new(b: &'a RefCell<Builder>) -> Result<Self> {
        let runtime = GraphRuntime(b);
        let zero = runtime.mem_f32(&[0.])?;
        Ok(Self {
            builder: b,
            runtime,
            zero,
        })
    }
    fn run(&self, name: &str, args: Vec<Argument>) -> Result<Memory> {
        let base = name.strip_prefix("flux.").unwrap_or(name);
        let modes = args
            .iter()
            .take(buffers(base))
            .map(|a| match a {
                Argument::Memory(m) if m.owned => 'o',
                _ => 'r',
            })
            .collect::<String>();
        let mut op = name.to_string();
        if modes.contains('o') {
            op += &format!("-{modes}");
        }
        if matches!(
            base,
            "dual-block"
                | "single-block"
                | "vae-resblock"
                | "vae-resblock-shortcut"
                | "vae-attention"
        ) {
            op += "-owned";
        }
        let dtype = if base.contains("f32")
            || base.starts_with("vae-")
            || matches!(
                base,
                "prepare-vae" | "upsample" | "rope-table" | "image-rgb"
            ) {
            "f32"
        } else {
            "bf16"
        };
        let mut b = self.builder.borrow_mut();
        let out = Memory {
            name: format!("m{}", b.operations.len()),
            bytes: 0,
            owned: true,
        };
        b.operations.push(Operation {
            op,
            args,
            out: out.clone(),
            dtype,
        });
        Ok(out)
    }
    fn linear(&self, x: Memory, w: Memory, rows: usize, ci: usize, co: usize) -> Result<Memory> {
        self.run(
            "flux.linear-bf16",
            vec![
                x.into(),
                w.into(),
                (rows as u64).into(),
                (ci as u64).into(),
                (co as u64).into(),
            ],
        )
    }
    fn slice(
        &self,
        x: Memory,
        rows: usize,
        stride: usize,
        cols: usize,
        offset: usize,
        bf16: bool,
    ) -> Result<Memory> {
        self.run(
            if bf16 {
                "flux.slice-bf16"
            } else {
                "flux.slice-f32"
            },
            vec![
                x.into(),
                ((rows * cols) as u64).into(),
                (stride as u64).into(),
                (cols as u64).into(),
                (offset as u64).into(),
            ],
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn norm(
        &self,
        x: Memory,
        w: Memory,
        b: Memory,
        rows: usize,
        cols: usize,
        spatial: usize,
        kind: u64,
        affine: u64,
        eps: f32,
        bf16: bool,
    ) -> Result<Memory> {
        self.run(
            if bf16 {
                "flux.norm-bf16"
            } else {
                "flux.norm-f32"
            },
            vec![
                x.into(),
                w.into(),
                b.into(),
                (rows as u64).into(),
                (cols as u64).into(),
                (spatial as u64).into(),
                kind.into(),
                affine.into(),
                eps.into(),
            ],
        )
    }
    fn capture(
        &self,
        dir: Option<&Path>,
        name: &str,
        m: &Memory,
        shape: &[usize],
        bf16: bool,
    ) -> Result<()> {
        if dir.is_some() {
            let copy = self.run(
                if bf16 {
                    "flux.copy-bf16"
                } else {
                    "flux.copy-f32"
                },
                vec![
                    m.as_ref().into(),
                    (shape.iter().product::<usize>() as u64).into(),
                ],
            )?;
            self.builder
                .borrow_mut()
                .captures
                .push((name.into(), copy, shape.into(), bf16));
        }
        Ok(())
    }
}
struct Transformer<'a> {
    ops: GraphOps<'a>,
    weights: GraphWeights<'a>,
    config: &'a TransformerConfig,
    time_freq: Memory,
    rope: Memory,
}
struct VaeDecoder<'a> {
    ops: GraphOps<'a>,
    weights: GraphWeights<'a>,
    config: &'a VaeConfig,
}

pub(crate) const DUAL_NAMES: [&str; 16] = [
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

impl Transformer<'_> {
    fn projection(
        &self,
        x: Memory,
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
        latents: Memory,
        conditioning: Memory,
        time: Argument,
        step: usize,
        dump: Option<&Path>,
    ) -> Result<MemOwn> {
        let rows = latents.byte_len() as usize / 2 / 128;
        let all_rows = rows + 512;
        let ops = &self.ops;
        let embedding = ops.run(
            "flux.time-embedding",
            vec![self.time_freq.as_ref().into(), 256u64.into(), time],
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
            self.ops.capture(
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
            let mut args: Vec<Argument> = vec![
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
                self.ops.capture(
                    dump,
                    &format!("step_{step}_dual_{layer}"),
                    &joint,
                    &[1, all_rows, 3072],
                    true,
                )?;
            }
            image = ops.slice(joint.as_ref(), rows, 3072, 3072, 512 * 3072, true)?;
            text = ops.slice(joint.as_ref(), 512, 3072, 3072, 0, true)?;
            discard_log!("step {step}: dual block {}/5", layer + 1);
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
                self.ops.capture(
                    dump,
                    &format!("step_{step}_single_{layer}"),
                    &joint,
                    &[1, all_rows, 3072],
                    true,
                )?;
            }
            discard_log!("step {step}: single block {}/20", layer + 1);
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
        self.ops.capture(
            dump,
            &format!("step_{step}_velocity"),
            &velocity,
            &[1, rows, 128],
            true,
        )?;
        Ok(velocity)
    }
}
impl VaeDecoder<'_> {
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
        let mut args: Vec<Argument> = vec![x.as_ref().into()];
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
        self.ops
            .capture(dump, "vae_latents", &x, &[1, 32, h, w], false)?;
        x = self.conv(&x, "post_quant_conv", 32, 32, h, w, 1)?;
        self.ops
            .capture(dump, "vae_post_quant", &x, &[1, 32, h, w], false)?;
        x = self.conv(&x, "decoder.conv_in", 32, 512, h, w, 3)?;
        self.ops
            .capture(dump, "vae_conv_in", &x, &[1, 512, h, w], false)?;
        x = self.resblock(&x, "decoder.mid_block.resnets.0", 512, 512, h, w)?;
        self.ops
            .capture(dump, "vae_res0", &x, &[1, 512, h, w], false)?;
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
        self.ops
            .capture(dump, "vae_attention", &x, &[1, 512, h, w], false)?;
        x = self.resblock(&x, "decoder.mid_block.resnets.1", 512, 512, h, w)?;
        self.ops
            .capture(dump, "vae_mid", &x, &[1, 512, h, w], false)?;
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
                self.ops.capture(
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
            self.ops
                .capture(dump, &format!("vae_up_{block}"), &x, &[1, co, h, w], false)?;
            discard_log!("VAE up block {}/4", block + 1);
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
        self.ops.capture(dump, "rgb", &rgb, &[1, 3, h, w], false)?;
        Ok(rgb)
    }
}

pub struct Program {
    pub source: String,
    pub weights: Vec<String>,
    pub captures: Vec<(String, Vec<usize>, bool)>,
}
fn owned_network(source: &str, activation_inputs: usize) -> String {
    let mut owned = BTreeSet::new();
    let mut lines = source.lines();
    let comment = lines.next().unwrap();
    let header = lines.next().unwrap();
    let mut header = header.replacen(" :", "-owned :", 1);
    for _ in 0..activation_inputs {
        header = header.replacen("(cap.ref mem)", "(cap.own mem)", 1);
    }
    let params = lines.next().unwrap();
    let names = params
        .trim()
        .strip_prefix("= ([")
        .unwrap()
        .trim_end_matches("]")
        .trim_end_matches('.')
        .split_whitespace()
        .collect::<Vec<_>>();
    owned.extend(names.iter().take(activation_inputs).map(|n| n.to_string()));
    let mut result = format!("{comment}\n{header}\n{params}\n");
    for line in lines {
        let mut line = line.to_string();
        if let Some(at) = line.find("} flux.") {
            let start = at + 2;
            let end = start + line[start..].find(' ').unwrap();
            let original = &line[start + 5..end];
            let base = original
                .rsplit_once('-')
                .filter(|(_, mode)| mode.chars().all(|c| c == 'o' || c == 'r'))
                .map_or(original, |(base, _)| base);
            let count = buffers(base);
            if count > 0 {
                let mut rest = &line[..at];
                let mut modes = String::new();
                for _ in 0..count {
                    let p = rest.find("[.").unwrap() + 2;
                    rest = &rest[p..];
                    let e = rest.find(']').unwrap();
                    modes.push(if owned.contains(&rest[..e]) { 'o' } else { 'r' });
                    rest = &rest[e + 1..];
                }
                let op = if modes.contains('o') {
                    format!("flux.{base}-{modes}")
                } else {
                    format!("flux.{base}")
                };
                line.replace_range(start..end, &op);
            }
        }
        // Every memory-producing call in these network blocks has one owned output.
        if line.contains("} flux.") || line.contains("} qwen3.free-") {
            if let Some(start) = line.rfind(" [") {
                let name = &line[start + 2..];
                if let Some(end) = name.find('.') {
                    owned.insert(name[..end].into());
                }
            }
        }
        result += &line;
        result.push('\n');
    }
    result
}
fn finish(
    builder: Builder,
    name: &str,
    inputs: &[(String, &str)],
    result: Memory,
    extra: &str,
) -> Program {
    let mut source = builder.extra + extra;
    source.push_str(
        "\n# Safetensors argument order (prepared dtypes follow the component loader):\n",
    );
    for (i, weight) in builder.weights.iter().enumerate() {
        source.push_str(&format!("# w{i}: {weight}\n"));
    }
    let mut outputs = vec![result.name.clone()];
    outputs.extend(builder.captures.iter().map(|(_, m, _, _)| m.name.clone()));
    let mut last = BTreeMap::new();
    let mut dtypes = BTreeMap::new();
    for (i, op) in builder.operations.iter().enumerate() {
        last.insert(op.out.name.clone(), builder.operations.len() - 1);
        dtypes.insert(op.out.name.clone(), op.dtype);
        for a in &op.args {
            if let Argument::Memory(m) = a {
                if m.owned && dtypes.contains_key(&m.name) {
                    last.insert(m.name.clone(), i);
                }
            }
        }
    }
    // Borrowed external inputs remain host-owned. Generated temporaries are released
    // through continuation dependencies after their final consumer has completed.
    let mut body = String::new();
    let mut aliases = BTreeMap::<String, String>::new();
    for (i, op) in builder.operations.iter().enumerate() {
        let args = op
            .args
            .iter()
            .map(|a| match a {
                Argument::Memory(m) => wire(aliases.get(&m.name).unwrap_or(&m.name)),
                _ => a.hex(),
            })
            .collect::<Vec<_>>()
            .join(" ");
        body += &format!(" ({{{args}}} {} [{}.])\n", op.op, op.out.name);
        let mut continuation = op.out.name.clone();
        for (memory, end) in &last {
            if *end != i || outputs.contains(memory) || memory == &op.out.name {
                continue;
            }
            let next = format!("free-{i}-{memory}");
            body += &format!(
                " ({{{} {}}} qwen3.free-{}-before [{next}.])\n",
                wire(aliases.get(memory).unwrap_or(memory)),
                wire(&continuation),
                dtypes[memory]
            );
            continuation = next;
        }
        aliases.insert(op.out.name.clone(), continuation);
    }
    body += &format!(
        " {{{}}}\n",
        outputs
            .iter()
            .map(|n| wire(aliases.get(n).unwrap_or(n)))
            .collect::<Vec<_>>()
            .join(" ")
    );
    let mut args = inputs.to_vec();
    args.extend(
        builder
            .weights
            .iter()
            .enumerate()
            .map(|(i, _)| (format!("w{i}"), "(cap.ref mem)")),
    );
    source += &format!(
        "\n(def program flux.{name} : {{{}}} -> {{{}}}\n= ([{}.]\n{body}))\n",
        args.iter().map(|(_, t)| *t).collect::<Vec<_>>().join(" "),
        vec!["(cap.own mem)"; outputs.len()].join(" "),
        args.iter()
            .map(|(n, _)| n.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    );
    Program {
        source,
        weights: builder.weights,
        captures: builder
            .captures
            .into_iter()
            .map(|(n, _, s, b)| (n, s, b))
            .collect(),
    }
}
fn input(name: &str, bytes: usize) -> Memory {
    Memory {
        name: name.into(),
        bytes,
        owned: false,
    }
}
const TRANSFORMER: &str = concat!(
    include_str!("../catena/flux_transformer_dual.hex"),
    "\n",
    include_str!("../catena/flux_transformer_single.hex")
);
pub fn transformer(
    config: &TransformerConfig,
    height: usize,
    width: usize,
    steps: usize,
    debug: bool,
) -> Result<Program> {
    config.validate()?;
    ensure!(
        height > 0 && width > 0 && height % 16 == 0 && width % 16 == 0 && steps > 0,
        "invalid generation dimensions or steps"
    );
    let rows = height / 16 * (width / 16);
    let b = RefCell::new(Builder::default());
    let ops = GraphOps::new(&b)?;
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
    let rope = ops.run(
        "flux.rope-table",
        vec![
            freq.into(),
            (((rows + 512) * 128) as u64).into(),
            512u64.into(),
            ((width / 16) as u64).into(),
        ],
    )?;
    let model = Transformer {
        ops,
        weights: GraphWeights(&b),
        config,
        time_freq,
        rope,
    };
    // Input preparation is F32. The generated graph establishes the BF16 boundary.
    let mut latents = model.ops.run(
        "flux.noise-to-bf16",
        vec![
            input("noise", rows * 128 * 4).into(),
            ((rows * 128) as u64).into(),
        ],
    )?;
    let sigmas = super::pipeline::schedule(rows, steps);
    let velocity_program = if debug {
        None
    } else {
        let inner = RefCell::new(Builder::default());
        // Table owners belong to denoise. This helper reads them but never frees them.
        let table = |name: &str| Memory {
            name: name.into(),
            bytes: 0,
            owned: true,
        };
        let inner_ops = GraphOps {
            builder: &inner,
            runtime: GraphRuntime(&inner),
            zero: table("zero"),
        };
        let network = Transformer {
            ops: inner_ops,
            weights: GraphWeights(&inner),
            config,
            time_freq: table("time-freq"),
            rope: table("rope"),
        };
        let out = network.predict_velocity(
            Memory {
                name: "latents".into(),
                bytes: rows * 128 * 2,
                owned: true,
            },
            input("conditioning", 0),
            Argument::Scalar("time".into()),
            0,
            None,
        )?;
        drop(network);
        // norm uses a shared zero owned by the outer graph; expose it as an input.
        let mut program = finish(
            inner.into_inner(),
            "velocity",
            &[
                ("latents".into(), "(cap.own mem)"),
                ("conditioning".into(), "(cap.ref mem)"),
                ("time-freq".into(), "(cap.own mem)"),
                ("rope".into(), "(cap.own mem)"),
                ("time".into(), "(f32 val)"),
                ("zero".into(), "(cap.own mem)"),
            ],
            out,
            "",
        );
        // Reserve weight arguments in identical order for every generated call.
        for name in &program.weights {
            model.weights.gpu_ref(name)?;
        }
        program.source.push('\n');
        Some(program)
    };
    for step in 0..steps {
        // byte_len is only a generator shape, never inferred from runtime contents.
        latents.bytes = rows * 128 * 2;
        let quant = |v: f32| super::pipeline::bf16_to_f32(super::pipeline::f32_to_bf16(v));
        let time = quant(quant(quant(sigmas[step] * 1000.) / 1000.) * 1000.);
        let velocity = if let Some(program) = &velocity_program {
            let mut args = vec![
                latents.as_ref().into(),
                input("conditioning", 0).into(),
                model.time_freq.as_ref().into(),
                model.rope.as_ref().into(),
                time.into(),
                model.ops.zero.as_ref().into(),
            ];
            for name in &program.weights {
                args.push(model.weights.gpu_ref(name)?.into());
            }
            model.ops.run("flux.velocity", args)?
        } else {
            model.predict_velocity(
                latents.as_ref(),
                input("conditioning", 512 * 7680 * 2),
                time.into(),
                step,
                debug.then_some(Path::new("debug")),
            )?
        };
        latents = model.ops.run(
            "flux.euler",
            vec![
                latents.into(),
                velocity.into(),
                ((rows * 128) as u64).into(),
                (sigmas[step + 1] - sigmas[step]).into(),
            ],
        )?;
        model.ops.capture(
            debug.then_some(Path::new("debug")),
            &format!("step_{step}_latents"),
            &latents,
            &[1, rows, 128],
            true,
        )?;
    }
    drop(model);
    let mut extra = TRANSFORMER.to_string();
    if let Some(program) = velocity_program {
        extra += &program.source;
    }
    for (src, n) in [
        (include_str!("../catena/flux_transformer_dual.hex"), 5),
        (include_str!("../catena/flux_transformer_single.hex"), 3),
    ] {
        extra += &owned_network(src, n);
    }
    Ok(finish(
        b.into_inner(),
        "denoise",
        &[
            ("noise".into(), "(cap.ref mem)"),
            ("conditioning".into(), "(cap.ref mem)"),
        ],
        latents,
        &extra,
    ))
}
pub fn vae(config: &VaeConfig, height: usize, width: usize, debug: bool) -> Result<Program> {
    config.validate()?;
    ensure!(
        height > 0 && width > 0 && height % 16 == 0 && width % 16 == 0,
        "invalid image dimensions"
    );
    let b = RefCell::new(Builder::default());
    let model = VaeDecoder {
        ops: GraphOps {
            builder: &b,
            runtime: GraphRuntime(&b),
            zero: input("unused-zero", 0),
        },
        weights: GraphWeights(&b),
        config,
    };
    let rgb = model.decode(
        &input("latents", 0),
        height,
        width,
        debug.then_some(Path::new("debug")),
    )?;
    let image = model.ops.run(
        "flux.image-rgb",
        vec![
            rgb.into(),
            ((height * width * 3) as u64).into(),
            ((height * width) as u64).into(),
        ],
    )?;
    drop(model);
    let mut extra = String::new();
    for src in [
        include_str!("../catena/flux_vae_resblock.hex"),
        include_str!("../catena/flux_vae_resblock_shortcut.hex"),
        include_str!("../catena/flux_vae_attention.hex"),
    ] {
        extra += src;
        extra += &owned_network(src, 1);
    }
    Ok(finish(
        b.into_inner(),
        "decode-image",
        &[("latents".into(), "(cap.ref mem)")],
        image,
        &extra,
    ))
}

pub(crate) fn execute<'a>(
    artifact: &catena_lang::runtime::Artifact,
    name: &str,
    args: Vec<catena_lang::runtime::Value<'a>>,
    program: &Program,
    dump: Option<&Path>,
) -> Result<catena_lang::runtime::MemOwn> {
    use catena_lang::runtime::Value;
    let outputs = artifact.exec_values(name, args)?;
    ensure!(
        outputs.len() == 1 + program.captures.len(),
        "wrong generated stage output count"
    );
    let mut outputs = outputs.into_iter();
    let result = match outputs.next().unwrap() {
        Value::MemOwn(m) => m,
        _ => anyhow::bail!("expected stage memory"),
    };
    for (value, (name, shape, bf16)) in outputs.zip(&program.captures) {
        let Value::MemOwn(memory) = value else {
            anyhow::bail!("expected capture memory")
        };
        super::debug::dump(dump, name, &memory, shape, *bf16)?;
    }
    Ok(result)
}

/// Assemble a deployable stage asset. Nodes supply the compiler stdlib and bind
/// the safetensors weights listed in the generated source comments.
pub fn bundle(source: &str, backend: super::ops::Backend) -> String {
    let helpers = match backend {
        super::ops::Backend::Fast => super::ops::SOURCE,
        super::ops::Backend::Legacy => include_str!("../catena/flux_legacy_helpers.hex"),
    };
    [
        crate::catena::BF16_HELPER_SOURCE,
        include_str!("../catena/inference_helpers.hex"),
        helpers,
        source,
    ]
    .join("\n")
}
