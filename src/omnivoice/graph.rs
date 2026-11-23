use crate::{catena, qwen, spec::ModelConfig};
use catena_lang::runtime::GpuDialect;
use qwen::{c, call, header};
use std::{collections::BTreeMap, fmt::Write};

pub(super) const OWN: &str = "((cap.own mem) val)";
pub(super) const REF: &str = "((cap.ref mem) val)";
const U64: &str = "(u64 val)";
const U32: &str = "(u32 val)";
const PREFIX: &str = "stdlib.omnivoice.";
const FREE_BF16: &str = "stdlib.models.resources.free-bf16-before";
const FREE: &str = "stdlib.models.resources.free-f32-before";
const CONV: &str = "acoustic_decoder";
pub(super) const STRIDES: [usize; 5] = [8, 5, 4, 2, 3];
pub(super) const DILATIONS: [usize; 3] = [1, 3, 9];

#[derive(Clone, Copy)]
pub(super) enum Packing {
    Bf16,
    F32,
    TransposedConv,
}

pub(super) struct Tensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub packing: Packing,
}

impl Tensor {
    fn new(name: String, shape: Vec<usize>, packing: Packing) -> Self {
        Self {
            name,
            shape,
            packing,
        }
    }
}

fn f(value: f32) -> String {
    format!("(const.u32.0x{:08X} u32.bitcast-f32)", value.to_bits())
}

fn mul(s: &mut String, a: &str, b: usize, out: &str) {
    call(s, &[a, &c(b)], "stdlib.models.memory.u64.mul", out);
}

fn add(s: &mut String, a: &str, b: &str, out: &str) {
    call(s, &[a, b], "stdlib.models.memory.u64.add", out);
}

fn kernel(s: &mut String, op: &str, args: &[&str], count: &str, out: &str) {
    let mut args = args.to_vec();
    args.extend([count, count]);
    call(s, &args, &format!("{PREFIX}{op}.run"), out);
}

fn finish(s: &mut String, out: &str) {
    writeln!(s, "  [.{out}]\n ))").unwrap();
}

fn free(s: &mut String, old: &[&str], new: &str, output: &str, op: &str) {
    let mut current = new.to_owned();
    for (i, memory) in old.iter().enumerate() {
        let next = if i + 1 == old.len() {
            output.to_owned()
        } else {
            format!("{output}-free{i}")
        };
        call(s, &[memory, &current], op, &next);
        current = next;
    }
}

fn fold(s: &mut String, op: &str, captures: &[&str], count: &str, initial: &str, out: &str) {
    call(
        s,
        captures,
        &format!("partial.{PREFIX}{op}.{}", captures.len()),
        "callback",
    );
    writeln!(
        s,
        "  ({{{count} (unit.intro meta.master.end32)}} smolcat.value.named_at [end.])\n  ({{[.end] [.{initial}] [.callback] core.truth}} core.fold.bounded [{out} proof.])\n  ([.proof] unsafe.discard unit.elim)"
    )
    .unwrap();
}

fn fold_arg(state: &str) -> String {
    format!(
        "({{({{master.index32 u32}} :) ({{{state} ({{(({{master.index32 master.end32}} prop.Less) |-) (prop.Truth |-)}} *)}} *)}} *)"
    )
}

fn fold_output(state: &str) -> String {
    format!("({{{state} (prop.Truth |-)}} *)")
}

fn unfold(s: &mut String) {
    s.push_str("  ([.pair] *.elim [index rest.])\n  ([.rest] *.elim [state proofs.])\n  ([.proofs] *.elim [less truth.])\n  ([.less] unsafe.discard unit.elim)\n  ([.index] :.forget [i32.])\n");
}

fn ref_inputs<'a>(inputs: &mut Vec<(&'a str, &'a str)>, names: &'a [String]) {
    inputs.extend(names.iter().map(|name| (name.as_str(), REF)));
}

fn backbone(s: &mut String, config: &ModelConfig, weights: &str) {
    let group_names = [
        "input-norm",
        "wq",
        "wk",
        "wv",
        "wo",
        "q-norm",
        "k-norm",
        "post-norm",
        "wg",
        "wu",
        "wd",
    ];
    let group = qwen::nested_product(&vec![REF.to_owned(); group_names.len()]);
    qwen::emit_layer_selector(s, config.num_layers, &group, weights);
    let mut layer_captures = vec![
        ("weights", weights),
        ("rows", U64),
        ("rope", OWN),
        ("identity", OWN),
    ];
    let pair = fold_arg(OWN);
    layer_captures.push(("pair", &pair));
    header(
        s,
        "stdlib.omnivoice.layer",
        &layer_captures,
        &fold_output(OWN),
    );
    unfold(s);
    call(
        s,
        &["weights", "i32"],
        "stdlib.qwen3.layer-weights",
        "selected",
    );
    qwen::unpack(s, "selected", &group_names.map(String::from));
    let hidden = config.hidden_size;
    let query = config.query_size();
    let kv = config.key_value_size();
    let intermediate = config.intermediate_size;
    let dim = config.head_dim;
    let norm = |s: &mut String, x: &str, w: &str, width, rows: &str, out: &str| {
        call(
            s,
            &[x, w, &c(width), &f(width as f32), rows],
            "qwen3.normalize",
            out,
        )
    };
    let project = |s: &mut String, x, w, ci, co, out| {
        call(s, &[x, w, &c(ci), &c(co)], "stdlib.omnivoice.project", out)
    };
    norm(s, "state", "input-norm", hidden, "rows", "normalized");
    project(s, "normalized", "wq", hidden, query, "q");
    project(s, "normalized", "wk", hidden, kv, "k");
    project(s, "normalized", "wv", hidden, kv, "v-raw");
    free(s, &["normalized"], "v-raw", "v", FREE_BF16);
    for (name, width) in [("q", query), ("k", kv)] {
        let heads = width / dim;
        let norm_rows = format!("{name}-norm-rows");
        let elements = format!("{name}-elements");
        let normalized = format!("{name}-normalized");
        let rotated = format!("{name}-rotated");
        mul(s, "rows", heads, &norm_rows);
        norm(
            s,
            name,
            &format!("{name}-norm"),
            dim,
            &norm_rows,
            &normalized,
        );
        mul(s, "rows", width, &elements);
        kernel(
            s,
            "rotate",
            &[&normalized, "rope", &c(width)],
            &elements,
            &rotated,
        );
        free(
            s,
            &[name, &normalized],
            &rotated,
            &format!("{name}-ready"),
            FREE_BF16,
        );
    }
    let zero = c(0);
    call(
        s,
        &[
            "q-ready",
            "k-ready",
            "v",
            "rows",
            "rows",
            &c(config.num_attention_heads),
            &c(config.num_key_value_heads),
            &f((dim as f32).sqrt().recip()),
            &zero,
            &zero,
            "identity",
            &zero,
            &zero,
            &zero,
        ],
        "stdlib.lfm3.online.text.d128.run",
        "attention",
    );
    free(
        s,
        &["q-ready", "k-ready", "v"],
        "attention",
        "context",
        FREE_BF16,
    );
    project(s, "context", "wo", query, hidden, "projected");
    call(s, &["state", "projected"], "qwen3.add", "residual-raw");
    free(
        s,
        &["state", "context", "projected"],
        "residual-raw",
        "residual",
        FREE_BF16,
    );
    norm(s, "residual", "post-norm", hidden, "rows", "ffn");
    project(s, "ffn", "wg", hidden, intermediate, "gate");
    project(s, "ffn", "wu", hidden, intermediate, "up");
    call(s, &["gate", "up"], "qwen3.swiglu", "activated");
    free(s, &["ffn", "gate", "up"], "activated", "gated", FREE_BF16);
    project(s, "gated", "wd", intermediate, hidden, "down");
    call(s, &["residual", "down"], "qwen3.add", "result");
    free(
        s,
        &["gated", "down", "residual"],
        "result",
        "ready",
        FREE_BF16,
    );
    s.push_str("  ({[.ready] [.truth]} *.intro)\n ))\n");

    header(
        s,
        "stdlib.omnivoice.logits",
        &[
            ("text", REF),
            ("text-embedding", REF),
            ("audio-embedding", REF),
            ("audio-heads", REF),
            ("output-norm", REF),
            ("weights", weights),
            ("codes", OWN),
            ("rope", OWN),
            ("identity", OWN),
            ("prefix", U64),
            ("frames", U64),
        ],
        OWN,
    );
    add(s, "prefix", "frames", "rows");
    mul(s, "rows", hidden, "hidden-count");
    kernel(
        s,
        "embed",
        &[
            "codes",
            "text",
            "text-embedding",
            "audio-embedding",
            "prefix",
        ],
        "hidden-count",
        "input",
    );
    fold(
        s,
        "layer",
        &["weights", "rows", "rope", "identity"],
        "(const.u32.0x0000001C)",
        "input",
        "hidden",
    );
    mul(s, "prefix", hidden, "offset");
    mul(s, "frames", hidden, "count");
    call(s, &["count"], "stdlib.models.memory.alloc-bf16", "target");
    call(
        s,
        &["hidden", "target", "offset", &zero, "count"],
        "stdlib.qwen35.prefill.copy-range",
        "selected",
    );
    free(s, &["hidden"], "selected", "target-ready", FREE_BF16);
    norm(
        s,
        "target-ready",
        "output-norm",
        config.hidden_size,
        "frames",
        "normalized",
    );
    project(
        s,
        "normalized",
        "audio-heads",
        config.hidden_size,
        8200,
        "logits",
    );
    free(
        s,
        &["target-ready", "normalized"],
        "logits",
        "ready",
        FREE_BF16,
    );
    finish(s, "ready");
}

fn sampler(s: &mut String, weights: &str) {
    let state = "({((cap.own mem) val) (u64 val)} *)";
    let captures = [
        ("text", REF),
        ("text-embedding", REF),
        ("audio-embedding", REF),
        ("audio-heads", REF),
        ("output-norm", REF),
        ("weights", weights),
        ("rope", OWN),
        ("identity", OWN),
        ("prefix", U64),
        ("frames", U64),
        ("slots", U64),
        ("seed", U64),
        ("num-step", U32),
    ];
    let mut inputs = captures.to_vec();
    let pair = fold_arg(state);
    inputs.push(("pair", &pair));
    header(s, "stdlib.omnivoice.step", &inputs, &fold_output(state));
    unfold(s);
    s.push_str("  ([.state] *.elim [codes remaining.])\n  ([.i32] stdlib.models.memory.u32.to-u64 [step.])\n");
    call(s, &["num-step"], "stdlib.models.memory.u32.to-u64", "steps");
    call(
        s,
        &["step", "steps", "slots", "remaining"],
        "stdlib.omnivoice.schedule",
        "take",
    );
    let mut logits = [
        "text",
        "text-embedding",
        "audio-embedding",
        "audio-heads",
        "output-norm",
        "weights",
        "codes",
        "rope",
        "identity",
        "prefix",
        "frames",
    ];
    call(s, &logits, "stdlib.omnivoice.logits", "conditional");
    let zero = c(0);
    logits[9] = &zero;
    call(s, &logits, "stdlib.omnivoice.logits", "unconditional");
    mul(s, "slots", 2, "pred-count");
    call(
        s,
        &[
            "conditional",
            "unconditional",
            "codes",
            "seed",
            "step",
            "slots",
            "pred-count",
        ],
        "stdlib.omnivoice.predict.run",
        "predictions",
    );
    free(
        s,
        &["conditional", "unconditional"],
        "predictions",
        "scores",
        FREE_BF16,
    );
    kernel(
        s,
        "select-codes",
        &["codes", "scores", "slots", "take"],
        "slots",
        "updated",
    );
    free(s, &["codes", "scores"], "updated", "ready", FREE);
    call(
        s,
        &["remaining", "take"],
        "stdlib.models.memory.u64.sub",
        "left",
    );
    s.push_str("  ({[.ready] [.left]} *.intro [next.])\n  ({[.next] [.truth]} *.intro)\n ))\n");

    let mut inputs = captures.to_vec();
    inputs.retain(|(name, _)| !matches!(*name, "rope" | "identity" | "slots"));
    header(s, "stdlib.omnivoice.generate", &inputs, OWN);
    mul(s, "frames", 8, "slots");
    kernel(s, "fill-codes", &[], "slots", "codes");
    add(s, "prefix", "frames", "rows");
    mul(s, "rows", 128, "rope-count");
    kernel(s, "rope-table", &[], "rope-count", "rope");
    mul(s, "rows", 16, "gate-count");
    kernel(s, "identity-gate", &[], "gate-count", "identity");
    call(s, &["codes", "slots"], "*.intro", "initial");
    fold(
        s,
        "step",
        &captures.map(|(name, _)| name),
        "[.num-step]",
        "initial",
        "result",
    );
    s.push_str("  ([.result] *.elim [final-codes remaining.])\n");
    free(s, &["rope"], "final-codes", "result-rope", FREE);
    free(s, &["identity"], "result-rope", "ready", FREE_BF16);
    finish(s, "ready");
}

struct Codec {
    body: String,
    weights: Vec<Tensor>,
}

impl Codec {
    fn bind(&mut self, name: String, shape: Vec<usize>, packing: Packing) -> String {
        let arg = format!("codec-{}", self.weights.len());
        self.weights.push(Tensor::new(name, shape, packing));
        arg
    }

    #[allow(clippy::too_many_arguments)]
    fn conv(
        &mut self,
        name: &str,
        x: &str,
        frames: &str,
        rows: &str,
        ci: usize,
        co: usize,
        taps: usize,
        stride: usize,
        dilation: usize,
        transpose: bool,
        out: &str,
    ) {
        let (shape, packing) = if transpose {
            (vec![ci, co, taps], Packing::TransposedConv)
        } else if name == "fc2" {
            (vec![co, ci], Packing::Bf16)
        } else {
            (vec![co, ci, taps], Packing::Bf16)
        };
        let weight = self.bind(format!("{name}.weight"), shape, packing);
        let bias = self.bind(format!("{name}.bias"), vec![co], Packing::Bf16);
        let pad = if transpose {
            stride.div_ceil(2)
        } else {
            (taps / 2) * dilation
        };
        // Keep the materialized BF16 input patches within 16 MiB.
        let tile_rows = (16 * 1024 * 1024 / (2 * ci * taps)).min(8192) / 256 * 256;
        call(
            &mut self.body,
            &[
                x,
                &weight,
                &bias,
                frames,
                rows,
                &c(ci),
                &c(co),
                &c(taps),
                &c(dilation),
                &c(stride),
                &c(pad),
                &c(usize::from(transpose)),
                &c(1),
                &c(tile_rows),
            ],
            "tts.conv-tiled",
            out,
        );
    }

    fn snake(&mut self, name: &str, x: &str, frames: &str, channels: usize, out: &str) {
        let alpha = self.bind(format!("{name}.alpha"), vec![1, channels, 1], Packing::F32);
        let count = format!("{out}-count");
        mul(&mut self.body, frames, channels, &count);
        kernel(
            &mut self.body,
            "snake",
            &[x, &alpha, &c(channels)],
            &count,
            out,
        );
    }

    fn new() -> Self {
        let mut codec = Self {
            body: String::new(),
            weights: Vec::new(),
        };
        mul(&mut codec.body, "frames", 1024, "latent-count");
        kernel(
            &mut codec.body,
            "quantizer",
            &["codes", "embeddings", "projections", "biases"],
            "latent-count",
            "quantized",
        );
        free(&mut codec.body, &["codes"], "quantized", "latent", FREE);
        codec.conv(
            "fc2", "latent", "frames", "frames", 1024, 256, 1, 1, 1, false, "fc2",
        );
        codec.conv(
            &format!("{CONV}.conv1"),
            "fc2",
            "frames",
            "frames",
            256,
            1024,
            7,
            1,
            1,
            false,
            "decoder-input",
        );
        free(
            &mut codec.body,
            &["latent", "fc2"],
            "decoder-input",
            "initial",
            FREE_BF16,
        );
        let (mut x, mut frames, mut channels) = ("initial".to_owned(), "frames".to_owned(), 1024);
        for (index, stride) in STRIDES.into_iter().enumerate() {
            let block = format!("{CONV}.block.{index}");
            let stem = format!("block-{index}");
            let rows = format!("{stem}-frames");
            let activated = format!("{stem}-activated");
            let upsampled = format!("{stem}-upsampled");
            mul(&mut codec.body, &frames, stride, &rows);
            codec.snake(
                &format!("{block}.snake1"),
                &x,
                &frames,
                channels,
                &activated,
            );
            codec.conv(
                &format!("{block}.conv_t1"),
                &activated,
                &frames,
                &rows,
                channels,
                channels / 2,
                stride * 2,
                stride,
                1,
                true,
                &upsampled,
            );
            let block_input = format!("{stem}-input");
            free(
                &mut codec.body,
                &[&x, &activated],
                &upsampled,
                &block_input,
                FREE_BF16,
            );
            channels /= 2;
            frames = rows;
            x = block_input;
            for (index, dilation) in DILATIONS.into_iter().enumerate() {
                let unit = format!("{block}.res_unit{}", index + 1);
                let stem = format!("{stem}-unit-{index}");
                let first = format!("{stem}-snake1");
                let conv = format!("{stem}-conv1");
                let second = format!("{stem}-snake2");
                let last = format!("{stem}-conv2");
                let added = format!("{stem}-add");
                let next = format!("{stem}-output");
                codec.snake(&format!("{unit}.snake1"), &x, &frames, channels, &first);
                codec.conv(
                    &format!("{unit}.conv1"),
                    &first,
                    &frames,
                    &frames,
                    channels,
                    channels,
                    7,
                    1,
                    dilation,
                    false,
                    &conv,
                );
                codec.snake(&format!("{unit}.snake2"), &conv, &frames, channels, &second);
                codec.conv(
                    &format!("{unit}.conv2"),
                    &second,
                    &frames,
                    &frames,
                    channels,
                    channels,
                    1,
                    1,
                    1,
                    false,
                    &last,
                );
                call(&mut codec.body, &[&x, &last], "qwen3.add", &added);
                free(
                    &mut codec.body,
                    &[&x, &first, &conv, &second, &last],
                    &added,
                    &next,
                    FREE_BF16,
                );
                x = next;
            }
        }
        codec.snake(
            &format!("{CONV}.snake1"),
            &x,
            &frames,
            channels,
            "last-snake",
        );
        codec.conv(
            &format!("{CONV}.conv2"),
            "last-snake",
            &frames,
            &frames,
            channels,
            1,
            7,
            1,
            1,
            false,
            "raw",
        );
        free(
            &mut codec.body,
            &[&x, "last-snake"],
            "raw",
            "wave",
            FREE_BF16,
        );
        finish(&mut codec.body, "wave");
        codec
    }
}

fn pack_layers(source: &mut String, layer_names: &[String]) {
    let groups = (0..layer_names.len() / 11)
        .map(|layer| {
            let name = format!("layer{layer}");
            qwen::pack(source, &layer_names[layer * 11..][..11], &name);
            name
        })
        .collect::<Vec<_>>();
    qwen::pack(source, &groups, "weights");
}

pub(super) fn build(
    config: &ModelConfig,
    diagnostics: bool,
    dialect: GpuDialect,
) -> (String, Vec<Tensor>) {
    let mut source = include_str!("../catena/memory.hex").to_owned();
    for fragment in [
        include_str!("../catena/numeric.hex"),
        include_str!("../catena/resources.hex"),
        include_str!("../catena/cooperative.hex"),
        include_str!("../catena/normalization.hex"),
        include_str!("../catena/projection.hex"),
        include_str!("../catena/qwen35_batch_wmma.hex"),
        include_str!("../catena/qwen35_batch_projection.hex"),
        include_str!("../catena/bf16_simt.hex"),
        include_str!("../catena/bf16_attention.hex"),
        include_str!("../catena/whisper.hex"),
        include_str!("../catena/qwen35_prefill.hex"),
        include_str!("../catena/state_buffers.hex"),
        include_str!("../catena/lfm3_online_attention.hex"),
        include_str!("../catena/omnivoice.hex"),
    ] {
        qwen::fused::append_hex(&mut source, fragment);
    }
    let tts = include_str!("../catena/tts_model.hex")
        .replace("stdlib.whisper.linear", "stdlib.omnivoice.conv-project");
    qwen::fused::append_hex(&mut source, &tts);
    source.push_str(&catena::rope_frequency_program(config));
    source = source
        .replace(
            "{{RMS_EPSILON_BITS}}",
            &format!("{:08X}", config.rms_norm_eps.to_bits()),
        )
        .replace("{{HIDDEN_SIZE}}", &format!("{:016X}", config.hidden_size));

    // The shared patch builder supplies explicit rows, already represented by the input buffer.
    source.push_str(
        "(def program stdlib.omnivoice.conv-project : {((cap.own mem) val) ((cap.ref mem) val) (u64 val) (u64 val) (u64 val)} -> ((cap.own mem) val) =
  ([input weight rows ci co.] ({[.input] [.weight] [.ci] [.co]} stdlib.qwen35.batch.wmma-project)))
",
    );
    let group = qwen::nested_product(&vec![REF.to_owned(); 11]);
    let weights = qwen::nested_product(&vec![group; config.num_layers]);
    // K64 won the HIP benchmark. On sm_80, K32 avoids the K64 spills.
    let wide_tile = match dialect {
        GpuDialect::Hip => "m128n64k64",
        GpuDialect::Cuda => "m128n64k32",
    };
    source.push_str(&catena::definitions(
        include_str!("../catena/lfm3_wmma.hex"),
        &[&format!("stdlib.lfm3.wmma.{wide_tile}.")],
    ));
    source.push_str(
        &include_str!("../catena/omnivoice_projection.hex").replace("{{WIDE_TILE}}", wide_tile),
    );
    backbone(&mut source, config, &weights);
    sampler(&mut source, &weights);
    let codec = Codec::new();
    let codec_names: Vec<_> = (0..codec.weights.len())
        .map(|i| format!("codec-{i}"))
        .collect();
    let mut codec_args = vec![
        ("codes", OWN),
        ("frames", U64),
        ("embeddings", REF),
        ("projections", REF),
        ("biases", REF),
    ];
    ref_inputs(&mut codec_args, &codec_names);
    header(&mut source, "stdlib.omnivoice.decode", &codec_args, OWN);
    source.push_str(&codec.body);

    let mut inputs = vec![
        ("text", REF),
        ("prefix", U64),
        ("frames", U64),
        ("seed", U64),
        ("num-step", U32),
        ("text-embedding", REF),
        ("audio-embedding", REF),
        ("audio-heads", REF),
        ("output-norm", REF),
    ];
    let layer_names: Vec<_> = (0..config.num_layers * 11)
        .map(|i| format!("layer-weight{i}"))
        .collect();
    ref_inputs(&mut inputs, &layer_names);
    let logits_input_count = inputs.len();
    inputs.extend_from_slice(&codec_args[2..]);

    header(&mut source, "omnivoice.synthesize", &inputs, OWN);
    pack_layers(&mut source, &layer_names);
    call(
        &mut source,
        &[
            "text",
            "text-embedding",
            "audio-embedding",
            "audio-heads",
            "output-norm",
            "weights",
            "prefix",
            "frames",
            "seed",
            "num-step",
        ],
        "stdlib.omnivoice.generate",
        "codes",
    );
    let names = codec_args.iter().map(|(name, _)| *name).collect::<Vec<_>>();
    call(&mut source, &names, "stdlib.omnivoice.decode", "raw");
    mul(&mut source, "frames", 960, "samples");
    kernel(&mut source, "peak", &["raw", "samples"], &c(1), "peak");
    add(&mut source, "samples", &c(4800), "length");
    kernel(
        &mut source,
        "waveform",
        &["raw", "peak", "samples"],
        "length",
        "waveform",
    );
    free(&mut source, &["raw"], "waveform", "wave-ready", FREE_BF16);
    free(&mut source, &["peak"], "wave-ready", "ready", FREE);
    finish(&mut source, "ready");

    if diagnostics {
        source.push_str(
            "(def program omnivoice.schedule : {(u64 val) (u64 val) (u64 val) (u64 val)} -> (u64 val) =
 ([step steps slots remaining.]
  ({[.step] [.steps] [.slots] [.remaining]} stdlib.omnivoice.schedule)
 ))\n",
        );
        let forward_args = inputs[..logits_input_count]
            .iter()
            .filter_map(|&(name, ty)| match name {
                "seed" => Some(("codes", OWN)),
                "num-step" => None,
                _ => Some((name, ty)),
            })
            .collect::<Vec<_>>();
        header(&mut source, "omnivoice.logits", &forward_args, OWN);
        pack_layers(&mut source, &layer_names);
        add(&mut source, "prefix", "frames", "rows");
        mul(&mut source, "rows", 128, "rope-count");
        kernel(&mut source, "rope-table", &[], "rope-count", "rope");
        mul(&mut source, "rows", 16, "gate-count");
        kernel(&mut source, "identity-gate", &[], "gate-count", "identity");
        call(
            &mut source,
            &[
                "text",
                "text-embedding",
                "audio-embedding",
                "audio-heads",
                "output-norm",
                "weights",
                "codes",
                "rope",
                "identity",
                "prefix",
                "frames",
            ],
            "stdlib.omnivoice.logits",
            "logits",
        );
        free(
            &mut source,
            &["codes", "rope"],
            "logits",
            "logits-ready",
            FREE,
        );
        free(
            &mut source,
            &["identity"],
            "logits-ready",
            "forward-result",
            FREE_BF16,
        );
        finish(&mut source, "forward-result");

        header(
            &mut source,
            "omnivoice.select",
            &[
                ("codes", OWN),
                ("scores", OWN),
                ("slots", U64),
                ("count", U64),
            ],
            OWN,
        );
        kernel(
            &mut source,
            "select-codes",
            &["codes", "scores", "slots", "count"],
            "slots",
            "next",
        );
        free(
            &mut source,
            &["codes", "scores"],
            "next",
            "next-ready",
            FREE,
        );
        finish(&mut source, "next-ready");

        header(&mut source, "omnivoice.decode", &codec_args, OWN);
        call(&mut source, &names, "stdlib.omnivoice.decode", "raw");
        finish(&mut source, "raw");
    }
    // Restrict the exported ABI to OmniVoice. Shared helpers are pruned by reachability.
    let mapping: BTreeMap<_, _> = source
        .split("(def program ")
        .skip(1)
        .filter_map(|definition| {
            let name = definition.split_whitespace().next()?;
            (!name.starts_with("stdlib.") && !name.starts_with("omnivoice."))
                .then(|| (name.to_owned(), format!("stdlib.omnivoice.shared.{name}")))
        })
        .collect();
    (catena::rename_programs(&source, &mapping), codec.weights)
}
