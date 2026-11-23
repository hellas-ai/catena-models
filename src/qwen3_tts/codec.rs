//! The decoder half of the bundled Qwen3-TTS 12 Hz speech tokenizer.
use super::{Binding, FRAME_SAMPLES, Storage, kernels::*};
use crate::{
    catena,
    qwen::{self, c, call, header, nested_product, pack, unpack},
    spec::ModelConfig,
};
use anyhow::{Result, ensure};
use serde_json::{Value, json};

const CHUNK: usize = 300;
const HISTORY: usize = 25;
const LINEAR: &str = "stdlib.qwen35.batch.wmma-project";
const LAYER: &str = "stdlib.qwen-tts.codec-layer";

pub(super) fn validate(config: &Value) -> Result<()> {
    for (field, expected) in [
        ("latent_dim", json!(1024)),
        ("codebook_dim", json!(512)),
        ("codebook_size", json!(2048)),
        ("decoder_dim", json!(1536)),
        ("hidden_size", json!(512)),
        ("intermediate_size", json!(1024)),
        ("num_hidden_layers", json!(8)),
        ("num_attention_heads", json!(16)),
        ("num_key_value_heads", json!(16)),
        ("head_dim", json!(64)),
        ("rope_theta", json!(10000)),
        ("sliding_window", json!(72)),
        ("rms_norm_eps", json!(1e-5)),
        ("num_quantizers", json!(16)),
        ("num_semantic_quantizers", json!(1)),
        ("upsampling_ratios", json!([2, 2])),
        ("upsample_rates", json!([8, 5, 4, 3])),
    ] {
        ensure!(
            config["decoder_config"][field] == expected,
            "unsupported Qwen3-TTS speech decoder {field}: expected {expected}"
        );
    }
    ensure!(
        config["output_sample_rate"].as_u64() == Some(24_000)
            && config["decode_upsample_rate"].as_u64() == Some(FRAME_SAMPLES as u64),
        "unsupported Qwen3-TTS waveform sample rate"
    );
    Ok(())
}

#[derive(Default)]
struct Decoder {
    body: String,
    bindings: Vec<Binding>,
    stages: String,
}

impl Decoder {
    fn stage(
        &mut self,
        name: &str,
        captures: &[(&str, &str)],
        result: &str,
        build: impl FnOnce(&mut Self),
    ) {
        let previous = std::mem::take(&mut self.body);
        let start = self.bindings.len();
        build(self);
        let labels = (start..self.bindings.len())
            .map(|i| format!("codec-w{i}"))
            .collect::<Vec<_>>();
        let mut inputs = captures.to_vec();
        inputs.extend(labels.iter().map(|label| (label.as_str(), REF)));
        let program = format!("stdlib.qwen-tts.codec-{name}");
        header(&mut self.stages, &program, &inputs, OWN);
        self.stages.push_str(&self.body);
        finish(&mut self.stages, &format!("[.{result}]"));
        self.body = previous;
        let args = inputs.iter().map(|(name, _)| *name).collect::<Vec<_>>();
        call(&mut self.body, &args, &program, result);
    }

    fn tensor(&mut self, path: &str, shape: impl Into<Vec<usize>>, storage: Storage) -> String {
        let label = format!("codec-w{}", self.bindings.len());
        self.bindings
            .push(Binding::new(format!("decoder.{path}"), shape, storage));
        label
    }

    fn conv(
        &mut self,
        path: &str,
        input: &str,
        source_rows: &str,
        rows: &str,
        channels: [usize; 2],
        taps: usize,
        dilation: usize,
        transpose_stride: Option<usize>,
        out: &str,
    ) {
        let [ci, co] = channels;
        let (weight, stride, pad, transpose) = if let Some(stride) = transpose_stride {
            (
                self.tensor(
                    &format!("{path}.weight"),
                    [ci, co, taps],
                    Storage::TransposeConv,
                ),
                stride,
                0,
                1,
            )
        } else {
            (
                self.tensor(&format!("{path}.weight"), [co, ci, taps], Storage::Bf16),
                1,
                (taps - 1) * dilation,
                0,
            )
        };
        let bias = self.tensor(&format!("{path}.bias"), [co], Storage::Bf16);
        let tile = (8 * 1024 * 1024 / (2 * ci * taps)).min(2048) / 128 * 128;
        call(
            &mut self.body,
            &[
                input,
                &weight,
                &bias,
                source_rows,
                rows,
                &c(ci),
                &c(co),
                &c(taps),
                &c(dilation),
                &c(stride),
                &c(pad),
                &c(transpose),
                &c(1),
                &c(tile),
            ],
            "tts.conv-tiled",
            out,
        );
    }

    fn linear(&mut self, path: &str, input: &str, rows: &str, ci: usize, co: usize, out: &str) {
        let w = self.tensor(&format!("{path}.weight"), [co, ci], Storage::Bf16);
        let b = self.tensor(&format!("{path}.bias"), [co], Storage::Bf16);
        call(
            &mut self.body,
            &[input, &w, &c(ci), &c(co)],
            LINEAR,
            &format!("{out}-linear"),
        );
        mem(
            &mut self.body,
            "u64.mul",
            &[rows, &c(co)],
            &format!("{out}-count"),
        );
        kernel(
            &mut self.body,
            "bias",
            &[
                &format!("{out}-linear"),
                &b,
                &c(co),
                &format!("{out}-count"),
            ],
            &format!("{out}-biased"),
        );
        free(
            &mut self.body,
            &[&format!("{out}-linear")],
            &format!("{out}-biased"),
            out,
            FREE_BF16,
        );
    }

    fn snake(&mut self, path: &str, input: &str, rows: &str, channels: usize, out: &str) {
        let alpha = self.tensor(&format!("{path}.alpha"), [channels], Storage::F32);
        let beta = self.tensor(&format!("{path}.beta"), [channels], Storage::F32);
        mem(
            &mut self.body,
            "u64.mul",
            &[rows, &c(channels)],
            &format!("{out}-count"),
        );
        kernel(
            &mut self.body,
            "snake-beta",
            &[input, &alpha, &beta, &c(channels), &format!("{out}-count")],
            out,
        );
    }

    fn quantize(&mut self) {
        mem(
            &mut self.body,
            "u64.mul",
            &["rows", &c(256)],
            "book-elements",
        );
        let mut groups = Vec::new();
        for (kind, count, first_book) in [("first", 1, 0), ("rest", 15, 1)] {
            let mut codes = String::new();
            for book in 0..count {
                let base = format!("quantizer.rvq_{kind}.vq.layers.{book}._codebook");
                let embed =
                    self.tensor(&format!("{base}.embedding_sum"), [2048, 256], Storage::F32);
                let usage = self.tensor(&format!("{base}.cluster_usage"), [2048], Storage::F32);
                let raw = format!("{kind}-codes-{book}");
                kernel(
                    &mut self.body,
                    "rvq-lookup",
                    &[
                        "codes",
                        "progress",
                        "start",
                        &c(first_book + book),
                        &embed,
                        &usage,
                        "book-elements",
                    ],
                    &raw,
                );
                if book == 0 {
                    codes = raw;
                } else {
                    let sum = format!("{kind}-sum-{book}");
                    kernel(
                        &mut self.body,
                        "rvq-sum.into",
                        &[&raw, &codes, "book-elements"],
                        &sum,
                    );
                    free(
                        &mut self.body,
                        &[&raw],
                        &sum,
                        &format!("{sum}-ready"),
                        FREE_F32,
                    );
                    codes = format!("{sum}-ready");
                }
            }
            let weight = self.tensor(
                &format!("quantizer.rvq_{kind}.output_proj.weight"),
                [512, 256, 1],
                Storage::Bf16,
            );
            let latent = format!("{kind}-latent");
            call(
                &mut self.body,
                &[&codes, "book-elements"],
                "tts.from-f32",
                &format!("{kind}-bf"),
            );
            free(
                &mut self.body,
                &[&codes],
                &format!("{kind}-bf"),
                &format!("{kind}-ready"),
                FREE_F32,
            );
            call(
                &mut self.body,
                &[&format!("{kind}-ready"), &weight, &c(256), &c(512)],
                LINEAR,
                &format!("{latent}-raw"),
            );
            free(
                &mut self.body,
                &[&format!("{kind}-ready")],
                &format!("{latent}-raw"),
                &latent,
                FREE_BF16,
            );
            groups.push(latent);
        }
        call(
            &mut self.body,
            &[&groups[0], &groups[1]],
            "qwen3.add",
            "latent-raw",
        );
        free(
            &mut self.body,
            &[&groups[0], &groups[1]],
            "latent-raw",
            "latent",
            FREE_BF16,
        );
    }

    fn transformer(&mut self) {
        self.conv(
            "pre_conv.conv",
            "latent",
            "rows",
            "rows",
            [512, 1024],
            3,
            1,
            None,
            "preconv",
        );
        self.linear(
            "pre_transformer.input_proj",
            "preconv",
            "rows",
            1024,
            512,
            "input",
        );
        free(
            &mut self.body,
            &["latent", "preconv"],
            "input",
            "pre-transformer",
            FREE_BF16,
        );
        let mut groups = Vec::new();
        for layer in 0..8 {
            let mut weights = Vec::new();
            for (name, shape, storage) in [
                ("input_layernorm.weight", vec![512], Storage::F32),
                ("self_attn.q_proj.weight", vec![1024, 512], Storage::Bf16),
                ("self_attn.k_proj.weight", vec![1024, 512], Storage::Bf16),
                ("self_attn.v_proj.weight", vec![1024, 512], Storage::Bf16),
                ("self_attn.o_proj.weight", vec![512, 1024], Storage::Bf16),
                ("self_attn_layer_scale.scale", vec![512], Storage::F32),
                ("post_attention_layernorm.weight", vec![512], Storage::F32),
                ("mlp.gate_proj.weight", vec![1024, 512], Storage::Bf16),
                ("mlp.up_proj.weight", vec![1024, 512], Storage::Bf16),
                ("mlp.down_proj.weight", vec![512, 1024], Storage::Bf16),
                ("mlp_layer_scale.scale", vec![512], Storage::F32),
            ] {
                weights.push(self.tensor(
                    &format!("pre_transformer.layers.{layer}.{name}"),
                    shape,
                    storage,
                ));
            }
            let group = format!("codec-layer-{layer}");
            pack(&mut self.body, &weights, &group);
            groups.push(group);
        }
        pack(&mut self.body, &groups, "layer-weights");
        mem(
            &mut self.body,
            "u64.mul",
            &["rows", &c(64)],
            "rope-elements",
        );
        kernel(&mut self.body, "codec-rope", &["rope-elements"], "rope");
        fold(
            &mut self.body,
            LAYER,
            &["layer-weights", "rows", "rope"],
            &u(8),
            "pre-transformer",
            "transformed",
        );
        let norm = self.tensor("pre_transformer.norm.weight", [512], Storage::F32);
        rms(&mut self.body, "transformed", &norm, "normalized");
        free(
            &mut self.body,
            &["transformed"],
            "normalized",
            "hidden-ready",
            FREE_BF16,
        );
        free(
            &mut self.body,
            &["rope"],
            "hidden-ready",
            "hidden",
            FREE_F32,
        );
        self.linear(
            "pre_transformer.output_proj",
            "hidden",
            "rows",
            512,
            1024,
            "projected",
        );
        free(
            &mut self.body,
            &["hidden"],
            "projected",
            "upsample-input",
            FREE_BF16,
        );
    }

    fn upsample(&mut self) -> (String, String) {
        let (mut input, mut rows) = ("upsample-input".to_owned(), "rows".to_owned());
        for stage in 0..2 {
            let path = format!("upsample.{stage}");
            let next_rows = format!("upsample-rows-{stage}");
            mem(&mut self.body, "u64.mul", &[&rows, &c(2)], &next_rows);
            let up = format!("upsample-{stage}");
            self.conv(
                &format!("{path}.0.conv"),
                &input,
                &rows,
                &next_rows,
                [1024, 1024],
                2,
                1,
                Some(2),
                &up,
            );
            let dw_weight = self.tensor(
                &format!("{path}.1.dwconv.conv.weight"),
                [1024, 1, 7],
                Storage::Bf16,
            );
            let dw_bias = self.tensor(&format!("{path}.1.dwconv.conv.bias"), [1024], Storage::Bf16);
            let norm_weight = self.tensor(&format!("{path}.1.norm.weight"), [1024], Storage::F32);
            let norm_bias = self.tensor(&format!("{path}.1.norm.bias"), [1024], Storage::F32);
            let scale = self.tensor(&format!("{path}.1.gamma"), [1024], Storage::F32);
            let ready = format!("{up}-ready");
            free(&mut self.body, &[&input], &up, &ready, FREE_BF16);
            let dw = format!("{up}-dw");
            let count = format!("{up}-count");
            mem(&mut self.body, "u64.mul", &[&next_rows, &c(1024)], &count);
            kernel(
                &mut self.body,
                "depthwise",
                &[&ready, &dw_weight, &dw_bias, &c(1024), &count],
                &dw,
            );
            let stats = format!("{up}-stats");
            mem(
                &mut self.body,
                "u64.mul",
                &[&next_rows, &c(2)],
                &format!("{stats}-count"),
            );
            mem(
                &mut self.body,
                "alloc-f32",
                &[&format!("{stats}-count")],
                &format!("{stats}-raw"),
            );
            kernel(
                &mut self.body,
                "layer-stats.into",
                &[&dw, &format!("{stats}-raw"), &next_rows],
                &stats,
            );
            let norm = format!("{up}-norm");
            kernel(
                &mut self.body,
                "layer-norm",
                &[&dw, &stats, &norm_weight, &norm_bias, &count],
                &norm,
            );
            let fc1 = format!("{up}-fc1");
            self.linear(
                &format!("{path}.1.pwconv1"),
                &norm,
                &next_rows,
                1024,
                4096,
                &fc1,
            );
            let gelu_count = format!("{up}-gelu-count");
            mem(
                &mut self.body,
                "u64.mul",
                &[&next_rows, &c(4096)],
                &gelu_count,
            );
            let gelu = format!("{up}-gelu");
            kernel(&mut self.body, "gelu", &[&fc1, &gelu_count], &gelu);
            let activation = format!("{up}-activation");
            free(
                &mut self.body,
                &[&dw, &norm, &fc1],
                &gelu,
                &format!("{activation}-bf"),
                FREE_BF16,
            );
            free(
                &mut self.body,
                &[&stats],
                &format!("{activation}-bf"),
                &activation,
                FREE_F32,
            );
            let fc2 = format!("{up}-fc2");
            self.linear(
                &format!("{path}.1.pwconv2"),
                &activation,
                &next_rows,
                4096,
                1024,
                &fc2,
            );
            let added = format!("{up}-added");
            kernel(
                &mut self.body,
                "scaled-residual",
                &[&ready, &fc2, &scale, &c(1024), &count],
                &added,
            );
            input = format!("{up}-result");
            free(
                &mut self.body,
                &[&ready, &activation, &fc2],
                &added,
                &input,
                FREE_BF16,
            );
            rows = next_rows;
        }
        (input, rows)
    }

    fn waveform(&mut self, input: &str, mut rows: String) {
        self.conv(
            "decoder.0.conv",
            input,
            &rows,
            &rows,
            [1024, 1536],
            7,
            1,
            None,
            "generator-input",
        );
        free(
            &mut self.body,
            &[input],
            "generator-input",
            "generator-start",
            FREE_BF16,
        );
        let (mut input, mut width) = ("generator-start".to_owned(), 1536);
        for (stage, stride) in [8, 5, 4, 3].into_iter().enumerate() {
            let path = format!("decoder.{}.block", stage + 1);
            let stem = format!("gen-{stage}");
            let next_rows = format!("{stem}-rows");
            mem(&mut self.body, "u64.mul", &[&rows, &c(stride)], &next_rows);
            let output = format!("{stem}-output");
            let input_name = input.clone();
            self.stage(
                &stem,
                &[(&input_name, OWN), (&rows, U64), (&next_rows, U64)],
                &output,
                |this| {
                    let mut input = input_name.clone();
                    let activated = format!("{stem}-activated");
                    this.snake(&format!("{path}.0"), &input, &rows, width, &activated);
                    let up = format!("{stem}-up");
                    this.conv(
                        &format!("{path}.1.conv"),
                        &activated,
                        &rows,
                        &next_rows,
                        [width, width / 2],
                        2 * stride,
                        1,
                        Some(stride),
                        &up,
                    );
                    let ready = format!("{stem}-up-ready");
                    free(
                        &mut this.body,
                        &[&input, &activated],
                        &up,
                        &ready,
                        FREE_BF16,
                    );
                    input = ready;
                    let rows = &next_rows;
                    let width = width / 2;
                    for (unit, dilation) in [1, 3, 9].into_iter().enumerate() {
                        let path = format!("{path}.{}", unit + 2);
                        let stem = format!("{stem}-res-{unit}");
                        let first = format!("{stem}-s1");
                        let conv1 = format!("{stem}-c1");
                        let second = format!("{stem}-s2");
                        let conv2 = format!("{stem}-c2");
                        this.snake(&format!("{path}.act1"), &input, &rows, width, &first);
                        this.conv(
                            &format!("{path}.conv1.conv"),
                            &first,
                            &rows,
                            &rows,
                            [width, width],
                            7,
                            dilation,
                            None,
                            &conv1,
                        );
                        this.snake(&format!("{path}.act2"), &conv1, &rows, width, &second);
                        this.conv(
                            &format!("{path}.conv2.conv"),
                            &second,
                            &rows,
                            &rows,
                            [width, width],
                            1,
                            1,
                            None,
                            &conv2,
                        );
                        call(
                            &mut this.body,
                            &[&input, &conv2],
                            "qwen3.add",
                            &format!("{stem}-added"),
                        );
                        let result = format!("{stem}-result");
                        free(
                            &mut this.body,
                            &[&input, &first, &conv1, &second, &conv2],
                            &format!("{stem}-added"),
                            &result,
                            FREE_BF16,
                        );
                        input = result;
                    }
                    this.body.push_str(&format!("  ([.{input}] [{output}.])\n"));
                },
            );
            input = output;
            rows = next_rows;
            width /= 2;
        }
        self.snake("decoder.5", &input, &rows, width, "final-snake");
        self.conv(
            "decoder.6.conv",
            "final-snake",
            &rows,
            &rows,
            [width, 1],
            7,
            1,
            None,
            "final-conv",
        );
        free(
            &mut self.body,
            &[&input, "final-snake"],
            "final-conv",
            "pcm",
            FREE_BF16,
        );
    }
}

fn rms(s: &mut String, x: &str, weight: &str, out: &str) {
    call(
        s,
        &[x, weight, &c(512), &f(512.0), "rows"],
        "stdlib.qwen-tts.codec-normalize",
        out,
    );
}

fn layer(s: &mut String) {
    let group = nested_product(&vec![REF.to_owned(); 11]);
    let weights = nested_product(&vec![group.clone(); 8]);
    let mut selector = String::new();
    qwen::emit_layer_selector(&mut selector, 8, &group, &weights);
    s.push_str(
        &selector
            .replace(
                "stdlib.qwen3.select-layer",
                "stdlib.qwen-tts.codec-select-layer",
            )
            .replace(
                "stdlib.qwen3.layer-weights",
                "stdlib.qwen-tts.codec-layer-weights",
            ),
    );
    step_header(
        s,
        LAYER,
        &[("weights", &weights), ("rows", U64), ("rope", OWN)],
        OWN,
    );
    call(
        s,
        &["weights", "i32"],
        "stdlib.qwen-tts.codec-layer-weights",
        "selected",
    );
    unpack(
        s,
        "selected",
        &[
            "input-norm",
            "q-w",
            "k-w",
            "v-w",
            "o-w",
            "attention-scale",
            "post-norm",
            "gate-w",
            "up-w",
            "down-w",
            "mlp-scale",
        ]
        .map(str::to_owned),
    );
    rms(s, "state", "input-norm", "normalized");
    mem(s, "u64.mul", &["rows", &c(1024)], "attention-elements");
    for name in ["q", "k", "v"] {
        call(
            s,
            &["normalized", &format!("{name}-w"), &c(512), &c(1024)],
            LINEAR,
            &format!("{name}-projected"),
        );
        if name == "v" {
            s.push_str("  ([.v-projected] [v.])\n");
        } else {
            kernel(
                s,
                "codec-rotate",
                &[&format!("{name}-projected"), "rope", "attention-elements"],
                &format!("{name}-rotated"),
            );
            free(
                s,
                &[&format!("{name}-projected")],
                &format!("{name}-rotated"),
                name,
                FREE_BF16,
            );
        }
    }
    mem(s, "u64.mul", &["rows", &c(16)], "query-heads");
    mem(s, "u64.mul", &["query-heads", "rows"], "score-count");
    call(
        s,
        &[
            "q",
            "k",
            "rows",
            &c(16),
            &c(64),
            &f(0.125),
            "score-count",
            "score-count",
        ],
        "stdlib.whisper.scores-o.run",
        "scores-raw",
    );
    kernel(
        s,
        "causal-mask",
        &["scores-raw", "rows", "score-count"],
        "masked",
    );
    free(
        s,
        &["q", "k", "normalized"],
        "masked",
        "scores-bf",
        FREE_BF16,
    );
    free(s, &["scores-raw"], "scores-bf", "scores", FREE_F32);
    mem(s, "u64.mul", &["query-heads", &c(2)], "stats-count");
    call(
        s,
        &["scores", "rows", "query-heads", "stats-count"],
        "stdlib.whisper.softmax-stats.run",
        "stats",
    );
    call(
        s,
        &["scores", "stats", "rows", "score-count", "score-count"],
        "stdlib.whisper.softmax.run",
        "probs",
    );
    free(s, &["scores", "stats"], "probs", "probabilities", FREE_F32);
    call(
        s,
        &[
            "probabilities",
            "v",
            "rows",
            &c(16),
            &c(64),
            "attention-elements",
            "attention-elements",
        ],
        "stdlib.whisper.context-o.run",
        "attention",
    );
    free(
        s,
        &["probabilities"],
        "attention",
        "attention-computed",
        FREE_F32,
    );
    free(s, &["v"], "attention-computed", "context", FREE_BF16);
    call(
        s,
        &["context", "o-w", &c(1024), &c(512)],
        LINEAR,
        "attn-projected",
    );
    mem(s, "u64.mul", &["rows", &c(512)], "elements");
    kernel(
        s,
        "scaled-residual",
        &[
            "state",
            "attn-projected",
            "attention-scale",
            &c(512),
            "elements",
        ],
        "attention-added",
    );
    free(
        s,
        &["state", "context", "attn-projected"],
        "attention-added",
        "residual",
        FREE_BF16,
    );
    rms(s, "residual", "post-norm", "ffn");
    call(s, &["ffn", "gate-w", &c(512), &c(1024)], LINEAR, "gate");
    call(s, &["ffn", "up-w", &c(512), &c(1024)], LINEAR, "up");
    call(s, &["gate", "up"], "qwen3.swiglu", "activated");
    free(s, &["ffn", "gate", "up"], "activated", "gated", FREE_BF16);
    call(s, &["gated", "down-w", &c(1024), &c(512)], LINEAR, "down");
    kernel(
        s,
        "scaled-residual",
        &["residual", "down", "mlp-scale", &c(512), "elements"],
        "added",
    );
    free(
        s,
        &["residual", "gated", "down"],
        "added",
        "result",
        FREE_BF16,
    );
    step_finish(s, "result");
}

pub(super) fn emit() -> (String, Vec<Binding>) {
    let mut s = String::new();
    let mut rope = ModelConfig::qwen3_0_6b();
    rope.rope_theta = 10_000.0;
    rope.rope_dimension_count = 64;
    s.push_str(&catena::rope_frequency_program(&rope).replace(
        "qwen3.rope-frequency",
        "stdlib.qwen-tts.codec-rope-frequency",
    ));
    s.push_str(
        &include_str!("../catena/fused_normalization.hex")
            .replace(
                "{{RMS_EPSILON_BITS}}",
                &format!("{:08X}", 1e-5f32.to_bits()),
            )
            .replace("stdlib.qwen3.fused.norm.", "stdlib.qwen-tts.codec-norm.")
            .replace("qwen3.fused.normalize", "stdlib.qwen-tts.codec-normalize"),
    );
    layer(&mut s);
    emit_kernels(&mut s);
    let mut decoder = Decoder::default();
    decoder.stage(
        "rvq",
        &[
            ("codes", OWN),
            ("progress", OWN),
            ("start", U64),
            ("rows", U64),
        ],
        "latent",
        Decoder::quantize,
    );
    decoder.stage(
        "transformer",
        &[("latent", OWN), ("rows", U64)],
        "upsample-input",
        Decoder::transformer,
    );
    let (input, rows) = decoder.upsample();
    decoder.waveform(&input, rows);
    let labels = (0..decoder.bindings.len())
        .map(|i| format!("codec-w{i}"))
        .collect::<Vec<_>>();
    let mut chunk = vec![
        ("codes", OWN),
        ("progress", OWN),
        ("start", U64),
        ("rows", U64),
    ];
    chunk.extend(labels.iter().map(|l| (l.as_str(), REF)));
    s.push_str(&decoder.stages);
    header(&mut s, "stdlib.qwen-tts.decode-chunk", &chunk, OWN);
    s.push_str(&decoder.body);
    finish(&mut s, "[.pcm]");

    let mut decode = vec![("codes", OWN), ("progress", OWN), ("budget", U64)];
    decode.extend(labels.iter().map(|l| (l.as_str(), REF)));
    let groups = labels.chunks(16).collect::<Vec<_>>();
    let group_names = (0..groups.len())
        .map(|i| format!("group-{i}"))
        .collect::<Vec<_>>();
    let weights = nested_product(
        &groups
            .iter()
            .map(|group| nested_product(&vec![REF.to_owned(); group.len()]))
            .collect::<Vec<_>>(),
    );
    step_header(
        &mut s,
        "stdlib.qwen-tts.decode-step",
        &[
            ("codes", OWN),
            ("progress", OWN),
            ("budget", U64),
            ("weights", &weights),
        ],
        OWN,
    );
    unpack(&mut s, "weights", &group_names);
    for (group, names) in group_names.iter().zip(&groups) {
        unpack(&mut s, group, names);
    }
    mem(&mut s, "u64.mul", &["i", &c(CHUNK)], "start");
    mem(&mut s, "u64.sub", &["budget", "start"], "remaining");
    mem(&mut s, "u64.less", &["remaining", &c(CHUNK)], "tail");
    mem(
        &mut s,
        "select-u64",
        &["tail", "remaining", &c(CHUNK)],
        "new-rows",
    );
    mem(&mut s, "u64.equal", &["i", &c(0)], "first");
    mem(
        &mut s,
        "select-u64",
        &["first", &c(0), &c(HISTORY)],
        "context",
    );
    mem(&mut s, "u64.sub", &["start", "context"], "chunk-start");
    mem(&mut s, "u64.add", &["new-rows", "context"], "rows");
    let mut args = vec!["codes", "progress", "chunk-start", "rows"];
    args.extend(labels.iter().map(String::as_str));
    kernel(&mut s, "decode-chunk", &args, "decoded");
    mem(&mut s, "u64.mul", &["context", &c(FRAME_SAMPLES)], "skip");
    mem(
        &mut s,
        "u64.mul",
        &["start", &c(FRAME_SAMPLES)],
        "output-offset",
    );
    mem(
        &mut s,
        "u64.mul",
        &["new-rows", &c(FRAME_SAMPLES)],
        "samples",
    );
    kernel(
        &mut s,
        "copy-wave.into",
        &["decoded", "skip", "output-offset", "state", "samples"],
        "copied",
    );
    free(&mut s, &["decoded"], "copied", "pcm", FREE_BF16);
    step_finish(&mut s, "pcm");

    header(&mut s, "stdlib.qwen-tts.decode", &decode, OWN);
    mem(
        &mut s,
        "u64.mul",
        &["budget", &c(FRAME_SAMPLES)],
        "capacity",
    );
    mem(&mut s, "alloc-f32", &["capacity"], "output");
    mem(&mut s, "u64.add", &["budget", &c(CHUNK - 1)], "rounded");
    mem(&mut s, "u64.div", &["rounded", &c(CHUNK)], "chunks");
    mem(&mut s, "fold-count-u32", &["chunks"], "n");
    for (group, labels) in group_names.iter().zip(&groups) {
        pack(&mut s, labels, group);
    }
    pack(&mut s, &group_names, "weights");
    fold(
        &mut s,
        "stdlib.qwen-tts.decode-step",
        &["codes", "progress", "budget", "weights"],
        "[.n]",
        "output",
        "wave",
    );
    finish(&mut s, "[.wave]");
    (s, decoder.bindings)
}

fn emit_kernels(s: &mut String) {
    let (functions, elements) = include_str!("../catena/qwen3_tts_codec.hex")
        .split_once("\n# Element bodies\n")
        .unwrap();
    s.push_str(functions);
    let mut elements = elements.split("# Element ").skip(1);
    let mut emit = |label, args: &[(&str, &str)], dtype| {
        let (name, body) = elements.next().unwrap().split_once('\n').unwrap();
        assert_eq!(label, name);
        launch(s, label, args, dtype, body);
    };
    emit(
        "rvq-lookup",
        &[
            ("codes", OWN),
            ("progress", OWN),
            ("start", U64),
            ("book", U64),
            ("embedding", REF),
            ("usage", REF),
        ],
        "f32",
    );
    emit("rvq-sum", &[("input", OWN)], "f32");
    emit("codec-rope", &[], "f32");
    emit("codec-rotate", &[("input", OWN), ("rope", OWN)], "bf16");
    emit("causal-mask", &[("scores", OWN), ("rows", U64)], "f32");
    emit(
        "scaled-residual",
        &[
            ("input", OWN),
            ("delta", OWN),
            ("scale", REF),
            ("width", U64),
        ],
        "bf16",
    );
    emit(
        "depthwise",
        &[
            ("input", OWN),
            ("weight", REF),
            ("bias", REF),
            ("cols", U64),
        ],
        "bf16",
    );
    emit("layer-stats", &[("input", OWN)], "f32");
    emit(
        "layer-norm",
        &[
            ("input", OWN),
            ("stats", OWN),
            ("weight", REF),
            ("bias", REF),
        ],
        "bf16",
    );
    emit("gelu", &[("input", OWN)], "bf16");
    emit(
        "snake-beta",
        &[
            ("input", OWN),
            ("alpha", REF),
            ("beta", REF),
            ("width", U64),
        ],
        "bf16",
    );
    emit(
        "copy-wave",
        &[("input", OWN), ("skip", U64), ("offset", U64)],
        "f32",
    );
    assert!(elements.next().is_none());
}
