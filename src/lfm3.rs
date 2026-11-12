//! LFM3 text graphs with model-specific projections and shared attention helpers.
use crate::{
    catena::{InferenceOptions, KernelKind},
    qwen::{c, call, header},
    spec::{ModelConfig, weight_specs},
};
use anyhow::{Result, ensure};
use std::fmt::Write;
const OWN: &str = "((cap.own mem) val)";
const REF: &str = "((cap.ref mem) val)";
const U: &str = "(u64 val)";

pub(crate) fn validate(cfg: &ModelConfig, options: InferenceOptions) -> Result<()> {
    cfg.validate()?;
    ensure!(
        options.kernels == KernelKind::Fused,
        "LFM3 requires --kernels fused"
    );
    let l = cfg.lfm3.as_ref().unwrap();
    ensure!(
        cfg.num_attention_heads.is_multiple_of(8),
        "LFM3 grouped projections require head counts divisible by eight"
    );
    ensure!(
        cfg.head_dim == 128,
        "LFM3 fused attention requires 128-channel heads"
    );
    ensure!(
        cfg.hidden_size.is_multiple_of(256),
        "LFM3 fused kernels require hidden widths divisible by 256"
    );
    ensure!(
        l.conv_kernel == 7,
        "LFM3 currently supports seven convolution taps"
    );
    ensure!(
        cfg.hidden_size.is_multiple_of(16)
            && cfg.intermediate_size.is_multiple_of(16)
            && cfg.head_dim.is_multiple_of(16),
        "LFM3 matrix widths must be multiples of 16"
    );
    ensure!(
        l.kv_heads
            .iter()
            .all(|&n| n == 0 || n == cfg.num_key_value_heads),
        "LFM3 attention layers must share their KV width"
    );
    Ok(())
}

/// Select shared helper families without importing another architecture's graphs.
fn definitions(source: &str, prefixes: &[&str]) -> String {
    let mut out = String::new();
    for (start, _) in source.match_indices("(def program ") {
        let tail = &source[start + 13..];
        let name = tail.split_whitespace().next().unwrap();
        if !prefixes.iter().any(|p| {
            p.strip_suffix(':')
                .map_or_else(|| name.starts_with(p), |exact| name == exact)
        }) {
            continue;
        }
        let mut depth = 0;
        for (i, ch) in source[start..].char_indices() {
            match ch {
                '(' => depth += 1,
                ')' => depth -= 1,
                _ => (),
            }
            if depth == 0 {
                out.push_str(&source[start..start + i + 1]);
                out.push('\n');
                break;
            }
        }
    }
    out
}

struct Emit<'a> {
    s: &'a mut String,
    next: usize,
}
impl Emit<'_> {
    fn op(&mut self, args: &[&str], op: &str) -> String {
        let n = format!("v{}", self.next);
        self.next += 1;
        call(self.s, args, op, &n);
        n
    }
    fn free(&mut self, old: &str, after: &str, dtype: &str) -> String {
        self.op(
            &[old, after],
            &format!("stdlib.models.resources.free-{dtype}-before"),
        )
    }
    fn after(&mut self, owner: &str, value: &str) -> String {
        let n = self.next;
        self.next += 1;
        writeln!(self.s, "  ([.{value}] meta.master.complete-own [completed{n} phase{n}.])\n  ({{[.{owner}] [.phase{n}]}} meta.master.after-own [after{n}.])").unwrap();
        format!("after{n}")
    }
    fn norm(&mut self, x: &str, w: &str, width: usize, rows: &str) -> String {
        self.op(
            &[
                x,
                w,
                &c(width),
                &float(width as f32),
                rows,
                &c(width.next_power_of_two().trailing_zeros() as usize),
                &c(width.next_power_of_two()),
            ],
            "qwen35.normalize",
        )
    }
    fn project(&mut self, x: &str, w: &str, features: usize, outputs: usize) -> String {
        let op = if features.is_multiple_of(256) && outputs.is_multiple_of(8) && outputs >= 128 {
            if outputs >= 32768 {
                "lfm3.project.fast"
            } else if features <= 4096 {
                "lfm3.project.wide-k64"
            } else {
                "lfm3.project.wide"
            }
        } else {
            "lfm3.project"
        };
        self.op(&[x, w, &c(features), &c(outputs)], op)
    }
    fn normres(&mut self, input: &str, residual: &str, post: &str, next: &str) -> (String, String) {
        let pair = self.op(&[input, residual, post, next], "stdlib.lfm3.normres");
        let out = format!("residual{}", self.next);
        let norm = format!("normalized{}", self.next);
        self.next += 1;
        writeln!(self.s, "  ([.{pair}] *.elim [{out} {norm}.])").unwrap();
        let out = self.free(input, &out, "bf16");
        let out = self.free(residual, &out, "bf16");
        let norm = self.after(&norm, &out);
        (out, norm)
    }
    fn element(&mut self, args: &[&str], name: &str, work: &str) -> String {
        let mut a = args.to_vec();
        a.extend([work, work]);
        self.op(&a, &format!("stdlib.lfm3.{name}.run"))
    }
}
fn float(x: f32) -> String {
    format!("(const.u32.0x{:08X} u32.bitcast-f32)", x.to_bits())
}

pub(crate) fn source(cfg: &ModelConfig, options: InferenceOptions) -> Result<String> {
    validate(cfg, options)?;
    let l = cfg.lfm3.as_ref().unwrap();
    let mut s = [
        include_str!("catena/memory.hex"),
        include_str!("catena/numeric.hex"),
        include_str!("catena/resources.hex"),
    ]
    .join("\n");
    crate::qwen::fused::append_hex(&mut s, include_str!("catena/cooperative.hex"));
    // The shared reducer's default is 32 lanes. LFM3 head RMS needs seven
    // stages for 128 lanes; keep this specialization local to this model.
    s.push_str(
        &definitions(
            include_str!("catena/cooperative.hex"),
            &["stdlib.models.coop.reduce-add:"],
        )
        .replace(
            "(def program stdlib.models.coop.reduce-add :",
            "(def program stdlib.lfm3.reduce-head :",
        )
        .replace("const.u32.0x00000005", "const.u32.0x00000007"),
    );
    s.push_str(include_str!("catena/bf16_simt.hex"));
    s.push_str(&definitions(
        include_str!("catena/lfm3_wmma.hex"),
        &[
            "stdlib.lfm3.wmma.m128n64k32.",
            "stdlib.lfm3.wmma.m128n64k64.",
            "stdlib.lfm3.gate.m128n64k32.",
            "stdlib.lfm3.qkvg.m128n64k32.",
        ],
    ));
    s.push_str(&definitions(
        include_str!("catena/lfm3_gemv.hex"),
        &[
            "stdlib.lfm3.gemv.r2:",
            "stdlib.lfm3.gemv.r2.",
            "stdlib.lfm3.gemv.r2wide:",
            "stdlib.lfm3.gemv.r2wide.",
            "stdlib.lfm3.gemv.qkvg",
        ],
    ));
    s.push_str(include_str!("catena/fused_normalization.hex"));
    // LFM's learned normalization multiplies in F32 before the storage boundary.
    s = s
        .replace("([.scaled] f32.to-bf16 [rounded.])", "")
        .replace("([.learned] f32.to-bf16 [learned-bf.])", "")
        .replace(
            "({[.rounded] [.learned-bf]} stdlib.models.memory.bf16.mul [result.])",
            "({[.scaled] [.learned]} stdlib.models.memory.f32.mul f32.to-bf16 [result.])",
        );
    s.push_str(&definitions(
        include_str!("catena/qwen35.hex"),
        &["stdlib.qwen35.norm.", "qwen35.normalize"],
    ));
    s.push_str(&definitions(
        include_str!("catena/qwen35_attention.hex"),
        &[
            "stdlib.qwen35.pow2",
            "stdlib.models.memory.u64.ceil_div",
            "stdlib.qwen35.context.dot",
        ],
    ));
    s.push_str(include_str!("catena/qwen35_batch_wmma.hex"));
    s.push_str(&definitions(
        include_str!("catena/qwen35_batch_projection.hex"),
        &["stdlib.qwen35.batch.project.write"],
    ));
    s.push_str(include_str!("catena/qwen35_prefill.hex"));
    s.push_str(include_str!("catena/state_buffers.hex"));
    s.push_str(include_str!("catena/session_copy.hex"));
    s.push_str(include_str!("catena/fused_argmax.hex"));
    s.push_str(include_str!("catena/lfm3.hex"));
    let attn = include_str!("catena/qwen35_wmma_attention.hex");
    for (name, window) in [("full", 0), ("slide", l.sliding_window)] {
        let mut a = format!(
            "{attn}\n{}",
            include_str!("catena/qwen35_exact_attention.hex")
        );
        // Replace the per-channel Qwen gate with LFM3's per-head gate.
        let old = definitions(&a, &["stdlib.qwen35.wmma.attention.context-write"]);
        a = a.replace(old.trim(), "");
        a = a.replace(
            "stdlib.qwen35.wmma.attention.context-write",
            "stdlib.lfm3.context-write",
        );
        a = a.replace(
            "({[.v4] [.key]} stdlib.models.memory.u64.less [v5.])",
            &format!("({{[.v4] [.key] {}}} stdlib.lfm3.masked [v5.])", c(window)),
        );
        if window > 0 {
            a = bounded_window(a, window, cfg.key_value_size());
        }
        a = a.replace(
            "stdlib.qwen35.wmma.attention.",
            &format!("stdlib.lfm3.attention.{name}."),
        );
        s.push_str(&a);
    }
    for (key, n) in [
        ("HIDDEN_SIZE", cfg.hidden_size),
        ("HEAD_DIM", cfg.head_dim),
        ("KV_HEADS", cfg.num_key_value_heads),
        ("QUERY_SIZE", cfg.query_size()),
        ("QUERY_HEADS", cfg.num_attention_heads),
        ("KEY_VALUE_SIZE", cfg.key_value_size()),
        (
            "QUERY_HEADS_PER_KV_HEAD",
            cfg.num_attention_heads / cfg.num_key_value_heads,
        ),
        ("CONV_KERNEL", l.conv_kernel),
    ] {
        s = s.replace(&format!("{{{{{key}}}}}"), &format!("{n:016X}"));
    }
    s = s
        .replace(
            "{{RMS_EPSILON_BITS}}",
            &format!("{:08X}", cfg.rms_norm_eps.to_bits()),
        )
        .replace(
            "{{HEAD_F32}}",
            &format!("{:08X}", (cfg.head_dim as f32).to_bits()),
        )
        .replace(
            "{{HIDDEN_F32}}",
            &format!("{:08X}", (cfg.hidden_size as f32).to_bits()),
        )
        .replace(
            "{{SCALE_BITS}}",
            &format!("{:08X}", (cfg.head_dim as f32).sqrt().recip().to_bits()),
        );
    for (kind, theta) in [("full", cfg.rope_theta), ("slide", l.sliding_rope_theta)] {
        let mut cc = cfg.clone();
        cc.rope_theta = theta;
        s.push_str(
            &crate::catena::rope_frequency_program(&cc)
                .replace("qwen3.rope-frequency", &format!("lfm3.frequency.{kind}")),
        );
    }
    // Dynamic prefill size chooses SIMT for a single row, WMMA for a batch.
    emit_projection(&mut s, options.wmma_prefill);
    let mut fast = String::new();
    emit_projection(&mut fast, options.wmma_prefill);
    fast = fast
        .replace("lfm3.project", "lfm3.project.fast")
        .replace(
            "stdlib.qwen35.batch.wmma-project",
            "stdlib.lfm3.wmma.m128n64k32.linear-or.native",
        )
        .replace(
            "[.x] [.w] [.features] [.outputs] } stdlib.lfm3.wmma",
            "[.x] [.w] [.rows] [.features] [.outputs] } stdlib.lfm3.wmma",
        )
        .replace(
            "[.x] [.w] [.rows] [.features] [.outputs] } stdlib.models.simt.or",
            "[.x] [.w] [.features] [.outputs] } stdlib.lfm3.gemv.r2",
        );
    let wide = fast
        .replace("lfm3.project.fast", "lfm3.project.wide")
        .replace("stdlib.lfm3.gemv.r2", "stdlib.lfm3.gemv.r2wide");
    let wide64 = wide
        .replace("lfm3.project.wide", "lfm3.project.wide-k64")
        .replace("m128n64k32", "m128n64k64");
    s.push_str(&fast);
    s.push_str(&wide);
    s.push_str(&wide64);
    emit_attention_dispatch(&mut s, cfg, options.wmma_prefill);
    emit_convolution(&mut s, cfg, options.wmma_prefill);
    emit_gate(&mut s, cfg, options.wmma_prefill);
    emit_qkvg(&mut s, cfg, options.wmma_prefill);
    let specs = weight_specs(cfg)?;
    let weight = |name: &str| {
        format!(
            "w{}",
            specs.iter().position(|x| x.gguf_name == name).unwrap()
        )
    };
    let attention_count = l.kv_heads.iter().filter(|&&n| n > 0).count();
    let convolution_count = cfg.num_layers - attention_count;
    let history_count = convolution_count * (l.conv_kernel - 1) * cfg.hidden_size;
    header(
        &mut s,
        "qwen3.init",
        &[("capacity", U)],
        &format!("{{{OWN} {OWN}}}"),
    );
    let mut e = Emit { s: &mut s, next: 0 };
    let count = e.op(
        &["capacity", &c(attention_count * 2 * cfg.key_value_size())],
        "stdlib.models.memory.u64.mul",
    );
    let cache = e.op(&[&count], "stdlib.qwen3.state-zero-bf16");
    let history = e.op(&[&c(history_count)], "stdlib.qwen3.state-zero-f32");
    writeln!(e.s, "  {{[.{cache}] [.{history}]}}\n ))").unwrap();

    let names = (0..specs.len())
        .map(|i| format!("w{i}"))
        .collect::<Vec<_>>();
    let mut args = vec![
        ("tokens", REF),
        ("cache", OWN),
        ("history", OWN),
        ("capacity", U),
        ("start", U),
    ];
    args.extend(names.iter().map(|n| (n.as_str(), REF)));
    header(
        &mut s,
        "qwen3",
        &args,
        &format!("{{{OWN} {OWN} {OWN} {OWN}}}"),
    );
    let mut e = Emit { s: &mut s, next: 0 };
    let rows = e.op(&["tokens"], "stdlib.models.memory.ref-count-u64");
    let mut hidden = e.op(&["tokens", "w0", &c(0)], "stdlib.qwen35.prefill.embed");
    let mut cache = "cache".to_owned();
    let mut history = "history".to_owned();
    let mut ai = 0;
    let mut ci = 0;
    let rope_weight = weight("rope_freqs.weight");
    let mut prepared = e.norm(
        &hidden,
        &weight("blk.0.attn_norm.weight"),
        cfg.hidden_size,
        &rows,
    );
    for layer in 0..cfg.num_layers {
        let w = |name: &str| weight(&format!("blk.{layer}.{name}.weight"));
        let input = prepared;
        let operator = if l.kv_heads[layer] == 0 {
            let base = c(ci * (l.conv_kernel - 1) * cfg.hidden_size);
            ci += 1;
            let state = e.op(
                &[
                    &input,
                    &w("shortconv.in_proj"),
                    &w("shortconv.conv"),
                    &history,
                    &base,
                    &rows,
                ],
                "lfm3.convolution",
            );
            let hname = format!("history{}", e.next);
            let conv = format!("conv{}", e.next);
            e.next += 1;
            writeln!(e.s, "  ([.{state}] *.elim [{hname} {conv}.])").unwrap();
            history = hname;
            let out = e.project(
                &conv,
                &w("shortconv.out_proj"),
                cfg.hidden_size,
                cfg.hidden_size,
            );
            e.free(&conv, &out, "bf16")
        } else {
            let grouped = e.op(
                &[
                    &input,
                    &w("attn_q"),
                    &w("attn_k"),
                    &w("attn_v"),
                    &w("attn_gate"),
                    &rows,
                ],
                "lfm3.qkvg",
            );
            let base = e.op(
                &["capacity", &c(ai * 2 * cfg.key_value_size())],
                "stdlib.models.memory.u64.mul",
            );
            let stride = e.op(
                &["capacity", &c(cfg.key_value_size())],
                "stdlib.models.memory.u64.mul",
            );
            let vb = e.op(&[&base, &stride], "stdlib.models.memory.u64.add");
            ai += 1;
            let prepared = e.op(
                &[
                    &grouped,
                    &w("attn_q_norm"),
                    &w("attn_k_norm"),
                    &rope_weight,
                    &cache,
                    &base,
                    &vb,
                    "start",
                    &c(usize::from(l.sliding_layers[layer])),
                    &rows,
                ],
                "stdlib.lfm3.prepare",
            );
            let nextcache = format!("cache{}", e.next);
            let tail = format!("prep{}", e.next);
            let q = format!("query{}", e.next);
            let gate = format!("gate{}", e.next);
            e.next += 1;
            writeln!(
                e.s,
                "  ([.{prepared}] *.elim [{nextcache} {tail}.])\n  ([.{tail}] *.elim [{q} {gate}.])"
            )
            .unwrap();
            cache = nextcache;
            let kind = if l.sliding_layers[layer] {
                "slide"
            } else {
                "full"
            };
            let context = e.op(
                &[&q, &cache, &gate, "start", &base, &vb],
                &format!("lfm3.attention.{kind}"),
            );
            let context = e.free(&grouped, &context, "bf16");
            let context = e.free(&q, &context, "bf16");
            let context = e.free(&gate, &context, "bf16");
            let out = e.project(
                &context,
                &w("attn_output"),
                cfg.query_size(),
                cfg.hidden_size,
            );
            e.free(&context, &out, "bf16")
        };
        let operator = e.free(&input, &operator, "bf16");
        let (residual, ffn) = e.normres(
            &operator,
            &hidden,
            &w("post_attention_norm"),
            &w("ffn_norm"),
        );
        let gated = e.op(&[&ffn, &w("ffn_gate"), &w("ffn_up"), &rows], "lfm3.gate");
        let gated = e.free(&ffn, &gated, "bf16");
        let down = e.project(
            &gated,
            &w("ffn_down"),
            cfg.intermediate_size,
            cfg.hidden_size,
        );
        let down = e.free(&gated, &down, "bf16");
        let next_weight = if layer + 1 < cfg.num_layers {
            weight(&format!("blk.{}.attn_norm.weight", layer + 1))
        } else {
            weight("output_norm.weight")
        };
        (hidden, prepared) = e.normres(&down, &residual, &w("post_ffw_norm"), &next_weight);
    }
    let last = e.op(&[&rows, &c(1)], "stdlib.models.memory.u64.sub");
    let norm = e.op(&[&prepared, &last], "stdlib.qwen35.prefill.row");
    let norm = e.free(&prepared, &norm, "bf16");
    let norm = e.free(&hidden, &norm, "bf16");
    let logits = e.project(
        &norm,
        names.last().unwrap(),
        cfg.hidden_size,
        cfg.vocabulary_size,
    );
    let logits = e.free(&norm, &logits, "bf16");
    let token = e.op(&[&logits], "qwen3.fused.argmax");
    cache = e.after(&cache, &token);
    history = e.after(&history, &token);
    writeln!(
        e.s,
        "  {{[.{cache}] [.{history}] [.{logits}] [.{token}]}}\n ))"
    )
    .unwrap();
    ensure!(
        !s.split("{{")
            .skip(1)
            .any(|x| x.chars().next().is_some_and(|c| c.is_ascii_uppercase())),
        "unresolved LFM3 source constant"
    );
    ensure!(
        !s.contains("bf16x2.dot-f32"),
        "packed BF16 dot in LFM3 source"
    );
    Ok(s)
}

/// Add an embedding-input entrypoint while retaining the ordinary text graph.
/// Image and text rows share the same linear RoPE positions and decoder state.
pub(crate) fn multimodal_source(cfg: &ModelConfig, options: InferenceOptions) -> Result<String> {
    let mut s = source(cfg, options)?;
    let original = definitions(&s, &["qwen3:"]);
    let mut args = vec![
        ("hidden", OWN),
        ("positions", REF),
        ("cache", OWN),
        ("history", OWN),
        ("capacity", U),
        ("start", U),
        ("sequence", U),
    ];
    let specs = weight_specs(cfg)?;
    let names = (0..specs.len())
        .map(|i| format!("w{i}"))
        .collect::<Vec<_>>();
    args.extend(names.iter().map(|n| (n.as_str(), REF)));
    let mut entry = String::new();
    header(
        &mut entry,
        "qwen3.multimodal",
        &args,
        &format!("{{{OWN} {OWN} {OWN} {OWN}}}"),
    );
    let mut body = original[original.find("  ({").unwrap()..].to_owned();
    let mut row_call = String::new();
    call(
        &mut row_call,
        &["tokens"],
        "stdlib.models.memory.ref-count-u64",
        "v0",
    );
    let mut embed_call = String::new();
    call(
        &mut embed_call,
        &["tokens", "w0", &c(0)],
        "stdlib.qwen35.prefill.embed",
        "v1",
    );
    ensure!(
        body.contains(&row_call) && body.contains(&embed_call),
        "LFM3 embedding entrypoint template changed"
    );
    let rows = format!(
        "  ([.hidden] stdlib.models.memory.own-count-bf16 [hidden-count.])\n  ({{[.hidden-count] {}}} stdlib.models.memory.u64.div [v0.])\n",
        c(cfg.hidden_size)
    );
    body = body
        .replace(&row_call, &rows)
        .replace(&embed_call, "  ([.hidden] [v1.])\n");
    entry.push_str(&body);
    s.push_str(&entry);
    header(
        &mut s,
        "qwen3.embedding",
        &[("tokens", REF), ("weight", REF)],
        OWN,
    );
    call(
        &mut s,
        &["tokens", "weight", &c(0)],
        "stdlib.qwen35.prefill.embed",
        "embedding",
    );
    s.push_str("  [.embedding]\n ))\n");
    s.push_str(include_str!("catena/lfm3_multimodal.hex"));
    Ok(s)
}

fn emit_projection(s: &mut String, wmma: bool) {
    for (branch, proof, op) in [
        (
            "batch",
            "((master.predicate prop.IsTrue) |-)",
            "stdlib.qwen35.batch.wmma-project",
        ),
        (
            "single",
            "(({(master.predicate prop.IsTrue)} stdlib.prelude.prop.Not) |-)",
            "stdlib.models.simt.or",
        ),
    ] {
        let p = format!("({{{U} ({{{proof} 1}} *)}} *)");
        header(
            s,
            &format!("stdlib.lfm3.project.{branch}"),
            &[
                ("x", OWN),
                ("w", REF),
                ("features", U),
                ("outputs", U),
                ("rows", U),
                ("pair", &p),
            ],
            OWN,
        );
        s.push_str("  ([.pair] *.elim [ignored proofs.])\n  ([.proofs] *.elim [proof unit.])\n  ([.proof] unsafe.discard unit.elim)\n  ([.unit] unit.elim)\n");
        if branch == "batch" {
            call(s, &["x", "w", "features", "outputs"], op, "result");
        } else {
            call(s, &["x", "w", "rows", "features", "outputs"], op, "result");
        }
        s.push_str("  [.result]\n ))\n");
    }
    header(
        s,
        "lfm3.project",
        &[("x", OWN), ("w", REF), ("features", U), ("outputs", U)],
        OWN,
    );
    s.push_str("  ([.x] stdlib.models.memory.own-count-bf16 [count.])\n  ({[.count] [.features]} stdlib.models.memory.u64.div [rows.])\n");
    if wmma {
        s.push_str("  ({(const.u64.0x0000000000000001) [.rows]} stdlib.models.memory.u64.less [batch.])\n  ({[.batch] (unit.intro meta.master.predicate)} smolcat.value.named_at :.forget [condition.])\n  ({[.x] [.w] [.features] [.outputs] [.rows]} partial.stdlib.lfm3.project.batch.5 [yes.])\n  ({[.x] [.w] [.features] [.outputs] [.rows]} partial.stdlib.lfm3.project.single.5 [no.])\n  ({[.condition] (const.u64.0x0000000000000000) [.yes] [.no] unit.intro} core.if)\n ))\n");
    } else {
        s.push_str("  ({[.x] [.w] [.rows] [.features] [.outputs]} stdlib.models.simt.or)\n ))\n");
    }
}

fn emit_attention_dispatch(s: &mut String, cfg: &ModelConfig, wmma: bool) {
    let args = [
        ("query", OWN),
        ("cache", OWN),
        ("gate", OWN),
        ("start", U),
        ("keybase", U),
        ("valuebase", U),
    ];
    let values = args.iter().map(|(n, _)| *n).collect::<Vec<_>>();
    for kind in ["full", "slide"] {
        for (branch, proof, target) in [
            ("batch", "((master.predicate prop.IsTrue) |-)", "run"),
            (
                "single",
                "(({(master.predicate prop.IsTrue)} stdlib.prelude.prop.Not) |-)",
                "run-safe",
            ),
        ] {
            let pair = format!("({{{U} ({{{proof} 1}} *)}} *)");
            let mut inputs = args.to_vec();
            inputs.push(("pair", &pair));
            header(
                s,
                &format!("stdlib.lfm3.dispatch.{kind}.{branch}"),
                &inputs,
                OWN,
            );
            s.push_str("  ([.pair] *.elim [ignored proofs.])\n  ([.proofs] *.elim [proof unit.])\n  ([.proof] unsafe.discard unit.elim)\n  ([.unit] unit.elim)\n");
            call(
                s,
                &values,
                &format!("stdlib.lfm3.attention.{kind}.{target}"),
                "result",
            );
            s.push_str("  [.result]\n ))\n");
        }
        header(s, &format!("lfm3.attention.{kind}"), &args, OWN);
        if wmma {
            s.push_str("  ([.query] stdlib.models.memory.own-count-bf16 [count.])\n");
            call(
                s,
                &[&c(cfg.query_size()), "count"],
                "stdlib.models.memory.u64.less",
                "batch",
            );
            s.push_str("  ({[.batch] (unit.intro meta.master.predicate)} smolcat.value.named_at :.forget [condition.])\n");
            call(
                s,
                &values,
                &format!("partial.stdlib.lfm3.dispatch.{kind}.batch.6"),
                "yes",
            );
            call(
                s,
                &values,
                &format!("partial.stdlib.lfm3.dispatch.{kind}.single.6"),
                "no",
            );
            s.push_str("  ({[.condition] (const.u64.0x0000000000000000) [.yes] [.no] unit.intro} core.if)\n ))\n");
        } else {
            call(
                s,
                &values,
                &format!("stdlib.lfm3.attention.{kind}.run-safe"),
                "result",
            );
            s.push_str("  [.result]\n ))\n");
        }
    }
}

fn emit_convolution(s: &mut String, cfg: &ModelConfig, fast: bool) {
    let args = [
        ("input", OWN),
        ("weight", REF),
        ("convweight", REF),
        ("history", OWN),
        ("base", U),
        ("rows", U),
    ];
    let result = format!("({{{OWN} {OWN}}} *)");
    header(s, "stdlib.lfm3.conv-prefill", &args, &result);
    let mut e = Emit { s, next: 0 };
    let count = e.op(
        &["rows", &c(cfg.hidden_size)],
        "stdlib.models.memory.u64.mul",
    );
    let bcx = e.project("input", "weight", cfg.hidden_size, 3 * cfg.hidden_size);
    let bx = e.element(&[&bcx], "bx", &count);
    let conv = e.element(
        &[&bx, &bcx, "history", "convweight", "base", "rows"],
        "conv",
        &count,
    );
    let history = e.after("history", &conv);
    let tail = e.element(
        &[&bx, &history, "base", "rows"],
        "tail",
        &c(6 * cfg.hidden_size),
    );
    let history = e.op(
        &[&tail, "base", &history, &c(6 * cfg.hidden_size)],
        "stdlib.lfm3.persist.into",
    );
    let history = e.free(&tail, &history, "f32");
    let conv = e.after(&conv, &history);
    let conv = e.free(&bx, &conv, "f32");
    let conv = e.free(&bcx, &conv, "bf16");
    let pair = e.op(&[&history, &conv], "*.intro");
    writeln!(e.s, "  [.{pair}]\n ))").unwrap();
    let names = args.iter().map(|(n, _)| *n).collect::<Vec<_>>();
    for (branch, proof) in [
        ("single", "((master.predicate prop.IsTrue) |-)"),
        (
            "batch",
            "(({(master.predicate prop.IsTrue)} stdlib.prelude.prop.Not) |-)",
        ),
    ] {
        let p = format!("({{{U} ({{{proof} 1}} *)}} *)");
        let mut inputs = args.to_vec();
        inputs.push(("pair", &p));
        header(
            s,
            &format!("stdlib.lfm3.convolution.{branch}"),
            &inputs,
            &result,
        );
        s.push_str("  ([.pair] *.elim [ignored proofs.])\n  ([.proofs] *.elim [proof unit.])\n  ([.proof] unsafe.discard unit.elim)\n  ([.unit] unit.elim)\n");
        if branch == "single" {
            call(s, &names[..5], "stdlib.lfm3.convdecode", "result");
        } else {
            call(s, &names, "stdlib.lfm3.conv-prefill", "result");
        }
        s.push_str("  [.result]\n ))\n");
    }
    header(s, "lfm3.convolution", &args, &result);
    if fast {
        call(
            s,
            &["rows", &c(1)],
            "stdlib.models.memory.u64.equal",
            "single",
        );
        s.push_str("  ({[.single] (unit.intro meta.master.predicate)} smolcat.value.named_at :.forget [condition.])\n");
        call(s, &names, "partial.stdlib.lfm3.convolution.single.6", "yes");
        call(s, &names, "partial.stdlib.lfm3.convolution.batch.6", "no");
        s.push_str("  ({[.condition] (const.u64.0x0000000000000000) [.yes] [.no] unit.intro} core.if)\n ))\n");
    } else {
        call(s, &names, "stdlib.lfm3.conv-prefill", "result");
        s.push_str("  [.result]\n ))\n");
    }
}

/// Limit sliding attention's key traversal, not just the score mask. Cache
/// addresses stay absolute; each query tile gets a local contiguous key range.
fn bounded_window(mut source: String, window: usize, kv: usize) -> String {
    let prefix = "stdlib.qwen35.wmma.attention.";
    let begin = |query: &str| {
        let mut s = String::new();
        call(
            &mut s,
            &["start", query],
            "stdlib.models.memory.u64.add",
            "window-first",
        );
        call(
            &mut s,
            &["window-first", &c(1)],
            "stdlib.models.memory.u64.add",
            "window-end",
        );
        call(
            &mut s,
            &["window-end", &c(window)],
            "stdlib.models.memory.u64.less",
            "window-short",
        );
        call(
            &mut s,
            &["window-end", &c(window)],
            "stdlib.models.memory.u64.sub",
            "window-raw",
        );
        call(
            &mut s,
            &["window-short", &c(0), "window-raw"],
            "stdlib.models.memory.select-u64",
            "window-base",
        );
        s
    };
    for name in ["tile", "tile-exact", "tile-safe"] {
        let old = definitions(&source, &[&format!("{prefix}{name}:")]);
        if old.is_empty() {
            continue;
        }
        let mut shifted = begin("v0");
        call(
            &mut shifted,
            &["global-end", "window-base"],
            "stdlib.models.memory.u64.sub",
            "v5",
        );
        call(
            &mut shifted,
            &["window-base", &c(kv)],
            "stdlib.models.memory.u64.mul",
            "window-offset",
        );
        call(
            &mut shifted,
            &["key-base", "window-offset"],
            "stdlib.models.memory.u64.add",
            "window-key-base",
        );
        call(
            &mut shifted,
            &["value-base", "window-offset"],
            "stdlib.models.memory.u64.add",
            "window-value-base",
        );
        let new = old
            .replace(
                "({[.start] [.v4]} stdlib.models.memory.u64.add [v5.])",
                &format!(
                    "({{[.start] [.v4]}} stdlib.models.memory.u64.add [global-end.])\n{shifted}"
                ),
            )
            .replace(
                "[.start] [.key-base]} stdlib.",
                "[.start] [.window-key-base]} stdlib.",
            )
            .replace(
                "[.v5] [.value-base]} stdlib.",
                "[.v5] [.window-value-base]} stdlib.",
            );
        source = source.replace(old.trim(), new.trim());
    }
    for name in ["run", "run-exact", "run-safe"] {
        let old = definitions(&source, &[&format!("{prefix}{name}:")]);
        if old.is_empty() {
            continue;
        }
        let mut bounded = String::new();
        call(
            &mut bounded,
            &["global-count", &c(window + 63)],
            "stdlib.models.memory.u64.less",
            "window-short",
        );
        call(
            &mut bounded,
            &["window-short", "global-count", &c(window + 63)],
            "stdlib.models.memory.select-u64",
            "v2",
        );
        let new = old.replace(
            "({[.start] [.v1]} stdlib.models.memory.u64.add [v2.])",
            &format!(
                "({{[.start] [.v1]}} stdlib.models.memory.u64.add [global-count.])\n{bounded}"
            ),
        );
        source = source.replace(old.trim(), new.trim());
    }
    for (name, query, key, position, predicate) in [
        ("score-write", "first", "key", "v4", "v5"),
        ("qk-exact-one", "first", "v0", "v15", "v16"),
    ] {
        let old = definitions(&source, &[&format!("{prefix}{name}:")]);
        let mut shifted = begin(query);
        call(
            &mut shifted,
            &[key, "window-base"],
            "stdlib.models.memory.u64.add",
            "window-key",
        );
        let from = if name == "score-write" {
            format!(
                "({{[.{position}] [.{key}] {}}} stdlib.lfm3.masked [{predicate}.])",
                c(window)
            )
        } else {
            format!("({{[.{position}] [.{key}]}} stdlib.models.memory.u64.less [{predicate}.])")
        };
        let to = if name == "score-write" {
            format!(
                "{shifted}({{[.{position}] [.window-key] {}}} stdlib.lfm3.masked [{predicate}.])",
                c(window)
            )
        } else {
            format!(
                "{shifted}({{[.{position}] [.window-key]}} stdlib.models.memory.u64.less [{predicate}.])"
            )
        };
        assert!(old.contains(&from), "missing sliding attention site {name}");
        source = source.replace(old.trim(), old.replace(&from, &to).trim());
    }
    source
}

fn emit_gate(s: &mut String, cfg: &ModelConfig, fast: bool) {
    let args = [("input", OWN), ("gate", REF), ("up", REF), ("rows", U)];
    for (branch, proof) in [
        ("batch", "((master.predicate prop.IsTrue) |-)"),
        (
            "single",
            "(({(master.predicate prop.IsTrue)} stdlib.prelude.prop.Not) |-)",
        ),
    ] {
        let p = format!("({{{U} ({{{proof} 1}} *)}} *)");
        let mut inputs = args.to_vec();
        inputs.push(("pair", &p));
        header(s, &format!("stdlib.lfm3.gate.{branch}"), &inputs, OWN);
        s.push_str("  ([.pair] *.elim [ignored proofs.])\n  ([.proofs] *.elim [proof unit.])\n  ([.proof] unsafe.discard unit.elim)\n  ([.unit] unit.elim)\n");
        if branch == "batch" {
            call(
                s,
                &[
                    "input",
                    "gate",
                    "rows",
                    &c(cfg.hidden_size),
                    &c(cfg.intermediate_size),
                    "up",
                ],
                "stdlib.lfm3.gate.m128n64k32.linear-or.native",
                "result",
            );
        } else {
            // Independent gate/up loads outperform the fused GEMV on Strix Halo.
            let mut e = Emit { s, next: 0 };
            let gate = e.op(
                &[
                    "input",
                    "gate",
                    &c(cfg.hidden_size),
                    &c(cfg.intermediate_size),
                ],
                "stdlib.lfm3.gemv.r2wide",
            );
            let up = e.op(
                &[
                    "input",
                    "up",
                    &c(cfg.hidden_size),
                    &c(cfg.intermediate_size),
                ],
                "stdlib.lfm3.gemv.r2wide",
            );
            let result = e.element(&[&gate, &up], "swiglu", &c(cfg.intermediate_size));
            let result = e.free(&gate, &result, "bf16");
            let result = e.free(&up, &result, "bf16");
            writeln!(e.s, "  [.{result}] [result.]").unwrap();
        }
        s.push_str("  [.result]\n ))\n");
    }
    header(s, "lfm3.gate", &args, OWN);
    if fast {
        call(
            s,
            &[&c(1), "rows"],
            "stdlib.models.memory.u64.less",
            "batch",
        );
        s.push_str("  ({[.batch] (unit.intro meta.master.predicate)} smolcat.value.named_at :.forget [condition.])\n");
        call(
            s,
            &["input", "gate", "up", "rows"],
            "partial.stdlib.lfm3.gate.batch.4",
            "yes",
        );
        call(
            s,
            &["input", "gate", "up", "rows"],
            "partial.stdlib.lfm3.gate.single.4",
            "no",
        );
        s.push_str("  ({[.condition] (const.u64.0x0000000000000000) [.yes] [.no] unit.intro} core.if)\n ))\n");
    } else {
        let mut e = Emit { s, next: 0 };
        let gate = e.op(
            &[
                "input",
                "gate",
                "rows",
                &c(cfg.hidden_size),
                &c(cfg.intermediate_size),
            ],
            "stdlib.models.simt.or",
        );
        let up = e.op(
            &[
                "input",
                "up",
                "rows",
                &c(cfg.hidden_size),
                &c(cfg.intermediate_size),
            ],
            "stdlib.models.simt.or",
        );
        let count = e.op(
            &["rows", &c(cfg.intermediate_size)],
            "stdlib.models.memory.u64.mul",
        );
        let result = e.element(&[&gate, &up], "swiglu", &count);
        let result = e.free(&gate, &result, "bf16");
        let result = e.free(&up, &result, "bf16");
        writeln!(e.s, "  [.{result}]\n ))").unwrap();
    }
}

fn emit_qkvg(s: &mut String, cfg: &ModelConfig, fast: bool) {
    let args = [
        ("input", OWN),
        ("q", REF),
        ("k", REF),
        ("v", REF),
        ("gate", REF),
        ("rows", U),
    ];
    let width = cfg.query_size() + 2 * cfg.key_value_size() + cfg.num_attention_heads;
    for (branch, proof) in [
        ("batch", "((master.predicate prop.IsTrue) |-)"),
        (
            "single",
            "(({(master.predicate prop.IsTrue)} stdlib.prelude.prop.Not) |-)",
        ),
    ] {
        let p = format!("({{{U} ({{{proof} 1}} *)}} *)");
        let mut inputs = args.to_vec();
        inputs.push(("pair", &p));
        header(s, &format!("stdlib.lfm3.qkvg.{branch}"), &inputs, OWN);
        s.push_str("  ([.pair] *.elim [ignored proofs.])\n  ([.proofs] *.elim [proof unit.])\n  ([.proof] unsafe.discard unit.elim)\n  ([.unit] unit.elim)\n");
        if branch == "batch" {
            call(
                s,
                &[
                    "input",
                    "q",
                    "rows",
                    &c(cfg.hidden_size),
                    &c(width),
                    "k",
                    "v",
                    "gate",
                ],
                "stdlib.lfm3.qkvg.m128n64k32.linear-or.native",
                "result",
            );
        } else {
            call(
                s,
                &[
                    "input",
                    "q",
                    &c(cfg.hidden_size),
                    &c(width),
                    "k",
                    "v",
                    "gate",
                ],
                "stdlib.lfm3.gemv.qkvg",
                "result",
            );
        }
        s.push_str("  [.result]\n ))\n");
    }
    header(s, "lfm3.qkvg", &args, OWN);
    if fast {
        call(
            s,
            &[&c(1), "rows"],
            "stdlib.models.memory.u64.less",
            "batch",
        );
        s.push_str("  ({[.batch] (unit.intro meta.master.predicate)} smolcat.value.named_at :.forget [condition.])\n");
        call(
            s,
            &["input", "q", "k", "v", "gate", "rows"],
            "partial.stdlib.lfm3.qkvg.batch.6",
            "yes",
        );
        call(
            s,
            &["input", "q", "k", "v", "gate", "rows"],
            "partial.stdlib.lfm3.qkvg.single.6",
            "no",
        );
        s.push_str("  ({[.condition] (const.u64.0x0000000000000000) [.yes] [.no] unit.intro} core.if)\n ))\n");
    } else {
        let mut e = Emit { s, next: 0 };
        let mut parts = Vec::new();
        for (name, n) in [
            ("q", cfg.query_size()),
            ("k", cfg.key_value_size()),
            ("v", cfg.key_value_size()),
            ("gate", cfg.num_attention_heads),
        ] {
            parts.push(e.op(
                &["input", name, "rows", &c(cfg.hidden_size), &c(n)],
                "stdlib.models.simt.or",
            ));
        }
        let count = e.op(&["rows", &c(width)], "stdlib.models.memory.u64.mul");
        let mut result = e.element(
            &parts.iter().map(String::as_str).collect::<Vec<_>>(),
            "pack-qkvg",
            &count,
        );
        for part in parts {
            result = e.free(&part, &result, "bf16");
        }
        writeln!(e.s, "  [.{result}]\n ))").unwrap();
    }
}
