//! Batched prefill composition; runtime execution and allocation remain in Hex.
use super::*;

fn fold_result(s: &mut String) {
    s.push_str("  ({[.result] [.truth]} *.intro)");
    finish(s);
}
fn batch_norm(s: &mut String, input: &str, weight: &str, output: &str, c0: &ModelConfig) {
    let width = c0.hidden_size;
    call(
        s,
        &[
            input,
            weight,
            &c(width),
            &floats(width),
            "sequence",
            &c(width.next_power_of_two().trailing_zeros() as usize),
            &c(width.next_power_of_two()),
        ],
        "qwen35.normalize",
        output,
    );
}
fn finish_state(s: &mut String, cache: &str, shadow: &str, hidden: &str) {
    s.push_str(&format!("  ([.{hidden}] meta.master.complete-own [completed phase.])\n  ({{[.{cache}] [.phase]}} meta.master.after-own [finished-cache.])\n  ({{[.{shadow}] [.phase]}} meta.master.after-own [finished-shadow.])\n"));
    pack(
        s,
        &names(&["finished-cache", "finished-shadow", "completed"]),
        "result",
    );
    s.push_str("  [.result]");
    finish(s);
}

pub(super) fn emit(
    s: &mut String,
    cfg: &ModelConfig,
    full_tree: &str,
    linear_tree: &str,
    wmma: WmmaStages,
) {
    let source_start = s.len();
    let moe = cfg.architecture == ModelArchitecture::Qwen35Moe;
    let dense_blocks = (!moe || wmma.fp8) && wmma.projections;
    let dense_scan = dense_blocks && cfg.linear_inner_size.unwrap().is_multiple_of(16);
    let project = if dense_blocks {
        "stdlib.qwen35.dense.project"
    } else if wmma.projections {
        "stdlib.qwen35.batch.wmma-project"
    } else {
        "stdlib.qwen35.batch.project"
    };
    let moe_ffn = if wmma.fp8 {
        "stdlib.qwen35.fp8.moe.ffn"
    } else if wmma.safe_experts {
        "stdlib.qwen35.wmma.moe.ffn-safe"
    } else if wmma.exact_gate {
        "stdlib.qwen35.wmma.moe.ffn-exact"
    } else if wmma.experts {
        "stdlib.qwen35.wmma.moe.ffn"
    } else {
        "stdlib.qwen35.batch.moe.ffn"
    };
    let attention = if wmma.safe_scores && cfg.head_dim.is_multiple_of(16) {
        "stdlib.qwen35.wmma.attention.run-safe"
    } else if wmma.exact_context && cfg.head_dim.is_multiple_of(16) {
        "stdlib.qwen35.wmma.attention.run-exact"
    } else if wmma.attention && cfg.head_dim.is_multiple_of(16) {
        "stdlib.qwen35.wmma.attention.run"
    } else {
        "stdlib.qwen35.batch.staged"
    };
    let fcount = (if moe { 16 } else { 11 })
        + if wmma.fp8 {
            2 * super::super::fp8_projection_indices(moe, false).len()
        } else {
            0
        };
    let lcount = (if moe { 19 } else { 14 })
        + if wmma.fp8 {
            2 * super::super::fp8_projection_indices(moe, true).len()
        } else {
            0
        };
    let fg = nested_product(&vec![REF.to_owned(); fcount]);
    let lg = nested_product(&vec![REF.to_owned(); lcount]);
    s.push_str(
        &include_str!("../catena/qwen35_prefill.hex")
            .replace("{{HIDDEN_SIZE}}", &format!("{:016X}", cfg.hidden_size)),
    );
    let scratch = if moe { SCRATCH.to_vec() } else { vec![] };
    let mut common = vec![
        ("capacity", U64),
        ("start", U64),
        ("layer", U64),
        ("sequence", U64),
    ];
    common.extend(scratch.iter().map(|n| (*n, OWN)));
    let cn = common.iter().map(|(n, _)| *n).collect::<Vec<_>>();
    // Batch full-layer projections/preparation and FFNs; preserve staged BF16 attention.
    let mut args = vec![("weights", fg.as_str())];
    args.extend(common.clone());
    args.push(("state", TRIPLE));
    header(s, "stdlib.qwen35.batch.full", &args, TRIPLE);
    unpack(s, "state", &names(&["cache", "shadow", "hidden"]));
    let fw = (0..fcount).map(|i| format!("w{i}")).collect::<Vec<_>>();
    unpack(s, "weights", &fw);
    batch_norm(s, "hidden", "w0", "normalized", cfg);
    for (out, weight, size) in [
        ("qg", "w1", cfg.query_size() * 2),
        ("key", "w2", cfg.key_value_size()),
        ("value", "w3", cfg.key_value_size()),
    ] {
        call(
            s,
            &["normalized", weight, &c(cfg.hidden_size), &c(size)],
            project,
            out,
        );
    }
    call(
        s,
        &["layer", "capacity"],
        "stdlib.models.memory.u64.mul",
        "layer-capacity",
    );
    call(
        s,
        &["layer-capacity", &c(cfg.key_value_size() * 2)],
        "stdlib.models.memory.u64.mul",
        "key-base",
    );
    call(
        s,
        &["capacity", &c(cfg.key_value_size())],
        "stdlib.models.memory.u64.mul",
        "value-offset",
    );
    call(
        s,
        &["key-base", "value-offset"],
        "stdlib.models.memory.u64.add",
        "value-base",
    );
    call(
        s,
        &["start", &c(cfg.key_value_size())],
        "stdlib.models.memory.u64.mul",
        "start-offset",
    );
    call(
        s,
        &["key-base", "start-offset"],
        "stdlib.models.memory.u64.add",
        "key-target",
    );
    call(
        s,
        &["value-base", "start-offset"],
        "stdlib.models.memory.u64.add",
        "value-target",
    );
    call(
        s,
        &[
            "qg",
            "key",
            "value",
            "start",
            "cache",
            "w5",
            "w6",
            "key-target",
            "value-target",
        ],
        "stdlib.qwen35.batch.full-prepare",
        "query cache-ready",
    );
    call(
        s,
        &[
            "query",
            "cache-ready",
            "qg",
            "start",
            "key-base",
            "value-base",
        ],
        attention,
        "context",
    );
    let mut ready = "context".to_owned();
    for old in ["normalized", "qg", "key", "value", "query"] {
        let next = format!("context-after-{old}");
        free(s, old, &ready, &next);
        ready = next;
    }
    call(
        s,
        &[
            &ready,
            "w4",
            &c(cfg.query_size()),
            &c(cfg.hidden_size),
            "hidden",
        ],
        if dense_blocks && cfg.query_size().is_multiple_of(16) {
            "stdlib.qwen35.dense.projection.residual"
        } else if wmma.projections && cfg.query_size().is_multiple_of(16) {
            "stdlib.qwen35.wmma.projection.residual"
        } else {
            "stdlib.qwen35.batch.residual"
        },
        "attention",
    );
    free(s, &ready, "attention", "attention-after-context");
    free(s, "hidden", "attention-after-context", "attention-ready");
    batch_norm(s, "attention-ready", "w7", "ffn-input", cfg);
    if moe {
        let mut args = names(&["ffn-input", "attention-ready"]);
        args.extend(fw[8..16].iter().cloned());
        if wmma.fp8 {
            args.extend(fw[24..].iter().cloned());
        }
        let refs = args.iter().map(String::as_str).collect::<Vec<_>>();
        call(s, &refs, moe_ffn, "final-hidden");
    } else {
        call(
            s,
            &[
                "ffn-input",
                "w8",
                &c(cfg.hidden_size),
                &c(cfg.intermediate_size),
                "w9",
            ],
            if dense_blocks {
                "stdlib.qwen35.dense.projection.gate"
            } else {
                "stdlib.qwen35.batch.gate"
            },
            "active",
        );
        free(s, "ffn-input", "active", "active-ready");
        call(
            s,
            &[
                "active-ready",
                "w10",
                &c(cfg.intermediate_size),
                &c(cfg.hidden_size),
                "attention-ready",
            ],
            if dense_blocks {
                "stdlib.qwen35.dense.projection.residual"
            } else if wmma.projections {
                "stdlib.qwen35.wmma.projection.residual"
            } else {
                "stdlib.qwen35.batch.residual"
            },
            "output",
        );
        free(s, "active-ready", "output", "output-ready");
        free(s, "attention-ready", "output-ready", "final-hidden");
    }
    finish_state(s, "cache-ready", "shadow", "final-hidden");
    let mut args = vec![("weights", lg.as_str())];
    args.extend(common.clone());
    args.push(("state", TRIPLE));
    header(s, "stdlib.qwen35.batch.linear", &args, TRIPLE);
    unpack(s, "state", &names(&["cache", "shadow", "hidden"]));
    let lw = (0..lcount).map(|i| format!("w{i}")).collect::<Vec<_>>();
    unpack(s, "weights", &lw);
    batch_norm(s, "hidden", "w0", "normalized", cfg);
    for (out, w, size) in [
        ("qkv", "w1", cfg.linear_qkv_size().unwrap()),
        ("z", "w2", cfg.linear_inner_size.unwrap()),
        ("alpha", "w3", cfg.linear_time_step_rank.unwrap()),
        ("beta", "w4", cfg.linear_time_step_rank.unwrap()),
    ] {
        call(
            s,
            &["normalized", w, &c(cfg.hidden_size), &c(size)],
            if wmma.fp8 && cfg.hidden_size < 16 && matches!(out, "alpha" | "beta") {
                "stdlib.qwen35.batch.project"
            } else {
                project
            },
            out,
        );
    }
    let recurrent = cfg.linear_state_size.unwrap() * cfg.linear_inner_size.unwrap();
    let conv = 3 * cfg.linear_qkv_size().unwrap();
    call(
        s,
        &["capacity", &c(cfg.kv_cache_elements_per_token().unwrap())],
        "stdlib.models.memory.u64.mul",
        "kv-end",
    );
    call(
        s,
        &["layer", &c(recurrent + conv)],
        "stdlib.models.memory.u64.mul",
        "layer-offset",
    );
    call(
        s,
        &["kv-end", "layer-offset"],
        "stdlib.models.memory.u64.add",
        "conv-base",
    );
    call(
        s,
        &["layer", &c(recurrent)],
        "stdlib.models.memory.u64.mul",
        "shadow-base",
    );
    call(
        s,
        &["qkv", "cache", "w9", "conv-base", "start"],
        "stdlib.qwen35.batch.prepare",
        "convolved query key",
    );
    s.push_str("  ([.convolved] meta.master.complete-own [conv-ready conv-phase.])\n  ({[.cache] [.conv-phase]} meta.master.after-own [cache-after-read.])\n");
    call(
        s,
        &["qkv", "cache-after-read", "conv-base", "start"],
        "stdlib.qwen35.batch.history.run",
        "history",
    );
    s.push_str("  ([.history] meta.master.complete-own [history-ready history-phase.])\n  ({[.conv-ready] [.history-phase]} meta.master.after-own [prepared.])\n");
    call(
        s,
        &[
            "prepared",
            "query",
            "key",
            "alpha",
            "beta",
            "w5",
            "w6",
            "shadow",
            "shadow-base",
        ],
        if dense_scan {
            "stdlib.qwen35.dense.scan-l8"
        } else {
            "stdlib.qwen35.batch.scan"
        },
        "core",
    );
    call(
        s,
        &["core", "z", "w7"],
        "stdlib.qwen35.batch.gated",
        "context",
    );
    let mut ready = "context".to_owned();
    for old in [
        "normalized",
        "qkv",
        "z",
        "alpha",
        "beta",
        "prepared",
        "query",
        "key",
        "core",
    ] {
        let next = format!("context-after-{old}");
        free(s, old, &ready, &next);
        ready = next;
    }
    call(
        s,
        &[
            &ready,
            "w8",
            &c(cfg.linear_inner_size.unwrap()),
            &c(cfg.hidden_size),
            "hidden",
        ],
        if dense_blocks && cfg.linear_inner_size.unwrap().is_multiple_of(16) {
            "stdlib.qwen35.dense.projection.residual"
        } else if wmma.projections && cfg.linear_inner_size.unwrap().is_multiple_of(16) {
            "stdlib.qwen35.wmma.projection.residual"
        } else {
            "stdlib.qwen35.batch.residual"
        },
        "attention",
    );
    free(s, &ready, "attention", "attention-after-context");
    free(s, "hidden", "attention-after-context", "attention-ready");
    batch_norm(s, "attention-ready", "w10", "ffn-input", cfg);
    if moe {
        let mut args = names(&["ffn-input", "attention-ready"]);
        args.extend(lw[11..19].iter().cloned());
        if wmma.fp8 {
            args.extend(lw[25..].iter().cloned());
        }
        let refs = args.iter().map(String::as_str).collect::<Vec<_>>();
        call(s, &refs, moe_ffn, "final-hidden");
        finish_state(s, "history-ready", "shadow", "final-hidden");
    } else {
        call(
            s,
            &[
                "ffn-input",
                "w11",
                &c(cfg.hidden_size),
                &c(cfg.intermediate_size),
                "w12",
            ],
            if dense_blocks {
                "stdlib.qwen35.dense.projection.gate"
            } else {
                "stdlib.qwen35.batch.gate"
            },
            "active",
        );
        free(s, "ffn-input", "active", "active-ready");
        call(
            s,
            &[
                "active-ready",
                "w13",
                &c(cfg.intermediate_size),
                &c(cfg.hidden_size),
                "attention-ready",
            ],
            if dense_blocks {
                "stdlib.qwen35.dense.projection.residual"
            } else if wmma.projections {
                "stdlib.qwen35.wmma.projection.residual"
            } else {
                "stdlib.qwen35.batch.residual"
            },
            "output",
        );
        free(s, "active-ready", "output", "output-after-active");
        free(s, "attention-ready", "output-after-active", "final-hidden");
        finish_state(s, "history-ready", "shadow", "final-hidden");
    }
    // Branches share the same opaque state ABI as scalar execution.
    for (kind, group, proof) in [
        ("full", fg.as_str(), "((master.predicate prop.IsTrue) |-)"),
        (
            "linear",
            lg.as_str(),
            "(({(master.predicate prop.IsTrue)} stdlib.prelude.prop.Not) |-)",
        ),
    ] {
        let present = (0..cfg.num_layers)
            .any(|l| (cfg.layer_type(l) == LayerType::FullAttention) == (kind == "full"));
        if !present {
            continue;
        }
        let pair = format!("({{{TRIPLE} ({{{proof} 1}} *)}} *)");
        let mut args = vec![("weights", group)];
        args.extend(common.clone());
        args.push(("pair", &pair));
        header(
            s,
            &format!("stdlib.qwen35.batch.branch-{kind}"),
            &args,
            TRIPLE,
        );
        s.push_str("  ([.pair] *.elim [state proofs.])\n  ([.proofs] *.elim [proof unit.])\n  ([.proof] unsafe.discard unit.elim)\n  ([.unit] unit.elim)\n");
        call(
            s,
            &["weights"]
                .into_iter()
                .chain(cn.iter().copied())
                .chain(["state"])
                .collect::<Vec<_>>(),
            &format!("stdlib.qwen35.batch.{kind}"),
            "result",
        );
        s.push_str("  [.result]");
        finish(s);
    }
    let mut captures = vec![
        ("full-weights", full_tree),
        ("linear-weights", linear_tree),
        ("capacity", U64),
        ("start", U64),
        ("sequence", U64),
    ];
    captures.extend(scratch.iter().map(|n| (*n, OWN)));
    let capnames = captures.iter().map(|(n, _)| *n).collect::<Vec<_>>();
    let mut args = captures.clone();
    args.push(("pair", FOLDED));
    header(
        s,
        "stdlib.qwen35.batch.layer.fold",
        &args,
        &format!("({{{TRIPLE} (prop.Truth |-)}} *)"),
    );
    fold_prefix(s);
    let has_full = (0..cfg.num_layers).any(|l| cfg.layer_type(l) == LayerType::FullAttention);
    let has_linear = (0..cfg.num_layers).any(|l| cfg.layer_type(l) == LayerType::LinearAttention);
    if !has_full || !has_linear {
        let kind = if has_full { "full" } else { "linear" };
        call(
            s,
            &[&format!("{kind}-weights"), "index-value"],
            &format!("stdlib.qwen35.{kind}-weights"),
            "selected",
        );
        s.push_str("  ([.index-value] stdlib.models.memory.u32.to-u64 [layer.])\n");
        let args = ["selected"]
            .into_iter()
            .chain(cn.iter().copied())
            .chain(["state"])
            .collect::<Vec<_>>();
        call(s, &args, &format!("stdlib.qwen35.batch.{kind}"), "result");
    } else {
        call(
            s,
            &["full-weights", "index-value"],
            "stdlib.qwen35.full-weights",
            "full-selected",
        );
        call(
            s,
            &["linear-weights", "index-value"],
            "stdlib.qwen35.linear-weights",
            "linear-selected",
        );
        s.push_str("  ([.index-value] stdlib.models.memory.u32.to-u64 [layer.])\n");
        let mut previous = "(const.u32.0x00000000)".to_owned();
        for (i, layer) in (0..cfg.num_layers)
            .filter(|&l| cfg.layer_type(l) == LayerType::FullAttention)
            .enumerate()
        {
            call(
                s,
                &["index-value", &format!("(const.u32.0x{layer:08X})")],
                "stdlib.models.memory.u32.equal",
                &format!("eq{i}"),
            );
            call(
                s,
                &[&format!("eq{i}"), "(const.u32.0x00000001)", &previous],
                "stdlib.models.memory.select-u32",
                &format!("flag{i}"),
            );
            previous = format!("flag{i}");
        }
        call(
            s,
            &[&previous, "(const.u32.0x00000001)"],
            "stdlib.models.memory.u32.equal",
            "is-full",
        );
        s.push_str("  ({[.is-full] (unit.intro meta.master.predicate)} smolcat.value.named_at :.forget [condition.])\n");
        for kind in ["full", "linear"] {
            call(
                s,
                &[format!("{kind}-selected").as_str()]
                    .into_iter()
                    .chain(cn.iter().copied())
                    .collect::<Vec<_>>(),
                &format!("partial.stdlib.qwen35.batch.branch-{kind}.{}", cn.len() + 1),
                &format!("{kind}-branch"),
            );
        }
        call(
            s,
            &[
                "condition",
                "state",
                "full-branch",
                "linear-branch",
                "(unit.intro)",
            ],
            "core.if",
            "result",
        );
    }
    fold_result(s);
    let mut common = vec![
        ("tokens", REF),
        ("embedding", REF),
        ("full-weights", full_tree),
        ("linear-weights", linear_tree),
        ("capacity", U64),
        ("start", U64),
    ];
    common.extend(scratch.iter().map(|n| (*n, OWN)));
    let cn = common.iter().map(|(n, _)| *n).collect::<Vec<_>>();
    for (kind, proof) in [
        ("scalar", "((master.predicate prop.IsTrue) |-)"),
        (
            "batch",
            "(({(master.predicate prop.IsTrue)} stdlib.prelude.prop.Not) |-)",
        ),
    ] {
        let pair = format!("({{{TRIPLE} ({{{proof} 1}} *)}} *)");
        let mut args = common.clone();
        args.push(("pair", &pair));
        header(s, &format!("stdlib.qwen35.text-{kind}"), &args, TRIPLE);
        s.push_str("  ([.pair] *.elim [state proofs.])\n  ([.proofs] *.elim [condition-proof unit.])\n  ([.condition-proof] unsafe.discard unit.elim)\n  ([.unit] unit.elim)\n");
        if kind == "scalar" {
            call(
                s,
                &cn,
                &format!("partial.stdlib.qwen35.token.fold.{}", cn.len()),
                "callback",
            );
            s.push_str("  ({([.tokens] stdlib.models.memory.ref-count-u64 stdlib.models.memory.fold-count-u32) (unit.intro meta.master.end32)} smolcat.value.named_at [end.])\n  ({[.end] [.state] [.callback] core.truth} core.fold.bounded [result proof.])\n  ([.proof] unsafe.discard unit.elim)\n  [.result]");
        } else {
            unpack(s, "state", &names(&["cache", "shadow", "previous"]));
            call(
                s,
                &["tokens"],
                "stdlib.models.memory.ref-count-u64",
                "sequence",
            );
            call(
                s,
                &["tokens", "embedding", &c(0)],
                "stdlib.qwen35.prefill.embed",
                "embedded",
            );
            free(s, "previous", "embedded", "batch");
            pack(s, &names(&["cache", "shadow", "batch"]), "initial");
            call(
                s,
                &capnames,
                &format!("partial.stdlib.qwen35.batch.layer.fold.{}", capnames.len()),
                "callback",
            );
            writeln!(s,"  ({{(const.u32.0x{:08X}) (unit.intro meta.master.end32)}} smolcat.value.named_at [end.])",cfg.num_layers).unwrap();
            s.push_str("  ({[.end] [.initial] [.callback] core.truth} core.fold.bounded [decoded proof.])\n  ([.proof] unsafe.discard unit.elim)\n");
            unpack(
                s,
                "decoded",
                &names(&["cache-final", "shadow-final", "batch-final"]),
            );
            call(
                s,
                &["sequence", &c(1)],
                "stdlib.models.memory.u64.sub",
                "last",
            );
            call(
                s,
                &["batch-final", "last"],
                "stdlib.qwen35.prefill.row",
                "last-hidden",
            );
            free(s, "batch-final", "last-hidden", "hidden-final");
            pack(
                s,
                &names(&["cache-final", "shadow-final", "hidden-final"]),
                "result",
            );
            s.push_str("  [.result]");
        }
        finish(s);
    }
    let mut args = common;
    args.push(("state", TRIPLE));
    header(s, "stdlib.qwen35.text-dispatch", &args, TRIPLE);
    call(
        s,
        &["tokens"],
        "stdlib.models.memory.ref-count-u64",
        "sequence",
    );
    call(
        s,
        &["sequence", &c(2)],
        "stdlib.models.memory.u64.less",
        "small",
    );
    s.push_str("  ({[.small] (unit.intro meta.master.predicate)} smolcat.value.named_at :.forget [condition.])\n");
    for kind in ["scalar", "batch"] {
        call(
            s,
            &cn,
            &format!("partial.stdlib.qwen35.text-{kind}.{}", cn.len()),
            kind,
        );
    }
    call(
        s,
        &["condition", "state", "scalar", "batch", "(unit.intro)"],
        "core.if",
        "result",
    );
    s.push_str("  [.result]");
    finish(s);
    if wmma.fp8 {
        let graph = s.split_off(source_start);
        s.push_str(&fp8_graph(&graph, moe));
    }
}

/// Replace only batched projection calls; decode keeps the original operands.
fn fp8_graph(graph: &str, moe: bool) -> String {
    let mut out = String::new();
    let mut linear = false;
    for line in graph.lines() {
        if line.starts_with("(def program stdlib.qwen35.batch.full :") {
            linear = false;
        }
        if line.starts_with("(def program stdlib.qwen35.batch.linear :") {
            linear = true;
        }
        let projection = line.contains("} stdlib.qwen35.dense.project [")
            || line.contains("} stdlib.qwen35.dense.projection.residual [")
            || line.contains("} stdlib.qwen35.dense.projection.gate [");
        if !projection {
            out.push_str(&line.replace(
                "stdlib.qwen35.wmma.attention.run [context.]",
                "stdlib.qwen3.fp8.attention.run [context.]",
            ));
            out.push('\n');
            continue;
        }
        let (args, result) = line.trim().split_once("} ").unwrap();
        let args = args.strip_prefix("({").unwrap();
        let args: Vec<_> = args.split_whitespace().collect();
        let weight: usize = args[1]
            .strip_prefix("[.w")
            .unwrap()
            .strip_suffix(']')
            .unwrap()
            .parse()
            .unwrap();
        let indices = super::super::fp8_projection_indices(moe, linear);
        let Some(index) = indices.iter().position(|&w| w == weight) else {
            // The small alpha/beta projections stay in checkpoint precision.
            out.push_str(line);
            out.push('\n');
            continue;
        };
        let base = if linear {
            if moe { 19 } else { 14 }
        } else if moe {
            16
        } else {
            11
        };
        let native = format!("w{}", base + index * 2);
        let scale = format!("w{}", base + index * 2 + 1);
        let input = args[0]
            .strip_prefix("[.")
            .unwrap()
            .strip_suffix(']')
            .unwrap();
        let bf16 = format!("w{weight}");
        let output = result
            .split_once(" [")
            .unwrap()
            .1
            .strip_suffix(".])")
            .unwrap();
        let shared_input = input == "normalized";
        let quant_name = if shared_input {
            "projection-input"
        } else {
            output
        };
        let quant = format!("quant-{quant_name}");
        let bytes = format!("bytes-{quant_name}");
        let scales = format!("scales-{quant_name}");
        if !shared_input || index == 0 {
            if moe {
                call(
                    &mut out,
                    &[input, args[2]],
                    "stdlib.qwen35.fp8.quantize-matrix",
                    &quant,
                );
            } else {
                call(&mut out, &[input], "stdlib.qwen3.fp8.quantize", &quant);
            }
            unpack(&mut out, &quant, &names(&[&bytes, &scales]));
        }
        let mut operands = vec![
            input, &bytes, &scales, &bf16, &native, &scale, args[2], args[3],
        ];
        let up = format!("w{}", weight + 1);
        let up_native = format!("w{}", base + (index + 1) * 2);
        let up_scale = format!("w{}", base + (index + 1) * 2 + 1);
        let operation = if result.starts_with("stdlib.qwen35.dense.projection.residual") {
            operands.push(
                args[4]
                    .strip_prefix("[.")
                    .unwrap()
                    .strip_suffix(']')
                    .unwrap(),
            );
            "project-residual"
        } else if result.starts_with("stdlib.qwen35.dense.projection.gate") {
            operands.extend([up.as_str(), up_native.as_str(), up_scale.as_str()]);
            call(
                &mut out,
                &operands,
                "stdlib.qwen3.fp8.project-gate",
                &format!("raw-{output}"),
            );
            operands.truncate(8);
            ""
        } else {
            "project"
        };
        if !operation.is_empty() && moe {
            let rows = format!("rows-{output}");
            call(
                &mut out,
                &[input],
                "stdlib.models.memory.own-count-bf16",
                &format!("elements-{output}"),
            );
            call(
                &mut out,
                &[&format!("elements-{output}"), args[2]],
                "stdlib.models.memory.u64.div",
                &rows,
            );
            let mut native_args = vec![
                bytes.as_str(),
                native.as_str(),
                scales.as_str(),
                scale.as_str(),
                rows.as_str(),
                args[2],
                args[3],
            ];
            if operation == "project-residual" {
                native_args.push(operands[8]);
            }
            call(
                &mut out,
                &native_args,
                if operation == "project-residual" {
                    "stdlib.qwen35.fp8.tail.project-residual"
                } else {
                    "stdlib.qwen35.fp8.tail.project"
                },
                &format!("raw-{output}"),
            );
        } else if !operation.is_empty() {
            call(
                &mut out,
                &operands,
                &format!("stdlib.qwen3.fp8.{operation}"),
                &format!("raw-{output}"),
            );
        }
        if shared_input && weight != if linear { 2 } else { 3 } {
            writeln!(out, "  ([.raw-{output}] [{output}.])").unwrap();
        } else {
            call(
                &mut out,
                &[&bytes, &format!("raw-{output}")],
                "stdlib.models.resources.free-f32-before",
                &format!("freed-{output}"),
            );
            call(
                &mut out,
                &[&scales, &format!("freed-{output}")],
                "stdlib.models.resources.free-f32-before",
                output,
            );
        }
    }
    out
}
