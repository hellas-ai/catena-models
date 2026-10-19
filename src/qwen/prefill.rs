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

pub(super) fn emit(s: &mut String, cfg: &ModelConfig, full_tree: &str, linear_tree: &str) {
    let moe = cfg.architecture == ModelArchitecture::Qwen35Moe;
    let fcount = if moe { 16 } else { 11 };
    let lcount = if moe { 19 } else { 14 };
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
            "stdlib.qwen35.batch.project",
            out,
        );
    }
    call(
        s,
        &["layer", "capacity"],
        "stdlib.qwen3.memory.u64.mul",
        "layer-capacity",
    );
    call(
        s,
        &["layer-capacity", &c(cfg.key_value_size() * 2)],
        "stdlib.qwen3.memory.u64.mul",
        "key-base",
    );
    call(
        s,
        &["capacity", &c(cfg.key_value_size())],
        "stdlib.qwen3.memory.u64.mul",
        "value-offset",
    );
    call(
        s,
        &["key-base", "value-offset"],
        "stdlib.qwen3.memory.u64.add",
        "value-base",
    );
    call(
        s,
        &["start", &c(cfg.key_value_size())],
        "stdlib.qwen3.memory.u64.mul",
        "start-offset",
    );
    call(
        s,
        &["key-base", "start-offset"],
        "stdlib.qwen3.memory.u64.add",
        "key-target",
    );
    call(
        s,
        &["value-base", "start-offset"],
        "stdlib.qwen3.memory.u64.add",
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
        "stdlib.qwen35.batch.staged",
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
        "stdlib.qwen35.batch.residual",
        "attention",
    );
    free(s, &ready, "attention", "attention-after-context");
    free(s, "hidden", "attention-after-context", "attention-ready");
    batch_norm(s, "attention-ready", "w7", "ffn-input", cfg);
    if moe {
        let mut args = names(&["ffn-input", "attention-ready"]);
        args.extend(fw[8..].iter().cloned());
        let refs = args.iter().map(String::as_str).collect::<Vec<_>>();
        call(s, &refs, "stdlib.qwen35.batch.moe.ffn", "final-hidden");
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
            "stdlib.qwen35.batch.gate",
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
            "stdlib.qwen35.batch.residual",
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
            "stdlib.qwen35.batch.project",
            out,
        );
    }
    let recurrent = cfg.linear_state_size.unwrap() * cfg.linear_inner_size.unwrap();
    let conv = 3 * cfg.linear_qkv_size().unwrap();
    call(
        s,
        &["capacity", &c(cfg.kv_cache_elements_per_token().unwrap())],
        "stdlib.qwen3.memory.u64.mul",
        "kv-end",
    );
    call(
        s,
        &["layer", &c(recurrent + conv)],
        "stdlib.qwen3.memory.u64.mul",
        "layer-offset",
    );
    call(
        s,
        &["kv-end", "layer-offset"],
        "stdlib.qwen3.memory.u64.add",
        "conv-base",
    );
    call(
        s,
        &["layer", &c(recurrent)],
        "stdlib.qwen3.memory.u64.mul",
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
        "stdlib.qwen35.batch.scan",
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
        "stdlib.qwen35.batch.residual",
        "attention",
    );
    free(s, &ready, "attention", "attention-after-context");
    free(s, "hidden", "attention-after-context", "attention-ready");
    batch_norm(s, "attention-ready", "w10", "ffn-input", cfg);
    if moe {
        let mut args = names(&["ffn-input", "attention-ready"]);
        args.extend(lw[11..].iter().cloned());
        let refs = args.iter().map(String::as_str).collect::<Vec<_>>();
        call(s, &refs, "stdlib.qwen35.batch.moe.ffn", "final-hidden");
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
            "stdlib.qwen35.batch.gate",
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
            "stdlib.qwen35.batch.residual",
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
    s.push_str("  ([.index-value] stdlib.qwen3.memory.u32.to-u64 [layer.])\n");
    let mut previous = "(const.u32.0x00000000)".to_owned();
    for (i, layer) in (0..cfg.num_layers)
        .filter(|&l| cfg.layer_type(l) == LayerType::FullAttention)
        .enumerate()
    {
        call(
            s,
            &["index-value", &format!("(const.u32.0x{layer:08X})")],
            "stdlib.qwen3.memory.u32.equal",
            &format!("eq{i}"),
        );
        call(
            s,
            &[&format!("eq{i}"), "(const.u32.0x00000001)", &previous],
            "stdlib.qwen3.memory.select-u32",
            &format!("flag{i}"),
        );
        previous = format!("flag{i}");
    }
    call(
        s,
        &[&previous, "(const.u32.0x00000001)"],
        "stdlib.qwen3.memory.u32.equal",
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
            s.push_str("  ({([.tokens] stdlib.qwen3.memory.ref-count-u64 stdlib.qwen3.memory.fold-count-u32) (unit.intro meta.master.end32)} smolcat.value.named_at [end.])\n  ({[.end] [.state] [.callback] core.truth} core.fold.bounded [result proof.])\n  ([.proof] unsafe.discard unit.elim)\n  [.result]");
        } else {
            unpack(s, "state", &names(&["cache", "shadow", "previous"]));
            call(
                s,
                &["tokens"],
                "stdlib.qwen3.memory.ref-count-u64",
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
                "stdlib.qwen3.memory.u64.sub",
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
        "stdlib.qwen3.memory.ref-count-u64",
        "sequence",
    );
    call(
        s,
        &["sequence", &c(2)],
        "stdlib.qwen3.memory.u64.less",
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
}
