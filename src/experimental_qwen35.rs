//! Dense Qwen3.5 hybrid graph for the experimental compiler.
use super::*;
use crate::spec::LayerType;
const TRIPLE: &str = "({((cap.own mem) val) ({((cap.own mem) val) ((cap.own mem) val)} *)} *)";
const FOLDED: &str = "({({master.index32 u32} :) ({({((cap.own mem) val) ({((cap.own mem) val) ((cap.own mem) val)} *)} *) ({(({master.index32 master.end32} prop.Less) |-) (prop.Truth |-)} *)} *)} *)";

fn finish(s: &mut String) {
    s.push_str("\n ))\n");
}
fn free(s: &mut String, old: &str, after: &str, output: &str) {
    call(s, &[old, after], "qwen3.free-bf16-before", output);
}
fn floats(n: usize) -> String {
    format!("(const.u32.0x{:08X} u32.bitcast-f32)", (n as f32).to_bits())
}
fn names(list: &[&str]) -> Vec<String> {
    list.iter().map(|x| (*x).to_owned()).collect()
}
fn fold_prefix(s: &mut String) {
    s.push_str("  ([.pair] *.elim [index rest.])\n  ([.rest] *.elim [state proofs.])\n  ([.proofs] *.elim [less truth.])\n  ([.less] unsafe.discard unit.elim)\n  ([.index] :.forget [index-value.])\n");
}
fn selector(s: &mut String, layers: &[usize], group: &str, tree: &str, kind: &str) {
    let mut selector = String::new();
    emit_layer_selector(&mut selector, layers.len(), group, tree);
    // Select by absolute cache layer index, even though weight groups are compact.
    for (compact, absolute) in layers.iter().enumerate().skip(1).rev() {
        selector = selector.replace(
            &format!("const.u32.0x{compact:08X}"),
            &format!("const.u32.0x{absolute:08X}"),
        );
    }
    selector = selector
        .replace(
            "stdlib.qwen3.select-layer",
            &format!("stdlib.qwen35.select-{kind}"),
        )
        .replace(
            "stdlib.qwen3.layer-weights",
            &format!("stdlib.qwen35.{kind}-weights"),
        );
    s.push_str(&selector);
}

pub(crate) fn source(config: &ModelConfig, options: InferenceOptions) -> Result<String> {
    let full = (0..config.num_layers)
        .filter(|&l| config.layer_type(l) == LayerType::FullAttention)
        .collect::<Vec<_>>();
    let linear = (0..config.num_layers)
        .filter(|&l| config.layer_type(l) == LayerType::LinearAttention)
        .collect::<Vec<_>>();
    ensure!(
        !full.is_empty() && !linear.is_empty(),
        "experimental Qwen3.5 requires both full and linear layers"
    );
    ensure!(
        config.linear_conv_kernel == Some(4),
        "experimental Qwen3.5 requires four convolution taps"
    );
    let mut base = config.clone();
    base.architecture = ModelArchitecture::Qwen3Dense;
    let mut s = super::fused::source(&base, options)?;
    let cut = s
        .find("(def program stdlib.qwen3.select-layer.yes")
        .unwrap();
    // Preserve kernel definitions appended after the reference entrypoint.
    let kernels = s[s.find("(def program stdlib.qwen3.perf.grid").unwrap()..].to_owned();
    s.truncate(cut);
    s.push_str(&kernels);
    // Match the default BF16 GEMV's 256-lane accumulation tree for the first
    // Qwen3.5 port; its recurrent state amplifies small projection differences.
    let projection_start = s.find("(def program stdlib.qwen3.fused.dot :").unwrap();
    let projection_end = s
        .find("(def program stdlib.qwen3.fused.norm.square :")
        .unwrap();
    let projection = s[projection_start..projection_end]
        .replace(&c(32), &c(256))
        .replace(&c(31), &c(255))
        .replace(&c(8), &c(1))
        .replace(&c(7), &c(0));
    s.replace_range(projection_start..projection_end, &projection);
    // The fused base merges shared-memory metatypes into its initial theories.
    let start = s.find("(def program qwen3.decoder-layer").unwrap();
    let end = start + s[start..].find("\n ))").unwrap() + 5;
    let mut decoder = s[start..end].to_owned();
    let qline = format!(
        "{{[.normalized] [.q-weight] {} {} }} qwen3.fused.project [query-raw.]",
        c(config.hidden_size),
        c(config.query_size())
    );
    // Decoder Hex uses compact spacing rather than the renderer's spacing.
    let qline = if decoder.contains(&qline) {
        qline
    } else {
        format!(
            "{{[.normalized] [.q-weight] {} {}}} qwen3.fused.project [query-raw.]",
            c(config.hidden_size),
            c(config.query_size())
        )
    };
    ensure!(
        decoder.contains(&qline),
        "missing full-attention query projection"
    );
    decoder = decoder.replace(&qline, &format!("{{[.normalized] [.q-weight] {} {}}} qwen3.fused.project [qg.])\n  ({{[.qg]}} qwen35.query [query-raw.]", c(config.hidden_size), c(config.query_size()*2)));
    let marker = "  ({[.context-ready] [.o-weight]";
    let begin = decoder.find(marker).unwrap();
    let mut gate = String::new();
    call(
        &mut gate,
        &["qg", "context-ready"],
        "qwen35.gate-context",
        "gated-context",
    );
    free(&mut gate, "qg", "gated-context", "context-after-qg");
    free(
        &mut gate,
        "context-ready",
        "context-after-qg",
        "full-context",
    );
    decoder.insert_str(begin, &gate);
    let tail = decoder[begin + gate.len()..].replace("context-ready", "full-context");
    decoder.truncate(begin + gate.len());
    decoder.push_str(&tail);
    s.replace_range(start..end, &decoder);

    let key_dim = config.linear_state_size.unwrap();
    let key_heads = config.linear_group_count.unwrap();
    let value_dim = config.linear_value_head_size().unwrap();
    let value_heads = config.linear_time_step_rank.unwrap();
    let value_size = config.linear_inner_size.unwrap();
    let qkv_size = config.linear_qkv_size().unwrap();
    let recurrent = key_dim
        .checked_mul(value_dim)
        .and_then(|v| v.checked_mul(value_heads))
        .ok_or_else(|| anyhow::anyhow!("recurrent workspace overflow"))?;
    let conv = qkv_size
        .checked_mul(3)
        .ok_or_else(|| anyhow::anyhow!("convolution workspace overflow"))?;
    let mut kernels = [
        include_str!("catena/experimental/qwen35.hex"),
        include_str!("catena/experimental/qwen35_attention.hex"),
    ]
    .join("\n");
    for (key, value) in [
        ("KEY_DIM", key_dim),
        ("KEY_HEADS", key_heads),
        ("KEY_SIZE", config.linear_key_size().unwrap()),
        ("VALUE_DIM", value_dim),
        ("VALUE_SIZE", value_size),
        ("QKV_SIZE", qkv_size),
        ("QUERY_SIZE", config.query_size()),
        ("KEY_VALUE_SIZE", config.key_value_size()),
        ("QUERY_HEADS", config.num_attention_heads),
        (
            "QUERY_HEADS_PER_KV_HEAD",
            config.num_attention_heads / config.num_key_value_heads,
        ),
        ("HEAD_DIM", config.head_dim),
        ("RECURRENT_SIZE", recurrent),
        ("CONV_STATE_SIZE", conv),
        ("LINEAR_STRIDE", conv + recurrent),
        ("KV_STRIDE", config.kv_cache_elements_per_token()?),
        (
            "SHADOW_SIZE",
            recurrent
                .checked_mul(config.num_layers)
                .ok_or_else(|| anyhow::anyhow!("shadow workspace overflow"))?,
        ),
    ] {
        kernels = kernels.replace(&format!("{{{{{key}}}}}"), &format!("{value:016X}"));
    }
    for (key, value) in [
        ("DELTA_SCALE_BITS", (key_dim as f32).sqrt().recip()),
        ("VALUE_DIM_F32", value_dim as f32),
        ("RMS_EPSILON_BITS", config.rms_norm_eps),
    ] {
        kernels = kernels.replace(
            &format!("{{{{{key}}}}}"),
            &format!("{:08X}", value.to_bits()),
        );
    }
    ensure!(!kernels.contains("{{KEY"), "unresolved Qwen3.5 dimensions");
    s = s
        .replace("} qwen3.softmax [", "} qwen35.softmax [")
        .replace("} qwen3.context [", "} qwen35.context [");
    s.push_str(&kernels);
    // The default cached Q.K reduction accumulates scalar products in order.
    let score_start = s.find("(def program stdlib.qwen3.score-dot :").unwrap();
    let score_end = score_start + s[score_start..].find("\n ))").unwrap() + 5;
    let mut scalar_score = String::new();
    header(
        &mut scalar_score,
        "stdlib.qwen3.score-dot",
        &[
            ("query", OWN),
            ("cache", OWN),
            ("query-base", U64),
            ("key-base", U64),
            ("pair", "({(u64 val) (f32 val)} *)"),
        ],
        "(f32 val)",
    );
    scalar_score.push_str("  ([.pair] *.elim [i acc.])\n");
    call(
        &mut scalar_score,
        &["query-base", &c(2)],
        "stdlib.qwen3.memory.u64.mul",
        "qb",
    );
    call(
        &mut scalar_score,
        &["key-base", &c(2)],
        "stdlib.qwen3.memory.u64.mul",
        "kb",
    );
    for (buffer, base, prefix) in [("query", "qb", "q"), ("cache", "kb", "k")] {
        call(
            &mut scalar_score,
            &[base, "i"],
            "stdlib.qwen3.memory.u64.add",
            &format!("{prefix}-address"),
        );
        call(
            &mut scalar_score,
            &[buffer, &format!("{prefix}-address")],
            "stdlib.qwen3.memory.own-read-bf16",
            &format!("{prefix}-bf"),
        );
        writeln!(scalar_score, "  ([.{prefix}-bf] bf16.to-f32 [{prefix}.])").unwrap();
    }
    call(
        &mut scalar_score,
        &["q", "k"],
        "stdlib.qwen3.memory.f32.mul",
        "product",
    );
    call(
        &mut scalar_score,
        &["acc", "product"],
        "stdlib.qwen3.memory.f32.add",
        "sum",
    );
    scalar_score.push_str("  [.sum]");
    finish(&mut scalar_score);
    s.replace_range(score_start..score_end, &scalar_score);
    let scores_start = s.find("(def program stdlib.qwen3.scores :").unwrap();
    let scores_end = scores_start + s[scores_start..].find("\n ))").unwrap() + 5;
    let old_bound = format!("const.u64.0x{:016X} [pairs.]", config.head_dim / 2);
    let new_bound = format!("const.u64.0x{:016X} [pairs.]", config.head_dim);
    ensure!(
        s[scores_start..scores_end].contains(&old_bound),
        "missing scalar attention reduction bound"
    );
    let scores = s[scores_start..scores_end].replace(&old_bound, &new_bound);
    s.replace_range(scores_start..scores_end, &scores);
    emit_linear_decoder(&mut s, config, recurrent, conv);
    let full_group = nested_product(&vec![REF.to_owned(); 11]);
    let linear_group = nested_product(&vec![REF.to_owned(); 14]);
    let full_tree = nested_product(&vec![full_group.clone(); full.len()]);
    let linear_tree = nested_product(&vec![linear_group.clone(); linear.len()]);
    selector(&mut s, &full, &full_group, &full_tree, "full");
    selector(&mut s, &linear, &linear_group, &linear_tree, "linear");
    // Typed branches keep the distinct full-attention and DeltaNet weight ABIs.
    for (kind, group, proof) in [
        (
            "full",
            full_group.as_str(),
            "((master.predicate prop.IsTrue) |-)",
        ),
        (
            "linear",
            linear_group.as_str(),
            "(({(master.predicate prop.IsTrue)} stdlib.prelude.prop.Not) |-)",
        ),
    ] {
        let pair = format!("({{{TRIPLE} ({{{proof} 1}} *)}} *)");
        header(
            &mut s,
            &format!("stdlib.qwen35.branch-{kind}"),
            &[
                ("weights", group),
                ("rope", OWN),
                ("capacity", U64),
                ("position", U64),
                ("layer", U64),
                ("pair", &pair),
            ],
            TRIPLE,
        );
        s.push_str("  ([.pair] *.elim [state proofs.])\n  ([.proofs] *.elim [proof unit.])\n  ([.proof] unsafe.discard unit.elim)\n  ([.unit] unit.elim)\n");
        unpack(&mut s, "state", &names(&["cache", "shadow", "hidden"]));
        let ws = (0..if kind == "full" { 11 } else { 14 })
            .map(|i| format!("w{i}"))
            .collect::<Vec<_>>();
        unpack(&mut s, "weights", &ws);
        let mut args = vec!["hidden", "cache", "rope", "capacity", "position", "layer"];
        if kind == "linear" {
            args.push("shadow");
        }
        args.extend(ws.iter().map(String::as_str));
        call(
            &mut s,
            &args,
            if kind == "full" {
                "qwen3.decoder-layer"
            } else {
                "qwen35.linear-decoder"
            },
            "decoded",
        );
        if kind == "full" {
            s.push_str("  ([.decoded] *.elim [next-cache next-hidden.])\n  ([.next-hidden] meta.master.complete-own [ready-hidden phase.])\n  ({[.shadow] [.phase]} meta.master.after-own [ready-shadow.])\n");
            pack(
                &mut s,
                &names(&["next-cache", "ready-shadow", "ready-hidden"]),
                "result",
            );
            s.push_str("  [.result]");
        } else {
            s.push_str("  [.decoded]");
        }
        finish(&mut s);
    }
    let captures = [
        ("full-weights", full_tree.as_str()),
        ("linear-weights", linear_tree.as_str()),
        ("rope", OWN),
        ("capacity", U64),
        ("position", U64),
    ];
    let mut args = captures.to_vec();
    args.push(("pair", FOLDED));
    header(
        &mut s,
        "stdlib.qwen35.layer.fold",
        &args,
        &format!("({{{TRIPLE} (prop.Truth |-)}} *)"),
    );
    fold_prefix(&mut s);
    call(
        &mut s,
        &["full-weights", "index-value"],
        "stdlib.qwen35.full-weights",
        "full-selected",
    );
    call(
        &mut s,
        &["linear-weights", "index-value"],
        "stdlib.qwen35.linear-weights",
        "linear-selected",
    );
    s.push_str("  ([.index-value] stdlib.qwen3.memory.u32.to-u64 [layer.])\n");
    // Count membership through scalar equality and select; no layout assumptions.
    let mut previous = "(const.u32.0x00000000)".to_owned();
    for (i, &l) in full.iter().enumerate() {
        call(
            &mut s,
            &["index-value", &format!("(const.u32.0x{l:08X})")],
            "stdlib.qwen3.memory.u32.equal",
            &format!("is-full{i}"),
        );
        call(
            &mut s,
            &[&format!("is-full{i}"), "(const.u32.0x00000001)", &previous],
            "stdlib.qwen3.memory.select-u32",
            &format!("full-flag{i}"),
        );
        previous = format!("full-flag{i}");
    }
    call(
        &mut s,
        &[&previous, "(const.u32.0x00000001)"],
        "stdlib.qwen3.memory.u32.equal",
        "is-full",
    );
    s.push_str("  ({[.is-full] (unit.intro meta.master.predicate)} smolcat.value.named_at :.forget [condition.])\n");
    for kind in ["full", "linear"] {
        call(
            &mut s,
            &[
                &format!("{kind}-selected"),
                "rope",
                "capacity",
                "position",
                "layer",
            ],
            &format!("partial.stdlib.qwen35.branch-{kind}.5"),
            &format!("{kind}-branch"),
        );
    }
    call(
        &mut s,
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
    s.push_str("  ({[.result] [.truth]} *.intro)");
    finish(&mut s);
    emit_entry(
        &mut s,
        config,
        &full,
        &linear,
        &full_tree,
        &linear_tree,
        recurrent,
    )?;
    // Normalization's column count is the hidden or full-attention head width.
    for (width, float, rows) in [
        (config.hidden_size, floats(config.hidden_size), 1),
        (
            config.head_dim,
            floats(config.head_dim),
            config.num_attention_heads,
        ),
        (
            config.head_dim,
            floats(config.head_dim),
            config.num_key_value_heads,
        ),
    ] {
        let columns = c(width);
        let stages = c(width.trailing_zeros() as usize);
        let pattern = format!("{columns} {float} {} }} qwen3.fused.normalize", c(rows));
        s = s.replace(
            &pattern,
            &format!("{columns} {float} {} {stages} }} qwen35.normalize", c(rows)),
        );
        let pattern = format!("{columns} {float} {}}} qwen3.fused.normalize", c(rows));
        s = s.replace(
            &pattern,
            &format!("{columns} {float} {} {stages}}} qwen35.normalize", c(rows)),
        );
    }
    Ok(s)
}

fn emit_linear_decoder(s: &mut String, c0: &ModelConfig, recurrent: usize, conv: usize) {
    let ws = [
        "input-norm",
        "qkv-weight",
        "z-weight",
        "alpha-weight",
        "beta-weight",
        "dt",
        "a",
        "linear-norm",
        "out-weight",
        "conv-weight",
        "post-norm",
        "gate-weight",
        "up-weight",
        "down-weight",
    ];
    let mut args = vec![
        ("hidden", OWN),
        ("cache", OWN),
        ("rope", OWN),
        ("capacity", U64),
        ("position", U64),
        ("layer", U64),
        ("shadow", OWN),
    ];
    args.extend(ws.iter().map(|w| (*w, REF)));
    header(s, "qwen35.linear-decoder", &args, TRIPLE);
    let hidden = c(c0.hidden_size);
    let hf = floats(c0.hidden_size);
    let one = c(1);
    call(
        s,
        &["hidden", "input-norm", &hidden, &hf, &one],
        "qwen3.fused.normalize",
        "normalized",
    );
    for (out, weight, size) in [
        ("qkv", "qkv-weight", c0.linear_qkv_size().unwrap()),
        ("z", "z-weight", c0.linear_inner_size.unwrap()),
        ("alpha", "alpha-weight", c0.linear_time_step_rank.unwrap()),
        ("beta", "beta-weight", c0.linear_time_step_rank.unwrap()),
    ] {
        call(
            s,
            &["normalized", weight, &hidden, &c(size)],
            "qwen3.fused.project",
            out,
        );
    }
    call(
        s,
        &["capacity", &c(c0.kv_cache_elements_per_token().unwrap())],
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
        &["qkv", "cache", "conv-weight", "conv-base", "position"],
        "qwen35.conv",
        "convolved",
    );
    for (out, offset) in [("query", 0), ("key", c0.linear_key_size().unwrap())] {
        call(
            s,
            &["convolved", &c(offset), &c(c0.linear_state_size.unwrap())],
            "qwen35.l2",
            out,
        );
    }
    call(
        s,
        &[
            "convolved",
            "query",
            "key",
            "alpha",
            "beta",
            "dt",
            "a",
            "shadow",
            "shadow-base",
        ],
        "qwen35.delta",
        "core-output",
    );
    call(
        s,
        &["core-output", "z", "linear-norm"],
        "qwen35.gated-norm",
        "context",
    );
    let mut ready = "context".to_owned();
    for old in [
        "normalized",
        "qkv",
        "convolved",
        "query",
        "key",
        "alpha",
        "beta",
        "core-output",
        "z",
    ] {
        let next = format!("context-after-{old}");
        free(s, old, &ready, &next);
        ready = next;
    }
    call(
        s,
        &[
            &ready,
            "out-weight",
            &c(c0.linear_inner_size.unwrap()),
            &hidden,
            "hidden",
        ],
        "qwen3.fused.residual",
        "attention",
    );
    free(s, &ready, "attention", "attention-after-context");
    free(s, "hidden", "attention-after-context", "attention-ready");
    call(
        s,
        &["attention-ready", "post-norm", &hidden, &hf, &one],
        "qwen3.fused.normalize",
        "ffn-input",
    );
    call(
        s,
        &[
            "ffn-input",
            "gate-weight",
            &hidden,
            &c(c0.intermediate_size),
            "up-weight",
        ],
        "qwen3.fused.gate",
        "activated",
    );
    free(s, "ffn-input", "activated", "activated-ready");
    call(
        s,
        &[
            "activated-ready",
            "down-weight",
            &c(c0.intermediate_size),
            &hidden,
            "attention-ready",
        ],
        "qwen3.fused.residual",
        "result",
    );
    free(s, "activated-ready", "result", "result-after-activation");
    free(
        s,
        "attention-ready",
        "result-after-activation",
        "result-ready",
    );
    s.push_str("  ([.result-ready] meta.master.complete-own [completed phase.])\n  ({[.cache] [.phase]} meta.master.after-own [cache-ready.])\n  ({[.shadow] [.phase]} meta.master.after-own [shadow-ready.])\n");
    pack(
        s,
        &names(&["cache-ready", "shadow-ready", "completed"]),
        "state",
    );
    s.push_str("  [.state]");
    finish(s);
}

fn emit_entry(
    s: &mut String,
    config: &ModelConfig,
    full: &[usize],
    linear: &[usize],
    full_tree: &str,
    linear_tree: &str,
    _recurrent: usize,
) -> Result<()> {
    let captures = [
        ("tokens", REF),
        ("embedding", REF),
        ("full-weights", full_tree),
        ("linear-weights", linear_tree),
        ("capacity", U64),
        ("start", U64),
    ];
    let mut args = captures.to_vec();
    args.push(("pair", FOLDED));
    header(
        s,
        "stdlib.qwen35.token.fold",
        &args,
        &format!("({{{TRIPLE} (prop.Truth |-)}} *)"),
    );
    fold_prefix(s);
    unpack(s, "state", &names(&["cache", "shadow", "previous"]));
    s.push_str("  ([.index-value] stdlib.qwen3.memory.u32.to-u64 [token-index.])\n");
    call(
        s,
        &["start", "token-index"],
        "stdlib.qwen3.memory.u64.add",
        "position",
    );
    call(
        s,
        &["tokens", "embedding", "token-index"],
        "qwen3.embedding-one",
        "embedded",
    );
    free(s, "previous", "embedded", "hidden");
    call(s, &["position"], "qwen3.rope-one", "rope");
    pack(s, &names(&["cache", "shadow", "hidden"]), "initial");
    call(
        s,
        &[
            "full-weights",
            "linear-weights",
            "rope",
            "capacity",
            "position",
        ],
        "partial.stdlib.qwen35.layer.fold.5",
        "layer-callback",
    );
    writeln!(s,"  ({{(const.u32.0x{:08X}) (unit.intro meta.master.end32)}} smolcat.value.named_at [layer-end.])",config.num_layers).unwrap();
    s.push_str("  ({[.layer-end] [.initial] [.layer-callback] core.truth} core.fold.bounded [decoded layer-proof.])\n  ([.layer-proof] unsafe.discard unit.elim)\n");
    unpack(
        s,
        "decoded",
        &names(&["final-cache", "final-shadow", "final-hidden"]),
    );
    call(
        s,
        &["rope", "final-hidden"],
        "qwen3.free-f32-before",
        "result-hidden",
    );
    pack(
        s,
        &names(&["final-cache", "final-shadow", "result-hidden"]),
        "result",
    );
    s.push_str("  ({[.result] [.truth]} *.intro)");
    finish(s);
    let specs = weight_specs(config)?;
    let ws = (0..specs.len())
        .map(|i| format!("w{i}"))
        .collect::<Vec<_>>();
    let mut inputs = vec![
        ("tokens", REF),
        ("cache", OWN),
        ("workspace", OWN),
        ("capacity", U64),
        ("start", U64),
    ];
    inputs.extend(ws.iter().map(|w| (w.as_str(), REF)));
    header(
        s,
        "qwen3",
        &inputs,
        "{((cap.own mem) val) ((cap.own mem) val) ((cap.own mem) val) ((cap.own mem) val)}",
    );
    let mut cursor = 1;
    for layer in 0..config.num_layers {
        let count = if full.contains(&layer) { 11 } else { 14 };
        pack(s, &ws[cursor..cursor + count], &format!("group{layer}"));
        cursor += count;
    }
    for (kind, layers) in [("full", full), ("linear", linear)] {
        pack(
            s,
            &layers
                .iter()
                .map(|l| format!("group{l}"))
                .collect::<Vec<_>>(),
            &format!("{kind}-weights"),
        );
    }
    s.push_str("  ([.workspace] [shadow.])\n");
    call(
        s,
        &["cache", "shadow", "capacity"],
        "qwen35.import",
        "shadow-ready",
    );
    call(
        s,
        &[&c(config.hidden_size)],
        "stdlib.qwen3.memory.alloc-bf16",
        "initial-hidden",
    );
    pack(
        s,
        &names(&["cache", "shadow-ready", "initial-hidden"]),
        "initial-state",
    );
    call(
        s,
        &[
            "tokens",
            "w0",
            "full-weights",
            "linear-weights",
            "capacity",
            "start",
        ],
        "partial.stdlib.qwen35.token.fold.6",
        "token-callback",
    );
    s.push_str("  ({([.tokens] stdlib.qwen3.memory.ref-count-u64 stdlib.qwen3.memory.fold-count-u32) (unit.intro meta.master.end32)} smolcat.value.named_at [token-end.])\n  ({[.token-end] [.initial-state] [.token-callback] core.truth} core.fold.bounded [final-state proof.])\n  ([.proof] unsafe.discard unit.elim)\n");
    unpack(
        s,
        "final-state",
        &names(&["cache-final", "shadow-final", "hidden-final"]),
    );
    call(
        s,
        &["cache-final", "shadow-final", "capacity"],
        "qwen35.persist",
        "cache-persisted",
    );
    s.push_str("  ([.cache-persisted] [cache-result.])\n");
    call(
        s,
        &[
            "hidden-final",
            &ws[ws.len() - 2],
            &c(config.hidden_size),
            &floats(config.hidden_size),
            &c(1),
        ],
        "qwen3.fused.normalize",
        "normalized",
    );
    free(s, "hidden-final", "normalized", "normalized-final");
    call(
        s,
        &[
            "normalized-final",
            &ws[ws.len() - 1],
            &c(config.hidden_size),
            &c(config.vocabulary_size),
        ],
        "qwen3.fused.project",
        "logits",
    );
    free(s, "normalized-final", "logits", "logits-final");
    call(s, &["logits-final"], "qwen3.argmax", "token");
    s.push_str("  {[.cache-result] [.shadow-final] [.logits-final] [.token]}");
    finish(s);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn qwen35_dense_compiles() -> Result<()> {
        let mut c = ModelConfig::qwen35_0_8b();
        c.hidden_size = 32;
        c.intermediate_size = 64;
        c.num_layers = 4;
        c.num_attention_heads = 2;
        c.num_key_value_heads = 1;
        c.head_dim = 16;
        c.rope_dimension_count = 8;
        c.vocabulary_size = 33;
        c.linear_state_size = Some(16);
        c.linear_group_count = Some(1);
        c.linear_time_step_rank = Some(2);
        c.linear_inner_size = Some(32);
        let options =
            InferenceOptions::new(InferencePrecision::Bf16).with_codegen(CodegenKind::Experimental);
        let source = source(&c, options)?;
        let module = compile(
            crate::catena::runtime_sources_for_options(&source, options),
            GpuDialect::Hip,
        )?;
        assert_eq!(
            module
                .functions
                .iter()
                .find(|f| f.source_name == "qwen3")
                .unwrap()
                .outputs
                .len(),
            4
        );
        Ok(())
    }
}
