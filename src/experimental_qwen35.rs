//! Dense and MoE Qwen3.5/3.6 hybrid graphs for the experimental compiler.
use super::*;
use crate::spec::LayerType;
#[path = "experimental_qwen35_prefill.rs"]
mod prefill;

const SCRATCH: [&str; 7] = [
    "scratch0", "scratch1", "scratch2", "scratch3", "scratch4", "scratch5", "scratch6",
];
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

fn emit_moe(s: &mut String, c0: &ModelConfig, _shadow_bytes: usize) -> Result<()> {
    let active = c0.experts_per_token * c0.intermediate_size;
    let shared = c0.shared_expert_intermediate_size.unwrap();
    let mut hex = [
        include_str!("catena/experimental/qwen35_moe.hex"),
        include_str!("catena/experimental/qwen35_batch_moe.hex"),
    ]
    .join("\n");
    for (key, value) in [
        ("EXPERT_COUNT", c0.num_experts),
        ("TOP_K", c0.experts_per_token),
        ("HIDDEN_SIZE", c0.hidden_size),
        ("HIDDEN_PAIRS", c0.hidden_size / 2),
        ("HIDDEN_STAGES", c0.hidden_size.trailing_zeros() as usize),
        ("INTERMEDIATE_SIZE", c0.intermediate_size),
        ("INTERMEDIATE_PAIRS", c0.intermediate_size / 2),
        ("ACTIVE_SIZE", active),
        ("ACTIVE_WIDTH_PAIRS", active / 2),
        ("TOTAL_ACTIVE_SIZE", active + shared),
        ("SHARED_PAIRS", shared / 2),
        ("ROUTER_ROWS", c0.num_experts + 1),
    ] {
        hex = hex.replace(&format!("{{{{{key}}}}}"), &format!("{value:016X}"));
    }
    for (key, operation) in [
        ("ROUTER_F32", "router-f32"),
        ("SHARED_ROUTER_F32", "shared-router-f32"),
        ("IDS_U64", "ids-u64"),
        ("PROBS_F32", "probs-f32"),
        ("ACTIVE_BF16", "active-bf16"),
        ("ACTIVE_PAIRS", "active-pairs"),
        ("SHARED_ACTIVE_PAIRS", "shared-active-pairs"),
    ] {
        hex = hex.replace(
            &format!("(const.u64.0x{{{{{key}}}}})"),
            &format!("(unit.intro stdlib.qwen35.layout.{operation})"),
        );
    }
    ensure!(!hex.contains("{{"), "unresolved MoE kernel constant");
    s.push_str(&hex);
    let weights = [
        "router",
        "gate",
        "up",
        "down",
        "shared-router",
        "shared-gate",
        "shared-up",
        "shared-down",
    ];
    let mut args = vec![("input", OWN), ("residual", OWN), ("scratch", OWN)];
    args.extend(weights.iter().map(|w| (*w, REF)));
    header(s, "qwen35.moe.ffn", &args, &format!("({{{OWN} {OWN}}} *)"));
    call(
        s,
        &["input", "router", "shared-router", "scratch"],
        "qwen35.moe.router",
        "routed",
    );
    call(s, &["routed"], "qwen35.moe.select", "selected");
    call(
        s,
        &[
            "input",
            "gate",
            "up",
            "shared-gate",
            "shared-up",
            "selected",
        ],
        "qwen35.moe.gate",
        "activated",
    );
    call(
        s,
        &["activated", "down", "shared-down", "residual"],
        "qwen35.moe.down",
        "output",
    );
    free(s, "input", "output", "output-after-input");
    free(s, "residual", "output-after-input", "result");
    s.push_str("  ([.result] meta.master.complete-own [completed phase.])\n  ({[.activated] [.phase]} meta.master.after-own [scratch-ready.])\n  ({[.scratch-ready] [.completed]} *.intro)");
    finish(s);
    Ok(())
}

fn emit_attention_branches(s: &mut String) {
    let output = format!("({{{OWN} {OWN}}} *)");
    for (kind, proof) in [
        ("online", "((master.predicate prop.IsTrue) |-)"),
        (
            "staged",
            "(({(master.predicate prop.IsTrue)} stdlib.prelude.prop.Not) |-)",
        ),
    ] {
        let pair = format!("({{{TRIPLE} ({{{proof} 1}} *)}} *)");
        header(
            s,
            &format!("qwen35.attention.{kind}"),
            &[
                ("keys", U64),
                ("key-offset", U64),
                ("value-offset", U64),
                ("pair", &pair),
            ],
            &output,
        );
        s.push_str("  ([.pair] *.elim [state proofs.])\n  ([.proofs] *.elim [proof unit.])\n  ([.proof] unsafe.discard unit.elim)\n  ([.unit] unit.elim)\n");
        unpack(s, "state", &names(&["query", "cache", "qg"]));
        if kind == "online" {
            call(
                s,
                &["query", "cache", "qg", "keys", "key-offset", "value-offset"],
                "qwen35.online",
                "context",
            );
            free(s, "query", "context", "context-query");
            free(s, "qg", "context-query", "result");
        } else {
            call(
                s,
                &["query", "cache", "keys", "key-offset"],
                "qwen3.scores",
                "scores",
            );
            free(s, "query", "scores", "scores-ready");
            call(
                s,
                &["scores-ready", "keys"],
                "qwen35.softmax",
                "probabilities",
            );
            call(
                s,
                &["scores-ready", "probabilities"],
                "qwen3.free-f32-before",
                "prob-ready",
            );
            call(
                s,
                &["prob-ready", "cache", "keys", "value-offset"],
                "qwen35.context",
                "context",
            );
            free(s, "prob-ready", "context", "context-ready");
            call(s, &["qg", "context-ready"], "qwen35.gate-context", "gated");
            free(s, "qg", "gated", "gated-ready");
            free(s, "context-ready", "gated-ready", "result");
        }
        s.push_str("  ([.result] meta.master.complete-own [ready phase.])\n  ({[.cache] [.phase]} meta.master.after-own [cache-ready.])\n  ({[.cache-ready] [.ready]} *.intro)");
        finish(s);
    }
}

fn replace_moe_decoder(s: &mut String, name: &str, linear: bool) -> Result<()> {
    let start = s.find(&format!("(def program {name} :")).unwrap();
    let end = start + s[start..].find("\n ))").unwrap() + 5;
    let old = &s[start..end];
    let body_start = old.find("\n  (").unwrap() + 1;
    let gate = old.find("} qwen3.fused.gate [").unwrap();
    let ffn = old[..gate].rfind("\n").unwrap() + 1;
    let mut new = String::new();
    let full_ws = [
        "input-norm",
        "q-weight",
        "k-weight",
        "v-weight",
        "o-weight",
        "q-norm",
        "k-norm",
        "post-norm",
    ];
    let linear_ws = [
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
    ];
    let moe_ws = [
        "router",
        "gate",
        "up",
        "down",
        "shared-router",
        "shared-gate",
        "shared-up",
        "shared-down",
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
    args.extend(
        if linear {
            linear_ws.as_slice()
        } else {
            full_ws.as_slice()
        }
        .iter()
        .map(|w| (*w, REF)),
    );
    args.extend(moe_ws.iter().map(|w| (*w, REF)));
    header(&mut new, name, &args, TRIPLE);
    let mut body = old[body_start..ffn].to_owned();
    args.insert(7, ("sequence", U64));
    args.splice(8..8, SCRATCH.iter().map(|name| (*name, OWN)));
    new.clear();
    header(&mut new, name, &args, TRIPLE);
    if !linear {
        let start = body.find("  ({[.query] [.cache-ready] [.keys]").unwrap();
        let end = body.find("  ({[.full-context] [.o-weight]").unwrap();
        let mut attention = String::new();
        pack(
            &mut attention,
            &names(&["query", "cache-ready", "qg"]),
            "attention-input",
        );
        call(
            &mut attention,
            &["sequence", &c(1)],
            "stdlib.qwen3.memory.u64.equal",
            "decode",
        );
        attention.push_str("  ({[.decode] (unit.intro meta.master.predicate)} smolcat.value.named_at :.forget [attention-condition.])\n");
        for kind in ["online", "staged"] {
            call(
                &mut attention,
                &["keys", "key-offset", "value-offset"],
                &format!("partial.qwen35.attention.{kind}.3"),
                &format!("{kind}-attention"),
            );
        }
        call(
            &mut attention,
            &[
                "attention-condition",
                "attention-input",
                "online-attention",
                "staged-attention",
                "(unit.intro)",
            ],
            "core.if",
            "attention-output",
        );
        attention.push_str("  ([.attention-output] *.elim [attention-cache full-context.])\n");
        body.replace_range(start..end, &attention);
    }
    // The four linear and three full-attention matrices share one projection launch.
    if linear {
        let start = body.find("  ({[.normalized] [.qkv-weight]").unwrap();
        let end = start + body[start..].find("  ({[.capacity]").unwrap();
        let mut projected = String::new();
        call(
            &mut projected,
            &[
                "normalized",
                "qkv-weight",
                "z-weight",
                "alpha-weight",
                "beta-weight",
                "scratch0",
                "scratch1",
                "scratch2",
                "scratch3",
            ],
            "qwen35.linear-projections-reuse",
            "qkv z alpha beta",
        );
        body.replace_range(start..end, &projected);
    } else {
        let start = body.find("  ({[.normalized] [.q-weight]").unwrap();
        let end = start + body[start..].find("  ({[.query-raw] [.q-norm]").unwrap();
        let mut projected = String::new();
        call(
            &mut projected,
            &[
                "normalized",
                "q-weight",
                "k-weight",
                "v-weight",
                "scratch0",
                "scratch1",
                "scratch2",
            ],
            "qwen35.full-projections-reuse",
            "qg key-raw value",
        );
        call(&mut projected, &["qg"], "qwen35.query", "query-raw");
        body.replace_range(start..end, &projected);
        let start = body.find("  ({[.qg] } qwen35.query").unwrap();
        let offsets = start + body[start..].find("  ({[.layer]").unwrap();
        let append = offsets + body[offsets..].find("  ({[.key] [.cache]").unwrap();
        let end = append + body[append..].find("  ({[.position]").unwrap();
        let mut prep = body[offsets..append].to_owned();
        call(
            &mut prep,
            &[
                "qg",
                "key-raw",
                "value",
                "rope",
                "cache",
                "q-norm",
                "k-norm",
                "key-target",
                "value-target",
                "scratch4",
            ],
            "qwen35.full-prepare-reuse",
            "query cache-prepared",
        );
        free(&mut prep, "key-raw", "cache-prepared", "cache-prepared-k");
        free(&mut prep, "value", "cache-prepared-k", "cache-ready");
        body.replace_range(start..end, &prep);
    }
    if linear {
        let start = body.find("  ({[.qkv] [.cache] [.conv-weight]").unwrap();
        let end = start
            + body[start..]
                .find("  ({[.convolved] [.query] [.key]")
                .unwrap();
        let mut prep = String::new();
        call(
            &mut prep,
            &[
                "qkv",
                "cache",
                "conv-weight",
                "conv-base",
                "position",
                "scratch4",
                "scratch5",
                "scratch6",
            ],
            "qwen35.linear-prepare-reuse",
            "convolved query key",
        );
        body.replace_range(start..end, &prep);
        let target = "[.shadow] [.shadow-base] } qwen35.delta";
        let replacement =
            "[.shadow] [.shadow-base] [.cache] [.conv-base] [.sequence] } qwen35.delta-dispatch";
        ensure!(body.contains(target), "missing linear DeltaNet call");
        body = body.replace(target, replacement);
    }
    new.push_str(&body);
    let residual = if linear {
        "attention-ready"
    } else {
        "attention-result"
    };
    let mut ffn_args = vec!["ffn-input", residual, "shadow"];
    ffn_args.extend(moe_ws);
    call(&mut new, &ffn_args, "qwen35.moe.ffn", "ffn-result");
    new.push_str("  ([.ffn-result] *.elim [shadow-result hidden-result.])\n  ([.hidden-result] meta.master.complete-own [completed phase.])\n");
    call(
        &mut new,
        &[if linear { "cache" } else { "attention-cache" }, "phase"],
        "meta.master.after-own",
        "cache-result",
    );
    pack(
        &mut new,
        &names(&["cache-result", "shadow-result", "completed"]),
        "result",
    );
    new.push_str("  [.result]");
    finish(&mut new);
    s.replace_range(start..end, &new);
    Ok(())
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
    base.num_experts = 0;
    base.experts_per_token = 0;
    base.shared_expert_intermediate_size = None;
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
    let mut fused_delta = include_str!("catena/experimental/qwen35_fused_delta.hex").to_owned();
    if key_dim == 128 {
        let start = fused_delta
            .find("(def program stdlib.qwen35.fused-delta.kernel :")
            .unwrap();
        let end = fused_delta
            .find("(def program stdlib.qwen35.fused-delta.host :")
            .unwrap();
        fused_delta.replace_range(
            start..end,
            include_str!("catena/experimental/qwen35_register_delta.hex"),
        );
    }
    let mut kernels = [
        include_str!("catena/experimental/qwen35.hex"),
        include_str!("catena/experimental/qwen35_batch_projection.hex"),
        include_str!("catena/experimental/qwen35_batch_prepare.hex"),
        include_str!("catena/experimental/qwen35_batch_gated.hex"),
        include_str!("catena/experimental/qwen35_batch_scan.hex"),
        include_str!("catena/experimental/qwen35_batch_full_prepare.hex"),
        include_str!("catena/experimental/qwen35_batch_staged_attention.hex"),
        include_str!("catena/experimental/qwen35_attention.hex"),
        include_str!("catena/experimental/qwen35_online.hex"),
        include_str!("catena/experimental/qwen35_cooperative_norm.hex"),
        fused_delta.as_str(),
        include_str!("catena/experimental/qwen35_cached_delta.hex"),
    ]
    .join("\n");
    if config.architecture == ModelArchitecture::Qwen35Moe {
        kernels.push_str(include_str!(
            "catena/experimental/qwen35_group_projection.hex"
        ));
    }
    if config.architecture == ModelArchitecture::Qwen35Moe {
        kernels.push_str(include_str!(
            "catena/experimental/qwen35_linear_prepare.hex"
        ));
    }
    if config.architecture == ModelArchitecture::Qwen35Moe {
        kernels.push_str(include_str!("catena/experimental/qwen35_full_prepare.hex"));
    }
    for (key, value) in [
        ("KEY_DIM", key_dim),
        ("KEY_HEADS", key_heads),
        ("VALUE_HEADS", value_heads),
        ("KEY_SIZE", config.linear_key_size().unwrap()),
        ("VALUE_DIM", value_dim),
        ("VALUE_SIZE", value_size),
        ("VALUE_BLOCKS", value_size / 8),
        ("QKV_SIZE", qkv_size),
        ("CONV_HEADS", qkv_size.div_ceil(key_dim)),
        ("QUERY_SIZE", config.query_size()),
        ("DOUBLE_QUERY_SIZE", config.query_size() * 2),
        ("HIDDEN_PAIRS", config.hidden_size / 2),
        ("KEY_VALUE_SIZE", config.key_value_size()),
        ("QUERY_HEADS", config.num_attention_heads),
        (
            "QUERY_HEADS_PER_KV_HEAD",
            config.num_attention_heads / config.num_key_value_heads,
        ),
        ("HEAD_DIM", config.head_dim),
        ("ROPE_DIM", config.rope_dimension_count),
        ("ROPE_HALF_DIM", config.rope_dimension_count / 2),
        ("PAD_HEAD_DIM", config.head_dim.next_power_of_two()),
        (
            "HEAD_STAGES",
            config.head_dim.next_power_of_two().trailing_zeros() as usize,
        ),
        (
            "FULL_PREP_HEADS",
            config.num_attention_heads + config.num_key_value_heads,
        ),
        (
            "HEADS_PER_KV",
            config.num_attention_heads / config.num_key_value_heads,
        ),
        ("DOT_STEPS", config.head_dim.div_ceil(32)),
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
        ("SCALE_BITS", (config.head_dim as f32).sqrt().recip()),
        ("HEAD_DIM_F32", config.head_dim as f32),
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
    let moe = config.architecture == ModelArchitecture::Qwen35Moe;
    if moe {
        let shadow_bytes = recurrent
            .checked_mul(config.num_layers)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| anyhow::anyhow!("experimental recurrent workspace overflow"))?;
        emit_moe(&mut s, config, shadow_bytes)?;
        emit_attention_branches(&mut s);
        replace_moe_decoder(&mut s, "qwen3.decoder-layer", false)?;
        replace_moe_decoder(&mut s, "qwen35.linear-decoder", true)?;
    }
    if moe {
        s = s
            .replace("} qwen35.l2 [", "} qwen35.coop-l2 [")
            .replace("} qwen35.gated-norm [", "} qwen35.coop-gated [")
            .replace("} qwen35.delta [", "} qwen35.fused-delta [");
    }
    if !moe && key_dim == 128 {
        s = s.replace("} qwen35.delta [", "} qwen35.fused-delta [");
    }
    let full_count = if moe { 16 } else { 11 };
    let linear_count = if moe { 19 } else { 14 };
    let full_group = nested_product(&vec![REF.to_owned(); full_count]);
    let linear_group = nested_product(&vec![REF.to_owned(); linear_count]);
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
        let mut branch_args = vec![
            ("weights", group),
            ("rope", OWN),
            ("capacity", U64),
            ("position", U64),
            ("layer", U64),
            ("sequence", U64),
        ];
        if moe {
            branch_args.extend(SCRATCH.iter().map(|name| (*name, OWN)));
        }
        branch_args.push(("pair", &pair));

        header(
            &mut s,
            &format!("stdlib.qwen35.branch-{kind}"),
            &branch_args,
            TRIPLE,
        );
        s.push_str("  ([.pair] *.elim [state proofs.])\n  ([.proofs] *.elim [proof unit.])\n  ([.proof] unsafe.discard unit.elim)\n  ([.unit] unit.elim)\n");
        unpack(&mut s, "state", &names(&["cache", "shadow", "hidden"]));
        let ws = (0..if kind == "full" {
            full_count
        } else {
            linear_count
        })
            .map(|i| format!("w{i}"))
            .collect::<Vec<_>>();
        unpack(&mut s, "weights", &ws);
        let mut args = vec!["hidden", "cache", "rope", "capacity", "position", "layer"];
        if kind == "linear" || moe {
            args.push("shadow");
        }
        if moe {
            args.push("sequence");
            args.extend(SCRATCH);
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
        if kind == "full" && !moe {
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
    let mut captures = vec![
        ("full-weights", full_tree.as_str()),
        ("linear-weights", linear_tree.as_str()),
        ("rope", OWN),
        ("capacity", U64),
        ("position", U64),
        ("sequence", U64),
    ];
    if moe {
        captures.extend(SCRATCH.iter().map(|name| (*name, OWN)));
    }
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
                "sequence",
            ]
            .into_iter()
            .chain(if moe { SCRATCH.to_vec() } else { vec![] })
            .collect::<Vec<_>>(),
            &format!(
                "partial.stdlib.qwen35.branch-{kind}.{}",
                if moe { 13 } else { 6 }
            ),
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
        let padded = c(width.next_power_of_two());
        let stages = c(width.next_power_of_two().trailing_zeros() as usize);
        let pattern = format!("{columns} {float} {} }} qwen3.fused.normalize", c(rows));
        s = s.replace(
            &pattern,
            &format!(
                "{columns} {float} {} {stages} {padded} }} qwen35.normalize",
                c(rows)
            ),
        );
        let pattern = format!("{columns} {float} {}}} qwen3.fused.normalize", c(rows));
        s = s.replace(
            &pattern,
            &format!(
                "{columns} {float} {} {stages} {padded}}} qwen35.normalize",
                c(rows)
            ),
        );
    }
    if moe {
        for name in [
            "qkv",
            "z",
            "alpha",
            "beta",
            "query",
            "key",
            "convolved",
            "qg",
            "key-raw",
            "value",
        ] {
            let prefix = format!("  ({{[.{name}]");
            s = s
                .lines()
                .map(|line| {
                    if line.starts_with(&prefix) && line.contains("qwen3.free-bf16-before") {
                        line.replace("qwen3.free-bf16-before", "qwen3.keep-bf16-before")
                    } else {
                        line.to_owned()
                    }
                })
                .collect::<Vec<_>>()
                .join("\n");
        }
    }
    if key_dim == 128 {
        let mut batch = String::new();
        prefill::emit(&mut batch, config, &full_tree, &linear_tree);
        let entry = s.rfind("(def program qwen3 :").unwrap();
        s.insert_str(entry, &batch);
    }
    crate::session_graph::append_layout(&mut s, config, options);
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
    let moe = config.architecture == ModelArchitecture::Qwen35Moe;
    let mut captures = vec![
        ("tokens", REF),
        ("embedding", REF),
        ("full-weights", full_tree),
        ("linear-weights", linear_tree),
        ("capacity", U64),
        ("start", U64),
    ];
    if moe {
        captures.extend(SCRATCH.iter().map(|name| (*name, OWN)));
    }
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
    call(
        s,
        &["tokens"],
        "stdlib.qwen3.memory.ref-count-u64",
        "sequence",
    );
    pack(s, &names(&["cache", "shadow", "hidden"]), "initial");
    call(
        s,
        &[
            "full-weights",
            "linear-weights",
            "rope",
            "capacity",
            "position",
            "sequence",
        ]
        .into_iter()
        .chain(if moe { SCRATCH.to_vec() } else { vec![] })
        .collect::<Vec<_>>(),
        &format!(
            "partial.stdlib.qwen35.layer.fold.{}",
            if moe { 13 } else { 6 }
        ),
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
    if moe {
        inputs.splice(3..3, SCRATCH.iter().map(|name| (*name, OWN)));
    }
    inputs.extend(ws.iter().map(|w| (w.as_str(), REF)));
    header(
        s,
        "qwen3",
        &inputs,
        &format!("{{{}}}", vec![OWN; if moe { 11 } else { 4 }].join(" ")),
    );
    let mut cursor = 1;
    for layer in 0..config.num_layers {
        let count = match (
            config.architecture == ModelArchitecture::Qwen35Moe,
            full.contains(&layer),
        ) {
            (true, true) => 16,
            (true, false) => 19,
            (false, true) => 11,
            (false, false) => 14,
        };
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
    let moe = config.architecture == ModelArchitecture::Qwen35Moe;
    if moe {
        call(
            s,
            &["tokens"],
            "stdlib.qwen3.memory.ref-count-u64",
            "sequence",
        );
    }

    call(
        s,
        &if moe {
            vec!["cache", "shadow", "capacity", "sequence"]
        } else {
            vec!["cache", "shadow", "capacity"]
        },
        if moe {
            "qwen35.import-optional"
        } else {
            "qwen35.import"
        },
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
    if config.linear_state_size == Some(128) {
        call(
            s,
            &[
                "tokens",
                "w0",
                "full-weights",
                "linear-weights",
                "capacity",
                "start",
            ]
            .into_iter()
            .chain(if moe { SCRATCH.to_vec() } else { vec![] })
            .chain(["initial-state"])
            .collect::<Vec<_>>(),
            "stdlib.qwen35.text-dispatch",
            "final-state",
        );
    } else {
        call(
            s,
            &[
                "tokens",
                "w0",
                "full-weights",
                "linear-weights",
                "capacity",
                "start",
            ]
            .into_iter()
            .chain(if moe { SCRATCH.to_vec() } else { vec![] })
            .collect::<Vec<_>>(),
            &format!(
                "partial.stdlib.qwen35.token.fold.{}",
                if moe { 13 } else { 6 }
            ),
            "token-callback",
        );
        s.push_str("  ({([.tokens] stdlib.qwen3.memory.ref-count-u64 stdlib.qwen3.memory.fold-count-u32) (unit.intro meta.master.end32)} smolcat.value.named_at [token-end.])\n  ({[.token-end] [.initial-state] [.token-callback] core.truth} core.fold.bounded [final-state proof.])\n  ([.proof] unsafe.discard unit.elim)\n");
    }
    unpack(
        s,
        "final-state",
        &names(&["cache-final", "shadow-final", "hidden-final"]),
    );
    call(
        s,
        &if moe {
            vec!["cache-final", "shadow-final", "capacity", "sequence"]
        } else {
            vec!["cache-final", "shadow-final", "capacity"]
        },
        if moe {
            "qwen35.persist-optional"
        } else {
            "qwen35.persist"
        },
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
    call(
        s,
        &["logits-final"],
        if config.architecture == ModelArchitecture::Qwen35Moe {
            "qwen3.fused.argmax"
        } else {
            "qwen3.argmax"
        },
        "token",
    );
    let mut result = vec!["cache-result", "shadow-final"];
    if moe {
        result.extend(SCRATCH);
    }
    result.extend(["logits-final", "token"]);
    writeln!(
        s,
        "  {{{}}}",
        result
            .iter()
            .map(|name| format!("[.{name}]"))
            .collect::<Vec<_>>()
            .join(" ")
    )
    .unwrap();
    finish(s);
    Ok(())
}

/// Reuse the hybrid layer fold with supplied embeddings and independent MRoPE.
pub(crate) fn multimodal_source(config: &ModelConfig, options: InferenceOptions) -> Result<String> {
    let half = config.rope_dimension_count / 2;
    let sections = config.mrope_sections;
    ensure!(
        sections.iter().sum::<usize>() == half
            && sections[1] * 3 <= half + 2
            && sections[2] * 3 <= half + 2,
        "unsupported interleaved MRoPE sections {sections:?}"
    );
    let mut s = source(config, options)?;
    let mut helpers = include_str!("catena/experimental/qwen35_multimodal.hex").to_owned();
    for (name, value) in [
        ("HIDDEN_SIZE", config.hidden_size),
        ("ROPE_DIM", config.rope_dimension_count),
        ("ROPE_HALF_DIM", half),
        ("HEIGHT_END", sections[1] * 3),
        ("WIDTH_END", sections[2] * 3),
    ] {
        helpers = helpers.replace(&format!("{{{{{name}}}}}"), &format!("{value:016X}"));
    }
    let entry = &s[s.rfind("(def program qwen3 :").unwrap()..];
    let body = &entry[entry.find("\n  ({").unwrap()..];
    let allocation = body
        .find(&format!(
            "  ({{{} }} stdlib.qwen3.memory.alloc-bf16",
            c(config.hidden_size)
        ))
        .unwrap();
    let tail = body.find("  ({[.cache-final] [.shadow-final]").unwrap();
    let mut mm = String::new();
    let specs = weight_specs(config)?;
    let ws = (0..specs.len())
        .map(|i| format!("w{i}"))
        .collect::<Vec<_>>();
    let mut args = vec![
        ("hidden", OWN),
        ("positions", REF),
        ("cache", OWN),
        ("workspace", OWN),
        ("capacity", U64),
        ("start", U64),
        ("sequence", U64),
    ];
    let moe = config.architecture == ModelArchitecture::Qwen35Moe;
    if moe {
        args.splice(4..4, SCRATCH.iter().map(|name| (*name, OWN)));
    }
    args.extend(ws.iter().map(|w| (w.as_str(), REF)));
    header(
        &mut mm,
        "qwen3.multimodal",
        &args,
        &format!("{{{}}}", vec![OWN; if moe { 11 } else { 4 }].join(" ")),
    );
    mm.push_str(&body[..allocation].replace(
        "  ({[.tokens] } stdlib.qwen3.memory.ref-count-u64 [sequence.])\n",
        "",
    ));
    call(&mut mm, &["positions"], "qwen3.mm-rope-table", "rope");
    pack(
        &mut mm,
        &names(&["cache", "shadow-ready", "hidden"]),
        "initial-state",
    );
    call(
        &mut mm,
        &[
            "full-weights",
            "linear-weights",
            "rope",
            "capacity",
            "start",
            "sequence",
        ]
        .into_iter()
        .chain(if moe { SCRATCH.to_vec() } else { vec![] })
        .collect::<Vec<_>>(),
        &format!(
            "partial.stdlib.qwen35.layer.fold.{}",
            if moe { 13 } else { 6 }
        ),
        "layer-callback",
    );
    writeln!(mm, "  ({{(const.u32.0x{:08X}) (unit.intro meta.master.end32)}} smolcat.value.named_at [layer-end.])", config.num_layers).unwrap();
    mm.push_str("  ({[.layer-end] [.initial-state] [.layer-callback] core.truth} core.fold.bounded [final-state proof.])\n  ([.proof] unsafe.discard unit.elim)\n");
    unpack(
        &mut mm,
        "final-state",
        &names(&["cache-final", "shadow-final", "decoded-hidden"]),
    );
    call(
        &mut mm,
        &["rope", "decoded-hidden"],
        "qwen3.free-f32-before",
        "hidden-final",
    );
    mm.push_str(&body[tail..]);
    s.push_str(&helpers);
    s.push_str(&mm);
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn qwen35_moe_compiles() -> Result<()> {
        let mut c = ModelConfig::qwen36_35b_a3b();
        c.hidden_size = 32;
        c.intermediate_size = 16;
        c.shared_expert_intermediate_size = Some(16);
        c.num_experts = 4;
        c.experts_per_token = 2;
        c.num_layers = 4;
        c.num_attention_heads = 2;
        c.num_key_value_heads = 1;
        c.head_dim = 16;
        c.rope_dimension_count = 8;
        c.mrope_sections = [2, 1, 1];
        c.vocabulary_size = 33;
        c.linear_state_size = Some(16);
        c.linear_group_count = Some(1);
        c.linear_time_step_rank = Some(2);
        c.linear_inner_size = Some(32);
        let options =
            InferenceOptions::new(InferencePrecision::Bf16).with_codegen(CodegenKind::Experimental);
        let source = multimodal_source(&c, options)?;
        let module = compile(
            crate::catena::runtime_sources_for_options(&source, options),
            GpuDialect::Hip,
        )?;
        assert_eq!(
            module
                .functions
                .iter()
                .find(|f| f.source_name == "qwen3.multimodal")
                .unwrap()
                .outputs
                .len(),
            11
        );
        Ok(())
    }
    #[test]
    fn qwen35_dense_compiles() -> Result<()> {
        let mut c = ModelConfig::qwen35_0_8b();
        c.hidden_size = 48;
        c.intermediate_size = 64;
        c.num_layers = 4;
        c.num_attention_heads = 2;
        c.num_key_value_heads = 1;
        c.head_dim = 24;
        c.rope_dimension_count = 8;
        c.vocabulary_size = 33;
        c.linear_state_size = Some(16);
        c.linear_group_count = Some(1);
        c.linear_time_step_rank = Some(2);
        c.linear_inner_size = Some(32);
        let options =
            InferenceOptions::new(InferencePrecision::Bf16).with_codegen(CodegenKind::Experimental);
        validate(&c, options, Some(GpuDialect::Hip))?;
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
