//! Batched talker prefill. The cache uses the same layer / plane / row layout as decode.
use super::kernels::*;
use crate::{
    catena,
    qwen::{c, call, header, unpack},
    spec::ModelConfig,
};

const PAIR: &str = "({((cap.own mem) val) ((cap.own mem) val)} *)";
const LAYER: &str = "stdlib.qwen-tts.prefill-layer";
const PROJECT: &str = "stdlib.qwen35.batch.wmma-project";

pub(super) fn emit(s: &mut String, config: &ModelConfig, weights: &str) {
    s.push_str(&catena::definitions(
        include_str!("../catena/omnivoice.hex"),
        &[
            "stdlib.omnivoice.identity-gate.",
            "stdlib.omnivoice.rope-table.",
            "stdlib.omnivoice.rotate.",
        ],
    ));
    s.push_str(include_str!("../catena/lfm3_online_attention.hex"));

    let width = config.hidden_size;
    let query = config.query_size();
    let kv = config.key_value_size();
    let dim = config.head_dim;
    let ff = config.intermediate_size;
    step_header(
        s,
        LAYER,
        &[
            ("weights", weights),
            ("rows", U64),
            ("capacity", U64),
            ("rope", OWN),
            ("identity", OWN),
        ],
        PAIR,
    );
    s.push_str("  ([.state] *.elim [cache hidden.])\n");
    call(
        s,
        &["weights", "i32"],
        "stdlib.qwen3.layer-weights",
        "selected",
    );
    unpack(
        s,
        "selected",
        &[
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
        ]
        .map(str::to_owned),
    );
    let norm =
        |s: &mut String, input: &str, weight: &str, columns: usize, rows: &str, out: &str| {
            call(
                s,
                &[input, weight, &c(columns), &f(columns as f32), rows],
                "qwen3.fused.normalize",
                out,
            );
        };
    let project = |s: &mut String, input: &str, weight: &str, ci, co, out: &str| {
        call(s, &[input, weight, &c(ci), &c(co)], PROJECT, out);
    };
    norm(s, "hidden", "input-norm", width, "rows", "normalized");
    for (name, features) in [("q", query), ("k", kv)] {
        let raw = format!("{name}-raw");
        let weight = format!("w{name}");
        project(s, "normalized", &weight, width, features, &raw);
        let heads = format!("{name}-heads");
        mem(s, "u64.mul", &["rows", &c(features / dim)], &heads);
        let normalized = format!("{name}-normalized");
        let norm_weight = format!("{name}-norm");
        norm(s, &raw, &norm_weight, dim, &heads, &normalized);
        let count = format!("{name}-elements");
        mem(s, "u64.mul", &["rows", &c(features)], &count);
        let rotated = format!("{name}-rotated");
        call(
            s,
            &[&normalized, "rope", &c(features), &count, &count],
            "stdlib.omnivoice.rotate.run",
            &rotated,
        );
        free(s, &[&raw, &normalized], &rotated, name, FREE_BF16);
    }
    project(s, "normalized", "wv", width, kv, "v-raw");
    free(s, &["normalized"], "v-raw", "v", FREE_BF16);

    mem(s, "u64.mul", &["capacity", &c(kv)], "plane");
    mem(s, "u64.mul", &["i", &c(2)], "layer-plane");
    mem(s, "u64.mul", &["layer-plane", "plane"], "key-offset");
    mem(s, "u64.add", &["key-offset", "plane"], "value-offset");
    call(s, &["k", "cache", "key-offset"], "qwen3.append", "cache-k");
    call(
        s,
        &["v", "cache-k", "value-offset"],
        "qwen3.append",
        "cache-v",
    );
    free(s, &["k", "v"], "cache-v", "cache-ready", FREE_BF16);

    call(
        s,
        &[
            "q",
            "cache-ready",
            "cache-ready",
            "rows",
            "rows",
            &c(config.num_attention_heads),
            &c(config.num_key_value_heads),
            &f((dim as f32).sqrt().recip()),
            &c(1),
            &c(0),
            "identity",
            &c(0),
            "key-offset",
            "value-offset",
        ],
        "stdlib.lfm3.online.text.d128.run",
        "attended",
    );
    free(s, &["q"], "attended", "context", FREE_BF16);
    project(s, "context", "wo", query, width, "projected");
    call(s, &["hidden", "projected"], "qwen3.add", "residual-raw");
    free(
        s,
        &["hidden", "context", "projected"],
        "residual-raw",
        "residual",
        FREE_BF16,
    );
    norm(s, "residual", "post-norm", width, "rows", "ffn");
    project(s, "ffn", "wg", width, ff, "gate");
    project(s, "ffn", "wu", width, ff, "up");
    call(s, &["gate", "up"], "qwen3.swiglu", "activated");
    free(s, &["ffn", "gate", "up"], "activated", "gated", FREE_BF16);
    project(s, "gated", "wd", ff, width, "down");
    call(s, &["residual", "down"], "qwen3.add", "added");
    free(
        s,
        &["gated", "down", "residual"],
        "added",
        "ready",
        FREE_BF16,
    );
    call(s, &["cache-ready", "ready"], "*.intro", "result");
    step_finish(s, "result");

    header(
        s,
        "stdlib.qwen-tts.prefill",
        &[
            ("weights", weights),
            ("rows", U64),
            ("capacity", U64),
            ("cache", OWN),
            ("prompt", OWN),
        ],
        PAIR,
    );
    mem(s, "u64.mul", &["rows", &c(dim)], "rope-count");
    call(
        s,
        &["rope-count", "rope-count"],
        "stdlib.omnivoice.rope-table.run",
        "rope",
    );
    mem(
        s,
        "u64.mul",
        &["rows", &c(config.num_attention_heads)],
        "gate-count",
    );
    call(
        s,
        &["gate-count", "gate-count"],
        "stdlib.omnivoice.identity-gate.run",
        "identity",
    );
    call(s, &["cache", "prompt"], "*.intro", "initial");
    fold(
        s,
        LAYER,
        &["weights", "rows", "capacity", "rope", "identity"],
        &u(config.num_layers),
        "initial",
        "prefilled",
    );
    s.push_str("  ([.prefilled] *.elim [prefilled-cache hidden.])\n");
    mem(s, "u64.sub", &["rows", &c(1)], "last");
    kernel(
        s,
        "row",
        &["hidden", "last", &c(width), &c(width)],
        "last-row",
    );
    free(
        s,
        &["hidden", "identity"],
        "last-row",
        "hidden-bf",
        FREE_BF16,
    );
    free(s, &["rope"], "hidden-bf", "final-hidden", FREE_F32);
    finish(s, "({[.prefilled-cache] [.final-hidden]} *.intro)");
}
