//! Native kernels for the shared Flux graphs. This module only renders Hex.
use anyhow::Result;

pub(super) fn bundle(source: &str) -> String {
    let mut out = include_str!("../catena/memory.hex").to_owned();
    for helper in [
        include_str!("../catena/numeric.hex"),
        include_str!("../catena/resources.hex"),
        include_str!("../catena/cooperative.hex"),
        include_str!("../catena/qwen35_batch_projection.hex"),
        include_str!("../catena/flux.hex"),
        source,
    ] {
        crate::qwen::fused::append_hex(&mut out, helper);
    }
    // Helpers remain private so source pruning compiles only reachable kernels.
    out = out
        .replace("flux.", "stdlib.flux.")
        .replace("stdlib.stdlib.flux.", "stdlib.flux.");
    for entry in ["encode-text", "denoise", "decode-image"] {
        out = out.replace(
            &format!("(def program stdlib.flux.{entry} :"),
            &format!("(def program flux.{entry} :"),
        );
    }
    out.replace(" qwen3.", " stdlib.flux.qwen3.")
        .replace("partial.qwen3.", "partial.stdlib.flux.qwen3.")
}

pub(super) fn text(cfg: &crate::spec::ModelConfig, debug: bool) -> Result<String> {
    use std::fmt::Write;
    let c = |n: usize| format!("(const.u64.0x{n:016X})");
    let f = |n: f32| format!("(const.u32.0x{:08X} u32.bitcast-f32)", n.to_bits());
    let own = "((cap.own mem) val)";
    let rf = "((cap.ref mem) val)";
    let h = cfg.hidden_size;
    let q = cfg.query_size();
    let kv = cfg.num_key_value_heads * cfg.head_dim;
    let ff = cfg.intermediate_size;
    let rows = super::text::SEQUENCE;
    let dim = cfg.head_dim;
    let mut out = crate::catena::rope_frequency_program(cfg);
    let mut body = String::new();
    let mut call = |args: String, op: &str, name: &str| {
        writeln!(body, " ({{{args}}} {op} [{name}.])").unwrap();
    };
    call(
        format!(
            "[.hidden] [.w0] [.w0] {} {} {} {} {} {}",
            c(rows),
            c(h),
            c(1),
            c(1),
            c(1),
            f(cfg.rms_norm_eps)
        ),
        "flux.norm-text-orr",
        "normalized",
    );
    for (weight, cols, name) in [(1, q, "query"), (2, kv, "key"), (3, kv, "value")] {
        call(
            format!(
                "[.normalized] [.w{weight}] {} {} {}",
                c(rows),
                c(h),
                c(cols)
            ),
            "flux.linear-bf16-or",
            name,
        );
    }
    for (input, weight, cols, heads, name) in [
        ("query", 5, q, cfg.num_attention_heads, "qn"),
        ("key", 6, kv, cfg.num_key_value_heads, "kn"),
    ] {
        call(
            format!(
                "[.{input}] [.w{weight}] [.w{weight}] {} {} {} {} {} {}",
                c(rows * heads),
                c(dim),
                c(1),
                c(1),
                c(1),
                f(cfg.rms_norm_eps)
            ),
            "flux.norm-text-orr",
            name,
        );
        call(
            format!("[.{name}] {} {} {}", c(cols), c(dim), c(rows)),
            "flux.text-rope",
            &format!("{name}-rotated"),
        );
    }
    call(
        format!(
            "[.qn-rotated] [.kn-rotated] [.value] {} {} {} {} {} {} {} [.valid]",
            c(rows),
            c(rows),
            c(cfg.num_attention_heads),
            c(cfg.num_key_value_heads),
            c(dim),
            f(1. / (dim as f32).sqrt()),
            c(1)
        ),
        "flux.attention-bf16-ooo",
        "context",
    );
    call(
        format!("[.context] [.w4] {} {} {}", c(rows), c(q), c(h)),
        "flux.linear-bf16-or",
        "projected",
    );
    call(
        format!("[.hidden] [.projected] {}", c(rows * h)),
        "flux.add-bf16-oo",
        "residual",
    );
    call(
        format!(
            "[.residual] [.w7] [.w7] {} {} {} {} {} {}",
            c(rows),
            c(h),
            c(1),
            c(1),
            c(1),
            f(cfg.rms_norm_eps)
        ),
        "flux.norm-text-orr",
        "ffn",
    );
    call(
        format!("[.ffn] [.w8] {} {} {}", c(rows), c(h), c(ff)),
        "flux.linear-bf16-or",
        "gate",
    );
    call(
        format!("[.ffn] [.w9] {} {} {}", c(rows), c(h), c(ff)),
        "flux.linear-bf16-or",
        "up",
    );
    // Separate gate/up projections retain the default SwiGLU rounding boundaries.
    call(
        format!("[.gate] [.up] {}", c(rows * ff)),
        "flux.text-swiglu",
        "gated",
    );
    call(
        format!("[.gated] [.w10] {} {} {}", c(rows), c(ff), c(h)),
        "flux.linear-bf16-or",
        "down",
    );
    call(
        format!("[.residual] [.down] {}", c(rows * h)),
        "flux.add-bf16-oo",
        "result",
    );
    let mut result = "result".to_owned();
    for (i, name) in [
        "hidden",
        "normalized",
        "query",
        "key",
        "value",
        "qn",
        "kn",
        "qn-rotated",
        "kn-rotated",
        "context",
        "projected",
        "residual",
        "ffn",
        "gate",
        "up",
        "gated",
        "down",
    ]
    .iter()
    .enumerate()
    {
        writeln!(
            body,
            " ({{[.{name}] [.{result}]}} qwen3.free-bf16-before [free{i}.])"
        )
        .unwrap();
        result = format!("free{i}");
    }
    body += &format!(" [.{result}]\n");
    writeln!(out,"(def program flux.text-layer : {{{own} {} (u64 val)}} -> {own} = ([hidden {} valid.]\n{body}))",vec![rf;11].join(" "),(0..11).map(|i|format!("w{i}")).collect::<Vec<_>>().join(" ")).unwrap();
    body = format!(
        " ({{[.ids] [.w0] {} {}}} flux.text-embedding [h0.])\n",
        c(h),
        c(rows)
    );
    for layer in 0..27 {
        writeln!(
            body,
            " ({{[.h{layer}] {} [.valid]}} flux.text-layer [h{}.])",
            (1 + layer * 11..1 + (layer + 1) * 11)
                .map(|i| format!("[.w{i}]"))
                .collect::<Vec<_>>()
                .join(" "),
            layer + 1
        )
        .unwrap();
        if super::text::CAPTURES.contains(&(layer + 1)) {
            writeln!(
                body,
                " ({{[.h{}] {}}} flux.copy-bf16-o [capture{}.])",
                layer + 1,
                c(rows * h),
                layer + 1
            )
            .unwrap();
        }
    }
    writeln!(body," ({{[.capture9] [.capture18] {} {} {}}} flux.concat-bf16-oo [first.])\n ({{[.first] [.capture27] {} {} {}}} flux.concat-bf16-oo [result.])",c(rows*h*2),c(h),c(h),c(rows*h*3),c(h*2),c(h)).unwrap();
    let mut result = "result".to_owned();
    let mut free = vec!["h27", "first"];
    if !debug {
        free.extend(["capture9", "capture18", "capture27"]);
    }
    for (i, name) in free.iter().enumerate() {
        writeln!(
            body,
            " ({{[.{name}] [.{result}]}} qwen3.free-bf16-before [final{i}.])"
        )
        .unwrap();
        result = format!("final{i}");
    }
    if debug {
        body += &format!(" {{[.{result}] [.capture9] [.capture18] [.capture27]}}\n");
    } else {
        body += &format!(" [.{result}]\n");
    }
    // Weight argument order is shared with TextEncoder's existing loader.
    let weights = 1 + 27 * 11;
    out += &super::text::weight_comments();
    writeln!(
        out,
        "(def program flux.encode-text : {{{rf} (u64 val) {}}} -> {} = ([ids valid {}.]\n{body}))",
        vec![rf; weights].join(" "),
        if debug {
            format!("{{{own} {own} {own} {own}}}")
        } else {
            own.to_owned()
        },
        (0..weights)
            .map(|i| format!("w{i}"))
            .collect::<Vec<_>>()
            .join(" ")
    )
    .unwrap();
    Ok(out)
}
