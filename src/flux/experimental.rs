//! Experimental kernels for the shared Flux graphs. This module only renders Hex.
use anyhow::{Context, Result};
const M: &str = "stdlib.qwen3.memory.";
pub(super) fn convert(source: &str) -> String {
    let mut source = source
        .replace("((cap.ref mem) val)", "__FLUX_REF__")
        .replace("((cap.own mem) val)", "__FLUX_OWN__")
        .replace("(cap.ref mem)", "((cap.ref mem) val)")
        .replace("(cap.own mem)", "((cap.own mem) val)")
        .replace("__FLUX_REF__", "((cap.ref mem) val)")
        .replace("__FLUX_OWN__", "((cap.own mem) val)");
    for (old, new) in [
        ("u64.ne bool.not", "u64.equal"),
        ("f32.neg", "f32.negate"),
        ("u64.lt", "u64.less"),
        ("u64.lte", "u64.less_equal"),
        ("f32.lt", "f32.less"),
        ("f32.select", "select-f32"),
        ("u64.select", "select-u64"),
        ("u64.add", "u64.add"),
        ("u64.sub", "u64.sub"),
        ("u64.mul", "u64.mul"),
        ("u64.div", "u64.div"),
        ("u64.rem", "u64.rem"),
        ("f32.add", "f32.add"),
        ("f32.sub", "f32.sub"),
        ("f32.mul", "f32.mul"),
        ("f32.div", "f32.div"),
    ] {
        // Legacy graphs contain bare operators, never the qualified adapters.
        source = source.replace(&format!(" {old} "), &format!(" {M}{new} "));
    }
    source
}
fn definition_end(source: &str, start: usize) -> usize {
    let mut depth = 0;
    for (i, b) in source.as_bytes().iter().enumerate().skip(start) {
        match b {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return i + 1;
                }
            }
            _ => {}
        }
    }
    panic!("unclosed generated Hex definition")
}
pub(super) fn graph(source: &str) -> Result<String> {
    let mut source = source.to_owned();
    while let Some(start) = source.find("(def program flux.constant-") {
        let end = definition_end(&source, start);
        let at = source[start..end].to_owned();
        let name = at
            .split_whitespace()
            .nth(2)
            .context("constant name")?
            .trim_end_matches("-at")
            .to_owned();
        let count_start = source[end..]
            .find("(const.u64.0x")
            .context("constant count")?
            + end
            + 13;
        let count = &source[count_start..count_start + 16];
        let body = at
            .split_once("([.index] ix.to-u64 [i.])")
            .context("constant body")?
            .1
            .trim();
        let body = body.strip_suffix("))").context("constant closing")?.trim();
        let body = format!(" ([.element] [i.])\n ({body} [value.])\n");
        let rendered = include_str!("../catena/experimental/flux_constant.hex")
            .replace("{{VALUE_BODY}}", &convert(&body))
            .replace("CONSTANT-NAME", name.strip_prefix("flux.").unwrap())
            .replace("{{COUNT}}", count);
        let next = source[end..]
            .find("(def program ")
            .context("constant wrapper")?
            + end;
        let next_end = definition_end(&source, next);
        // Converted definitions must not be selected again by the loop.
        source.replace_range(
            start..next_end,
            &rendered.replace("flux.constant-", "flux.table-"),
        );
        source = source.replace(
            &format!("{name} "),
            &format!("{} ", name.replace("constant-", "table-")),
        );
    }
    Ok(convert(&source))
}
pub(super) fn bundle(source: &str) -> String {
    let mut out = include_str!("../catena/experimental/memory.hex").to_owned();
    for helper in [
        include_str!("../catena/experimental/numeric.hex"),
        include_str!("../catena/experimental/resources.hex"),
        include_str!("../catena/experimental/cooperative.hex"),
        include_str!("../catena/experimental/qwen35_batch_projection.hex"),
        include_str!("../catena/experimental/flux.hex"),
        source,
    ] {
        crate::experimental::fused::append_hex(&mut out, helper);
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
    let mut out = convert(&crate::catena::rope_frequency_program(cfg));
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

#[cfg(test)]
mod tests {
    use super::*;
    use catena_lang::{
        codegen::{CodegenKind, GpuDialect},
        runtime::RuntimeModule,
    };
    fn compile(source: &str) -> Result<RuntimeModule> {
        let raw = metacat::theory::RawTheorySet::from_texts(
            catena_lang::stdlib::experimental_core_sources()
                .map(|s| -> &str { s })
                .chain([source]),
        )?;
        let raw = crate::catena::prepare_sources(raw)?;
        Ok(catena_lang::compile::compile(
            &mut catena_lang::report::CompileReport::new(raw),
            CodegenKind::Experimental,
            GpuDialect::Hip,
        )?)
    }
    #[test]
    fn experimental_kernels_compile() -> Result<()> {
        let source = bundle(
            "(def program flux.denoise : {((cap.ref mem) val) ((cap.ref mem) val) ((cap.ref mem) val)} -> ((cap.own mem) val) = ([x w b.] ({[.x] [.w] [.b] (const.u64.0x0000000000000001) (const.u64.0x0000000000000004) (const.u64.0x0000000000000001) (const.u64.0x0000000000000000) (const.u64.0x0000000000000002) (const.u32.0x3727C5AC u32.bitcast-f32)} flux.norm-f32)))",
        );
        assert!(!source.contains("materializec."));
        let module = compile(&source)?;
        assert_eq!(module.functions.len(), 1);
        Ok(())
    }
    #[test]
    fn experimental_stages_compile() -> Result<()> {
        let tc: super::super::config::TextConfig = serde_json::from_str(
            r#"{"hidden_size":2560,"intermediate_size":9728,"num_hidden_layers":36,"num_attention_heads":32,"num_key_value_heads":8,"head_dim":128,"vocab_size":151936,"rms_norm_eps":0.000001,"rope_theta":1000000.0,"max_position_embeddings":40960}"#,
        )?;
        let dc: super::super::config::TransformerConfig = serde_json::from_str(
            r#"{"attention_head_dim":128,"axes_dims_rope":[32,32,32,32],"eps":0.000001,"guidance_embeds":false,"in_channels":128,"joint_attention_dim":7680,"mlp_ratio":3.0,"num_attention_heads":24,"num_layers":5,"num_single_layers":20,"rope_theta":2000.0,"timestep_guidance_channels":256}"#,
        )?;
        let vc: super::super::config::VaeConfig = serde_json::from_str(
            r#"{"block_out_channels":[128,256,512,512],"latent_channels":32,"norm_num_groups":32,"batch_norm_eps":0.00001,"patch_size":[2,2],"force_upcast":true,"layers_per_block":2}"#,
        )?;
        for debug in [false, true] {
            for (name, source) in [
                ("text", text(&tc.qwen()?, debug)?),
                (
                    "transformer",
                    graph(&super::super::graph::transformer(&dc, 32, 48, 2, debug)?.source)?,
                ),
                (
                    "vae",
                    graph(&super::super::graph::vae(&vc, 32, 48, debug)?.source)?,
                ),
            ] {
                if std::env::var("FLUX_TEST_STAGE").is_ok_and(|stage| stage != name) {
                    continue;
                }
                eprintln!("compiling {name}, debug={debug}");
                let module =
                    compile(&bundle(&source)).with_context(|| format!("{name} debug={debug}"))?;
                assert_eq!(module.functions.len(), 1);
            }
        }
        Ok(())
    }
}
