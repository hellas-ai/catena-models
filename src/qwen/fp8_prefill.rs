//! FP8 prefill uses a separate native-weight tail in each layer's weight tree.
use super::*;

pub(super) fn append(source: &mut String) {
    fused::append_hex(source, include_str!("../catena/qwen3_fp8.hex"));
}

pub(super) fn graph() -> String {
    let mut graph = include_str!("../catena/qwen3_prefill.hex").to_owned();
    graph = graph.replace(
        "  ([.weights9] *.elim [wu wd.])",
        "  ([.weights9] *.elim [wu weights10.])\n  ([.weights10] *.elim [wd native.])",
    );
    let mut extra = String::new();
    let operands = ["q", "k", "v", "o", "g", "u", "d"]
        .into_iter()
        .flat_map(|n| [format!("{n}-fp8"), format!("{n}-scale")])
        .collect::<Vec<_>>();
    unpack(&mut extra, "native", &operands);
    graph = graph.replace(
        "  ([.weights10] *.elim [wd native.])",
        &format!("  ([.weights10] *.elim [wd native.])\n{extra}"),
    );
    // Quantize once for each shared projection input and free scratch after its last consumer.
    let mut qkv = String::new();
    call(
        &mut qkv,
        &["normalized"],
        "stdlib.qwen3.fp8.quantize",
        "qkv-quantized",
    );
    unpack(
        &mut qkv,
        "qkv-quantized",
        &["qkv-bytes".into(), "qkv-scales".into()],
    );
    for (name, weight, width) in [
        ("q", "wq", "QUERY_SIZE"),
        ("k", "wk", "KEY_VALUE_SIZE"),
        ("v", "wv", "KEY_VALUE_SIZE"),
    ] {
        project(
            &mut qkv,
            "normalized",
            "qkv-bytes",
            "qkv-scales",
            weight,
            name,
            "HIDDEN_SIZE",
            width,
            "project",
            &[],
            &format!("{name}-raw"),
        );
    }
    let start = graph.find("  ({[.normalized] [.wq]").unwrap();
    let end = graph[start..].find("\n\n").unwrap() + start;
    graph.replace_range(start..end, &qkv);
    graph = graph.replace("  ({[.normalized] [.attended]}", "  ({[.qkv-bytes] [.attended]} stdlib.models.resources.free-f32-before [attended-after-bytes.])\n  ({[.qkv-scales] [.attended-after-bytes]} stdlib.models.resources.free-f32-before [attended-after-scales.])\n  ({[.normalized] [.attended-after-scales]}");
    let start = graph.find("  ({[.context] [.wo]").unwrap();
    let end = graph[start..].find('\n').unwrap() + start;
    let mut out = String::new();
    quantized_project(
        &mut out,
        "context",
        "wo",
        "o",
        "QUERY_SIZE",
        "HIDDEN_SIZE",
        "hidden",
        "residual",
    );
    graph.replace_range(start..end, &out);
    let start = graph.find("  ({[.ffn] [.wg]").unwrap();
    let end = graph[start..].find('\n').unwrap() + start;
    let mut gate = String::new();
    call(
        &mut gate,
        &["ffn"],
        "stdlib.qwen3.fp8.quantize",
        "ffn-quantized",
    );
    unpack(
        &mut gate,
        "ffn-quantized",
        &["ffn-bytes".into(), "ffn-scales".into()],
    );
    project(
        &mut gate,
        "ffn",
        "ffn-bytes",
        "ffn-scales",
        "wg",
        "g",
        "HIDDEN_SIZE",
        "INTERMEDIATE_SIZE",
        "project-gate",
        &["wu", "u-fp8", "u-scale"],
        "gate-active",
    );
    call(
        &mut gate,
        &["ffn-bytes", "gate-active"],
        "stdlib.models.resources.free-f32-before",
        "gate-after-bytes",
    );
    call(
        &mut gate,
        &["ffn-scales", "gate-after-bytes"],
        "stdlib.models.resources.free-f32-before",
        "active",
    );
    graph.replace_range(start..end, &gate);
    let start = graph.find("  ({[.active-ready] [.wd]").unwrap();
    let end = graph[start..].find('\n').unwrap() + start;
    let mut down = String::new();
    quantized_project(
        &mut down,
        "active-ready",
        "wd",
        "d",
        "INTERMEDIATE_SIZE",
        "HIDDEN_SIZE",
        "attention",
        "output",
    );
    graph.replace_range(start..end, &down);
    graph
}

#[allow(clippy::too_many_arguments)]
fn project(
    s: &mut String,
    input: &str,
    bytes: &str,
    scales: &str,
    bf16: &str,
    native: &str,
    k: &str,
    n: &str,
    operation: &str,
    extra: &[&str],
    out: &str,
) {
    let native_weight = format!("{native}-fp8");
    let native_scale = format!("{native}-scale");
    let features = format!("(const.u64.0x{{{{{k}}}}})");
    let outputs = format!("(const.u64.0x{{{{{n}}}}})");
    let mut args = vec![
        input,
        bytes,
        scales,
        bf16,
        &native_weight,
        &native_scale,
        &features,
        &outputs,
    ];
    args.extend_from_slice(extra);
    call(s, &args, &format!("stdlib.qwen3.fp8.{operation}"), out);
}
#[allow(clippy::too_many_arguments)]
fn quantized_project(
    s: &mut String,
    input: &str,
    bf16: &str,
    native: &str,
    k: &str,
    n: &str,
    residual: &str,
    out: &str,
) {
    let quant = format!("{native}-quantized");
    let bytes = format!("{native}-bytes");
    let scales = format!("{native}-scales");
    call(s, &[input], "stdlib.qwen3.fp8.quantize", &quant);
    unpack(s, &quant, &[bytes.clone(), scales.clone()]);
    let projected = format!("{native}-raw");
    project(
        s,
        input,
        &bytes,
        &scales,
        bf16,
        native,
        k,
        n,
        "project-residual",
        &[residual],
        &projected,
    );
    let freed = format!("{native}-after-bytes");
    call(
        s,
        &[&bytes, &projected],
        "stdlib.models.resources.free-f32-before",
        &freed,
    );
    call(
        s,
        &[&scales, &freed],
        "stdlib.models.resources.free-f32-before",
        out,
    );
}
