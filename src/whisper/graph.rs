use super::Config;
use anyhow::Result;
pub const HELPERS: &str = include_str!("../catena/whisper_helpers.hex");
const MODEL: &str = include_str!("../catena/whisper_model.hex");
pub const ATTN: &[&str] = &[
    "self_attn_layer_norm.weight",
    "self_attn_layer_norm.bias",
    "self_attn.q_proj.weight",
    "self_attn.q_proj.bias",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.v_proj.bias",
    "self_attn.out_proj.weight",
    "self_attn.out_proj.bias",
];
pub const FF: &[&str] = &[
    "final_layer_norm.weight",
    "final_layer_norm.bias",
    "fc1.weight",
    "fc1.bias",
    "fc2.weight",
    "fc2.bias",
];
pub const CROSS: &[&str] = &[
    "encoder_attn_layer_norm.weight",
    "encoder_attn_layer_norm.bias",
    "encoder_attn.q_proj.weight",
    "encoder_attn.q_proj.bias",
    "encoder_attn.out_proj.weight",
    "encoder_attn.out_proj.bias",
];
fn c(n: usize) -> String {
    format!("(const.u64.0x{n:016X})")
}
fn v(n: &str) -> String {
    format!("[.{n}]")
}
fn call(args: &[String], op: &str, outs: &[&str]) -> String {
    format!(" ({{{}}} {op} [{}.])\n", args.join(" "), outs.join(" "))
}
fn entry(name: &str, inputs: &[(String, &'static str)], outputs: &[&str], body: &str) -> String {
    format!(
        "\n(def program whisper.{name} : {{{}}} -> {{{}}}\n= ([{}.]\n{body}))\n",
        inputs.iter().map(|(_, t)| *t).collect::<Vec<_>>().join(" "),
        outputs.join(" "),
        inputs
            .iter()
            .map(|(n, _)| n.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    )
}
pub fn layer_names(kind: &str, layer: usize, suffixes: &[&str]) -> Vec<String> {
    suffixes
        .iter()
        .map(|s| format!("model.{kind}.layers.{layer}.{s}"))
        .collect()
}
pub fn encoder_names(cfg: &Config) -> Vec<String> {
    let mut n = [
        "conv1.weight",
        "conv1.bias",
        "conv2.weight",
        "conv2.bias",
        "embed_positions.weight",
    ]
    .map(|s| format!("model.encoder.{s}"))
    .to_vec();
    for l in 0..cfg.encoder_layers {
        n.extend(layer_names("encoder", l, ATTN));
        n.extend(layer_names("encoder", l, FF));
    }
    n.extend([
        "model.encoder.layer_norm.weight".into(),
        "model.encoder.layer_norm.bias".into(),
    ]);
    n
}
pub fn cross_names(cfg: &Config) -> Vec<String> {
    (0..cfg.decoder_layers)
        .flat_map(|l| {
            layer_names(
                "decoder",
                l,
                &[
                    "encoder_attn.k_proj.weight",
                    "encoder_attn.v_proj.weight",
                    "encoder_attn.v_proj.bias",
                ],
            )
        })
        .collect()
}
pub fn decoder_names(cfg: &Config) -> Vec<String> {
    let mut n = vec![
        "model.decoder.embed_tokens.weight".into(),
        "model.decoder.embed_positions.weight".into(),
    ];
    for l in 0..cfg.decoder_layers {
        n.extend(layer_names("decoder", l, ATTN));
        n.extend(layer_names("decoder", l, CROSS));
        n.extend(layer_names("decoder", l, FF));
    }
    n.extend([
        "model.decoder.layer_norm.weight".into(),
        "model.decoder.layer_norm.bias".into(),
    ]);
    n
}
pub fn source(cfg: &Config) -> Result<String> {
    source_with_debug(cfg, false)
}
pub fn source_with_debug(cfg: &Config, debug: bool) -> Result<String> {
    cfg.validate()?;
    let mut s = model_source(cfg);
    s += &super::generation::source(cfg, &s, debug);
    Ok(s)
}

/// Metadata-derived model composition shared by both compiler backends.
pub(crate) fn model_source(cfg: &Config) -> String {
    let mut s = MODEL.to_string();
    for (k, n) in [
        ("DIM", cfg.d_model),
        ("FF", cfg.encoder_ffn_dim),
        ("HEADS", cfg.encoder_attention_heads),
        ("HEAD_DIM", cfg.d_model / cfg.encoder_attention_heads),
        ("SOURCE", cfg.max_source_positions),
        ("CAPACITY", cfg.max_target_positions),
    ] {
        s = s.replace(&format!("{{{{{k}}}}}"), &format!("{n:016X}"));
    }
    let own = "(cap.own mem)";
    let borrowed = "(cap.ref mem)";
    let scalar = "(u64 val)";
    let mut inputs = vec![("mel".into(), own)];
    inputs.extend((0..encoder_names(cfg).len()).map(|i| (format!("w{i}"), borrowed)));
    let d = cfg.d_model;
    let rows = cfg.max_source_positions;
    let mut b = call(
        &[
            v("mel"),
            v("w0"),
            v("w1"),
            c(rows * 2),
            c(rows * 2),
            c(cfg.num_mel_bins),
            c(d),
            c(1),
        ],
        "whisper.conv",
        &["conv1"],
    );
    b += &call(
        &[
            v("conv1"),
            v("w2"),
            v("w3"),
            c(rows * 2),
            c(rows),
            c(d),
            c(d),
            c(2),
        ],
        "whisper.conv",
        &["conv2"],
    );
    b += &call(
        &[v("conv2"), v("w4"), c(0), c(rows * d)],
        "whisper.position",
        &["h0"],
    );
    b += &call(&[v("conv2"), v("h0")], "qwen3.free-bf16-before", &["start"]);
    let mut hidden = "start".to_string();
    let mut wi = 5;
    for l in 0..cfg.encoder_layers {
        let mut args = vec![v(&hidden), c(rows)];
        args.extend((wi..wi + ATTN.len() + FF.len()).map(|i| v(&format!("w{i}"))));
        wi += ATTN.len() + FF.len();
        let next = format!("h{}", l + 1);
        b += &call(&args, "whisper.encoder-layer", &[&next]);
        hidden = next;
    }
    b += &call(
        &[
            v(&hidden),
            v(&format!("w{wi}")),
            v(&format!("w{}", wi + 1)),
            c(rows),
            c(d),
        ],
        "whisper.norm",
        &["normalized"],
    );
    b += &call(
        &[v(&hidden), v("normalized")],
        "qwen3.free-bf16-before",
        &["result"],
    );
    b += " [.result]\n";
    s += &entry("encode", &inputs, &[own], &b);
    let mut inputs = vec![("encoder".into(), own)];
    inputs.extend((0..cross_names(cfg).len()).map(|i| (format!("w{i}"), borrowed)));
    let mut b = String::new();
    let mut outputs = Vec::new();
    for l in 0..cfg.decoder_layers {
        let k = format!("k{l}");
        let vr = format!("vr{l}");
        let vv = format!("v{l}");
        b += &call(
            &[v("encoder"), v(&format!("w{}", l * 3)), c(rows), c(d), c(d)],
            "whisper.linear",
            &[&k],
        );
        b += &call(
            &[
                v("encoder"),
                v(&format!("w{}", l * 3 + 1)),
                c(rows),
                c(d),
                c(d),
            ],
            "whisper.linear",
            &[&vr],
        );
        b += &call(
            &[
                v(&vr),
                v(&format!("w{}", l * 3 + 2)),
                c(d),
                c(0),
                c(rows * d),
            ],
            "whisper.bias-gelu",
            &[&vv],
        );
        let ready = format!("vready{l}");
        b += &call(&[v(&vr), v(&vv)], "qwen3.free-bf16-before", &[&ready]);
        outputs.push(k);
        outputs.push(ready);
    }
    let last = outputs.pop().unwrap();
    b += &call(
        &[v("encoder"), v(&last)],
        "qwen3.free-bf16-before",
        &["last"],
    );
    outputs.push("last".into());
    b += &format!(
        " {{{}}}\n",
        outputs.iter().map(|n| v(n)).collect::<Vec<_>>().join(" ")
    );
    s += &entry("prepare-cross-kv", &inputs, &vec![own; outputs.len()], &b);
    let mut inputs = vec![("token".into(), scalar), ("pos".into(), scalar)];
    for l in 0..cfg.decoder_layers {
        inputs.extend([
            (format!("sk{l}"), own),
            (format!("sv{l}"), own),
            (format!("ck{l}"), borrowed),
            (format!("cv{l}"), borrowed),
        ]);
    }
    inputs.extend((0..decoder_names(cfg).len()).map(|i| (format!("w{i}"), borrowed)));
    let mut b = call(
        &[v("w0"), v("w1"), v("token"), v("pos"), c(d)],
        "whisper.embedding",
        &["h0"],
    );
    let mut wi = 2;
    let mut hidden = "h0".to_string();
    let mut caches = Vec::new();
    for l in 0..cfg.decoder_layers {
        let mut args = vec![
            v(&hidden),
            v(&format!("sk{l}")),
            v(&format!("sv{l}")),
            v(&format!("ck{l}")),
            v(&format!("cv{l}")),
            v("pos"),
        ];
        let nw = ATTN.len() + CROSS.len() + FF.len();
        args.extend((wi..wi + nw).map(|i| v(&format!("w{i}"))));
        wi += nw;
        let next = format!("h{}", l + 1);
        let k = format!("kn{l}");
        let vv = format!("vn{l}");
        b += &call(&args, "whisper.decoder-layer", &[&next, &k, &vv]);
        hidden = next;
        caches.extend([k, vv]);
    }
    b += &call(
        &[
            v(&hidden),
            v(&format!("w{wi}")),
            v(&format!("w{}", wi + 1)),
            c(1),
            c(d),
        ],
        "whisper.norm",
        &["normalized"],
    );
    b += &call(
        &[v("normalized"), v("w0"), c(1), c(d), c(cfg.vocab_size)],
        "whisper.linear",
        &["logits"],
    );
    b += &call(
        &[v(&hidden), v("logits")],
        "qwen3.free-bf16-before",
        &["after-hidden"],
    );
    b += &call(
        &[v("normalized"), v("after-hidden")],
        "qwen3.free-bf16-before",
        &["result"],
    );
    b += &format!(
        " {{[.result] {}}}\n",
        caches.iter().map(|n| v(n)).collect::<Vec<_>>().join(" ")
    );
    s += &entry("decode-step", &inputs, &vec![own; 1 + caches.len()], &b);
    s
}

/// Full deployable model asset; the receiving node supplies only compiler stdlib.
pub fn bundle(cfg: &Config, debug: bool) -> Result<String> {
    Ok([
        crate::catena::BF16_HELPER_SOURCE,
        include_str!("../catena/inference_helpers.hex"),
        HELPERS,
        &source_with_debug(cfg, debug)?,
    ]
    .join("\n"))
}

pub fn bundle_with_codegen(
    cfg: &Config,
    debug: bool,
    codegen: catena_lang::codegen::CodegenKind,
) -> Result<String> {
    match codegen {
        catena_lang::codegen::CodegenKind::Default => bundle(cfg, debug),
        catena_lang::codegen::CodegenKind::Experimental => super::experimental::source(cfg, debug),
    }
}
