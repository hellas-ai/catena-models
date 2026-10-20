//! Render Whisper for the compiler. No model execution happens in Rust.
//! The encoder/decoder topology is shared; generation state and decisions live in Hex.
use super::{Config, graph};
use anyhow::Result;
use std::collections::BTreeMap;
const O: &str = "((cap.own mem) val)";
const R: &str = "((cap.ref mem) val)";
const U: &str = "(u64 val)";
fn c(n: usize) -> String {
    format!("(const.u64.0x{n:016X})")
}
fn v(n: &str) -> String {
    format!("[.{n}]")
}
fn call(names: &[String], op: &str, outs: &[String]) -> String {
    if outs.is_empty() {
        return format!(" ({{{}}} {op})\n", names.join(" "));
    }
    format!(" ({{{}}} {op} [{}.])\n", names.join(" "), outs.join(" "))
}
fn invoke(args: &[&str], op: &str, outs: &[&str]) -> String {
    call(
        &args.iter().map(|s| v(s)).collect::<Vec<_>>(),
        op,
        &outs.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
    )
}
fn product(ts: &[String]) -> String {
    if ts.len() == 1 {
        return ts[0].clone();
    }
    let m = ts.len() / 2;
    format!("({{{} {}}} *)", product(&ts[..m]), product(&ts[m..]))
}
fn pack(ns: &[String], out: &str) -> String {
    if ns.len() == 1 {
        return format!(" ({} [{out}.])\n", v(&ns[0]));
    }
    let m = ns.len() / 2;
    let a = format!("{out}-l");
    let b = format!("{out}-r");
    pack(&ns[..m], &a) + &pack(&ns[m..], &b) + &invoke(&[&a, &b], "*.intro", &[out])
}
fn unpack(ns: &[String], input: &str) -> String {
    if ns.len() == 1 {
        return format!(" ({} [{}.])\n", v(input), ns[0]);
    }
    let m = ns.len() / 2;
    let a = format!("{input}-l");
    let b = format!("{input}-r");
    invoke(&[input], "*.elim", &[&a, &b]) + &unpack(&ns[..m], &a) + &unpack(&ns[m..], &b)
}
fn def(name: &str, args: &[(String, String)], output: &str, body: &str) -> String {
    format!(
        "\n(def program {name} : {{{}}} -> {output} =\n ([{}.]\n{body}))\n",
        args.iter()
            .map(|(_, t)| t.as_str())
            .collect::<Vec<_>>()
            .join(" "),
        args.iter()
            .map(|(n, _)| n.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    )
}
fn extract(source: &str, name: &str) -> String {
    let start = source.find(&format!("(def program {name} :")).unwrap();
    let end = source[start + 1..]
        .find("(def program ")
        .map_or(source.len(), |i| start + 1 + i);
    source[start..end].to_owned()
}
fn free(old: &str, after: &str, result: &str, bf16: bool) -> String {
    invoke(
        &[old, after],
        if bf16 {
            "qwen3.free-bf16-before"
        } else {
            "qwen3.free-f32-before"
        },
        &[result],
    )
}
fn decode(
    cfg: &Config,
    control: &str,
    prefix: &str,
    caches: &[String],
    cross: &[String],
    weights: &[String],
) -> (String, Vec<String>, String) {
    let mut args = vec![v(control)];
    for l in 0..cfg.decoder_layers {
        args.extend([
            v(&caches[2 * l]),
            v(&caches[2 * l + 1]),
            v(&cross[2 * l]),
            v(&cross[2 * l + 1]),
        ]);
    }
    args.extend(weights.iter().map(|w| v(w)));
    let next = caches
        .iter()
        .map(|n| format!("{prefix}-{n}"))
        .collect::<Vec<_>>();
    let logits = format!("{prefix}-logits");
    let mut out = vec![logits.clone()];
    out.extend(next.clone());
    (call(&args, "whisper.decode-state", &out), next, logits)
}
fn generation(cfg: &Config, debug: bool) -> String {
    let caches = (0..cfg.decoder_layers)
        .flat_map(|i| [format!("sk{i}"), format!("sv{i}")])
        .collect::<Vec<_>>();
    let cross = (0..cfg.decoder_layers)
        .flat_map(|i| [format!("ck{i}"), format!("cv{i}")])
        .collect::<Vec<_>>();
    let dw = (0..graph::decoder_names(cfg).len())
        .map(|i| format!("dw{i}"))
        .collect::<Vec<_>>();
    let mut env = vec!["eos".to_owned(), "text-mask".into(), "begin-mask".into()];
    env.extend(cross.clone());
    env.extend(dw.clone());
    let mut et = vec![U.to_owned(), R.into(), R.into()];
    et.extend(vec![O.into(); cross.len()]);
    et.extend(vec![R.into(); dw.len()]);
    let et = product(&et);
    let mut state = vec!["control".to_owned(), "tokens".into()];
    state.extend(caches.clone());
    if debug {
        state.push("trace".into());
    }
    let st = product(&vec![O.to_owned(); state.len()]);
    let mut body = unpack(&env, "env") + &unpack(&state, "state");
    let (step, next, mut logits) = decode(cfg, "control", "next", &caches, &cross, &dw);
    body += &step;
    let mut next_state = vec!["ready-control".to_owned(), "tokens".into()];
    next_state.extend(next);
    let control = if debug {
        body += &call(
            &[v("trace"), v(&logits), v("control"), c(cfg.vocab_size)],
            "stdlib.whisper.trace",
            &["next-trace".into()],
        );
        body += " ([.next-trace] meta.master.complete-own [trace-ready trace-phase.])\n";
        body += &format!(
            " ({{{} [.trace-phase]}} meta.master.after-own [logits-after-trace.])\n",
            v(&logits)
        );
        body += " ({[.control] [.trace-phase]} meta.master.after-own [control-after-trace.])\n";
        logits = "logits-after-trace".into();
        next_state.push("trace-ready".into());
        "control-after-trace"
    } else {
        "control"
    };
    body += &call(
        &[
            v(&logits),
            v("text-mask"),
            v("begin-mask"),
            v(control),
            c(0),
        ],
        "stdlib.whisper.select-state",
        &["scores".into(), "indices".into()],
    );
    body += &invoke(
        &[control, "scores", "indices", "tokens", "eos"],
        "stdlib.whisper.advance",
        &["updated"],
    );
    body += &free("scores", "updated", "free-scores", true);
    body += &free("indices", "free-scores", "free-indices", false);
    body += &free(&logits, "free-indices", "ready-control", true);
    body += &pack(&next_state, "result");
    body += " [.result]\n";
    let mut source = def(
        "stdlib.whisper.iterate",
        &[("env".into(), et.clone()), ("state".into(), st.clone())],
        &st,
        &body,
    );
    let pair = format!(
        "({{({{master.index32 u32}} :) ({{{st} ({{(({{master.index32 master.end32}} prop.Less) |-) (prop.Truth |-)}} *)}} *)}} *)"
    );
    let body=" ([.pair] *.elim [index rest.])\n ([.rest] *.elim [state proofs.])\n ([.proofs] *.elim [less truth.])\n ([.less] unsafe.discard unit.elim)\n".to_owned()+&invoke(&["env","state"],"stdlib.whisper.iterate",&["next"])+&invoke(&["next","truth"],"*.intro",&[]);
    source += &def(
        "stdlib.whisper.iterate.fold",
        &[("env".into(), et.clone()), ("pair".into(), pair)],
        &format!("({{{st} (prop.Truth |-)}} *)"),
        &body,
    );
    let mut inputs = vec![("features".into(), R.into())];
    inputs.extend(
        [
            "limit",
            "language-override",
            "start",
            "task",
            "no-timestamps",
            "eos",
        ]
        .map(|n| (n.into(), U.into())),
    );
    inputs.extend(["language-mask", "text-mask", "begin-mask"].map(|n| (n.into(), R.into())));
    for (prefix, names) in [
        ("ew", graph::encoder_names(cfg)),
        ("cw", graph::cross_names(cfg)),
        ("dw", graph::decoder_names(cfg)),
    ] {
        inputs.extend((0..names.len()).map(|i| (format!("{prefix}{i}"), R.into())));
    }
    let mut body = call(
        &[
            v("features"),
            c(cfg.max_source_positions * 2 * cfg.num_mel_bins),
        ],
        "inference.f32-to-bf16",
        &["mel".into()],
    );
    let mut args = vec![v("mel")];
    args.extend((0..graph::encoder_names(cfg).len()).map(|i| v(&format!("ew{i}"))));
    body += &call(&args, "whisper.encode", &["encoder".into()]);
    if debug {
        body += &call(
            &[v("encoder"), c(cfg.max_source_positions * cfg.d_model)],
            "stdlib.whisper.copy-bf16",
            &["encoder-snapshot".into()],
        );
        body += " ([.encoder-snapshot] meta.master.complete-own [snapshot-ready snapshot-phase.])\n ({[.encoder] [.snapshot-phase]} meta.master.after-own [encoder-ready.])\n";
    }
    let mut args = vec![v(if debug { "encoder-ready" } else { "encoder" })];
    args.extend((0..graph::cross_names(cfg).len()).map(|i| v(&format!("cw{i}"))));
    body += &call(&args, "whisper.prepare-cross-kv", &cross);
    for n in &caches {
        body += &call(
            &[c(cfg.max_target_positions * cfg.d_model)],
            "inference.zeros-bf16",
            std::slice::from_ref(n),
        );
    }
    body += &invoke(
        &["start", "language-override"],
        "stdlib.whisper.control-init",
        &["control0"],
    );
    let (step, mut cache, logits) = decode(cfg, "control0", "prefix0", &caches, &cross, &dw);
    body += &step;
    body += &call(
        &[
            v(&logits),
            v("language-mask"),
            v("begin-mask"),
            v("control0"),
            c(1),
        ],
        "stdlib.whisper.select-state",
        &["language-scores".into(), "language-indices".into()],
    );
    body += &invoke(
        &["control0", "language-scores", "language-indices"],
        "stdlib.whisper.language-update",
        &["language-control"],
    );
    body += &free(
        "language-scores",
        "language-control",
        "language-free-scores",
        true,
    );
    body += &free(
        "language-indices",
        "language-free-scores",
        "language-free-indices",
        false,
    );
    body += &free(&logits, "language-free-indices", "control1", true);
    let mut control = "control1".to_owned();
    for (position, token) in [(2, "task"), (3, "no-timestamps")] {
        let prefix = format!("prefix{}", position - 1);
        let (step, next, logits) = decode(cfg, &control, &prefix, &cache, &cross, &dw);
        body += &step;
        cache = next;
        body += &format!(
            " ({{{} }} meta.master.complete-own [prefix-logits{position} prefix-phase{position}.])\n ({{{} [.prefix-phase{position}]}} meta.master.after-own [prefix-control{position}.])\n",
            v(&logits),
            v(&control)
        );
        body += &call(
            &[
                v(&format!("prefix-control{position}")),
                v(token),
                c(position),
            ],
            "stdlib.whisper.force",
            &[format!("forced{position}")],
        );
        let ready = format!("control{position}");
        body += &free(
            &format!("prefix-logits{position}"),
            &format!("forced{position}"),
            &ready,
            true,
        );
        control = ready;
    }
    let bound = cfg.max_target_positions - 4;
    body += &call(&[c(bound + 3)], "inference.zeros-u64", &["tokens".into()]);
    let mut initial = vec![control, "tokens".into()];
    initial.extend(cache);
    if debug {
        body += &call(
            &[c(bound * cfg.vocab_size)],
            "inference.zeros-bf16",
            &["initial-trace".into()],
        );
        initial.push("initial-trace".into());
    }
    body += &pack(&initial, "initial");
    body += &pack(&env, "env");
    body += &call(
        &[c(bound), v("limit")],
        "stdlib.qwen3.memory.u64.less",
        &["clamp".into()],
    );
    body += &call(
        &[v("clamp"), c(bound), v("limit")],
        "stdlib.qwen3.memory.select-u64",
        &["bounded".into()],
    );
    body += " ([.bounded] stdlib.qwen3.memory.fold-count-u32 [bound32.])\n ({[.bound32] (unit.intro meta.master.end32)} smolcat.value.named_at [end.])\n ({[.env]} partial.stdlib.whisper.iterate.fold.1 [callback.])\n ({[.end] [.initial] [.callback] core.truth} core.fold.bounded [final proof.])\n ([.proof] unsafe.discard unit.elim)\n";
    let final_names = state
        .iter()
        .map(|n| format!("final-{n}"))
        .collect::<Vec<_>>();
    body += &unpack(&final_names, "final");
    body += &invoke(
        &["final-tokens", "final-control"],
        "stdlib.whisper.result-header",
        &["header"],
    );
    let mut result = "header".to_owned();
    for (i, (name, is_bf16)) in std::iter::once(("final-control".to_owned(), false))
        .chain(caches.iter().map(|n| (format!("final-{n}"), true)))
        .chain(cross.into_iter().map(|n| (n, true)))
        .enumerate()
    {
        let next = format!("result{i}");
        body += &free(&name, &result, &next, is_bf16);
        result = next;
    }
    let mut outputs = vec![v(&result)];
    if debug {
        outputs.extend([v("snapshot-ready"), v("final-trace")]);
    }
    body += &format!(" {{{}}}\n", outputs.join(" "));
    source += &def(
        "whisper.transcribe",
        &inputs,
        &format!("{{{}}}", vec![O; if debug { 3 } else { 1 }].join(" ")),
        &body,
    );
    source
}

pub(super) fn source(cfg: &Config, debug: bool) -> Result<String> {
    cfg.validate()?;
    Ok(render(cfg, debug))
}
fn render(cfg: &Config, debug: bool) -> String {
    let model = graph::model_source(cfg);
    let mut layer = extract(&model, "whisper.decoder-layer")
        .replacen(
            "whisper.decoder-layer :",
            "whisper.decoder-layer-state :",
            1,
        )
        .replacen(U, O, 1)
        .replacen(R, O, 2)
        .replace(
            " ({[.pos] (const.u64.0x0000000000000001)} stdlib.qwen3.memory.u64.add [valid.])\n",
            "",
        )
        .replace("[.valid]", "[.pos]")
        .replace("stdlib.whisper.append ", "stdlib.whisper.append-state ")
        .replace(
            "stdlib.whisper.attention-o [a.]",
            "stdlib.whisper.attention-state [a.]",
        )
        .replace("stdlib.whisper.attention-r ", "stdlib.whisper.attention-o ");
    layer.push('\n');
    let mut step = extract(&model, "whisper.decode-step");
    let end = step.find(" ->").unwrap();
    let mut signature = step[..end].replacen(&format!("{U} {U}"), O, 1);
    for _ in 0..2 * cfg.decoder_layers {
        signature = signature.replacen(R, O, 1);
    }
    step = (signature + &step[end..])
        .replacen("whisper.decode-step :", "whisper.decode-state :", 1)
        .replacen("([token pos ", "([pos ", 1)
        .replace("[.token] [.pos]", "[.pos]")
        .replace(
            "stdlib.whisper.embedding ",
            "stdlib.whisper.embedding-state ",
        )
        .replace("whisper.decoder-layer ", "whisper.decoder-layer-state ");
    let composition = [model, layer, step, generation(cfg, debug)].join("\n");
    let mut source = include_str!("../catena/memory.hex").to_owned();
    for fragment in [
        include_str!("../catena/numeric.hex"),
        include_str!("../catena/resources.hex"),
        include_str!("../catena/cooperative.hex"),
        include_str!("../catena/qwen35_batch_projection.hex"),
        include_str!("../catena/fused_argmax.hex"),
        include_str!("../catena/whisper.hex"),
        &composition,
    ] {
        crate::qwen::fused::append_hex(&mut source, fragment);
    }
    // Only these four entries belong to the runtime ABI; helper definitions prune normally.
    let public = [
        "whisper.encode",
        "whisper.prepare-cross-kv",
        "whisper.decode-step",
        "whisper.transcribe",
    ];
    let mut mapping = BTreeMap::new();
    for def in source.split("(def program ").skip(1) {
        let name = def.split_whitespace().next().unwrap();
        if !name.starts_with("stdlib.") && !public.contains(&name) {
            mapping.insert(name.to_owned(), format!("stdlib.{name}"));
        }
    }
    crate::catena::rename_programs(&source, &mapping)
}
