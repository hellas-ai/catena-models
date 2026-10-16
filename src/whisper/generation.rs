//! Hex generation control. These functions assemble source before compilation.
use super::{Config, graph};
const OWN: &str = "(cap.own mem)";
const REF: &str = "(cap.ref mem)";
const U: &str = "(u64 val)";
fn c(n: usize) -> String {
    format!("(const.u64.0x{n:016X})")
}
fn v(n: &str) -> String {
    format!("[.{n}]")
}
fn product(types: &[&str]) -> String {
    if types.len() == 1 {
        types[0].into()
    } else {
        let mid = types.len() / 2;
        format!(
            "({{{} {}}} *)",
            product(&types[..mid]),
            product(&types[mid..])
        )
    }
}
fn pack(names: &[String], out: &str) -> String {
    if names.len() == 1 {
        return format!(" ({} [{out}.])\n", v(&names[0]));
    }
    let mid = names.len() / 2;
    let left = format!("{out}-l");
    let right = format!("{out}-r");
    pack(&names[..mid], &left)
        + &pack(&names[mid..], &right)
        + &format!(" ({{{} {}}} *.intro [{out}.])\n", v(&left), v(&right))
}
fn unpack(names: &[String], input: &str) -> String {
    if names.len() == 1 {
        return format!(" ({} [{}.])\n", v(input), names[0]);
    }
    let mid = names.len() / 2;
    let left = format!("{input}-l");
    let right = format!("{input}-r");
    format!(" ({} *.elim [{left} {right}.])\n", v(input))
        + &unpack(&names[..mid], &left)
        + &unpack(&names[mid..], &right)
}
// Unpack only the context fields used by a control helper. The decoder alone
// needs every weight; guards should not repeat hundreds of weight bindings.
fn unpack_selected(names: &[String], input: &str, needed: &[&str]) -> String {
    if !names.iter().any(|name| needed.contains(&name.as_str())) {
        return String::new();
    }
    if names.len() == 1 {
        return format!(" ({} [{}.])\n", v(input), names[0]);
    }
    let mid = names.len() / 2;
    let left = format!("{input}-l");
    let right = format!("{input}-r");
    format!(" ({} *.elim [{left} {right}.])\n", v(input))
        + &unpack_selected(&names[..mid], &left, needed)
        + &unpack_selected(&names[mid..], &right, needed)
}
fn call(args: &[String], op: &str, outs: &[String]) -> String {
    format!(" ({{{}}} {op} [{}.])\n", args.join(" "), outs.join(" "))
}
fn def(name: &str, args: &[(String, String)], out: &str, body: &str) -> String {
    format!(
        "\n(def program whisper.{name} : {{{}}} -> {out}\n= ([{}.]\n{body}))\n",
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
    let start = source
        .find(&format!("(def program whisper.{name} :"))
        .unwrap();
    let end = source[start + 1..]
        .find("(def program ")
        .map_or(source.len(), |i| start + 1 + i);
    source[start..end].into()
}

pub fn source(cfg: &Config, model_source: &str, debug: bool) -> String {
    let mut source = include_str!("../catena/whisper_generation.hex").to_string();
    // The generated transcription owns cross K/V. They remain alive until its end.
    let mut layer = extract(model_source, "decoder-layer");
    layer = layer
        .replacen(
            "whisper.decoder-layer :",
            "whisper.decoder-layer-owned-cross :",
            1,
        )
        .replacen(REF, OWN, 2)
        .replace("whisper.attention-r ", "whisper.attention-o ");
    source += &layer;
    let mut step = extract(model_source, "decode-step");
    let signature_end = step.find(" ->").unwrap();
    let mut signature = step[..signature_end].to_string();
    for _ in 0..cfg.decoder_layers * 2 {
        signature = signature.replacen(REF, OWN, 1);
    }
    step = signature + &step[signature_end..];
    step = step
        .replacen(
            "whisper.decode-step :",
            "whisper.decode-step-owned-cross :",
            1,
        )
        .replace(
            "whisper.decoder-layer ",
            "whisper.decoder-layer-owned-cross ",
        );
    source += &step;
    let caches = (0..cfg.decoder_layers)
        .flat_map(|i| [format!("sk{i}"), format!("sv{i}")])
        .collect::<Vec<_>>();
    let cross = (0..cfg.decoder_layers)
        .flat_map(|i| [format!("ck{i}"), format!("cv{i}")])
        .collect::<Vec<_>>();
    let weights = (0..graph::decoder_names(cfg).len())
        .map(|i| format!("dw{i}"))
        .collect::<Vec<_>>();
    let mut env = vec![
        "limit".into(),
        "eos".into(),
        "text-mask".into(),
        "begin-mask".into(),
    ];
    env.extend(cross.clone());
    env.extend(weights.clone());
    let mut env_types = vec![U, U, REF, REF];
    env_types.extend(vec![OWN; cross.len()]);
    env_types.extend(vec![REF; weights.len()]);
    let et = product(&env_types);
    let mut state = vec![
        "logits".into(),
        "tokens".into(),
        "count".into(),
        "pos".into(),
        "status".into(),
        "token".into(),
    ];
    state.extend(caches.clone());
    if debug {
        state.push("trace".into());
    }
    let mut stypes = vec![OWN, OWN, U, U, U, U];
    stypes.extend(vec![OWN; caches.len()]);
    if debug {
        stypes.push(OWN);
    }
    let st = product(&stypes);
    let args = vec![("env".into(), et.clone()), ("state".into(), st.clone())];
    let unpack_both = |needed: Option<&[&str]>| {
        needed.map_or_else(
            || unpack(&env, "env"),
            |names| unpack_selected(&env, "env", names),
        ) + &unpack(&state, "state")
    };
    let return_state = |names: &[String]| pack(names, "result") + " [.result]\n";
    // A branch which leaves all state owners untouched.
    source += &def("hold-state", &args, &st, " [.state]\n");
    let mut body = unpack(&state, "state");
    body += &format!(" {} [failed.]\n", c(2));
    let mut failed_state = state.clone();
    failed_state[4] = "failed".into();
    body += &return_state(&failed_state);
    source += &def("failed-state", &args, &st, &body);
    let mut body = unpack(&state, "state");
    body += &format!(" {} [finished.]\n", c(1));
    let mut eos_state = state.clone();
    eos_state[4] = "finished".into();
    body += &return_state(&eos_state);
    source += &def("eos-state", &args, &st, &body);
    // A decoder call is a shared definition, regardless of the requested token limit.
    let mut body = unpack_both(None);
    body += &call(&[v("pos"), c(1)], "u64.add", &["next-pos".into()]);
    let mut decoder_args = vec![v("token"), v("next-pos")];
    for i in 0..cfg.decoder_layers {
        decoder_args.extend([
            v(&format!("sk{i}")),
            v(&format!("sv{i}")),
            v(&format!("ck{i}")),
            v(&format!("cv{i}")),
        ]);
    }
    decoder_args.extend(weights.iter().map(|n| v(n)));
    let mut outs = vec!["next-logits".into()];
    outs.extend(caches.iter().map(|n| format!("next-{n}")));
    body += &call(&decoder_args, "whisper.decode-step-owned-cross", &outs);
    body += &call(
        &[v("logits"), v("next-logits")],
        "qwen3.free-bf16-before",
        &["ready-logits".into()],
    );
    let mut nextstate = state.clone();
    nextstate[0] = "ready-logits".into();
    nextstate[3] = "next-pos".into();
    for i in 0..caches.len() {
        nextstate[6 + i] = format!("next-{}", caches[i]);
    }
    body += &return_state(&nextstate);
    source += &def("decode-next", &args, &st, &body);
    let mut body = unpack_both(Some(&["limit"]));
    body += &call(&[v("count"), c(3)], "u64.add", &["token-offset".into()]);
    body += &call(
        &[v("tokens"), v("token-offset"), v("token")],
        "whisper.write-token",
        &["next-tokens".into()],
    );
    body += &call(&[v("count"), c(1)], "u64.add", &["next-count".into()]);
    let mut nextstate = state.clone();
    nextstate[1] = "next-tokens".into();
    nextstate[2] = "next-count".into();
    body += &pack(&nextstate, "next-state");
    body += &call(&[v("next-count"), v("limit")], "u64.lt", &["more".into()]);
    body += &call(
        &[
            v("env"),
            "name.whisper.decode-next".into(),
            v("env"),
            "name.whisper.hold-state".into(),
            v("more"),
            v("next-state"),
        ],
        "bool.ifc",
        &["result".into()],
    );
    body += " [.result]\n";
    source += &def("emit-token", &args, &st, &body);
    let mut body = unpack_both(Some(&["eos"]));
    body += &call(&[v("token"), v("eos")], "u64.lt", &["text-token".into()]);
    body += &call(
        &[
            v("env"),
            "name.whisper.emit-token".into(),
            v("env"),
            "name.whisper.eos-state".into(),
            v("text-token"),
            v("state"),
        ],
        "bool.ifc",
        &["result".into()],
    );
    body += " [.result]\n";
    source += &def("accept-token", &args, &st, &body);
    let mut body = unpack_both(Some(&["text-mask", "begin-mask"]));
    if debug {
        body += &call(
            &[v("trace"), v("logits"), v("count"), c(cfg.vocab_size)],
            "whisper.write-logits",
            &["next-trace".into()],
        );
    }
    body += &call(&[v("count"), c(1)], "u64.lt", &["first".into()]);
    body += &call(
        &[v("logits"), v("text-mask"), v("begin-mask"), v("first")],
        "whisper.select-token",
        &["ready-logits".into(), "selected".into(), "valid".into()],
    );
    let mut nextstate = state.clone();
    nextstate[0] = "ready-logits".into();
    nextstate[5] = "selected".into();
    if debug {
        *nextstate.last_mut().unwrap() = "next-trace".into();
    }
    body += &pack(&nextstate, "next-state");
    body += &call(
        &[
            v("env"),
            "name.whisper.accept-token".into(),
            v("env"),
            "name.whisper.failed-state".into(),
            v("valid"),
            v("next-state"),
        ],
        "bool.ifc",
        &["result".into()],
    );
    body += " [.result]\n";
    source += &def("advance", &args, &st, &body);
    let mut body = unpack_both(Some(&["limit"]));
    body += &call(&[v("count"), v("limit")], "u64.lt", &["in-limit".into()]);
    body += &call(&[v("status"), c(1)], "u64.lt", &["running".into()]);
    body += &call(
        &[v("in-limit"), v("running")],
        "bool.and",
        &["active".into()],
    );
    body += &call(
        &[
            v("env"),
            "name.whisper.advance".into(),
            v("env"),
            "name.whisper.hold-state".into(),
            v("active"),
            v("state"),
        ],
        "bool.ifc",
        &["result".into()],
    );
    body += " [.result]\n";
    source += &def("iterate", &args, &st, &body);
    // Complete transcription entrypoint. The host supplies only features and metadata.
    let mut inputs = vec![
        ("features".into(), REF.into()),
        ("limit".into(), U.into()),
        ("language-override".into(), U.into()),
        ("start".into(), U.into()),
        ("task".into(), U.into()),
        ("no-timestamps".into(), U.into()),
        ("eos".into(), U.into()),
        ("language-mask".into(), REF.into()),
        ("text-mask".into(), REF.into()),
        ("begin-mask".into(), REF.into()),
    ];
    for (prefix, names) in [
        ("ew", graph::encoder_names(cfg)),
        ("cw", graph::cross_names(cfg)),
        ("dw", graph::decoder_names(cfg)),
    ] {
        inputs.extend(
            names
                .iter()
                .enumerate()
                .map(|(i, _)| (format!("{prefix}{i}"), REF.into())),
        );
    }
    let mut body = call(
        &[v("features"), c(3000 * 80)],
        "inference.f32-to-bf16",
        &["mel".into()],
    );
    let mut encoder_args = vec![v("mel")];
    encoder_args.extend((0..graph::encoder_names(cfg).len()).map(|i| v(&format!("ew{i}"))));
    body += &call(&encoder_args, "whisper.encode", &["encoder".into()]);
    if debug {
        body += &call(
            &[v("encoder"), c(cfg.max_source_positions * cfg.d_model)],
            "whisper.copy-bf16",
            &["encoder-snapshot".into()],
        );
    }
    let mut cross_args = vec![v("encoder")];
    cross_args.extend((0..graph::cross_names(cfg).len()).map(|i| v(&format!("cw{i}"))));
    body += &call(&cross_args, "whisper.prepare-cross-kv", &cross);
    for name in &caches {
        body += &call(
            &[c(cfg.max_target_positions * cfg.d_model)],
            "inference.zeros-bf16",
            std::slice::from_ref(name),
        );
    }
    let mut decoder_args = vec![v("start"), c(0)];
    for i in 0..cfg.decoder_layers {
        decoder_args.extend([
            v(&format!("sk{i}")),
            v(&format!("sv{i}")),
            v(&format!("ck{i}")),
            v(&format!("cv{i}")),
        ]);
    }
    decoder_args.extend(weights.iter().map(|n| v(n)));
    let mut outs = vec!["prefix0-logits".into()];
    outs.extend(caches.iter().map(|n| format!("p0-{n}")));
    body += &call(&decoder_args, "whisper.decode-step-owned-cross", &outs);
    body += " (bool.f [not-first.])\n";
    body += &call(
        &[
            v("prefix0-logits"),
            v("language-mask"),
            v("begin-mask"),
            v("not-first"),
        ],
        "whisper.select-token",
        &[
            "language-logits".into(),
            "detected".into(),
            "language-valid".into(),
        ],
    );
    body += &call(
        &[c(0), v("language-override")],
        "u64.lt",
        &["override".into()],
    );
    body += &call(
        &[v("override"), v("language-override"), v("detected")],
        "u64.select",
        &["language".into()],
    );
    let mut previous = "language-logits".to_string();
    for (pos, token) in [(1, "language"), (2, "task"), (3, "no-timestamps")] {
        let mut args = vec![v(token), c(pos)];
        for i in 0..cfg.decoder_layers {
            args.extend([
                v(&format!("p{}-sk{i}", pos - 1)),
                v(&format!("p{}-sv{i}", pos - 1)),
                v(&format!("ck{i}")),
                v(&format!("cv{i}")),
            ]);
        }
        args.extend(weights.iter().map(|n| v(n)));
        let mut out = vec![format!("prefix{pos}-logits")];
        out.extend(caches.iter().map(|n| format!("p{pos}-{n}")));
        body += &call(&args, "whisper.decode-step-owned-cross", &out);
        body += &call(
            &[v(&previous), v(&out[0])],
            "qwen3.free-bf16-before",
            &[format!("ready-prefix{pos}")],
        );
        previous = format!("ready-prefix{pos}");
    }
    body += &call(
        &[c(cfg.max_target_positions - 4 + 3)],
        "inference.zeros-u64",
        &["tokens".into()],
    );
    body += &call(
        &[c(cfg.max_target_positions - 4), v("limit")],
        "u64.lt",
        &["clamp-limit".into()],
    );
    body += &call(
        &[
            v("clamp-limit"),
            c(cfg.max_target_positions - 4),
            v("limit"),
        ],
        "u64.select",
        &["bounded-limit".into()],
    );
    body += &call(
        &[v("override"), v("language-valid")],
        "bool.or",
        &["valid-language".into()],
    );
    body += &call(
        &[v("valid-language"), c(0), c(2)],
        "u64.select",
        &["status".into()],
    );
    let mut initial = vec![
        previous,
        "tokens".into(),
        "initial-count".into(),
        "initial-pos".into(),
        "status".into(),
        "initial-token".into(),
    ];
    initial.extend(caches.iter().map(|n| format!("p3-{n}")));
    if debug {
        body += &call(
            &[c((cfg.max_target_positions - 4) * cfg.vocab_size)],
            "inference.zeros-bf16",
            &["initial-trace".into()],
        );
        initial.push("initial-trace".into());
    }
    body += &format!(
        " {} [initial-count.]\n {} [initial-pos.]\n {} [initial-token.]\n",
        c(0),
        c(3),
        c(0)
    );
    body += &pack(&initial, "state0");
    let mut envnames = env.clone();
    envnames[0] = "bounded-limit".into();
    body += &pack(&envnames, "env");
    // A small tree of shared definitions expresses the bounded calls without
    // hundreds of repeated weight argument lists in one entrypoint.
    let bound = cfg.max_target_positions - 4;
    let mut power = 2;
    while power <= bound {
        let previous = if power == 2 {
            "whisper.iterate".into()
        } else {
            format!("whisper.repeat-{}", power / 2)
        };
        let mut repeat = call(&[v("env"), v("state")], &previous, &["middle".into()]);
        repeat += &call(&[v("env"), v("middle")], &previous, &["result".into()]);
        repeat += " [.result]\n";
        source += &def(&format!("repeat-{power}"), &args, &st, &repeat);
        power *= 2;
    }
    let mut current = "state0".to_string();
    let mut part = 1;
    let mut sequence = 0;
    while part <= bound {
        if bound & part != 0 {
            let op = if part == 1 {
                "whisper.iterate".into()
            } else {
                format!("whisper.repeat-{part}")
            };
            let next = format!("generation-{sequence}");
            body += &call(&[v("env"), v(&current)], &op, std::slice::from_ref(&next));
            current = next;
            sequence += 1;
        }
        part *= 2;
    }
    let final_state = state
        .iter()
        .map(|n| format!("final-{n}"))
        .collect::<Vec<_>>();
    body += &unpack(&final_state, &current);
    // Free all generated state after the final token buffer exists.
    let mut result = "final-tokens".to_string();
    for (i, name) in std::iter::once("final-logits".to_string())
        .chain(caches.iter().map(|n| format!("final-{n}")))
        .chain(cross)
        .enumerate()
    {
        let next = format!("freed{i}");
        body += &call(
            &[v(&name), v(&result)],
            "qwen3.free-bf16-before",
            std::slice::from_ref(&next),
        );
        result = next;
    }
    // Common output ABI: [count, language, status, token...]. Both backends
    // return device memory; the shared Rust harness only unpacks the result.
    for (offset, value) in [(0, "final-count"), (1, "language"), (2, "final-status")] {
        let next = format!("result-header-{offset}");
        body += &call(
            &[v(&result), c(offset), v(value)],
            "whisper.write-token",
            std::slice::from_ref(&next),
        );
        result = next;
    }
    body += &format!(" {{{}}}\n", v(&result));
    if debug {
        body = body.replace(
            &format!("{{{}}}", v(&result)),
            &format!("{{{} [.encoder-snapshot] [.final-trace]}}", v(&result)),
        );
    }
    let output_types = if debug {
        "{(cap.own mem) (cap.own mem) (cap.own mem)}"
    } else {
        "{(cap.own mem)}"
    };
    for (prefix, names) in [
        ("ew", graph::encoder_names(cfg)),
        ("cw", graph::cross_names(cfg)),
        ("dw", graph::decoder_names(cfg)),
    ] {
        for (i, name) in names.iter().enumerate() {
            source += &format!("# {prefix}{i}: {name} (BF16)\n");
        }
    }
    source += &def("transcribe", &inputs, output_types, &body);
    source
}
