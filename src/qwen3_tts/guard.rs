//! Specialize the cached Qwen layer kernels with a device-side EOS guard.
use super::kernels::OWN;
use std::collections::BTreeSet;

pub(super) fn generation(source: &str) -> String {
    let names = source
        .split("(def program ")
        .skip(1)
        .filter_map(|program| program.split_whitespace().next())
        .filter(|name| {
            [
                "scores",
                "softmax",
                "context",
                "fused.project",
                "fused.residual",
                "fused.gate",
                "decoder-layer",
            ]
            .iter()
            .any(|op| *name == format!("qwen3.{op}"))
                || [
                    "stdlib.qwen3.scores",
                    "stdlib.qwen3.softmax",
                    "stdlib.qwen3.context",
                    "stdlib.qwen35.simt.project.run",
                    "stdlib.qwen35.simt.residual.run",
                    "stdlib.qwen35.simt.gate.run",
                ]
                .iter()
                .any(|root| name == root || name.starts_with(&format!("{root}.")))
                || [
                    "stdlib.qwen3.decoder-step.fold",
                    "qwen-tts.predictor.layer",
                    "stdlib.qwen-tts.predictor.layer-step",
                    "stdlib.qwen-tts.talker.token",
                    "stdlib.qwen-tts.predictor.token",
                    "stdlib.qwen-tts.predictor.input",
                ]
                .contains(name)
        })
        .collect::<BTreeSet<_>>();
    let (preamble, _) = source.split_once("(def program ").unwrap();
    let mut result = preamble.to_owned();
    for program in source.split("(def program ").skip(1) {
        let name = program.split_whitespace().next().unwrap();
        let caller = match name {
            "stdlib.qwen-tts.frame" => "progress",
            "stdlib.qwen-tts.predictor.book" | "stdlib.qwen-tts.predictor.last-book" => "progress",
            _ if names.contains(name) => "progress",
            _ => {
                result.push_str("(def program ");
                result.push_str(program);
                continue;
            }
        };
        let mut program = program.to_owned();
        if names.contains(name) {
            let input = program.find(" : {").unwrap() + " : {".len();
            program.insert_str(input, &format!("{OWN} "));
            let locals = program.find("([").unwrap() + 2;
            program.insert_str(locals, "progress ");
        }
        let body = program.find("([").unwrap();
        let limit = match name {
            "stdlib.qwen35.simt.project.run.kernel" => Some("v8"),
            "stdlib.qwen35.simt.residual.run.kernel" => Some("v9"),
            "stdlib.qwen35.simt.gate.run.kernel" => Some("v6"),
            "stdlib.qwen3.scores" => Some("pairs"),
            "stdlib.qwen3.softmax" | "stdlib.qwen3.context" => Some("keys"),
            _ => None,
        };
        result.push_str("(def program ");
        result.push_str(&program[..body]);
        let mut guarded_limit = false;
        for line in program[body..].lines() {
            let mut line = line.to_owned();
            if let Some(limit) = limit {
                // Keep the cache layout and residual stores unchanged, while
                // setting the expensive device reduction bounds to zero.
                if line.contains(&format!("[.{limit}]"))
                    && line.contains("(unit.intro meta.master.end")
                {
                    if !guarded_limit {
                        result.push_str(&format!(
                            "  ({{[.progress] [.{limit}]}} stdlib.qwen-tts.active-count [active-limit.])\n"
                        ));
                        guarded_limit = true;
                    }
                    line = line.replace(&format!("[.{limit}]"), "[.active-limit]");
                }
            }
            if let Some((args, tail)) = line.split_once("} ") {
                let operator = tail
                    .split_whitespace()
                    .next()
                    .unwrap_or_default()
                    .trim_end_matches(')');
                let partial = operator
                    .strip_prefix("partial.")
                    .and_then(|x| x.rsplit_once('.'));
                let callee = partial.map_or(operator, |(callee, _)| callee);
                if names.contains(callee) {
                    let progress = if name == "stdlib.qwen-tts.frame"
                        && (callee.starts_with("stdlib.qwen-tts.predictor.")
                            || callee == "stdlib.qwen-tts.talker.token")
                    {
                        "updated"
                    } else {
                        caller
                    };
                    let start = args.find("({").unwrap() + 2;
                    if let Some((callee, arity)) = partial {
                        let old = format!("partial.{callee}.{arity}");
                        let new =
                            format!("partial.{callee}.{}", arity.parse::<usize>().unwrap() + 1);
                        line = line.replace(&old, &new);
                    }
                    line.insert_str(start, &format!("[.{progress}] "));
                }
            }
            result.push_str(&line);
            result.push('\n');
        }
        assert!(
            limit.is_none() || guarded_limit,
            "missing guard bound for {name}"
        );
    }
    result
}
