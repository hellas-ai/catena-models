//! Small HEX launch and fold adapters used by the synthesis graph.
use crate::qwen::{call, header};
use std::fmt::Write;

pub(super) const OWN: &str = "((cap.own mem) val)";
pub(super) const REF: &str = "((cap.ref mem) val)";
pub(super) const U64: &str = "(u64 val)";
pub(super) const U32: &str = "(u32 val)";
pub(super) const F32: &str = "(f32 val)";
pub(super) const FREE_BF16: &str = "stdlib.models.resources.free-bf16-before";
pub(super) const FREE_F32: &str = "stdlib.models.resources.free-f32-before";
const GRID: &str = "({master.grid ({({master.blocks master.one} stdlib.gpu.geometry.sort.D2) ({master.threads master.one} stdlib.gpu.geometry.sort.D2)} stdlib.gpu.geometry.type.Grid)} :)";
const THREAD: &str = "({({master.block ({({master.blocks master.one} stdlib.gpu.geometry.sort.D2) ({master.threads master.one} stdlib.gpu.geometry.sort.D2) master.grid} stdlib.gpu.geometry.type.Block)} :) (({({master.blocks master.one} stdlib.gpu.geometry.sort.D2) ({master.threads master.one} stdlib.gpu.geometry.sort.D2) master.grid master.block} stdlib.gpu.geometry.type.Thread) val)} *)";

pub(super) fn f(value: f32) -> String {
    format!("(const.u32.0x{:08X} u32.bitcast-f32)", value.to_bits())
}

pub(super) fn u(value: usize) -> String {
    format!("(const.u32.0x{value:08X})")
}

pub(super) fn mem(s: &mut String, op: &str, args: &[&str], out: &str) {
    call(s, args, &format!("stdlib.models.memory.{op}"), out);
}

pub(super) fn finish(s: &mut String, expr: &str) {
    writeln!(s, "  {expr}\n ))").unwrap();
}

pub(super) fn fold_input(state: &str) -> String {
    format!(
        "({{({{master.index32 u32}} :) ({{{state} ({{(({{master.index32 master.end32}} prop.Less) |-) (prop.Truth |-)}} *)}} *)}} *)"
    )
}

pub(super) fn fold_output(state: &str) -> String {
    format!("({{{state} (prop.Truth |-)}} *)")
}

pub(super) fn unfold(s: &mut String) {
    s.push_str("  ([.pair] *.elim [index rest.])\n  ([.rest] *.elim [state proofs.])\n  ([.proofs] *.elim [less truth.])\n  ([.less] unsafe.discard unit.elim)\n  ([.index] :.forget [i32.])\n  ([.i32] stdlib.models.memory.u32.to-u64 [i.])\n");
}

pub(super) fn fold(s: &mut String, op: &str, args: &[&str], count: &str, init: &str, out: &str) {
    call(
        s,
        args,
        &format!("partial.{op}.{}", args.len()),
        &format!("{out}-callback"),
    );
    writeln!(s, "  ({{{count} (unit.intro meta.master.end32)}} smolcat.value.named_at [{out}-end.])\n  ({{[.{out}-end] [.{init}] [.{out}-callback] core.truth}} core.fold.bounded [{out} {out}-proof.])\n  ([.{out}-proof] unsafe.discard unit.elim)").unwrap();
}

pub(super) fn free(s: &mut String, old: &[&str], new: &str, out: &str, op: &str) {
    let mut current = new.to_owned();
    for (index, name) in old.iter().enumerate() {
        let next = if index + 1 == old.len() {
            out.to_owned()
        } else {
            format!("{out}-free{index}")
        };
        call(s, &[name, &current], op, &next);
        current = next;
    }
}

// Device kernels take an output buffer and the linear index after their captures.
// The one-iteration host fold preserves the launch's ordering with later users
// of the output buffer, including kernels that write into an existing buffer.
pub(super) fn launch(
    s: &mut String,
    label: &str,
    args: &[(&str, &str)],
    output_type: &str,
    body: &str,
) {
    let name = format!("stdlib.qwen-tts.{label}");
    let mut inputs = args.to_vec();
    inputs.extend([("out", OWN), ("i", U64)]);
    header(s, &format!("{name}.element"), &inputs, "1");
    s.push_str(body);
    finish(s, "(unit.intro)");

    inputs.pop();
    let mut captured = args.iter().map(|(n, _)| *n).collect::<Vec<_>>();
    captured.push("out");
    let n = captured.len();
    for (suffix, proof) in [
        ("yes", "((master.predicate prop.IsTrue) |-)"),
        (
            "no",
            "(({(master.predicate prop.IsTrue)} stdlib.prelude.prop.Not) |-)",
        ),
    ] {
        let pair = format!("({{{U64} ({{{proof} 1}} *)}} *)");
        let mut branch = inputs.clone();
        branch.push(("pair", &pair));
        header(s, &format!("{name}.{suffix}"), &branch, "1");
        s.push_str("  ([.pair] *.elim [i proofs.])\n  ([.proofs] *.elim [proof unit.])\n  ([.proof] unsafe.discard unit.elim)\n");
        if suffix == "yes" {
            let mut element = captured.clone();
            element.push("i");
            s.push_str("  ([.unit] unit.elim)\n");
            call(s, &element, &format!("{name}.element"), "written");
            finish(s, "[.written]");
        } else {
            finish(s, "[.unit]");
        }
    }
    inputs.push(("work", U64));
    let mut kernel = inputs.clone();
    kernel.push(("thread", THREAD));
    header(s, &format!("{name}.kernel"), &kernel, "1");
    s.push_str("  ([.thread] stdlib.models.memory.thread-index [i.])\n  ({[.i] [.work]} stdlib.models.memory.u64.less [valid.])\n  ({[.valid] (unit.intro meta.master.predicate)} smolcat.value.named_at :.forget [condition.])\n");
    call(s, &captured, &format!("partial.{name}.yes.{n}"), "yes");
    call(s, &captured, &format!("partial.{name}.no.{n}"), "no");
    finish(s, "({[.condition] [.i] [.yes] [.no] unit.intro} core.if)");

    captured.push("work");
    let pair = fold_input(OWN);
    let mut host = inputs.clone();
    host.extend([("grid", GRID), ("pair", &pair)]);
    header(s, &format!("{name}.host"), &host, &fold_output(OWN));
    unfold(s);
    call(
        s,
        &captured,
        &format!("partial.{name}.kernel.{}", n + 1),
        "kernel",
    );
    s.push_str("  ({[.grid] [.kernel]} unsafe.launch unit.elim)\n");
    finish(s, "({[.state] [.truth]} *.intro)");

    header(s, &format!("{name}.into"), &inputs, OWN);
    s.push_str("  ({[.work] (const.u64.0x00000000000000FF)} stdlib.models.memory.u64.add [padded.])\n  ({[.padded] (const.u64.0x0000000000000100)} stdlib.models.memory.u64.div stdlib.models.memory.grid [grid.])\n");
    captured.push("grid");
    fold(
        s,
        &format!("{name}.host"),
        &captured,
        &u(1),
        "out",
        "result",
    );
    finish(s, "[.result]");

    let mut alloc = args.to_vec();
    alloc.push(("work", U64));
    header(s, &name, &alloc, OWN);
    mem(s, &format!("alloc-{output_type}"), &["work"], "out");
    captured.pop(); // grid
    call(s, &captured, &format!("{name}.into"), "result");
    finish(s, "[.result]");
}

pub(super) fn kernel(s: &mut String, label: &str, args: &[&str], out: &str) {
    call(s, args, &format!("stdlib.qwen-tts.{label}"), out);
}

pub(super) fn step_header(s: &mut String, name: &str, captures: &[(&str, &str)], state: &str) {
    let pair = fold_input(state);
    let mut inputs = captures.to_vec();
    inputs.push(("pair", &pair));
    header(s, name, &inputs, &fold_output(state));
    unfold(s);
}

pub(super) fn step_finish(s: &mut String, state: &str) {
    finish(s, &format!("({{[.{state}] [.truth]}} *.intro)"));
}

pub(super) fn emit_generation(s: &mut String) {
    let formulas = include_str!("../catena/qwen3_tts.hex");
    // Each formula is also used directly by the launch adapter as a kernel body.
    let (functions, elements) = formulas.split_once("\n# Element bodies\n").unwrap();
    s.push_str(functions);
    let mut elements = elements.split("# Element ").skip(1);
    let mut emit = |label, args: &[(&str, &str)], dtype| {
        let (name, body) = elements.next().unwrap().split_once('\n').unwrap();
        assert_eq!(label, name);
        launch(s, label, args, dtype, body);
    };
    emit("zero-u64", &[], "u64");
    emit(
        "row",
        &[("input", OWN), ("row", U64), ("width", U64)],
        "bf16",
    );
    emit(
        "prompt-text",
        &[
            ("text", REF),
            ("text-len", U64),
            ("instruction", REF),
            ("instruction-len", U64),
            ("codec-len", U64),
            ("pad", U64),
            ("bos", U64),
            ("eos", U64),
            ("embedding", REF),
            ("width", U64),
        ],
        "bf16",
    );
    emit(
        "prompt-codec",
        &[
            ("projected", OWN),
            ("instruction-len", U64),
            ("codec", REF),
            ("codec-len", U64),
            ("embedding", REF),
            ("pad", U64),
            ("bos", U64),
            ("prompt-rows", U64),
            ("width", U64),
        ],
        "bf16",
    );
    for label in ["bias", "bias-silu"] {
        emit(
            label,
            &[("input", OWN), ("bias", REF), ("width", U64)],
            "bf16",
        );
    }
    emit(
        "embed-code",
        &[("code", OWN), ("embedding", REF), ("width", U64)],
        "bf16",
    );
    emit(
        "progress",
        &[("code", OWN), ("frame", U64), ("eos", U64)],
        "u64",
    );
    emit("store-code", &[("code", OWN), ("offset", U64)], "u64");
    emit("init-sum", &[("input", OWN)], "f32");
    emit("sum-embeddings", &[("input", OWN)], "f32");
    emit("sum-pad", &[("sum", OWN), ("pad", OWN)], "bf16");
    emit(
        "sample-logits",
        &[
            ("input", OWN),
            ("history", OWN),
            ("progress", OWN),
            ("frame", U64),
            ("eos", U64),
            ("temperature", F32),
            ("penalty", F32),
        ],
        "f32",
    );
    emit(
        "sample-noise",
        &[
            ("scores", OWN),
            ("threshold", OWN),
            ("frame", U64),
            ("group", U64),
            ("seed", U64),
        ],
        "bf16",
    );
    emit("sample-code", &[("token", OWN), ("eos", U64)], "u64");
    assert!(elements.next().is_none());
    s.push_str(include_str!("../catena/qwen3_tts_rank.hex"));

    header(
        s,
        "stdlib.qwen-tts.sample",
        &[
            ("input", OWN),
            ("progress", OWN),
            ("history", OWN),
            ("frame", U64),
            ("group", U64),
            ("seed", U64),
            ("eos", U64),
            ("top-k", U64),
            ("temperature", F32),
            ("penalty", F32),
            ("one", U64),
        ],
        OWN,
    );
    kernel(
        s,
        "sample-logits",
        &[
            "input",
            "history",
            "progress",
            "frame",
            "eos",
            "temperature",
            "penalty",
            "(const.u64.0x0000000000000801)",
        ],
        "scores",
    );
    kernel(
        s,
        "sample-threshold",
        &["scores", "progress", "top-k"],
        "threshold",
    );
    kernel(
        s,
        "sample-noise",
        &[
            "scores",
            "threshold",
            "frame",
            "group",
            "seed",
            "(const.u64.0x0000000000000801)",
        ],
        "perturbed",
    );
    call(s, &["perturbed"], "qwen3.fused.argmax", "token");
    kernel(s, "sample-code", &["token", "eos", "one"], "code");
    free(
        s,
        &["scores", "threshold", "token"],
        "code",
        "code-ready",
        FREE_F32,
    );
    free(s, &["perturbed"], "code-ready", "sampled", FREE_BF16);
    finish(s, "[.sampled]");
}
