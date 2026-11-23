//! Host-side HEX scheduling for Catena versions with device readback.
use super::kernels::*;
use crate::{
    catena,
    qwen::{c, call, header, nested_product, pack, unpack},
};

const READ: &str = "stdlib.gpu.memory.host_read_u64";
const SCHEDULE: &str = "stdlib.qwen-tts.schedule";
pub(super) const FRAMES_PER_CALL: usize = 4;

pub(super) fn supported() -> bool {
    catena::core_sources().any(|source| source.contains(&format!("(arr {READ}")))
}

pub(super) fn emit(
    s: &mut String,
    inputs: &[(&str, &str)],
    names: &[String],
    model_len: usize,
    state: &[&str],
) {
    let state_names = state
        .iter()
        .map(|name| (*name).to_owned())
        .collect::<Vec<_>>();
    let buffers = nested_product(&vec![OWN.to_owned(); state.len()]);
    let state_type = format!("({{{buffers} {U64}}} *)");
    let group_names = (0..model_len.div_ceil(16))
        .map(|group| format!("weights-{group}"))
        .collect::<Vec<_>>();
    let groups = names[..model_len].chunks(16).collect::<Vec<_>>();
    let weights = nested_product(
        &groups
            .iter()
            .map(|group| nested_product(&vec![REF.to_owned(); group.len()]))
            .collect::<Vec<_>>(),
    );
    let captures = [
        ("weights", weights.as_str()),
        ("budget", U64),
        ("capacity", U64),
        ("prompt-rows", U64),
        ("seed", U64),
    ];
    let context = format!("({{{buffers} {U64}}} *)");
    for (suffix, proof) in [
        ("advance", "((master.predicate prop.IsTrue) |-)"),
        (
            "done",
            "(({(master.predicate prop.IsTrue)} stdlib.prelude.prop.Not) |-)",
        ),
    ] {
        let branch = format!("({{{context} ({{{proof} 1}} *)}} *)");
        let mut args = if suffix == "advance" {
            captures.to_vec()
        } else {
            Vec::new()
        };
        args.push(("branch", &branch));
        header(s, &format!("{SCHEDULE}.{suffix}"), &args, &state_type);
        s.push_str("  ([.branch] *.elim [context proofs.])\n  ([.proofs] *.elim [proof unit.])\n  ([.proof] unsafe.discard unit.elim)\n  ([.unit] unit.elim)\n  ([.context] *.elim [buffers start.])\n");
        if suffix == "done" {
            finish(s, &format!("({{[.buffers] {}}} *.intro)", c(1)));
            continue;
        }
        unpack(s, "weights", &group_names);
        for (label, group) in group_names.iter().zip(&groups) {
            unpack(s, label, group);
        }
        unpack(s, "buffers", &state_names);
        mem(s, "u64.sub", &["budget", "start"], "remaining");
        mem(s, "u64.less", &["remaining", &c(FRAMES_PER_CALL)], "tail");
        mem(
            s,
            "select-u64",
            &["tail", "remaining", &c(FRAMES_PER_CALL)],
            "count",
        );
        mem(s, "fold-count-u32", &["count"], "steps");
        let mut args = vec!["start", "steps", "capacity", "prompt-rows", "seed"];
        args.extend_from_slice(state);
        args.extend(names[..model_len].iter().map(String::as_str));
        let next = state
            .iter()
            .map(|name| format!("next-{name}"))
            .collect::<Vec<_>>();
        call(s, &args, "qwen3-tts.advance", &next.join(" "));
        call(s, &["next-progress", &c(1)], READ, "stopped");
        pack(s, &next, "updated");
        finish(s, "({[.updated] [.stopped]} *.intro)");
    }

    step_header(s, SCHEDULE, &captures, &state_type);
    s.push_str("  ([.state] *.elim [buffers stopped.])\n");
    mem(s, "u64.equal", &["stopped", &c(0)], "active");
    mem(s, "u64.mul", &["i", &c(FRAMES_PER_CALL)], "start");
    call(s, &["buffers", "start"], "*.intro", "context");
    s.push_str("  ({[.active] (unit.intro meta.master.predicate)} smolcat.value.named_at :.forget [condition.])\n");
    call(
        s,
        &["weights", "budget", "capacity", "prompt-rows", "seed"],
        &format!("partial.{SCHEDULE}.advance.5"),
        "advance",
    );
    s.push_str(&format!("  ({{[.condition] [.context] [.advance] (name.{SCHEDULE}.done lift) unit.intro}} core.if [next.])\n"));
    step_finish(s, "next");

    let mut args = inputs.to_vec();
    args.push(("seed", U64));
    args.extend(names[model_len..].iter().map(|name| (name.as_str(), REF)));
    header(
        s,
        "qwen3-tts.synthesize",
        &args,
        &format!("{{{OWN} {OWN}}}"),
    );
    let prepare = inputs.iter().map(|(name, _)| *name).collect::<Vec<_>>();
    call(s, &prepare, "qwen3-tts.prepare", &state.join(" "));
    for (group, weights) in group_names.iter().zip(&groups) {
        pack(s, weights, group);
    }
    pack(s, &group_names, "weights");
    pack(s, &state_names, "initial-buffers");
    call(s, &["initial-buffers", &c(0)], "*.intro", "initial");
    mem(s, "u64.add", &["text-len", "instruction-len"], "text-rows");
    mem(
        s,
        "u64.add",
        &["text-rows", "codec-len"],
        "conditioned-rows",
    );
    mem(s, "u64.add", &["conditioned-rows", &c(2)], "prompt-rows");
    mem(s, "u32.to-u64", &["max-frames"], "budget");
    mem(s, "u64.add", &["prompt-rows", "budget"], "capacity");
    mem(
        s,
        "u64.add",
        &["budget", &c(FRAMES_PER_CALL - 1)],
        "rounded",
    );
    mem(s, "u64.div", &["rounded", &c(FRAMES_PER_CALL)], "groups");
    mem(s, "fold-count-u32", &["groups"], "count");
    fold(
        s,
        SCHEDULE,
        &["weights", "budget", "capacity", "prompt-rows", "seed"],
        "[.count]",
        "initial",
        "generated",
    );
    s.push_str("  ([.generated] *.elim [final-buffers stopped.])\n");
    let final_names = state
        .iter()
        .map(|name| format!("final-{name}"))
        .collect::<Vec<_>>();
    unpack(s, "final-buffers", &final_names);
    call(s, &["final-progress", &c(0)], READ, "frames");
    free(
        s,
        &[
            "final-prompt-cache",
            "final-ready-hidden",
            "final-predictor-cache",
            "final-pad",
        ],
        "final-codes",
        "codes-bf",
        FREE_BF16,
    );
    free(s, &["final-sum"], "codes-bf", "codes-ready", FREE_F32);
    let mut decode = vec!["codes-ready", "final-progress", "frames"];
    decode.extend(names[model_len..].iter().map(String::as_str));
    kernel(s, "decode", &decode, "wave");
    free(s, &["codes-ready"], "wave", "pcm", FREE_F32);
    finish(s, "{[.pcm] [.final-progress]}");
}
