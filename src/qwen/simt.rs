//! Portable Qwen3 reductions, applied after base graph composition so Qwen3.5's
//! own decode specialization and legacy prefill stay independent.
use crate::catena::KernelKind;

fn definition<'a>(source: &'a str, name: &str) -> &'a str {
    let start = source.find(&format!("(def program {name} :")).unwrap();
    let mut depth = 0isize;
    for (offset, ch) in source[start..].char_indices() {
        depth += match ch {
            '(' => 1,
            ')' => -1,
            _ => 0,
        };
        if depth == 0 {
            return &source[start..start + offset + 1];
        }
    }
    unreachable!("balanced Hex definition")
}

pub(super) fn specialize(mut source: String, kernels: KernelKind) -> String {
    source = source.replace(" bf16x2.dot-f32 ", " stdlib.models.math.dot-pair-f32 ");
    if kernels == KernelKind::Reference {
        super::fused::append_hex(&mut source, include_str!("../catena/cooperative.hex"));
    }
    source.push_str(include_str!("../catena/bf16_simt.hex"));
    let old = definition(&source, "qwen3.project").to_owned();
    source = source.replace(&old, "(def program qwen3.project : {((cap.own mem) val) ((cap.ref mem) val) (u64 val) (u64 val)} -> ((cap.own mem) val) =\n ([input weight features outputs.]\n ({[.input] [.weight] (const.u64.0x0000000000000001) [.features] [.outputs]} stdlib.models.simt.or)\n ))");
    // Cache addressing and BF16 score rounding stay in the existing graph.
    // Split the sequential pair accumulator into independent low/high sums.
    let pair_type = "({(f32 val) (f32 val)} *)";
    let old = definition(&source, "stdlib.qwen3.score-dot").to_owned();
    let mut dot = old.replace("(f32 val)", pair_type);
    dot = dot.replace(
        "([.pair] *.elim [i acc.])",
        "([.pair] *.elim [i sums.])\n  ([.sums] *.elim [low high.])",
    );
    dot = dot.replace(
        "({[.q] [.k] [.acc]} stdlib.models.math.dot-pair-f32 [result.])",
        r#"([.q] bf16x2.low bf16.to-f32 [ql.])
  ([.q] bf16x2.high bf16.to-f32 [qh.])
  ([.k] bf16x2.low bf16.to-f32 [kl.])
  ([.k] bf16x2.high bf16.to-f32 [kh.])
  ({[.ql] [.kl]} stdlib.models.memory.f32.mul [pl.])
  ({[.qh] [.kh]} stdlib.models.memory.f32.mul [ph.])
  ({[.low] [.pl]} stdlib.models.memory.f32.add [sl.])
  ({[.high] [.ph]} stdlib.models.memory.f32.add [sh.])
  ({[.sl] [.sh]} *.intro [result.])"#,
    );
    source = source.replace(&old, &dot);
    let old = definition(&source, "stdlib.qwen3.score-dot.fold").to_owned();
    let fold = old
        .replace("(f32 val)", pair_type)
        .replace("master.index u64", "master.index32 u32")
        .replace("master.index master.end", "master.index32 master.end32")
        .replace(
            "([.index] :.forget [i.])",
            "([.index] :.forget stdlib.models.memory.u32.to-u64 [i.])",
        );
    source = source.replace(&old, &fold);
    let old = definition(&source, "stdlib.qwen3.scores").to_owned();
    let scores = old.replace("[.end] (const.u32.0x00000000 u32.bitcast-f32)", "[.end] ({(const.u32.0x00000000 u32.bitcast-f32) (const.u32.0x00000000 u32.bitcast-f32)} *.intro)")
        .replace("[ sum proof.]", "[ sums proof.]")
        .replace("[.pairs] (unit.intro meta.master.end)", "([.pairs] stdlib.models.memory.fold-count-u32) (unit.intro meta.master.end32)")
        .replace("core.fold.bounded_u64", "core.fold.bounded")
        .replace("([.proof] unsafe.discard unit.elim)", "([.proof] unsafe.discard unit.elim)\n  ([.sums] *.elim [low high.])\n  ({[.low] [.high]} stdlib.models.memory.f32.add [sum.])");
    source.replace(&old, &scores)
}
