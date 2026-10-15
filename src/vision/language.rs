//! The language decoder consumes supplied embeddings but caches by sequence offset.
use crate::{catena::InferencePrecision, spec::ModelConfig};
use anyhow::{Result, ensure};
pub(crate) fn source(
    config: &ModelConfig,
    precision: InferencePrecision,
    entrypoint: String,
) -> Result<String> {
    let half = config.rope_dimension_count / 2;
    let sections = config.mrope_sections;
    ensure!(
        sections.iter().sum::<usize>() == half
            && sections[1] * 3 <= half + 2
            && sections[2] * 3 <= half + 2,
        "unsupported interleaved MRoPE sections {sections:?} for {} dimensions",
        config.rope_dimension_count
    );
    let prologue = "  ([.tokens] mem.cast.u64 [sequence _token-buffer.])\n  ([.sequence] :.forget [sequence-value.])\n  ({[.cache-start] [.sequence-value]} qwen3.rope-table [rope.])\n  ({[.tokens] [.embed-weight]} qwen3.embedding [hidden-00.])\n";
    ensure!(
        entrypoint.contains(prologue),
        "multimodal entrypoint prologue changed"
    );
    let entrypoint=entrypoint
        .replacen("(def program qwen3 :", "(def program qwen3.multimodal :",1)
        .replacen("    (cap.ref mem)\n", "    (cap.own mem)\n    (cap.ref mem)\n",1)
        .replacen("= ([tokens cache token-capacity cache-start embed-weight", "= ([hidden-00 positions cache token-capacity cache-start _embed-weight",1)
        .replacen("    (u64 val)\n    (u64 val)\n", "    (u64 val)\n    (u64 val)\n    (u64 val)\n", 1)
        .replacen("cache token-capacity cache-start _embed-weight", "cache token-capacity cache-start _sequence _embed-weight", 1)
        .replace(prologue,"  (const.u64.0x0000000000000001 [sequence-value.])\n  ([.positions] qwen3.mm-rope-table [rope.])\n");
    let mut out = format!(
        r#"
(def program qwen3.mm-rope-table-at : ([n.] {{(cap.ref mem) ([.n] ix val)}}) -> ([n.] (f32 val))
= ([positions index.]
 ([.index] ix.to-u64 [i.]) (const.u64.0x{half:016X} [half.])
 ([.i half] u64.rem [pair.]) ([.pair] qwen3.rope-frequency [frequency.])
 (const.u64.0x0000000000000003 [three.]) ([.pair three] u64.rem [mod.])
 (const.u64.0x0000000000000001 [one.]) (const.u64.0x0000000000000002 [two.])
 ([.mod one] u64.lte [height-upper.]) ([.mod one] u64.gte [height-lower.])
 ([.height-upper height-lower] bool.and [height-mod.]) ([.mod two] u64.gte [width-mod.])
 (const.u64.0x{height_end:016X} [height-end.]) (const.u64.0x{width_end:016X} [width-end.])
 ([.pair height-end] u64.lt [height-range.]) ([.pair width-end] u64.lt [width-range.])
 ([.height-mod height-range] bool.and [height.]) ([.width-mod width-range] bool.and [width.])
 (const.u64.0x0000000000000000 [zero.]) ([.height one zero] u64.select [axis0.])
 ([.width two axis0] u64.select [axis.])
 ([.positions] mem.cast.u64 [pc pb.]) ({{[.pc] [.pb] [.axis]}} qwen3.u64-buffer-at u64.to-f32 [position.])
 ([.position frequency] f32.mul qwen3.rope-sincos [sine cosine.])
 ([.i half] u64.gte [is-sine.]) ([.is-sine sine cosine] f32.select)
))
(def program qwen3.mm-rope-table : (cap.ref mem) -> (cap.own mem)
= ([positions.] (const.u64.0x{dim:016X} u64.name [n.])
 {{[.n] ({{[.n] [.positions] (([.n] :.param) name.qwen3.mm-rope-table-at)}} materializec)}} buf.to-mem))
"#,
        height_end = sections[1] * 3,
        width_end = sections[2] * 3,
        dim = config.rope_dimension_count
    );
    // Only the ref image-row functions are needed in the language runtime.
    let dtype = match precision {
        InferencePrecision::F32 => "f32",
        InferencePrecision::Bf16 => "bf16",
    };
    let narrow = if precision == InferencePrecision::Bf16 {
        "f32.to-bf16"
    } else {
        ""
    };
    out += &format!(
        r#"
(def program qwen3.mm-image-row-at : ([n.] {{({{(cap.ref mem) (u64 val)}} *) ([.n] ix val)}}) -> ([n.] ({dtype} val))
= ([env index.] ([.env] *.elim [image row.]) ([.index] ix.to-u64 [col.])
 (const.u64.0x{hidden:016X} [width.]) ([.row width] u64.mul [base.]) ([.base col] u64.add [i.])
 ([.image] mem.cast.f32 [count buffer.]) ({{[.count] [.buffer] [.i]}} qwen3.f32-buffer-at {narrow})))
(def program qwen3.mm-image-row : {{(cap.ref mem) (u64 val)}} -> (cap.own mem)
= ([image row.] (const.u64.0x{hidden:016X} u64.name [n.])
 ({{[.image] [.row]}} *.intro [env.])
 {{[.n] ({{[.n] [.env] (([.n] :.param) name.qwen3.mm-image-row-at)}} materializec)}} buf.to-mem))
"#,
        hidden = config.hidden_size
    );
    out += &entrypoint;
    Ok(out)
}
