use std::fmt::Write;

use crate::spec::{NUM_LAYERS, TENSORS_PER_LAYER, WEIGHT_COUNT};

pub const MODEL_CORE_SOURCE: &str = include_str!("catena/qwen3.hex");

const LAYER_WEIGHT_NAMES: [&str; TENSORS_PER_LAYER] = [
    "input-norm",
    "q-weight",
    "k-weight",
    "v-weight",
    "o-weight",
    "q-norm",
    "k-norm",
    "post-norm",
    "router-weight",
    "expert-gate-weight",
    "expert-up-weight",
    "expert-down-weight",
];

/// Generate the explicit 48-layer linear-use composition. Keeping this
/// mechanical ABI out of the handwritten Catena source makes the layer kernel
/// readable while retaining compile-time ownership checking for every weight.
pub fn model_source() -> String {
    let mut source = String::from(MODEL_CORE_SOURCE);
    source.push_str("\n# Explicit 48-layer Qwen3-30B-A3B composition.\n");
    source.push_str("(def program qwen3 :\n  {\n");
    for ty in [
        "    (cap.ref mem)\n", // tokens
        "    (cap.ref mem)\n", // RoPE
        "    (cap.own mem)\n", // cache
        "    (u64 val)\n",     // cache capacity
        "    (u64 val)\n",     // cache start
    ] {
        source.push_str(ty);
    }
    for _ in 0..WEIGHT_COUNT {
        source.push_str("    (cap.ref mem)\n");
    }
    source.push_str("  }\n  ->\n  {(cap.own mem) (cap.own mem) (cap.own mem)}\n= ([tokens rope cache token-capacity cache-start embed-weight");
    for layer in 0..NUM_LAYERS {
        for name in LAYER_WEIGHT_NAMES {
            write!(source, " layer-{layer:02}-{name}").unwrap();
        }
    }
    source.push_str(" final-norm lm-head.]\n");
    source.push_str("  ([.tokens] mem.cast.u64 [sequence _token-buffer.])\n  ([.sequence] :.forget [sequence-value.])\n  ({[.tokens] [.embed-weight]} qwen3.embedding [hidden-00.])\n");

    for layer in 0..NUM_LAYERS {
        let cache_in = if layer == 0 {
            "cache".to_owned()
        } else {
            format!("cache-{layer:02}")
        };
        writeln!(source, "  ({{").unwrap();
        writeln!(source, "    [.hidden-{layer:02}]").unwrap();
        source.push_str("    [.rope]\n    [.sequence-value]\n");
        writeln!(source, "    [.{cache_in}]").unwrap();
        source.push_str("    [.token-capacity]\n    [.cache-start]\n");
        writeln!(source, "    (const.u64.0x{layer:016X})").unwrap();
        for name in LAYER_WEIGHT_NAMES {
            writeln!(source, "    [.layer-{layer:02}-{name}]").unwrap();
        }
        writeln!(
            source,
            "  }} qwen3.cached-decoder-layer [hidden-{:02} cache-{:02}.])",
            layer + 1,
            layer + 1
        )
        .unwrap();
    }

    writeln!(
        source,
        "  ([.hidden-{NUM_LAYERS:02}] qwen3.rms-inv [final-inverse.])"
    )
    .unwrap();
    writeln!(source, "  ({{").unwrap();
    writeln!(source, "    [.hidden-{NUM_LAYERS:02}]").unwrap();
    source.push_str("    [.final-norm]\n    [.lm-head]\n    [.final-inverse]\n    [.sequence-value]\n  } qwen3.final-logits [logits.])\n");
    source.push_str(
        "  ({[.final-inverse] [.logits]} qwen3.free-f32-before [logits-after-inverse.])\n",
    );
    writeln!(
        source,
        "  ({{[.hidden-{NUM_LAYERS:02}] [.logits-after-inverse]}} qwen3.free-f32-before [final-logits.])"
    )
    .unwrap();
    source.push_str("  ([.final-logits] mem.cast.f32 [logit-count logit-buffer.])\n  (const.u64.0x0000000000000001 u64.name [one.])\n  {\n    [.logit-buffer]\n    [.logit-count]\n    [.one]\n  } materializec.borrow-argmax-f32 [logits-after-argmax next-token-buffer.]\n  {\n");
    writeln!(source, "    [.cache-{NUM_LAYERS:02}]").unwrap();
    source.push_str("    ({[.logit-count] [.logits-after-argmax]} buf.to-mem)\n    ({[.one] [.next-token-buffer]} buf.to-mem)\n  }\n))\n");
    source
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use catena_lang::{codegen::GpuDialect, runtime::Runtime, stdlib};

    #[test]
    fn entrypoint_has_all_layers_and_inputs() {
        let source = model_source();
        let entrypoint = source
            .split("# Explicit 48-layer Qwen3-30B-A3B composition.")
            .nth(1)
            .expect("entrypoint marker");
        assert_eq!(
            entrypoint.matches("(cap.ref mem)").count(),
            2 + WEIGHT_COUNT
        );
        assert_eq!(
            entrypoint.matches("qwen3.cached-decoder-layer").count(),
            NUM_LAYERS
        );
        assert_eq!(entrypoint.matches("(cap.own mem)").count(), 4);
        assert_eq!(entrypoint.matches("(u64 val)").count(), 2);
        for layer in 0..=NUM_LAYERS {
            assert!(entrypoint.contains(&format!("hidden-{layer:02}")));
        }
    }

    #[test]
    fn gpu_program_compiles_when_requested() -> Result<()> {
        if std::env::var_os("CATENA_QWEN3_GPU_TESTS").is_none() {
            return Ok(());
        }
        let source = model_source();
        let mut sources = stdlib::sources().collect::<Vec<&str>>();
        sources.push(source.as_str());
        Runtime::from_sources(sources, GpuDialect::Hip)?;
        Ok(())
    }
}
