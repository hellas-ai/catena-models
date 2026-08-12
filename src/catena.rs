use std::fmt::Write;

use crate::spec::{NUM_LAYERS, TENSORS_PER_LAYER, WEIGHT_COUNT};

/// Complete checked-in Catena program used by normal inference.
pub const MODEL_SOURCE: &str = include_str!("catena/qwen3.hex");

const ENTRYPOINT_MARKER: &str = "# Explicit 48-layer Qwen3-30B-A3B composition.";

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

/// Regenerate the complete Catena source, including the explicit 48-layer
/// linear-use composition.
///
/// Normal inference uses [`MODEL_SOURCE`] directly. This function exists only
/// for the opt-in `--generate-entrypoint` maintenance command.
pub fn generated_model_source() -> String {
    let core = MODEL_SOURCE
        .split_once(ENTRYPOINT_MARKER)
        .map_or(MODEL_SOURCE, |(core, _entrypoint)| core)
        .trim_end();
    let mut source = String::from(core);
    source.push_str("\n\n");
    source.push_str(ENTRYPOINT_MARKER);
    source.push('\n');
    source.push_str("(def program qwen3 :\n  {\n");
    for ty in [
        "    (cap.ref mem)\n", // tokens
        "    (cap.own mem)\n", // cache
        "    (u64 val)\n",     // cache capacity
        "    (u64 val)\n",     // cache start
    ] {
        source.push_str(ty);
    }
    for _ in 0..WEIGHT_COUNT {
        source.push_str("    (cap.ref mem)\n");
    }
    source.push_str("  }\n  ->\n  {(cap.own mem) (cap.own mem) (cap.own mem)}\n= ([tokens cache token-capacity cache-start embed-weight");
    for layer in 0..NUM_LAYERS {
        for name in LAYER_WEIGHT_NAMES {
            write!(source, " layer-{layer:02}-{name}").unwrap();
        }
    }
    source.push_str(" final-norm lm-head.]\n");
    source.push_str("  ([.tokens] mem.cast.u64 [sequence _token-buffer.])\n  ([.sequence] :.forget [sequence-value.])\n  ({[.cache-start] [.sequence-value]} qwen3.rope-table [rope.])\n  ({[.tokens] [.embed-weight]} qwen3.embedding [hidden-00.])\n");

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
    source.push_str("  ({[.rope] [.final-logits]} qwen3.free-f32-before [final-logits-after-rope.])\n  ([.final-logits-after-rope] mem.cast.f32 [logit-count logit-buffer.])\n  (const.u64.0x0000000000000001 u64.name [one.])\n  {\n    [.logit-buffer]\n    [.logit-count]\n    [.one]\n  } materializec.borrow-argmax-f32 [logits-after-argmax next-token-buffer.]\n  {\n");
    writeln!(source, "    [.cache-{NUM_LAYERS:02}]").unwrap();
    source.push_str("    ({[.logit-count] [.logits-after-argmax]} buf.to-mem)\n    ({[.one] [.next-token-buffer]} buf.to-mem)\n  }\n))\n");
    source
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use catena_lang::{
        codegen::GpuDialect,
        runtime::{Runtime, Value},
        stdlib,
    };

    #[test]
    fn entrypoint_has_all_layers_and_inputs() {
        let entrypoint = MODEL_SOURCE
            .split(ENTRYPOINT_MARKER)
            .nth(1)
            .expect("entrypoint marker");
        assert_eq!(
            entrypoint.matches("(cap.ref mem)").count(),
            1 + WEIGHT_COUNT
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
    fn checked_in_entrypoint_matches_generator() {
        assert_eq!(MODEL_SOURCE, generated_model_source());
    }

    #[test]
    fn gpu_program_compiles_when_requested() -> Result<()> {
        if std::env::var_os("CATENA_QWEN3_GPU_TESTS").is_none() {
            return Ok(());
        }
        let runtime =
            Runtime::from_sources(stdlib::sources().chain([MODEL_SOURCE]), GpuDialect::Hip)?;
        let sequence = crate::spec::MAX_POSITION_EMBEDDINGS;
        let [output] =
            runtime.exec("qwen3.rope-table", [0_u64.into(), (sequence as u64).into()])?;
        let Value::MemOwn(output) = output else {
            anyhow::bail!("qwen3.rope-table returned non-owned memory: {output:?}");
        };
        let actual = output.try_to_f32_vec()?;
        let half_dim = crate::spec::ROPE_HALF_DIM;
        let plane_len = sequence * half_dim;
        assert_eq!(actual.len(), 2 * plane_len);
        let mut max_sine_error = 0.0_f32;
        let mut max_cosine_error = 0.0_f32;
        for position in 0..sequence {
            for pair in 0..half_dim {
                let exponent = (2 * pair) as f32 / crate::spec::HEAD_DIM as f32;
                let frequency = crate::spec::ROPE_THETA.powf(-exponent);
                let angle = position as f32 * frequency;
                let index = position * half_dim + pair;
                max_cosine_error = max_cosine_error.max((actual[index] - angle.cos()).abs());
                max_sine_error =
                    max_sine_error.max((actual[plane_len + index] - angle.sin()).abs());
            }
        }
        eprintln!(
            "full-context Catena RoPE maximum absolute error: sin={max_sine_error:e}, cos={max_cosine_error:e}"
        );
        assert!(max_sine_error <= 1.0e-6, "sine error is too large");
        assert!(max_cosine_error <= 1.0e-6, "cosine error is too large");
        Ok(())
    }
}
