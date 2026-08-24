use std::fmt::Write;

use anyhow::{Result, ensure};
use catena_lang::stdlib;

use crate::spec::{ModelArchitecture, ModelConfig, NUM_LAYERS, TENSORS_PER_LAYER, WEIGHT_COUNT};

/// Fixed, dimension-free Catena helpers compiled before the generated model.
pub const HELPER_SOURCE: &str = include_str!("catena/qwen3_helpers.hex");

/// Model-specific Catena programs rendered from GGUF metadata at runtime.
const MODEL_TEMPLATE: &str = include_str!("catena/qwen3_model.hex");

/// Catena stdlib, fixed helpers, and one generated model in compilation order.
pub fn runtime_sources<'a>(model_source: &'a str) -> impl Iterator<Item = &'a str> {
    stdlib::sources()
        .map(|source| -> &'a str { source })
        .chain([HELPER_SOURCE, model_source])
}

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

fn generated_entrypoint() -> String {
    let mut source = String::new();
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

fn rope_frequency_program(config: &ModelConfig) -> String {
    let half_dim = config.rope_dimension_count / 2;
    let mut source = String::new();
    writeln!(
        source,
        "# Exact F32 values of {}^(-2 * pair / {}), computed once by the host.",
        config.rope_theta, config.rope_dimension_count
    )
    .unwrap();
    source.push_str("(def program qwen3.rope-frequency : (u64 val) -> (f32 val) =\n  ([pair.]\n");
    for pair in 0..half_dim {
        let exponent = (2 * pair) as f32 / config.rope_dimension_count as f32;
        let frequency = config.rope_theta.powf(-exponent);
        writeln!(
            source,
            "    (const.u32.0x{:08X} u32.bitcast-f32 [frequency-{pair:02}.])",
            frequency.to_bits()
        )
        .unwrap();
    }
    if half_dim == 1 {
        source.push_str("    [.frequency-00]\n");
    } else {
        for pair in 0..half_dim - 1 {
            writeln!(source, "    (const.u64.0x{pair:016X} [pair-{pair:02}.])").unwrap();
            writeln!(
                source,
                "    ([.pair pair-{pair:02}] u64.ne bool.not [is-pair-{pair:02}.])"
            )
            .unwrap();
        }
        for pair in (0..half_dim - 1).rev() {
            let fallback = if pair + 1 == half_dim - 1 {
                format!("frequency-{:02}", pair + 1)
            } else {
                format!("selected-{:02}", pair + 1)
            };
            writeln!(
                source,
                "    ([.is-pair-{pair:02} frequency-{pair:02} {fallback}] f32.select [selected-{pair:02}.])"
            )
            .unwrap();
        }
        source.push_str("    [.selected-00]\n");
    }
    source.push_str("  )\n)");
    source
}

fn render_model_template(config: &ModelConfig) -> Result<String> {
    ensure!(
        config.rope_dimension_count == config.head_dim,
        "the current Qwen3 attention template requires full-head RoPE"
    );
    ensure!(
        config.rope_dimension_count.is_multiple_of(2),
        "rotary dimension must be even"
    );
    let query_heads_per_kv = config.num_attention_heads / config.num_key_value_heads;
    let active_intermediate = config
        .intermediate_size
        .checked_mul(config.experts_per_token)
        .ok_or_else(|| anyhow::anyhow!("active expert intermediate size overflow"))?;
    let expert_stride = config
        .intermediate_size
        .checked_mul(config.hidden_size)
        .ok_or_else(|| anyhow::anyhow!("expert weight stride overflow"))?;
    let attention_scale = (config.head_dim as f32).sqrt().recip();
    let values = [
        ("{{HIDDEN_SIZE}}", format!("{:016X}", config.hidden_size)),
        (
            "{{INTERMEDIATE_SIZE}}",
            format!("{:016X}", config.intermediate_size),
        ),
        ("{{QUERY_SIZE}}", format!("{:016X}", config.query_size())),
        (
            "{{KEY_VALUE_SIZE}}",
            format!("{:016X}", config.key_value_size()),
        ),
        ("{{HEAD_DIM}}", format!("{:016X}", config.head_dim)),
        (
            "{{ROPE_HALF_DIM}}",
            format!("{:016X}", config.rope_dimension_count / 2),
        ),
        (
            "{{ROPE_DIM}}",
            format!("{:016X}", config.rope_dimension_count),
        ),
        (
            "{{QUERY_HEADS}}",
            format!("{:016X}", config.num_attention_heads),
        ),
        (
            "{{KEY_VALUE_HEADS}}",
            format!("{:016X}", config.num_key_value_heads),
        ),
        (
            "{{QUERY_HEADS_PER_KV_HEAD}}",
            format!("{query_heads_per_kv:016X}"),
        ),
        ("{{EXPERT_COUNT}}", format!("{:016X}", config.num_experts)),
        (
            "{{EXPERTS_PER_TOKEN}}",
            format!("{:016X}", config.experts_per_token),
        ),
        (
            "{{ACTIVE_INTERMEDIATE_SIZE}}",
            format!("{active_intermediate:016X}"),
        ),
        ("{{EXPERT_STRIDE}}", format!("{expert_stride:016X}")),
        (
            "{{VOCABULARY_SIZE}}",
            format!("{:016X}", config.vocabulary_size),
        ),
        (
            "{{RMS_EPSILON_BITS}}",
            format!("{:08X}", config.rms_norm_eps.to_bits()),
        ),
        (
            "{{ATTENTION_SCALE_BITS}}",
            format!("{:08X}", attention_scale.to_bits()),
        ),
        ("{{ROPE_FREQUENCY_PROGRAM}}", rope_frequency_program(config)),
    ];
    let mut source = MODEL_TEMPLATE.to_owned();
    for (placeholder, value) in values {
        ensure!(
            source.contains(placeholder),
            "model template is missing {placeholder}"
        );
        source = source.replace(placeholder, &value);
    }
    ensure!(
        !source.contains("{{"),
        "model template contains an unknown placeholder"
    );
    Ok(source)
}

/// Generate all model-specific programs plus the explicit linear-use entrypoint.
pub fn generated_model_source() -> String {
    generated_model_source_for(&ModelConfig::qwen3_30b())
        .expect("the built-in Qwen3 configuration must be renderable")
}

/// Generate the source for a selected GGUF configuration. The current Catena
/// core implements conventional Qwen3 MoE attention only; rejecting the hybrid
/// architecture here prevents writing a plausible-looking but invalid program.
pub fn generated_model_source_for(config: &ModelConfig) -> Result<String> {
    config.validate()?;
    let legacy = ModelConfig::qwen3_30b();
    ensure!(
        config.architecture == ModelArchitecture::Qwen3Moe,
        "Qwen3.5/3.6 source generation requires gated-delta-net, shared-expert, \
         gated-attention, partial-RoPE, and offset-RMSNorm Catena primitives"
    );
    ensure!(
        config.hidden_size == legacy.hidden_size
            && config.intermediate_size == legacy.intermediate_size
            && config.num_layers == legacy.num_layers
            && config.num_attention_heads == legacy.num_attention_heads
            && config.num_key_value_heads == legacy.num_key_value_heads
            && config.head_dim == legacy.head_dim
            && config.num_experts == legacy.num_experts
            && config.experts_per_token == legacy.experts_per_token,
        "the generated entrypoint still uses the fixed Qwen3-30B tensor ABI"
    );
    let mut source = render_model_template(config)?;
    source.push('\n');
    source.push_str(&generated_entrypoint());
    Ok(source)
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use catena_lang::{
        codegen::GpuDialect,
        runtime::{Runtime, Value},
    };
    use std::collections::{BTreeMap, BTreeSet};

    fn qwen3_program_references(source: &str) -> BTreeMap<String, BTreeSet<String>> {
        const DEFINITION: &str = "(def program ";
        let definitions = source
            .match_indices(DEFINITION)
            .filter_map(|(offset, _)| {
                let name_start = offset + DEFINITION.len();
                let name = source[name_start..].split_whitespace().next()?;
                (name == "qwen3" || name.starts_with("qwen3.")).then_some((offset, name))
            })
            .collect::<Vec<_>>();
        definitions
            .iter()
            .enumerate()
            .map(|(index, (offset, name))| {
                let end = definitions
                    .get(index + 1)
                    .map_or(source.len(), |(offset, _)| *offset);
                let mut references = source[*offset..end]
                    .lines()
                    .map(|line| line.split_once('#').map_or(line, |(code, _)| code))
                    .flat_map(|line| {
                        line.split(|character: char| {
                            !(character.is_ascii_alphanumeric()
                                || character == '.'
                                || character == '-')
                        })
                    })
                    .map(|token| token.strip_prefix("name.").unwrap_or(token))
                    .filter(|token| *token == "qwen3" || token.starts_with("qwen3."))
                    .map(str::to_owned)
                    .collect::<BTreeSet<_>>();
                references.remove(*name);
                ((*name).to_owned(), references)
            })
            .collect()
    }

    #[test]
    fn entrypoint_has_all_layers_and_inputs() {
        let source = generated_model_source();
        let entrypoint = source
            .split_once(ENTRYPOINT_MARKER)
            .map(|(_, entrypoint)| entrypoint)
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
    fn helpers_and_generated_model_are_separate_sources() {
        assert!(!HELPER_SOURCE.contains(ENTRYPOINT_MARKER));
        assert!(!HELPER_SOURCE.contains("(def program qwen3 :"));
        assert!(!HELPER_SOURCE.contains("(def program qwen3.embedding :"));
        assert!(!HELPER_SOURCE.contains("hidden-size"));
        let model = generated_model_source();
        assert!(model.contains(ENTRYPOINT_MARKER));
        assert!(model.contains("(def program qwen3.embedding :"));
        assert!(model.contains("(def program qwen3.cached-decoder-layer :"));
        assert!(!model.contains("{{"));
    }

    #[test]
    fn model_constants_are_rendered_from_configuration() -> Result<()> {
        let mut config = ModelConfig::qwen3_30b();
        config.vocabulary_size = 42;
        config.rms_norm_eps = 2.0e-5;
        config.rope_theta = 10_000.0;
        let model = generated_model_source_for(&config)?;
        assert!(model.contains("const.u64.0x000000000000002A [vocabulary-size-value.]"));
        assert!(model.contains(&format!(
            "const.u32.0x{:08X} u32.bitcast-f32 [epsilon.]",
            config.rms_norm_eps.to_bits()
        )));
        let second_frequency = config
            .rope_theta
            .powf(-2.0 / config.rope_dimension_count as f32);
        assert!(model.contains(&format!(
            "const.u32.0x{:08X} u32.bitcast-f32 [frequency-01.]",
            second_frequency.to_bits()
        )));
        Ok(())
    }

    #[test]
    fn built_in_rope_constants_preserve_the_previous_bits() {
        let model = generated_model_source();
        assert!(model.contains("const.u32.0x3F4E4BAC u32.bitcast-f32 [frequency-01.]"));
        assert!(model.contains("const.u32.0x35A68E4C u32.bitcast-f32 [frequency-63.]"));
    }

    #[test]
    fn every_qwen3_program_is_reachable_from_the_model_entrypoint() {
        let source = format!("{HELPER_SOURCE}\n{}", generated_model_source());
        let programs = qwen3_program_references(&source);
        let mut reachable = BTreeSet::new();
        let mut pending = vec!["qwen3".to_owned()];
        while let Some(program) = pending.pop() {
            if !reachable.insert(program.clone()) {
                continue;
            }
            pending.extend(programs.get(&program).into_iter().flatten().cloned());
        }
        let unreachable = programs
            .keys()
            .filter(|program| !reachable.contains(*program))
            .cloned()
            .collect::<Vec<_>>();
        assert!(
            unreachable.is_empty(),
            "programs unreachable from qwen3: {unreachable:?}"
        );
    }

    #[test]
    fn gpu_program_compiles_when_requested() -> Result<()> {
        if std::env::var_os("CATENA_QWEN3_GPU_TESTS").is_none() {
            return Ok(());
        }
        let model = generated_model_source();
        let runtime = Runtime::from_sources(runtime_sources(&model), GpuDialect::Hip)?;
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
