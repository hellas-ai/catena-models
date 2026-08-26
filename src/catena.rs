use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::Write,
};

use anyhow::{Result, ensure};
use catena_lang::stdlib;

use crate::spec::{LayerType, ModelArchitecture, ModelConfig, weight_specs};

/// Fixed, dimension-free Catena helpers compiled before the generated model.
pub const HELPER_SOURCE: &str = include_str!("catena/qwen3_helpers.hex");
pub const BF16_HELPER_SOURCE: &str = include_str!("catena/qwen3_bf16_helpers.hex");

/// Model-specific Catena programs rendered from GGUF metadata at runtime.
const MODEL_TEMPLATE: &str = include_str!("catena/qwen3_model.hex");
const BF16_MODEL_TEMPLATE: &str = include_str!("catena/qwen3_bf16_model.hex");
const QWEN35_TEMPLATE: &str = include_str!("catena/qwen35_model.hex");
const QWEN35_BF16_TEMPLATE: &str = include_str!("catena/qwen35_bf16_model.hex");

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferencePrecision {
    F32,
    Bf16,
}

/// Catena stdlib, fixed helpers, and one generated model in compilation order.
pub fn runtime_sources<'a>(
    model_source: &'a str,
    precision: InferencePrecision,
) -> impl Iterator<Item = &'a str> {
    let helpers = match precision {
        InferencePrecision::F32 => HELPER_SOURCE,
        InferencePrecision::Bf16 => BF16_HELPER_SOURCE,
    };
    stdlib::sources()
        .map(|source| -> &'a str { source })
        .chain([helpers, model_source])
}

const ENTRYPOINT_MARKER: &str = "# Explicit metadata-derived Qwen3 composition.";

const MOE_LAYER_WEIGHT_NAMES: [&str; 12] = [
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

const DENSE_LAYER_WEIGHT_NAMES: [&str; 11] = [
    "input-norm",
    "q-weight",
    "k-weight",
    "v-weight",
    "o-weight",
    "q-norm",
    "k-norm",
    "post-norm",
    "gate-weight",
    "up-weight",
    "down-weight",
];

const QWEN35_LINEAR_LAYER_WEIGHT_NAMES: [&str; 14] = [
    "input-norm",
    "qkv-weight",
    "z-weight",
    "alpha-weight",
    "beta-weight",
    "dt-bias",
    "a-log",
    "linear-norm",
    "linear-out-weight",
    "conv-weight",
    "post-norm",
    "gate-weight",
    "up-weight",
    "down-weight",
];

const QWEN35_MOE_FULL_LAYER_WEIGHT_NAMES: [&str; 16] = [
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
    "shared-router-weight",
    "shared-gate-weight",
    "shared-up-weight",
    "shared-down-weight",
];

const QWEN35_MOE_LINEAR_LAYER_WEIGHT_NAMES: [&str; 19] = [
    "input-norm",
    "qkv-weight",
    "z-weight",
    "alpha-weight",
    "beta-weight",
    "dt-bias",
    "a-log",
    "linear-norm",
    "linear-out-weight",
    "conv-weight",
    "post-norm",
    "router-weight",
    "expert-gate-weight",
    "expert-up-weight",
    "expert-down-weight",
    "shared-router-weight",
    "shared-gate-weight",
    "shared-up-weight",
    "shared-down-weight",
];

fn layer_weight_names(config: &ModelConfig, layer: usize) -> Result<&'static [&'static str]> {
    match (config.architecture, config.layer_type(layer)) {
        (ModelArchitecture::Qwen3Dense, _) => Ok(&DENSE_LAYER_WEIGHT_NAMES),
        (ModelArchitecture::Qwen3Moe, _) => Ok(&MOE_LAYER_WEIGHT_NAMES),
        (ModelArchitecture::Qwen35Dense, LayerType::FullAttention) => Ok(&DENSE_LAYER_WEIGHT_NAMES),
        (ModelArchitecture::Qwen35Dense, LayerType::LinearAttention) => {
            Ok(&QWEN35_LINEAR_LAYER_WEIGHT_NAMES)
        }
        (ModelArchitecture::Qwen35Moe, LayerType::FullAttention) => {
            Ok(&QWEN35_MOE_FULL_LAYER_WEIGHT_NAMES)
        }
        (ModelArchitecture::Qwen35Moe, LayerType::LinearAttention) => {
            Ok(&QWEN35_MOE_LINEAR_LAYER_WEIGHT_NAMES)
        }
    }
}

fn generated_entrypoint(config: &ModelConfig, precision: InferencePrecision) -> Result<String> {
    let weight_count = weight_specs(config)?.len();
    let described_layer_weight_count =
        (0..config.num_layers).try_fold(0_usize, |total, layer| {
            total
                .checked_add(layer_weight_names(config, layer)?.len())
                .ok_or_else(|| anyhow::anyhow!("Qwen3 entrypoint weight count overflowed"))
        })?;
    let described_weight_count = described_layer_weight_count
        .checked_add(3)
        .ok_or_else(|| anyhow::anyhow!("Qwen3 entrypoint weight count overflowed"))?;
    ensure!(
        weight_count == described_weight_count,
        "Qwen3 entrypoint describes {described_weight_count} weights, but metadata produced {weight_count}"
    );
    let mut source = String::new();
    source.push_str("(def program qwen3 :\n");
    source.push_str(ENTRYPOINT_MARKER);
    source.push_str("\n  {\n");
    for ty in [
        "    (cap.ref mem)\n", // tokens
        "    (cap.own mem)\n", // cache
        "    (u64 val)\n",     // cache capacity
        "    (u64 val)\n",     // cache start
    ] {
        source.push_str(ty);
    }
    for _ in 0..weight_count {
        source.push_str("    (cap.ref mem)\n");
    }
    source.push_str("  }\n  ->\n  {(cap.own mem) (cap.own mem) (cap.own mem)}\n= ([tokens cache token-capacity cache-start embed-weight");
    for layer in 0..config.num_layers {
        for name in layer_weight_names(config, layer)? {
            write!(source, " layer-{layer:02}-{name}").unwrap();
        }
    }
    source.push_str(" final-norm lm-head.]\n");
    source.push_str("  ([.tokens] mem.cast.u64 [sequence _token-buffer.])\n  ([.sequence] :.forget [sequence-value.])\n  ({[.cache-start] [.sequence-value]} qwen3.rope-table [rope.])\n  ({[.tokens] [.embed-weight]} qwen3.embedding [hidden-00.])\n");

    for layer in 0..config.num_layers {
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
        for name in layer_weight_names(config, layer)? {
            writeln!(source, "    [.layer-{layer:02}-{name}]").unwrap();
        }
        let decoder = match (config.architecture, config.layer_type(layer)) {
            (ModelArchitecture::Qwen3Dense, _) => "qwen3.cached-dense-decoder-layer",
            (ModelArchitecture::Qwen3Moe, _) => "qwen3.cached-decoder-layer",
            (ModelArchitecture::Qwen35Dense, LayerType::FullAttention) => {
                "qwen3.qwen35-full-decoder-layer"
            }
            (ModelArchitecture::Qwen35Dense, LayerType::LinearAttention) => {
                "qwen3.qwen35-linear-decoder-layer"
            }
            (ModelArchitecture::Qwen35Moe, LayerType::FullAttention) => {
                "qwen3.qwen35-full-moe-decoder-layer"
            }
            (ModelArchitecture::Qwen35Moe, LayerType::LinearAttention) => {
                "qwen3.qwen35-linear-moe-decoder-layer"
            }
        };
        writeln!(
            source,
            "  }} {decoder} [hidden-{:02} cache-{:02}.])",
            layer + 1,
            layer + 1
        )
        .unwrap();
    }

    writeln!(
        source,
        "  ([.hidden-{:02}] qwen3.rms-inv [final-inverse.])",
        config.num_layers
    )
    .unwrap();
    writeln!(source, "  ({{").unwrap();
    writeln!(source, "    [.hidden-{:02}]", config.num_layers).unwrap();
    source.push_str("    [.final-norm]\n    [.lm-head]\n    [.final-inverse]\n    [.sequence-value]\n  } qwen3.final-logits [logits.])\n");
    source.push_str(
        "  ({[.final-inverse] [.logits]} qwen3.free-f32-before [logits-after-inverse.])\n",
    );
    let activation_free = match precision {
        InferencePrecision::F32 => "qwen3.free-f32-before",
        InferencePrecision::Bf16 => "qwen3.free-bf16-before",
    };
    writeln!(
        source,
        "  ({{[.hidden-{:02}] [.logits-after-inverse]}} {activation_free} [final-logits.])",
        config.num_layers,
    )
    .unwrap();
    let (logit_cast, argmax) = match precision {
        InferencePrecision::F32 => ("mem.cast.f32", "materializec.borrow-argmax-f32"),
        InferencePrecision::Bf16 => ("mem.cast.bf16", "materializec.borrow-argmax-bf16"),
    };
    write!(source, "  ({{[.rope] [.final-logits]}} qwen3.free-f32-before [final-logits-after-rope.])\n  ([.final-logits-after-rope] {logit_cast} [logit-count logit-buffer.])\n  (const.u64.0x0000000000000001 u64.name [one.])\n  {{\n    [.logit-buffer]\n    [.logit-count]\n    [.one]\n  }} {argmax} [logits-after-argmax next-token-buffer.]\n  {{\n").unwrap();
    writeln!(source, "    [.cache-{:02}]", config.num_layers).unwrap();
    source.push_str("    ({[.logit-count] [.logits-after-argmax]} buf.to-mem)\n    ({[.one] [.next-token-buffer]} buf.to-mem)\n  }\n))\n");
    Ok(source)
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

fn render_model_template(config: &ModelConfig, precision: InferencePrecision) -> Result<String> {
    ensure!(
        config.rope_dimension_count.is_multiple_of(2),
        "rotary dimension must be even"
    );
    let query_heads_per_kv = config.num_attention_heads / config.num_key_value_heads;
    let routed_experts = if config.is_dense() {
        1
    } else {
        config.experts_per_token
    };
    let rendered_expert_count = if config.is_dense() {
        1
    } else {
        config.num_experts
    };
    let active_intermediate = config
        .intermediate_size
        .checked_mul(routed_experts)
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
        ("{{EXPERT_COUNT}}", format!("{rendered_expert_count:016X}")),
        ("{{EXPERTS_PER_TOKEN}}", format!("{routed_experts:016X}")),
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
    let mut source = match precision {
        InferencePrecision::F32 => MODEL_TEMPLATE,
        InferencePrecision::Bf16 => BF16_MODEL_TEMPLATE,
    }
    .to_owned();
    if config.is_qwen35() {
        source.push('\n');
        source.push_str(match precision {
            InferencePrecision::F32 => QWEN35_TEMPLATE,
            InferencePrecision::Bf16 => QWEN35_BF16_TEMPLATE,
        });
    }
    for (placeholder, value) in values {
        ensure!(
            source.contains(placeholder),
            "model template is missing {placeholder}"
        );
        source = source.replace(placeholder, &value);
    }
    if config.is_qwen35() {
        let linear_qkv_size = config.linear_qkv_size().unwrap();
        let linear_key_size = config.linear_key_size().unwrap();
        let linear_value_size = config.linear_inner_size.unwrap();
        let linear_key_heads = config.linear_group_count.unwrap();
        let linear_value_heads = config.linear_time_step_rank.unwrap();
        let linear_key_head_dim = config.linear_state_size.unwrap();
        let linear_value_head_dim = config.linear_value_head_size().unwrap();
        let linear_conv_kernel = config.linear_conv_kernel.unwrap();
        let linear_conv_state_size = linear_qkv_size
            .checked_mul(linear_conv_kernel - 1)
            .ok_or_else(|| anyhow::anyhow!("linear convolution state size overflowed"))?;
        let linear_recurrent_state_size = linear_value_heads
            .checked_mul(linear_key_head_dim)
            .and_then(|value| value.checked_mul(linear_value_head_dim))
            .ok_or_else(|| anyhow::anyhow!("linear recurrent state size overflowed"))?;
        let linear_attention_scale = (linear_key_head_dim as f32).sqrt().recip();
        let shared_intermediate_size = config.shared_expert_intermediate_size.unwrap_or(1);
        for (placeholder, value) in [
            (
                "{{QWEN35_QUERY_PROJECTION_SIZE}}",
                format!("{:016X}", config.query_size() * 2),
            ),
            (
                "{{KV_CACHE_ELEMENTS_PER_TOKEN}}",
                format!("{:016X}", config.kv_cache_elements_per_token()?),
            ),
            ("{{LINEAR_QKV_SIZE}}", format!("{linear_qkv_size:016X}")),
            ("{{LINEAR_KEY_SIZE}}", format!("{linear_key_size:016X}")),
            ("{{LINEAR_VALUE_SIZE}}", format!("{linear_value_size:016X}")),
            ("{{LINEAR_KEY_HEADS}}", format!("{linear_key_heads:016X}")),
            (
                "{{LINEAR_VALUE_HEADS}}",
                format!("{linear_value_heads:016X}"),
            ),
            (
                "{{LINEAR_KEY_HEAD_DIM}}",
                format!("{linear_key_head_dim:016X}"),
            ),
            (
                "{{LINEAR_VALUE_HEAD_DIM}}",
                format!("{linear_value_head_dim:016X}"),
            ),
            (
                "{{LINEAR_CONV_KERNEL}}",
                format!("{linear_conv_kernel:016X}"),
            ),
            (
                "{{LINEAR_CONV_STATE_SIZE}}",
                format!("{linear_conv_state_size:016X}"),
            ),
            (
                "{{LINEAR_RECURRENT_STATE_SIZE}}",
                format!("{linear_recurrent_state_size:016X}"),
            ),
            (
                "{{LINEAR_ATTENTION_SCALE_BITS}}",
                format!("{:08X}", linear_attention_scale.to_bits()),
            ),
            (
                "{{SHARED_INTERMEDIATE_SIZE}}",
                format!("{shared_intermediate_size:016X}"),
            ),
        ] {
            ensure!(
                source.contains(placeholder),
                "model template is missing {placeholder}"
            );
            source = source.replace(placeholder, &value);
        }
    }
    ensure!(
        !source.contains("{{"),
        "model template contains an unknown placeholder"
    );
    Ok(source)
}

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
                        !(character.is_ascii_alphanumeric() || character == '.' || character == '-')
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

fn retain_reachable_model_programs(source: &str) -> String {
    const DEFINITION: &str = "(def program ";
    let programs = qwen3_program_references(source);
    let mut reachable = BTreeSet::new();
    let mut pending = vec!["qwen3".to_owned()];
    while let Some(program) = pending.pop() {
        if !reachable.insert(program.clone()) {
            continue;
        }
        pending.extend(programs.get(&program).into_iter().flatten().cloned());
    }
    let definitions = source
        .match_indices(DEFINITION)
        .filter_map(|(offset, _)| {
            let name = source[offset + DEFINITION.len()..]
                .split_whitespace()
                .next()?;
            (name == "qwen3" || name.starts_with("qwen3.")).then_some((offset, name))
        })
        .collect::<Vec<_>>();
    let mut retained =
        source[..definitions.first().map_or(source.len(), |entry| entry.0)].to_owned();
    for (index, (start, name)) in definitions.iter().enumerate() {
        let end = definitions
            .get(index + 1)
            .map_or(source.len(), |entry| entry.0);
        if reachable.contains(*name) {
            retained.push_str(&source[*start..end]);
        }
    }
    retained
}

/// Generate all model-specific programs plus the explicit linear-use entrypoint.
pub fn generated_model_source() -> String {
    generated_model_source_for(&ModelConfig::qwen3_30b())
        .expect("the built-in Qwen3 configuration must be renderable")
}

/// Generate model source for the selected GGUF configuration. Qwen3.5 dense
/// models compose metadata-selected full-attention and Gated DeltaNet layers,
/// with either dense FFNs or routed plus shared-expert MoE FFNs.
pub fn generated_model_source_for(config: &ModelConfig) -> Result<String> {
    generated_model_source_for_precision(config, InferencePrecision::F32)
}

pub fn generated_model_source_for_precision(
    config: &ModelConfig,
    precision: InferencePrecision,
) -> Result<String> {
    config.validate()?;
    ensure!(
        precision == InferencePrecision::F32
            || matches!(
                config.architecture,
                ModelArchitecture::Qwen3Dense
                    | ModelArchitecture::Qwen35Dense
                    | ModelArchitecture::Qwen35Moe
            ),
        "BF16 inference currently supports Qwen3 dense and Qwen3.5/3.6 dense or MoE models"
    );
    for layer in 0..config.num_layers {
        layer_weight_names(config, layer)?;
    }
    let mut source = render_model_template(config, precision)?;
    source.push('\n');
    source.push_str(&generated_entrypoint(config, precision)?);
    Ok(retain_reachable_model_programs(&source))
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use catena_lang::{
        codegen::GpuDialect,
        runtime::{Runtime, Value},
    };

    #[test]
    fn entrypoint_uses_metadata_layer_and_input_counts() -> Result<()> {
        let mut config = ModelConfig::qwen3_30b();
        config.num_layers = 3;
        let source = generated_model_source_for(&config)?;
        let entrypoint = source
            .split_once(ENTRYPOINT_MARKER)
            .map(|(_, entrypoint)| entrypoint)
            .expect("entrypoint marker");
        let weight_count = weight_specs(&config)?.len();
        assert_eq!(
            entrypoint.matches("(cap.ref mem)").count(),
            1 + weight_count
        );
        assert_eq!(
            entrypoint.matches("qwen3.cached-decoder-layer").count(),
            config.num_layers
        );
        assert_eq!(entrypoint.matches("(cap.own mem)").count(), 4);
        assert_eq!(entrypoint.matches("(u64 val)").count(), 2);
        for layer in 0..=config.num_layers {
            assert!(entrypoint.contains(&format!("hidden-{layer:02}")));
        }
        assert!(!entrypoint.contains("layer-03-input-norm"));
        Ok(())
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
    fn exponential_helper_saturates_before_f32_exponent_bits_wrap() {
        assert!(HELPER_SOURCE.contains("const.u32.0x42B17218"));
        assert!(HELPER_SOURCE.contains("const.u32.0x7F800000"));
        assert!(HELPER_SOURCE.contains("[.magnitude maximum-log] f32.gt"));
    }

    #[test]
    fn model_constants_are_rendered_from_configuration() -> Result<()> {
        let mut config = ModelConfig::qwen3_30b();
        config.vocabulary_size = 42;
        config.hidden_size = 64;
        config.intermediate_size = 32;
        config.num_layers = 2;
        config.num_attention_heads = 4;
        config.num_key_value_heads = 2;
        config.head_dim = 16;
        config.rope_dimension_count = 16;
        config.num_experts = 8;
        config.experts_per_token = 2;
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
    fn dense_qwen3_uses_dense_layers_and_tensor_arity() -> Result<()> {
        let config = ModelConfig::qwen3_0_6b();
        let model = generated_model_source_for(&config)?;
        let entrypoint = model.split_once(ENTRYPOINT_MARKER).unwrap().1;
        assert_eq!(
            entrypoint
                .matches("qwen3.cached-dense-decoder-layer")
                .count(),
            config.num_layers
        );
        assert_eq!(
            entrypoint.matches("(cap.ref mem)").count(),
            1 + weight_specs(&config)?.len()
        );
        assert!(model.contains("(def program qwen3.dense-activate :"));
        assert!(!model.contains("(def program qwen3.cached-decoder-layer :"));
        assert!(!model.contains("(def program qwen3.router-select-experts :"));
        assert!(!model.contains("(def program qwen3.moe-activate :"));
        Ok(())
    }

    #[test]
    fn dense_qwen3_bf16_uses_alternate_graph() -> Result<()> {
        let config = ModelConfig::qwen3_0_6b();
        let model = generated_model_source_for_precision(&config, InferencePrecision::Bf16)?;
        let entrypoint = model.split_once(ENTRYPOINT_MARKER).unwrap().1;
        assert_eq!(
            entrypoint
                .matches("qwen3.cached-dense-decoder-layer")
                .count(),
            config.num_layers
        );
        assert!(entrypoint.contains("mem.cast.bf16 [logit-count logit-buffer.]"));
        assert!(entrypoint.contains("materializec.borrow-argmax-bf16"));
        assert!(model.contains("qwen3.free-bf16-before"));
        assert_eq!(model.matches("materializec.bf16-gemv").count(), 3);
        assert!(!model.contains("} materializec.reduce-f32 [logits-buffer.]"));
        assert!(!model.contains("qwen3.expert-linear-pair"));
        assert!(!model.contains("qwen3.moe-down-residual"));
        let unexpected_f32_casts = model
            .lines()
            .filter(|line| line.contains("mem.cast.f32"))
            .filter(|line| {
                !["norm-weight", "inverse", "rope"]
                    .iter()
                    .any(|allowed| line.contains(allowed))
            })
            .collect::<Vec<_>>();
        assert!(
            unexpected_f32_casts.is_empty(),
            "persistent BF16 graph buffers cast as F32: {unexpected_f32_casts:?}"
        );
        Ok(())
    }

    #[test]
    fn bf16_graph_rejects_qwen3_moe() {
        let error = generated_model_source_for_precision(
            &ModelConfig::qwen3_30b(),
            InferencePrecision::Bf16,
        )
        .unwrap_err();
        assert!(error.to_string().contains("Qwen3.5/3.6 dense or MoE"));
    }

    #[test]
    fn dense_qwen35_bf16_uses_native_hybrid_graph() -> Result<()> {
        let config = ModelConfig::qwen35_0_8b();
        let model = generated_model_source_for_precision(&config, InferencePrecision::Bf16)?;
        let entrypoint = model.split_once(ENTRYPOINT_MARKER).unwrap().1;
        assert_eq!(
            entrypoint
                .matches("qwen3.qwen35-full-decoder-layer")
                .count(),
            6
        );
        assert_eq!(
            entrypoint
                .matches("qwen3.qwen35-linear-decoder-layer")
                .count(),
            18
        );
        assert!(model.contains("materializec.borrow-gated-delta-net-bf16"));
        assert!(!model.contains("materializec.borrow-gated-delta-net-f32"));
        assert!(!model.contains("qwen3.expert-linear-pair"));
        assert!(!model.contains("qwen3.moe-down-residual"));
        assert!(!model.contains("(def program qwen3.linear-norm :"));
        assert!(model.contains("materializec.bf16-gemv"));
        assert!(model.contains("mem.cast.bf16 [state-cap state-buffer.]"));
        assert!(model.contains("mem.cast.f32 [dt-count dt-buffer.]"));
        assert!(model.contains("mem.cast.f32 [a-count a-buffer.]"));
        assert!(model.contains("mem.cast.f32 [norm-count norm-buffer.]"));
        Ok(())
    }

    #[test]
    fn qwen35_moe_bf16_uses_native_hybrid_and_expert_graphs() -> Result<()> {
        let config = ModelConfig::qwen36_35b_a3b();
        let model = generated_model_source_for_precision(&config, InferencePrecision::Bf16)?;
        let entrypoint = model.split_once(ENTRYPOINT_MARKER).unwrap().1;
        assert_eq!(
            entrypoint
                .matches("qwen3.qwen35-full-moe-decoder-layer")
                .count(),
            10
        );
        assert_eq!(
            entrypoint
                .matches("qwen3.qwen35-linear-moe-decoder-layer")
                .count(),
            30
        );
        assert!(model.contains("materializec.borrow-gated-delta-net-bf16"));
        assert!(model.contains("materializec.borrow-routed-native-bf16-gemv-pair"));
        assert!(model.contains("materializec.borrow-routed-native-bf16-gemv-residual"));
        assert!(model.contains("qwen3.qwen35-moe-down-residual"));
        assert!(model.contains("qwen3.qwen35-router-linear"));
        assert!(model.contains("[.linear-weight] mem.cast.f32"));
        assert!(!model.contains("qwen3.qwen35-router-f32"));
        assert!(!model.contains("(def program qwen3.linear-norm :"));
        assert!(!model.contains("(def program qwen3.qwen35-moe-down-term-at :"));
        assert!(!model.contains("materializec.borrow-routed-bf16-gemv-pair"));
        Ok(())
    }

    #[test]
    fn dense_qwen35_uses_metadata_selected_hybrid_layers() -> Result<()> {
        let config = ModelConfig::qwen35_0_8b();
        let model = generated_model_source_for(&config)?;
        let entrypoint = model.split_once(ENTRYPOINT_MARKER).unwrap().1;
        assert_eq!(
            entrypoint
                .matches("qwen3.qwen35-full-decoder-layer")
                .count(),
            6
        );
        assert_eq!(
            entrypoint
                .matches("qwen3.qwen35-linear-decoder-layer")
                .count(),
            18
        );
        assert_eq!(
            entrypoint.matches("(cap.ref mem)").count(),
            1 + weight_specs(&config)?.len()
        );
        assert!(model.contains("(def program qwen3.qwen35-gated-delta-scan :"));
        assert!(!model.contains("(def program qwen3.qwen35-delta :"));
        assert!(HELPER_SOURCE.contains("(def program qwen3.softplus :"));
        Ok(())
    }

    #[test]
    fn qwen36_moe_uses_hybrid_layers_and_shared_experts() -> Result<()> {
        let config = ModelConfig::qwen36_35b_a3b();
        let model = generated_model_source_for(&config)?;
        let entrypoint = model.split_once(ENTRYPOINT_MARKER).unwrap().1;
        assert_eq!(
            entrypoint
                .matches("qwen3.qwen35-full-moe-decoder-layer")
                .count(),
            10
        );
        assert_eq!(
            entrypoint
                .matches("qwen3.qwen35-linear-moe-decoder-layer")
                .count(),
            30
        );
        assert_eq!(
            entrypoint.matches("(cap.ref mem)").count(),
            1 + weight_specs(&config)?.len()
        );
        assert!(model.contains("(def program qwen3.qwen35-moe-ffn :"));
        assert!(model.contains("(def program qwen3.qwen35-shared-linear-pair :"));
        Ok(())
    }

    #[test]
    fn every_qwen3_program_is_used_by_a_supported_model_entrypoint() {
        let mut all_programs = BTreeSet::new();
        let mut all_reachable = BTreeSet::new();
        for config in [
            ModelConfig::qwen3_30b(),
            ModelConfig::qwen3_0_6b(),
            ModelConfig::qwen35_0_8b(),
            ModelConfig::qwen36_35b_a3b(),
        ] {
            let model = generated_model_source_for(&config).unwrap();
            let source = format!("{HELPER_SOURCE}\n{model}");
            let programs = qwen3_program_references(&source);
            all_programs.extend(programs.keys().cloned());
            let mut reachable = BTreeSet::new();
            let mut pending = vec!["qwen3".to_owned()];
            while let Some(program) = pending.pop() {
                if !reachable.insert(program.clone()) {
                    continue;
                }
                pending.extend(programs.get(&program).into_iter().flatten().cloned());
            }
            all_reachable.extend(reachable);
        }
        let unreachable = all_programs
            .difference(&all_reachable)
            .cloned()
            .collect::<Vec<_>>();
        assert!(
            unreachable.is_empty(),
            "programs unreachable from every supported qwen3 entrypoint: {unreachable:?}"
        );
    }

    #[test]
    fn gpu_program_compiles_when_requested() -> Result<()> {
        if std::env::var_os("CATENA_QWEN3_GPU_TESTS").is_none() {
            return Ok(());
        }
        let model = generated_model_source();
        let runtime = Runtime::from_sources(
            runtime_sources(&model, InferencePrecision::F32),
            GpuDialect::Hip,
        )?;
        let config = ModelConfig::qwen3_30b();
        let sequence = config.max_position_embeddings;
        let [output] =
            runtime.exec("qwen3.rope-table", [0_u64.into(), (sequence as u64).into()])?;
        let Value::MemOwn(output) = output else {
            anyhow::bail!("qwen3.rope-table returned non-owned memory: {output:?}");
        };
        let actual = output.try_to_f32_vec()?;
        let half_dim = config.rope_dimension_count / 2;
        let plane_len = sequence * half_dim;
        assert_eq!(actual.len(), 2 * plane_len);
        let mut max_sine_error = 0.0_f32;
        let mut max_cosine_error = 0.0_f32;
        for position in 0..sequence {
            for pair in 0..half_dim {
                let exponent = (2 * pair) as f32 / config.rope_dimension_count as f32;
                let frequency = config.rope_theta.powf(-exponent);
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

    #[test]
    fn dense_gpu_program_compiles_when_requested() -> Result<()> {
        if std::env::var_os("CATENA_QWEN3_GPU_TESTS").is_none() {
            return Ok(());
        }
        for (index, config) in [ModelConfig::qwen3_0_6b(), ModelConfig::qwen3_1_7b()]
            .into_iter()
            .enumerate()
        {
            let model = generated_model_source_for(&config)?;
            let runtime = Runtime::from_sources(
                runtime_sources(&model, InferencePrecision::F32),
                GpuDialect::Hip,
            )?;
            if index == 0 {
                let [Value::F32(positive)] = runtime.exec("qwen3.exp", [100.0_f32.into()])? else {
                    anyhow::bail!("qwen3.exp returned an unexpected value")
                };
                let [Value::F32(negative)] = runtime.exec("qwen3.exp", [(-100.0_f32).into()])?
                else {
                    anyhow::bail!("qwen3.exp returned an unexpected value")
                };
                assert!(positive.is_infinite() && positive.is_sign_positive());
                assert_eq!(negative, 0.0);
            }
        }
        Ok(())
    }

    #[test]
    fn bf16_gpu_programs_compile_when_requested() -> Result<()> {
        if std::env::var_os("CATENA_QWEN3_GPU_TESTS").is_none() {
            return Ok(());
        }
        for config in [
            ModelConfig::qwen3_0_6b(),
            ModelConfig::qwen35_0_8b(),
            ModelConfig::qwen36_35b_a3b(),
        ] {
            let model = generated_model_source_for_precision(&config, InferencePrecision::Bf16)?;
            Runtime::from_sources(
                runtime_sources(&model, InferencePrecision::Bf16),
                GpuDialect::Hip,
            )?;
        }
        Ok(())
    }
}
