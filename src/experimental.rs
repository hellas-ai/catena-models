//! BF16 Qwen3-family text programs for Catena's experimental compiler.
use crate::{
    catena::{ExperimentalKernelKind, InferenceOptions, InferencePrecision, MatrixCoreMode},
    spec::{ModelArchitecture, ModelConfig, weight_specs},
};
use anyhow::{Result, ensure};
use catena_lang::{
    codegen::{CodegenKind, GpuDialect},
    runtime::RuntimeModule,
};
use std::fmt::Write;

#[path = "experimental_fused.rs"]
pub(crate) mod fused;
#[path = "experimental_qwen35.rs"]
mod qwen35;

const SOURCES: &[&str] = &[
    include_str!("catena/experimental/memory.hex"),
    include_str!("catena/experimental/numeric.hex"),
    include_str!("catena/experimental/resources.hex"),
    include_str!("catena/experimental/embedding.hex"),
    include_str!("catena/experimental/normalization.hex"),
    include_str!("catena/experimental/projection.hex"),
    include_str!("catena/experimental/rope.hex"),
    include_str!("catena/experimental/attention.hex"),
    include_str!("catena/experimental/argmax.hex"),
    include_str!("catena/experimental/decoder.hex"),
];
const OWN: &str = "((cap.own mem) val)";
const REF: &str = "((cap.ref mem) val)";
const U64: &str = "(u64 val)";
const STATE: &str = "({((cap.own mem) val) ((cap.own mem) val)} *)";

pub(crate) fn validate(
    config: &ModelConfig,
    options: InferenceOptions,
    dialect: Option<GpuDialect>,
) -> Result<()> {
    ensure!(
        matches!(
            config.architecture,
            ModelArchitecture::Qwen3Dense
                | ModelArchitecture::Qwen35Dense
                | ModelArchitecture::Qwen35Moe
        ),
        "experimental codegen supports dense Qwen3 and dense/MoE Qwen3.5 text models only"
    );
    ensure!(
        options.precision == InferencePrecision::Bf16,
        "experimental Qwen3-family inference requires BF16 (--bf16)"
    );
    ensure!(
        options.matrix_cores == MatrixCoreMode::Disabled && !options.fused_attention,
        "experimental Qwen3 does not support the legacy WMMA, BLAS or fused-attention options"
    );
    ensure!(
        dialect.is_none_or(|d| d == GpuDialect::Hip),
        "experimental BF16 Qwen3 requires HIP gfx11; CUDA is not supported"
    );
    ensure!(
        [
            config.hidden_size,
            config.intermediate_size,
            config.head_dim
        ]
        .iter()
        .all(|n| n.is_multiple_of(2)),
        "experimental packed BF16 projections require even hidden, intermediate and head dimensions"
    );
    if matches!(
        config.architecture,
        ModelArchitecture::Qwen35Dense | ModelArchitecture::Qwen35Moe
    ) {
        config.validate()?;
        ensure!(
            config.hidden_size.is_power_of_two() && config.head_dim.is_power_of_two(),
            "experimental Qwen3.5 requires power-of-two hidden and attention head widths"
        );
        ensure!(
            options.experimental_kernels == ExperimentalKernelKind::Fused,
            "experimental Qwen3.5 requires --kernels fused"
        );
        ensure!(
            config.linear_conv_kernel == Some(4),
            "experimental Qwen3.5 requires four convolution taps"
        );
        ensure!(
            config.linear_inner_size.unwrap().is_multiple_of(8),
            "experimental Qwen3.5 requires the linear value size to be a multiple of eight"
        );
        ensure!(
            config
                .linear_time_step_rank
                .unwrap()
                .is_multiple_of(config.linear_group_count.unwrap()),
            "experimental Qwen3.5 requires a whole number of value heads per key head"
        );
    }
    if config.architecture == ModelArchitecture::Qwen35Moe {
        ensure!(
            config.num_experts <= 256 && config.experts_per_token <= 8,
            "experimental MoE supports at most 256 experts and eight active experts"
        );
        let shared = config
            .shared_expert_intermediate_size
            .ok_or_else(|| anyhow::anyhow!("experimental Qwen3.5 MoE requires a shared expert"))?;
        ensure!(
            shared > 0 && shared.is_multiple_of(2),
            "experimental shared expert width must be positive and even"
        );
        ensure!(
            config
                .experts_per_token
                .checked_mul(config.intermediate_size)
                .is_some_and(|width| shared <= width),
            "experimental shared expert width exceeds routed activation width"
        );
    }
    ensure!(
        config.rope_dimension_count.is_multiple_of(2),
        "experimental Qwen3 rotary dimension must be even"
    );
    Ok(())
}

/// Persistent experimental scratch is returned by the artifact after every call.
pub(crate) fn workspace_bytes(config: &ModelConfig, options: InferenceOptions) -> Result<usize> {
    if options.codegen != CodegenKind::Experimental
        || options.experimental_kernels == ExperimentalKernelKind::Reference
    {
        return Ok(0);
    }
    if config.architecture == ModelArchitecture::Qwen3Dense {
        return Ok(4096);
    }
    let recurrent = config
        .num_layers
        .checked_mul(config.linear_time_step_rank.unwrap())
        .and_then(|n| n.checked_mul(config.linear_state_size.unwrap()))
        .and_then(|n| n.checked_mul(config.linear_value_head_size().unwrap()))
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| anyhow::anyhow!("experimental recurrent workspace overflow"))?;
    if config.architecture == ModelArchitecture::Qwen35Moe {
        Ok(qwen35::moe_workspace(config, recurrent)?.bytes)
    } else {
        Ok(recurrent)
    }
}

fn c(n: usize) -> String {
    format!("(const.u64.0x{n:016X})")
}
fn call(s: &mut String, args: &[&str], op: &str, outputs: &str) {
    write!(s, "  ({{").unwrap();
    for arg in args {
        if arg.starts_with('(') {
            write!(s, "{arg} ").unwrap();
        } else {
            write!(s, "[.{arg}] ").unwrap();
        }
    }
    writeln!(s, "}} {op} [{outputs}.])").unwrap();
}
fn header(s: &mut String, name: &str, inputs: &[(&str, &str)], output: &str) {
    writeln!(
        s,
        "(def program {name} : {{{}}} -> {output} =",
        inputs
            .iter()
            .map(|(_, ty)| *ty)
            .collect::<Vec<_>>()
            .join(" ")
    )
    .unwrap();
    writeln!(
        s,
        " ([{}. ]",
        inputs
            .iter()
            .map(|(name, _)| *name)
            .collect::<Vec<_>>()
            .join(" ")
    )
    .unwrap();
}

fn nested_product(types: &[String]) -> String {
    if types.len() == 1 {
        return types[0].clone();
    }
    format!("({{{} {}}} *)", types[0], nested_product(&types[1..]))
}
fn pack(s: &mut String, names: &[String], output: &str) {
    if names.len() == 1 {
        writeln!(s, "  ([.{}] [{}.] )", names[0], output).unwrap();
        return;
    }
    let mut tail = names.last().unwrap().clone();
    for i in (0..names.len() - 1).rev() {
        let next = if i == 0 {
            output.to_owned()
        } else {
            format!("{output}-tail{i}")
        };
        call(s, &[&names[i], &tail], "*.intro", &next);
        tail = next;
    }
}
fn unpack(s: &mut String, input: &str, names: &[String]) {
    if names.len() == 1 {
        writeln!(s, "  ([.{input}] [{}.] )", names[0]).unwrap();
        return;
    }
    let mut tail = input.to_owned();
    for i in 0..names.len() - 1 {
        let next = if i == names.len() - 2 {
            names[i + 1].clone()
        } else {
            format!("{input}-tail{i}")
        };
        writeln!(s, "  ([.{tail}] *.elim [{} {next}.])", names[i]).unwrap();
        tail = next;
    }
}
fn emit_layer_selector(s: &mut String, layers: usize, group: &str, weights: &str) {
    let pair = format!("({{{group} {group}}} *)");
    for (suffix, proof) in [
        ("yes", "((master.predicate prop.IsTrue) |-)"),
        (
            "no",
            "(({(master.predicate prop.IsTrue)} stdlib.prelude.prop.Not) |-)",
        ),
    ] {
        let arg = format!("({{{pair} ({{{proof} 1}} *)}} *)");
        header(
            s,
            &format!("stdlib.qwen3.select-layer.{suffix}"),
            &[("pair", &arg)],
            group,
        );
        s.push_str("  ([.pair] *.elim [values proofs.])\n  ([.values] *.elim [yes no.])\n  ([.proofs] *.elim [proof unit.])\n  ([.proof] unsafe.discard unit.elim)\n  ([.unit] unit.elim)\n");
        writeln!(s, "  [.{suffix}]\n ))").unwrap();
    }
    header(
        s,
        "stdlib.qwen3.select-layer",
        &[("condition", "(bool val)"), ("yes", group), ("no", group)],
        group,
    );
    s.push_str("  ({[.condition] (unit.intro meta.master.predicate)} smolcat.value.named_at :.forget [test.])\n  ({[.yes] [.no]} *.intro [values.])\n  ({[.test] [.values] (name.stdlib.qwen3.select-layer.yes lift) (name.stdlib.qwen3.select-layer.no lift) unit.intro} core.if)\n ))\n");
    header(
        s,
        "stdlib.qwen3.layer-weights",
        &[("weights", weights), ("layer", "(u32 val)")],
        group,
    );
    let groups = (0..layers).map(|i| format!("group{i}")).collect::<Vec<_>>();
    unpack(s, "weights", &groups);
    let mut previous = "group0".to_owned();
    for layer in 1..layers {
        let index = format!("(const.u32.0x{layer:08X})");
        call(
            s,
            &["layer", &index],
            "stdlib.qwen3.memory.u32.equal",
            &format!("is-layer{layer}"),
        );
        call(
            s,
            &[&format!("is-layer{layer}"), &groups[layer], &previous],
            "stdlib.qwen3.select-layer",
            &format!("selected{layer}"),
        );
        previous = format!("selected{layer}");
    }
    writeln!(s, "  [.{previous}]\n ))").unwrap();
}

pub(crate) fn source(config: &ModelConfig, options: InferenceOptions) -> Result<String> {
    config.validate()?;
    validate(config, options, None)?;
    if matches!(
        config.architecture,
        ModelArchitecture::Qwen35Dense | ModelArchitecture::Qwen35Moe
    ) {
        ensure!(
            options.experimental_kernels == ExperimentalKernelKind::Fused,
            "experimental Qwen3.5 requires --kernels fused"
        );
        return qwen35::source(config, options);
    }
    if options.experimental_kernels == ExperimentalKernelKind::Fused {
        return fused::source(config, options);
    }
    let mut s = SOURCES.join("\n");
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
        ("{{HEAD_HALF_DIM}}", format!("{:016X}", config.head_dim / 2)),
        (
            "{{ROPE_DIM}}",
            format!("{:016X}", config.rope_dimension_count),
        ),
        (
            "{{ROPE_HALF_DIM}}",
            format!("{:016X}", config.rope_dimension_count / 2),
        ),
        (
            "{{KV_HEADS}}",
            format!("{:016X}", config.num_key_value_heads),
        ),
        (
            "{{QUERY_HEADS}}",
            format!("{:016X}", config.num_attention_heads),
        ),
        (
            "{{QUERY_HEADS_PER_KV_HEAD}}",
            format!(
                "{:016X}",
                config.num_attention_heads / config.num_key_value_heads
            ),
        ),
        (
            "{{HIDDEN_F32}}",
            format!("{:08X}", (config.hidden_size as f32).to_bits()),
        ),
        (
            "{{HEAD_F32}}",
            format!("{:08X}", (config.head_dim as f32).to_bits()),
        ),
        (
            "{{RMS_EPSILON_BITS}}",
            format!("{:08X}", config.rms_norm_eps.to_bits()),
        ),
        (
            "{{ATTENTION_SCALE_BITS}}",
            format!("{:08X}", (config.head_dim as f32).sqrt().recip().to_bits()),
        ),
    ];
    for (key, value) in values {
        s = s.replace(key, &value);
    }
    s.push_str(
        &crate::catena::rope_frequency_program(config)
            .replace("u64.ne bool.not", "stdlib.qwen3.memory.u64.equal")
            .replace("f32.select", "stdlib.qwen3.memory.select-f32"),
    );
    s.push('\n');
    let specs = weight_specs(config)?;
    let names = (0..specs.len())
        .map(|i| format!("w{i}"))
        .collect::<Vec<_>>();
    let group_type = nested_product(&vec![REF.to_owned(); 11]);
    let weights_type = nested_product(&vec![group_type.clone(); config.num_layers]);
    emit_layer_selector(&mut s, config.num_layers, &group_type, &weights_type);
    let folded_state = "({({master.index32 u32} :) ({({((cap.own mem) val) ((cap.own mem) val)} *) ({(({master.index32 master.end32} prop.Less) |-) (prop.Truth |-)} *)} *)} *)";
    header(
        &mut s,
        "stdlib.qwen3.decoder-step.fold",
        &[
            ("weights", &weights_type),
            ("rope", OWN),
            ("capacity", U64),
            ("position", U64),
            ("pair", folded_state),
        ],
        "({({((cap.own mem) val) ((cap.own mem) val)} *) (prop.Truth |-)} *)",
    );
    s.push_str("  ([.pair] *.elim [index rest.])\n  ([.rest] *.elim [state proofs.])\n  ([.proofs] *.elim [less truth.])\n  ([.less] unsafe.discard unit.elim)\n  ([.index] :.forget [layer-u32.])\n  ([.layer-u32] stdlib.qwen3.memory.u32.to-u64 [layer.])\n  ([.state] *.elim [cache hidden.])\n");
    call(
        &mut s,
        &["weights", "layer-u32"],
        "stdlib.qwen3.layer-weights",
        "selected",
    );
    let layer_names = (0..11).map(|i| format!("layer-w{i}")).collect::<Vec<_>>();
    unpack(&mut s, "selected", &layer_names);
    let mut args = vec!["hidden", "cache", "rope", "capacity", "position", "layer"];
    args.extend(layer_names.iter().map(String::as_str));
    call(&mut s, &args, "qwen3.decoder-layer", "decoded");
    s.push_str("  ({[.decoded] [.truth]} *.intro)\n ))\n");

    let captures = vec![
        ("tokens", REF),
        ("embedding", REF),
        ("weights", weights_type.as_str()),
        ("capacity", U64),
        ("start", U64),
    ];
    let mut token_inputs = captures.clone();
    token_inputs.push((
        "pair",
        "({(u64 val) ({((cap.own mem) val) ((cap.own mem) val)} *)} *)",
    ));
    header(&mut s, "stdlib.qwen3.token-step", &token_inputs, STATE);
    s.push_str("  ([.pair] *.elim [index state.])\n  ([.state] *.elim [cache previous.])\n");
    call(
        &mut s,
        &["start", "index"],
        "stdlib.qwen3.memory.u64.add",
        "position",
    );
    call(
        &mut s,
        &["tokens", "embedding", "index"],
        "qwen3.embedding-one",
        "embedded",
    );
    call(
        &mut s,
        &["previous", "embedded"],
        "qwen3.free-bf16-before",
        "hidden",
    );
    call(&mut s, &["position"], "qwen3.rope-one", "rope");
    s.push_str("  ({[.cache] [.hidden]} *.intro [initial-state.])\n");
    call(
        &mut s,
        &["weights", "rope", "capacity", "position"],
        "partial.stdlib.qwen3.decoder-step.fold.4",
        "callback",
    );
    writeln!(
        s,
        "  ({{(const.u32.0x{:08X}) (unit.intro meta.master.end32)}} smolcat.value.named_at [end.])",
        config.num_layers
    )
    .unwrap();
    s.push_str("  ({[.end] [.initial-state] [.callback] core.truth} core.fold.bounded [final-state proof.])\n  ([.proof] unsafe.discard unit.elim)\n  ([.final-state] *.elim [final-cache final-hidden.])\n");
    call(
        &mut s,
        &["rope", "final-hidden"],
        "qwen3.free-f32-before",
        "hidden-result",
    );
    s.push_str("  ({[.final-cache] [.hidden-result]} *.intro)\n ))\n");
    let mut adapter_inputs = captures.clone();
    adapter_inputs.push(("pair", folded_state));
    header(
        &mut s,
        "stdlib.qwen3.token-step.fold",
        &adapter_inputs,
        "({({((cap.own mem) val) ((cap.own mem) val)} *) (prop.Truth |-)} *)",
    );
    s.push_str("  ([.pair] *.elim [index rest.])\n  ([.rest] *.elim [state proofs.])\n  ([.proofs] *.elim [less truth.])\n  ([.less] unsafe.discard unit.elim)\n  ([.index] :.forget stdlib.qwen3.memory.u32.to-u64 [i.])\n  ({[.i] [.state]} *.intro [arguments.])\n");
    call(
        &mut s,
        &[
            "tokens",
            "embedding",
            "weights",
            "capacity",
            "start",
            "arguments",
        ],
        "stdlib.qwen3.token-step",
        "state-next",
    );
    s.push_str("  ({[.state-next] [.truth]} *.intro)\n ))\n");
    let mut inputs = vec![
        ("tokens", REF),
        ("cache", OWN),
        ("capacity", U64),
        ("start", U64),
    ];
    inputs.extend(names.iter().map(|n| (n.as_str(), REF)));
    header(
        &mut s,
        "qwen3",
        &inputs,
        "{((cap.own mem) val) ((cap.own mem) val) ((cap.own mem) val)}",
    );
    for layer in 0..config.num_layers {
        pack(
            &mut s,
            &names[1 + layer * 11..1 + (layer + 1) * 11],
            &format!("group{layer}"),
        );
    }
    let groups = (0..config.num_layers)
        .map(|i| format!("group{i}"))
        .collect::<Vec<_>>();
    pack(&mut s, &groups, "weights");
    s.push_str("  ([.tokens] stdlib.qwen3.memory.ref-count-u64 [sequence.])\n");
    call(
        &mut s,
        &[&c(config.hidden_size)],
        "stdlib.qwen3.memory.alloc-bf16",
        "initial",
    );
    s.push_str("  ({[.cache] [.initial]} *.intro [state.])\n");
    call(
        &mut s,
        &["tokens", "w0", "weights", "capacity", "start"],
        "partial.stdlib.qwen3.token-step.fold.5",
        "callback",
    );
    s.push_str("  ({([.sequence] stdlib.qwen3.memory.fold-count-u32) (unit.intro meta.master.end32)} smolcat.value.named_at [end.])\n  ({[.end] [.state] [.callback] core.truth} core.fold.bounded [final-state proof.])\n  ([.proof] unsafe.discard unit.elim)\n  ([.final-state] *.elim [cache-final hidden-final.])\n");
    let norm = &names[names.len() - 2];
    let head = &names[names.len() - 1];
    let columns = c(config.hidden_size);
    let one = c(1);
    let columns_float = format!(
        "(const.u32.0x{:08X} u32.bitcast-f32)",
        (config.hidden_size as f32).to_bits()
    );
    call(
        &mut s,
        &["hidden-final", norm, &columns, &columns_float, &one],
        "qwen3.normalize",
        "normalized",
    );
    call(
        &mut s,
        &["hidden-final", "normalized"],
        "qwen3.free-bf16-before",
        "normalized-final",
    );
    call(
        &mut s,
        &[
            "normalized-final",
            head,
            &columns,
            &c(config.vocabulary_size),
        ],
        "qwen3.project",
        "logits",
    );
    call(
        &mut s,
        &["normalized-final", "logits"],
        "qwen3.free-bf16-before",
        "logits-final",
    );
    call(&mut s, &["logits-final"], "qwen3.argmax", "next-token");
    s.push_str("  {[.cache-final] [.logits-final] [.next-token]}\n ))\n");
    ensure!(
        !s.split("{{").skip(1).any(|part| part
            .chars()
            .next()
            .is_some_and(|ch| ch.is_ascii_uppercase())),
        "unresolved experimental Qwen3 template dimensions"
    );
    Ok(s)
}

/// Keep primitive declarations and exported programs, following callback dependencies.
fn reachable_sources(raw: metacat::theory::RawTheorySet) -> Result<metacat::theory::RawTheorySet> {
    use std::collections::BTreeSet;
    let mut raw = raw.with_extensions()?;
    let program = raw
        .theories
        .get_mut(&"program".parse().unwrap())
        .ok_or_else(|| anyhow::anyhow!("missing experimental program theory"))?;
    let available = program
        .arrows
        .keys()
        .map(|n| n.as_str().to_owned())
        .collect::<BTreeSet<_>>();
    let mut pending = program
        .arrows
        .iter()
        .filter_map(|(name, arrow)| {
            (!name.as_str().starts_with("meta.")
                && (arrow.definition.is_none() || !name.as_str().starts_with("stdlib.")))
            .then(|| name.as_str().to_owned())
        })
        .collect::<Vec<_>>();
    let mut used = BTreeSet::new();
    while let Some(name) = pending.pop() {
        if !used.insert(name.clone()) {
            continue;
        }
        let Some(arrow) = program.arrows.get(&name.parse().unwrap()) else {
            continue;
        };
        for expr in [&arrow.type_maps.0, &arrow.type_maps.1]
            .into_iter()
            .chain(arrow.definition.iter())
        {
            for token in expr
                .to_string()
                .split(|c: char| c.is_whitespace() || "(){}[]^".contains(c))
            {
                let token = token.strip_prefix("name.").unwrap_or(token);
                let token = token
                    .strip_prefix("partial.")
                    .and_then(|s| s.rsplit_once('.'))
                    .filter(|(_, n)| n.parse::<usize>().is_ok())
                    .map_or(token, |(s, _)| s);
                if available.contains(token) {
                    pending.push(token.to_owned());
                }
            }
        }
    }
    program.arrows.retain(|n, _| used.contains(n.as_str()));
    Ok(raw)
}

pub(crate) fn compile<'a>(
    sources: impl IntoIterator<Item = &'a str>,
    dialect: GpuDialect,
) -> Result<RuntimeModule> {
    let raw = metacat::theory::RawTheorySet::from_texts(sources)?;
    let raw = reachable_sources(raw)?;
    let module = catena_lang::compile::compile(
        &mut catena_lang::report::CompileReport::new(raw),
        CodegenKind::Experimental,
        dialect,
    )?;
    if let Some(path) = std::env::var_os("CATENA_QWEN3_EXPERIMENTAL_CPP") {
        std::fs::write(path, &module.source)?;
    }
    Ok(module)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn options() -> InferenceOptions {
        InferenceOptions::new(InferencePrecision::Bf16)
            .with_codegen(CodegenKind::Experimental)
            .with_experimental_kernels(ExperimentalKernelKind::Reference)
    }
    pub(super) fn tiny_config() -> ModelConfig {
        let mut c = ModelConfig::qwen3_0_6b();
        c.hidden_size = 32;
        c.intermediate_size = 64;
        c.num_layers = 2;
        c.num_attention_heads = 2;
        c.num_key_value_heads = 1;
        c.head_dim = 16;
        c.rope_dimension_count = 16;
        c.vocabulary_size = 33;
        c.max_position_embeddings = 128;
        c
    }
    #[test]
    fn dense_bf16_compiles_and_preserves_runtime_abi() -> Result<()> {
        for config in [
            tiny_config(),
            ModelConfig::qwen3_0_6b(),
            ModelConfig::qwen3_1_7b(),
        ] {
            let model = source(&config, options())?;
            let module = compile(
                crate::catena::runtime_sources_for_options(&model, options()),
                GpuDialect::Hip,
            )?;
            let entry = module
                .functions
                .iter()
                .find(|f| f.source_name == "qwen3")
                .expect("Qwen3 entrypoint");
            use catena_lang::runtime::ValueKind;
            assert_eq!(entry.inputs.len(), 4 + weight_specs(&config)?.len());
            assert_eq!(
                entry.inputs[..4],
                [
                    ValueKind::MemRef,
                    ValueKind::MemOwn,
                    ValueKind::U64,
                    ValueKind::U64
                ]
            );
            assert!(entry.inputs[4..].iter().all(|k| *k == ValueKind::MemRef));
            assert_eq!(entry.outputs, vec![ValueKind::MemOwn; 3]);
            assert!(module.source.contains("__builtin_amdgcn_fdot2_f32_bf16"));
            let symbol = &module
                .functions
                .iter()
                .find(|f| f.source_name == "qwen3.normalize")
                .unwrap()
                .symbol;
            let body = module
                .source
                .split(&format!("void {symbol}("))
                .nth(1)
                .unwrap()
                .split("extern \"C\" __host__")
                .next()
                .unwrap();
            assert!(
                body.rfind("<<<").unwrap() < body.find("hipFree(").unwrap(),
                "inverse RMS buffer must remain live until normalization completes"
            );
        }
        Ok(())
    }
    #[test]
    fn qwen35_moe_validates_routing_and_workspace() -> Result<()> {
        let options =
            InferenceOptions::new(InferencePrecision::Bf16).with_codegen(CodegenKind::Experimental);
        let config = ModelConfig::qwen36_35b_a3b();
        validate(&config, options, Some(GpuDialect::Hip))?;
        let bytes = workspace_bytes(&config, options)?;
        assert_eq!(bytes, 83_896_424);
        for (experts, selected, shared) in [
            (257, 8, Some(512)),
            (256, 9, Some(512)),
            (256, 8, None),
            (256, 8, Some(513)),
            (256, 8, Some(8192)),
        ] {
            let mut bad = config.clone();
            bad.num_experts = experts;
            bad.experts_per_token = selected;
            bad.shared_expert_intermediate_size = shared;
            assert!(validate(&bad, options, None).is_err());
        }
        assert!(
            validate(
                &config,
                options.with_experimental_kernels(ExperimentalKernelKind::Reference),
                None
            )
            .is_err()
        );
        let mut huge = config;
        huge.num_layers = usize::MAX;
        assert!(workspace_bytes(&huge, options).is_err());
        Ok(())
    }
    #[test]
    fn unsupported_options_are_rejected_before_compilation() {
        let c = tiny_config();
        assert!(source(&c, options().with_matrix_cores(MatrixCoreMode::Wmma)).is_err());
        assert!(source(&c, options().with_fused_attention(true)).is_err());
        assert!(
            source(
                &c,
                InferenceOptions::new(InferencePrecision::F32)
                    .with_codegen(CodegenKind::Experimental)
            )
            .is_err()
        );
        assert!(source(&ModelConfig::qwen35_0_8b(), options()).is_err());
        assert!(source(&ModelConfig::qwen3_30b(), options()).is_err());
        assert!(validate(&c, options(), Some(GpuDialect::Cuda)).is_err());
    }
}

#[cfg(test)]
#[path = "experimental_tests.rs"]
mod device_tests;
