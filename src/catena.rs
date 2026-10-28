//! Model source generation and the single compiler integration boundary.
use crate::spec::ModelConfig;
use anyhow::Result;
use std::fmt::Write;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub enum KernelKind {
    Reference,
    #[default]
    Fused,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct InferenceOptions {
    pub kernels: KernelKind,
    pub wmma_prefill: bool,
}
impl InferenceOptions {
    pub const fn with_kernels(mut self, kernels: KernelKind) -> Self {
        self.kernels = kernels;
        self
    }
    pub const fn with_wmma_prefill(mut self, enabled: bool) -> Self {
        self.wmma_prefill = enabled;
        self
    }
}
/// Default compiler primitives and core helpers.
pub fn core_sources() -> impl Iterator<Item = &'static str> {
    catena_lang::stdlib::core_sources()
}
pub fn runtime_sources(model: &str) -> impl Iterator<Item = &str> {
    core_sources().map(|s| -> &str { s }).chain([model])
}
pub fn compile(
    raw: metacat::theory::RawTheorySet,
    dialect: catena_lang::runtime::GpuDialect,
) -> Result<catena_lang::runtime::RuntimeModule> {
    let raw = prepare_sources(raw)?;
    let module = catena_lang::compile::compile(
        &mut catena_lang::report::CompileReport::new(raw),
        catena_lang::codegen::CodegenKind::Default,
        dialect,
    )?;
    if let Some(path) = std::env::var_os("CATENA_QWEN3_CPP") {
        std::fs::write(path, &module.source)?;
    }
    Ok(module)
}
/// Compile source texts through the persistent compiler module cache.
/// The cache includes this adapter's source-pruning policy and saves the full ABI.
pub fn compile_sources<'a>(
    sources: impl IntoIterator<Item = &'a str>,
    dialect: catena_lang::runtime::GpuDialect,
) -> Result<catena_lang::runtime::RuntimeModule> {
    use catena_lang::{codegen::CodegenKind, runtime::ModuleCache};
    let started = std::time::Instant::now();
    let sources = sources.into_iter().collect::<Vec<_>>();
    let build = || {
        compile(
            metacat::theory::RawTheorySet::from_texts(sources.iter().copied())?,
            dialect,
        )
    };
    let module = match ModuleCache::from_sources(&sources, CodegenKind::Default, dialect) {
        Some(cache) => cache
            .with_context(include_str!("catena.rs"))
            .get_or_compile(build)?,
        None => build()?,
    };
    if std::env::var_os("CATENA_COMPILE_TIMINGS").is_some() {
        eprintln!(
            "[compile] Hex module ready in {:.3}s",
            started.elapsed().as_secs_f64()
        );
    }
    // Preserve diagnostics on cache hits as well as uncached builds.
    if let Some(path) = std::env::var_os("CATENA_QWEN3_CPP") {
        std::fs::write(path, &module.source)?;
    }
    Ok(module)
}
pub(crate) fn load_runtime<'a>(
    sources: impl IntoIterator<Item = &'a str>,
    dialect: catena_lang::runtime::GpuDialect,
) -> Result<(
    catena_lang::runtime::Runtime,
    catena_lang::runtime::Artifact,
)> {
    let module = compile_sources(sources, dialect)?;
    let mut runtime = catena_lang::runtime::Runtime::new(dialect)?;
    let artifact = runtime.load(module)?;
    Ok((runtime, artifact))
}
pub fn generated_model_source_for(config: &ModelConfig) -> Result<String> {
    generated_model_source_for_options(config, InferenceOptions::default())
}
pub fn generated_model_source_for_options(
    config: &ModelConfig,
    options: InferenceOptions,
) -> Result<String> {
    crate::qwen::source(config, options)
}
pub(crate) fn generated_multimodal_source(
    config: &ModelConfig,
    options: InferenceOptions,
) -> Result<String> {
    crate::qwen::multimodal_source(config, options)
}
/// Keep primitive declarations and exported programs, following callback dependencies.
pub(crate) fn prepare_sources(
    raw: metacat::theory::RawTheorySet,
) -> Result<metacat::theory::RawTheorySet> {
    use std::collections::BTreeSet;
    let mut raw = raw.with_extensions()?;
    let program = raw
        .theories
        .get_mut(&"program".parse().unwrap())
        .ok_or_else(|| anyhow::anyhow!("missing program theory"))?;
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

pub(crate) fn rope_frequency_program(config: &ModelConfig) -> String {
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
                "    ([.pair pair-{pair:02}] stdlib.qwen3.memory.u64.equal [is-pair-{pair:02}.])"
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
                "    ([.is-pair-{pair:02} frequency-{pair:02} {fallback}] stdlib.qwen3.memory.select-f32 [selected-{pair:02}.])"
            )
            .unwrap();
        }
        source.push_str("    [.selected-00]\n");
    }
    source.push_str("  )\n)");
    source
}

pub(crate) fn rename_programs(
    source: &str,
    mapping: &std::collections::BTreeMap<String, String>,
) -> String {
    fn token<'a>(s: &'a str, m: &'a std::collections::BTreeMap<String, String>) -> String {
        if let Some(n) = m.get(s) {
            return n.clone();
        }
        if let Some(s) = s.strip_prefix("name.") {
            if let Some(n) = m.get(s) {
                return format!("name.{n}");
            }
        }
        if let Some(s) = s.strip_prefix("partial.") {
            if let Some((name, n)) = s.rsplit_once('.') {
                if let Some(name) = m.get(name) {
                    return format!("partial.{name}.{n}");
                }
            }
        }
        s.to_owned()
    }
    let mut out = String::new();
    let mut start = 0;
    for (i, c) in source.char_indices() {
        if c.is_whitespace() || "(){}[]".contains(c) {
            out += &token(&source[start..i], mapping);
            out.push(c);
            start = i + c.len_utf8();
        }
    }
    out += &token(&source[start..], mapping);
    out
}
