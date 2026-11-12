//! Select Hex helpers for the shared vision composition.
//! This renders a graph; allocation, kernels, scheduling and releases execute in Hex.
use super::{VisionConfig, graph};

pub(super) fn source(config: &VisionConfig, names: &[String]) -> String {
    let graph = graph::source(config, names);
    let mut source = [
        include_str!("../catena/memory.hex"),
        include_str!("../catena/numeric.hex"),
        include_str!("../catena/resources.hex"),
        include_str!("../catena/qwen35_vision.hex"),
        &graph,
    ]
    .join("\n");
    if config.kind == super::VisionKind::Lfm3 {
        crate::qwen::fused::append_hex(&mut source, include_str!("../catena/cooperative.hex"));
        source.push_str(include_str!("../catena/lfm3_vision.hex"));
    }
    source
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires LFM3_VL_PROJECTOR pointing to the cached BF16 projector"]
    fn lfm3_graph_compiles() -> anyhow::Result<()> {
        let path = std::path::PathBuf::from(std::env::var("LFM3_VL_PROJECTOR")?);
        let config = VisionConfig::from_gguf(&path)?;
        anyhow::ensure!(config.kind == super::super::VisionKind::Lfm3);
        let mut names = config
            .weight_specs(&path)?
            .into_iter()
            .map(|s| s.gguf_name)
            .collect::<Vec<_>>();
        names.push("__zero".into());
        let source = source(&config, &names);
        let module = crate::catena::compile_sources(
            crate::catena::runtime_sources(&source),
            crate::model::configured_gpu_dialect()?,
        )?;
        assert!(!module.source.contains("__builtin_amdgcn_fdot2"));
        Ok(())
    }
}
