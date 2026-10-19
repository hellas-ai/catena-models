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
    // Only the encoder and widening adapter form the public runtime ABI.
    for name in [
        "exp-nonneg",
        "exp",
        "silu",
        "sigmoid",
        "softplus",
        "sqrt",
        "rope-sincos",
        "free-f32-before",
        "free-bf16-before",
        "keep-bf16-before",
    ] {
        source = source.replace(&format!("qwen3.{name}"), &format!("stdlib.vision.{name}"));
    }
    source
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use catena_lang::runtime::GpuDialect;

    fn fixture() -> (VisionConfig, Vec<String>) {
        let config = VisionConfig {
            hidden_size: 8,
            intermediate_size: 16,
            output_size: 8,
            layers: 1,
            heads: 2,
            patch_size: 2,
            merge_size: 2,
            position_side: 4,
            eps: 1e-6,
            min_pixels: 16,
            max_pixels: 64,
            mean: [0.5; 3],
            std: [0.5; 3],
        };
        let mut names = [
            "v.patch_embd.weight",
            "v.patch_embd.weight.1",
            "v.patch_embd.bias",
            "v.position_embd.weight",
            "v.post_ln.weight",
            "v.post_ln.bias",
            "mm.0.weight",
            "mm.0.bias",
            "mm.2.weight",
            "mm.2.bias",
        ]
        .map(str::to_owned)
        .to_vec();
        for prefix in ["ln1", "ln2", "attn_qkv", "attn_out", "ffn_up", "ffn_down"] {
            for suffix in ["weight", "bias"] {
                names.push(format!("v.blk.0.{prefix}.{suffix}"));
            }
        }
        names.push("__zero".into());
        (config, names)
    }

    #[test]
    fn vision_graph_compiles_without_legacy_primitives() -> Result<()> {
        let (config, names) = fixture();
        let source = source(&config, &names);
        assert!(!source.contains("materializec."));
        assert!(!source.contains("flux."));
        let raw = metacat::theory::RawTheorySet::from_texts(
            crate::catena::core_sources()
                .map(|s| -> &str { s })
                .chain([source.as_str()]),
        )?;
        let raw = crate::catena::prepare_sources(raw)?;
        let module = crate::catena::compile(raw, GpuDialect::Hip)?;
        assert!(
            module
                .functions
                .iter()
                .any(|f| f.source_name == "vision.encode")
        );
        assert_eq!(module.functions.len(), 2);
        Ok(())
    }
}
