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
