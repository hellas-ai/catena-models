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
