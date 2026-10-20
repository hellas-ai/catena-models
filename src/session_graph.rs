//! Model-owned session allocation entrypoints.
//! This is source rendering only: the execution harness treats state as opaque.
use crate::{
    catena::{InferenceOptions, KernelKind},
    spec::{ModelArchitecture, ModelConfig},
};

pub(crate) fn append(source: &mut String, c: &ModelConfig, options: InferenceOptions) {
    append_layout(source, c, options);
    let start = source.len();
    let dtype = "bf16";
    source.push_str(include_str!("catena/state_buffers.hex"));

    let init = match (options.kernels, c.architecture) {
        (KernelKind::Reference, _) => include_str!("catena/state_init.hex"),
        (_, ModelArchitecture::Qwen35Moe) => include_str!("catena/state_init_moe.hex"),
        (_, ModelArchitecture::Qwen35Dense) => include_str!("catena/state_init_hybrid.hex"),
        _ => include_str!("catena/state_init_fused.hex"),
    };
    source.push_str(&init.replace("{{DTYPE}}", dtype));
    let helpers = source.split_off(start);
    let entry = source.find("(def program qwen3 :").unwrap_or(source.len());
    source.insert_str(entry, &helpers);
}

pub(crate) fn append_layout(source: &mut String, c: &ModelConfig, _options: InferenceOptions) {
    if source.contains("(def program stdlib.qwen35.layout.router :") {
        return;
    }
    let mut layout = include_str!("catena/state_layout.hex").to_owned();
    for (key, value) in [
        ("LAYERS", c.num_layers),
        ("QUERY_HEADS", c.num_attention_heads),
        ("KV_HEADS", c.num_key_value_heads),
        ("HEAD_DIM", c.head_dim),
        ("KEY_HEADS", c.linear_group_count.unwrap_or(0)),
        ("KEY_DIM", c.linear_state_size.unwrap_or(0)),
        ("VALUE_SIZE", c.linear_inner_size.unwrap_or(0)),
        ("VALUE_HEADS", c.linear_time_step_rank.unwrap_or(1)),
        ("CONV_KERNEL", c.linear_conv_kernel.unwrap_or(1)),
        ("EXPERTS", c.num_experts),
        ("TOP_K", c.experts_per_token),
        ("INTERMEDIATE", c.intermediate_size),
        (
            "SHARED_INTERMEDIATE",
            c.shared_expert_intermediate_size.unwrap_or(0),
        ),
    ] {
        layout = layout.replace(&format!("{{{{{key}}}}}"), &format!("{value:016X}"));
    }

    let entry = source.find("(def program qwen3 :").unwrap_or(source.len());
    source.insert_str(entry, &layout);
}
