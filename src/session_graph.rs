//! Model-owned session allocation and diagnostic entrypoints.
//! This is source rendering only: the execution harness treats state as opaque.
use crate::{
    catena::{InferenceOptions, KernelKind},
    spec::{ModelArchitecture, ModelConfig},
};
#[cfg(test)]
use std::fmt::Write;

pub(crate) fn append(source: &mut String, c: &ModelConfig, options: InferenceOptions) {
    append_layout(source, c, options);
    #[cfg(test)]
    let own = "((cap.own mem) val)";
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
    #[cfg(test)]
    {
        let workspace = options.kernels == KernelKind::Fused;
        let moe = workspace && c.architecture == ModelArchitecture::Qwen35Moe;
        let mut states = vec!["cache".to_owned()];
        if workspace {
            states.push("workspace".into());
        }
        if moe {
            states.extend((0..7).map(|i| format!("scratch{i}")));
        }
        let types = vec![own; states.len()].join(" ");
        let values = states
            .iter()
            .map(|s| format!("[.{s}]"))
            .collect::<Vec<_>>()
            .join(" ");
        let borrowed = "((cap.ref mem) val)";
        // Tests inspect/import cache through Hex, never by indexing opaque state.
        writeln!(
            source,
            "\n(def program qwen3.state-export : {{{types}}} -> {{{types} {own}}} =\n ([{}.]",
            states.join(" ")
        )
        .unwrap();

        writeln!(
            source,
            "  ([.cache] stdlib.qwen3.memory.own-count-{dtype} [count.])"
        )
        .unwrap();

        writeln!(source,"  ({{[.cache] [.count]}} stdlib.qwen3.state-copy-own-{dtype} [snapshot.])\n  {{{values} [.snapshot]}}\n ))").unwrap();
        writeln!(source,"\n(def program qwen3.state-import : {{{borrowed} {types}}} -> {{{types}}} =\n ([replacement {}.]",states.join(" ")).unwrap();

        writeln!(
            source,
            "  ([.cache] stdlib.qwen3.memory.own-count-{dtype} [count.])"
        )
        .unwrap();

        writeln!(source,"  ({{[.replacement] [.count]}} stdlib.qwen3.state-copy-ref-{dtype} [copied.])\n  ({{[.cache] [.copied]}} qwen3.free-{dtype}-before [new-cache.])\n  {{{}}}\n ))",states.iter().map(|s| if s=="cache" {"[.new-cache]".to_owned()} else {format!("[.{s}]")}).collect::<Vec<_>>().join(" ")).unwrap();
    }
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
