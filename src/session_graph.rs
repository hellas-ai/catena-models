//! Model-owned session allocation and diagnostic entrypoints.
//! This is source rendering only: the execution harness treats state as opaque.
use crate::{
    catena::{ExperimentalKernelKind, InferenceOptions, InferencePrecision},
    spec::{ModelArchitecture, ModelConfig},
};
use catena_lang::codegen::CodegenKind;
#[cfg(test)]
use std::fmt::Write;

pub(crate) fn append(source: &mut String, c: &ModelConfig, options: InferenceOptions) {
    let experimental = options.codegen == CodegenKind::Experimental;
    append_layout(source, c, options);
    let own = if experimental {
        "((cap.own mem) val)"
    } else {
        "(cap.own mem)"
    };
    let start = source.len();
    let dtype = if options.precision == InferencePrecision::Bf16 {
        "bf16"
    } else {
        "f32"
    };
    if experimental {
        source.push_str(include_str!("catena/experimental/state_buffers.hex"));
    } else {
        for definition in include_str!("catena/state_buffers.hex")
            .split("(def program ")
            .skip(1)
        {
            let name = definition.split_whitespace().next().unwrap();
            if name.contains(&format!("-{dtype}")) {
                source.push_str("(def program ");
                source.push_str(definition);
            }
        }
    }
    let init = match (
        options.codegen,
        options.experimental_kernels,
        c.architecture,
    ) {
        (CodegenKind::Default, _, _) | (_, ExperimentalKernelKind::Reference, _) => {
            include_str!("catena/state_init.hex")
        }
        (_, _, ModelArchitecture::Qwen35Moe) => {
            include_str!("catena/experimental/state_init_moe.hex")
        }
        (_, _, ModelArchitecture::Qwen35Dense) => {
            include_str!("catena/experimental/state_init_hybrid.hex")
        }
        _ => include_str!("catena/experimental/state_init_fused.hex"),
    };
    source.push_str(
        &init
            .replace("{{DTYPE}}", dtype)
            .replace("(cap.own mem)", own),
    );
    #[cfg(test)]
    {
        let workspace =
            experimental && options.experimental_kernels == ExperimentalKernelKind::Fused;
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
        let borrowed = if experimental {
            "((cap.ref mem) val)"
        } else {
            "(cap.ref mem)"
        };
        // Tests inspect/import cache through Hex, never by indexing opaque state.
        writeln!(
            source,
            "\n(def program qwen3.state-export : {{{types}}} -> {{{types} {own}}} =\n ([{}.]",
            states.join(" ")
        )
        .unwrap();
        if experimental {
            writeln!(
                source,
                "  ([.cache] stdlib.qwen3.memory.own-count-{dtype} [count.])"
            )
            .unwrap();
        } else {
            writeln!(source,"  ([.cache] mem.cast.{dtype} [named-count ignored.])\n  ([.named-count] :.forget [count.])").unwrap();
        }
        writeln!(source,"  ({{[.cache] [.count]}} stdlib.qwen3.state-copy-own-{dtype} [snapshot.])\n  {{{values} [.snapshot]}}\n ))").unwrap();
        writeln!(source,"\n(def program qwen3.state-import : {{{borrowed} {types}}} -> {{{types}}} =\n ([replacement {}.]",states.join(" ")).unwrap();
        if experimental {
            writeln!(
                source,
                "  ([.cache] stdlib.qwen3.memory.own-count-{dtype} [count.])"
            )
            .unwrap();
        } else {
            writeln!(source,"  ([.cache] mem.cast.{dtype} [named-count ignored.])\n  ([.named-count] :.forget [count.])").unwrap();
        }
        writeln!(source,"  ({{[.replacement] [.count]}} stdlib.qwen3.state-copy-ref-{dtype} [copied.])\n  ({{[.cache] [.copied]}} qwen3.free-{dtype}-before [new-cache.])\n  {{{}}}\n ))",states.iter().map(|s| if s=="cache" {"[.new-cache]".to_owned()} else {format!("[.{s}]")}).collect::<Vec<_>>().join(" ")).unwrap();
    }
    let helpers = source.split_off(start);
    let entry = source.find("(def program qwen3 :").unwrap_or(source.len());
    source.insert_str(entry, &helpers);
}

pub(crate) fn append_layout(source: &mut String, c: &ModelConfig, options: InferenceOptions) {
    if source.contains("(def program stdlib.qwen35.layout.router :") {
        return;
    }
    let experimental = options.codegen == CodegenKind::Experimental;
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
    if experimental {
        for (from, to) in [
            ("u64.mul", "u64.mul"),
            ("u64.add", "u64.add"),
            ("u64.div", "u64.div"),
            ("u64.sub", "u64.sub"),
            ("u64.lt", "u64.less"),
            ("u64.select", "select-u64"),
        ] {
            layout = layout.replace(from, &format!("stdlib.qwen3.memory.{to}"));
        }
    }
    let entry = source.find("(def program qwen3 :").unwrap_or(source.len());
    source.insert_str(entry, &layout);
}
