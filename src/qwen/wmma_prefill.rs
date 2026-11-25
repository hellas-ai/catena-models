//! Render the Qwen3 prefill graph with the shared dense matrix-core kernels.
use super::*;
use crate::catena::definitions;

pub(super) fn append(source: &mut String, config: &ModelConfig, options: InferenceOptions) {
    let mut kernels = definitions(
        include_str!("../catena/qwen35.hex"),
        &["stdlib.qwen35.norm.", "qwen35.normalize:"],
    );
    if options.kernels == KernelKind::Reference {
        kernels.push_str(&definitions(
            include_str!("../catena/fused_normalization.hex"),
            &["stdlib.qwen3.fused.norm.write"],
        ));
    }
    kernels.push_str(&definitions(
        include_str!("../catena/qwen35_cooperative_norm.hex"),
        &[
            "stdlib.qwen35.coop.write:",
            "stdlib.qwen35.coop.write.fold:",
        ],
    ));
    kernels.push_str(&definitions(
        include_str!("../catena/qwen35_attention.hex"),
        &["stdlib.qwen35.pow2", "stdlib.models.memory.u64.ceil_div:"],
    ));
    kernels.push_str(&definitions(
        include_str!("../catena/qwen35_batch_projection.hex"),
        &["stdlib.qwen35.batch.project.write"],
    ));
    kernels.push_str(&definitions(
        include_str!("../catena/lfm3_wmma.hex"),
        &["stdlib.lfm3.wmma.m128n64k64."],
    ));
    kernels.push_str(&definitions(
        include_str!("../catena/qwen35_dense.hex"),
        &[
            "stdlib.qwen35.dense.project:",
            "stdlib.qwen35.dense.project.",
            "stdlib.qwen35.dense.projection.",
        ],
    ));
    kernels.push_str(include_str!("../catena/qwen35_batch_wmma.hex"));
    kernels.push_str(include_str!("../catena/qwen35_batch_full_prepare.hex"));
    kernels.push_str(include_str!("../catena/qwen35_prefill.hex"));
    kernels.push_str(&ungated_attention());
    let fp8 = options.prefill_dtype == crate::catena::PrefillDtype::Fp8;
    let prefill = if fp8 {
        super::fp8_prefill::graph()
    } else {
        include_str!("../catena/qwen3_prefill.hex").to_owned()
    };
    kernels.push_str(&prefill);

    let width = config.hidden_size;
    let head = config.head_dim;
    for (key, value) in [
        ("HIDDEN_SIZE", width),
        ("PAD_HIDDEN_SIZE", width.next_power_of_two()),
        (
            "HIDDEN_STAGES",
            width.next_power_of_two().trailing_zeros() as usize,
        ),
        ("INTERMEDIATE_SIZE", config.intermediate_size),
        ("HEAD_DIM", head),
        ("PAD_HEAD_DIM", head.next_power_of_two()),
        (
            "HEAD_STAGES",
            head.next_power_of_two().trailing_zeros() as usize,
        ),
        ("ROPE_DIM", config.rope_dimension_count),
        ("ROPE_HALF_DIM", config.rope_dimension_count / 2),
        ("KEY_VALUE_SIZE", config.key_value_size()),
        ("KV_LAYER_SIZE", 2 * config.key_value_size()),
        ("QUERY_HEADS", config.num_attention_heads),
        (
            "QUERY_HEADS_PER_KV_HEAD",
            config.num_attention_heads / config.num_key_value_heads,
        ),
        (
            "FULL_PREP_HEADS",
            config.num_attention_heads + config.num_key_value_heads,
        ),
        ("QUERY_SIZE", config.query_size()),
        ("DOUBLE_QUERY_SIZE", config.query_size()),
        ("QUERY_PACKING", 1),
    ] {
        kernels = kernels.replace(&format!("{{{{{key}}}}}"), &format!("{value:016X}"));
    }
    for (key, value) in [
        ("RMS_EPSILON_BITS", config.rms_norm_eps),
        ("SCALE_BITS", (head as f32).sqrt().recip()),
        ("HEAD_DIM_F32", head as f32),
        ("HIDDEN_F32", width as f32),
    ] {
        kernels = kernels.replace(
            &format!("{{{{{key}}}}}"),
            &format!("{:08X}", value.to_bits()),
        );
    }
    let layer = nested_product(&vec![REF.to_owned(); if fp8 { 25 } else { 11 }]);
    let weights = nested_product(&vec![layer; config.num_layers]);
    for (key, value) in [
        ("WEIGHTS", weights),
        ("LAYERS", format!("{:08X}", config.num_layers)),
    ] {
        kernels = kernels.replace(&format!("{{{{{key}}}}}"), &value);
    }
    source.push_str(&kernels);
    if fp8 {
        super::fp8_prefill::append(source);
    }
}

fn ungated_attention() -> String {
    let shared = include_str!("../catena/qwen35_wmma_attention.hex");
    let epilogue = definitions(shared, &["stdlib.qwen35.wmma.attention.context-write:"]);
    // Preserve the output addressing and BF16 rounding. Qwen3 has no output gate.
    let gate = epilogue.find("\n  ({(const.u64.0x{{QUERY_SIZE}})").unwrap();
    let store = epilogue.find("\n  ({[.out] [.v4]").unwrap();
    let mut ungated = epilogue.clone();
    ungated.replace_range(gate..store, "\n  ([.value] f32.to-bf16 [v18.])");
    shared.replace(epilogue.trim(), ungated.trim())
}
