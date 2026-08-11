//! Fixed architecture and compact device-weight ABI for Qwen3-30B-A3B-Base.

pub const VOCAB_SIZE: usize = 151_936;
pub const HIDDEN_SIZE: usize = 2_048;
pub const INTERMEDIATE_SIZE: usize = 768;
pub const NUM_LAYERS: usize = 48;
pub const NUM_ATTENTION_HEADS: usize = 32;
pub const NUM_KEY_VALUE_HEADS: usize = 4;
pub const HEAD_DIM: usize = 128;
pub const QUERY_SIZE: usize = NUM_ATTENTION_HEADS * HEAD_DIM;
pub const KEY_VALUE_SIZE: usize = NUM_KEY_VALUE_HEADS * HEAD_DIM;
pub const ROPE_HALF_DIM: usize = HEAD_DIM / 2;
pub const MAX_POSITION_EMBEDDINGS: usize = 32_768;
pub const RMS_NORM_EPS: f32 = 1.0e-6;
pub const ROPE_THETA: f32 = 1_000_000.0;
pub const NUM_EXPERTS: usize = 128;
pub const EXPERTS_PER_TOKEN: usize = 8;
pub const EOS_TOKEN_ID: u64 = 151_643;

pub const TENSORS_PER_LAYER: usize = 12;
pub const WEIGHT_COUNT: usize = 1 + NUM_LAYERS * TENSORS_PER_LAYER + 2;
pub const MODEL_INPUT_COUNT: usize = 5 + WEIGHT_COUNT;
pub const KV_CACHE_ELEMENTS_PER_TOKEN: usize = NUM_LAYERS * 2 * NUM_KEY_VALUE_HEADS * HEAD_DIM;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorSource {
    pub name: String,
    pub shape: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorSpec {
    /// Stable name of the device-side buffer. Grouped expert buffers do not
    /// exist as a single tensor in the Hugging Face checkpoint.
    pub name: String,
    pub shape: Vec<usize>,
    pub sources: Vec<TensorSource>,
}

impl TensorSpec {
    pub fn element_count(&self) -> usize {
        self.shape.iter().product()
    }

    fn single(name: String, shape: Vec<usize>) -> Self {
        Self {
            sources: vec![TensorSource {
                name: name.clone(),
                shape: shape.clone(),
            }],
            name,
            shape,
        }
    }

    fn experts(layer: usize, projection: &str, source_shape: Vec<usize>) -> Self {
        let sources = (0..NUM_EXPERTS)
            .map(|expert| TensorSource {
                name: format!("model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"),
                shape: source_shape.clone(),
            })
            .collect();
        let mut shape = vec![NUM_EXPERTS];
        shape.extend(source_shape);
        Self {
            name: format!("model.layers.{layer}.mlp.experts.{projection}.weight"),
            shape,
            sources,
        }
    }
}

/// Device weight order shared by checkpoint loading and the generated Catena ABI.
/// The 128 per-expert tensors for each projection are packed consecutively.
pub fn weight_specs() -> Vec<TensorSpec> {
    let mut tensors = Vec::with_capacity(WEIGHT_COUNT);
    tensors.push(TensorSpec::single(
        "model.embed_tokens.weight".into(),
        vec![VOCAB_SIZE, HIDDEN_SIZE],
    ));

    for layer in 0..NUM_LAYERS {
        let prefix = format!("model.layers.{layer}");
        tensors.extend([
            TensorSpec::single(
                format!("{prefix}.input_layernorm.weight"),
                vec![HIDDEN_SIZE],
            ),
            TensorSpec::single(
                format!("{prefix}.self_attn.q_proj.weight"),
                vec![QUERY_SIZE, HIDDEN_SIZE],
            ),
            TensorSpec::single(
                format!("{prefix}.self_attn.k_proj.weight"),
                vec![KEY_VALUE_SIZE, HIDDEN_SIZE],
            ),
            TensorSpec::single(
                format!("{prefix}.self_attn.v_proj.weight"),
                vec![KEY_VALUE_SIZE, HIDDEN_SIZE],
            ),
            TensorSpec::single(
                format!("{prefix}.self_attn.o_proj.weight"),
                vec![HIDDEN_SIZE, QUERY_SIZE],
            ),
            TensorSpec::single(format!("{prefix}.self_attn.q_norm.weight"), vec![HEAD_DIM]),
            TensorSpec::single(format!("{prefix}.self_attn.k_norm.weight"), vec![HEAD_DIM]),
            TensorSpec::single(
                format!("{prefix}.post_attention_layernorm.weight"),
                vec![HIDDEN_SIZE],
            ),
            TensorSpec::single(
                format!("{prefix}.mlp.gate.weight"),
                vec![NUM_EXPERTS, HIDDEN_SIZE],
            ),
            TensorSpec::experts(layer, "gate_proj", vec![INTERMEDIATE_SIZE, HIDDEN_SIZE]),
            TensorSpec::experts(layer, "up_proj", vec![INTERMEDIATE_SIZE, HIDDEN_SIZE]),
            TensorSpec::experts(layer, "down_proj", vec![HIDDEN_SIZE, INTERMEDIATE_SIZE]),
        ]);
    }

    tensors.push(TensorSpec::single(
        "model.norm.weight".into(),
        vec![HIDDEN_SIZE],
    ));
    tensors.push(TensorSpec::single(
        "lm_head.weight".into(),
        vec![VOCAB_SIZE, HIDDEN_SIZE],
    ));
    debug_assert_eq!(tensors.len(), WEIGHT_COUNT);
    tensors
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkpoint_abi_has_the_expected_inventory() {
        let specs = weight_specs();
        assert_eq!(specs.len(), 579);
        assert_eq!(MODEL_INPUT_COUNT, 584);
        assert_eq!(KV_CACHE_ELEMENTS_PER_TOKEN, 49_152);
        assert_eq!(specs.first().unwrap().name, "model.embed_tokens.weight");
        assert_eq!(specs.last().unwrap().name, "lm_head.weight");
        assert_eq!(
            specs.iter().map(|spec| spec.sources.len()).sum::<usize>(),
            18_867
        );
        assert_eq!(
            specs.iter().map(TensorSpec::element_count).sum::<usize>(),
            30_532_122_624
        );
    }

    #[test]
    fn source_tensor_names_are_unique() {
        let specs = weight_specs();
        let mut names = specs
            .iter()
            .flat_map(|spec| spec.sources.iter().map(|source| &source.name))
            .collect::<Vec<_>>();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), 18_867);
    }
}
