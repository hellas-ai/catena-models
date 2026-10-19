//! Native Qwen3.5 still-image encoding and single-turn prompt preparation.
mod config;
mod graph;
pub mod image;
#[cfg(test)]
mod kernel_tests;
mod kernels;
pub use config::VisionConfig;

use crate::{
    checkpoint::{Checkpoint, GgufDtype},
    tokenizer::QwenTokenizer,
};
use anyhow::{Context, Result, ensure};
use catena_lang::{
    codegen::GpuDialect,
    runtime::{Artifact, MemOwn, Runtime, Value},
};
use std::{collections::BTreeMap, path::Path, time::Instant};

pub struct VisionModel {
    runtime: Runtime,
    artifact: Artifact,
    checkpoint: Checkpoint,
    weights: Vec<crate::checkpoint::TensorView>,
    converted: BTreeMap<usize, MemOwn>,
    zero: MemOwn,
    frequencies: MemOwn,
    config: VisionConfig,
    dialect: GpuDialect,
}
pub struct ImageEmbedding {
    pub(crate) values: MemOwn,
    pub(crate) grid: [usize; 3],
    pub(crate) width: usize,
    pub(crate) dialect: GpuDialect,
}
impl ImageEmbedding {
    pub fn grid(&self) -> [usize; 3] {
        self.grid
    }
    pub fn token_count(&self) -> usize {
        self.grid[1] * self.grid[2] / 4
    }
    pub fn to_f32_vec(&self) -> Result<Vec<f32>> {
        Ok(self.values.try_to_f32_vec()?)
    }
}
fn one_memory(mut outputs: Vec<Value<'_>>) -> Result<MemOwn> {
    ensure!(outputs.len() == 1, "vision output ABI mismatch");
    match outputs.pop().unwrap() {
        Value::MemOwn(m) => Ok(m),
        _ => anyhow::bail!("vision returned non-memory output"),
    }
}
impl VisionModel {
    pub fn load_gguf(
        path: impl AsRef<Path>,
        dialect: GpuDialect,
        max_pixels: Option<usize>,
    ) -> Result<Self> {
        let path = path.as_ref();
        let mut config = VisionConfig::from_gguf(path)?;
        if let Some(max) = max_pixels {
            config.max_pixels = max;
        }
        config.validate()?;
        let specs = config.weight_specs(path)?;
        let checkpoint = Checkpoint::open_inventory(path, &specs, None, dialect)
            .context("loading vision/projector GGUF (convert F16/quantized projectors with --mmproj --outtype bf16)")?;
        let mut names: Vec<_> = specs.iter().map(|s| s.gguf_name.clone()).collect();
        names.push("__zero".into());
        let source = kernels::source(&config, &names);
        let (runtime, artifact) =
            crate::catena::load_runtime(crate::catena::runtime_sources(&source), dialect)
                .context("compiling Qwen3.5 vision graph")?;
        let weights = specs
            .iter()
            .map(|s| checkpoint.tensor_view(s))
            .collect::<Result<Vec<_>>>()?;
        let mut converted = BTreeMap::new();
        // Vision kernels accept F32 matrices. Widen BF16 once on the GPU;
        // keep the immutable GGUF mapping alive for native F32 tensors.
        for (i, tensor) in weights.iter().enumerate() {
            if tensor.dtype == GgufDtype::Bf16 {
                let mem = one_memory(artifact.exec_values(
                    "vision.widen-bf16",
                    vec![
                        checkpoint.gpu_ref(tensor)?.into(),
                        (tensor.byte_len as u64 / 2).into(),
                    ],
                )?)?;
                converted.insert(i, mem);
            }
        }
        let zero = runtime.mem_f32(&vec![0.0; config.hidden_size])?;
        let dim = config.hidden_size / config.heads;
        let inv: Vec<_> = (0..dim / 4)
            .map(|i| 10000_f32.powf(-4. * i as f32 / dim as f32))
            .collect();
        let frequencies = runtime.mem_f32(&inv)?;
        Ok(Self {
            runtime,
            artifact,
            checkpoint,
            weights,
            converted,
            zero,
            frequencies,
            config,
            dialect,
        })
    }
    pub fn config(&self) -> &VisionConfig {
        &self.config
    }
    pub fn encode_image(&self, path: impl AsRef<Path>) -> Result<ImageEmbedding> {
        let started = Instant::now();
        let patches = image::ImagePatches::open(path.as_ref(), &self.config)?;
        eprintln!(
            "image {} resized to {}x{}, {} vision tokens ({:.2?} preprocessing)",
            path.as_ref().display(),
            patches.resized_size[1],
            patches.resized_size[0],
            patches.grid[1] * patches.grid[2] / 4,
            started.elapsed()
        );
        self.encode_patches(&patches)
    }
    pub fn encode_patches(&self, patches: &image::ImagePatches) -> Result<ImageEmbedding> {
        let rows = patches.grid[1]
            .checked_mul(patches.grid[2])
            .context("image grid overflow")?;
        ensure!(
            patches.grid[0] == 1
                && patches.grid[1].is_multiple_of(2)
                && patches.grid[2].is_multiple_of(2)
                && rows > 0,
            "invalid still-image grid"
        );
        ensure!(
            patches.pixels.len() == rows * 3 * self.config.patch_size * self.config.patch_size
                && patches.positions.len() == rows * 2
                && patches.bilinear_indices.len() == rows * 4
                && patches.bilinear_weights.len() == rows * 4,
            "image patch buffer mismatch"
        );
        ensure!(
            patches
                .bilinear_indices
                .iter()
                .all(|&i| i < (self.config.position_side * self.config.position_side) as u64),
            "image interpolation index out of range"
        );
        ensure!(
            patches
                .positions
                .as_chunks::<2>()
                .0
                .iter()
                .all(|p| p[0] < patches.grid[1] as u64 && p[1] < patches.grid[2] as u64),
            "image rotary position out of range"
        );
        ensure!(
            patches.pixels.iter().all(|v| v.is_finite())
                && patches
                    .bilinear_weights
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .all(|w| w.iter().all(|v| v.is_finite() && *v >= 0.)
                        && (w.iter().sum::<f32>() - 1.).abs() < 1e-5),
            "invalid image pixels or interpolation weights"
        );
        let pixels = self.runtime.mem_f32(&patches.pixels)?;
        let positions = self.runtime.mem_u64(&patches.positions)?;
        let indices = self.runtime.mem_u64(&patches.bilinear_indices)?;
        let coefficients = self.runtime.mem_f32(&patches.bilinear_weights)?;
        let mut args = vec![
            pixels.as_ref().into(),
            positions.as_ref().into(),
            indices.as_ref().into(),
            coefficients.as_ref().into(),
            self.frequencies.as_ref().into(),
        ];
        for (i, tensor) in self.weights.iter().enumerate() {
            args.push(
                if let Some(mem) = self.converted.get(&i) {
                    mem.as_ref()
                } else {
                    self.checkpoint.gpu_ref(tensor)?
                }
                .into(),
            );
        }
        args.push(self.zero.as_ref().into());
        let started = Instant::now();
        let values = one_memory(self.artifact.exec_values("vision.encode", args)?)?;
        ensure!(
            values.byte_len() == (rows / 4 * self.config.output_size * 4) as u64,
            "vision embedding length mismatch"
        );
        eprintln!("vision encoding: {:.2?}", started.elapsed());
        Ok(ImageEmbedding {
            values,
            grid: patches.grid,
            width: self.config.output_size,
            dialect: self.dialect,
        })
    }
}

pub(crate) struct ImageSpan {
    pub start: usize,
    pub len: usize,
}
/// Validated single-turn prompt. GPU image embeddings remain owned by this value.
pub struct PreparedMultimodalPrompt {
    pub(crate) tokens: Vec<u64>,
    pub(crate) positions: Vec<[u64; 3]>,
    pub(crate) spans: Vec<ImageSpan>,
    pub(crate) images: Vec<ImageEmbedding>,
    pub(crate) next_rope_position: u64,
}
impl PreparedMultimodalPrompt {
    pub fn new(
        tokenizer: &QwenTokenizer,
        prompt: &str,
        images: Vec<ImageEmbedding>,
    ) -> Result<Self> {
        ensure!(
            !images.is_empty(),
            "multimodal prompt requires at least one image"
        );
        let start = tokenizer.special_token_id("<|vision_start|>")?;
        let end = tokenizer.special_token_id("<|vision_end|>")?;
        let pad = tokenizer.special_token_id("<|image_pad|>")?;
        for token in ["<|im_start|>", "<|im_end|>", "<think>", "</think>"] {
            tokenizer.special_token_id(token)?;
        }
        // Tokenize ordinary content separately so prompt text cannot introduce unmatched image spans.
        let content = tokenizer.encode_without_special_tokens(prompt.trim())?;
        let video = tokenizer.special_token_id("<|video_pad|>").ok();
        ensure!(
            !content
                .iter()
                .any(|&t| t == start || t == end || t == pad || video == Some(t)),
            "prompt text contains reserved vision tokens"
        );
        let mut tokens = tokenizer.encode_without_special_tokens("<|im_start|>user\n")?;
        let mut spans = Vec::new();
        for image in &images {
            tokens.push(start);
            let len = image.token_count();
            ensure!(
                len > 0 && image.grid[0] == 1,
                "empty or non-still image embedding"
            );
            spans.push(ImageSpan {
                start: tokens.len(),
                len,
            });
            tokens.resize(
                tokens
                    .len()
                    .checked_add(len)
                    .context("image prompt length overflow")?,
                pad,
            );
            tokens.push(end);
        }
        tokens.extend(content);
        tokens.extend(tokenizer.encode_without_special_tokens(
            "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        )?);
        let (positions, next_rope_position) = prompt_positions(tokens.len(), &spans, &images)?;
        Ok(Self {
            tokens,
            positions,
            spans,
            images,
            next_rope_position,
        })
    }
    pub fn tokens(&self) -> &[u64] {
        &self.tokens
    }
    pub fn positions(&self) -> &[[u64; 3]] {
        &self.positions
    }
    pub fn token_count(&self) -> usize {
        self.tokens.len()
    }
    pub fn rope_delta(&self) -> i64 {
        self.next_rope_position as i64 - self.tokens.len() as i64
    }
}
fn prompt_positions(
    len: usize,
    spans: &[ImageSpan],
    images: &[ImageEmbedding],
) -> Result<(Vec<[u64; 3]>, u64)> {
    let grids: Vec<_> = images.iter().map(|image| image.grid).collect();
    position_ids(len, spans, &grids)
}
fn position_ids(
    len: usize,
    spans: &[ImageSpan],
    grids: &[[usize; 3]],
) -> Result<(Vec<[u64; 3]>, u64)> {
    ensure!(spans.len() == grids.len(), "image span count mismatch");
    let mut out = Vec::with_capacity(len);
    let mut pos = 0_u64;
    for (span, grid) in spans.iter().zip(grids) {
        ensure!(
            span.start >= out.len()
                && grid[0] == 1
                && grid[1] > 0
                && grid[2] > 0
                && grid[1].is_multiple_of(2)
                && grid[2].is_multiple_of(2),
            "invalid image span/grid"
        );
        while out.len() < span.start {
            out.push([pos; 3]);
            pos += 1;
        }
        let h = grid[1] / 2;
        let w = grid[2] / 2;
        ensure!(
            h.checked_mul(w) == Some(span.len)
                && span.start.checked_add(span.len).is_some_and(|n| n <= len),
            "image span length mismatch"
        );
        for y in 0..h {
            for x in 0..w {
                out.push([pos, pos + y as u64, pos + x as u64]);
            }
        }
        pos += h.max(w) as u64;
    }
    while out.len() < len {
        out.push([pos; 3]);
        pos += 1;
    }
    ensure!(out.len() == len, "position length mismatch");
    Ok((out, pos))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn multimodal_positions_keep_cache_and_rope_separate() {
        let spans = vec![
            ImageSpan { start: 3, len: 6 },
            ImageSpan { start: 11, len: 4 },
        ];
        let (p, next) = position_ids(17, &spans, &[[1, 4, 6], [1, 4, 4]]).unwrap();
        assert_eq!(p[3], [3, 3, 3]);
        assert_eq!(p[8], [3, 4, 5]);
        assert_eq!(p[9], [6; 3]);
        assert_eq!(p[11], [8; 3]);
        assert_eq!(p[15], [10; 3]);
        assert_eq!(next, 12);
    }
    #[test]
    fn invalid_span_rejected() {
        assert!(position_ids(5, &[ImageSpan { start: 3, len: 4 }], &[[1, 4, 4]]).is_err());
    }
    #[test]
    fn multimodal_language_graphs_compile_without_gpu() -> Result<()> {
        // The current compiler supports these graphs through BF16/HIP;
        // the default backend's fused model operations have been removed.
        for mut config in [
            crate::spec::ModelConfig::qwen35_0_8b(),
            crate::spec::ModelConfig::qwen36_35b_a3b(),
        ] {
            config.num_layers = 4;
            config.num_nextn_layers = 0;
            config.hidden_size = 32;
            config.intermediate_size = 64;
            config.num_attention_heads = 1;
            config.num_key_value_heads = 1;
            config.head_dim = 64;
            config.num_experts = if config.is_dense() { 0 } else { 2 };
            config.experts_per_token = if config.is_dense() { 0 } else { 1 };
            if !config.is_dense() {
                config.shared_expert_intermediate_size = Some(64);
            }
            config.rope_dimension_count = 8;
            config.mrope_sections = [2, 1, 1];
            config.vocabulary_size = 33;
            config.linear_state_size = Some(16);
            config.linear_group_count = Some(1);
            config.linear_time_step_rank = Some(2);
            config.linear_inner_size = Some(32);
            let options = crate::catena::InferenceOptions::default();
            let source = crate::catena::generated_multimodal_source(&config, options)?;
            let raw =
                metacat::theory::RawTheorySet::from_texts(crate::catena::runtime_sources(&source))?;
            let raw = crate::catena::prepare_sources(raw)?;
            let module = crate::catena::compile(raw, GpuDialect::Hip)?;
            ensure!(
                module
                    .functions
                    .iter()
                    .any(|f| f.source_name == "qwen3.multimodal"),
                "missing multimodal entrypoint"
            );
            if config.is_dense()
                && let Some(directory) = std::env::var_os("QWEN35_TEST_CPP")
            {
                std::fs::create_dir_all(&directory)?;
                std::fs::write(
                    std::path::Path::new(&directory).join("language-Hip.cpp"),
                    module.source,
                )?;
            }
        }
        Ok(())
    }
    #[test]
    fn vision_graph_compiles_without_gpu() -> Result<()> {
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
        let mut names = vec![
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
        .into_iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();
        for prefix in ["ln1", "ln2", "attn_qkv", "attn_out", "ffn_up", "ffn_down"] {
            for suffix in ["weight", "bias"] {
                names.push(format!("v.blk.0.{prefix}.{suffix}"));
            }
        }
        names.push("__zero".into());
        let source = kernels::source(&config, &names);
        let raw = metacat::theory::RawTheorySet::from_texts(
            crate::catena::core_sources()
                .map(|s| -> &str { s })
                .chain([source.as_str()]),
        )?;
        let raw = crate::catena::prepare_sources(raw)?;

        for dialect in [GpuDialect::Hip, GpuDialect::Cuda] {
            let module = crate::catena::compile(raw.clone(), dialect)?;
            ensure!(
                module
                    .functions
                    .iter()
                    .any(|f| f.source_name == "vision.encode"),
                "missing encoder entrypoint"
            );
            if let Some(directory) = std::env::var_os("QWEN35_TEST_CPP") {
                std::fs::create_dir_all(&directory)?;
                std::fs::write(
                    std::path::Path::new(&directory).join(format!("vision-{dialect:?}.cpp")),
                    module.source,
                )?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod reference_tests {
    use super::*;
    #[test]
    fn positions_match_pinned_transformers_two_image_fixture() -> Result<()> {
        let reference: serde_json::Value =
            serde_json::from_str(include_str!("../../tests/fixtures/qwen35-images.json"))?;
        let tokens = reference["tokens"].as_array().unwrap();
        let mut spans = vec![];
        let mut index = 0;
        while index < tokens.len() {
            if tokens[index].as_u64() == Some(248056) {
                let start = index;
                while index < tokens.len() && tokens[index].as_u64() == Some(248056) {
                    index += 1;
                }
                spans.push(ImageSpan {
                    start,
                    len: index - start,
                });
            } else {
                index += 1;
            }
        }
        let grids: Vec<[usize; 3]> = serde_json::from_value(reference["grids"].clone())?;
        let (positions, next) = position_ids(tokens.len(), &spans, &grids)?;
        assert_eq!(serde_json::json!(positions), reference["positions"]);
        assert_eq!(
            next as i64 - tokens.len() as i64,
            reference["rope_delta"].as_i64().unwrap()
        );
        Ok(())
    }
    #[test]
    #[ignore = "requires QWEN35_MMPROJ_MODELS (comma-separated HF names), no GPU"]
    fn real_projector_inventories() -> Result<()> {
        let models = std::env::var("QWEN35_MMPROJ_MODELS").context("set QWEN35_MMPROJ_MODELS")?;
        for model in models.split(',') {
            let path = crate::hfhub::download_mmproj(model.trim())?;
            let config = VisionConfig::from_gguf(&path)?;
            let specs = config.weight_specs(&path)?;
            let actual = crate::checkpoint::tensor_inventory(&path)?;
            ensure!(specs.len() == actual.len(), "unexpected projector tensors");
        }
        Ok(())
    }
}
