//! Text-only k2-fsa/OmniVoice synthesis with the bundled Higgs audio decoder.
mod graph;
mod prompt;

use crate::{
    catena, hfhub, model,
    spec::{ModelConfig, WeightDtype, weight_specs},
    tts::Audio,
    weights::{Weights, bf16},
};
use anyhow::{Context, Result, anyhow, ensure};
use catena_lang::runtime::{Artifact, MemOwn, Runtime, Value};
use graph::{Packing, Tensor};
use serde_json::{Value as Json, json};
use std::path::Path;
use tokenizers::Tokenizer;

pub const DEFAULT_MODEL: &str = "k2-fsa/OmniVoice";
pub const DEFAULT_NUM_STEP: u32 = 32;
const SAMPLE_RATE: u32 = 24_000;
const FRAME_SAMPLES: usize = 960;
const MAX_FRAMES: usize = 30 * SAMPLE_RATE as usize / FRAME_SAMPLES;
const FILES: &[&str] = &[
    "config.json",
    "tokenizer.json",
    "model.safetensors",
    "audio_tokenizer/config.json",
    "audio_tokenizer/model.safetensors",
];

/// Conditioning and refinement settings for one utterance.
pub struct SynthesisOptions<'a> {
    pub instruct: Option<&'a str>,
    /// Number of masked refinement steps. Must be positive.
    pub num_step: u32,
}

impl Default for SynthesisOptions<'_> {
    fn default() -> Self {
        Self {
            instruct: None,
            num_step: DEFAULT_NUM_STEP,
        }
    }
}

pub struct OmniVoice {
    runtime: Runtime,
    artifact: Artifact,
    model: Weights,
    model_names: Vec<String>,
    codec: Weights,
    codec_weights: Vec<Tensor>,
    quantizer: [MemOwn; 3],
    tokenizer: Tokenizer,
    config: ModelConfig,
}

impl OmniVoice {
    /// Resolve the OmniVoice checkpoint through Hugging Face Hub and load the HEX graph.
    pub fn load() -> Result<Self> {
        let directory = hfhub::resolve_model_files(DEFAULT_MODEL, FILES)?;
        Self::load_snapshot(&directory, false)
    }

    fn load_snapshot(directory: &Path, diagnostics: bool) -> Result<Self> {
        let config = config(directory)?;
        let tokenizer = Tokenizer::from_file(directory.join("tokenizer.json"))
            .map_err(|err| anyhow!("loading OmniVoice tokenizer: {err}"))?;
        let dialect = model::configured_gpu_dialect()?;
        let (source, codec_weights) = graph::build(&config, diagnostics, dialect);
        let mut model = Weights::open(&directory.join("model.safetensors"))?;
        let (model_names, model_f32, model_bf16) = model_bindings(&config, &model)?;
        let mut codec = Weights::open(&directory.join("audio_tokenizer/model.safetensors"))?;
        for tensor in &codec_weights {
            codec.validate(&tensor.name, &tensor.shape)?;
        }
        let (runtime, artifact) = catena::load_runtime(catena::runtime_sources(&source), dialect)
            .context("compiling and loading OmniVoice HEX")?;
        model.prepare(&runtime, &model_f32, true)?;
        model.prepare(&runtime, &model_bf16, false)?;
        let codec_names = |f32_storage| {
            codec_weights
                .iter()
                .filter(|tensor| {
                    matches!(
                        (tensor.packing, f32_storage),
                        (Packing::F32, true) | (Packing::Bf16, false)
                    )
                })
                .map(|tensor| tensor.name.clone())
                .collect::<Vec<_>>()
        };
        codec.prepare(&runtime, &codec_names(true), true)?;
        codec.prepare(&runtime, &codec_names(false), false)?;
        for tensor in &codec_weights {
            if let Packing::TransposedConv = tensor.packing {
                let [input, output, taps] = *tensor.shape.as_slice() else {
                    unreachable!("transposed-convolution dimensions")
                };
                let data = codec.host_f32(&tensor.name)?;
                let mut packed = Vec::with_capacity(data.len());
                for co in 0..output {
                    for ci in 0..input {
                        let start = (ci * output + co) * taps;
                        packed.extend_from_slice(&data[start..start + taps]);
                    }
                }
                codec.insert_prepared(&tensor.name, runtime.mem_u16(&bf16(&packed))?)?;
            }
        }
        let quantizer = [
            prepare_quantizer(&runtime, &codec, "codebook.embed", &[1024, 64])?,
            prepare_quantizer(&runtime, &codec, "project_out.weight", &[1024, 64])?,
            prepare_quantizer(&runtime, &codec, "project_out.bias", &[1024])?,
        ];
        Ok(Self {
            runtime,
            artifact,
            model,
            model_names,
            codec,
            codec_weights,
            quantizer,
            tokenizer,
            config,
        })
    }

    /// Generate a single text-only utterance with reproducible sampling.
    pub fn synthesize(&self, text: &str, seed: u64) -> Result<Audio> {
        self.synthesize_with_instruct(text, seed, None)
    }

    /// Generate an utterance with optional English voice directions.
    /// Free-form emotion directions are experimental.
    pub fn synthesize_with_instruct(
        &self,
        text: &str,
        seed: u64,
        instruct: Option<&str>,
    ) -> Result<Audio> {
        self.synthesize_with_options(
            text,
            seed,
            SynthesisOptions {
                instruct,
                ..Default::default()
            },
        )
    }

    /// Generate an utterance using the requested refinement step count.
    pub fn synthesize_with_options(
        &self,
        text: &str,
        seed: u64,
        options: SynthesisOptions<'_>,
    ) -> Result<Audio> {
        ensure!(options.num_step > 0, "OmniVoice num_step must be positive");
        let tokens = prompt::tokens(&self.tokenizer, text, options.instruct)?;
        let frames = prompt::duration_frames(text);
        ensure!(
            frames <= MAX_FRAMES,
            "OmniVoice supports up to 30 seconds per utterance"
        );
        ensure!(
            tokens.len() + frames <= self.config.max_position_embeddings
                && tokens
                    .iter()
                    .all(|&id| id < self.config.vocabulary_size as u64),
            "OmniVoice prompt exceeds the checkpoint's context or vocabulary"
        );
        let text = self.runtime.mem_u64(&tokens)?;
        let mut args = vec![
            text.as_ref().into(),
            (tokens.len() as u64).into(),
            (frames as u64).into(),
            seed.into(),
            options.num_step.into(),
        ];
        for name in &self.model_names {
            args.push(self.model.gpu_ref(name)?.into());
        }
        self.codec_args(&mut args)?;
        let wave = self.execute("omnivoice.synthesize", args)?;
        let samples = wave.try_to_f32_vec()?;
        ensure!(
            samples.len() == frames * FRAME_SAMPLES + SAMPLE_RATE as usize / 5
                && samples.iter().all(|x| x.is_finite()),
            "OmniVoice returned an invalid waveform"
        );
        Ok(Audio {
            samples,
            sample_rate: SAMPLE_RATE,
        })
    }

    fn codec_args<'a>(&'a self, args: &mut Vec<Value<'a>>) -> Result<()> {
        args.extend(self.quantizer.iter().map(|mem| mem.as_ref().into()));
        for tensor in &self.codec_weights {
            args.push(self.codec.gpu_ref(&tensor.name)?.into());
        }
        Ok(())
    }

    fn execute(&self, entry: &str, args: Vec<Value<'_>>) -> Result<MemOwn> {
        let mut results = self
            .artifact
            .exec_values(entry, args)
            .context("executing OmniVoice HEX")?;
        ensure!(results.len() == 1, "invalid OmniVoice output count");
        match results.pop() {
            Some(Value::MemOwn(samples)) => Ok(samples),
            _ => Err(anyhow!("invalid OmniVoice output type")),
        }
    }
}

fn config(directory: &Path) -> Result<ModelConfig> {
    let model: Json = hfhub::read_json(directory.join("config.json"))?;
    let codec: Json = hfhub::read_json(directory.join("audio_tokenizer/config.json"))?;
    let mut qwen = ModelConfig::qwen3_0_6b();
    qwen.vocabulary_size = 151_676;
    qwen.eos_token_id = Some(151_645);
    let fields = [
        (&model, "/model_type", json!("omnivoice")),
        (&model, "/num_audio_codebook", json!(8)),
        (&model, "/audio_vocab_size", json!(1025)),
        (&model, "/audio_mask_id", json!(1024)),
        (&model, "/llm_config/model_type", json!("qwen3")),
        (
            &model,
            "/llm_config/vocab_size",
            json!(qwen.vocabulary_size),
        ),
        (&model, "/llm_config/hidden_size", json!(qwen.hidden_size)),
        (
            &model,
            "/llm_config/intermediate_size",
            json!(qwen.intermediate_size),
        ),
        (
            &model,
            "/llm_config/num_hidden_layers",
            json!(qwen.num_layers),
        ),
        (
            &model,
            "/llm_config/num_attention_heads",
            json!(qwen.num_attention_heads),
        ),
        (
            &model,
            "/llm_config/num_key_value_heads",
            json!(qwen.num_key_value_heads),
        ),
        (&model, "/llm_config/head_dim", json!(qwen.head_dim)),
        (
            &model,
            "/llm_config/rope_parameters/rope_theta",
            json!(qwen.rope_theta as u64),
        ),
        (
            &model,
            "/llm_config/rope_parameters/rope_type",
            json!("default"),
        ),
        (&model, "/llm_config/rms_norm_eps", json!(1.0e-6)),
        (&model, "/llm_config/attention_bias", json!(false)),
        (&model, "/llm_config/hidden_act", json!("silu")),
        (
            &model,
            "/llm_config/max_position_embeddings",
            json!(qwen.max_position_embeddings),
        ),
        (&codec, "/model_type", json!("higgs_audio_v2_tokenizer")),
        (&codec, "/sample_rate", json!(SAMPLE_RATE)),
        (&codec, "/codebook_size", json!(1024)),
        (&codec, "/codebook_dim", json!(64)),
        (&codec, "/acoustic_model_config/hidden_size", json!(256)),
        (
            &codec,
            "/acoustic_model_config/decoder_hidden_size",
            json!(1024),
        ),
        (
            &codec,
            "/acoustic_model_config/hop_length",
            json!(FRAME_SAMPLES),
        ),
        (
            &codec,
            "/acoustic_model_config/upsampling_ratios",
            json!(graph::STRIDES),
        ),
    ];
    for (config, path, expected) in fields {
        let actual = config.pointer(path);
        ensure!(
            actual == Some(&expected),
            "unsupported OmniVoice config field {path}: expected {expected}, got {actual:?}"
        );
    }
    Ok(qwen)
}

fn model_bindings(
    config: &ModelConfig,
    weights: &Weights,
) -> Result<(Vec<String>, Vec<String>, Vec<String>)> {
    let specs = weight_specs(config)?;
    let mut names = vec![
        "llm.embed_tokens.weight".to_owned(),
        "audio_embeddings.weight".to_owned(),
        "audio_heads.weight".to_owned(),
        "llm.norm.weight".to_owned(),
    ];
    let mut f32_names = vec![names[3].clone()];
    let mut bf16_names = names[..3].to_vec();
    weights.validate(&names[0], &[config.vocabulary_size, config.hidden_size])?;
    for name in &names[1..3] {
        weights.validate(name, &[8200, config.hidden_size])?;
    }
    weights.validate(&names[3], &[config.hidden_size])?;
    for spec in specs
        .iter()
        .filter(|spec| spec.name.starts_with("model.layers."))
    {
        let name = spec.name.replacen("model.", "llm.", 1);
        weights.validate(&name, &spec.shape)?;
        names.push(name.clone());
        match spec.dtype {
            WeightDtype::F32 => f32_names.push(name),
            WeightDtype::Bf16 | WeightDtype::F16 => bf16_names.push(name),
        }
    }
    ensure!(
        names.len() == 4 + config.num_layers * 11,
        "invalid Qwen3 layer bindings"
    );
    Ok((names, f32_names, bf16_names))
}

fn prepare_quantizer(
    runtime: &Runtime,
    weights: &Weights,
    suffix: &str,
    shape: &[usize],
) -> Result<MemOwn> {
    let mut values = Vec::with_capacity(8 * shape.iter().product::<usize>());
    for codebook in 0..8 {
        let name = format!("quantizer.quantizers.{codebook}.{suffix}");
        weights.validate(&name, shape)?;
        values.extend(weights.host_f32(&name)?);
    }
    Ok(runtime.mem_f32(&values)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use catena_lang::runtime::GpuDialect;

    #[test]
    #[ignore = "compiles the full OmniVoice graph for HIP and CUDA"]
    fn graph_compiles_for_both_gpu_dialects() -> Result<()> {
        for dialect in [GpuDialect::Hip, GpuDialect::Cuda] {
            let (source, _) = graph::build(&ModelConfig::qwen3_0_6b(), false, dialect);
            let module = catena::compile_sources(catena::runtime_sources(&source), dialect)?;
            ensure!(
                !module.source.contains("catena_host_read"),
                "unexpected model readback"
            );
        }
        Ok(())
    }

    fn compare_reference(output: &[u16], variable: &str, minimum: f32) -> Result<()> {
        let Some(path) = std::env::var_os(variable) else {
            return Ok(());
        };
        let bytes = std::fs::read(path)?;
        ensure!(
            bytes.len() == 4 * output.len(),
            "unexpected reference length"
        );
        let reference = bytes
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>();
        let wave: Vec<_> = output
            .iter()
            .map(|&x| crate::weights::bf16_to_f32(x))
            .collect();
        let dot = |a: &[f32], b: &[f32]| -> f32 { a.iter().zip(b).map(|(x, y)| x * y).sum() };
        let similarity =
            dot(&wave, &reference) / (dot(&wave, &wave) * dot(&reference, &reference)).sqrt();
        eprintln!("{variable} cosine similarity {similarity:.4}");
        ensure!(
            similarity > minimum,
            "GPU output diverged from the reference"
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires a GPU and the OmniVoice checkpoint"]
    fn gpu_reference() -> Result<()> {
        let directory = hfhub::resolve_model_files(DEFAULT_MODEL, FILES)?;
        let model = OmniVoice::load_snapshot(&directory, true)?;
        // Expected unmasking counts from the Python reference schedule.
        for (slots, expected) in [
            (536u64, &[536u64][..]),
            (536, &[9, 12, 17, 26, 45, 94, 333]),
            (6000, &[286, 715, 4999]),
        ] {
            let mut remaining = slots;
            for (step, &expected_take) in expected.iter().enumerate() {
                let [Value::U64(take)] = model.artifact.exec(
                    "omnivoice.schedule",
                    [
                        (step as u64).into(),
                        (expected.len() as u64).into(),
                        slots.into(),
                        remaining.into(),
                    ],
                )?
                else {
                    anyhow::bail!("invalid OmniVoice schedule output");
                };
                assert_eq!(take, expected_take);
                remaining -= take;
            }
            assert_eq!(remaining, 0);
        }
        let text = "This is a sentence without any voice prompt.";
        let ids = prompt::tokens(&model.tokenizer, text, None)?;
        assert_eq!(
            ids,
            [
                151670, 4064, 151671, 151672, 4064, 151673, 151674, 1986, 374, 264, 11652, 2041,
                894, 7743, 9934, 13, 151675
            ]
        );
        let conditioned = prompt::tokens(&model.tokenizer, text, Some(" excited and happy "))?;
        assert_eq!(
            &conditioned[..9],
            &[151670, 4064, 151671, 151672, 39441, 1608, 323, 6247, 151673]
        );
        assert_eq!(&conditioned[9..], &ids[6..]);
        assert_eq!(prompt::duration_frames(text), 67);
        let multilingual = "你好 （世界） [sigh]！";
        assert_eq!(
            prompt::tokens(&model.tokenizer, multilingual, None)?,
            [
                151670, 4064, 151671, 151672, 4064, 151673, 151674, 108386, 7, 99489, 8, 220,
                11880, 1090, 60, 6313, 151675
            ]
        );
        assert_eq!(prompt::duration_frames("你好，世界！"), 38);

        let frames = 3u64;
        let codes: Vec<u64> = (0..frames * 8).map(|i| (i * 37 + 13) % 1024).collect();
        let mut args = vec![model.runtime.mem_u64(&codes)?.into(), frames.into()];
        model.codec_args(&mut args)?;
        let wave = model.execute("omnivoice.decode", args)?.try_to_u16_vec()?;
        ensure!(
            wave.len() == frames as usize * FRAME_SAMPLES,
            "unexpected decoder length"
        );
        compare_reference(&wave, "OMNIVOICE_REFERENCE_DECODER", 0.95)?;

        let tokens = model.runtime.mem_u64(&ids)?;
        for (frames, reference) in [
            (3u64, "OMNIVOICE_REFERENCE_LOGITS"),
            (127, "OMNIVOICE_REFERENCE_LOGITS_127"),
            (128, "OMNIVOICE_REFERENCE_LOGITS_128"),
            (129, "OMNIVOICE_REFERENCE_WIDE_LOGITS"),
        ] {
            let mut masked = vec![1024; frames as usize * 8];
            masked[8..16].copy_from_slice(&codes[..8]);
            let mut args = vec![
                tokens.as_ref().into(),
                (ids.len() as u64).into(),
                frames.into(),
                model.runtime.mem_u64(&masked)?.into(),
            ];
            for name in &model.model_names {
                args.push(model.model.gpu_ref(name)?.into());
            }
            let logits = model.execute("omnivoice.logits", args)?.try_to_u16_vec()?;
            ensure!(
                logits.len() == frames as usize * 8200,
                "unexpected logit count"
            );
            compare_reference(&logits, reference, 0.98)?;
        }

        let codes = model.runtime.mem_u64(&[1024, 1024, 7, 1024, 1024])?;
        let scores = model
            .runtime
            .mem_f32(&[0.5, 3., 1., 5., f32::MIN, 6., 1., 9., -2., 4.])?;
        let selected = model
            .execute(
                "omnivoice.select",
                vec![codes.into(), scores.into(), 5u64.into(), 2u64.into()],
            )?
            .try_to_u64_vec()?;
        assert_eq!(selected, [1024, 5, 7, 9, 1024]);
        Ok(())
    }
}
