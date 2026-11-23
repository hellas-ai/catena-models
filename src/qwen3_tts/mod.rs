//! Qwen3-TTS 12 Hz preset-voice synthesis.
mod codec;
mod graph;
mod guard;
mod kernels;
mod prefill;
mod schedule;

use crate::{
    catena, hfhub, model,
    spec::ModelConfig,
    tokenizer::QwenTokenizer,
    tts::Audio,
    weights::{Weights, bf16},
};
use anyhow::{Context, Result, bail, ensure};
use catena_lang::runtime::{Artifact, Runtime, Value};
use serde::Deserialize;
use std::{collections::BTreeMap, path::Path};

pub const DEFAULT_MODEL: &str = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice";
pub const LARGE_MODEL: &str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice";
pub const DEFAULT_MAX_FRAMES: u32 = 512;
pub const MAX_FRAMES: u32 = 8192;
const FRAME_SAMPLES: usize = 1920;
const FILES: &[&str] = &[
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "vocab.json",
    "merges.txt",
    "tokenizer_config.json",
    "speech_tokenizer/config.json",
    "speech_tokenizer/model.safetensors",
];

pub struct SynthesisOptions<'a> {
    pub language: &'a str,
    pub speaker: &'a str,
    pub instruct: Option<&'a str>,
    pub max_frames: u32,
    pub seed: u64,
}

impl Default for SynthesisOptions<'_> {
    fn default() -> Self {
        Self {
            language: "English",
            speaker: "Vivian",
            instruct: None,
            max_frames: DEFAULT_MAX_FRAMES,
            seed: 0,
        }
    }
}

pub struct Synthesis {
    pub audio: Audio,
    pub frame_limit_reached: bool,
}

pub struct Qwen3Tts {
    artifact: Artifact,
    model: Weights,
    codec: Weights,
    model_names: Vec<Binding>,
    codec_names: Vec<Binding>,
    tokenizer: QwenTokenizer,
    config: Config,
    runtime: Runtime,
}

#[derive(Deserialize)]
struct Config {
    model_type: String,
    tts_model_type: String,
    tokenizer_type: String,
    tts_pad_token_id: u64,
    tts_bos_token_id: u64,
    tts_eos_token_id: u64,
    talker_config: TalkerConfig,
}

#[derive(Deserialize)]
struct TalkerConfig {
    #[serde(flatten)]
    decoder: DecoderConfig,
    code_predictor_config: DecoderConfig,
    num_code_groups: usize,
    text_hidden_size: usize,
    text_vocab_size: usize,
    codec_bos_id: u64,
    codec_eos_token_id: u64,
    codec_pad_id: u64,
    codec_think_id: u64,
    codec_nothink_id: u64,
    codec_think_bos_id: u64,
    codec_think_eos_id: u64,
    codec_language_id: BTreeMap<String, u64>,
    spk_id: BTreeMap<String, u64>,
    spk_is_dialect: BTreeMap<String, serde_json::Value>,
}

#[derive(Deserialize)]
struct DecoderConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    max_position_embeddings: usize,
    vocab_size: usize,
    rope_theta: f32,
    rms_norm_eps: f32,
}

impl DecoderConfig {
    fn model_config(&self) -> ModelConfig {
        let mut config = ModelConfig::qwen3_0_6b();
        config.hidden_size = self.hidden_size;
        config.intermediate_size = self.intermediate_size;
        config.num_layers = self.num_hidden_layers;
        config.num_attention_heads = self.num_attention_heads;
        config.num_key_value_heads = self.num_key_value_heads;
        config.head_dim = self.head_dim;
        config.rope_dimension_count = self.head_dim;
        config.max_position_embeddings = self.max_position_embeddings;
        config.vocabulary_size = self.vocab_size;
        config.eos_token_id = None;
        config.rope_theta = self.rope_theta;
        config.rms_norm_eps = self.rms_norm_eps;
        config
    }
}

#[derive(Deserialize)]
struct Sampling {
    do_sample: bool,
    repetition_penalty: f32,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    subtalker_dosample: bool,
    subtalker_temperature: f32,
    subtalker_top_k: usize,
    subtalker_top_p: f32,
}

#[derive(Clone, Copy, PartialEq)]
enum Storage {
    Bf16,
    F32,
    TransposeConv,
}

struct Binding {
    name: String,
    shape: Vec<usize>,
    storage: Storage,
}

impl Binding {
    fn new(name: impl Into<String>, shape: impl Into<Vec<usize>>, storage: Storage) -> Self {
        Self {
            name: name.into(),
            shape: shape.into(),
            storage,
        }
    }
}

impl Qwen3Tts {
    pub fn load(model_name: &str) -> Result<Self> {
        ensure!(
            matches!(model_name, DEFAULT_MODEL | LARGE_MODEL),
            "unsupported Qwen3-TTS checkpoint {model_name:?}"
        );
        let directory = hfhub::resolve_model_files(model_name, FILES)?;
        let (config, sampling) = read_config(&directory)?;
        let tokenizer = QwenTokenizer::from_qwen2_files(&directory)?;
        let dialect = model::configured_gpu_dialect()?;
        let (source, model_names, codec_names) = graph::build(&config, &sampling)?;
        let mut model = Weights::open(&directory.join("model.safetensors"))?;
        let mut codec = Weights::open(&directory.join("speech_tokenizer/model.safetensors"))?;
        for (weights, names) in [(&model, &model_names), (&codec, &codec_names)] {
            for binding in names {
                weights.validate(&binding.name, &binding.shape)?;
            }
        }
        let (runtime, artifact) = catena::load_runtime(catena::runtime_sources(&source), dialect)
            .context("compiling and loading Qwen3-TTS HEX")?;
        prepare(&mut model, &model_names, &runtime)?;
        prepare(&mut codec, &codec_names, &runtime)?;
        Ok(Self {
            artifact,
            model,
            codec,
            model_names,
            codec_names,
            tokenizer,
            config,
            runtime,
        })
    }

    pub fn synthesize(&self, text: &str, options: SynthesisOptions<'_>) -> Result<Synthesis> {
        ensure!(!text.trim().is_empty(), "text must contain a spoken prompt");
        ensure!(
            (1..=MAX_FRAMES).contains(&options.max_frames),
            "max_frames must be between 1 and {MAX_FRAMES}"
        );
        let (text, instruction, codec) = self.config.prompt(&self.tokenizer, text, &options)?;
        let talker = &self.config.talker_config;
        let prompt_len = instruction.len() + text.len() + codec.len() + 2;
        ensure!(
            prompt_len + options.max_frames as usize <= talker.decoder.max_position_embeddings,
            "Qwen3-TTS prompt and frame budget exceed the checkpoint context"
        );
        ensure!(
            text.iter()
                .chain(&instruction)
                .all(|&id| id < talker.text_vocab_size as u64),
            "Qwen3-TTS input exceeds the checkpoint vocabulary"
        );
        let text_len = text.len() as u64;
        let instruction_len = instruction.len() as u64;
        let codec_len = codec.len() as u64;
        let text = self.runtime.mem_u64(&text)?;
        // Keep a valid device pointer for an empty instruction.
        let instruction = self.runtime.mem_u64(if instruction.is_empty() {
            &[0]
        } else {
            &instruction
        })?;
        let codec = self.runtime.mem_u64(&codec)?;
        let mut args = vec![
            text.as_ref().into(),
            text_len.into(),
            instruction.as_ref().into(),
            instruction_len.into(),
            codec.as_ref().into(),
            codec_len.into(),
            options.max_frames.into(),
        ];
        if schedule::supported() {
            for binding in &self.model_names {
                args.push(self.model.gpu_ref(&binding.name)?.into());
            }
            args.push(options.seed.into());
            for binding in &self.codec_names {
                args.push(self.codec.gpu_ref(&binding.name)?.into());
            }
            let result = self
                .artifact
                .exec_values("qwen3-tts.synthesize", args)
                .context("executing Qwen3-TTS HEX")?;
            let [Value::MemOwn(pcm), Value::MemOwn(progress)] = result.as_slice() else {
                bail!("invalid Qwen3-TTS graph outputs");
            };
            let progress = progress.try_to_u64_vec()?;
            let [frames, stopped] = progress.as_slice() else {
                bail!("invalid Qwen3-TTS progress");
            };
            ensure!(
                *frames <= u64::from(options.max_frames),
                "invalid Qwen3-TTS frame count"
            );
            return waveform(pcm, *frames, *stopped);
        }

        let model_weights = self
            .model_names
            .iter()
            .map(|binding| self.model.gpu_ref(&binding.name))
            .collect::<Result<Vec<_>>>()?;
        args.extend(model_weights.iter().copied().map(Value::from));
        let mut state = self
            .artifact
            .exec_values("qwen3-tts.prepare", args)
            .context("preparing Qwen3-TTS generation")?;

        // Keep caches, activations and codes on device across calls. The small
        // descriptor lets the host stop submitting work once HEX detects EOS.
        let capacity = prompt_len as u64 + u64::from(options.max_frames);
        let mut start = 0;
        let (frames, stopped) = loop {
            let steps = (options.max_frames - start).min(schedule::FRAMES_PER_CALL as u32);
            let mut args = vec![
                u64::from(start).into(),
                steps.into(),
                capacity.into(),
                (prompt_len as u64).into(),
                options.seed.into(),
            ];
            args.extend(state);
            args.extend(model_weights.iter().copied().map(Value::from));
            state = self
                .artifact
                .exec_values("qwen3-tts.advance", args)
                .context("generating Qwen3-TTS frames")?;
            let [_, _, _, _, _, Value::MemOwn(progress), _] = state.as_slice() else {
                bail!("invalid Qwen3-TTS generation state");
            };
            let progress = progress.try_to_u64_vec()?;
            let [frames, stopped] = progress.as_slice() else {
                bail!("invalid Qwen3-TTS progress");
            };
            let end = start + steps;
            ensure!(
                (u64::from(start)..=u64::from(end)).contains(frames)
                    && *stopped <= 1
                    && (*stopped == 1 || *frames == u64::from(end)),
                "invalid Qwen3-TTS frame count"
            );
            if *stopped == 1 || end == options.max_frames {
                break (*frames, *stopped);
            }
            start = end;
        };

        let Ok(
            [
                cache,
                hidden,
                predictor_cache,
                sum,
                codes @ Value::MemOwn(_),
                progress @ Value::MemOwn(_),
                pad,
            ],
        ) = <[_; 7]>::try_from(state)
        else {
            bail!("invalid Qwen3-TTS codec inputs");
        };
        drop([cache, hidden, predictor_cache, sum, pad]);
        ensure!(frames > 0, "Qwen3-TTS returned no speech frames");
        let mut args = vec![codes, progress, frames.into()];
        for binding in &self.codec_names {
            args.push(self.codec.gpu_ref(&binding.name)?.into());
        }
        let results = self
            .artifact
            .exec_values("qwen3-tts.decode", args)
            .context("decoding Qwen3-TTS waveform")?;
        let [Value::MemOwn(pcm)] = results.as_slice() else {
            bail!("invalid Qwen3-TTS decoder output");
        };
        waveform(pcm, frames, stopped)
    }
}

fn waveform(pcm: &catena_lang::runtime::MemOwn, frames: u64, stopped: u64) -> Result<Synthesis> {
    let samples = pcm.try_to_f32_vec()?;
    ensure!(
        frames > 0
            && stopped <= 1
            && samples.len() == frames as usize * FRAME_SAMPLES
            && samples.iter().all(|x| x.is_finite()),
        "invalid Qwen3-TTS waveform"
    );
    Ok(Synthesis {
        audio: Audio {
            samples,
            sample_rate: 24_000,
        },
        frame_limit_reached: stopped == 0,
    })
}

impl Config {
    fn prompt(
        &self,
        tokenizer: &QwenTokenizer,
        text: &str,
        options: &SynthesisOptions<'_>,
    ) -> Result<(Vec<u64>, Vec<u64>, Vec<u64>)> {
        let cfg = &self.talker_config;
        let speaker = options.speaker.to_ascii_lowercase();
        let speaker_id = cfg
            .spk_id
            .get(&speaker)
            .with_context(|| format!("unknown Qwen3-TTS speaker {:?}", options.speaker))?;
        let language = options.language.to_ascii_lowercase();
        let mut language_id = if language == "auto" {
            None
        } else {
            Some(
                *cfg.codec_language_id
                    .get(&language)
                    .filter(|_| !language.ends_with("_dialect"))
                    .with_context(|| {
                        format!("unknown Qwen3-TTS language {:?}", options.language)
                    })?,
            )
        };
        if matches!(language.as_str(), "auto" | "chinese") {
            if let Some(dialect) = cfg
                .spk_is_dialect
                .get(&speaker)
                .and_then(serde_json::Value::as_str)
            {
                language_id = Some(
                    *cfg.codec_language_id
                        .get(dialect)
                        .context("missing Qwen3-TTS speaker dialect")?,
                );
            }
        }
        let mut codec = vec![
            if language_id.is_some() {
                cfg.codec_think_id
            } else {
                cfg.codec_nothink_id
            },
            cfg.codec_think_bos_id,
        ];
        codec.extend(language_id);
        codec.extend([cfg.codec_think_eos_id, *speaker_id, cfg.codec_pad_id]);

        let text = tokenizer.encode_without_special_tokens(&format!(
            "<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        ))?;
        ensure!(text.len() > 8, "text must contain a spoken prompt");
        let text = text[..text.len() - 5].to_vec();
        let instruction = options
            .instruct
            .filter(|s| !s.trim().is_empty())
            .map(|s| {
                tokenizer
                    .encode_without_special_tokens(&format!("<|im_start|>user\n{s}<|im_end|>\n"))
            })
            .transpose()?
            .unwrap_or_default();
        Ok((text, instruction, codec))
    }
}

fn read_config(directory: &Path) -> Result<(Config, Sampling)> {
    let config: Config = hfhub::read_json(directory.join("config.json"))?;
    let sampling: Sampling = hfhub::read_json(directory.join("generation_config.json"))?;
    let codec: serde_json::Value =
        hfhub::read_json(directory.join("speech_tokenizer/config.json"))?;
    let talker = &config.talker_config;
    ensure!(
        config.model_type == "qwen3_tts"
            && config.tts_model_type == "custom_voice"
            && config.tokenizer_type == "qwen3_tts_tokenizer_12hz"
            && talker.num_code_groups == 16
            && talker.decoder.vocab_size == 3072
            && talker.code_predictor_config.vocab_size == 2048,
        "unsupported Qwen3-TTS model configuration"
    );
    for decoder in [&talker.decoder, &talker.code_predictor_config] {
        ensure!(
            (
                decoder.num_attention_heads,
                decoder.num_key_value_heads,
                decoder.head_dim
            ) == (16, 8, 128)
                && decoder.rope_theta == 1_000_000.0
                && decoder.rms_norm_eps == 1e-6,
            "unsupported Qwen3-TTS attention configuration"
        );
    }
    codec::validate(&codec)?;
    ensure!(
        sampling.do_sample
            && sampling.subtalker_dosample
            && sampling.top_p == 1.0
            && sampling.subtalker_top_p == 1.0
            && sampling.top_k > 0
            && sampling.top_k <= 2048
            && sampling.subtalker_top_k > 0
            && sampling.subtalker_top_k <= 2048
            && sampling.temperature > 0.0
            && sampling.subtalker_temperature > 0.0
            && sampling.repetition_penalty >= 1.0,
        "unsupported Qwen3-TTS sampling configuration"
    );
    Ok((config, sampling))
}

fn prepare(weights: &mut Weights, bindings: &[Binding], runtime: &Runtime) -> Result<()> {
    for (storage, f32_storage) in [(Storage::F32, true), (Storage::Bf16, false)] {
        let names = bindings
            .iter()
            .filter(|binding| binding.storage == storage)
            .map(|binding| binding.name.clone())
            .collect::<Vec<_>>();
        weights.prepare(runtime, &names, f32_storage)?;
    }
    for binding in bindings
        .iter()
        .filter(|binding| matches!(binding.storage, Storage::TransposeConv))
    {
        let [input, output, taps] = *binding.shape.as_slice() else {
            unreachable!()
        };
        let source = weights.host_f32(&binding.name)?;
        let mut packed = Vec::with_capacity(source.len());
        for co in 0..output {
            for ci in 0..input {
                let offset = (ci * output + co) * taps;
                packed.extend_from_slice(&source[offset..offset + taps]);
            }
        }
        weights.insert_prepared(&binding.name, runtime.mem_u16(&bf16(&packed))?)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use catena_lang::runtime::GpuDialect;

    #[test]
    #[ignore = "requires both Qwen3-TTS checkpoints in the Hugging Face cache"]
    fn custom_voice_checkpoints() -> Result<()> {
        for id in [DEFAULT_MODEL, LARGE_MODEL] {
            let directory = hfhub::resolve_model_files(id, FILES)?;
            let (config, sampling) = read_config(&directory)?;
            let tokenizer = QwenTokenizer::from_qwen2_files(&directory)?;
            let options = SynthesisOptions {
                instruct: Some("Tired, warm voice"),
                ..SynthesisOptions::default()
            };
            let (text, instruct, codec) = config.prompt(&tokenizer, "Hello world", &options)?;
            assert_eq!(text, [151644, 77091, 198, 9707, 1879]);
            assert_eq!(
                instruct,
                [151644, 872, 198, 51, 2690, 11, 8205, 7743, 151645, 198]
            );
            assert_eq!(codec, [2154, 2156, 2050, 2157, 3065, 2148]);
            let (source, model, codec) = graph::build(&config, &sampling)?;
            for (path, weights) in [
                ("model.safetensors", model),
                ("speech_tokenizer/model.safetensors", codec),
            ] {
                let tensors = Weights::open(&directory.join(path))?;
                for binding in weights {
                    tensors.validate(&binding.name, &binding.shape)?;
                }
            }
            for dialect in [GpuDialect::Hip, GpuDialect::Cuda] {
                let module = catena::compile_sources(catena::runtime_sources(&source), dialect)
                    .with_context(|| format!("{id} on {dialect:?}"))?;
                let entries = module
                    .functions
                    .iter()
                    .map(|entry| entry.source_name.as_str())
                    .collect::<std::collections::BTreeSet<_>>();
                let expected = if schedule::supported() {
                    ["qwen3-tts.synthesize"].into_iter().collect()
                } else {
                    ["qwen3-tts.prepare", "qwen3-tts.advance", "qwen3-tts.decode"]
                        .into_iter()
                        .collect()
                };
                assert_eq!(entries, expected);
            }
        }
        Ok(())
    }
}
