//! English MMS/VITS speech synthesis through one Catena entrypoint.
pub mod graph;
pub mod tokenizer;
use crate::{
    hfhub,
    weights::{Weights, bf16},
};
use anyhow::{Context, Result, bail, ensure};
use catena_lang::runtime::{Artifact, MemOwn, Runtime, Value};
use serde::Deserialize;
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    time::Instant,
};

const FILES: &[&str] = &[
    "config.json",
    "vocab.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "model.safetensors",
];
pub const SAMPLE_RATE: u32 = 16000;
pub const MAX_TOKENS: usize = 512;
pub const MAX_SAMPLES: usize = 480000;
pub fn snapshot(model: &str) -> Result<PathBuf> {
    hfhub::resolve_model_files(model, FILES)
}
#[derive(Debug)]
pub struct Config(serde_json::Value);
impl Config {
    pub fn read(dir: &Path) -> Result<Self> {
        let c = Self(hfhub::read_json(dir.join("config.json"))?);
        c.validate()?;
        Ok(c)
    }
    fn expected() -> serde_json::Value {
        serde_json::json!({
            "model_type":"vits", "hidden_size":192, "flow_size":192, "vocab_size":38,
            "num_hidden_layers":6, "num_attention_heads":2, "ffn_dim":768,
            "ffn_kernel_size":3, "hidden_act":"relu", "window_size":4,
            "num_speakers":1, "speaker_embedding_size":0, "sampling_rate":16000,
            "use_stochastic_duration_prediction":true, "duration_predictor_num_flows":4,
            "duration_predictor_flow_bins":10, "duration_predictor_tail_bound":5.0,
            "duration_predictor_kernel_size":3, "depth_separable_channels":2,
            "depth_separable_num_layers":3, "prior_encoder_num_flows":4,
            "prior_encoder_num_wavenet_layers":4, "wavenet_kernel_size":5,
            "wavenet_dilation_rate":1, "layer_norm_eps":1e-5, "use_bias":true,
            "upsample_initial_channel":512, "upsample_rates":[8,8,2,2],
            "upsample_kernel_sizes":[16,16,4,4], "resblock_kernel_sizes":[3,7,11],
            "resblock_dilation_sizes":[[1,3,5],[1,3,5],[1,3,5]], "leaky_relu_slope":0.1
        })
    }
    pub fn validate(&self) -> Result<()> {
        let expected = Self::expected();
        for (key, value) in expected.as_object().unwrap() {
            ensure!(
                self.0.get(key) == Some(value),
                "unsupported MMS English config {key}: expected {value}, got {:?}",
                self.0.get(key)
            );
        }
        Ok(())
    }
}
#[derive(Clone, Debug)]
pub struct Options {
    pub seed: u64,
    pub speaking_rate: f32,
    pub noise_scale: f32,
    pub noise_scale_duration: f32,
    pub timings: bool,
    pub dump_dir: Option<PathBuf>,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            seed: 0,
            speaking_rate: 1.,
            noise_scale: 0.667,
            noise_scale_duration: 0.8,
            timings: false,
            dump_dir: None,
        }
    }
}
impl Options {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.speaking_rate.is_finite() && self.speaking_rate > 0.,
            "speaking rate must be finite and positive"
        );
        for (name, value) in [
            ("noise scale", self.noise_scale),
            ("duration noise scale", self.noise_scale_duration),
        ] {
            ensure!(
                value.is_finite() && value >= 0.,
                "{name} must be finite and nonnegative"
            );
        }
        Ok(())
    }
}
#[derive(Clone, Debug)]
pub struct Audio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}
impl Audio {
    pub fn write_wav(&self, path: &Path) -> Result<()> {
        ensure!(
            !self.samples.is_empty()
                && self.samples.len() <= MAX_SAMPLES
                && self.samples.iter().all(|x| x.is_finite()),
            "invalid synthesized waveform"
        );
        let mut wav = hound::WavWriter::create(
            path,
            hound::WavSpec {
                channels: 1,
                sample_rate: self.sample_rate,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
        )?;
        for &x in &self.samples {
            wav.write_sample((x.clamp(-1., 1.) * 32767.).round() as i16)?;
        }
        wav.finalize().context("writing synthesized WAV")
    }
}
#[derive(Debug, Deserialize)]
pub struct Binding {
    pub shape: Vec<usize>,
    pub normalized: bool,
    pub transpose: bool,
}
pub fn bindings() -> Vec<(String, Binding)> {
    serde_json::from_str(include_str!("weights.json")).expect("generated TTS weight manifest")
}
pub fn validate_weights(weights: &Weights, config: &Config) -> Result<()> {
    config.validate()?;
    for (name, binding) in bindings() {
        if binding.normalized {
            let prefix = name.strip_suffix(".weight").unwrap();
            weights.validate(&format!("{prefix}.weight_v"), &binding.shape)?;
            weights.validate(&format!("{prefix}.weight_g"), &[binding.shape[0], 1, 1])?;
        } else {
            weights.validate(&name, &binding.shape)?;
        }
    }
    Ok(())
}
/// Materialize checkpoint weight normalization before BF16 rounding. Transpose
/// convolution weights are rearranged to the ordinary [out,in,k] GEMM layout.
pub fn materialize(weights: &Weights, name: &str, binding: &Binding) -> Result<Vec<f32>> {
    let mut values = if binding.normalized {
        let prefix = name.strip_suffix(".weight").unwrap();
        let v = weights.host_f32(&format!("{prefix}.weight_v"))?;
        let g = weights.host_f32(&format!("{prefix}.weight_g"))?;
        let terms = binding.shape[1] * binding.shape[2];
        let mut out = v;
        for (row, &gain) in out.chunks_exact_mut(terms).zip(&g) {
            let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            ensure!(
                norm.is_finite() && norm > 0.,
                "invalid normalized weight {name}"
            );
            for x in row {
                *x *= gain / norm;
            }
        }
        out
    } else {
        weights.host_f32(name)?
    };
    ensure!(
        values.iter().all(|x| x.is_finite()),
        "non-finite weight {name}"
    );
    if binding.transpose {
        let (ci, co, k) = (binding.shape[0], binding.shape[1], binding.shape[2]);
        let mut out = vec![0.; values.len()];
        for i in 0..ci {
            for o in 0..co {
                for tap in 0..k {
                    out[(o * ci + i) * k + tap] = values[(i * co + o) * k + tap];
                }
            }
        }
        values = out;
    }
    Ok(values)
}
pub struct Tts {
    runtime: Runtime,
    artifact: Artifact,
    checkpoint: Weights,
    weights: BTreeMap<String, MemOwn>,
    tokenizer: tokenizer::Tokenizer,
}
impl Tts {
    pub fn load(dir: &Path, timings: bool) -> Result<Self> {
        let start = Instant::now();
        let config = Config::read(dir)?;
        let mut checkpoint = Weights::open(&dir.join("model.safetensors"))?;
        validate_weights(&checkpoint, &config)?;
        let tokenizer = tokenizer::Tokenizer::read(dir)?;
        let source = graph::bundle(&config)?;
        let (runtime, artifact) = crate::catena::load_runtime(
            crate::catena::core_sources()
                .map(|s| -> &str { s })
                .chain([source.as_str()]),
            crate::model::configured_gpu_dialect()?,
        )
        .context("compiling MMS TTS Hex")?;
        let ordinary = bindings()
            .into_iter()
            .filter(|(_, b)| !b.normalized && !b.transpose)
            .map(|(n, _)| n)
            .collect::<Vec<_>>();
        checkpoint.prepare(&runtime, &ordinary, false)?;
        let mut weights = BTreeMap::new();
        for (name, binding) in bindings()
            .into_iter()
            .filter(|(_, b)| b.normalized || b.transpose)
        {
            weights.insert(
                name.clone(),
                runtime.mem_u16(&bf16(&materialize(&checkpoint, &name, &binding)?))?,
            );
        }
        if timings {
            eprintln!(
                "load/compile/weight preparation: {:.3}s",
                start.elapsed().as_secs_f64()
            );
        }
        Ok(Self {
            runtime,
            artifact,
            checkpoint,
            weights,
            tokenizer,
        })
    }
    pub fn synthesize(&self, text: &str, options: &Options) -> Result<Audio> {
        options.validate()?;
        let ids = self.tokenizer.encode(text)?;
        let input = self.runtime.mem_u64(&ids)?;
        let inputs = || -> Result<Vec<Value<'_>>> {
            let mut args = vec![
                input.as_ref().into(),
                (ids.len() as u64).into(),
                (options.seed as u32).into(),
                ((options.seed >> 32) as u32).into(),
                options.noise_scale_duration.into(),
                options.noise_scale.into(),
                options.speaking_rate.into(),
            ];
            for (name, _) in bindings() {
                args.push(if let Some(weight) = self.weights.get(&name) {
                    weight.as_ref().into()
                } else {
                    self.checkpoint.gpu_ref(&name)?.into()
                });
            }
            Ok(args)
        };
        let start = Instant::now();
        let mut prepared = self.artifact.exec_values("tts.prepare", inputs()?)?;
        ensure!(prepared.len() == 5, "invalid TTS preparation ABI");
        let header = memory(prepared.pop())?.try_to_f32_vec()?;
        let frames = checked_frame_count(&header)?;
        let mut args = inputs()?;
        args.extend(prepared);
        args.push((frames as u64).into());
        let debug = options.dump_dir.is_some();
        let outputs = self.artifact.exec_values(
            if debug {
                "tts.synthesize-debug"
            } else {
                "tts.synthesize"
            },
            args,
        )?;
        let elapsed = start.elapsed().as_secs_f64();
        let mut outputs = outputs.into_iter();
        let waveform = memory(outputs.next())?;
        match outputs.next() {
            Some(Value::U64(0)) => {}
            Some(Value::U64(_)) => {
                bail!("predicted audio exceeds 30 seconds or has invalid durations")
            }
            _ => bail!("invalid TTS status output"),
        }
        let samples = waveform.try_to_f32_vec()?;
        ensure!(
            !samples.is_empty()
                && samples.len() <= MAX_SAMPLES
                && samples.len() % 256 == 0
                && samples.iter().all(|x| x.is_finite()),
            "invalid TTS waveform"
        );
        if let Some(dir) = &options.dump_dir {
            std::fs::create_dir_all(dir)?;
            std::fs::write(dir.join("tokens.json"), serde_json::to_vec(&ids)?)?;
            std::fs::write(
                dir.join("options.json"),
                serde_json::to_vec(
                    &serde_json::json!({"seed":options.seed,"speaking_rate":options.speaking_rate,"noise_scale":options.noise_scale,"noise_scale_duration":options.noise_scale_duration}),
                )?,
            )?;
            let frames = samples.len() / 256;
            for (name, dtype, shape) in [
                ("durations", "f32", vec![ids.len()]),
                ("stats", "bf16", vec![ids.len(), 384]),
                ("duration_noise", "f32", vec![ids.len(), 2]),
                ("latent_noise", "f32", vec![192, frames]),
                ("flow_output", "bf16", vec![frames, 192]),
            ] {
                let mem = memory(outputs.next())?;
                let values = if dtype == "bf16" {
                    mem.try_to_u16_vec()?
                        .into_iter()
                        .map(|x| f32::from_bits((x as u32) << 16))
                        .collect()
                } else {
                    mem.try_to_f32_vec()?
                };
                dump(dir, name, &values, &shape)?;
            }
            dump(dir, "waveform", &samples, &[samples.len()])?;
        }
        if options.timings {
            let duration = samples.len() as f64 / SAMPLE_RATE as f64;
            eprintln!(
                "inference: {elapsed:.3}s; audio: {duration:.3}s; real-time factor: {:.3}",
                elapsed / duration
            );
        }
        Ok(Audio {
            samples,
            sample_rate: SAMPLE_RATE,
        })
    }
}
fn memory(value: Option<Value<'_>>) -> Result<MemOwn> {
    match value {
        Some(Value::MemOwn(m)) => Ok(m),
        _ => bail!("TTS returned invalid memory output"),
    }
}
fn dump(dir: &Path, name: &str, values: &[f32], shape: &[usize]) -> Result<()> {
    ensure!(
        shape.iter().product::<usize>() == values.len(),
        "invalid dump shape for {name}"
    );
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(dir.join(format!("{name}.f32")), bytes)?;
    std::fs::write(
        dir.join(format!("{name}.shape.json")),
        serde_json::to_vec(shape)?,
    )?;
    Ok(())
}

fn checked_frame_count(header: &[f32]) -> Result<usize> {
    ensure!(header.len() == 1, "invalid TTS length header");
    let frames = header[0];
    ensure!(
        frames.is_finite() && frames >= 1. && frames <= 1875. && frames.fract() == 0.,
        "predicted audio exceeds 30 seconds or has invalid durations"
    );
    Ok(frames as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn checks_device_predicted_length_before_decoder_allocation() {
        assert_eq!(checked_frame_count(&[1.]).unwrap(), 1);
        assert_eq!(checked_frame_count(&[1875.]).unwrap(), 1875);
        for value in [0., -1., 1.5, 1876., f32::NAN, f32::INFINITY] {
            assert!(checked_frame_count(&[value]).is_err());
        }
        assert!(checked_frame_count(&[]).is_err());
        assert!(checked_frame_count(&[1., 2.]).is_err());
    }
}
