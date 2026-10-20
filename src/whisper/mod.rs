pub mod audio;
pub mod graph;
pub mod hub;
mod kernels;
use crate::weights::Weights;
use anyhow::{Context, Result, bail, ensure};
use catena_lang::{
    codegen::GpuDialect,
    runtime::{Artifact, MemOwn, Runtime, Value},
};
use serde::Deserialize;
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
    time::Instant,
};

#[derive(Debug, Deserialize)]
pub struct Config {
    pub model_type: String,
    pub d_model: usize,
    pub num_mel_bins: usize,
    pub encoder_layers: usize,
    pub decoder_layers: usize,
    pub encoder_attention_heads: usize,
    pub decoder_attention_heads: usize,
    pub encoder_ffn_dim: usize,
    pub decoder_ffn_dim: usize,
    pub max_source_positions: usize,
    pub max_target_positions: usize,
    pub vocab_size: usize,
    pub decoder_start_token_id: u32,
    pub eos_token_id: u32,
    pub activation_function: String,
    #[serde(default)]
    pub scale_embedding: bool,
}
impl Config {
    pub fn read(dir: &Path) -> Result<Self> {
        let c: Self = read_json(&dir.join("config.json"))?;
        c.validate()?;
        Ok(c)
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.model_type == "whisper"
                && self.d_model > 0
                && self.d_model.is_multiple_of(2)
                && self.num_mel_bins == 80
                && self.encoder_layers > 0
                && self.decoder_layers > 0
                && self.encoder_attention_heads > 0
                && self.encoder_attention_heads == self.decoder_attention_heads
                && self.d_model.is_multiple_of(self.encoder_attention_heads)
                && self.encoder_ffn_dim > 0
                && self.encoder_ffn_dim.is_multiple_of(2)
                && self.encoder_ffn_dim == self.decoder_ffn_dim
                && self.max_source_positions == 1500
                && self.max_target_positions > 4
                && self.vocab_size > 0
                && self.activation_function == "gelu"
                && !self.scale_embedding,
            "expected an 80-mel Whisper model with matching encoder/decoder head and FFN dimensions, GELU, and a 1500-position encoder (128-mel large-v3 models are unsupported)"
        );
        ensure!(
            (self.decoder_start_token_id as usize) < self.vocab_size
                && (self.eos_token_id as usize) < self.vocab_size,
            "invalid special token IDs"
        );
        Ok(())
    }
}
use crate::hfhub::read_json;
#[derive(Deserialize)]
struct Generation {
    lang_to_id: BTreeMap<String, u32>,
    task_to_id: BTreeMap<String, u32>,
    no_timestamps_token_id: u32,
    #[serde(default)]
    suppress_tokens: Vec<u32>,
    #[serde(default)]
    begin_suppress_tokens: Vec<u32>,
}
impl Generation {
    // Non-learned token metadata becomes GPU masks; decisions stay in Hex.
    fn masks(&self, vocab: usize, eos: u32) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut language = vec![f32::NEG_INFINITY; vocab];
        for &id in self.lang_to_id.values() {
            language[id as usize] = 0.;
        }
        let mut text = vec![0.; vocab];
        for v in text.iter_mut().skip(eos as usize + 1) {
            *v = f32::NEG_INFINITY;
        }
        for &id in &self.suppress_tokens {
            text[id as usize] = f32::NEG_INFINITY;
        }
        let mut begin = vec![0.; vocab];
        for &id in &self.begin_suppress_tokens {
            begin[id as usize] = f32::NEG_INFINITY;
        }
        (language, text, begin)
    }
}

pub use crate::weights::bf16;
pub fn host(mem: &MemOwn) -> Result<Vec<f32>> {
    Ok(mem
        .try_to_u16_vec()?
        .into_iter()
        .map(|x| f32::from_bits((x as u32) << 16))
        .collect())
}
fn memories(values: Vec<Value<'_>>) -> Result<Vec<MemOwn>> {
    values
        .into_iter()
        .map(|v| match v {
            Value::MemOwn(m) => Ok(m),
            _ => bail!("Whisper graph returned a non-owned value"),
        })
        .collect()
}

pub fn validate_weights(w: &Weights, c: &Config) -> Result<()> {
    let d = c.d_model;
    let ff = c.encoder_ffn_dim;
    for (n, shape) in [
        ("model.encoder.conv1.weight", vec![d, c.num_mel_bins, 3]),
        ("model.encoder.conv1.bias", vec![d]),
        ("model.encoder.conv2.weight", vec![d, d, 3]),
        ("model.encoder.conv2.bias", vec![d]),
        (
            "model.encoder.embed_positions.weight",
            vec![c.max_source_positions, d],
        ),
        ("model.decoder.embed_tokens.weight", vec![c.vocab_size, d]),
        (
            "model.decoder.embed_positions.weight",
            vec![c.max_target_positions, d],
        ),
    ] {
        w.validate(n, &shape)?;
    }
    for kind in ["encoder", "decoder"] {
        for suffix in ["weight", "bias"] {
            w.validate(&format!("model.{kind}.layer_norm.{suffix}"), &[d])?;
        }
        let layers = if kind == "encoder" {
            c.encoder_layers
        } else {
            c.decoder_layers
        };
        for l in 0..layers {
            let mut names = graph::layer_names(kind, l, graph::ATTN);
            names.extend(graph::layer_names(kind, l, graph::FF));
            if kind == "decoder" {
                names.extend(graph::layer_names(kind, l, graph::CROSS));
                names.extend(graph::layer_names(
                    kind,
                    l,
                    &[
                        "encoder_attn.k_proj.weight",
                        "encoder_attn.v_proj.weight",
                        "encoder_attn.v_proj.bias",
                    ],
                ));
            }
            for name in names {
                let shape = if name.ends_with("fc1.weight") {
                    vec![ff, d]
                } else if name.ends_with("fc2.weight") {
                    vec![d, ff]
                } else if name.ends_with("fc1.bias") {
                    vec![ff]
                } else if name.ends_with("bias") || name.contains("layer_norm") {
                    vec![d]
                } else {
                    vec![d, d]
                };
                w.validate(&name, &shape)?;
            }
        }
    }
    if w.tensors.contains_key("proj_out.weight") {
        w.validate("proj_out.weight", &[c.vocab_size, d])?;
        ensure!(
            w.host_f32("proj_out.weight")? == w.host_f32("model.decoder.embed_tokens.weight")?,
            "untied output weights are unsupported"
        );
    }
    Ok(())
}
pub struct Options {
    pub language: Option<String>,
    pub max_new_tokens: usize,
    pub timings: bool,
    pub dump_dir: Option<PathBuf>,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            language: None,
            max_new_tokens: 256,
            timings: false,
            dump_dir: None,
        }
    }
}
pub struct Whisper {
    pub runtime: Runtime,
    pub artifact: Artifact,
    pub config: Config,
    weights: Weights,
    tokenizer: tokenizers::Tokenizer,
    generation: Generation,
    debug: bool,
}
impl Whisper {
    pub fn load(dir: &Path, timings: bool) -> Result<Self> {
        Self::load_with_debug(dir, timings, false)
    }
    pub fn load_with_debug(dir: &Path, timings: bool, debug: bool) -> Result<Self> {
        let start = Instant::now();
        let config = Config::read(dir)?;
        let mut checkpoint = Weights::open(&dir.join("model.safetensors"))?;
        validate_weights(&checkpoint, &config)?;
        let generation: Generation = read_json(&dir.join("generation_config.json"))?;
        ensure!(
            !generation.lang_to_id.is_empty(),
            "checkpoint has no language tokens"
        );
        ensure!(
            generation
                .lang_to_id
                .values()
                .chain(generation.task_to_id.values())
                .chain(generation.suppress_tokens.iter())
                .chain(generation.begin_suppress_tokens.iter())
                .chain(std::iter::once(&generation.no_timestamps_token_id))
                .all(|id| (*id as usize) < config.vocab_size),
            "generation config contains out-of-range tokens"
        );
        ensure!(
            generation.task_to_id.contains_key("transcribe"),
            "missing transcribe task token"
        );
        let tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        ensure!(
            tokenizer.get_vocab_size(true) == config.vocab_size,
            "tokenizer vocabulary does not match model"
        );
        let source = graph::bundle(&config, debug)?;
        let (runtime, artifact) =
            crate::catena::load_runtime(crate::catena::runtime_sources(&source), GpuDialect::Hip)
                .context("compiling Whisper Hex graphs")?;
        let names: BTreeSet<_> = graph::encoder_names(&config)
            .into_iter()
            .chain(graph::cross_names(&config))
            .chain(graph::decoder_names(&config))
            .collect();
        checkpoint.prepare(&runtime, &names.into_iter().collect::<Vec<_>>(), false)?;
        if timings {
            eprintln!(
                "load/compile/weight preparation: {:.3}s",
                start.elapsed().as_secs_f64()
            );
        }
        Ok(Self {
            runtime,
            artifact,
            config,
            weights: checkpoint,
            tokenizer,
            generation,
            debug,
        })
    }
    fn weight_args<'a>(&'a self, names: &[String]) -> Result<Vec<Value<'a>>> {
        names
            .iter()
            .map(|n| self.weights.gpu_ref(n).map(Into::into))
            .collect::<Result<Vec<_>>>()
    }
    pub fn encode(&self, features: &[f32]) -> Result<MemOwn> {
        ensure!(
            features.len() == audio::FRAMES * audio::MELS,
            "wrong mel feature dimensions"
        );
        let mut args = vec![self.runtime.mem_u16(&bf16(features))?.into()];
        args.extend(self.weight_args(&graph::encoder_names(&self.config))?);
        let mut output = memories(self.artifact.exec_values("whisper.encode", args)?)?;
        ensure!(output.len() == 1, "wrong encoder output count");
        Ok(output.remove(0))
    }
    pub fn prepare_cross(&self, encoder: MemOwn) -> Result<Vec<MemOwn>> {
        let mut args = vec![encoder.into()];
        args.extend(self.weight_args(&graph::cross_names(&self.config))?);
        let out = memories(
            self.artifact
                .exec_values("whisper.prepare-cross-kv", args)?,
        )?;
        ensure!(
            out.len() == self.config.decoder_layers * 2,
            "wrong cross cache count"
        );
        Ok(out)
    }
    pub fn step(
        &self,
        token: u32,
        pos: usize,
        cache: Vec<MemOwn>,
        cross: &[MemOwn],
    ) -> Result<(Vec<f32>, Vec<MemOwn>)> {
        ensure!(
            (token as usize) < self.config.vocab_size && pos < self.config.max_target_positions,
            "invalid decoder token or position"
        );
        ensure!(
            cache.len() == self.config.decoder_layers * 2 && cross.len() == cache.len(),
            "wrong decoder cache count"
        );
        let mut args = vec![(token as u64).into(), (pos as u64).into()];
        for (l, owner) in cache.into_iter().enumerate() {
            args.push(owner.into());
            if l % 2 == 1 {
                args.push(cross[l - 1].as_ref().into());
                args.push(cross[l].as_ref().into());
            }
        }
        args.extend(self.weight_args(&graph::decoder_names(&self.config))?);
        let mut out = memories(self.artifact.exec_values("whisper.decode-step", args)?)?;
        ensure!(
            out.len() == 1 + self.config.decoder_layers * 2,
            "wrong decoder output count"
        );
        let logits = host(&out.remove(0))?;
        Ok((logits, out))
    }
    pub fn transcribe(&self, samples: &[f32], options: &Options) -> Result<String> {
        ensure!(
            options.max_new_tokens > 0,
            "--max-new-tokens must be positive"
        );
        let language_override = options
            .language
            .as_ref()
            .map(|lang| {
                self.generation
                    .lang_to_id
                    .get(&format!("<|{lang}|>"))
                    .copied()
                    .with_context(|| format!("unknown language code {lang}"))
            })
            .transpose()?;
        let start = Instant::now();
        let features = audio::log_mel(samples)?;
        let preprocessing = start.elapsed();
        dump_f32(
            options.dump_dir.as_deref(),
            "log_mel",
            &features,
            &[3000, 80],
        )?;
        let features = self.runtime.mem_f32(&features)?;
        let (language_mask, text_mask, begin_mask) = self
            .generation
            .masks(self.config.vocab_size, self.config.eos_token_id);
        let language_mask = self.runtime.mem_f32(&language_mask)?;
        let text_mask = self.runtime.mem_f32(&text_mask)?;
        let begin_mask = self.runtime.mem_f32(&begin_mask)?;
        let mut args = vec![
            features.as_ref().into(),
            (options.max_new_tokens as u64).into(),
            (language_override.unwrap_or(0) as u64).into(),
            (self.config.decoder_start_token_id as u64).into(),
            (self.generation.task_to_id["transcribe"] as u64).into(),
            (self.generation.no_timestamps_token_id as u64).into(),
            (self.config.eos_token_id as u64).into(),
            language_mask.as_ref().into(),
            text_mask.as_ref().into(),
            begin_mask.as_ref().into(),
        ];
        args.extend(self.weight_args(&graph::encoder_names(&self.config))?);
        args.extend(self.weight_args(&graph::cross_names(&self.config))?);
        args.extend(self.weight_args(&graph::decoder_names(&self.config))?);
        let start = Instant::now();
        let outputs = self.artifact.exec_values("whisper.transcribe", args)?;
        let mut outputs = outputs.into_iter();
        let Some(Value::MemOwn(tokens)) = outputs.next() else {
            bail!("missing generated token buffer")
        };
        let result = tokens.try_to_u64_vec()?;
        ensure!(result.len() >= 3, "missing transcription result header");
        let (count, language, status) = (result[0], result[1], result[2]);
        ensure!(
            status != 2,
            "model produced nonfinite or entirely suppressed logits"
        );
        if self.debug {
            let Some(Value::MemOwn(encoder)) = outputs.next() else {
                bail!("missing encoder capture")
            };
            let Some(Value::MemOwn(trace)) = outputs.next() else {
                bail!("missing logits capture")
            };
            dump_f32(
                options.dump_dir.as_deref(),
                "encoder",
                &host(&encoder)?,
                &[1500, self.config.d_model],
            )?;
            let trace = host(&trace)?;
            let captured = (count as usize + usize::from(status == 1))
                .min(self.config.max_target_positions - 4);
            for i in 0..captured {
                dump_f32(
                    options.dump_dir.as_deref(),
                    &format!("logits_{i:03}"),
                    &trace[i * self.config.vocab_size..(i + 1) * self.config.vocab_size],
                    &[self.config.vocab_size],
                )?;
            }
        }
        let tokens = &result[3..];
        ensure!(
            count as usize <= tokens.len(),
            "invalid generated token count"
        );
        let generated = tokens[..count as usize]
            .iter()
            .map(|&t| u32::try_from(t).map_err(Into::into))
            .collect::<Result<Vec<_>>>()?;
        if status == 0 {
            eprintln!("transcription truncated at the token/position limit");
        }
        if let Some(dir) = &options.dump_dir {
            std::fs::create_dir_all(dir)?;
            std::fs::write(dir.join("tokens.json"), serde_json::to_vec(&generated)?)?;
            let prefix = [
                self.config.decoder_start_token_id,
                language as u32,
                self.generation.task_to_id["transcribe"],
                self.generation.no_timestamps_token_id,
            ];
            std::fs::write(dir.join("prefix.json"), serde_json::to_vec(&prefix)?)?;
        }
        if options.timings {
            let inference = start.elapsed().as_secs_f64();
            let name = self
                .generation
                .lang_to_id
                .iter()
                .find(|(_, id)| **id as u64 == language)
                .map(|(name, _)| name.as_str())
                .unwrap_or("unknown");
            eprintln!(
                "language: {name}; preprocessing: {:.3}s; generated inference: {inference:.3}s ({:.2} text tokens/s); warm RTF: {:.3}",
                preprocessing.as_secs_f64(),
                generated.len() as f64 / inference,
                (preprocessing.as_secs_f64() + inference) / (samples.len() as f64 / 16000.)
            );
        }

        self.tokenizer
            .decode(&generated, true)
            .map(|s| s.trim().to_string())
            .map_err(|e| anyhow::anyhow!(e.to_string()))
    }
}
pub fn dump_f32(dir: Option<&Path>, name: &str, values: &[f32], shape: &[usize]) -> Result<()> {
    if let Some(dir) = dir {
        std::fs::create_dir_all(dir)?;
        let bytes: Vec<_> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(dir.join(format!("{name}.f32")), bytes)?;
        std::fs::write(
            dir.join(format!("{name}.shape.json")),
            serde_json::to_vec(shape)?,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn generation() -> Generation {
        Generation {
            lang_to_id: BTreeMap::from([("<|en|>".into(), 5), ("<|fr|>".into(), 6)]),
            task_to_id: BTreeMap::new(),
            no_timestamps_token_id: 7,
            suppress_tokens: vec![1],
            begin_suppress_tokens: vec![4],
        }
    }
    #[test]
    fn nonlearned_generation_masks() {
        let (language, text, begin) = generation().masks(8, 4);
        assert_eq!(&language[5..7], &[0., 0.]);
        assert!(language[..5].iter().all(|v| *v == f32::NEG_INFINITY));
        assert_eq!(text[2], 0.);
        assert_eq!(text[1], f32::NEG_INFINITY);
        assert!(text[5..].iter().all(|v| *v == f32::NEG_INFINITY));
        assert_eq!(begin[4], f32::NEG_INFINITY);
    }
    #[test]
    fn bf16_rounds_ties_to_even() {
        assert_eq!(
            bf16(&[f32::from_bits(0x3f808000), f32::from_bits(0x3f818000)]),
            [0x3f80, 0x3f82]
        );
    }
}
