use anyhow::{Result, ensure};
use catena_lang::codegen::CodegenKind;
use catena_qwen3::whisper::{self, Config, Options, Whisper};
use clap::Parser;
use std::path::PathBuf;
#[derive(Parser)]
#[command(
    about = "Whisper ASR through Catena (16 kHz WAV, up to 30 seconds; defaults to Whisper small)"
)]
struct Args {
    wav: Option<PathBuf>,
    #[arg(long, default_value = "openai/whisper-small", value_name = "MODEL")]
    model: String,
    /// Compiler backend; experimental uses only generic GPU primitives.
    #[arg(long, value_enum, default_value = "default")]
    codegen: CodegenArg,
    #[arg(long)]
    language: Option<String>,
    #[arg(long, default_value_t = 256)]
    max_new_tokens: usize,
    #[arg(long)]
    timings: bool,
    #[arg(long)]
    dump_dir: Option<PathBuf>,
    /// Write the rendered model Hex and exit without initializing the GPU.
    #[arg(long, value_name = "PATH")]
    generate_entrypoint: Option<PathBuf>,
    /// Validate model configuration and checkpoint shapes without initializing the GPU.
    #[arg(long)]
    inspect: bool,
    /// Write CPU log-mel features and exit without initializing the GPU.
    #[arg(long)]
    features_only: bool,
}
#[derive(Clone, Copy, clap::ValueEnum)]
enum CodegenArg {
    Default,
    Experimental,
}
impl From<CodegenArg> for CodegenKind {
    fn from(value: CodegenArg) -> Self {
        match value {
            CodegenArg::Default => Self::Default,
            CodegenArg::Experimental => Self::Experimental,
        }
    }
}
fn main() -> Result<()> {
    let a = Args::parse();
    let model_dir = whisper::hub::snapshot(&a.model)?;
    let config = Config::read(&model_dir)?;
    if a.timings || a.inspect {
        eprintln!("Whisper snapshot: {}", model_dir.display());
    }
    if let Some(path) = a.generate_entrypoint {
        std::fs::write(
            path,
            whisper::graph::bundle_with_codegen(&config, a.dump_dir.is_some(), a.codegen.into())?,
        )?;
        return Ok(());
    }
    if a.inspect {
        let w = catena_qwen3::weights::Weights::open(&model_dir.join("model.safetensors"))?;
        whisper::validate_weights(&w, &config)?;
        eprintln!(
            "Whisper: {} tensors, {:.2} GiB",
            w.tensors.len(),
            w.byte_len() as f64 / (1u64 << 30) as f64
        );
        return Ok(());
    }
    ensure!(a.max_new_tokens > 0, "--max-new-tokens must be positive");
    let wav = a
        .wav
        .ok_or_else(|| anyhow::anyhow!("a WAV path is required"))?;
    let samples = whisper::audio::read_wav(&wav)?;
    if a.features_only {
        ensure!(a.dump_dir.is_some(), "--features-only requires --dump-dir");
        let mel = whisper::audio::log_mel(&samples)?;
        whisper::dump_f32(a.dump_dir.as_deref(), "log_mel", &mel, &[3000, 80])?;
        return Ok(());
    }
    let model = Whisper::load_with_codegen(
        &model_dir,
        a.timings,
        a.dump_dir.is_some(),
        a.codegen.into(),
    )?;
    let text = model.transcribe(
        &samples,
        &Options {
            language: a.language,
            max_new_tokens: a.max_new_tokens,
            timings: a.timings,
            dump_dir: a.dump_dir,
        },
    )?;
    println!("{text}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn defaults_to_whisper_small_and_selects_experimental() {
        let args = Args::try_parse_from(["whisper-asr", "--codegen", "experimental", "speech.wav"])
            .unwrap();
        assert_eq!(args.model, "openai/whisper-small");
        assert!(matches!(args.codegen, CodegenArg::Experimental));
        let args = Args::try_parse_from(["whisper-asr", "--model", "owner/whisper", "speech.wav"])
            .unwrap();
        assert_eq!(args.model, "owner/whisper");
    }
}
