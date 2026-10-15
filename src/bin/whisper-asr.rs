use anyhow::{Result, ensure};
use catena_qwen3::whisper::{self, Config, Options, Whisper};
use clap::Parser;
use std::path::PathBuf;
#[derive(Parser)]
#[command(about = "Whisper-small ASR through Catena (16 kHz WAV, up to 30 seconds)")]
struct Args {
    wav: Option<PathBuf>,
    #[arg(long, required = true, value_name = "MODEL")]
    model: String,
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
fn main() -> Result<()> {
    let a = Args::parse();
    let model_dir = whisper::hub::snapshot(&a.model)?;
    let config = Config::read(&model_dir)?;
    if a.timings || a.inspect {
        eprintln!("Whisper snapshot: {}", model_dir.display());
    }
    if let Some(path) = a.generate_entrypoint {
        std::fs::write(path, whisper::graph::bundle(&config, a.dump_dir.is_some())?)?;
        return Ok(());
    }
    if a.inspect {
        let w = catena_qwen3::weights::Weights::open(&model_dir.join("model.safetensors"))?;
        whisper::validate_weights(&w, &config)?;
        eprintln!(
            "Whisper-small: {} tensors, {:.2} GiB",
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
    let model = Whisper::load_with_debug(&model_dir, a.timings, a.dump_dir.is_some())?;
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
