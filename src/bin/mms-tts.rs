use anyhow::{Result, ensure};
use catena_models::{
    tts::{self, Config, Options, Tts},
    weights::Weights,
};
use clap::Parser;
use std::path::PathBuf;
#[derive(Parser)]
#[command(about = "MMS English text-to-speech through Catena (16 kHz WAV)")]
struct Args {
    text: Option<String>,
    #[arg(long, default_value = "speech.wav")]
    output: PathBuf,
    #[arg(long, default_value = "facebook/mms-tts-eng", value_name = "MODEL")]
    model: String,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long, default_value_t = 1.)]
    speaking_rate: f32,
    #[arg(long, default_value_t = 0.667)]
    noise_scale: f32,
    #[arg(long, default_value_t = 0.8)]
    noise_scale_duration: f32,
    #[arg(long)]
    timings: bool,
    #[arg(long)]
    dump_dir: Option<PathBuf>,
    #[arg(long, conflicts_with = "generate_entrypoint")]
    inspect: bool,
    #[arg(long, value_name = "PATH")]
    generate_entrypoint: Option<PathBuf>,
}
fn main() -> Result<()> {
    let a = Args::parse();
    let options = Options {
        seed: a.seed,
        speaking_rate: a.speaking_rate,
        noise_scale: a.noise_scale,
        noise_scale_duration: a.noise_scale_duration,
        timings: a.timings,
        dump_dir: a.dump_dir,
    };
    options.validate()?;
    ensure!(
        a.inspect || a.generate_entrypoint.is_some() || a.text.is_some(),
        "text is required"
    );
    let dir = tts::snapshot(&a.model)?;
    let config = Config::read(&dir)?;
    if a.timings || a.inspect {
        eprintln!("MMS snapshot: {}", dir.display());
    }
    if let Some(path) = a.generate_entrypoint {
        std::fs::write(path, tts::graph::bundle(&config)?)?;
        return Ok(());
    }
    if a.inspect {
        let weights = Weights::open(&dir.join("model.safetensors"))?;
        tts::validate_weights(&weights, &config)?;
        tts::tokenizer::Tokenizer::read(&dir)?;
        eprintln!(
            "MMS English: {} checkpoint tensors, {} inference bindings, {:.2} MiB, 16 kHz",
            weights.tensors.len(),
            tts::bindings().len(),
            weights.byte_len() as f64 / (1 << 20) as f64
        );
        return Ok(());
    }
    let text = a.text.unwrap();
    tts::tokenizer::Tokenizer::read(&dir)?.encode(&text)?;
    Tts::load(&dir, a.timings)?
        .synthesize(&text, &options)?
        .write_wav(&a.output)?;
    println!("{}", a.output.display());
    Ok(())
}
