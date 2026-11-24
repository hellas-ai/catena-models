use anyhow::{Result, ensure};
use catena_models::qwen3_tts::{
    DEFAULT_MAX_FRAMES, DEFAULT_MODEL, MAX_FRAMES, Qwen3Tts, SynthesisOptions,
};
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
#[command(about = "Qwen3-TTS CustomVoice synthesis through Catena (24 kHz WAV)")]
struct Args {
    text: String,
    #[arg(long, default_value = DEFAULT_MODEL)]
    model: String,
    #[arg(long, default_value = "English")]
    language: String,
    #[arg(long, default_value = "Vivian")]
    speaker: String,
    /// Voice directions, such as "speak softly and warmly"
    #[arg(long)]
    instruct: Option<String>,
    /// Maximum number of generated audio frames at 12.5 frames per second
    #[arg(long, default_value_t = DEFAULT_MAX_FRAMES, value_parser = clap::value_parser!(u32).range(1..=i64::from(MAX_FRAMES)))]
    max_frames: u32,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long, default_value = "speech.wav")]
    output: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        !args.text.trim().is_empty(),
        "text must contain a spoken prompt"
    );
    let result = Qwen3Tts::load(&args.model)?.synthesize(
        &args.text,
        SynthesisOptions {
            language: &args.language,
            speaker: &args.speaker,
            instruct: args.instruct.as_deref(),
            max_frames: args.max_frames,
            seed: args.seed,
        },
    )?;
    result.audio.write_wav(&args.output)?;
    if result.frame_limit_reached {
        eprintln!("Reached the {}-frame generation limit", args.max_frames);
    }
    println!("{}", args.output.display());
    Ok(())
}
