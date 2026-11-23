use anyhow::{Result, ensure};
use catena_models::omnivoice::{DEFAULT_NUM_STEP, OmniVoice, SynthesisOptions};
use clap::Parser;
use std::{path::PathBuf, time::Instant};

#[derive(Parser)]
#[command(about = "OmniVoice text-to-speech through Catena (24 kHz WAV)")]
struct Args {
    text: String,
    /// English voice directions, such as "whisper" or an experimental emotion prompt
    #[arg(long, value_name = "DIRECTIONS")]
    instruct: Option<String>,
    /// Number of refinement steps. Fewer steps reduce synthesis time
    #[arg(long, default_value_t = DEFAULT_NUM_STEP, value_parser = clap::value_parser!(u32).range(1..))]
    num_step: u32,
    #[arg(long, default_value = "speech.wav")]
    output: PathBuf,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long)]
    timings: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        !args.text.trim().is_empty(),
        "text must contain a spoken prompt"
    );
    let start = Instant::now();
    let model = OmniVoice::load()?;
    if args.timings {
        eprintln!("load and compile {:.3}s", start.elapsed().as_secs_f64());
    }
    let start = Instant::now();
    let audio = model.synthesize_with_options(
        &args.text,
        args.seed,
        SynthesisOptions {
            instruct: args.instruct.as_deref(),
            num_step: args.num_step,
        },
    )?;
    if args.timings {
        eprintln!("synthesis {:.3}s", start.elapsed().as_secs_f64());
    }
    audio.write_wav(&args.output)?;
    println!("{}", args.output.display());
    Ok(())
}
