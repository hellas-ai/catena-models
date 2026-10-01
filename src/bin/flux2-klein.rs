use anyhow::Result;
use catena_qwen3::flux::{config, weights::Weights};
use clap::Parser;
use std::path::PathBuf;
#[derive(Parser)]
#[command(about = "Offline FLUX.2-klein-4B text-to-image generation")]
struct Args {
    #[arg(long, value_enum, default_value_t = catena_qwen3::flux::ops::Backend::Fast)]
    backend: catena_qwen3::flux::ops::Backend,
    #[arg(long)]
    prompt: Option<String>,
    #[arg(long, default_value = "image.png")]
    output: PathBuf,
    #[arg(long)]
    model_dir: Option<PathBuf>,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long, default_value_t = 1024)]
    width: usize,
    #[arg(long, default_value_t = 1024)]
    height: usize,
    #[arg(long, default_value_t = 4)]
    steps: usize,
    #[arg(long)]
    inspect: bool,
    /// Print checkpoint, compiler and weight preparation startup timings.
    #[arg(long)]
    startup_timings: bool,
    #[arg(long)]
    dump_dir: Option<PathBuf>,
    #[arg(long)]
    noise: Option<PathBuf>,
}
fn main() -> Result<()> {
    let a = Args::parse();
    if a.startup_timings {
        // This single-threaded entry point has not initialized any runtime or worker threads.
        unsafe { std::env::set_var("CATENA_STARTUP_TIMINGS", "1") };
    }
    let p = config::snapshot(a.model_dir.as_deref())?;
    let t: config::TextConfig = config::read(p.join("text_encoder/config.json"))?;
    t.qwen()?;
    let d: config::TransformerConfig = config::read(p.join("transformer/config.json"))?;
    d.validate()?;
    let v: config::VaeConfig = config::read(p.join("vae/config.json"))?;
    v.validate()?;
    for component in ["text_encoder", "transformer", "vae"] {
        let w = Weights::open(&p.join(component))?;
        eprintln!(
            "{component}: {} tensors, {:.2} GiB",
            w.tensors.len(),
            w.byte_len() as f64 / (1u64 << 30) as f64
        );
    }
    if a.inspect {
        return Ok(());
    }
    let options = catena_qwen3::flux::pipeline::GenerationOptions {
        backend: a.backend,
        prompt: a
            .prompt
            .ok_or_else(|| anyhow::anyhow!("--prompt is required for generation"))?,
        output: a.output,
        seed: a.seed,
        width: a.width,
        height: a.height,
        steps: a.steps,
        dump_dir: a.dump_dir,
        noise: a.noise,
    };
    catena_qwen3::flux::pipeline::FluxPipeline::new(p)?.generate(&options)
}
