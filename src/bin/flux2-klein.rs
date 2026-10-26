use anyhow::Result;
use catena_models::{flux::config, weights::Weights};
use clap::Parser;
use std::path::PathBuf;
#[derive(Parser)]
#[command(about = "FLUX.2-klein-4B text-to-image generation")]
struct Args {
    #[arg(long)]
    prompt: Option<String>,
    #[arg(long, default_value = "image.png")]
    output: PathBuf,
    #[arg(
        long,
        default_value = "black-forest-labs/FLUX.2-klein-4B",
        value_name = "MODEL"
    )]
    model: String,
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
    /// Write the three complete Hex stage assets and exit without initializing a GPU.
    #[arg(long, value_name = "DIR", conflicts_with = "inspect")]
    generate_entrypoint: Option<PathBuf>,
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
    let p = config::snapshot(&a.model)?;
    let t: config::TextConfig = config::read(p.join("text_encoder/config.json"))?;
    t.qwen()?;
    let d: config::TransformerConfig = config::read(p.join("transformer/config.json"))?;
    d.validate()?;
    let v: config::VaeConfig = config::read(p.join("vae/config.json"))?;
    v.validate()?;
    if let Some(dir) = &a.generate_entrypoint {
        let options = catena_models::flux::pipeline::GenerationOptions {
            prompt: String::new(),
            output: a.output.clone(),
            seed: a.seed,
            width: a.width,
            height: a.height,
            steps: a.steps,
            dump_dir: a.dump_dir.clone(),
            noise: a.noise.clone(),
        };
        options.validate()?;
        let debug = a.dump_dir.is_some();
        let text = catena_models::flux::text::generated_source_with_debug(&t.qwen()?, debug)?;
        let denoise =
            catena_models::flux::graph::transformer(&d, a.height, a.width, a.steps, debug)?;
        let vae = catena_models::flux::graph::vae(&v, a.height, a.width, debug)?;
        std::fs::create_dir_all(dir)?;
        for (name, source) in [
            ("flux-encode-text.hex", text),
            ("flux-denoise.hex", denoise.source),
            ("flux-decode-image.hex", vae.source),
        ] {
            let path = dir.join(name);
            std::fs::write(&path, catena_models::flux::graph::bundle(&source))?;
            println!("generated {}", path.display());
        }
        return Ok(());
    }
    for (component, filename) in [
        ("text_encoder", "model.safetensors.index.json"),
        ("transformer", "diffusion_pytorch_model.safetensors"),
        ("vae", "diffusion_pytorch_model.safetensors"),
    ] {
        let w = Weights::open(&p.join(component).join(filename))?;
        eprintln!(
            "{component}: {} tensors, {:.2} GiB",
            w.tensors.len(),
            w.byte_len() as f64 / (1u64 << 30) as f64
        );
    }
    if a.inspect {
        return Ok(());
    }
    let options = catena_models::flux::pipeline::GenerationOptions {
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
    catena_models::flux::pipeline::FluxPipeline::new(p)?.generate(&options)
}
