use anyhow::Result;
use catena_qwen3::{flux::config, weights::Weights};
use clap::Parser;
use std::path::PathBuf;
#[derive(Parser)]
#[command(about = "FLUX.2-klein-4B text-to-image generation")]
struct Args {
    #[arg(long, value_enum, default_value_t = catena_qwen3::flux::ops::Backend::Fast)]
    backend: catena_qwen3::flux::ops::Backend,
    #[arg(long)]
    prompt: Option<String>,
    /// Compiler backend; experimental uses generic Hex GPU kernels.
    #[arg(long, value_enum, default_value = "default")]
    codegen: CodegenArg,
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
#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum CodegenArg {
    Default,
    Experimental,
}
impl From<CodegenArg> for catena_lang::codegen::CodegenKind {
    fn from(value: CodegenArg) -> Self {
        match value {
            CodegenArg::Default => Self::Default,
            CodegenArg::Experimental => Self::Experimental,
        }
    }
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
        let options = catena_qwen3::flux::pipeline::GenerationOptions {
            backend: a.backend,
            codegen: a.codegen.into(),
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
        let text = catena_qwen3::flux::text::generated_source_with_codegen(
            &t.qwen()?,
            debug,
            a.codegen.into(),
        )?;
        let mut denoise =
            catena_qwen3::flux::graph::transformer(&d, a.height, a.width, a.steps, debug)?;
        let mut vae = catena_qwen3::flux::graph::vae(&v, a.height, a.width, debug)?;
        if matches!(a.codegen, CodegenArg::Experimental) {
            denoise.source = catena_qwen3::flux::graph::experimental_source(&denoise.source)?;
            vae.source = catena_qwen3::flux::graph::experimental_source(&vae.source)?;
        }
        std::fs::create_dir_all(dir)?;
        for (name, source) in [
            ("flux-encode-text.hex", text),
            ("flux-denoise.hex", denoise.source),
            ("flux-decode-image.hex", vae.source),
        ] {
            let path = dir.join(name);
            std::fs::write(
                &path,
                catena_qwen3::flux::graph::bundle_with_codegen(
                    &source,
                    a.backend,
                    a.codegen.into(),
                ),
            )?;
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
    let options = catena_qwen3::flux::pipeline::GenerationOptions {
        backend: a.backend,
        codegen: a.codegen.into(),
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn compiler_selection() {
        let args = Args::try_parse_from([
            "flux2-klein",
            "--codegen",
            "experimental",
            "--prompt",
            "a fox",
        ])
        .unwrap();
        assert_eq!(args.model, "black-forest-labs/FLUX.2-klein-4B");
        assert!(matches!(args.codegen, CodegenArg::Experimental));
        assert!(matches!(
            args.backend,
            catena_qwen3::flux::ops::Backend::Fast
        ));
        let args = Args::try_parse_from([
            "flux2-klein",
            "--model",
            "owner/model",
            "--generate-entrypoint",
            "/tmp/assets",
        ])
        .unwrap();
        assert_eq!(args.model, "owner/model");
        assert!(matches!(args.codegen, CodegenArg::Default));
        assert!(args.generate_entrypoint.is_some());
    }
}
