//! One 256x256 same-latent VAE comparison, independent of text and denoising.
use anyhow::{Result, ensure};
use catena_lang::runtime::{GpuDialect, Runtime, Value};
use catena_models::{
    catena::{compile_sources, runtime_sources},
    flux::{config, graph, vae::VaeDecoder},
    weights::Weights,
};
use std::{path::PathBuf, time::Instant};
#[test]
#[ignore = "requires gfx11 HIP GPU and cached Klein weights; only decodes 256x256"]
fn decode_saved_256() -> Result<()> {
    let dir =
        PathBuf::from(std::env::var("FLUX_VAE_REFERENCE").unwrap_or("/tmp/flux-vae-256".into()));
    std::fs::create_dir_all(&dir)?;
    let baseline = std::env::var_os("FLUX_VAE_SAVE_BASELINE").is_some();
    if baseline && dir.join("rgb.bin").exists() {
        eprintln!("reusing saved F32 reference in {}", dir.display());
        return Ok(());
    }
    let path = std::env::var_os("FLUX_VAE_LATENTS")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            if dir.join("latents.bin").exists() {
                dir.join("latents.bin")
            } else {
                "/tmp/flux-wmma-256/latents.bin".into()
            }
        });
    let bytes = std::fs::read(&path)?;
    let raw = if path.extension().is_some_and(|s| s == "safetensors") {
        let tensors = safetensors::SafeTensors::deserialize(&bytes)?;
        let tensor = tensors.tensor("value")?;
        ensure!(
            tensor.dtype() == safetensors::Dtype::BF16,
            "expected BF16 latents"
        );
        tensor.data().to_vec()
    } else {
        bytes
    };
    let latent: Vec<u16> = raw
        .chunks_exact(2)
        .map(|v| u16::from_le_bytes(v.try_into().unwrap()))
        .collect();
    ensure!(latent.len() == 256 * 128, "expected 256x256 latent tensor");
    let snapshot = config::snapshot("black-forest-labs/FLUX.2-klein-4B")?;
    let (rgb, compute, readback) = if baseline {
        // Test-only reference: restore the original F32 operations. Production
        // has one VAE path and no backend/precision selector.
        let cfg = config::read(snapshot.join("vae/config.json"))?;
        let mut program = graph::vae(&cfg, 256, 256, false)?;
        program.source = program
            .source
            .replace("} flux.vae-conv-f32", "} flux.conv2d-f32")
            .replace("} flux.vae-norm-f32", "} flux.norm-f32")
            .replace("} flux.vae-linear-f32-orr", "} flux.linear-f32-orr")
            .replace("} flux.vae-attention-f32-ooo", "} flux.attention-f32-ooo");
        let source = graph::bundle(&program.source);
        let mut runtime = Runtime::new(GpuDialect::Hip)?;
        let artifact = runtime.load(compile_sources(runtime_sources(&source), GpuDialect::Hip)?)?;
        let mut weights = Weights::open(&snapshot.join("vae/diffusion_pytorch_model.safetensors"))?;
        weights.prepare(&runtime, &program.weights, true)?;
        let input = runtime.mem_u16(&latent)?;
        let mut args = vec![input.as_ref().into()];
        for name in &program.weights {
            args.push(weights.gpu_ref(name)?.into());
        }
        let start = Instant::now();
        let mut values = artifact.exec_values("flux.decode-image", args)?;
        let compute = start.elapsed().as_secs_f64();
        let Value::MemOwn(out) = values.remove(0) else {
            anyhow::bail!("expected RGB memory")
        };
        let start = Instant::now();
        let rgb = out.try_to_f32_vec()?;
        (rgb, compute, start.elapsed().as_secs_f64())
    } else {
        let start = Instant::now();
        let vae = VaeDecoder::load(&snapshot, 256, 256)?;
        eprintln!("VAE load/compile {:.6}s", start.elapsed().as_secs_f64());
        let input = vae.ops.runtime.mem_u16(&latent)?;
        let start = Instant::now();
        let out = vae.decode(&input, None)?;
        let compute = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let rgb = out.try_to_f32_vec()?;
        (rgb, compute, start.elapsed().as_secs_f64())
    };
    ensure!(rgb.iter().all(|x| x.is_finite()), "non-finite VAE output");
    eprintln!("VAE decode {compute:.6}s; readback {readback:.6}s");
    if baseline {
        std::fs::write(
            dir.join("latents.bin"),
            latent
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        std::fs::write(
            dir.join("rgb.bin"),
            rgb.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
        )?;
    } else {
        let reference = std::fs::read(dir.join("rgb.bin"))?;
        let reference: Vec<f32> = reference
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
            .collect();
        ensure!(reference.len() == rgb.len(), "RGB shape mismatch");
        let mse = rgb
            .iter()
            .zip(&reference)
            .map(|(a, b)| (f64::from(*a) - f64::from(*b)).powi(2))
            .sum::<f64>()
            / rgb.len() as f64;
        let psnr = -10. * mse.log10();
        eprintln!("same-latent VAE PSNR {psnr:.6} dB");
        ensure!(psnr >= 35., "VAE PSNR below 35dB");
        let image = dir.join("wmma.png");
        let mut writer = png::Encoder::new(std::fs::File::create(image)?, 256, 256);
        writer.set_color(png::ColorType::Rgb);
        writer.set_depth(png::BitDepth::Eight);
        writer.write_header()?.write_image_data(
            &rgb.iter()
                .map(|v| (v * 255.).round_ties_even() as u8)
                .collect::<Vec<_>>(),
        )?;
    }
    Ok(())
}
