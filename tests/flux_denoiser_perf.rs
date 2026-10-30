//! One saved-conditioning 256x256 denoiser run, independent of text encoding.
use anyhow::{Result, ensure};
use catena_models::{
    flux::{config, transformer::Transformer},
    weights::bf16_to_f32,
};
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use std::{path::PathBuf, time::Instant};
#[test]
#[ignore = "requires gfx11 HIP GPU and cached Klein weights; only 256x256"]
fn saved_denoiser_256() -> Result<()> {
    let dir = PathBuf::from("/tmp/flux-denoiser-tune");
    std::fs::create_dir_all(&dir)?;
    let baseline = std::env::var_os("FLUX_DENOISER_SAVE_BASELINE").is_some();
    let path = dir.join("baseline.bin");
    if baseline && path.exists() {
        return Ok(());
    }
    let bytes = std::fs::read("/tmp/flux-wmma-256/conditioning.bin")?;
    let conditioning: Vec<u16> = bytes
        .chunks_exact(2)
        .map(|x| u16::from_le_bytes(x.try_into().unwrap()))
        .collect();
    ensure!(
        conditioning.len() == 512 * 7680,
        "expected saved BF16 conditioning"
    );
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
    let noise: Vec<f32> = (0..256 * 128)
        .map(|_| StandardNormal.sample(&mut rng))
        .collect();
    let snapshot = config::snapshot("black-forest-labs/FLUX.2-klein-4B")?;
    let start = Instant::now();
    let model = Transformer::load(&snapshot, 256, 256, 4)?;
    eprintln!(
        "denoiser load/compile {:.6}s",
        start.elapsed().as_secs_f64()
    );
    let n = model.ops.runtime.mem_f32(&noise)?;
    let c = model.ops.runtime.mem_u16(&conditioning)?;
    let start = Instant::now();
    let out = model.denoise(n.as_ref(), c.as_ref(), None)?;
    eprintln!("denoise256 {:.6}s", start.elapsed().as_secs_f64());
    let out = out.try_to_u16_vec()?;
    ensure!(
        out.iter().all(|x| bf16_to_f32(*x).is_finite()),
        "nonfinite denoiser output"
    );
    let raw: Vec<u8> = out.iter().flat_map(|x| x.to_le_bytes()).collect();
    if baseline {
        std::fs::write(path, raw)?;
    } else {
        let bytes = std::fs::read(path)?;
        let reference: Vec<f32> = bytes
            .chunks_exact(2)
            .map(|x| bf16_to_f32(u16::from_le_bytes(x.try_into().unwrap())))
            .collect();
        ensure!(reference.len() == out.len(), "latent shape mismatch");
        let err = out
            .iter()
            .zip(&reference)
            .map(|(a, b)| (f64::from(bf16_to_f32(*a)) - f64::from(*b)).powi(2))
            .sum::<f64>();
        let norm = reference.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>();
        let nrms = (err / norm).sqrt();
        eprintln!("same-input latent NRMS {nrms:.8}");
        ensure!(
            raw == bytes,
            "denoiser output differs from preceding WMMA path"
        );
        std::fs::write(dir.join("optimized.bin"), raw)?;
    }
    Ok(())
}
