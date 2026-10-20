//! Opt-in checks against external Flux model references.
use anyhow::{Context, Result, ensure};

#[test]
#[ignore = "requires HIP GPU, model weights, FLUX_TEST_REFERENCE and FLUX_TEST_DUMP directories"]
fn cached_components_match_reference() -> Result<()> {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use rand_distr::{Distribution, StandardNormal};
    use std::path::PathBuf;
    let reference =
        PathBuf::from(std::env::var("FLUX_TEST_REFERENCE").context("set FLUX_TEST_REFERENCE")?);
    let output = PathBuf::from(std::env::var("FLUX_TEST_DUMP").context("set FLUX_TEST_DUMP")?);
    let width = std::env::var("FLUX_TEST_WIDTH")
        .unwrap_or("64".into())
        .parse::<usize>()?;
    let height = std::env::var("FLUX_TEST_HEIGHT")
        .unwrap_or("96".into())
        .parse::<usize>()?;
    let seed = std::env::var("FLUX_TEST_SEED")
        .unwrap_or("42".into())
        .parse::<u64>()?;
    let model =
        std::env::var("FLUX_TEST_MODEL").unwrap_or("black-forest-labs/FLUX.2-klein-4B".into());
    let snapshot = super::config::snapshot(&model)?;
    let read_bf16 = |name: &str| -> Result<Vec<u16>> {
        let bytes = std::fs::read(reference.join(format!("{name}.safetensors")))?;
        let tensors = safetensors::SafeTensors::deserialize(&bytes)?;
        let tensor = tensors.tensor("value")?;
        ensure!(
            tensor.dtype() == safetensors::Dtype::BF16,
            "expected BF16 fixture {name}"
        );
        Ok(tensor
            .data()
            .chunks_exact(2)
            .map(|x| u16::from_le_bytes(x.try_into().unwrap()))
            .collect())
    };
    let read_f32 = |name: &str| -> Result<Vec<f32>> {
        let bytes = std::fs::read(reference.join(format!("{name}.safetensors")))?;
        let tensors = safetensors::SafeTensors::deserialize(&bytes)?;
        let tensor = tensors.tensor("value")?;
        ensure!(
            tensor.dtype() == safetensors::Dtype::F32,
            "expected F32 fixture {name}"
        );
        Ok(tensor
            .data()
            .chunks_exact(4)
            .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
            .collect())
    };
    let nrms = |actual: &[f32], expected: &[f32]| -> f64 {
        assert_eq!(actual.len(), expected.len());
        assert!(actual.iter().all(|x| x.is_finite()));
        let error = actual
            .iter()
            .zip(expected)
            .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
            .sum::<f64>();
        (error
            / expected
                .iter()
                .map(|x| (*x as f64).powi(2))
                .sum::<f64>()
                .max(1e-30))
        .sqrt()
    };
    let transformer =
        super::transformer::Transformer::load_with_debug(&snapshot, height, width, 1, true)?;
    let conditioning = read_bf16("conditioning")?;
    ensure!(conditioning.len() == 512 * 7680, "wrong conditioning shape");
    let conditioning = transformer.ops.runtime.mem_u16(&conditioning)?;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let noise = (0..width / 16 * (height / 16) * 128)
        .map(|_| StandardNormal.sample(&mut rng))
        .collect::<Vec<f32>>();
    let noise = transformer.ops.runtime.mem_f32(&noise)?;
    let latents = transformer.denoise(noise.as_ref(), conditioning.as_ref(), Some(&output))?;
    let actual = latents
        .try_to_u16_vec()?
        .into_iter()
        .map(super::pipeline::bf16_to_f32)
        .collect::<Vec<_>>();
    let expected = read_bf16("step_0_latents")?
        .into_iter()
        .map(super::pipeline::bf16_to_f32)
        .collect::<Vec<_>>();
    eprintln!(
        "same-conditioning transformer latent NRMS: {:.6}",
        nrms(&actual, &expected)
    );
    // BF16 reduction order differs from WMMA and can amplify through late
    // blocks and the Euler subtraction. Keep this comparison diagnostic;
    // the VAE comparison below enforces a numerical error bound.
    ensure!(
        actual.iter().all(|x| x.is_finite()),
        "non-finite transformer output"
    );
    drop(latents);
    drop(noise);
    drop(conditioning);
    drop(transformer);
    let vae = super::vae::VaeDecoder::load_with_debug(&snapshot, height, width, true)?;
    let latents = read_bf16("step_0_latents")?;
    ensure!(
        latents.len() == width / 16 * (height / 16) * 128,
        "wrong latent shape"
    );
    let latents = vae.ops.runtime.mem_u16(&latents)?;
    let rgb = vae.decode(&latents, Some(&output))?.try_to_f32_vec()?;
    // Diagnostic rgb is raw NCHW; the stage result is normalized HWC.
    let raw = read_f32("rgb")?;
    ensure!(raw.len() == width * height * 3, "wrong RGB fixture shape");
    let expected = (0..width * height * 3)
        .map(|i| (raw[i % 3 * width * height + i / 3] * 0.5 + 0.5).clamp(0., 1.))
        .collect::<Vec<_>>();
    let error = nrms(&rgb, &expected);
    eprintln!("same-latent F32 VAE NRMS: {error:.8}");
    ensure!(error < 0.0005, "F32 VAE diverged with identical inputs");
    Ok(())
}
