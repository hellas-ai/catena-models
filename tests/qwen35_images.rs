//! Checkpoint-backed checks are opt-in; ordinary tests need no GPU or downloads.
use anyhow::{Context, Result, ensure};
use catena_qwen3::{
    catena::{InferenceOptions, InferencePrecision},
    hfhub::{download_gguf, download_mmproj},
    model::{Model, configured_gpu_dialect},
    tokenizer::QwenTokenizer,
    vision::{PreparedMultimodalPrompt, VisionModel},
};
use std::{
    path::{Path, PathBuf},
    time::Instant,
};
fn reference_dir() -> Result<PathBuf> {
    Ok(std::env::var_os("QWEN35_REFERENCE")
        .context("set QWEN35_REFERENCE")?
        .into())
}
fn f32_file(path: &Path) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path)?;
    ensure!(bytes.len().is_multiple_of(4), "invalid float fixture");
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|v| f32::from_le_bytes(*v))
        .collect())
}
fn compare(actual: &[f32], expected: &[f32], max_limit: f32, mean_limit: f32) -> Result<()> {
    ensure!(
        actual.len() == expected.len() && !actual.is_empty(),
        "fixture shape mismatch"
    );
    ensure!(actual.iter().all(|v| v.is_finite()), "nonfinite output");
    let errors: Vec<_> = actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .collect();
    let max = errors.iter().copied().fold(0_f32, f32::max);
    let mean = errors.iter().sum::<f32>() / errors.len() as f32;
    eprintln!("max absolute error {max}, mean absolute error {mean}");
    ensure!(
        max <= max_limit && mean <= mean_limit,
        "numerical fixture mismatch: max={max}, mean={mean}"
    );
    Ok(())
}
#[test]
#[ignore = "requires QWEN35_REFERENCE and QWEN35_MODEL; tokenizer/prompt parity only, no GPU"]
fn tokenizer_prompt_matches_reference() -> Result<()> {
    let dir = reference_dir()?;
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("metadata.json"))?)?;
    let tokenizer = QwenTokenizer::from_gguf(
        download_gguf(&std::env::var("QWEN35_MODEL").context("set QWEN35_MODEL")?)?.primary,
    )?
    .context("GGUF tokenizer missing")?;
    let tokens = tokenizer.encode_without_special_tokens(
        meta["template_text"].as_str().context("missing template")?,
    )?;
    // Replace each single image pad with its merged image-token count.
    let pad = tokenizer.special_token_id("<|image_pad|>")?;
    let mut expanded = vec![];
    let mut grids = meta["grids"].as_array().unwrap().iter();
    for t in tokens {
        if t == pad {
            let g = grids.next().context("image span mismatch")?;
            expanded.resize(
                expanded.len() + (g[1].as_u64().unwrap() * g[2].as_u64().unwrap() / 4) as usize,
                pad,
            );
        } else {
            expanded.push(t);
        }
    }
    ensure!(
        serde_json::json!(expanded) == meta["tokens"],
        "GGUF tokenizer differs from reference"
    );
    Ok(())
}
#[test]
#[ignore = "requires a GPU, QWEN35_MODEL, QWEN35_MMPROJ_MODEL and QWEN35_REFERENCE"]
fn native_images_prefill_and_cached_decode_match_reference() -> Result<()> {
    let dir = reference_dir()?;
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("metadata.json"))?)?;
    let precision = match meta["precision"].as_str() {
        Some("f32") => InferencePrecision::F32,
        _ => InferencePrecision::Bf16,
    };
    let dialect = configured_gpu_dialect()?;
    let started = Instant::now();
    let selector = std::env::var("QWEN35_MODEL").context("set QWEN35_MODEL")?;
    let checkpoint = download_gguf(&selector)?;
    let codegen = match std::env::var("QWEN35_CODEGEN").as_deref() {
        Ok("experimental") => catena_lang::codegen::CodegenKind::Experimental,
        Ok("default") | Err(_) => catena_lang::codegen::CodegenKind::Default,
        Ok(value) => anyhow::bail!("invalid QWEN35_CODEGEN {value}"),
    };
    let model = Model::load_gguf_with_multimodal_options_and_dialect(
        &checkpoint.primary,
        InferenceOptions::new(precision).with_codegen(codegen),
        dialect,
    )?;
    let tokenizer =
        QwenTokenizer::from_gguf(&checkpoint.primary)?.context("GGUF tokenizer missing")?;
    let vision = VisionModel::load_gguf(
        download_mmproj(&std::env::var("QWEN35_MMPROJ_MODEL").unwrap_or(selector))?,
        dialect,
        meta["max_pixels"].as_u64().map(|n| n as usize),
    )?;
    eprintln!("load/compile {:?}", started.elapsed());
    let mut images = vec![];
    for (i, name) in ["square", "portrait"].iter().enumerate() {
        let image = vision.encode_image(dir.join(format!("{name}.png")))?;
        compare(
            &image.to_f32_vec()?,
            &f32_file(&dir.join(format!("image-{i}-embedding.f32")))?,
            0.01,
            0.001,
        )?;
        images.push(image);
    }
    let prepared =
        PreparedMultimodalPrompt::new(&tokenizer, meta["prompt"].as_str().unwrap(), images)?;
    ensure!(
        serde_json::json!(prepared.tokens()) == meta["tokens"],
        "expanded prompt differs"
    );
    ensure!(
        serde_json::json!(prepared.positions()) == meta["positions"],
        "MRoPE positions differ"
    );
    ensure!(
        prepared.rope_delta() == meta["rope_delta"].as_i64().unwrap(),
        "RoPE decode delta differs"
    );
    let mut session = model.session(prepared.token_count() + 4)?;
    let started = Instant::now();
    let mut logits = session.prefill_multimodal_logits(&prepared)?;
    eprintln!("prefill {:?}", started.elapsed());
    let (max, mean) = if precision == InferencePrecision::F32 {
        (0.05, 0.005)
    } else {
        // Catena's mixed BF16 graph retains F32 norm weights, reductions and
        // RoPE tables; Transformers rounds more intermediate values to BF16.
        // Trained dense models amplify these rounding differences. Keep exact
        // greedy-token agreement in addition to the bounded logit comparison.
        (1.0, 0.2)
    };
    compare(
        &logits,
        &f32_file(&dir.join("prefill-logits.f32"))?,
        max,
        mean,
    )?;
    ensure!(
        session.position() == prepared.token_count(),
        "image tokens did not advance cache"
    );
    for step in 0..3 {
        let next = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0 as u64;
        ensure!(
            Some(next) == meta["generated"][step].as_u64(),
            "greedy output differs at step {step}"
        );
        let started = Instant::now();
        logits = session.decode_logits(next)?;
        eprintln!("decode-{step} {:?}", started.elapsed());
        compare(
            &logits,
            &f32_file(&dir.join(format!("decode-{step}-logits.f32")))?,
            max,
            mean,
        )?;
    }
    Ok(())
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Memory {
    data: *mut std::ffi::c_void,
    len: u64,
}
impl Memory {
    fn of<T>(values: &[T]) -> Self {
        Self {
            data: values.as_ptr() as *mut _,
            len: std::mem::size_of_val(values) as u64,
        }
    }
}
#[test]
#[ignore = "requires QWEN35_HOST_LIBRARY, compiled with host-backed scalar loads; no GPU"]
fn generated_vision_producers_match_reference_math() -> Result<()> {
    let path = std::env::var_os("QWEN35_HOST_LIBRARY").context("set QWEN35_HOST_LIBRARY")?;
    // SAFETY: the library is our generated HIP code; only host scalar producers run.
    let lib = unsafe { libloading::Library::new(path)? };
    type Gelu = unsafe extern "C" fn(Memory, u64, u64, *mut f32);
    type Position = unsafe extern "C" fn(Memory, Memory, Memory, u64, u64, u64, *mut f32);
    type Rope = unsafe extern "C" fn(Memory, Memory, Memory, u64, u64, u64, u64, *mut f32);
    unsafe {
        let exact = lib.get::<Gelu>(b"program_vision_gelu_erf_at")?;
        let tanh = lib.get::<Gelu>(b"program_vision_gelu_tanh_at")?;
        let values: Vec<f32> = (0..2001).map(|i| -10. + i as f32 / 100.).collect();
        for (i, &x) in values.iter().enumerate() {
            let mut actual = 0.;
            tanh(
                Memory::of(&values),
                values.len() as u64,
                i as u64,
                &mut actual,
            );
            let xd = x as f64;
            let expected = (0.5
                * xd
                * (1.
                    + (std::f64::consts::FRAC_2_SQRT_PI / std::f64::consts::SQRT_2
                        * (xd + 0.044715 * xd.powi(3)))
                    .tanh())) as f32;
            ensure!(
                (actual - expected).abs() < 2e-6,
                "tanh GELU mismatch for {x}: {actual} vs {expected}"
            );
            exact(
                Memory::of(&values),
                values.len() as u64,
                i as u64,
                &mut actual,
            );
            // Independent erf from the system libm.
            unsafe extern "C" {
                fn erf(x: f64) -> f64;
            }
            let expected = (0.5 * xd * (1. + erf(xd / std::f64::consts::SQRT_2))) as f32;
            ensure!(
                (actual - expected).abs() < 2e-6,
                "erf GELU mismatch for {x}"
            );
        }
        let interpolate = lib.get::<Position>(b"program_vision_position_at")?;
        let embedding: Vec<f32> = (0..24).map(|i| (i as f32 * 0.1).sin()).collect();
        let indices = [0_u64, 2, 3, 5];
        let weights = [0.1_f32, 0.2, 0.3, 0.4];
        for col in 0..4 {
            let mut actual = 0.;
            interpolate(
                Memory::of(&embedding),
                Memory::of(&indices),
                Memory::of(&weights),
                4,
                4,
                col,
                &mut actual,
            );
            let expected = (0..4)
                .map(|k| embedding[indices[k] as usize * 4 + col as usize] * weights[k])
                .sum::<f32>();
            ensure!(
                (actual - expected).abs() < 2e-7,
                "learned position interpolation mismatch"
            );
        }
        let rotate = lib.get::<Rope>(b"program_vision_rope_at")?;
        let x: Vec<f32> = (0..16).map(|i| (i as f32 * 0.2).sin()).collect();
        let positions = [2_u64, 3, 5, 7];
        let frequencies = [1_f32, 0.01];
        for i in 0..16 {
            let channel = i % 8;
            let pair = channel % 4;
            let angle = positions[(i / 8) * 2 + pair / 2] as f64 * frequencies[pair % 2] as f64;
            let other = if channel < 4 { -x[i + 4] } else { x[i - 4] };
            let expected = (x[i] as f64 * angle.cos() + other as f64 * angle.sin()) as f32;
            let mut actual = 0.;
            rotate(
                Memory::of(&x),
                Memory::of(&positions),
                Memory::of(&frequencies),
                16,
                8,
                8,
                i as u64,
                &mut actual,
            );
            ensure!(
                (actual - expected).abs() < 2e-6,
                "vision spatial RoPE mismatch"
            );
        }
    }
    Ok(())
}
#[test]
#[ignore = "requires QWEN35_LANGUAGE_HOST_LIBRARY with host-backed scalar loads; no GPU"]
fn generated_interleaved_mrope_matches_reference_math() -> Result<()> {
    let path = std::env::var_os("QWEN35_LANGUAGE_HOST_LIBRARY")
        .context("set QWEN35_LANGUAGE_HOST_LIBRARY")?;
    // SAFETY: generated scalar ABI, with three host-backed u64 position values.
    unsafe {
        let lib = libloading::Library::new(path)?;
        type Softplus = unsafe extern "C" fn(f32, *mut f32);
        let softplus = lib.get::<Softplus>(b"program_qwen3_softplus")?;
        for i in -2000..=2000 {
            let x = i as f32 / 100.;
            let expected = ((x as f64).max(0.) + (-(x as f64).abs()).exp().ln_1p()) as f32;
            let mut actual = 0.;
            softplus(x, &mut actual);
            ensure!(
                (actual - expected).abs() <= 1e-6 * expected.max(1e-8),
                "DeltaNet softplus mismatch for {x}: {actual} vs {expected}"
            );
        }
        type Table = unsafe extern "C" fn(Memory, u64, *mut f32);
        let table = lib.get::<Table>(b"program_qwen3_mm_rope_table_at")?;
        let positions = [1_u64, 7, 11];
        for i in 0..64 {
            let pair = i % 32;
            let axis = if pair % 3 == 1 && pair < 33 {
                1
            } else if pair % 3 == 2 && pair < 30 {
                2
            } else {
                0
            };
            let angle = positions[axis] as f64 * 10000000_f32.powf(-2. * pair as f32 / 64.) as f64;
            let expected = if i < 32 { angle.cos() } else { angle.sin() } as f32;
            let mut actual = 0.;
            table(Memory::of(&positions), i as u64, &mut actual);
            ensure!(
                (actual - expected).abs() < 2e-6,
                "interleaved MRoPE mismatch for pair {pair}"
            );
        }
    }
    Ok(())
}
