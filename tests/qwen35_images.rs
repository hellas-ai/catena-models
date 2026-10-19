//! Checkpoint-backed checks are opt-in; ordinary tests need no GPU or downloads.
use anyhow::{Context, Result, ensure};
use catena_qwen3::{
    catena::InferenceOptions,
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
    let dialect = configured_gpu_dialect()?;
    let started = Instant::now();
    let selector = std::env::var("QWEN35_MODEL").context("set QWEN35_MODEL")?;
    let checkpoint = download_gguf(&selector)?;
    let model = Model::load_gguf_with_multimodal_options_and_dialect(
        &checkpoint.primary,
        InferenceOptions::default(),
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
    // Language inference always uses BF16, including comparisons with F32 HF
    // references. Keep exact greedy-token agreement alongside bounded logits.
    let (max, mean) = (1.0, 0.2);
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
