//! Checkpoint integration checks, deliberately kept to one hardware test.
use catena_models::{
    catena::{self, InferenceOptions},
    checkpoint,
    model::{Model, configured_gpu_dialect},
    tokenizer::QwenTokenizer,
};

#[path = "support/qwen_tiles.rs"]
#[allow(dead_code)]
mod tiles;

fn checkpoint_path() -> anyhow::Result<std::path::PathBuf> {
    if let Some(path) = std::env::var_os("LFM3_GGUF") {
        return Ok(path.into());
    }
    let root = std::path::PathBuf::from(std::env::var_os("HF_HOME").unwrap_or_else(|| {
        std::path::PathBuf::from(std::env::var_os("HOME").unwrap())
            .join(".cache/huggingface")
            .into_os_string()
    }))
    .join("hub/models--LiquidAI--LFM3-1.7B-GGUF/snapshots");
    for snapshot in std::fs::read_dir(root)? {
        let path = snapshot?.path().join("LFM3-1.7B-BF16.gguf");
        if path.exists() {
            return Ok(path);
        }
    }
    anyhow::bail!("set LFM3_GGUF to the cached BF16 checkpoint")
}

#[test]
#[ignore = "requires cached LFM3 BF16 checkpoint"]
fn checkpoint_metadata_and_source() -> anyhow::Result<()> {
    let path = checkpoint_path()?;
    let cfg = checkpoint::read_gguf_model_config(&path)?;
    assert_eq!(cfg.hidden_size, 2048);
    assert_eq!(cfg.num_layers, 24);
    use catena_models::spec::LayerType;
    assert_eq!(cfg.layer_type(0), LayerType::Convolution);
    assert_eq!(cfg.layer_type(2), LayerType::FullAttention);
    assert_eq!(cfg.layer_type(4), LayerType::SlidingAttention);
    checkpoint::validate_gguf_tensors(&path, &cfg)?;
    let tokenizer = QwenTokenizer::from_gguf(&path)?.unwrap();
    let text = "Hello, world! 1234567 café e\u{301}\n中文 🦙";
    let ids = tokenizer.encode(text)?;
    assert_eq!(tokenizer.decode(&ids, false)?, text);
    // Pinned against llama.cpp using this GGUF, including combining marks and emoji.
    assert_eq!(
        ids,
        vec![
            35808, 20, 1530, 9, 229, 9792, 25677, 31, 34998, 320, 9685, 207, 108820, 23304, 108,
            256
        ]
    );
    assert_eq!(
        tokenizer.encode("<|im_end|>")?,
        vec![cfg.eos_token_id.unwrap()]
    );
    for accelerated in [false, true] {
        let source = catena::generated_model_source_for_options(
            &cfg,
            InferenceOptions::default().with_wmma_prefill(accelerated),
        )?;
        assert!(!source.contains("bf16x2.dot-f32"));
        let module =
            catena::compile_sources(catena::runtime_sources(&source), configured_gpu_dialect()?)?;
        assert!(!module.source.contains("catena_bf16_dot"));
        assert!(!module.source.contains("__builtin_amdgcn_fdot2"));
        if accelerated {
            tiles::check_native(&module, configured_gpu_dialect()?);
            if configured_gpu_dialect()? == catena_lang::runtime::GpuDialect::Cuda {
                tiles::check_toolkit(module, "lfm3_text")?;
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires a supported HIP/CUDA GPU and cached LFM3 BF16 checkpoint"]
fn wmma_prefill_decode_and_resume() -> anyhow::Result<()> {
    let path = checkpoint_path()?;
    let tokenizer = QwenTokenizer::from_gguf(&path)?.unwrap();
    let length = std::env::var("LFM3_TEST_LENGTH")
        .ok()
        .map(|s| s.parse::<usize>())
        .transpose()?
        .unwrap_or(273);
    let tokens = tokenizer.encode(
        &"The river flows through the valley. ".repeat(if length > 351 { 200 } else { 50 }),
    )?;
    if let Ok(path) = std::env::var("LFM3_REFERENCE_TOKENS") {
        let reference = std::fs::read_to_string(path)?
            .split_whitespace()
            .map(str::parse)
            .collect::<Result<Vec<u64>, _>>()?;
        assert_eq!(tokens, reference, "tokenization differs from llama.cpp");
    }
    anyhow::ensure!(
        length > 129 && length <= tokens.len(),
        "test length must be between 130 and {}",
        tokens.len()
    );
    let tokens = &tokens[..length];
    let model =
        Model::load_gguf_with_options(&path, InferenceOptions::default().with_wmma_prefill(true))?;
    let started = std::time::Instant::now();
    let mut full = model.session(length + 32)?;
    let actual = full.prefill_logits(tokens)?;
    if let Ok(path) = std::env::var("LFM3_REFERENCE_LOGITS") {
        let bytes = std::fs::read(path)?;
        let reference = bytes
            .chunks_exact(4)
            .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
            .collect::<Vec<_>>();
        compare(&actual, &reference);
    }
    eprintln!(
        "LFM3 WMMA prefill {} tokens: {:.3}s",
        tokens.len(),
        started.elapsed().as_secs_f64()
    );
    let mut chunked = model.session(length + 32)?;
    chunked.prefill_next_token(&tokens[..129])?;
    let reference = chunked.append_logits(&tokens[129..])?;
    compare(&actual, &reference);
    let checkpoint = chunked.into_checkpoint()?;
    let mut resumed = model.resume_checkpoint(&checkpoint)?;
    let next = actual
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0 as u64;
    compare(&full.decode_logits(next)?, &resumed.decode_logits(next)?);
    let mut serial = model.session(length + 32)?;
    serial.prefill_next_token(&tokens[..1])?;
    for &token in &tokens[1..tokens.len() - 1] {
        serial.decode_next_token(token)?;
    }
    compare(&actual, &serial.decode_logits(*tokens.last().unwrap())?);
    Ok(())
}
fn compare(a: &[f32], b: &[f32]) {
    let error =
        (a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f32>() / a.len() as f32).sqrt();
    eprintln!("LFM3 logit RMS difference: {error}");
    assert!(error < 0.3, "LFM3 logit RMS error {error}");
    let top = |x: &[f32]| {
        x.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0
    };
    assert_eq!(top(a), top(b));
}
