use anyhow::{Context, Result};
use catena_models::{
    catena::{InferenceOptions, PrefillDtype, generated_model_source_for_options},
    hfhub::download_model,
    model::Model,
    spec::ModelConfig,
};

fn options() -> InferenceOptions {
    InferenceOptions::default()
        .with_wmma_prefill(true)
        .with_prefill_dtype(PrefillDtype::Fp8)
}

#[test]
fn dense_fp8_graph_uses_native_projections_and_gated_attention() -> Result<()> {
    let source = generated_model_source_for_options(&ModelConfig::qwen35_0_8b(), options())?;
    assert!(!source.contains("{{QUERY_SIZE}}"));
    // Appended FP8 operands must never displace the final norm or vocabulary weights.
    let specs = catena_models::spec::weight_specs(&ModelConfig::qwen35_0_8b())?;
    assert!(source.contains(&format!("[.hidden-final] [.w{}]", specs.len() - 2)));
    assert!(source.contains(&format!("[.normalized-final] [.w{}]", specs.len() - 1)));
    let full = source
        .split("(def program stdlib.qwen35.batch.full :")
        .nth(1)
        .unwrap();
    let full = full
        .split("(def program stdlib.qwen35.batch.linear :")
        .next()
        .unwrap();
    assert!(full.contains("stdlib.qwen3.fp8.attention.run [context.]"));
    assert!(full.contains("stdlib.qwen3.fp8.project-gate"));
    assert!(!full.contains("stdlib.qwen35.dense.project ["));
    let linear = source
        .split("(def program stdlib.qwen35.batch.linear :")
        .nth(1)
        .unwrap();
    let linear = linear
        .split("(def program stdlib.qwen35.batch.branch-full :")
        .next()
        .unwrap();
    assert!(linear.contains("stdlib.qwen3.fp8.project-residual"));
    assert!(linear.contains("stdlib.qwen35.dense.scan-l8"));
    assert!(linear.contains("[.normalized] [.w3]"));
    assert!(linear.contains("[.normalized] [.w4]"));
    // No token-count branch to BF16 attention in the FP8 adapter.
    let adapter = source
        .split("(def program stdlib.qwen3.fp8.attention.run :")
        .nth(1)
        .unwrap();
    assert!(
        adapter
            .lines()
            .take(5)
            .any(|l| l.contains("attention.native"))
    );
    Ok(())
}

#[test]
fn fp8_rejects_unsupported_attention_instead_of_falling_back() {
    let mut config = ModelConfig::qwen35_0_8b();
    config.head_dim = 16;
    config.rope_dimension_count = 16;
    let error = generated_model_source_for_options(&config, options()).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("FP8 attention requires head dimensions")
    );
}

#[test]
fn qwen3_short_attention_uses_fp8_residuals() -> Result<()> {
    let source = generated_model_source_for_options(&ModelConfig::qwen3_0_6b(), options())?;
    assert_eq!(
        source
            .matches("(def program stdlib.qwen3.fp8.attention.run :")
            .count(),
        1
    );
    assert!(source.contains("stdlib.qwen3.fp8.precise-attention.short"));
    assert!(source.contains("stdlib.qwen3.fp8.precise-attention.native"));
    Ok(())
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(&a, &b)| a as f64 * b as f64).sum();
    let aa: f64 = a.iter().map(|&x| (x as f64).powi(2)).sum();
    let bb: f64 = b.iter().map(|&x| (x as f64).powi(2)).sum();
    dot / (aa * bb).sqrt()
}

#[test]
#[ignore = "requires CUDA and downloads a dynamic blockwise FP8 Qwen3.5 checkpoint"]
fn dense_fp8_prefill_matches_bf16_and_continues_decode() -> Result<()> {
    let repo =
        std::env::var("QWEN35_FP8_MODEL").unwrap_or_else(|_| "surogate/Qwen3.5-0.8B-FP8".into());
    let mut checkpoint = download_model(&repo)?;
    if let Ok(layers) = std::env::var("QWEN35_FP8_LAYERS") {
        if let catena_models::hfhub::HubCheckpoint::Safetensors { config, .. } = &mut checkpoint {
            config.num_layers = layers.parse()?;
        }
    }

    let bf16 = Model::load_checkpoint_with_options(
        &checkpoint,
        options().with_prefill_dtype(PrefillDtype::Bf16),
    )?;
    let fp8 = Model::load_checkpoint_with_options(&checkpoint, options())?;
    let tokenizer = checkpoint.tokenizer()?;
    let mut cases: Vec<Vec<u64>> = [1, 2, 17, 65, 128, 512]
        .into_iter()
        .map(|n| (0..n).map(|i| 100 + (i as u64 * 17) % 5000).collect())
        .collect();
    cases.push(tokenizer.encode("The capital of France is")?);
    for tokens in cases {
        let a = bf16
            .logits(&tokens)
            .with_context(|| format!("BF16 pp{}", tokens.len()))?;
        let mut session = fp8.session(tokens.len() + 4)?;
        let b = session
            .prefill_logits(&tokens)
            .with_context(|| format!("FP8 pp{}", tokens.len()))?;
        assert!(b.iter().all(|x| x.is_finite()));
        let similarity = cosine(&a, &b);
        eprintln!("Qwen3.5 FP8 pp{} cosine={similarity:.8}", tokens.len());
        assert!(similarity > 0.99, "logit cosine {similarity}");
        if tokens.len() == 1 {
            assert_eq!(a, b);
        }
        let mut reference = if tokens.len() == 1 {
            let mut reference = bf16.session(5)?;
            reference.prefill_logits(&tokens)?;
            Some(reference)
        } else {
            None
        };
        for token in [100, 200, 300] {
            let next = session.decode_logits(token)?;
            assert!(next.iter().all(|x| x.is_finite()));
            if let Some(reference) = &mut reference {
                assert_eq!(next, reference.decode_logits(token)?, "decode remains BF16");
            }
        }
    }
    let tokens: Vec<_> = (0..128).map(|i| 100 + i * 17 % 5000).collect();
    let whole = fp8.logits(&tokens)?;
    let mut session = fp8.session(132)?;
    session.append_logits(&tokens[..17])?;
    let chunked = session.append_logits(&tokens[17..])?;
    let similarity = cosine(&whole, &chunked);
    eprintln!("Qwen3.5 chunked FP8 cosine={similarity:.8}");
    assert!(similarity > 0.99);
    Ok(())
}

#[test]
#[ignore = "manual CUDA end-to-end prefill benchmark"]
fn dense_fp8_prefill_benchmark() -> Result<()> {
    let repo =
        std::env::var("QWEN35_FP8_MODEL").unwrap_or_else(|_| "surogate/Qwen3.5-0.8B-FP8".into());
    let checkpoint = download_model(&repo)?;
    let bf16 = Model::load_checkpoint_with_options(
        &checkpoint,
        options().with_prefill_dtype(PrefillDtype::Bf16),
    )?;
    let fp8 = Model::load_checkpoint_with_options(&checkpoint, options())?;
    eprintln!(
        "Qwen3.5 weights BF16={} FP8={} bytes",
        bf16.weight_byte_len(),
        fp8.weight_byte_len()
    );
    for rows in [17, 128, 512, 1024] {
        let tokens: Vec<u64> = (0..rows).map(|i| 100 + (i as u64 * 17) % 5000).collect();
        let mut medians = Vec::new();
        for (name, model) in [("BF16", &bf16), ("FP8", &fp8)] {
            for _ in 0..2 {
                model.logits(&tokens)?;
            }
            let mut samples = Vec::new();
            for _ in 0..7 {
                let mut session = model.session(rows)?;
                let start = std::time::Instant::now();
                session.prefill_next_token(&tokens)?;
                samples.push(start.elapsed().as_secs_f64());
            }
            samples.sort_by(f64::total_cmp);
            let median = samples[samples.len() / 2];
            medians.push(median);
            eprintln!(
                "Qwen3.5 pp{rows} {name}: {:.3} ms, {:.2} tok/s",
                median * 1000.0,
                rows as f64 / median
            );
        }
        eprintln!("Qwen3.5 pp{rows} speedup {:.3}x", medians[0] / medians[1]);
    }
    Ok(())
}
