use catena_models::{
    catena::InferenceOptions, hfhub::download_gguf, model::Model, tokenizer::QwenTokenizer,
};

#[test]
#[ignore = "requires gfx11 HIP GPU and a cached BF16 Qwen3.5 MoE model"]
fn compile_accelerated_model() -> anyhow::Result<()> {
    use catena_lang::runtime::{GpuDialect, Runtime};
    use catena_models::catena::{
        compile_sources, generated_model_source_for_options, runtime_sources,
    };
    let selector = std::env::var("QWEN35_MODEL")
        .unwrap_or_else(|_| "unsloth/Qwen3.6-35B-A3B-GGUF:BF16".into());
    let checkpoint = download_gguf(&selector)?;
    let config = catena_models::checkpoint::read_gguf_model_config(&checkpoint.primary)?;
    let source = generated_model_source_for_options(
        &config,
        InferenceOptions::default().with_wmma_prefill(true),
    )?;
    let module = compile_sources(runtime_sources(&source), GpuDialect::Hip)?;
    let mut runtime = Runtime::new(GpuDialect::Hip)?;
    runtime.load(module)?;
    Ok(())
}

struct Case {
    name: String,
    tokens: Vec<u64>,
}
struct ResultRow {
    logits: Vec<f32>,
    generated: Vec<u64>,
}
fn run(selector: &str, wmma: bool, cases: &[Case]) -> anyhow::Result<Vec<ResultRow>> {
    let model = Model::load_hf_with_options(
        selector,
        InferenceOptions::default().with_wmma_prefill(wmma),
    )?;
    let mut results = Vec::new();
    for case in cases {
        eprintln!("parity: wmma={wmma}, {}", case.name);
        let mut session = model.session(case.tokens.len() + 20)?;
        let logits = session.prefill_logits(&case.tokens)?;
        let mut next = 0_usize;
        for i in 1..logits.len() {
            if logits[i] > logits[next] {
                next = i;
            }
        }
        let mut next = next as u64;
        let mut generated = vec![next];
        for _ in 1..20 {
            next = session.decode_next_token(next)?;
            generated.push(next);
        }
        results.push(ResultRow { logits, generated });
    }
    Ok(results)
}

#[test]
#[ignore = "requires gfx11 HIP GPU and a cached BF16 Qwen3.5 MoE model"]
fn wmma_prefill_matches_simt_model() -> anyhow::Result<()> {
    let selector = std::env::var("QWEN35_MODEL")
        .unwrap_or_else(|_| "unsloth/Qwen3.6-35B-A3B-GGUF:BF16".to_owned());
    let tokenizer = QwenTokenizer::from_gguf(download_gguf(&selector)?.primary)?
        .ok_or_else(|| anyhow::anyhow!("missing tokenizer"))?;
    let paragraph = "The river flows through the valley, carrying water from the mountains to the sea. People have built towns along its banks for centuries. Explain how seasonal rainfall affects the river and the surrounding farmland. ";
    let corpus = tokenizer.encode(&paragraph.repeat(200))?;
    let mut cases = vec![
        Case {
            name: "synthetic32".into(),
            tokens: (0..32).map(|i| 100 + i * 29).collect(),
        },
        Case {
            name: "short-text".into(),
            tokens: tokenizer.encode("The capital of France is")?,
        },
    ];
    let lengths = std::env::var("QWEN35_PARITY_LENGTHS").unwrap_or_else(|_| "512,2048,4096".into());
    for n in lengths.split(',').filter(|n| !n.is_empty()) {
        let n: usize = n.parse()?;
        anyhow::ensure!(
            n > 0 && n <= corpus.len(),
            "invalid parity prompt length {n}"
        );
        cases.push(Case {
            name: format!("text{n}"),
            tokens: corpus[..n].to_vec(),
        });
    }
    let reference = run(&selector, false, &cases)?;
    let actual = run(&selector, true, &cases)?;
    let require_exact = matches!(
        std::env::var("CATENA_WMMA_COMPONENTS").as_deref(),
        Ok("safe" | "safe-experts" | "safe-attention" | "none")
    );
    let mut failures = Vec::new();
    for ((case, reference), actual) in cases.iter().zip(&reference).zip(&actual) {
        assert_eq!(actual.logits.len(), reference.logits.len());
        let (error, magnitude) = actual
            .logits
            .iter()
            .zip(&reference.logits)
            .fold((0.0_f64, 0.0_f64), |(err, mag), (a, b)| {
                (err + f64::from(a - b).powi(2), mag + f64::from(*b).powi(2))
            });
        let rms = (error / magnitude).sqrt();
        let different_logits = actual
            .logits
            .iter()
            .zip(&reference.logits)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        eprintln!(
            "{}: relative logit RMS {rms:.6}, different logits={different_logits}, tokens match={}",
            case.name,
            actual.generated == reference.generated
        );
        if require_exact && different_logits != 0 {
            failures.push(format!(
                "{}: exact variant changed {different_logits} logits",
                case.name
            ));
        }
        if !rms.is_finite() || rms > 0.05 {
            failures.push(format!("{}: logit RMS {rms:.6} exceeds 5%", case.name));
        }
        if actual.generated != reference.generated {
            failures.push(format!(
                "{}: greedy tokens differ: {:?} vs {:?}",
                case.name, actual.generated, reference.generated
            ));
        }
    }
    anyhow::ensure!(failures.is_empty(), "{}", failures.join("\n"));
    Ok(())
}
