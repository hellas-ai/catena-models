use anyhow::Result;
use catena_lang::runtime::GpuDialect;
use catena_models::{
    catena::{
        InferenceOptions, KernelKind, compile_sources, generated_model_source_for_options,
        runtime_sources,
    },
    spec::ModelConfig,
};

#[test]
fn qwen3_portable_reductions_compile() -> Result<()> {
    let mut cfg = ModelConfig::qwen3_0_6b();
    cfg.hidden_size = 64;
    cfg.intermediate_size = 128;
    cfg.num_attention_heads = 2;
    cfg.num_key_value_heads = 1;
    cfg.head_dim = 32;
    cfg.rope_dimension_count = 32;
    cfg.num_layers = 1;
    cfg.vocabulary_size = 256;
    for kernels in [KernelKind::Reference, KernelKind::Fused] {
        let source = generated_model_source_for_options(
            &cfg,
            InferenceOptions::default().with_kernels(kernels),
        )?;
        let raw = metacat::theory::RawTheorySet::from_texts(runtime_sources(&source))?;
        let prepared = catena_models::catena::prepare_sources(raw)?;
        let program = prepared.theories.get(&"program".parse().unwrap()).unwrap();
        assert!(
            !program
                .arrows
                .keys()
                .any(|n| n.as_str() == "stdlib.models.math.dot-pair-f32")
        );
        for dialect in [GpuDialect::Hip, GpuDialect::Cuda] {
            let module = compile_sources(runtime_sources(&source), dialect)?;
            assert!(!module.source.contains("catena_bf16_dot"));
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires a GPU"]
fn standalone_scores_match_reference() -> Result<()> {
    use catena_lang::runtime::{Runtime, Value};
    use catena_models::weights::{bf16_to_f32, f32_to_bf16};
    let mut cfg = ModelConfig::qwen3_0_6b();
    cfg.hidden_size = 64;
    cfg.intermediate_size = 128;
    cfg.num_attention_heads = 2;
    cfg.num_key_value_heads = 1;
    cfg.head_dim = 32;
    cfg.rope_dimension_count = 32;
    cfg.num_layers = 1;
    cfg.vocabulary_size = 256;
    let dialect = catena_models::model::configured_gpu_dialect()?;
    let source = generated_model_source_for_options(
        &cfg,
        InferenceOptions::default().with_kernels(KernelKind::Reference),
    )?;
    let mut rt = Runtime::new(dialect)?;
    let artifact = rt.load(compile_sources(runtime_sources(&source), dialect)?)?;
    for keys in [1, 3, 65] {
        let data = |n, salt| {
            (0..n)
                .map(|i| f32_to_bf16((((i * 13 + salt) % 51) as f32 - 25.) / 16.))
                .collect::<Vec<_>>()
        };
        let q = data(64, 3);
        let k = data(keys * 32, 7);
        let mut expected = vec![];
        for head in 0..2 {
            for key in 0..keys {
                let mut sums = [0f32; 2];
                for pair in 0..16 {
                    for j in 0..2 {
                        let ch = pair * 2 + j;
                        sums[j] += bf16_to_f32(q[head * 32 + ch]) * bf16_to_f32(k[key * 32 + ch]);
                    }
                }
                expected.push(bf16_to_f32(f32_to_bf16(
                    (sums[0] + sums[1]) * (1f32 / 32f32.sqrt()),
                )));
            }
        }
        let [Value::MemOwn(out)] = artifact.exec(
            "qwen3.scores",
            [
                rt.mem_u16(&q)?.into(),
                rt.mem_u16(&k)?.into(),
                (keys as u64).into(),
                0u64.into(),
            ],
        )?
        else {
            anyhow::bail!("score output ABI")
        };
        anyhow::ensure!(out.try_to_f32_vec()? == expected, "keys={keys}");
    }
    Ok(())
}

#[test]
#[ignore = "requires GPU, cached Qwen3 BF16 weights and CATENA_QWEN3_SIMT_DUMP_DIR"]
fn teacher_forced_model_logits() -> Result<()> {
    use catena_models::{model::Model, tokenizer::QwenTokenizer};
    use std::path::PathBuf;
    let dir = PathBuf::from(std::env::var("CATENA_QWEN3_SIMT_DUMP_DIR")?);
    std::fs::create_dir_all(&dir)?;
    let path = catena_models::hfhub::download_gguf("unsloth/Qwen3-0.6B-GGUF:BF16")?.primary;
    let tokenizer = QwenTokenizer::from_gguf(&path)?.unwrap();
    let model = Model::load_gguf_with_options(&path, InferenceOptions::default())?;
    for (i, prompt) in [
        "The capital of France is",
        "The capital of France is Paris. The capital of",
        "Two plus two equals",
    ]
    .iter()
    .enumerate()
    {
        let logits = model.logits(&tokenizer.encode(prompt)?)?;
        anyhow::ensure!(logits.iter().all(|x| x.is_finite()), "nonfinite logits");
        std::fs::write(
            dir.join(format!("{i}.f32")),
            logits
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        let mut order = (0..logits.len()).collect::<Vec<_>>();
        order.sort_by(|a, b| logits[*b].total_cmp(&logits[*a]));
        eprintln!(
            "prompt {i}: top {:?}",
            order[..5]
                .iter()
                .map(|j| (*j, logits[*j]))
                .collect::<Vec<_>>()
        );
        if let Some(reference) = std::env::var_os("CATENA_QWEN3_SIMT_REFERENCE_DIR") {
            let bytes = std::fs::read(PathBuf::from(reference).join(format!("{i}.f32")))?;
            let reference = bytes
                .chunks_exact(4)
                .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
                .collect::<Vec<_>>();
            anyhow::ensure!(reference.len() == logits.len(), "logit shape mismatch");
            let error = logits
                .iter()
                .zip(&reference)
                .map(|(x, y)| f64::from(x - y).powi(2))
                .sum::<f64>();
            let norm = reference.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>();
            let nrms = (error / norm).sqrt();
            eprintln!("prompt {i}: logit NRMS {nrms:.8}");
            let mae = logits
                .iter()
                .zip(&reference)
                .map(|(x, y)| f64::from((x - y).abs()))
                .sum::<f64>()
                / logits.len() as f64;
            let mut reference_order = (0..reference.len()).collect::<Vec<_>>();
            reference_order.sort_by(|a, b| reference[*b].total_cmp(&reference[*a]));
            let overlap = order[..5]
                .iter()
                .filter(|i| reference_order[..5].contains(i))
                .count();
            // Reordered BF16 sums can swap close candidates. Preserve rankings
            // with a clear margin and bound drift in the full vocabulary too.
            anyhow::ensure!(nrms <= 0.03 && mae <= 0.1, "excessive logit drift");
            anyhow::ensure!(overlap >= 4, "top-five predictions changed excessively");
            if reference[reference_order[0]] - reference[reference_order[1]] >= 0.5 {
                anyhow::ensure!(
                    order[0] == reference_order[0],
                    "high-margin prediction changed"
                );
            }
        }
    }
    Ok(())
}
