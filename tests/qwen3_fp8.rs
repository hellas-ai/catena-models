use anyhow::Result;
use catena_lang::runtime::GpuDialect;
use catena_models::{
    catena::{
        InferenceOptions, PrefillDtype, compile_sources, generated_model_source_for_options,
        runtime_sources,
    },
    spec::ModelConfig,
};

#[test]
fn fp8_prefill_compiles_to_native_mma() -> Result<()> {
    let mut config = ModelConfig::qwen3_0_6b();
    config.num_layers = 1;
    let options = InferenceOptions::default()
        .with_wmma_prefill(true)
        .with_prefill_dtype(PrefillDtype::Fp8);
    let source = generated_model_source_for_options(&config, options)?;
    let module = compile_sources(runtime_sources(&source), GpuDialect::Cuda)?;
    assert!(
        module
            .source
            .contains("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32")
    );
    assert!(module.source.contains("// CATENA_CUDA_MIN_SM=89"));
    Ok(())
}

fn kernel_module() -> Result<catena_lang::runtime::RuntimeModule> {
    let mut config = ModelConfig::qwen3_0_6b();
    config.num_layers = 1;
    let options = InferenceOptions::default()
        .with_wmma_prefill(true)
        .with_prefill_dtype(PrefillDtype::Fp8);
    let mut source = generated_model_source_for_options(&config, options)?;
    for tile in [16, 64] {
        for mode in ["", "-residual", "-gate"] {
            let (extra_types, extra_names, extra_args, free_residual) = match mode {
                "-residual" => (
                    "((cap.own mem) val)",
                    "residual",
                    "[.residual]",
                    "({[.residual] [.after-scales]} stdlib.models.resources.free-bf16-before [result.])",
                ),
                "-gate" => (
                    "((cap.ref mem) val) ((cap.ref mem) val)",
                    "up-weight up-scales",
                    "[.up-weight] [.up-scales]",
                    "([.after-scales] [result.])",
                ),
                _ => ("", "", "", "([.after-scales] [result.])"),
            };
            source.push_str(&format!(r#"
(def program fp8-test-m{tile}{mode} : {{((cap.own mem) val) ((cap.ref mem) val) ((cap.ref mem) val) (u64 val) (u64 val) {extra_types}}} -> ((cap.own mem) val) =
 ([input weight weight-scales features outputs {extra_names}.]
  ({{[.input]}} stdlib.qwen3.fp8.quantize [quantized.])
  ([.quantized] *.elim [packed scales.])
  ({{[.input]}} stdlib.models.memory.own-count-bf16 [elements.])
  ({{[.elements] [.features]}} stdlib.models.memory.u64.div [rows.])
  ({{[.packed] [.weight] [.scales] [.weight-scales] [.rows] [.features] [.outputs] {extra_args}}} stdlib.qwen3.fp8.m{tile}{mode} [projected.])
  ({{[.packed] [.projected]}} stdlib.models.resources.free-f32-before [after-packed.])
  ({{[.scales] [.after-packed]}} stdlib.models.resources.free-f32-before [after-scales.])
  {free_residual}
  ({{[.input] [.result]}} stdlib.models.resources.free-bf16-before)
 ))
"#));
        }
    }
    source.push_str(r#"
(def program bf16-test : {((cap.own mem) val) ((cap.ref mem) val) (u64 val) (u64 val)} -> ((cap.own mem) val) =
 ([input weight features outputs.]
  ({[.input] [.weight] [.features] [.outputs]} stdlib.qwen35.dense.project [out.])
  ({[.input] [.out]} stdlib.models.resources.free-bf16-before)
 ))
"#);
    // Keep only the fixture entrypoints public so kernel checks do not compile
    // the model's unrelated decode, attention, and vocabulary entrypoints.
    let source = source
        .replace("stdlib.qwen3.", "stdlib.fixture-internal.")
        .replace("qwen3.", "stdlib.fixture-public.")
        .replace("qwen3 :", "stdlib.fixture-root :");
    compile_sources(runtime_sources(&source), GpuDialect::Cuda)
}

fn decode_fp8(bits: u8) -> f32 {
    let e = (bits >> 3) & 15;
    let m = bits & 7;
    let v = if e == 0 {
        m as f32 / 512.0
    } else if e == 15 && m == 7 {
        f32::NAN
    } else {
        (1.0 + m as f32 / 8.0) * 2.0_f32.powi(e as i32 - 7)
    };
    if bits & 128 == 0 { v } else { -v }
}

fn quantize_reference(values: &[u16]) -> Vec<f32> {
    use catena_models::weights::bf16_to_f32;
    values
        .chunks(128)
        .flat_map(|block| {
            let amax = block
                .iter()
                .map(|&v| bf16_to_f32(v).abs())
                .fold(0.0_f32, f32::max);
            let scale = if amax == 0.0 { 1.0 } else { amax / 448.0 };
            block.iter().map(move |&v| {
                let v = bf16_to_f32(v) / scale;
                let best = (0..=255u8)
                    .filter(|&b| decode_fp8(b).is_finite())
                    .min_by(|&a, &b| {
                        (decode_fp8(a) - v)
                            .abs()
                            .total_cmp(&(decode_fp8(b) - v).abs())
                            .then_with(|| (a & 1).cmp(&(b & 1)))
                    })
                    .unwrap();
                decode_fp8(best) * scale
            })
        })
        .collect()
}

#[test]
#[ignore = "requires CUDA SM89+ and nvcc"]
fn fp8_kernels_match_independent_blockwise_reference() -> Result<()> {
    use catena_lang::runtime::{Runtime, Value};
    use catena_models::weights::{bf16_to_f32, f32_to_bf16};
    let mut runtime = Runtime::new(GpuDialect::Cuda)?;
    let artifact = runtime.load(kernel_module()?)?;
    for (rows, outputs, features, zero) in [
        (2, 128, 128, false),
        (17, 256, 256, false),
        (65, 128, 384, false),
        (33, 128, 128, true),
    ] {
        let input: Vec<_> = (0..rows * features)
            .map(|i| {
                f32_to_bf16(if zero || i / 128 % 7 == 0 {
                    0.0
                } else {
                    ((i * 17 % 113) as f32 - 56.0) / 19.0
                })
            })
            .collect();
        let reference_a = quantize_reference(&input);
        let weight: Vec<_> = (0..outputs * features)
            .map(|i| {
                [0x38, 0xb8, 0x30, 0xb0, 0x40, 0xc0, 0x00, 0x01, 0x7e, 0xfe]
                    [(i * 13 + i / features) % 10]
            })
            .collect();
        let scales: Vec<_> = (0..outputs / 128 * features / 128)
            .map(|i| (i as f32 + 1.0) / 32.0)
            .collect();
        let b = runtime.mem_u8(&weight)?;
        let bs = runtime.mem_f32(&scales)?;
        let up_scales = runtime.mem_f32(&scales.iter().map(|s| s * 2.0).collect::<Vec<_>>())?;
        let residual: Vec<_> = (0..rows * outputs)
            .map(|i| f32_to_bf16((i % 17) as f32 / 16.0 - 0.5))
            .collect();
        for tile in [16, 64] {
            for mode in ["", "-residual", "-gate"] {
                let a = runtime.mem_u16(&input)?;
                let mut args = vec![
                    a.into(),
                    b.as_ref().into(),
                    bs.as_ref().into(),
                    (features as u64).into(),
                    (outputs as u64).into(),
                ];
                if mode == "-residual" {
                    args.push(runtime.mem_u16(&residual)?.into());
                }
                if mode == "-gate" {
                    args.extend([b.as_ref().into(), up_scales.as_ref().into()]);
                }
                let mut result = artifact.exec_values(&format!("fp8-test-m{tile}{mode}"), args)?;
                let Value::MemOwn(out) = result.remove(0) else {
                    anyhow::bail!("expected memory");
                };
                let actual = out.try_to_u16_vec()?;
                for row in 0..rows {
                    for col in 0..outputs {
                        let expected: f32 = (0..features)
                            .map(|k| {
                                reference_a[row * features + k]
                                    * decode_fp8(weight[col * features + k])
                                    * scales[col / 128 * (features / 128) + k / 128]
                            })
                            .sum();
                        let mut expected = bf16_to_f32(f32_to_bf16(expected));
                        if mode == "-gate" {
                            let up = bf16_to_f32(f32_to_bf16(expected * 2.0));
                            let silu =
                                bf16_to_f32(f32_to_bf16(expected / (1.0 + (-expected).exp())));
                            expected = bf16_to_f32(f32_to_bf16(silu * up));
                        }
                        if mode == "-residual" {
                            expected = bf16_to_f32(f32_to_bf16(
                                expected + bf16_to_f32(residual[row * outputs + col]),
                            ));
                        }
                        let got = bf16_to_f32(actual[row * outputs + col]);
                        assert!(
                            (got - expected).abs() <= 0.01 * expected.abs() + 0.002,
                            "m{tile}{mode} [{rows},{outputs},{features}] ({row},{col}): {got} != {expected}"
                        );
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
fn fp8_option_rejects_incompatible_graphs() {
    use catena_models::spec::ModelArchitecture;
    let mut config = ModelConfig::qwen3_0_6b();
    let options = InferenceOptions::default().with_prefill_dtype(PrefillDtype::Fp8);
    assert!(generated_model_source_for_options(&config, options).is_err());
    config.architecture = ModelArchitecture::Qwen35Dense;
    assert!(generated_model_source_for_options(&config, options.with_wmma_prefill(true)).is_err());
}

#[test]
#[ignore = "requires CUDA SM89+ and cached unsloth/Qwen3-0.6B-FP8"]
fn unsloth_fp8_prefill_matches_bf16_and_continues_decode() -> Result<()> {
    use catena_models::{hfhub::download_model, model::Model};
    let checkpoint = download_model("unsloth/Qwen3-0.6B-FP8")?;
    let options = InferenceOptions::default().with_wmma_prefill(true);
    let baseline = Model::load_checkpoint_with_options(&checkpoint, options)?;
    let fp8 = Model::load_checkpoint_with_options(
        &checkpoint,
        options.with_prefill_dtype(PrefillDtype::Fp8),
    )?;
    assert!(fp8.weight_byte_len() > baseline.weight_byte_len());
    let mut cases: Vec<Vec<u64>> = [1, 2, 17, 128, 512, 1024]
        .into_iter()
        .map(|len| (0..len).map(|i| 100 + (i as u64 * 17) % 5000).collect())
        .collect();
    let tokenizer = checkpoint.tokenizer()?;
    for prompt in [
        "The capital of France is",
        "Explain why the sky is blue.",
        "fn fibonacci(n: u32) -> u32 {",
        "Translate to French: Good morning.",
    ] {
        cases.push(tokenizer.encode(prompt)?);
    }
    for tokens in cases {
        let expected = baseline.logits(&tokens)?;
        let mut session = fp8.session(tokens.len() + 4)?;
        let actual = session.prefill_logits(&tokens)?;
        assert!(actual.iter().all(|v| v.is_finite()));
        let cosine = cosine(&expected, &actual);
        eprintln!("{} tokens: logit cosine {cosine:.8}", tokens.len());
        assert!(cosine >= 0.99);
        if tokens.len() == 1 {
            assert_eq!(expected, actual, "single-token prefill must retain BF16");
            let mut reference = baseline.session(5)?;
            reference.prefill_logits(&tokens)?;
            for token in [42, 123, 77] {
                assert_eq!(
                    reference.decode_logits(token)?,
                    session.decode_logits(token)?,
                    "decode must retain BF16"
                );
            }
        } else {
            let token = actual
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0 as u64;
            assert!(session.decode_logits(token)?.iter().all(|v| v.is_finite()));
        }
    }
    let tokens: Vec<_> = (0..128).map(|i| 100 + (i * 17) % 5000).collect();
    let expected = fp8.logits(&tokens)?;
    let mut chunked = fp8.session(132)?;
    chunked.append_logits(&tokens[..17])?;
    let actual = chunked.append_logits(&tokens[17..])?;
    assert!(
        cosine(&expected, &actual) >= 0.99,
        "chunked prefill must use absolute positions"
    );
    Ok(())
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(&x, &y)| x as f64 * y as f64).sum();
    let aa: f64 = a.iter().map(|&x| (x as f64).powi(2)).sum();
    let bb: f64 = b.iter().map(|&x| (x as f64).powi(2)).sum();
    dot / (aa * bb).sqrt()
}

#[test]
#[ignore = "manual CUDA performance measurement using the Unsloth checkpoint"]
fn fp8_prefill_benchmark() -> Result<()> {
    use catena_models::{hfhub::download_model, model::Model};
    use std::time::Instant;
    let checkpoint = download_model("unsloth/Qwen3-0.6B-FP8")?;
    let options = InferenceOptions::default().with_wmma_prefill(true);
    let bf16 = Model::load_checkpoint_with_options(&checkpoint, options)?;
    let fp8 = Model::load_checkpoint_with_options(
        &checkpoint,
        options.with_prefill_dtype(PrefillDtype::Fp8),
    )?;
    eprintln!(
        "weight bytes: BF16={}, FP8={}",
        bf16.weight_byte_len(),
        fp8.weight_byte_len()
    );
    for len in [17, 128, 512, 1024] {
        let tokens: Vec<_> = (0..len).map(|i| 100 + (i as u64 * 17) % 5000).collect();
        let mut medians = Vec::new();
        for (name, model) in [("bf16", &bf16), ("fp8", &fp8)] {
            for _ in 0..2 {
                model.logits(&tokens)?;
            }
            let mut samples = Vec::new();
            for _ in 0..9 {
                let mut session = model.session(len)?;
                let start = Instant::now();
                session.prefill_next_token(&tokens)?;
                samples.push(start.elapsed().as_secs_f64());
            }
            samples.sort_by(f64::total_cmp);
            let seconds = samples[samples.len() / 2];
            eprintln!(
                "pp{len} {name}: {:.3} ms, {:.2} tok/s (median of 9 after 2 warmups)",
                seconds * 1000.0,
                len as f64 / seconds
            );
            medians.push(seconds);
        }
        eprintln!("pp{len} speedup: {:.3}x", medians[0] / medians[1]);
    }
    Ok(())
}

#[test]
#[ignore = "manual CUDA projection measurement including activation quantization"]
fn fp8_projection_benchmark() -> Result<()> {
    use catena_lang::runtime::Runtime;
    use catena_models::weights::f32_to_bf16;
    use std::time::Instant;
    let mut runtime = Runtime::new(GpuDialect::Cuda)?;
    let artifact = runtime.load(kernel_module()?)?;
    for (features, outputs) in [(1024, 2048), (1024, 3072), (3072, 1024)] {
        let weight = vec![0x38; features * outputs];
        let bf16_weight = runtime.mem_u16(&vec![f32_to_bf16(1.0); features * outputs])?;
        let fp8_weight = runtime.mem_u8(&weight)?;
        let scales = runtime.mem_f32(&vec![1.0; features / 128 * outputs / 128])?;
        for rows in [128, 512, 1024] {
            let input = vec![f32_to_bf16(0.125); rows * features];
            for fp8 in [false, true] {
                let mut samples = Vec::new();
                for run in 0..11 {
                    let input = runtime.mem_u16(&input)?;
                    let start = Instant::now();
                    let result = if fp8 {
                        artifact.exec_values(
                            "fp8-test-m64",
                            vec![
                                input.into(),
                                fp8_weight.as_ref().into(),
                                scales.as_ref().into(),
                                (features as u64).into(),
                                (outputs as u64).into(),
                            ],
                        )?
                    } else {
                        artifact.exec_values(
                            "bf16-test",
                            vec![
                                input.into(),
                                bf16_weight.as_ref().into(),
                                (features as u64).into(),
                                (outputs as u64).into(),
                            ],
                        )?
                    };
                    let elapsed = start.elapsed().as_secs_f64();
                    drop(result);
                    if run >= 2 {
                        samples.push(elapsed);
                    }
                }
                samples.sort_by(f64::total_cmp);
                eprintln!(
                    "projection M={rows} N={outputs} K={features} {}: {:.3} ms",
                    if fp8 { "FP8 + quantize" } else { "BF16" },
                    samples[4] * 1000.0
                );
            }
        }
    }
    Ok(())
}
