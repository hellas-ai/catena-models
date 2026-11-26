use anyhow::Result;
use catena_lang::runtime::{GpuDialect, Runtime, RuntimeModule, Value};
use catena_models::{
    catena::{
        InferenceOptions, PrefillDtype, compile_sources, generated_model_source_for_options,
        runtime_sources,
    },
    spec::ModelConfig,
    weights::{bf16_to_f32, f32_to_bf16},
};

fn module(d: usize) -> Result<RuntimeModule> {
    module_for(d, if d == 32 { 4 } else { 2 })
}

fn module_for(d: usize, kvh: usize) -> Result<RuntimeModule> {
    module_for_kind(d, kvh, false, false)
}

fn module_for_kind(d: usize, kvh: usize, gated: bool, precise: bool) -> Result<RuntimeModule> {
    let mut config = if gated {
        ModelConfig::qwen35_0_8b()
    } else {
        ModelConfig::qwen3_0_6b()
    };
    config.num_layers = if gated { 4 } else { 1 };
    config.hidden_size = 256;
    config.intermediate_size = 512;
    config.vocabulary_size = 256;
    config.head_dim = d;
    config.rope_dimension_count = d;
    config.num_key_value_heads = kvh;
    config.num_attention_heads = config.num_key_value_heads * 2;
    let options = InferenceOptions::default()
        .with_wmma_prefill(true)
        .with_prefill_dtype(PrefillDtype::Fp8);
    let mut source = generated_model_source_for_options(&config, options)?;
    source.push_str(r#"
(def program attention-test : {((cap.own mem) val) ((cap.own mem) val) (u64 val) (u64 val) (u64 val)} -> ((cap.own mem) val) =
 ([query cache start kb vb.]
  ({[.query] [.cache] [.query] [.start] [.kb] [.vb]} stdlib.qwen3.fp8.attention.native [out.])
  ({[.query] [.out]} stdlib.models.resources.free-bf16-before [after-query.])
  ({[.cache] [.after-query]} stdlib.models.resources.free-bf16-before)
 ))
(def program bf16-attention-test : {((cap.own mem) val) ((cap.own mem) val) (u64 val) (u64 val) (u64 val)} -> ((cap.own mem) val) =
 ([query cache start kb vb.]
  ({[.query] [.cache] [.query] [.start] [.kb] [.vb]} stdlib.qwen35.wmma.attention.run [out.])
  ({[.query] [.out]} stdlib.models.resources.free-bf16-before [after-query.])
  ({[.cache] [.after-query]} stdlib.models.resources.free-bf16-before)
 ))
"#);
    if gated {
        source.push_str(r#"
(def program gated-attention-test : {((cap.own mem) val) ((cap.own mem) val) ((cap.own mem) val) (u64 val) (u64 val) (u64 val)} -> ((cap.own mem) val) =
 ([query cache qg start kb vb.]
  ({[.query] [.cache] [.qg] [.start] [.kb] [.vb]} stdlib.qwen3.fp8.attention.run [out.])
  ({[.qg] [.out]} stdlib.models.resources.free-bf16-before [after-gate.])
  ({[.query] [.after-gate]} stdlib.models.resources.free-bf16-before [after-query.])
  ({[.cache] [.after-query]} stdlib.models.resources.free-bf16-before)
 ))
"#);
    }
    source.push_str(r#"
(def program prepare-test : {((cap.own mem) val) ((cap.own mem) val) ((cap.own mem) val) (u64 val) ((cap.own mem) val) ((cap.ref mem) val) ((cap.ref mem) val) (u64 val) (u64 val)} -> {((cap.own mem) val) ((cap.own mem) val)} =
 ([q k v start cache qw kw kt vt.]
  ({[.q] [.k] [.v] [.start] [.cache] [.qw] [.kw] [.kt] [.vt]} stdlib.qwen3.fp8.attention.prepare [query updated.])
  ({[.q] [.query]} stdlib.models.resources.free-bf16-before [after-q.])
  ({[.k] [.after-q]} stdlib.models.resources.free-bf16-before [after-k.])
  ({[.v] [.after-k]} stdlib.models.resources.free-bf16-before [after-v.])
  {[.after-v] [.updated]}
 ))
"#);
    if precise {
        source = source.replace(
            "stdlib.qwen3.fp8.attention.native [out.]",
            "stdlib.qwen3.fp8.precise-attention.native [out.]",
        );
    }
    let source = source
        .replace("stdlib.qwen3.", "stdlib.fixture-internal.")
        .replace("qwen3.", "stdlib.fixture-public.")
        .replace("qwen3 :", "stdlib.fixture-root :");
    compile_sources(runtime_sources(&source), GpuDialect::Cuda)
}

#[test]
fn attention_shuffle_and_mma_compile() -> Result<()> {
    let module = module(128)?;
    assert!(module.source.contains("__shfl_xor_sync"));
    assert!(
        module
            .source
            .contains("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32")
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA SM89+ and nvcc"]
fn fp8_attention_matches_independent_causal_reference() -> Result<()> {
    causal_reference(false, false)
}

#[test]
#[ignore = "requires CUDA SM89+ and nvcc"]
fn fp8_gated_attention_matches_independent_causal_reference() -> Result<()> {
    causal_reference(true, false)
}

#[test]
#[ignore = "requires CUDA SM89+ and nvcc"]
fn fp8_precise_attention_matches_independent_causal_reference() -> Result<()> {
    causal_reference(false, true)
}

fn causal_reference(gated: bool, precise: bool) -> Result<()> {
    let mut runtime = Runtime::new(GpuDialect::Cuda)?;
    let dims: &[usize] = &[32, 64, 128, 256];
    for &d in dims {
        let artifact = runtime.load(module_for_kind(
            d,
            if d == 32 { 4 } else { 2 },
            gated,
            precise,
        )?)?;
        let kvh = if d == 32 { 4 } else { 2 };
        let qh = kvh * 2;
        let kvsize = kvh * d;
        for (rows, start, amplitude) in [
            (2, 0, 1.0),
            (17, 5, 1.0),
            (33, 0, 0.0),
            (65, 0, 1.0),
            (64, 67, 1.0),
            (129, 128, 3.0),
        ] {
            let keys = start + rows;
            let capacity = keys + 17;
            let kb = 16;
            let vb = kb + capacity * kvsize + 32;
            let query: Vec<_> = (0..rows * qh * d)
                .map(|i| {
                    f32_to_bf16((((i * 17 + i / d * 7) % 97) as f32 - 48.0) / 64.0 * amplitude)
                })
                .collect();
            let mut cache = vec![0x7fc1; vb + capacity * kvsize];
            for key in 0..keys {
                for h in 0..kvh {
                    for ch in 0..d {
                        cache[kb + key * kvsize + h * d + ch] = f32_to_bf16(
                            (((key * 23 + h * 3 + ch * 11) % 89) as f32 - 44.0) / 64.0 * amplitude,
                        );
                        cache[vb + key * kvsize + h * d + ch] = f32_to_bf16(
                            (((key * 7 + h * 17 + ch * 13) % 113) as f32 - 51.0) / 64.0,
                        );
                    }
                }
            }
            let q = runtime.mem_u16(&query)?;
            let c = runtime.mem_u16(&cache)?;
            let qg: Vec<_> = (0..rows * qh * d * 2)
                .map(|i| f32_to_bf16(((i * 7 % 101) as f32 - 50.0) / 16.0))
                .collect();
            let result = if gated {
                artifact.exec(
                    "gated-attention-test",
                    [
                        q.into(),
                        c.into(),
                        runtime.mem_u16(&qg)?.into(),
                        (start as u64).into(),
                        (kb as u64).into(),
                        (vb as u64).into(),
                    ],
                )?
            } else {
                artifact.exec(
                    "attention-test",
                    [
                        q.into(),
                        c.into(),
                        (start as u64).into(),
                        (kb as u64).into(),
                        (vb as u64).into(),
                    ],
                )?
            };
            let [Value::MemOwn(out)] = result else {
                anyhow::bail!("expected output");
            };
            let actual = out.try_to_u16_vec()?;
            assert_eq!(actual.len(), query.len());
            let mut error = 0.0_f64;
            let mut norm = 0.0_f64;
            let mut dot = 0.0_f64;
            let mut actual_norm = 0.0_f64;
            for row in 0..rows {
                for h in 0..qh {
                    let head = h / 2;
                    let scores: Vec<f32> = (0..=start + row)
                        .map(|key| {
                            (0..d)
                                .map(|ch| {
                                    bf16_to_f32(query[(row * qh + h) * d + ch])
                                        * bf16_to_f32(cache[kb + key * kvsize + head * d + ch])
                                })
                                .sum::<f32>()
                                / (d as f32).sqrt()
                        })
                        .collect();
                    let scores: Vec<_> = scores
                        .into_iter()
                        .map(|s| {
                            if precise {
                                bf16_to_f32(f32_to_bf16(s))
                            } else {
                                s
                            }
                        })
                        .collect();
                    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let exps: Vec<_> = scores.iter().map(|s| (s - max).exp()).collect();
                    let sum: f32 = exps.iter().sum();
                    for ch in 0..d {
                        let expected: f32 = exps
                            .iter()
                            .enumerate()
                            .map(|(key, &p)| {
                                let probability = p / sum;
                                let probability = if precise && keys <= 64 {
                                    bf16_to_f32(f32_to_bf16(probability))
                                } else {
                                    probability
                                };
                                probability * bf16_to_f32(cache[vb + key * kvsize + head * d + ch])
                            })
                            .sum();
                        let expected = if gated {
                            let gate = bf16_to_f32(qg[row * qh * d * 2 + h * d * 2 + d + ch]);
                            bf16_to_f32(f32_to_bf16(expected)) / (1.0 + (-gate).exp())
                        } else {
                            expected
                        };
                        let got = bf16_to_f32(actual[(row * qh + h) * d + ch]);
                        assert!(
                            got.is_finite(),
                            "d{d} rows{rows} start{start}: non-finite output"
                        );
                        error += (got as f64 - expected as f64).powi(2);
                        norm += (expected as f64).powi(2);
                        dot += got as f64 * expected as f64;
                        actual_norm += (got as f64).powi(2);
                    }
                }
            }
            let rms = (error / norm).sqrt();
            let cosine = dot / (norm * actual_norm).sqrt();
            eprintln!(
                "D={d} rows={rows} start={start} amplitude={amplitude}: NRMSE={rms:.6}, cosine={cosine:.6}"
            );
            assert!(
                rms < if precise { 0.004 } else { 0.12 } && cosine > 0.99,
                "attention accuracy: D={d} rows={rows} start={start} rms={rms} cosine={cosine}"
            );
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA SM89+ and nvcc"]
fn warp_prepare_preserves_bf16_kv_and_absolute_rope() -> Result<()> {
    let mut runtime = Runtime::new(GpuDialect::Cuda)?;
    for d in [32, 64, 128, 256] {
        let artifact = runtime.load(module(d)?)?;
        let kvh = if d == 32 { 4 } else { 2 };
        let qh = kvh * 2;
        let kvsize = kvh * d;
        for (rows, start) in [(3, 0), (17, 67)] {
            let capacity = start + rows + 7;
            let kb = 16;
            let vb = kb + capacity * kvsize + 32;
            let q: Vec<_> = (0..rows * qh * d)
                .map(|i| f32_to_bf16(((i * 17 % 97) as f32 - 48.0) / 64.0))
                .collect();
            let k: Vec<_> = (0..rows * kvsize)
                .map(|i| f32_to_bf16(((i * 13 % 89) as f32 - 44.0) / 64.0))
                .collect();
            let v: Vec<_> = (0..rows * kvsize)
                .map(|i| f32_to_bf16(((i * 7 % 113) as f32 - 51.0) / 64.0))
                .collect();
            let qw: Vec<_> = (0..d).map(|i| 0.8 + (i % 7) as f32 / 32.0).collect();
            let kw: Vec<_> = (0..d).map(|i| 0.7 + (i % 11) as f32 / 32.0).collect();
            let initial = vec![0x3e80; vb + capacity * kvsize];
            let qm = runtime.mem_u16(&q)?;
            let km = runtime.mem_u16(&k)?;
            let vm = runtime.mem_u16(&v)?;
            let cm = runtime.mem_u16(&initial)?;
            let qwm = runtime.mem_f32(&qw)?;
            let kwm = runtime.mem_f32(&kw)?;
            let mut result = artifact.exec_values(
                "prepare-test",
                vec![
                    qm.into(),
                    km.into(),
                    vm.into(),
                    (start as u64).into(),
                    cm.into(),
                    qwm.as_ref().into(),
                    kwm.as_ref().into(),
                    ((kb + start * kvsize) as u64).into(),
                    ((vb + start * kvsize) as u64).into(),
                ],
            )?;
            let Value::MemOwn(cache) = result.pop().unwrap() else {
                anyhow::bail!("cache");
            };
            let Value::MemOwn(query) = result.pop().unwrap() else {
                anyhow::bail!("query");
            };
            let cache = cache.try_to_u16_vec()?;
            let query = query.try_to_u16_vec()?;
            let reference = |values: &[u16], weights: &[f32], heads: usize| -> Vec<f32> {
                let mut output = Vec::new();
                for row in 0..rows {
                    for h in 0..heads {
                        let x: Vec<_> = values[(row * heads + h) * d..(row * heads + h + 1) * d]
                            .iter()
                            .copied()
                            .map(bf16_to_f32)
                            .collect();
                        let variance = x.iter().map(|v| v * v).sum::<f32>() / d as f32 + 1e-6;
                        let inverse = 1.0 / variance.sqrt();
                        let n: Vec<_> = x
                            .iter()
                            .enumerate()
                            .map(|(i, x)| {
                                bf16_to_f32(f32_to_bf16(
                                    bf16_to_f32(f32_to_bf16(x * inverse))
                                        * bf16_to_f32(f32_to_bf16(weights[i])),
                                ))
                            })
                            .collect();
                        let mut rotated = vec![0.0; d];
                        for ch in 0..d / 2 {
                            let frequency = 1_000_000.0_f32.powf(-((2 * ch) as f32) / d as f32);
                            let angle = (start + row) as f32 * frequency;
                            let (sin, cos) = angle.sin_cos();
                            rotated[ch] =
                                bf16_to_f32(f32_to_bf16(n[ch] * cos - n[ch + d / 2] * sin));
                            rotated[ch + d / 2] =
                                bf16_to_f32(f32_to_bf16(n[ch + d / 2] * cos + n[ch] * sin));
                        }
                        output.extend(rotated);
                    }
                }
                output
            };
            let qr = reference(&q, &qw, qh);
            let kr = reference(&k, &kw, kvh);
            for (i, &bits) in query.iter().enumerate() {
                let actual = bf16_to_f32(bits);
                assert!(
                    (actual - qr[i]).abs() < 0.025,
                    "Q D={d} rows={rows} start={start} i={i}: {actual} vs {}",
                    qr[i]
                );
            }
            for i in 0..cache.len() {
                if (kb + start * kvsize..kb + (start + rows) * kvsize).contains(&i) {
                    let actual = bf16_to_f32(cache[i]);
                    let expected = kr[i - kb - start * kvsize];
                    assert!(
                        (actual - expected).abs() < 0.025,
                        "K D={d} i={i}: {actual} vs {expected}"
                    );
                } else if (vb + start * kvsize..vb + (start + rows) * kvsize).contains(&i) {
                    assert_eq!(cache[i], v[i - vb - start * kvsize]);
                } else {
                    assert_eq!(cache[i], initial[i], "unwritten KV changed at {i}");
                }
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "manual CUDA attention benchmark"]
fn fp8_attention_benchmark() -> Result<()> {
    use std::time::Instant;
    let mut runtime = Runtime::new(GpuDialect::Cuda)?;
    let artifact = runtime.load(module_for(128, 8)?)?;
    let d = 128;
    let qh = 16;
    let kvh = 8;
    let kvsize = d * kvh;
    for rows in [64, 128, 512, 1024] {
        let query: Vec<_> = (0..rows * qh * d)
            .map(|i| f32_to_bf16(((i * 17 % 97) as f32 - 48.0) / 64.0))
            .collect();
        let cache: Vec<_> = (0..rows * kvsize * 2)
            .map(|i| f32_to_bf16(((i * 13 % 89) as f32 - 44.0) / 64.0))
            .collect();
        for name in ["bf16-attention-test", "attention-test"] {
            let mut samples = Vec::new();
            for run in 0..11 {
                let q = runtime.mem_u16(&query)?;
                let c = runtime.mem_u16(&cache)?;
                let t = Instant::now();
                let [out] = artifact.exec(
                    name,
                    [
                        q.into(),
                        c.into(),
                        0u64.into(),
                        0u64.into(),
                        ((rows * kvsize) as u64).into(),
                    ],
                )?;
                let dt = t.elapsed().as_secs_f64();
                drop(out);
                if run >= 2 {
                    samples.push(dt);
                }
            }
            samples.sort_by(f64::total_cmp);
            eprintln!(
                "rows={rows} {name}: {:.3} ms median including snapshots",
                samples[4] * 1000.0
            );
        }
    }
    Ok(())
}
