//! Independent CPU oracle for the complete vision tower.
use super::{VisionConfig, kernels};
use anyhow::{Result, ensure};
use catena_lang::runtime::{GpuDialect, Value};
use std::collections::BTreeMap;

fn linear(x: &[f32], w: &[f32], b: &[f32], ci: usize, co: usize) -> Vec<f32> {
    x.chunks_exact(ci)
        .flat_map(|row| {
            (0..co).map(move |j| {
                row.iter()
                    .zip(&w[j * ci..(j + 1) * ci])
                    .map(|(a, b)| a * b)
                    .sum::<f32>()
                    + b[j]
            })
        })
        .collect()
}
fn norm(x: &[f32], w: &[f32], b: &[f32], eps: f32) -> Vec<f32> {
    x.chunks_exact(w.len())
        .flat_map(|row| {
            let mean = row.iter().sum::<f32>() / row.len() as f32;
            let variance =
                row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / row.len() as f32;
            row.iter()
                .enumerate()
                .map(move |(i, v)| (v - mean) / (variance + eps).sqrt() * w[i] + b[i])
        })
        .collect()
}
fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(a, b)| a + b).collect()
}
fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + (std::f32::consts::FRAC_2_PI.sqrt() * (x + 0.044715 * x * x * x)).tanh())
}
fn gelu_erf(x: f32) -> f32 {
    // Same documented erf approximation as the default vision graph.
    let z = x * std::f32::consts::FRAC_1_SQRT_2;
    let t = 1.0 / (1.0 + 0.3275911 * z.abs());
    let poly =
        ((((1.0614054 * t - 1.4531521) * t + 1.4214138) * t - 0.28449672) * t + 0.2548296) * t;
    let erf = (1.0 - poly * (-z * z).exp()).copysign(z);
    x * 0.5 * (1.0 + erf)
}
fn oracle(
    config: &VisionConfig,
    weights: &BTreeMap<String, Vec<f32>>,
    pixels: &[f32],
    positions: &[u64],
    indices: &[u64],
    coefficients: &[f32],
    frequencies: &[f32],
) -> Vec<f32> {
    let h = config.hidden_size;
    let ci = 3 * config.patch_size * config.patch_size;
    let rows = pixels.len() / ci;
    let lin = |x: &[f32], prefix: &str, ci, co| {
        linear(
            x,
            &weights[&format!("{prefix}.weight")],
            &weights[&format!("{prefix}.bias")],
            ci,
            co,
        )
    };
    let ln = |x: &[f32], prefix: &str, eps| {
        norm(
            x,
            &weights[&format!("{prefix}.weight")],
            &weights[&format!("{prefix}.bias")],
            eps,
        )
    };
    let p0 = linear(
        pixels,
        &weights["v.patch_embd.weight"],
        &weights["v.patch_embd.bias"],
        ci,
        h,
    );
    let p1 = linear(
        pixels,
        &weights["v.patch_embd.weight.1"],
        &weights["__zero"],
        ci,
        h,
    );
    let mut x = add(&p0, &p1);
    for r in 0..rows {
        for c in 0..h {
            let terms: Vec<f32> = (0..4)
                .map(|j| {
                    weights["v.position_embd.weight"][indices[r * 4 + j] as usize * h + c]
                        * coefficients[r * 4 + j]
                })
                .collect();
            x[r * h + c] += (terms[0] + terms[1]) + (terms[2] + terms[3]);
        }
    }
    let dim = h / config.heads;
    for layer in 0..config.layers {
        let p = format!("v.blk.{layer}");
        let n = ln(&x, &format!("{p}.ln1"), config.eps);
        let qkv = lin(&n, &format!("{p}.attn_qkv"), h, 3 * h);
        let slice = |offset| {
            qkv.chunks_exact(3 * h)
                .flat_map(|r| r[offset..offset + h].iter().copied())
                .collect::<Vec<_>>()
        };
        let rotate = |input: Vec<f32>| {
            let mut out = vec![0.; input.len()];
            for i in 0..input.len() {
                let r = i / h;
                let ch = i % dim;
                let pair = ch % (dim / 2);
                let axis = pair / (dim / 4);
                let angle = positions[r * 2 + axis] as f32 * frequencies[pair % (dim / 4)];
                let (s, c) = angle.sin_cos();
                let other = if ch < dim / 2 {
                    -input[i + dim / 2]
                } else {
                    input[i - dim / 2]
                };
                out[i] = input[i] * c + other * s;
            }
            out
        };
        let q = rotate(slice(0));
        let k = rotate(slice(h));
        let v = slice(2 * h);
        let mut context = vec![0.; rows * h];
        for r in 0..rows {
            for head in 0..config.heads {
                let qb = r * h + head * dim;
                let mut scores = (0..rows)
                    .map(|key| {
                        (0..dim)
                            .map(|j| q[qb + j] * k[key * h + head * dim + j])
                            .sum::<f32>()
                            / (dim as f32).sqrt()
                    })
                    .collect::<Vec<_>>();
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                for s in &mut scores {
                    *s = (*s - max).exp();
                }
                let denominator = scores.iter().sum::<f32>();
                for j in 0..dim {
                    context[qb + j] = (0..rows)
                        .map(|key| scores[key] / denominator * v[key * h + head * dim + j])
                        .sum();
                }
            }
        }
        x = add(&x, &lin(&context, &format!("{p}.attn_out"), h, h));
        let n = ln(&x, &format!("{p}.ln2"), config.eps);
        let up = lin(&n, &format!("{p}.ffn_up"), h, config.intermediate_size)
            .into_iter()
            .map(gelu)
            .collect::<Vec<_>>();
        x = add(
            &x,
            &lin(&up, &format!("{p}.ffn_down"), config.intermediate_size, h),
        );
    }
    let x = ln(&x, "v.post_ln", 1e-6);
    let merged = h * 4;
    let x = lin(&x, "mm.0", merged, merged)
        .into_iter()
        .map(gelu_erf)
        .collect::<Vec<_>>();
    lin(&x, "mm.2", merged, config.output_size)
}

#[test]
fn vision_matches_cpu_on_gpu() -> Result<()> {
    if std::env::var_os("CATENA_QWEN3_GPU_TESTS").is_none() {
        return Ok(());
    }
    for (hidden_size, intermediate_size) in [(12, 20), (192, 320)] {
        let config = VisionConfig {
            hidden_size,
            intermediate_size,
            output_size: 10,
            layers: 2,
            heads: 3,
            patch_size: 2,
            merge_size: 2,
            position_side: 4,
            eps: 1e-6,
            min_pixels: 16,
            max_pixels: 256,
            mean: [0.5; 3],
            std: [0.5; 3],
        };
        let h = config.hidden_size;
        let ci = 12;
        let merged = h * 4;
        let mut shapes = BTreeMap::from([
            ("v.patch_embd.weight".to_owned(), ci * h),
            ("v.patch_embd.weight.1".into(), ci * h),
            ("v.patch_embd.bias".into(), h),
            ("v.position_embd.weight".into(), 16 * h),
            ("v.post_ln.weight".into(), h),
            ("v.post_ln.bias".into(), h),
            ("mm.0.weight".into(), merged * merged),
            ("mm.0.bias".into(), merged),
            ("mm.2.weight".into(), merged * config.output_size),
            ("mm.2.bias".into(), config.output_size),
            ("__zero".into(), h),
        ]);
        for l in 0..config.layers {
            for (name, input, output) in [
                ("ln1", 1, h),
                ("ln2", 1, h),
                ("attn_qkv", h, 3 * h),
                ("attn_out", h, h),
                ("ffn_up", h, config.intermediate_size),
                ("ffn_down", config.intermediate_size, h),
            ] {
                shapes.insert(format!("v.blk.{l}.{name}.weight"), input * output);
                shapes.insert(format!("v.blk.{l}.{name}.bias"), output);
            }
        }
        let mut weights = BTreeMap::new();
        for (i, (name, n)) in shapes.iter().enumerate() {
            let values = (0..*n)
                .map(|j| {
                    if name == "__zero" {
                        0.
                    } else {
                        let shift = if name.ends_with(".weight")
                            && (name.contains("ln") || name.contains("post_ln"))
                        {
                            1.
                        } else {
                            0.
                        };
                        shift + 0.15 * ((j * 17 + i * 31) as f32 * 0.113).sin()
                    }
                })
                .collect::<Vec<_>>();
            weights.insert(name.clone(), values);
        }
        let names = weights.keys().cloned().collect::<Vec<_>>();
        let source = kernels::source(&config, &names);
        let (runtime, artifact) = crate::catena::load_runtime(
            crate::catena::core_sources()
                .map(|s| -> &str { s })
                .chain([source.as_str()]),
            GpuDialect::Hip,
        )?;
        let device_weights = names
            .iter()
            .map(|n| runtime.mem_f32(&weights[n]))
            .collect::<Result<Vec<_>, _>>()?;
        let frequencies = (0..config.hidden_size / config.heads / 4)
            .map(|i| 0.4 * 0.7_f32.powi(i as i32))
            .collect::<Vec<_>>();
        let df = runtime.mem_f32(&frequencies)?;
        for rows in [4, 12, 4] {
            let pixels = (0..rows * ci)
                .map(|i| 0.8 * (i as f32 * 0.73).sin())
                .collect::<Vec<_>>();
            let positions = (0..rows)
                .flat_map(|r| [(r / 2) as u64, (r % 2) as u64])
                .collect::<Vec<_>>();
            let indices = (0..rows * 4)
                .map(|i| (i * 7 % 16) as u64)
                .collect::<Vec<_>>();
            let coefficients = [0.1, 0.2, 0.3, 0.4].repeat(rows);
            let expected = oracle(
                &config,
                &weights,
                &pixels,
                &positions,
                &indices,
                &coefficients,
                &frequencies,
            );
            let dp = runtime.mem_f32(&pixels)?;
            let dpos = runtime.mem_u64(&positions)?;
            let di = runtime.mem_u64(&indices)?;
            let dc = runtime.mem_f32(&coefficients)?;
            let mut args = vec![
                dp.as_ref().into(),
                dpos.as_ref().into(),
                di.as_ref().into(),
                dc.as_ref().into(),
                df.as_ref().into(),
            ];
            args.extend(device_weights.iter().map(|w| Value::from(w.as_ref())));
            let actual = super::one_memory(artifact.exec_values("vision.encode", args)?)?
                .try_to_f32_vec()?;
            ensure!(actual.len() == expected.len(), "output shape mismatch");
            let max = actual
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0_f32, f32::max);
            eprintln!("vision hidden={hidden_size}, rows={rows}: maximum CPU/GPU error {max}");
            ensure!(
                actual.iter().all(|x| x.is_finite()) && max < 2e-4,
                "vision CPU/GPU mismatch: {max}"
            );
        }
        let bf = runtime.mem_u16(&[0x3f80, 0xbf00, 0x0000, 0x4040])?;
        let wide = super::one_memory(
            artifact.exec_values("vision.widen-bf16", vec![bf.as_ref().into(), 4_u64.into()])?,
        )?
        .try_to_f32_vec()?;
        assert_eq!(wide, [1.0, -0.5, 0.0, 3.0]);
    }
    Ok(())
}
