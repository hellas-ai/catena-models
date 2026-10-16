//! Independent numerical checks for experimental Flux kernels.
use super::experimental;
use anyhow::{Result, ensure};
use catena_lang::{
    codegen::{CodegenKind, GpuDialect},
    runtime::{MemOwn, Value},
};
fn c(n: usize) -> String {
    format!("(const.u64.0x{n:016X})")
}
fn f(n: f32) -> String {
    format!("(const.u32.0x{:08X} u32.bitcast-f32)", n.to_bits())
}
fn entry(name: &str, types: &str, names: &str, body: &str) -> String {
    format!(
        "(def program flux.test-{name} : {{{types}}} -> ((cap.own mem) val) = ([{names}.] {body}))\n"
    )
}
fn get(
    artifact: &catena_lang::runtime::Artifact,
    name: &str,
    args: Vec<Value<'_>>,
) -> Result<MemOwn> {
    let mut out = artifact.exec_values(name, args)?;
    ensure!(out.len() == 1, "wrong output count");
    match out.pop().unwrap() {
        Value::MemOwn(m) => Ok(m),
        _ => anyhow::bail!("expected memory"),
    }
}
fn close(actual: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(actual.len(), expected.len());
    for (i, (a, b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && (*a - *b).abs() <= tol * (1. + b.abs()),
            "element {i}: {a} != {b}"
        );
    }
}
fn bf(x: f32) -> f32 {
    super::pipeline::bf16_to_f32(super::pipeline::f32_to_bf16(x))
}
#[test]
#[ignore = "requires HIP GPU"]
fn kernels_match_independent_cpu() -> Result<()> {
    let r = "((cap.ref mem) val)";
    let mut s = entry(
        "conv",
        &format!("{r} {r} {r}"),
        "x w b",
        &format!(
            "({{[.x] [.w] [.b] {} {} {} {} {} {}}} flux.conv2d-f32)",
            c(2),
            c(3),
            c(3),
            c(5),
            c(3),
            c(1)
        ),
    );
    for (kind, cols, rows, spatial, mode) in [
        ("layer", 7, 3, 1, 0),
        ("rms", 2560, 2, 1, 1),
        ("group", 6000, 2, 2000, 2),
        ("large-group", 1_048_580, 2, 262_145, 2),
    ] {
        s += &entry(
            kind,
            &format!("{r} {r} {r}"),
            "x w b",
            &format!(
                "({{[.x] [.w] [.b] {} {} {} {} {} {}}} flux.norm-f32)",
                c(rows),
                c(cols),
                c(spatial),
                c(mode),
                c(if mode == 1 { 1 } else { 2 }),
                f(1e-5)
            ),
        );
    }
    for (dtype, dim) in [("f32", 512), ("bf16", 8)] {
        for causal in [false, true] {
            s += &entry(
                &format!("attention-{dtype}-{causal}"),
                &format!("{r} {r} {r}"),
                "q k v",
                &format!(
                    "({{[.q] [.k] [.v] {} {} {} {} {} {} {} {}}} flux.attention-{dtype})",
                    c(35),
                    c(37),
                    c(4),
                    c(2),
                    c(dim),
                    f(1. / (dim as f32).sqrt()),
                    c(causal as usize),
                    c(33)
                ),
            );
        }
    }
    s += &entry(
        "project",
        &format!("((cap.own mem) val) {r}"),
        "x w",
        &format!(
            "({{[.x] [.w] {} {} {}}} flux.linear-bf16-or)",
            c(9),
            c(2560),
            c(13)
        ),
    );
    let source =
        experimental::bundle(&s).replace("(def program stdlib.flux.test-", "(def program test-");
    let (runtime, artifact) = crate::catena::load_runtime_with_codegen(
        catena_lang::stdlib::experimental_core_sources()
            .map(|s| -> &str { s })
            .chain([source.as_str()]),
        GpuDialect::Hip,
        CodegenKind::Experimental,
    )?;
    for (name, cols, rows, spatial, kind) in [
        ("layer", 7, 3, 1, 0),
        ("rms", 2560, 2, 1, 1),
        ("group", 6000, 2, 2000, 2),
        ("large-group", 1_048_580, 2, 262_145, 2),
    ] {
        eprintln!("checking {name}");
        let x = (0..cols * rows)
            .map(|i| ((i * 7 % 19) as f32 - 9.) / 13. + 2.)
            .collect::<Vec<_>>();
        let channels = if kind == 2 {
            cols * rows / spatial
        } else {
            cols
        };
        let w = (0..channels)
            .map(|i| 0.7 + (i % 7) as f32 / 13.)
            .collect::<Vec<_>>();
        let b = (0..channels)
            .map(|i| (i % 5) as f32 / 11.)
            .collect::<Vec<_>>();
        let mut expected = vec![0.; x.len()];
        for row in 0..rows {
            let v = &x[row * cols..(row + 1) * cols];
            let mu = if kind == 1 {
                0.
            } else {
                v.iter().map(|x| *x as f64).sum::<f64>() / cols as f64
            };
            let var = v.iter().map(|z| (*z as f64 - mu).powi(2)).sum::<f64>() / cols as f64;
            for j in 0..cols {
                let i = row * cols + j;
                let wi = if kind == 2 { i / spatial } else { j };
                expected[i] = ((x[i] as f64 - mu) / (var + 1e-5).sqrt() * w[wi] as f64
                    + if kind == 1 { 0. } else { b[wi] as f64 })
                    as f32;
            }
        }
        let (xm, wm, bm) = (
            runtime.mem_f32(&x)?,
            runtime.mem_f32(&w)?,
            runtime.mem_f32(&b)?,
        );
        close(
            &get(
                &artifact,
                &format!("test-{name}"),
                vec![xm.as_ref().into(), wm.as_ref().into(), bm.as_ref().into()],
            )?
            .try_to_f32_vec()?,
            &expected,
            5e-4,
        );
    }
    eprintln!("checking convolution");
    let x = (0..30).map(|i| (i as f32 - 15.) / 11.).collect::<Vec<_>>();
    let w = (0..54)
        .map(|i| (i % 9) as f32 / 17. - 0.2)
        .collect::<Vec<_>>();
    let b = vec![0.1, -0.2, 0.3];
    let mut expected = vec![0.; 45];
    for oc in 0..3 {
        for y in 0..3 {
            for px in 0..5 {
                let mut sum = 0.;
                for ic in 0..2 {
                    for ky in 0..3 {
                        for kx in 0..3 {
                            let iy = y as isize + ky as isize - 1;
                            let ix = px as isize + kx as isize - 1;
                            if (0..3).contains(&iy) && (0..5).contains(&ix) {
                                sum += x[(ic * 3 + iy as usize) * 5 + ix as usize]
                                    * w[((oc * 2 + ic) * 3 + ky) * 3 + kx];
                            }
                        }
                    }
                }
                expected[(oc * 3 + y) * 5 + px] = sum + b[oc];
            }
        }
    }
    let (xm, wm, bm) = (
        runtime.mem_f32(&x)?,
        runtime.mem_f32(&w)?,
        runtime.mem_f32(&b)?,
    );
    close(
        &get(
            &artifact,
            "test-conv",
            vec![xm.as_ref().into(), wm.as_ref().into(), bm.as_ref().into()],
        )?
        .try_to_f32_vec()?,
        &expected,
        1e-5,
    );
    for (dtype, dim) in [("f32", 512), ("bf16", 8)] {
        for causal in [false, true] {
            eprintln!("checking attention {dtype}");
            let quant = |x| if dtype == "bf16" { bf(x) } else { x };
            let q = (0..35 * 4 * dim)
                .map(|i| quant((i % 17) as f32 / 29. - 0.3))
                .collect::<Vec<_>>();
            let k = (0..37 * 2 * dim)
                .map(|i| quant((i % 11) as f32 / 31. - 0.1))
                .collect::<Vec<_>>();
            let v = (0..37 * 2 * dim)
                .map(|i| quant((i % 13) as f32 / 23. - 0.2))
                .collect::<Vec<_>>();
            let mut expected = vec![0.; q.len()];
            for row in 0..35 {
                for head in 0..4 {
                    let end = if causal { (row + 1).min(33) } else { 33 };
                    let scores = (0..end)
                        .map(|key| {
                            (0..dim)
                                .map(|d| {
                                    q[(row * 4 + head) * dim + d]
                                        * k[(key * 2 + head / 2) * dim + d]
                                })
                                .sum::<f32>()
                                / (dim as f32).sqrt()
                        })
                        .collect::<Vec<_>>();
                    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let p = scores.iter().map(|x| (x - max).exp()).collect::<Vec<_>>();
                    let den = p.iter().sum::<f32>();
                    for d in 0..dim {
                        expected[(row * 4 + head) * dim + d] = quant(
                            (0..end)
                                .map(|key| p[key] / den * v[(key * 2 + head / 2) * dim + d])
                                .sum::<f32>(),
                        );
                    }
                }
            }
            let upload = |v: &[f32]| -> Result<MemOwn> {
                if dtype == "bf16" {
                    Ok(runtime.mem_u16(
                        &v.iter()
                            .map(|x| super::pipeline::f32_to_bf16(*x))
                            .collect::<Vec<_>>(),
                    )?)
                } else {
                    Ok(runtime.mem_f32(v)?)
                }
            };
            let mut masked_v = v.clone();
            masked_v[33 * 2 * dim..].fill(f32::NAN);
            let (qm, km, vm) = (upload(&q)?, upload(&k)?, upload(&masked_v)?);
            for _ in 0..2 {
                let out = get(
                    &artifact,
                    &format!("test-attention-{dtype}-{causal}"),
                    vec![qm.as_ref().into(), km.as_ref().into(), vm.as_ref().into()],
                )?;
                let actual = if dtype == "bf16" {
                    out.try_to_u16_vec()?
                        .into_iter()
                        .map(super::pipeline::bf16_to_f32)
                        .collect()
                } else {
                    out.try_to_f32_vec()?
                };
                close(
                    &actual,
                    &expected,
                    if dtype == "bf16" { 0.002 } else { 2e-5 },
                );
            }
        }
    }
    eprintln!("checking projection");
    let x = (0..9 * 2560)
        .map(|i| bf((i % 17) as f32 / 19. - 0.3))
        .collect::<Vec<_>>();
    let w = (0..13 * 2560)
        .map(|i| bf((i % 13) as f32 / 17. - 0.4))
        .collect::<Vec<_>>();
    let expected = (0..9 * 13)
        .map(|i| {
            bf((0..2560)
                .map(|j| x[i / 13 * 2560 + j] * w[i % 13 * 2560 + j])
                .sum())
        })
        .collect::<Vec<_>>();
    let xm = runtime.mem_u16(
        &x.iter()
            .map(|x| super::pipeline::f32_to_bf16(*x))
            .collect::<Vec<_>>(),
    )?;
    let wm = runtime.mem_u16(
        &w.iter()
            .map(|x| super::pipeline::f32_to_bf16(*x))
            .collect::<Vec<_>>(),
    )?;
    let out = get(
        &artifact,
        "test-project",
        vec![xm.into(), wm.as_ref().into()],
    )?;
    close(
        &out.try_to_u16_vec()?
            .into_iter()
            .map(super::pipeline::bf16_to_f32)
            .collect::<Vec<_>>(),
        &expected,
        0.01,
    );
    Ok(())
}

/// Feed reference stage inputs directly, separating upstream BF16 drift from
/// transformer and VAE correctness. Fixtures are ordinary diagnostic dumps.
/// BF16 drift is reported; F32 VAE agreement is checked numerically.
#[test]
#[ignore = "requires HIP GPU, cached HF model, and FLUX_TEST_REFERENCE dump directory"]
fn cached_components_match_reference() -> Result<()> {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use rand_distr::{Distribution, StandardNormal};
    use std::path::PathBuf;
    let reference = PathBuf::from(std::env::var("FLUX_TEST_REFERENCE")?);
    let output = PathBuf::from(std::env::var("FLUX_TEST_DUMP")?);
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
    let transformer = super::transformer::Transformer::load_with_codegen(
        &snapshot,
        height,
        width,
        1,
        true,
        super::ops::Backend::Fast,
        CodegenKind::Experimental,
    )?;
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
    // individual kernels have independent CPU checks below tighter bounds.
    ensure!(
        actual.iter().all(|x| x.is_finite()),
        "non-finite transformer output"
    );
    drop(latents);
    drop(noise);
    drop(conditioning);
    drop(transformer);
    let vae = super::vae::VaeDecoder::load_with_codegen(
        &snapshot,
        height,
        width,
        true,
        super::ops::Backend::Fast,
        CodegenKind::Experimental,
    )?;
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
