//! D512 VAE attention and biased F32-interface projections, without image generation.
use anyhow::Result;
use catena_lang::runtime::{GpuDialect, Runtime, Value};
use catena_models::{
    catena::{compile_sources, core_sources},
    weights::{bf16_to_f32, f32_to_bf16},
};
use std::time::Instant;
fn source() -> String {
    catena_models::flux::graph::bundle(&[
        include_str!("../src/catena/flux_vae_wmma_linear.hex"),
        include_str!("../src/catena/flux_vae_wmma_attention.hex"),
        r#"
(def program test-attention : {((cap.own mem) val) ((cap.own mem) val) ((cap.own mem) val) (u64 val) (u64 val)} -> ((cap.own mem) val) =
 ([q k v rows keys.] {[.q] [.k] [.v] [.rows] [.keys] (const.u64.0x0000000000000001) (const.u64.0x0000000000000001) (const.u64.0x0000000000000200) (const.u32.0x3D3504F3 u32.bitcast-f32) (const.u64.0x0000000000000000) [.keys]} flux.vae-attention-f32-ooo))
(def program old-attention : {((cap.own mem) val) ((cap.own mem) val) ((cap.own mem) val) (u64 val) (u64 val)} -> ((cap.own mem) val) =
 ([q k v rows keys.] {[.q] [.k] [.v] [.rows] [.keys] (const.u64.0x0000000000000001) (const.u64.0x0000000000000001) (const.u64.0x0000000000000200) (const.u32.0x3D3504F3 u32.bitcast-f32) (const.u64.0x0000000000000000) [.keys]} flux.attention-f32-ooo))
(def program test-linear : {((cap.own mem) val) ((cap.ref mem) val) ((cap.ref mem) val) (u64 val)} -> ((cap.own mem) val) =
 ([x w b rows.] {[.x] [.w] [.b] [.rows] (const.u64.0x0000000000000200) (const.u64.0x0000000000000200)} flux.vae-linear-f32-orr))
"#,
    ].join("\n"))
}
#[test]
fn vae_attention_compiles() -> Result<()> {
    let s = source();
    let m = compile_sources(
        core_sources().map(|s| -> &str { s }).chain([s.as_str()]),
        GpuDialect::Hip,
    )?;
    assert!(
        m.source
            .contains("__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32")
    );
    Ok(())
}
#[test]
#[ignore = "requires gfx11 HIP GPU; optional isolated larger-shape timings"]
fn attention_and_projection() -> Result<()> {
    let s = source();
    let mut rt = Runtime::new(GpuDialect::Hip)?;
    let m = rt.load(compile_sources(
        core_sources().map(|s| -> &str { s }).chain([s.as_str()]),
        GpuDialect::Hip,
    )?)?;
    let make = |n: usize, salt: usize| -> Vec<f32> {
        (0..n)
            .map(|i| (((i * 17 + i / 512 * 7 + salt) % 97) as f32 - 48.) / 97.)
            .collect()
    };
    let bf = |x| bf16_to_f32(f32_to_bf16(x));
    let (rows, keys, d) = (65, 67, 512);
    let q = make(rows * d, 3);
    let mut k = make(keys * d, 7);
    let v = make(keys * d, 11);
    // Raise the online maximum in the second tile; exercise all four D128 slabs.
    k[64 * d..65 * d].copy_from_slice(&q[..d].iter().map(|x| x * 4.).collect::<Vec<_>>());
    let [Value::MemOwn(out)] = m.exec(
        "test-attention",
        [
            rt.mem_f32(&q)?.into(),
            rt.mem_f32(&k)?.into(),
            rt.mem_f32(&v)?.into(),
            (rows as u64).into(),
            (keys as u64).into(),
        ],
    )?
    else {
        panic!()
    };
    let out = out.try_to_f32_vec()?;
    for r in 0..rows {
        let scores: Vec<f32> = (0..keys)
            .map(|j| {
                (0..d)
                    .map(|c| bf(q[r * d + c]) * bf(k[j * d + c]))
                    .sum::<f32>()
                    / (d as f32).sqrt()
            })
            .collect();
        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let p: Vec<f32> = scores.iter().map(|x| (x - max).exp()).collect();
        let den: f32 = p.iter().sum();
        for c in 0..d {
            let want = (0..keys).map(|j| p[j] * bf(v[j * d + c])).sum::<f32>() / den;
            assert!(
                (out[r * d + c] - want).abs() < 0.002,
                "attention [{r},{c}]: {} vs {want}",
                out[r * d + c]
            );
        }
    }
    let x = make(rows * d, 9);
    let w = make(d * d, 17);
    let b = make(d, 29);
    let wg = rt.mem_f32(&w)?;
    let bg = rt.mem_f32(&b)?;
    let [Value::MemOwn(out)] = m.exec(
        "test-linear",
        [
            rt.mem_f32(&x)?.into(),
            wg.as_ref().into(),
            bg.as_ref().into(),
            (rows as u64).into(),
        ],
    )?
    else {
        panic!()
    };
    let out = out.try_to_f32_vec()?;
    for r in 0..rows {
        for c in 0..d {
            let want = (0..d)
                .map(|i| bf(x[r * d + i]) * bf(w[c * d + i]))
                .sum::<f32>()
                + b[c];
            assert!(
                (out[r * d + c] - want).abs() < 0.001 * want.abs().max(1.),
                "projection [{r},{c}]: {} vs {want}",
                out[r * d + c]
            );
        }
    }
    if std::env::var_os("FLUX_VAE_BENCH").is_some() {
        for rows in [1024, 4096, 16384] {
            let q = make(rows * d, 3);
            let k = make(rows * d, 7);
            let v = make(rows * d, 11);
            for name in ["old-attention", "test-attention"] {
                let a = rt.mem_f32(&q)?;
                let b = rt.mem_f32(&k)?;
                let c = rt.mem_f32(&v)?;
                let start = Instant::now();
                let _: [Value; 1] = m.exec(
                    name,
                    [
                        a.into(),
                        b.into(),
                        c.into(),
                        (rows as u64).into(),
                        (rows as u64).into(),
                    ],
                )?;
                eprintln!(
                    "VAE {name} tokens={rows}: {:.6}s",
                    start.elapsed().as_secs_f64()
                );
            }
        }
    }
    Ok(())
}
