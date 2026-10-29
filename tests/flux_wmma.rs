use catena_lang::runtime::{GpuDialect, Runtime, Value};
use catena_models::{
    catena::{compile_sources, core_sources},
    weights::{bf16_to_f32, f32_to_bf16},
};
fn source() -> String {
    catena_models::flux::graph::bundle(&[
        include_str!("../src/catena/flux_wmma_linear.hex"),
        r#"(def program test-linear : {((cap.ref mem) val) ((cap.ref mem) val) (u64 val) (u64 val) (u64 val)} -> ((cap.own mem) val) = ([x w m k n.] {[.x] [.w] [.m] [.k] [.n]} flux.wmma-linear-bf16))"#,
    ].join("\n"))
}
#[test]
fn linear_compiles() -> anyhow::Result<()> {
    let s = source();
    let module = compile_sources(
        core_sources().map(|s| -> &str { s }).chain([s.as_str()]),
        GpuDialect::Hip,
    )?;
    assert!(
        module
            .source
            .contains("__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32")
    );
    Ok(())
}
#[test]
#[ignore = "requires gfx11 HIP GPU"]
fn linear_matches_cpu() -> anyhow::Result<()> {
    let s = source();
    let mut rt = Runtime::new(GpuDialect::Hip)?;
    let module = rt.load(compile_sources(
        core_sources().map(|s| -> &str { s }).chain([s.as_str()]),
        GpuDialect::Hip,
    )?)?;
    for (m, k, n) in [
        (1, 32, 17),
        (17, 16, 19),
        (17, 32, 19),
        (65, 96, 67),
        (17, 3072, 32),
    ] {
        let x: Vec<_> = (0..m * k)
            .map(|i| f32_to_bf16((i as i32 % 29 - 14) as f32 / 32.))
            .collect();
        let w: Vec<_> = (0..k * n)
            .map(|i| f32_to_bf16((i as i32 % 31 - 15) as f32 / 64.))
            .collect();
        let a = rt.mem_u16(&x)?;
        let b = rt.mem_u16(&w)?;
        let [Value::MemOwn(out)] = module.exec(
            "test-linear",
            [
                a.as_ref().into(),
                b.as_ref().into(),
                (m as u64).into(),
                (k as u64).into(),
                (n as u64).into(),
            ],
        )?
        else {
            panic!()
        };
        let out = out.try_to_u16_vec()?;
        for row in 0..m {
            for col in 0..n {
                let want = (0..k)
                    .map(|i| bf16_to_f32(x[row * k + i]) * bf16_to_f32(w[col * k + i]))
                    .sum::<f32>();
                let got = bf16_to_f32(out[row * n + col]);
                assert!(
                    (got - want).abs() <= 0.005 * want.abs().max(1.),
                    "{m}x{k}x{n} [{row},{col}] {got} != {want}"
                );
            }
        }
    }
    Ok(())
}

fn attention_source() -> String {
    let entry = r#"(def program test-attention : {((cap.ref mem) val) ((cap.ref mem) val) ((cap.ref mem) val) (u64 val) (u64 val) (u64 val) (u64 val) (u64 val) (u64 val)} -> ((cap.own mem) val) =
      ([q k v rows keys heads kvheads causal valid.]
       {[.q] [.k] [.v] [.rows] [.keys] [.heads] [.kvheads] (const.u64.0x0000000000000080) (const.u32.0x3DB504F3 u32.bitcast-f32) [.causal] [.valid]} flux.wmma-fused-attention-bf16))"#;
    catena_models::flux::graph::bundle(
        &[include_str!("../src/catena/flux_wmma_attention.hex"), entry].join("\n"),
    )
}
#[test]
fn attention_compiles() -> anyhow::Result<()> {
    let source = attention_source();
    let module = compile_sources(
        core_sources()
            .map(|s| -> &str { s })
            .chain([source.as_str()]),
        GpuDialect::Hip,
    )?;
    assert!(module.source.contains("catena_wmma_f32x8_t{{"));
    Ok(())
}
#[test]
#[ignore = "requires gfx11 HIP GPU"]
fn attention_matches_cpu() -> anyhow::Result<()> {
    let source = attention_source();
    let mut rt = Runtime::new(GpuDialect::Hip)?;
    let module = rt.load(compile_sources(
        core_sources()
            .map(|s| -> &str { s })
            .chain([source.as_str()]),
        GpuDialect::Hip,
    )?)?;
    let dim = 128;
    for (rows, keys, heads, kvheads, causal, valid) in [
        (1, 17, 2, 2, false, 17),
        (65, 67, 2, 2, false, 67),
        (65, 67, 4, 2, true, 37),
        (17, 67, 32, 8, true, 67),
    ] {
        let make = |n: usize| -> Vec<u16> {
            (0..n)
                .map(|i| f32_to_bf16((i as i32 % 31 - 15) as f32 / 32.))
                .collect()
        };
        let q = make(rows * heads * dim);
        let mut k = make(keys * kvheads * dim);
        if keys > 64 {
            // Force a later tile to raise the maximum; causal/padding masks
            // must also prevent this key affecting rows that cannot attend it.
            for head in 0..kvheads {
                for ch in 0..dim {
                    k[(64 * kvheads + head) * dim + ch] =
                        f32_to_bf16(2. * bf16_to_f32(q[(head * (heads / kvheads)) * dim + ch]));
                }
            }
        }
        let v = make(keys * kvheads * dim);
        let a = rt.mem_u16(&q)?;
        let b = rt.mem_u16(&k)?;
        let c = rt.mem_u16(&v)?;
        let [Value::MemOwn(out)] = module.exec(
            "test-attention",
            [
                a.as_ref().into(),
                b.as_ref().into(),
                c.as_ref().into(),
                (rows as u64).into(),
                (keys as u64).into(),
                (heads as u64).into(),
                (kvheads as u64).into(),
                u64::from(causal).into(),
                (valid as u64).into(),
            ],
        )?
        else {
            panic!()
        };
        let out = out.try_to_u16_vec()?;
        for row in 0..rows {
            for head in 0..heads {
                let kvhead = head / (heads / kvheads);
                let limit = valid.min(keys).min(if causal { row + 1 } else { keys });
                let scores: Vec<f32> = (0..limit)
                    .map(|key| {
                        (0..dim)
                            .map(|ch| {
                                bf16_to_f32(q[(row * heads + head) * dim + ch])
                                    * bf16_to_f32(k[(key * kvheads + kvhead) * dim + ch])
                            })
                            .sum::<f32>()
                            / (dim as f32).sqrt()
                    })
                    .collect();
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let probs: Vec<f32> = scores.iter().map(|score| (score - max).exp()).collect();
                let denominator = probs.iter().sum::<f32>();
                for ch in 0..dim {
                    let want = (0..limit)
                        .map(|key| {
                            probs[key] / denominator
                                * bf16_to_f32(v[(key * kvheads + kvhead) * dim + ch])
                        })
                        .sum::<f32>();
                    let got = bf16_to_f32(out[(row * heads + head) * dim + ch]);
                    assert!(
                        (got - want).abs() <= 0.005,
                        "rows={rows} keys={keys} heads={heads}/{kvheads} causal={causal} valid={valid} row={row} head={head} ch={ch}: {got} vs {want}"
                    );
                }
            }
        }
    }
    Ok(())
}
