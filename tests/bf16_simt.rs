use anyhow::{Result, ensure};
use catena_lang::runtime::{GpuDialect, Runtime, Value};
use catena_models::{
    catena::{compile_sources, core_sources},
    weights::{bf16_to_f32, f32_to_bf16},
};

fn sources() -> Vec<String> {
    let mut source = String::new();
    for (mode, x, w) in [
        ("or", "own", "ref"),
        ("rr", "ref", "ref"),
        ("ro", "ref", "own"),
        ("oo", "own", "own"),
    ] {
        source += &format!(
            "\n(def program test-{mode} : {{((cap.{x} mem) val) ((cap.{w} mem) val) (u64 val) (u64 val) (u64 val)}} -> ((cap.own mem) val) = ([x w m k n.] {{[.x] [.w] [.m] [.k] [.n]}} stdlib.models.simt.{mode}))\n"
        );
    }
    source += "\n(def program test-or-into : {((cap.own mem) val) ((cap.ref mem) val) (u64 val) (u64 val) (u64 val) ((cap.own mem) val)} -> ((cap.own mem) val) = ([x w m k n out.] {[.x] [.w] [.m] [.k] [.n] [.out]} stdlib.models.simt.or.into))";
    vec![catena_models::flux::graph::bundle(&source)]
}

#[test]
fn portable_matrix_compiles() -> Result<()> {
    let source = sources();
    for dialect in [GpuDialect::Hip, GpuDialect::Cuda] {
        let module = compile_sources(
            core_sources()
                .map(|s| -> &str { s })
                .chain(source.iter().map(String::as_str)),
            dialect,
        )?;
        ensure!(!module.source.contains("catena_bf16_dot"));
        ensure!(!module.source.contains("catena_wmma_mma"));
    }
    Ok(())
}

#[test]
#[ignore = "requires a GPU"]
fn matrix_matches_reference() -> Result<()> {
    let dialect = catena_models::model::configured_gpu_dialect()?;
    let mut rt = Runtime::new(dialect)?;
    let source = sources();
    let artifact = rt.load(compile_sources(
        core_sources()
            .map(|s| -> &str { s })
            .chain(source.iter().map(String::as_str)),
        dialect,
    )?)?;
    for (m, k, n) in [
        (1usize, 1usize, 1usize),
        (1, 2, 5),
        (3, 63, 7),
        (4, 64, 4),
        (5, 65, 9),
        (9, 258, 5),
        (17, 192, 33),
        (4, 2048, 16),
    ] {
        let data = |len, salt| {
            (0..len)
                .map(|i| {
                    f32_to_bf16(
                        ((((i * 17 + salt) % 71) as f32 - 35.) / 32.)
                            * 2f32.powi(((i * 7 + salt) % 17) as i32 - 8),
                    )
                })
                .collect::<Vec<_>>()
        };
        let mut x = data(m * k, 3);
        x.extend(vec![0x7fc0; 2 * k]); // inactive input capacity must not become extra rows
        let w = data(k * n, 11);
        let mut expected = Vec::new();
        for row in 0..m {
            for col in 0..n {
                let mut partial = [0f32; 32];
                for lane in 0..32 {
                    let mut sum = [0f32; 2];
                    for pair in (lane..k.div_ceil(2)).step_by(32) {
                        for j in 0..2 {
                            let ch = pair * 2 + j;
                            if ch < k {
                                sum[j] +=
                                    bf16_to_f32(x[row * k + ch]) * bf16_to_f32(w[col * k + ch]);
                            }
                        }
                    }
                    partial[lane] = sum[0] + sum[1];
                }
                for stride in [16, 8, 4, 2, 1] {
                    for lane in 0..stride {
                        partial[lane] += partial[lane + stride];
                    }
                }
                expected.push(f32_to_bf16(partial[0]));
            }
        }
        for mode in ["or", "rr", "ro", "oo"] {
            let xx = rt.mem_u16(&x)?;
            let ww = rt.mem_u16(&w)?;
            let xv: Value = if mode.starts_with('o') {
                xx.into()
            } else {
                xx.as_ref().into()
            };
            let wv: Value = if mode.ends_with('o') {
                ww.into()
            } else {
                ww.as_ref().into()
            };
            let [Value::MemOwn(out)] = artifact.exec(
                &format!("test-{mode}"),
                [
                    xv,
                    wv,
                    (m as u64).into(),
                    (k as u64).into(),
                    (n as u64).into(),
                ],
            )?
            else {
                anyhow::bail!("output ABI")
            };
            ensure!(
                out.try_to_u16_vec()? == expected,
                "{mode}: M={m} K={k} N={n}"
            );
        }
        let [Value::MemOwn(out)] = artifact.exec(
            "test-or-into",
            [
                rt.mem_u16(&x)?.into(),
                rt.mem_u16(&w)?.as_ref().into(),
                (m as u64).into(),
                (k as u64).into(),
                (n as u64).into(),
                rt.mem_u16(&vec![0x7fc0; m * n + 7])?.into(),
            ],
        )?
        else {
            anyhow::bail!("into output ABI")
        };
        let actual = out.try_to_u16_vec()?;
        ensure!(actual[..m * n] == expected, "into M={m} K={k} N={n}");
        ensure!(
            actual[m * n..].iter().all(|x| *x == 0x7fc0),
            "output padding modified"
        );
    }
    Ok(())
}

#[test]
#[ignore = "GPU microbenchmark; compares legacy token-tiled scalar emulation"]
fn benchmark_matrix_layout() -> Result<()> {
    use std::time::Instant;
    let dialect = catena_models::model::configured_gpu_dialect()?;
    let mut rt = Runtime::new(dialect)?;
    let mut source = sources();
    source[0] += "\n(def program test-old : {((cap.own mem) val) ((cap.ref mem) val) (u64 val) (u64 val) (u64 val)} -> ((cap.own mem) val) = ([x w m k n.] {[.x] [.w] [.k] [.n]} stdlib.qwen35.batch.project))";
    let artifact = rt.load(compile_sources(
        core_sources()
            .map(|s| -> &str { s })
            .chain(source.iter().map(String::as_str)),
        dialect,
    )?)?;
    for (m, k, n) in [
        (1, 768, 768),
        (1, 768, 51865),
        (4, 192, 192),
        (32, 192, 768),
        (256, 768, 768),
        (1500, 768, 768),
        (256, 768, 3072),
    ] {
        let w = rt.mem_u16(&vec![0x3c00; k * n])?;
        let mut times = [Vec::new(), Vec::new()];
        for run in 0..12 {
            for which in if run % 2 == 0 { [0, 1] } else { [1, 0] } {
                let x = rt.mem_u16(&vec![0x3d00; m * k])?;
                let start = Instant::now();
                let [Value::MemOwn(out)] = artifact.exec(
                    ["test-old", "test-or"][which],
                    [
                        x.into(),
                        w.as_ref().into(),
                        (m as u64).into(),
                        (k as u64).into(),
                        (n as u64).into(),
                    ],
                )?
                else {
                    anyhow::bail!("output ABI")
                };
                std::hint::black_box(out.try_to_u16_vec()?);
                if run >= 2 {
                    times[which].push(start.elapsed().as_secs_f64() * 1e6);
                }
            }
        }
        let mean = |i: usize| times[i].iter().sum::<f64>() / times[i].len() as f64;
        eprintln!(
            "M={m} K={k} N={n}: old {:.1} us, new {:.1} us, {:.3}x",
            mean(0),
            mean(1),
            mean(0) / mean(1)
        );
    }
    Ok(())
}
