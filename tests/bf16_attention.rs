use anyhow::{Result, ensure};
use catena_lang::runtime::{GpuDialect, Runtime, Value};
use catena_models::{
    catena::{compile_sources, runtime_sources},
    weights::{bf16_to_f32, f32_to_bf16},
};

fn source() -> String {
    let mut s = include_str!("../src/catena/bf16_attention.hex").to_owned();
    for mode in ["o", "r"] {
        let ty = if mode == "o" { "own" } else { "ref" };
        s += &format!(
            "\n(def program scores-{mode} : {{((cap.own mem) val) ((cap.{ty} mem) val) (u64 val) (u64 val) (u64 val) (u64 val)}} -> ((cap.own mem) val) = ([x w count valid heads dim.] ([.count] stdlib.models.memory.alloc-f32 [out.]) {{[.x] [.w] [.valid] [.heads] [.dim] (const.u32.0x3E000000 u32.bitcast-f32) [.count] [.out]}} stdlib.models.simt.attention.scores-{mode}))"
        );
        s += &format!(
            "\n(def program context-{mode} : {{((cap.own mem) val) ((cap.{ty} mem) val) (u64 val) (u64 val) (u64 val) (u64 val)}} -> ((cap.own mem) val) = ([x w count valid heads dim.] ([.count] stdlib.models.memory.alloc-bf16 [out.]) {{[.x] [.w] [.valid] [.heads] [.dim] [.count] [.out]}} stdlib.models.simt.attention.context-{mode}))"
        );
    }
    catena_models::flux::graph::bundle(&s)
}
#[test]
fn attention_compiles() -> Result<()> {
    for dialect in [GpuDialect::Hip, GpuDialect::Cuda] {
        compile_sources(runtime_sources(&source()), dialect)?;
    }
    Ok(())
}
fn reduce(width: usize, count: usize, mut term: impl FnMut(usize) -> f32) -> f32 {
    let mut partial = vec![0f32; width];
    for lane in 0..width {
        let mut sum = [0f32; 2];
        for pair in (lane..count.div_ceil(2)).step_by(width) {
            for j in 0..2 {
                if pair * 2 + j < count {
                    sum[j] += term(pair * 2 + j);
                }
            }
        }
        partial[lane] = sum[0] + sum[1];
    }
    let mut stride = width / 2;
    while stride > 0 {
        for lane in 0..stride {
            partial[lane] += partial[lane + stride];
        }
        stride /= 2;
    }
    partial[0]
}
#[test]
#[ignore = "requires a GPU"]
fn attention_matches_reference() -> Result<()> {
    let dialect = catena_models::model::configured_gpu_dialect()?;
    let mut rt = Runtime::new(dialect)?;
    let artifact = rt.load(compile_sources(runtime_sources(&source()), dialect)?)?;
    for (rows, valid, heads, dim) in [
        (1, 1, 1, 1),
        (3, 7, 2, 5),
        (2, 65, 3, 64),
        (1, 1500, 12, 64),
    ] {
        let data = |len, salt| {
            (0..len)
                .map(|i| {
                    f32_to_bf16(
                        ((((i * 13 + salt) % 31) as f32 - 15.) / 16.)
                            * 2f32.powi(((i * 7 + salt) % 17) as i32 - 8),
                    )
                })
                .collect::<Vec<_>>()
        };
        let q = data(rows * heads * dim, 3);
        let k = data(valid * heads * dim, 7);
        let vv = data(valid * heads * dim, 9);
        let p = (0..rows * heads * valid)
            .map(|i| ((i % 17) as f32) / 32.)
            .collect::<Vec<_>>();
        let mut scores = vec![];
        let mut context = vec![];
        for qh in 0..rows * heads {
            let head = qh % heads;
            for key in 0..valid {
                scores.push(
                    reduce(32, dim, |ch| {
                        bf16_to_f32(q[qh * dim + ch])
                            * bf16_to_f32(k[(key * heads + head) * dim + ch])
                    }) * 0.125,
                );
            }
            for ch in 0..dim {
                context.push(f32_to_bf16(reduce(8, valid, |key| {
                    p[qh * valid + key] * bf16_to_f32(vv[(key * heads + head) * dim + ch])
                })));
            }
        }
        for mode in ["o", "r"] {
            for kind in ["scores", "context"] {
                let input = if kind == "scores" {
                    rt.mem_u16(&q)?
                } else {
                    rt.mem_f32(&p)?
                };
                let w = rt.mem_u16(if kind == "scores" { &k } else { &vv })?;
                let w: Value = if mode == "o" {
                    w.into()
                } else {
                    w.as_ref().into()
                };
                let count = if kind == "scores" {
                    scores.len()
                } else {
                    context.len()
                };
                let [Value::MemOwn(out)] = artifact.exec(
                    &format!("{kind}-{mode}"),
                    [
                        input.into(),
                        w,
                        (count as u64).into(),
                        (valid as u64).into(),
                        (heads as u64).into(),
                        (dim as u64).into(),
                    ],
                )?
                else {
                    anyhow::bail!("output ABI")
                };
                if kind == "scores" {
                    ensure!(
                        out.try_to_f32_vec()? == scores,
                        "scores {rows} {valid} {dim} {mode}"
                    );
                } else {
                    ensure!(
                        out.try_to_u16_vec()? == context,
                        "context {rows} {valid} {dim} {mode}"
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "GPU microbenchmark; requires CATENA_SIMT_BASELINE_DIR"]
fn benchmark_attention_layout() -> Result<()> {
    use std::{path::PathBuf, time::Instant};
    let dialect = catena_models::model::configured_gpu_dialect()?;
    let mut rt = Runtime::new(dialect)?;
    let baseline = std::fs::read_to_string(
        PathBuf::from(std::env::var("CATENA_SIMT_BASELINE_DIR")?).join("src/catena/whisper.hex"),
    )?;
    let mut s = source();
    s += &baseline.replace("stdlib.whisper.", "stdlib.baseline.whisper.");
    for kind in ["scores", "context"] {
        let scale = if kind == "scores" {
            "(const.u32.0x3E000000 u32.bitcast-f32) "
        } else {
            ""
        };
        s += &format!(
            "\n(def program old-{kind} : {{((cap.own mem) val) ((cap.own mem) val) (u64 val) (u64 val) (u64 val) (u64 val)}} -> ((cap.own mem) val) = ([x w count valid heads dim.] {{[.x] [.w] [.valid] [.heads] [.dim] {scale}[.count] [.count]}} stdlib.baseline.whisper.{kind}-o.run))"
        );
    }
    let artifact = rt.load(compile_sources(runtime_sources(&s), dialect)?)?;
    for (rows, keys, heads, dim) in [(1, 1500, 12, 64), (1500, 1500, 12, 64), (1, 64, 12, 64)] {
        for kind in ["scores", "context"] {
            let mut times = [Vec::new(), Vec::new()];
            for run in 0..12 {
                for which in if run % 2 == 0 { [0, 1] } else { [1, 0] } {
                    let x = if kind == "scores" {
                        rt.mem_u16(&vec![0x3d00; rows * heads * dim])?
                    } else {
                        rt.mem_f32(&vec![0.125; rows * heads * keys])?
                    };
                    let w = rt.mem_u16(&vec![0x3c00; keys * heads * dim])?;
                    let count = rows * heads * if kind == "scores" { keys } else { dim };
                    let start = Instant::now();
                    let [Value::MemOwn(out)] = artifact.exec(
                        &if which == 0 {
                            format!("old-{kind}")
                        } else {
                            format!("{kind}-o")
                        },
                        [
                            x.into(),
                            w.into(),
                            (count as u64).into(),
                            (keys as u64).into(),
                            (heads as u64).into(),
                            (dim as u64).into(),
                        ],
                    )?
                    else {
                        anyhow::bail!("output ABI")
                    };
                    if kind == "scores" {
                        std::hint::black_box(out.try_to_f32_vec()?);
                    } else {
                        std::hint::black_box(out.try_to_u16_vec()?);
                    }
                    if run >= 2 {
                        times[which].push(start.elapsed().as_secs_f64() * 1e3);
                    }
                }
            }
            let mean = |i: usize| times[i].iter().sum::<f64>() / times[i].len() as f64;
            eprintln!(
                "{kind} rows={rows} keys={keys}: old {:.3} ms new {:.3} ms {:.3}x",
                mean(0),
                mean(1),
                mean(0) / mean(1)
            );
        }
    }
    Ok(())
}
