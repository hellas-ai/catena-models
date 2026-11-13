//! Focused numerical and shape benchmarks for LFM3's dense projection kernels.
use anyhow::Result;
use catena_lang::runtime::{Runtime, RuntimeModule, Value};
use catena_models::{
    catena::{compile_sources, core_sources},
    weights::{bf16_to_f32, f32_to_bf16},
};

#[path = "support/qwen_tiles.rs"]
mod tiles;

type Variant = (&'static str, &'static str, bool);

fn projection_module() -> Result<(RuntimeModule, Vec<Variant>)> {
    let mut entries = String::new();
    let variants = [
        ("base", "stdlib.qwen35.batch.wmma-project", false),
        (
            "m128n64",
            "stdlib.lfm3.wmma.m128n64k32.linear-or.native",
            true,
        ),
        (
            "m128n64k64",
            "stdlib.lfm3.wmma.m128n64k64.linear-or.native",
            true,
        ),
        (
            "gate_wmma",
            "stdlib.lfm3.gate.m128n64k32.linear-or.native",
            true,
        ),
        ("gemv_wide", "stdlib.lfm3.gemv.r2wide", false),
        ("gemv_base", "stdlib.models.simt.or", true),
        ("gemv_r2", "stdlib.lfm3.gemv.r2", false),
    ];
    let filter = std::env::var("LFM3_PROJECTION_FILTER").ok();
    let variants = variants
        .into_iter()
        .filter(|(name, _, _)| {
            filter
                .as_ref()
                .is_none_or(|f| f.split(',').any(|s| s == *name))
        })
        .collect::<Vec<_>>();
    for &(name, op, rows) in &variants {
        let args = if rows {
            "[.x] [.w] [.rows] [.k] [.n]"
        } else {
            "[.x] [.w] [.k] [.n]"
        };
        let args = format!(
            "{args}{}",
            if name.starts_with("gate_") {
                " [.up]"
            } else {
                ""
            }
        );
        entries += &format!(
            "(def program {name} : {{((cap.own mem) val) ((cap.ref mem) val) ((cap.ref mem) val) (u64 val) (u64 val) (u64 val)}} -> {{((cap.own mem) val) ((cap.own mem) val)}} =\n ([x w up rows k n.]\n ({{{args}}} {op} [out.])\n ([.out] meta.master.complete-own [complete phase.])\n ({{[.x] [.phase]}} meta.master.after-own [input.])\n {{[.input] [.complete]}}\n ))\n"
        );
    }
    let wmma = include_str!("../src/catena/lfm3_wmma.hex")
        .replace("{{QUERY_SIZE}}", "0000000000000800")
        .replace("{{KEY_VALUE_SIZE}}", "0000000000000200");
    let gemv = include_str!("../src/catena/lfm3_gemv.hex")
        .replace("{{QUERY_SIZE}}", "0000000000000800")
        .replace("{{KEY_VALUE_SIZE}}", "0000000000000200");
    let mut sources = vec![
        include_str!("../src/catena/memory.hex"),
        include_str!("../src/catena/numeric.hex"),
        include_str!("../src/catena/resources.hex"),
        include_str!("../src/catena/cooperative.hex"),
        include_str!("../src/catena/bf16_simt.hex"),
        include_str!("../src/catena/qwen35_batch_projection.hex"),
        include_str!("../src/catena/qwen35_batch_wmma.hex"),
        wmma.as_str(),
        gemv.as_str(),
        &entries,
    ];
    for source in core_sources() {
        sources.push(source);
    }
    let module = compile_sources(sources, tiles::dialect())?;
    assert!(!module.source.contains("catena_bf16_dot"));
    Ok((module, variants))
}

#[test]
fn projections_compile() -> Result<()> {
    let (module, variants) = projection_module()?;
    if variants
        .iter()
        .any(|(name, op, _)| name.contains("wmma") || op.contains("wmma"))
    {
        tiles::check_native(&module, tiles::dialect());
        if tiles::dialect() == catena_lang::runtime::GpuDialect::Cuda {
            tiles::check_toolkit(module, "lfm3_projections")?;
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires a supported HIP/CUDA GPU"]
fn projections() -> Result<()> {
    let (module, variants) = projection_module()?;
    let mut rt = Runtime::new(tiles::dialect())?;
    let artifact = rt.load(module)?;
    for (rows, n, k) in [(1, 64, 256), (17, 67, 256), (65, 128, 2048)] {
        let x = (0..rows * k)
            .map(|i| f32_to_bf16(((i * 7 % 31) as f32 - 15.) / 16.))
            .collect::<Vec<_>>();
        let w = (0..n * k)
            .map(|i| f32_to_bf16(((i * 13 % 37) as f32 - 18.) / 32.))
            .collect::<Vec<_>>();
        let weights = rt.mem_u16(&w)?;
        let up_values = w
            .iter()
            .map(|&x| f32_to_bf16(bf16_to_f32(x) * 0.5 + 0.125))
            .collect::<Vec<_>>();
        let up = rt.mem_u16(&up_values)?;
        for &(name, _, _) in &variants {
            if (name.starts_with("gemv") || name.starts_with("gate_simt"))
                && (rows != 1 || n % 16 != 0)
            {
                continue;
            }
            let [Value::MemOwn(_input), Value::MemOwn(out)] = artifact.exec(
                name,
                [
                    rt.mem_u16(&x)?.into(),
                    weights.as_ref().into(),
                    up.as_ref().into(),
                    (rows as u64).into(),
                    (k as u64).into(),
                    (n as u64).into(),
                ],
            )?
            else {
                anyhow::bail!("projection ABI");
            };
            let got = out.try_to_u16_vec()?;
            for r in 0..rows {
                for c in 0..n {
                    let sum = (0..k)
                        .map(|p| bf16_to_f32(x[r * k + p]) * bf16_to_f32(w[c * k + p]))
                        .sum::<f32>();
                    let sum = if name.starts_with("gate_") {
                        let g = bf16_to_f32(f32_to_bf16(sum));
                        let u = (0..k)
                            .map(|p| bf16_to_f32(x[r * k + p]) * bf16_to_f32(up_values[c * k + p]))
                            .sum::<f32>();
                        g / (1. + (-g).exp()) * bf16_to_f32(f32_to_bf16(u))
                    } else {
                        sum
                    };
                    let expected = f32_to_bf16(sum);
                    let error = (bf16_to_f32(got[r * n + c]) - bf16_to_f32(expected)).abs();
                    assert!(
                        error
                            <= (if name.starts_with("gate_") {
                                0.02
                            } else {
                                0.01
                            }) * sum.abs().max(1.),
                        "{name} {rows}x{n}x{k} row={r} col={c} error={error}"
                    );
                }
            }
        }
    }
    if std::env::var_os("LFM3_PROJECTION_BENCH").is_none() {
        return Ok(());
    }
    for (rows, n, k) in [
        (512, 7680, 2048),
        (512, 2048, 7680),
        (512, 6144, 2048),
        (512, 512, 2048),
        (1, 7680, 2048),
        (1, 2048, 7680),
        (1, 128000, 2048),
    ] {
        let x = vec![f32_to_bf16(0.125); rows * k];
        let weights = rt.mem_u16(&vec![f32_to_bf16(0.25); n * k])?;
        let up = rt.mem_u16(&vec![f32_to_bf16(0.125); n * k])?;
        for &(name, _, _) in &variants {
            if (name.starts_with("gemv") || name.starts_with("gate_simt")) != (rows == 1) {
                continue;
            }
            let mut input = rt.mem_u16(&x)?;
            let mut timings = Vec::new();
            for i in 0..11 {
                let start = std::time::Instant::now();
                let [Value::MemOwn(next), Value::MemOwn(out)] = artifact.exec(
                    name,
                    [
                        input.into(),
                        weights.as_ref().into(),
                        up.as_ref().into(),
                        (rows as u64).into(),
                        (k as u64).into(),
                        (n as u64).into(),
                    ],
                )?
                else {
                    anyhow::bail!("projection ABI");
                };
                input = next;
                drop(out);
                if i > 0 {
                    timings.push(start.elapsed().as_secs_f64());
                }
            }
            timings.sort_by(f64::total_cmp);
            eprintln!(
                "projection {name} M={rows} N={n} K={k}: {:.3} ms",
                timings[5] * 1000.
            );
        }
    }
    Ok(())
}
