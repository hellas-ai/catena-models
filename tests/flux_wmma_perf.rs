//! Isolated WMMA timings; larger tensor shapes do not render images.
use anyhow::Result;
use catena_lang::runtime::{GpuDialect, Runtime, Value};
use catena_models::catena::{compile_sources, core_sources};
use std::time::Instant;
#[test]
#[ignore = "requires gfx11 HIP GPU; isolated projection timings"]
fn projection_shapes() -> Result<()> {
    let kernels = if let Ok(path) = std::env::var("FLUX_LINEAR_SOURCE") {
        std::fs::read_to_string(path)?
    } else {
        include_str!("../src/catena/flux_wmma_linear.hex").into()
    };
    let s=catena_models::flux::graph::bundle(&[kernels.as_str(),r#"(def program bench-linear : {((cap.ref mem) val) ((cap.ref mem) val) (u64 val) (u64 val) (u64 val)} -> ((cap.own mem) val) = ([x w m k n.] {[.x] [.w] [.m] [.k] [.n]} flux.wmma-linear-bf16))"#].join("\n"));
    let mut rt = Runtime::new(GpuDialect::Hip)?;
    let m = rt.load(compile_sources(
        core_sources().map(|s| -> &str { s }).chain([s.as_str()]),
        GpuDialect::Hip,
    )?)?;
    for (rows, ci, co) in [
        (256, 3072, 3072),
        (768, 3072, 27648),
        (768, 12288, 3072),
        (1536, 3072, 27648),
        (4608, 3072, 27648),
    ] {
        let x = rt.mem_u16(&vec![0x3e80; rows * ci])?;
        let w = rt.mem_u16(&vec![0x3e00; ci * co])?;
        let mut times = Vec::new();
        for rep in 0..4 {
            let start = Instant::now();
            let _: [Value; 1] = m.exec(
                "bench-linear",
                [
                    x.as_ref().into(),
                    w.as_ref().into(),
                    (rows as u64).into(),
                    (ci as u64).into(),
                    (co as u64).into(),
                ],
            )?;
            if rep > 0 {
                times.push(start.elapsed().as_secs_f64());
            }
        }
        times.sort_by(f64::total_cmp);
        eprintln!("projection M={rows} K={ci} N={co}: {:.6}s", times[1]);
    }
    Ok(())
}

#[test]
#[ignore = "requires gfx11 HIP GPU; isolated attention timings"]
fn attention_shapes() -> Result<()> {
    let kernels = if let Ok(path) = std::env::var("FLUX_ATTENTION_SOURCE") {
        std::fs::read_to_string(path)?
    } else {
        include_str!("../src/catena/flux_wmma_attention.hex").into()
    };
    let s=catena_models::flux::graph::bundle(&[kernels.as_str(),r#"(def program bench-attention : {((cap.ref mem) val) ((cap.ref mem) val) ((cap.ref mem) val) (u64 val)} -> ((cap.own mem) val) = ([q k v rows.] {[.q] [.k] [.v] [.rows] [.rows] (const.u64.0x0000000000000018) (const.u64.0x0000000000000018) (const.u64.0x0000000000000080) (const.u32.0x3DB504F3 u32.bitcast-f32) (const.u64.0x0000000000000000) [.rows]} flux.wmma-fused-attention-bf16))"#].join("\n"));
    let mut rt = Runtime::new(GpuDialect::Hip)?;
    let m = rt.load(compile_sources(
        core_sources().map(|s| -> &str { s }).chain([s.as_str()]),
        GpuDialect::Hip,
    )?)?;
    for rows in [768, 1536, 4608] {
        let x = rt.mem_u16(&vec![0x3e80; rows * 3072])?;
        let mut times = Vec::new();
        for rep in 0..4 {
            let start = Instant::now();
            let _: [Value; 1] = m.exec(
                "bench-attention",
                [
                    x.as_ref().into(),
                    x.as_ref().into(),
                    x.as_ref().into(),
                    (rows as u64).into(),
                ],
            )?;
            if rep > 0 {
                times.push(start.elapsed().as_secs_f64());
            }
        }
        times.sort_by(f64::total_cmp);
        eprintln!("attention M={rows} H=24 D=128: {:.6}s", times[1]);
    }
    Ok(())
}
