use anyhow::Result;
use catena_lang::runtime::{GpuDialect, Runtime, Value};
use catena_models::catena::{compile_sources, core_sources};
fn source() -> String {
    catena_models::flux::graph::bundle(&[include_str!("../src/catena/flux_vae_wmma.hex"),r#"
 (def program test-conv : {((cap.ref mem) val) ((cap.ref mem) val) ((cap.ref mem) val) (u64 val) (u64 val) (u64 val) (u64 val) (u64 val) (u64 val)} -> ((cap.own mem) val) =
 ([x w b ci co h width ks pad.] {[.x] [.w] [.b] [.ci] [.co] [.h] [.width] [.ks] [.pad]} flux.vae-conv-f32))
 (def program test-norm : {((cap.ref mem) val) ((cap.ref mem) val) ((cap.ref mem) val) (u64 val) (u64 val) (u64 val)} -> ((cap.own mem) val) =
 ([x w b groups cols spatial.] {[.x] [.w] [.b] [.groups] [.cols] [.spatial] (const.u64.0x0000000000000002) (const.u64.0x0000000000000002) (const.u32.0x358637BD u32.bitcast-f32)} flux.vae-norm-f32))
 "#].join("\n"))
}
#[test]
fn vae_kernels_compile() -> Result<()> {
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
#[ignore = "requires gfx11 HIP GPU"]
fn convolution_and_group_norm_match_cpu() -> Result<()> {
    let s = source();
    let mut rt = Runtime::new(GpuDialect::Hip)?;
    let module = rt.load(compile_sources(
        core_sources().map(|s| -> &str { s }).chain([s.as_str()]),
        GpuDialect::Hip,
    )?)?;
    for (ci, co, h, width, ks) in [
        (32, 64, 3, 5, 1),
        (32, 70, 5, 7, 3),
        (32, 32, 3, 5, 1),
        (32, 3, 5, 7, 3),
    ] {
        let make =
            |n: usize| -> Vec<f32> { (0..n).map(|i| (i as i32 % 29 - 14) as f32 / 31.).collect() };
        let x = make(ci * h * width);
        let w = make(co * ci * ks * ks);
        let bias = make(co);
        let a = rt.mem_f32(&x)?;
        let b = rt.mem_f32(&w)?;
        let c = rt.mem_f32(&bias)?;
        let [Value::MemOwn(out)] = module.exec(
            "test-conv",
            [
                a.as_ref().into(),
                b.as_ref().into(),
                c.as_ref().into(),
                (ci as u64).into(),
                (co as u64).into(),
                (h as u64).into(),
                (width as u64).into(),
                (ks as u64).into(),
                ((ks / 2) as u64).into(),
            ],
        )?
        else {
            panic!()
        };
        let out = out.try_to_f32_vec()?;
        for oc in 0..co {
            for y in 0..h {
                for px in 0..width {
                    let mut want = 0f32;
                    for ic in 0..ci {
                        for ky in 0..ks {
                            for kx in 0..ks {
                                let iy = y as isize + ky as isize - (ks / 2) as isize;
                                let ix = px as isize + kx as isize - (ks / 2) as isize;
                                if iy >= 0 && iy < h as isize && ix >= 0 && ix < width as isize {
                                    let mut a = x[(ic * h + iy as usize) * width + ix as usize];
                                    let mut b = w[((oc * ci + ic) * ks + ky) * ks + kx];
                                    if co >= 64 {
                                        use catena_models::weights::{bf16_to_f32, f32_to_bf16};
                                        a = bf16_to_f32(f32_to_bf16(a));
                                        b = bf16_to_f32(f32_to_bf16(b));
                                    }
                                    want += a * b;
                                }
                            }
                        }
                    }
                    want += bias[oc];
                    let got = out[(oc * h + y) * width + px];
                    assert!(
                        (got - want).abs() <= 0.001 * want.abs().max(1.),
                        "conv ci={ci} co={co} h={h} w={width} ks={ks} [{oc},{y},{px}] {got} vs {want}"
                    );
                }
            }
        }
    }
    for cols in [35, 1024] {
        let groups = 4;
        let spatial = cols;
        let x: Vec<_> = (0..groups * cols)
            .map(|i| (i as i32 % 23 - 11) as f32 / 11.)
            .collect();
        let weights = vec![1.25f32; groups];
        let bias = vec![-0.1f32; groups];
        let a = rt.mem_f32(&x)?;
        let b = rt.mem_f32(&weights)?;
        let c = rt.mem_f32(&bias)?;
        let [Value::MemOwn(out)] = module.exec(
            "test-norm",
            [
                a.as_ref().into(),
                b.as_ref().into(),
                c.as_ref().into(),
                (groups as u64).into(),
                (cols as u64).into(),
                (spatial as u64).into(),
            ],
        )?
        else {
            panic!()
        };
        let out = out.try_to_f32_vec()?;
        for g in 0..groups {
            let slice = &x[g * cols..(g + 1) * cols];
            let mean = slice.iter().sum::<f32>() / cols as f32;
            let var = slice.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / cols as f32;
            for j in 0..cols {
                let want = (slice[j] - mean) / (var + 1e-6).sqrt() * 1.25 - 0.1;
                assert!(
                    (out[g * cols + j] - want).abs() < 0.0001,
                    "norm g={g} cols={cols} j={j}: {} vs {want}",
                    out[g * cols + j]
                );
            }
        }
    }
    Ok(())
}
