use catena_lang::runtime::{GpuDialect, Runtime, Value};
use catena_models::catena::{compile_sources, runtime_sources};
use catena_models::weights::{bf16_to_f32, f32_to_bf16};

fn source() -> String {
    catena_models::flux::graph::bundle(&[
        include_str!("../src/catena/qwen35_simt.hex"),
        r#"
(def program test-project : {((cap.own mem) val) ((cap.ref mem) val) (u64 val) (u64 val)} -> ((cap.own mem) val) = ([x w k n.] {[.x] [.w] [.k] [.n]} qwen3.fused.project))
(def program test-gate : {((cap.own mem) val) ((cap.ref mem) val) (u64 val) (u64 val) ((cap.ref mem) val)} -> ((cap.own mem) val) = ([x w k n u.] {[.x] [.w] [.k] [.n] [.u]} qwen3.fused.gate))
(def program test-residual : {((cap.own mem) val) ((cap.ref mem) val) (u64 val) (u64 val) ((cap.own mem) val)} -> ((cap.own mem) val) = ([x w k n r.] {[.x] [.w] [.k] [.n] [.r]} qwen3.fused.residual))
"#,
    ].join("\n"))
}

fn moe_source() -> String {
    let mut source = [
        include_str!("../src/catena/qwen35_simt.hex")
            .replace("(def program qwen3.", "(def program stdlib.qwen3."),
        include_str!("../src/catena/qwen35_simt_moe.hex").into(),
        include_str!("../src/catena/qwen35_moe.hex")
            .replace("(def program qwen35.", "(def program stdlib.qwen35."),
        include_str!("../src/catena/qwen35_batch_moe.hex").into(),
        include_str!("../src/catena/qwen35_group_projection.hex").into(),
        r#"
(def program test-router : {((cap.own mem) val) ((cap.ref mem) val) ((cap.ref mem) val) ((cap.own mem) val)} -> ((cap.own mem) val) = ([x r s o.] {[.x] [.r] [.s] [.o]} stdlib.qwen35.moe.router))
(def program test-batch-router : {(u64 val) ((cap.own mem) val) ((cap.ref mem) val) ((cap.ref mem) val) ((cap.own mem) val)} -> ((cap.own mem) val) = ([t x r s o.] {[.t] [.x] [.r] [.s] [.o]} stdlib.qwen35.batch.moe.router))
(def program test-moe-gate : {((cap.own mem) val) ((cap.ref mem) val) ((cap.ref mem) val) ((cap.ref mem) val) ((cap.ref mem) val) ((cap.own mem) val)} -> ((cap.own mem) val) = ([x g u sg su o.] {[.x] [.g] [.u] [.sg] [.su] [.o]} stdlib.qwen35.moe.gate))
(def program test-moe-down : {((cap.own mem) val) ((cap.ref mem) val) ((cap.ref mem) val) ((cap.own mem) val)} -> ((cap.own mem) val) = ([x w s r.] {[.x] [.w] [.s] [.r]} stdlib.qwen35.moe.down))
"#.into(),
    ].join("\n");
    for (key, value) in [
        ("EXPERT_COUNT", 4),
        ("TOP_K", 2),
        ("HIDDEN_SIZE", 256),
        ("HIDDEN_PAIRS", 128),
        ("HIDDEN_STAGES", 8),
        ("INTERMEDIATE_SIZE", 66),
        ("INTERMEDIATE_PAIRS", 33),
        ("SHARED_SIZE", 258),
        ("SHARED_PAIRS", 129),
        ("ACTIVE_SIZE", 132),
        ("ACTIVE_WIDTH_PAIRS", 66),
        ("TOTAL_ACTIVE_SIZE", 390),
        ("ROUTER_ROWS", 5),
        ("DOUBLE_QUERY_SIZE", 126),
        ("KEY_VALUE_SIZE", 66),
        ("QKV_SIZE", 192),
        ("VALUE_HEADS", 4),
        ("VALUE_SIZE", 128),
    ] {
        source = source.replace(&format!("{{{{{key}}}}}"), &format!("{value:016X}"));
    }
    for (name, value) in [
        ("router", 0),
        ("ids", 0),
        ("probs", 16),
        ("active", 256),
        ("workspace-bytes", 1040),
    ] {
        source.push_str(&format!("\n(def program stdlib.qwen35.layout.{name} : 1 -> (u64 val) = (unit.elim (const.u64.0x{value:016X})))\n"));
    }
    for (key, name, value) in [
        ("IDS_U64", "ids-u64", 0),
        ("PROBS_F32", "probs-f32", 4),
        ("ROUTER_F32", "router-f32", 32),
        ("SHARED_ROUTER_F32", "shared-router-f32", 36),
        ("ACTIVE_BF16", "active-bf16", 128),
        ("ACTIVE_PAIRS", "active-pairs", 64),
        ("SHARED_ACTIVE_PAIRS", "shared-active-pairs", 130),
        ("unused", "stride-2", 520),
        ("unused", "stride-4", 260),
        ("unused", "stride-8", 130),
    ] {
        source = source.replace(&format!("{{{{{key}}}}}"), &format!("{value:016X}"));
        source.push_str(&format!("\n(def program stdlib.qwen35.layout.{name} : 1 -> (u64 val) = (unit.elim (const.u64.0x{value:016X})))\n"));
    }
    catena_models::flux::graph::bundle(&source)
}

#[test]
fn standalone_simt_compiles_for_both_dialects() -> anyhow::Result<()> {
    for source in [source(), moe_source()] {
        for dialect in [GpuDialect::Cuda, GpuDialect::Hip] {
            let module = compile_sources(runtime_sources(&source), dialect)?;
            assert!(!module.source.contains("catena_bf16_dot"));
            assert!(!module.source.contains("catena_wmma_mma"));
        }
    }
    Ok(())
}

// Independent host oracle for the specified split low/high accumulation and
// fixed tree. Inputs are expanded here only for the test oracle.
fn dot(a: &[u16], b: &[f32]) -> f32 {
    dot_width(a, b, 32)
}

fn dot_width(a: &[u16], b: &[f32], width: usize) -> f32 {
    let mut sums = vec![0.0_f32; width];
    for (lane, sum) in sums.iter_mut().enumerate() {
        let (mut low, mut high) = (0.0, 0.0);
        for pair in (lane..a.len() / 2).step_by(width) {
            low += bf16_to_f32(a[pair * 2]) * b[pair * 2];
            high += bf16_to_f32(a[pair * 2 + 1]) * b[pair * 2 + 1];
        }
        *sum = low + high;
    }
    for shift in 1..=width.trailing_zeros() {
        let stride = width >> shift;
        for lane in 0..stride {
            sums[lane] += sums[lane + stride];
        }
    }
    sums[0]
}

fn values(n: usize, seed: usize) -> Vec<u16> {
    (0..n)
        .map(|i| {
            // Exercise F32 rounding/cancellation, not only exactly summable
            // small dyadic products.
            let scale = 2.0_f32.powi(((i * seed / 37) % 17) as i32 - 8);
            f32_to_bf16(((i * seed % 37) as f32 - 18.0) / 64.0 * scale)
        })
        .collect()
}

#[test]
#[ignore = "requires HIP or CUDA GPU; select with CATENA_GPU_DIALECT"]
fn projections_match_reference() -> anyhow::Result<()> {
    let dialect = catena_models::model::configured_gpu_dialect()?;
    let mut rt = Runtime::new(dialect)?;
    let module = rt.load(compile_sources(runtime_sources(&source()), dialect)?)?;
    for (k, n) in [
        (2, 1),
        (62, 3),
        (64, 5),
        (66, 17),
        (254, 7),
        (256, 4),
        (258, 5),
        (2048, 33),
    ] {
        let x = values(k, 7);
        let w = values(k * n, 13);
        let residual = values(n, 11);
        let u = values(k * n, 17);
        let weight = rt.mem_u16(&w)?;
        let up = rt.mem_u16(&u)?;
        for name in ["test-project", "test-residual", "test-gate"] {
            let mut args: Vec<Value> = vec![
                rt.mem_u16(&x)?.into(),
                weight.as_ref().into(),
                (k as u64).into(),
                (n as u64).into(),
            ];
            if name == "test-residual" {
                args.push(rt.mem_u16(&residual)?.into());
            }
            if name == "test-gate" {
                args.push(up.as_ref().into());
            }
            let Value::MemOwn(out) = module.exec_values(name, args)?.pop().unwrap() else {
                anyhow::bail!("wrong output")
            };
            let actual = out.try_to_u16_vec()?;
            for row in 0..n {
                let g = bf16_to_f32(f32_to_bf16(dot_width(
                    &x,
                    &w[row * k..(row + 1) * k]
                        .iter()
                        .copied()
                        .map(bf16_to_f32)
                        .collect::<Vec<_>>(),
                    if name == "test-project" {
                        32
                    } else if name == "test-residual" {
                        64
                    } else {
                        256
                    },
                )));
                let expected = if name == "test-residual" {
                    f32_to_bf16(g + bf16_to_f32(residual[row]))
                } else if name == "test-gate" {
                    let u = bf16_to_f32(f32_to_bf16(dot_width(
                        &x,
                        &u[row * k..(row + 1) * k]
                            .iter()
                            .copied()
                            .map(bf16_to_f32)
                            .collect::<Vec<_>>(),
                        256,
                    )));
                    let silu = bf16_to_f32(f32_to_bf16(g / (1.0 + (-g).exp())));
                    f32_to_bf16(silu * u)
                } else {
                    f32_to_bf16(g)
                };
                if name == "test-gate" {
                    // Host exp differs from Hex exp at F32 rounding boundaries.
                    anyhow::ensure!(
                        (bf16_to_f32(actual[row]) - bf16_to_f32(expected)).abs()
                            <= 0.01 * bf16_to_f32(expected).abs().max(0.01),
                        "{name} k={k} n={n} row={row}"
                    );
                } else {
                    assert_eq!(actual[row], expected, "{name} k={k} n={n} row={row}");
                }
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires HIP or CUDA GPU; select with CATENA_GPU_DIALECT"]
fn grouped_router_and_expert_kernels_match_reference() -> anyhow::Result<()> {
    let dialect = catena_models::model::configured_gpu_dialect()?;
    let mut rt = Runtime::new(dialect)?;
    let module = rt.load(compile_sources(runtime_sources(&moe_source()), dialect)?)?;
    let x = values(3 * 256, 7);
    let weights: Vec<_> = [126, 66, 66]
        .into_iter()
        .enumerate()
        .map(|(j, n)| values(n * 256, 11 + j * 2))
        .collect();
    let buffers = weights
        .iter()
        .map(|w| rt.mem_u16(w))
        .collect::<Result<Vec<_>, _>>()?;
    let result = module.exec_values(
        "qwen35.full-projections",
        vec![
            rt.mem_u16(&x[..256])?.into(),
            buffers[0].as_ref().into(),
            buffers[1].as_ref().into(),
            buffers[2].as_ref().into(),
        ],
    )?;
    for (bank, value) in result.into_iter().enumerate() {
        let Value::MemOwn(out) = value else {
            anyhow::bail!("wrong grouped output")
        };
        for (row, actual) in out.try_to_u16_vec()?.into_iter().enumerate() {
            let expected = dot_width(
                &x[..256],
                &weights[bank][row * 256..(row + 1) * 256]
                    .iter()
                    .copied()
                    .map(bf16_to_f32)
                    .collect::<Vec<_>>(),
                256,
            );
            assert_eq!(actual, f32_to_bf16(expected), "group bank={bank} row={row}");
        }
    }
    let weights: Vec<f32> = (0..4 * 256)
        .map(|i| ((i * 13 % 37) as f32 - 18.0) / 64.0)
        .collect();
    let shared = vec![0.125_f32; 256];
    let w = rt.mem_f32(&weights)?;
    let sw = rt.mem_f32(&shared)?;
    for batched in [false, true] {
        let tokens = if batched { 3 } else { 1 };
        let mut args: Vec<Value> = vec![];
        if batched {
            args.push(0u64.into());
        }
        args.extend([
            rt.mem_u16(&x[..tokens * 256])?.into(),
            w.as_ref().into(),
            sw.as_ref().into(),
            rt.mem_f32(&vec![f32::NAN; tokens * 260])?.into(),
        ]);
        let Value::MemOwn(out) = module
            .exec_values(
                if batched {
                    "test-batch-router"
                } else {
                    "test-router"
                },
                args,
            )?
            .pop()
            .unwrap()
        else {
            anyhow::bail!("wrong output")
        };
        let actual = out.try_to_f32_vec()?;
        for token in 0..tokens {
            for row in 0..5 {
                let weight = if row < 4 {
                    &weights[row * 256..(row + 1) * 256]
                } else {
                    &shared
                };
                let expected = dot(&x[token * 256..(token + 1) * 256], weight);
                let index = if batched { token * 260 + row } else { 32 + row };
                assert_eq!(
                    actual[index].to_bits(),
                    expected.to_bits(),
                    "router batch={batched} token={token} row={row}"
                );
            }
        }
    }
    // The shared expert is wider than both selected routed experts combined;
    // both stages and tails must be included without reading poisoned padding.
    let mut scratch = vec![0x7fc0_u16; 520];
    for (slot, expert) in [1_u64, 3].into_iter().enumerate() {
        for j in 0..4 {
            scratch[slot * 4 + j] = (expert >> (16 * j)) as u16;
        }
    }
    for (index, value) in [(4, 0.25_f32), (5, 0.75), (36, 0.125)] {
        scratch[index * 2] = value.to_bits() as u16;
        scratch[index * 2 + 1] = (value.to_bits() >> 16) as u16;
    }
    let g = values(4 * 66 * 256, 11);
    let u = values(4 * 66 * 256, 17);
    let sg = values(258 * 256, 19);
    let su = values(258 * 256, 23);
    let (gw, uw, sgw, suw) = (
        rt.mem_u16(&g)?,
        rt.mem_u16(&u)?,
        rt.mem_u16(&sg)?,
        rt.mem_u16(&su)?,
    );
    let [Value::MemOwn(out)] = module.exec(
        "test-moe-gate",
        [
            rt.mem_u16(&x[..256])?.into(),
            gw.as_ref().into(),
            uw.as_ref().into(),
            sgw.as_ref().into(),
            suw.as_ref().into(),
            rt.mem_u16(&scratch)?.into(),
        ],
    )?
    else {
        anyhow::bail!("wrong gate output")
    };
    let actual = out.try_to_u16_vec()?;
    assert_eq!(&actual[..128], &scratch[..128]);
    assert_eq!(&actual[518..], &scratch[518..]);
    for row in 0..390 {
        let routed = row < 132;
        let wr = if routed {
            [1, 3][row / 66] * 66 + row % 66
        } else {
            row - 132
        };
        let (g, u) = if routed { (&g, &u) } else { (&sg, &su) };
        let project = |w: &[u16]| {
            bf16_to_f32(f32_to_bf16(dot_width(
                &x[..256],
                &w[wr * 256..(wr + 1) * 256]
                    .iter()
                    .copied()
                    .map(bf16_to_f32)
                    .collect::<Vec<_>>(),
                256,
            )))
        };
        let (g, u) = (project(g), project(u));
        let silu = g / (1.0 + (-g).exp());
        let hidden = if routed {
            silu
        } else {
            bf16_to_f32(f32_to_bf16(silu))
        } * u;
        let hidden = if routed {
            hidden
        } else {
            bf16_to_f32(f32_to_bf16(hidden))
        };
        let scale = if routed {
            [0.25, 0.75][row / 66]
        } else {
            1.0 / (1.0 + (-0.125_f32).exp())
        };
        let expected = bf16_to_f32(f32_to_bf16(hidden * scale));
        anyhow::ensure!(
            (bf16_to_f32(actual[128 + row]) - expected).abs() <= 0.01 * expected.abs().max(0.0001),
            "expert gate row={row}"
        );
    }
    let activations = values(132 + 258, 19);
    scratch[128..518].copy_from_slice(&activations);
    let down = values(4 * 256 * 66, 11);
    let sd = values(256 * 258, 17);
    let residual = values(256, 5);
    let d = rt.mem_u16(&down)?;
    let s = rt.mem_u16(&sd)?;
    let [Value::MemOwn(out)] = module.exec(
        "test-moe-down",
        [
            rt.mem_u16(&scratch)?.into(),
            d.as_ref().into(),
            s.as_ref().into(),
            rt.mem_u16(&residual)?.into(),
        ],
    )?
    else {
        anyhow::bail!("wrong output")
    };
    let actual = out.try_to_u16_vec()?;
    for row in 0..256 {
        let weights: Vec<_> = [1, 3]
            .into_iter()
            .flat_map(|e| {
                down[(e * 256 + row) * 66..(e * 256 + row + 1) * 66]
                    .iter()
                    .copied()
                    .map(bf16_to_f32)
            })
            .collect();
        let routed = f32_to_bf16(dot(&activations[..132], &weights) + bf16_to_f32(residual[row]));
        let shared = f32_to_bf16(dot(
            &activations[132..],
            &sd[row * 258..(row + 1) * 258]
                .iter()
                .copied()
                .map(bf16_to_f32)
                .collect::<Vec<_>>(),
        ));
        assert_eq!(
            actual[row],
            f32_to_bf16(bf16_to_f32(routed) + bf16_to_f32(shared)),
            "expert down row={row}"
        );
    }
    Ok(())
}
