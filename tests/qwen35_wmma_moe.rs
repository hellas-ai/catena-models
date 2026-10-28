use catena_lang::runtime::{GpuDialect, Runtime, RuntimeModule, Value};
use catena_models::{
    catena::{compile_sources, core_sources},
    weights::{bf16_to_f32, f32_to_bf16},
};

const E: usize = 256;
const K: usize = 8;
const H: usize = 2048;
const I: usize = 512;
const S: usize = 512;
const ACTIVE: usize = 64;
const STRIDE: usize = ACTIVE + K * I + S;
fn compile_moe(exact: bool) -> anyhow::Result<RuntimeModule> {
    compile_moe_kind(if exact { 1 } else { 0 })
}
fn compile_moe_kind(kind: u8) -> anyhow::Result<RuntimeModule> {
    let exact = kind > 0;
    let mut source = include_str!("../src/catena/qwen35_wmma_moe.hex").to_owned();
    if exact {
        source.push_str(include_str!("../src/catena/qwen35_exact_moe.hex"));
    }
    if exact {
        let original = include_str!("../src/catena/qwen35_batch_moe.hex");
        source.push_str(&original[..original.find("# Batched router").unwrap()]);
        source.push_str(
            &original[original
                .find("(def program stdlib.qwen35.batch.moe.gate-write :")
                .unwrap()
                ..original
                    .find("(def program stdlib.qwen35.batch.moe.down-dot :")
                    .unwrap()],
        );
        source.push_str(include_str!("../src/catena/qwen35_batch_projection.hex"));
        source.push_str(r#"(def program reference-gate : {((cap.own mem) val) ((cap.own mem) val) ((cap.ref mem) val) ((cap.ref mem) val) ((cap.ref mem) val) ((cap.ref mem) val)} -> ((cap.own mem) val) =
            ([input scratch gate up sg su.] {(const.u64.0x0000000000000000) [.input] [.gate] [.up] [.sg] [.su] [.scratch]} stdlib.qwen35.batch.moe.gate))"#);
    }
    if kind == 2 {
        source.push_str(include_str!("../src/catena/qwen35_safe_moe.hex"));
        let old = include_str!("../src/catena/qwen35_batch_moe.hex");
        source.push_str(
            &old[old
                .find("(def program stdlib.qwen35.batch.moe.down-dot :")
                .unwrap()
                ..old
                    .find("(def program stdlib.qwen35.batch.moe.ffn :")
                    .unwrap()],
        );
        source.push_str(r#"(def program reference-down : {((cap.own mem) val) ((cap.ref mem) val) ((cap.ref mem) val) ((cap.own mem) val)} -> ((cap.own mem) val) =
            ([scratch down sd residual.] {(const.u64.0x0000000000000000) [.scratch] [.down] [.sd] [.residual]} stdlib.qwen35.batch.moe.down))"#);
    }
    for (key, n) in [
        ("EXPERT_COUNT", E),
        ("TOP_K", K),
        ("HIDDEN_SIZE", H),
        ("INTERMEDIATE_SIZE", I),
        ("SHARED_SIZE", S),
        ("HIDDEN_PAIRS", H / 2),
        ("INTERMEDIATE_PAIRS", I / 2),
        ("SHARED_PAIRS", S / 2),
        ("ACTIVE_WIDTH_PAIRS", K * I / 2),
        ("ACTIVE_SIZE", K * I),
        ("TOTAL_ACTIVE_SIZE", K * I + S),
    ] {
        source = source.replace(&format!("{{{{{key}}}}}"), &format!("{n:016X}"));
    }
    for (name, n) in [
        ("stride-2", STRIDE),
        ("stride-4", STRIDE / 2),
        ("stride-8", STRIDE / 4),
        ("ids-u64", 0),
        ("probs-f32", K * 2),
        ("shared-router-f32", K * 3),
        ("active-bf16", ACTIVE),
        ("active-pairs", ACTIVE / 2),
        ("shared-active-pairs", (ACTIVE + K * I) / 2),
    ] {
        source.push_str(&format!("\n(def program stdlib.qwen35.batch.layout.{name} : 1 -> (u64 val) = (unit.elim (const.u64.0x{n:016X})))\n"));
    }
    source.push_str(r#"(def program test-moe : {((cap.own mem) val) ((cap.own mem) val) ((cap.own mem) val) ((cap.ref mem) val) ((cap.ref mem) val) ((cap.ref mem) val) ((cap.ref mem) val) ((cap.ref mem) val) ((cap.ref mem) val)} -> ({((cap.own mem) val) ((cap.own mem) val)} *) =
      ([input scratch residual gate up down sg su sd.]
       ({[.input] [.scratch] [.residual] [.gate] [.up] [.down] [.sg] [.su] [.sd]} stdlib.qwen35.wmma.moe.run [out.])
       ({[.scratch] [.out]} *.intro)))"#);
    if exact {
        source = source.replace(
            "} stdlib.qwen35.wmma.moe.run [",
            "} stdlib.qwen35.wmma.moe.run-exact [",
        );
    }
    if kind == 2 {
        source = source.replace(
            "} stdlib.qwen35.wmma.moe.run-exact [",
            "} stdlib.qwen35.wmma.moe.run-safe [",
        );
    }
    compile_sources(
        core_sources().map(|s| -> &str { s }).chain([
            include_str!("../src/catena/memory.hex"),
            include_str!("../src/catena/numeric.hex"),
            include_str!("../src/catena/cooperative.hex"),
            include_str!("../src/catena/resources.hex"),
            &source,
        ]),
        GpuDialect::Hip,
    )
}
#[test]
fn grouped_moe_compiles() -> anyhow::Result<()> {
    compile_moe(false)?;
    compile_moe(true)?;
    compile_moe_kind(2)?;
    Ok(())
}
#[test]
#[ignore = "requires gfx11 HIP GPU"]
fn grouped_moe_matches_cpu() -> anyhow::Result<()> {
    check(false)
}
#[test]
#[ignore = "requires gfx11 HIP GPU"]
fn exact_gate_matches_cpu() -> anyhow::Result<()> {
    check(true)
}
fn check(exact: bool) -> anyhow::Result<()> {
    check_kind(if exact { 1 } else { 0 })
}
#[test]
#[ignore = "requires gfx11 HIP GPU"]
fn safe_moe_matches_original_bits() -> anyhow::Result<()> {
    check_kind(2)
}
fn check_kind(kind: u8) -> anyhow::Result<()> {
    let exact = kind > 0;
    let mut runtime = Runtime::new(GpuDialect::Hip)?;
    let artifact = runtime.load(compile_moe_kind(kind)?)?;
    let round = |x: f32| bf16_to_f32(f32_to_bf16(x));
    for rows in [1, 17, 33] {
        let input: Vec<u16> = (0..rows * H)
            .map(|i| f32_to_bf16(((i * 7 % 31) as f32 - 15.0) / 32.0))
            .collect();
        let make = |n: usize| -> Vec<u16> {
            (0..n)
                .map(|i| f32_to_bf16(((i * 13 % 37) as f32 - 18.0) / 64.0))
                .collect()
        };
        let gate = make(E * I * H);
        let up = make(E * I * H);
        let down = make(E * H * I);
        let sg = make(S * H);
        let su = make(S * H);
        let sd = make(H * S);
        let mut scratch = vec![0_u16; rows * STRIDE];
        for t in 0..rows {
            for slot in 0..K {
                let expert = ((t * 17 + slot * 31) % E) as u64;
                let bytes = expert.to_ne_bytes();
                for j in 0..4 {
                    scratch[t * STRIDE + slot * 4 + j] =
                        u16::from_ne_bytes([bytes[2 * j], bytes[2 * j + 1]]);
                }
                let p = (if slot == 0 { 0.4_f32 } else { 0.6_f32 }).to_ne_bytes();
                let at = t * STRIDE + K * 4 + slot * 2;
                scratch[at] = u16::from_ne_bytes([p[0], p[1]]);
                scratch[at + 1] = u16::from_ne_bytes([p[2], p[3]]);
            }
        }
        let residual = make(rows * H);
        let a = runtime.mem_u16(&input)?;
        let work = runtime.mem_u16(&scratch)?;
        let res = runtime.mem_u16(&residual)?;
        let weights: Vec<_> = [&gate, &up, &down, &sg, &su, &sd]
            .into_iter()
            .map(|v| runtime.mem_u16(v))
            .collect::<Result<_, _>>()?;
        let mut args = vec![a.into(), work.into(), res.into()];
        args.extend(weights.iter().map(|w| w.as_ref().into()));
        let values = artifact.exec_values("test-moe", args)?;
        let Value::MemOwn(actual) = &values[1] else {
            panic!("not memory")
        };
        let actual = actual.try_to_u16_vec()?;
        let Value::MemOwn(active_memory) = &values[0] else {
            panic!("not memory")
        };
        let gpu_active = active_memory.try_to_u16_vec()?;
        if exact {
            let input2 = runtime.mem_u16(&input)?;
            let work2 = runtime.mem_u16(&scratch)?;
            let [Value::MemOwn(reference)] = artifact.exec(
                "reference-gate",
                [
                    input2.into(),
                    work2.into(),
                    weights[0].as_ref().into(),
                    weights[1].as_ref().into(),
                    weights[3].as_ref().into(),
                    weights[4].as_ref().into(),
                ],
            )?
            else {
                panic!("not memory")
            };
            let ref_values = reference.try_to_u16_vec()?;
            for t in 0..rows {
                assert_eq!(
                    &gpu_active[t * STRIDE + ACTIVE..(t + 1) * STRIDE],
                    &ref_values[t * STRIDE + ACTIVE..(t + 1) * STRIDE],
                    "gate activations differ, rows={rows} token={t}"
                );
            }
            if kind == 2 {
                let residual2 = runtime.mem_u16(&residual)?;
                let [Value::MemOwn(expected)] = artifact.exec(
                    "reference-down",
                    [
                        reference.into(),
                        weights[2].as_ref().into(),
                        weights[5].as_ref().into(),
                        residual2.into(),
                    ],
                )?
                else {
                    panic!("not memory")
                };
                assert_eq!(
                    actual,
                    expected.try_to_u16_vec()?,
                    "down outputs differ, rows={rows}"
                );
            }
        }
        if kind == 2 {
            continue;
        }
        let mut expected = vec![0_f32; rows * H];
        for t in 0..rows {
            let mut routed = vec![0_f32; H];
            let mut shared = vec![0_f32; H];
            for slot in 0..=K {
                let e = (t * 17 + slot * 31) % E;
                let n = if slot < K { I } else { S };
                let gw = if slot < K { &gate } else { &sg };
                let uw = if slot < K { &up } else { &su };
                let dw = if slot < K { &down } else { &sd };
                let mut active = vec![0_f32; n];
                for j in 0..n {
                    let base = if slot < K { (e * I + j) * H } else { j * H };
                    let dot = |w: &[u16]| {
                        (0..H)
                            .map(|x| bf16_to_f32(input[t * H + x]) * bf16_to_f32(w[base + x]))
                            .sum::<f32>()
                    };
                    let g = round(dot(gw));
                    let u = round(dot(uw));
                    let silu = g / (1.0 + (-g).exp());
                    let expected_activation = if slot < K {
                        round(silu * u * if slot == 0 { 0.4 } else { 0.6 })
                    } else {
                        round(round(round(silu) * u) * 0.5)
                    };
                    let bits = gpu_active[t * STRIDE + ACTIVE + slot * I + j];
                    let got = bf16_to_f32(bits);
                    // Validate both stages independently with the same 1%
                    // tolerance. Feeding the measured activations into the CPU
                    // down reference isolates its dot products from the first
                    // stage's BF16 midpoint and exp-approximation differences.
                    assert!(
                        (got - expected_activation).abs()
                            <= 0.01 * expected_activation.abs().max(1.0),
                        "activation rows={rows} token={t} slot={slot} channel={j}: {got} vs {expected_activation}",
                    );
                    active[j] = got;
                }
                for ch in 0..H {
                    let base = if slot < K { (e * H + ch) * I } else { ch * S };
                    let sum = (0..n)
                        .map(|j| active[j] * bf16_to_f32(dw[base + j]))
                        .sum::<f32>();
                    if slot < K {
                        routed[ch] += sum;
                    } else {
                        shared[ch] = sum;
                    }
                }
            }
            for ch in 0..H {
                expected[t * H + ch] = round(
                    round(routed[ch] + bf16_to_f32(residual[t * H + ch])) + round(shared[ch]),
                );
            }
        }
        for (ix, (&got, &want)) in actual.iter().zip(&expected).enumerate() {
            let got = bf16_to_f32(got);
            assert!(
                (got - want).abs() <= 0.01 * want.abs().max(1.0),
                "rows={rows} ix={ix}: {got} vs {want}"
            );
        }
    }
    Ok(())
}
