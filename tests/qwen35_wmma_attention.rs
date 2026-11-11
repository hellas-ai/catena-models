#[path = "support/qwen_tiles.rs"]
mod tiles;
use catena_lang::runtime::{GpuDialect, Runtime, RuntimeModule, Value};
use catena_models::{
    catena::{compile_sources, core_sources},
    weights::{bf16_to_f32, f32_to_bf16},
};
const D: usize = 256;
const QH: usize = 16;
const KV: usize = 2;
fn compile_attention_config(kind: u8) -> anyhow::Result<RuntimeModule> {
    compile_attention_config_for(kind, tiles::dialect())
}
fn compile_attention_config_for(kind: u8, dialect: GpuDialect) -> anyhow::Result<RuntimeModule> {
    let exact = kind > 0;
    let norm = include_str!("../src/catena/qwen35.hex");
    let norm = &norm[norm
        .find("(def program stdlib.qwen35.norm.reduce-write :")
        .unwrap()
        ..norm
            .find("(def program stdlib.qwen35.norm.kernel :")
            .unwrap()];
    let softmax = include_str!("../src/catena/qwen35_attention.hex");
    let softmax = &softmax[softmax
        .find("(def program stdlib.qwen35.pow2.step :")
        .unwrap()
        ..softmax
            .find("(def program qwen35.softmax-shared :")
            .unwrap()];
    let div = include_str!("../src/catena/qwen35_attention.hex");
    let div = &div[div
        .find("(def program stdlib.models.memory.u64.ceil_div :")
        .unwrap()..];
    let mut source = [
        norm,
        softmax,
        div,
        include_str!("../src/catena/qwen35_wmma_attention.hex"),
    ]
    .join("\n");
    if exact {
        let original = include_str!("../src/catena/qwen35_attention.hex");
        source.push_str(
            &original[original
                .find("(def program stdlib.qwen35.context.dot :")
                .unwrap()
                ..original
                    .find("(def program stdlib.qwen35.context :")
                    .unwrap()],
        );
        source.push_str(include_str!("../src/catena/qwen35_exact_attention.hex"));
    }
    for (key, n) in [
        ("HEAD_DIM", D),
        ("QUERY_HEADS", QH),
        ("QUERY_HEADS_PER_KV_HEAD", QH / KV),
        ("QUERY_SIZE", QH * D),
        ("KEY_VALUE_SIZE", KV * D),
    ] {
        source = source.replace(&format!("{{{{{key}}}}}"), &format!("{n:016X}"));
    }
    source = source.replace(
        "{{SCALE_BITS}}",
        &format!("{:08X}", (1.0 / (D as f32).sqrt()).to_bits()),
    );
    for (prefix, zero) in [
        ("const.u64.0x{{", "0000000000000000"),
        ("const.u32.0x{{", "00000000"),
    ] {
        while let Some(at) = source.find(prefix) {
            let end = at + source[at..].find("}}").unwrap() + 2;
            source.replace_range(at..end, &format!("{}{}", &prefix[..prefix.len() - 2], zero));
        }
    }
    source.push_str(r#"(def program test-attention : {((cap.own mem) val) ((cap.own mem) val) ((cap.own mem) val) (u64 val) (u64 val) (u64 val)} -> ((cap.own mem) val) =
      ([query cache qg start kb vb.] {[.query] [.cache] [.qg] [.start] [.kb] [.vb]} stdlib.qwen35.wmma.attention.run))"#);
    source.push_str(r#"
(def program test-softmax : {((cap.own mem) val) ((cap.own mem) val) (u64 val) (u64 val) (u64 val) (u64 val)} -> ((cap.own mem) val) =
 ([scores prob keys stages shared rows.] {[.scores] [.prob] [.keys] [.stages] [.shared] [.rows]} stdlib.qwen35.wmma.attention.normalize))
(def program test-stream-softmax : {((cap.own mem) val) ((cap.own mem) val) (u64 val) (u64 val) (u64 val) (u64 val)} -> ((cap.own mem) val) =
 ([scores prob keys stages shared rows.] {[.scores] [.prob] [.keys] [.stages] [.shared] [.rows]} stdlib.qwen35.wmma.attention.normalize-stream))
"#);
    if exact {
        source = source.replace(
            "stdlib.qwen35.wmma.attention.run))",
            "stdlib.qwen35.wmma.attention.run-exact))",
        );
    }
    if kind == 2 {
        source = source.replace(
            "stdlib.qwen35.wmma.attention.run-exact))",
            "stdlib.qwen35.wmma.attention.run-safe))",
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
        dialect,
    )
}
#[test]
fn tiled_attention_compiles() -> anyhow::Result<()> {
    for dialect in [GpuDialect::Hip, GpuDialect::Cuda] {
        for kind in 0..=2 {
            let module = compile_attention_config_for(kind, dialect)?;
            if kind < 2 {
                tiles::check_native(&module, dialect);
            }
        }
    }
    Ok(())
}
#[test]
#[ignore = "requires a supported HIP or CUDA GPU"]
fn tiled_attention_matches_cpu() -> anyhow::Result<()> {
    check(false)
}
#[test]
#[ignore = "requires a supported HIP or CUDA GPU"]
fn exact_context_matches_cpu() -> anyhow::Result<()> {
    check(true)
}
fn check(exact: bool) -> anyhow::Result<()> {
    check_kind(if exact { 1 } else { 0 })
}
#[test]
#[ignore = "requires a supported HIP or CUDA GPU"]
fn safe_attention_matches_cpu() -> anyhow::Result<()> {
    check_kind(2)
}
fn check_kind(kind: u8) -> anyhow::Result<()> {
    let mut runtime = Runtime::new(tiles::dialect())?;
    let artifact = runtime.load(compile_attention_config(kind)?)?;
    let round = |x: f32| bf16_to_f32(f32_to_bf16(x));
    for (rows, start) in [(1, 0), (17, 3), (65, 0), (1, 16200)] {
        let keys = rows + start;
        let capacity = keys + 1;
        let vb = capacity * KV * D;
        let query: Vec<u16> = (0..rows * QH * D)
            .map(|i| f32_to_bf16(((i * 7 % 31) as f32 - 15.0) / 16.0))
            .collect();
        let cache: Vec<u16> = (0..2 * capacity * KV * D)
            .map(|i| f32_to_bf16(((i * 13 % 37) as f32 - 18.0) / 32.0))
            .collect();
        let qg = vec![0_u16; rows * QH * D * 2];
        let a = runtime.mem_u16(&query)?;
        let b = runtime.mem_u16(&cache)?;
        let c = runtime.mem_u16(&qg)?;
        let [result] = artifact.exec(
            "test-attention",
            [
                a.into(),
                b.into(),
                c.into(),
                (start as u64).into(),
                0_u64.into(),
                (vb as u64).into(),
            ],
        )?;
        let Value::MemOwn(result) = result else {
            panic!("not memory")
        };
        let actual = result.try_to_u16_vec()?;
        for t in 0..rows {
            for h in 0..QH {
                let end = start + t + 1;
                let kh = h / (QH / KV);
                let scores: Vec<f32> = (0..end)
                    .map(|k| {
                        round(
                            (0..D)
                                .map(|d| {
                                    bf16_to_f32(query[(t * QH + h) * D + d])
                                        * bf16_to_f32(cache[(k * KV + kh) * D + d])
                                })
                                .sum::<f32>()
                                / (D as f32).sqrt(),
                        )
                    })
                    .collect();
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let ex: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
                let den = ex.iter().sum::<f32>();
                for d in 0..D {
                    let expected = round(
                        round(
                            (0..end)
                                .map(|k| {
                                    round(ex[k] / den)
                                        * bf16_to_f32(cache[vb + (k * KV + kh) * D + d])
                                })
                                .sum::<f32>(),
                        ) * 0.5,
                    );
                    let ix = (t * QH + h) * D + d;
                    let got = bf16_to_f32(actual[ix]);
                    assert!(
                        (got - expected).abs() < 0.01,
                        "rows={rows} start={start} t={t} h={h} d={d}: {got} vs {expected}"
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA toolkit; no GPU needed"]
fn cuda_toolkit_compiles() -> anyhow::Result<()> {
    tiles::check_toolkit(
        compile_attention_config_for(0, GpuDialect::Cuda)?,
        "attention",
    )
}

#[test]
#[ignore = "requires a supported HIP or CUDA GPU"]
fn long_context_softmax_matches_cpu() -> anyhow::Result<()> {
    let mut runtime = Runtime::new(tiles::dialect())?;
    let artifact = runtime.load(compile_attention_config(0)?)?;
    // Dispatcher boundary, former AMD/CUDA shared-memory boundaries, and the
    // checkpoint's maximum context. Also exercise non-power-of-two row lengths.
    for keys in [
        17usize, 8192, 8193, 12032, 16112, 16128, 16200, 16208, 16384, 32768, 131072, 262144,
    ] {
        let rows = 2;
        let live = rows * QH * keys;
        let mut scores = vec![f32::NAN; live + 37];
        let mut expected = Vec::with_capacity(live);
        for row in 0..rows * QH {
            let visible = if row % 3 == 0 { 1 } else { keys - row % 7 };
            // Uniform rows check exact BF16 probabilities. Other rows cover
            // stable exponentiation, causal -infinity masks, and varied maxima.
            let values = (0..visible)
                .map(|k| {
                    if row % 2 == 0 {
                        0.0
                    } else {
                        ((k * 13 + row * 7) % 97) as f32 / 8.0 - 6.0
                    }
                })
                .collect::<Vec<_>>();
            let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let ex = values
                .iter()
                .map(|x| f64::from(*x - max).exp())
                .collect::<Vec<_>>();
            let sum = ex.iter().sum::<f64>();
            for key in 0..keys {
                scores[row * keys + key] = if key < visible {
                    values[key]
                } else {
                    f32::NEG_INFINITY
                };
                expected.push(if key < visible {
                    f32_to_bf16((ex[key] / sum) as f32)
                } else {
                    0
                });
            }
        }
        for entry in ["test-softmax", "test-stream-softmax"] {
            let [Value::MemOwn(out)] = artifact.exec(
                entry,
                [
                    runtime.mem_f32(&scores)?.into(),
                    runtime.mem_u16(&vec![0x7fc0; live + 37])?.into(),
                    (keys as u64).into(),
                    (keys.next_power_of_two().trailing_zeros() as u64).into(),
                    ((keys + 256) as u64).into(),
                    (rows as u64).into(),
                ],
            )?
            else {
                anyhow::bail!("softmax output ABI")
            };
            let actual = out.try_to_u16_vec()?;
            for (i, (&a, &b)) in actual[..live].iter().zip(&expected).enumerate() {
                let row = i / keys;
                let tolerance = u16::from(row % 2 == 1);
                anyhow::ensure!(
                    a.abs_diff(b) <= tolerance,
                    "{entry}: keys={keys} row={row} key={} got={} expected={}",
                    i % keys,
                    bf16_to_f32(a),
                    bf16_to_f32(b)
                );
            }
            anyhow::ensure!(
                actual[live..].iter().all(|v| *v == 0x7fc0),
                "output padding modified"
            );
        }
    }
    Ok(())
}
