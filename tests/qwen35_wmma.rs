#[path = "support/qwen_tiles.rs"]
mod tiles;
use catena_lang::runtime::{GpuDialect, Runtime, RuntimeModule, Value};
use catena_models::{
    catena::{compile_sources, core_sources},
    weights::{bf16_to_f32, f32_to_bf16},
};

fn compile_projection() -> anyhow::Result<RuntimeModule> {
    compile_projection_for(tiles::dialect())
}
fn compile_projection_for(dialect: GpuDialect) -> anyhow::Result<RuntimeModule> {
    let entry = r#"(def program wmma-project : {((cap.own mem) val) ((cap.ref mem) val) (u64 val) (u64 val)} -> ((cap.own mem) val) =
      ([input weight features outputs.] {[.input] [.weight] [.features] [.outputs]} stdlib.qwen35.batch.wmma-project))
(def program wmma-gate : {((cap.own mem) val) ((cap.ref mem) val) (u64 val) (u64 val) ((cap.ref mem) val)} -> ((cap.own mem) val) =
 ([input gate features outputs up.] {[.input] [.gate] [.features] [.outputs] [.up]} stdlib.qwen35.wmma.projection.gate))"#;
    let sources = core_sources().chain([
        include_str!("../src/catena/memory.hex"),
        include_str!("../src/catena/numeric.hex"),
        include_str!("../src/catena/resources.hex"),
        include_str!("../src/catena/qwen35_wmma_epilogues.hex"),
        include_str!("../src/catena/cooperative.hex"),
        include_str!("../src/catena/qwen35_batch_projection.hex"),
        include_str!("../src/catena/qwen35_batch_wmma.hex"),
        entry,
    ]);
    compile_sources(sources, dialect)
}

#[test]
fn experimental_wmma_projection_compiles() -> anyhow::Result<()> {
    for dialect in [GpuDialect::Hip, GpuDialect::Cuda] {
        tiles::check_native(&compile_projection_for(dialect)?, dialect);
    }
    Ok(())
}

#[test]
#[ignore = "requires a supported HIP or CUDA GPU"]
fn experimental_wmma_projection_matches_cpu() -> anyhow::Result<()> {
    let mut runtime = Runtime::new(tiles::dialect())?;
    let artifact = runtime.load(compile_projection()?)?;
    for (rows, cols, inner, identity) in [
        (1, 1, 16, false),
        (16, 16, 16, false),
        (17, 19, 32, false),
        (31, 7, 48, false),
        (17, 19, 2048, false),
        // Identity rows isolate every fragment coordinate and both column halves.
        (16, 16, 16, true),
        (32, 32, 32, true),
    ] {
        let input: Vec<u16> = (0..rows * inner)
            .map(|i| {
                f32_to_bf16(if identity {
                    f32::from(i / inner == i % inner)
                } else {
                    ((i * 7 % 31) as f32 - 15.0) / 8.0
                })
            })
            .collect();
        let weights: Vec<u16> = (0..cols * inner)
            .map(|i| {
                f32_to_bf16(if identity {
                    ((i % 251) as f32 - 125.0) / 16.0
                } else {
                    ((i * 13 % 37) as f32 - 18.0) / 16.0
                })
            })
            .collect();
        let a = runtime.mem_u16(&input)?;
        let b = runtime.mem_u16(&weights)?;
        let [result] = artifact.exec(
            "wmma-project",
            [
                a.into(),
                b.as_ref().into(),
                (inner as u64).into(),
                (cols as u64).into(),
            ],
        )?;
        let Value::MemOwn(result) = result else {
            anyhow::bail!("projection returned non-memory output");
        };
        let actual = result.try_to_u16_vec()?;
        assert_eq!(actual.len(), rows * cols);
        for row in 0..rows {
            for col in 0..cols {
                let expected = (0..inner)
                    .map(|k| {
                        bf16_to_f32(input[row * inner + k]) * bf16_to_f32(weights[col * inner + k])
                    })
                    .sum::<f32>();
                let got = bf16_to_f32(actual[row * cols + col]);
                let tolerance = if identity {
                    0.0
                } else {
                    0.01 * expected.abs().max(1.0)
                };
                assert!(
                    (got - expected).abs() <= tolerance,
                    "rows={rows} cols={cols} inner={inner} row={row} col={col}: {got} vs {expected}"
                );
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires gfx1151 HIP GPU and ROCm compiler"]
fn deltanet_input_projections_match_emulation_bits() -> anyhow::Result<()> {
    use catena_models::spec::ModelConfig;

    let mut native = compile_projection_for(GpuDialect::Hip)?;
    let mut emulated = native.clone();
    // Fix each artifact's launch mode without changing process-wide environment
    // variables, so parallel GPU tests cannot switch one another's arithmetic.
    let setting = "getenv(\"CATENA_WMMA_EMULATE\")";
    anyhow::ensure!(
        native.source.contains(setting),
        "missing WMMA launch selector"
    );
    native.source = native.source.replace(setting, "\"0\"");
    emulated.source = emulated.source.replace(setting, "\"1\"");
    let mut runtime = Runtime::new(GpuDialect::Hip)?;
    let native = runtime.load(native)?;
    let emulated = runtime.load(emulated)?;
    for config in [ModelConfig::qwen35_0_8b(), ModelConfig::qwen36_35b_a3b()] {
        for (name, cols) in [
            ("qkv", config.linear_qkv_size().unwrap()),
            ("z", config.linear_inner_size.unwrap()),
            ("alpha", config.linear_time_step_rank.unwrap()),
            ("beta", config.linear_time_step_rank.unwrap()),
        ] {
            let inner = config.hidden_size;
            let salt = if name == "beta" { 11 } else { 3 };
            let weights: Vec<u16> = (0..cols * inner)
                .map(|i| f32_to_bf16(((i * 13 + salt) % 97) as f32 / 97. - 0.5))
                .collect();
            let weights = runtime.mem_u16(&weights)?;
            for rows in [2, 17] {
                let input: Vec<u16> = (0..rows * inner)
                    .map(|i| f32_to_bf16(((i * 7 + i / inner * 11) % 89) as f32 / 89. - 0.5))
                    .collect();
                let mut results = Vec::new();
                for artifact in [&native, &emulated] {
                    let [Value::MemOwn(result)] = artifact.exec(
                        "wmma-project",
                        [
                            runtime.mem_u16(&input)?.into(),
                            weights.as_ref().into(),
                            (inner as u64).into(),
                            (cols as u64).into(),
                        ],
                    )?
                    else {
                        anyhow::bail!("projection returned non-memory output");
                    };
                    let words = result.try_to_u16_vec()?;
                    assert_eq!(words.len(), rows * cols);
                    assert!(words.iter().all(|&x| bf16_to_f32(x).is_finite()));
                    results.push(words);
                }
                assert_eq!(
                    results[0],
                    results[1],
                    "{} {name}: rows={rows} inner={inner} cols={cols}",
                    config.model_name.as_deref().unwrap()
                );
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires a supported HIP or CUDA GPU"]
fn dense_gate_matches_bf16_boundaries() -> anyhow::Result<()> {
    let mut runtime = Runtime::new(tiles::dialect())?;
    let artifact = runtime.load(compile_projection()?)?;
    for (rows, inner, cols) in [(1, 16, 1), (17, 48, 19), (65, 64, 67)] {
        let make = |n: usize, salt: usize| -> Vec<u16> {
            (0..n)
                .map(|i| f32_to_bf16(((i * 7 + salt) % 31) as f32 / 31. - 0.5))
                .collect()
        };
        let x = make(rows * inner, 0);
        let gate = make(cols * inner, 3);
        let up = make(cols * inner, 11);
        let a = runtime.mem_u16(&x)?;
        let g = runtime.mem_u16(&gate)?;
        let u = runtime.mem_u16(&up)?;
        let [Value::MemOwn(result)] = artifact.exec(
            "wmma-gate",
            [
                a.into(),
                g.as_ref().into(),
                (inner as u64).into(),
                (cols as u64).into(),
                u.as_ref().into(),
            ],
        )?
        else {
            anyhow::bail!("expected gate output")
        };
        let result = result.try_to_u16_vec()?;
        for row in 0..rows {
            for col in 0..cols {
                let dot = |w: &[u16]| {
                    let sum = (0..inner)
                        .map(|k| bf16_to_f32(x[row * inner + k]) * bf16_to_f32(w[col * inner + k]))
                        .sum::<f32>();
                    bf16_to_f32(f32_to_bf16(sum))
                };
                let g = dot(&gate);
                let active = bf16_to_f32(f32_to_bf16(g / (1. + (-g).exp())));
                let want = bf16_to_f32(f32_to_bf16(active * dot(&up)));
                let got = bf16_to_f32(result[row * cols + col]);
                assert!(
                    (got - want).abs() <= 0.01 * want.abs().max(0.1),
                    "gate [{row},{col}]: {got} vs {want}"
                );
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA toolkit; no GPU needed"]
fn cuda_toolkit_compiles() -> anyhow::Result<()> {
    tiles::check_toolkit(compile_projection_for(GpuDialect::Cuda)?, "projection")
}
