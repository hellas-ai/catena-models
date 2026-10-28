use catena_lang::runtime::{GpuDialect, Runtime, RuntimeModule, Value};
use catena_models::{
    catena::{compile_sources, core_sources},
    weights::{bf16_to_f32, f32_to_bf16},
};

fn compile_projection() -> anyhow::Result<RuntimeModule> {
    let entry = r#"(def program wmma-project : {((cap.own mem) val) ((cap.ref mem) val) (u64 val) (u64 val)} -> ((cap.own mem) val) =
      ([input weight features outputs.] {[.input] [.weight] [.features] [.outputs]} stdlib.qwen35.batch.wmma-project))"#;
    let sources = core_sources().chain([
        include_str!("../src/catena/memory.hex"),
        include_str!("../src/catena/numeric.hex"),
        include_str!("../src/catena/cooperative.hex"),
        include_str!("../src/catena/qwen35_batch_projection.hex"),
        include_str!("../src/catena/qwen35_batch_wmma.hex"),
        entry,
    ]);
    compile_sources(sources, GpuDialect::Hip)
}

#[test]
fn experimental_wmma_projection_compiles() -> anyhow::Result<()> {
    let module = compile_projection()?;
    assert!(
        module
            .source
            .contains("__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32")
    );
    assert!(module.source.contains("catena_wmma_mma("));
    Ok(())
}

#[test]
#[ignore = "requires a gfx11 HIP GPU and ROCm compiler"]
fn experimental_wmma_projection_matches_cpu() -> anyhow::Result<()> {
    let mut runtime = Runtime::new(GpuDialect::Hip)?;
    let artifact = runtime.load(compile_projection()?)?;
    for (rows, cols, inner) in [
        (1, 1, 16),
        (16, 16, 16),
        (17, 19, 32),
        (31, 7, 48),
        (17, 19, 2048),
    ] {
        let input: Vec<u16> = (0..rows * inner)
            .map(|i| f32_to_bf16(((i * 7 % 31) as f32 - 15.0) / 8.0))
            .collect();
        let weights: Vec<u16> = (0..cols * inner)
            .map(|i| f32_to_bf16(((i * 13 % 37) as f32 - 18.0) / 16.0))
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
                assert!(
                    (got - expected).abs() <= 0.01 * expected.abs().max(1.0),
                    "rows={rows} cols={cols} inner={inner} row={row} col={col}: {got} vs {expected}"
                );
            }
        }
    }
    Ok(())
}
