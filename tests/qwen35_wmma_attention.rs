use catena_lang::runtime::{GpuDialect, Runtime, RuntimeModule, Value};
use catena_models::{
    catena::{compile_sources, core_sources},
    weights::{bf16_to_f32, f32_to_bf16},
};
const D: usize = 256;
const QH: usize = 16;
const KV: usize = 2;
fn compile_attention() -> anyhow::Result<RuntimeModule> {
    compile_attention_kind(false)
}
fn compile_attention_kind(exact: bool) -> anyhow::Result<RuntimeModule> {
    compile_attention_config(if exact { 1 } else { 0 })
}
fn compile_attention_config(kind: u8) -> anyhow::Result<RuntimeModule> {
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
        .find("(def program stdlib.qwen3.memory.u64.ceil_div :")
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
        GpuDialect::Hip,
    )
}
#[test]
fn tiled_attention_compiles() -> anyhow::Result<()> {
    compile_attention()?;
    Ok(())
}
#[test]
#[ignore = "requires gfx11 HIP GPU"]
fn tiled_attention_matches_cpu() -> anyhow::Result<()> {
    check(false)
}
#[test]
#[ignore = "requires gfx11 HIP GPU"]
fn exact_context_matches_cpu() -> anyhow::Result<()> {
    check(true)
}
fn check(exact: bool) -> anyhow::Result<()> {
    check_kind(if exact { 1 } else { 0 })
}
#[test]
#[ignore = "requires gfx11 HIP GPU"]
fn safe_attention_matches_cpu() -> anyhow::Result<()> {
    check_kind(2)
}
fn check_kind(kind: u8) -> anyhow::Result<()> {
    let mut runtime = Runtime::new(GpuDialect::Hip)?;
    let artifact = runtime.load(compile_attention_config(kind)?)?;
    let round = |x: f32| bf16_to_f32(f32_to_bf16(x));
    for (rows, start) in [(1, 0), (17, 3), (65, 0)] {
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
