//! Focused DeltaNet state parity and optional shape timings for the dense 2B model.
use catena_lang::runtime::{GpuDialect, Runtime, Value};
use catena_models::{
    catena::{compile_sources, core_sources},
    weights::f32_to_bf16,
};

fn definitions(source: &str, prefix: &str) -> String {
    let mut out = String::new();
    for (at, _) in source.match_indices("(def program ") {
        let name = source[at + 13..].split_whitespace().next().unwrap();
        if !name.starts_with(prefix) {
            continue;
        }
        let mut depth = 0;
        for (i, ch) in source[at..].char_indices() {
            depth += i32::from(ch == '(') - i32::from(ch == ')');
            if depth == 0 {
                out.push_str(&source[at..at + i + 1]);
                out.push('\n');
                break;
            }
        }
    }
    out
}
#[test]
#[ignore = "requires HIP GPU"]
fn dense_scan_preserves_state() -> anyhow::Result<()> {
    let dense = include_str!("../src/catena/qwen35_dense.hex");
    let original = include_str!("../src/catena/qwen35_batch_scan.hex");
    let mut source = String::new();
    source += &definitions(
        include_str!("../src/catena/qwen35.hex"),
        "stdlib.qwen35.delta-update.write",
    );
    source += original;
    let mut variants = vec![("base".to_string(), "stdlib.qwen35.batch.scan".to_string())];
    source += &definitions(dense, "stdlib.qwen35.dense.scan");
    for lanes in [8] {
        variants.push((
            format!("lanes{lanes}"),
            format!("stdlib.qwen35.dense.scan-l{lanes}"),
        ));
    }
    for (name, op) in &variants {
        source += &format!(
            "(def program {name} : {{((cap.own mem) val) ((cap.own mem) val) ((cap.own mem) val) ((cap.own mem) val) ((cap.own mem) val) ((cap.ref mem) val) ((cap.ref mem) val) ((cap.own mem) val)}} -> {{((cap.own mem) val) ((cap.own mem) val)}} =\n ([conv q k alpha beta dt a shadow.]\n ({{[.conv] [.q] [.k] [.alpha] [.beta] [.dt] [.a] [.shadow] (const.u64.0x0000000000000000)}} {op} [out.])\n ([.out] meta.master.complete-own [ready phase.])\n ({{[.shadow] [.phase]}} meta.master.after-own [state.])\n {{[.state] [.ready]}}\n ))\n"
        );
    }
    for (key, value) in [
        ("KEY_SIZE", 2048),
        ("VALUE_SIZE", 2048),
        ("VALUE_HEADS", 16),
        ("KEY_HEADS", 16),
        ("KEY_DIM", 128),
        ("VALUE_DIM", 128),
        ("QKV_SIZE", 6144),
        ("VALUE_BLOCKS", 256),
    ] {
        source = source.replace(&format!("{{{{{key}}}}}"), &format!("{value:016X}"));
    }
    source = source.replace(
        "{{DELTA_SCALE_BITS}}",
        &format!("{:08X}", 128_f32.sqrt().recip().to_bits()),
    );
    let mut sources = vec![
        source.as_str(),
        include_str!("../src/catena/memory.hex"),
        include_str!("../src/catena/numeric.hex"),
        include_str!("../src/catena/resources.hex"),
        include_str!("../src/catena/cooperative.hex"),
    ];
    for core in core_sources() {
        sources.push(core);
    }
    let module = compile_sources(sources, GpuDialect::Hip)?;
    let mut rt = Runtime::new(GpuDialect::Hip)?;
    let artifact = rt.load(module)?;
    let dt = rt.mem_f32(&vec![0.125; 16])?;
    let a = rt.mem_f32(&vec![-0.75; 16])?;
    for rows in [1, 129, 512] {
        let data = |n: usize, scale: f32| {
            (0..n)
                .map(|i| f32_to_bf16(((i * 13 % 37) as f32 - 18.) * scale))
                .collect::<Vec<_>>()
        };
        let conv = data(rows * 6144, 0.02);
        let q = data(rows * 2048, 0.005);
        let k = data(rows * 2048, 0.005);
        let alpha = data(rows * 16, 0.02);
        let beta = data(rows * 16, 0.01);
        let mut reference = None;
        for (name, _) in &variants {
            let iterations = if std::env::var_os("QWEN35_SCAN_BENCH").is_some() {
                6
            } else {
                1
            };
            let mut times = Vec::new();
            for _ in 0..iterations {
                let initial = (0..2048 * 128)
                    .map(|i| ((i % 29) as f32 - 14.) * 0.0001)
                    .collect::<Vec<_>>();
                let state = rt.mem_f32(&initial)?;
                let conv = rt.mem_u16(&conv)?;
                let q = rt.mem_u16(&q)?;
                let k = rt.mem_u16(&k)?;
                let alpha = rt.mem_u16(&alpha)?;
                let beta = rt.mem_u16(&beta)?;
                let start = std::time::Instant::now();
                let [Value::MemOwn(state), Value::MemOwn(out)] = artifact.exec(
                    name,
                    [
                        conv.into(),
                        q.into(),
                        k.into(),
                        alpha.into(),
                        beta.into(),
                        dt.as_ref().into(),
                        a.as_ref().into(),
                        state.into(),
                    ],
                )?
                else {
                    anyhow::bail!("scan ABI");
                };
                times.push(start.elapsed().as_secs_f64());
                let values = (state.try_to_f32_vec()?, out.try_to_u16_vec()?);
                if let Some(expected) = &reference {
                    assert_eq!(&values, expected, "{name}, rows={rows}");
                } else {
                    reference = Some(values);
                }
            }
            times.sort_by(f64::total_cmp);
            eprintln!(
                "scan {name} rows={rows}: {:.3} ms",
                times[times.len() / 2] * 1000.
            );
        }
    }
    Ok(())
}
