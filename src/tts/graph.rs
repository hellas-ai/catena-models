use super::Config;
use anyhow::Result;
pub const SOURCE: &str = include_str!("../catena/tts_model.hex");
/// Complete portable Hex bundle; the receiver supplies only Catena's core stdlib.
pub fn bundle(config: &Config) -> Result<String> {
    config.validate()?;
    Ok(with_exports(&[
        "tts.prepare",
        "tts.synthesize",
        "tts.synthesize-debug",
    ]))
}

fn with_exports(exports: &[&str]) -> String {
    let mut source = include_str!("../catena/memory.hex").to_owned();
    for fragment in [
        include_str!("../catena/numeric.hex"),
        include_str!("../catena/resources.hex"),
        include_str!("../catena/cooperative.hex"),
        include_str!("../catena/state_buffers.hex"),
        include_str!("../catena/qwen35_batch_projection.hex"),
        include_str!("../catena/whisper.hex"),
        SOURCE,
    ] {
        crate::qwen::fused::append_hex(&mut source, fragment);
    }
    let mapping = source
        .split("(def program ")
        .skip(1)
        .filter_map(|definition| {
            let name = definition.split_whitespace().next()?;
            (!name.starts_with("stdlib.") && !exports.contains(&name))
                .then(|| (name.to_owned(), format!("stdlib.{name}")))
        })
        .collect();
    crate::catena::rename_programs(&source, &mapping)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn synthesis_graph_compiles_without_gpu() -> Result<()> {
        let config = Config(Config::expected());
        let source = bundle(&config)?;
        assert!(!source.contains("materializec"));
        let raw =
            metacat::theory::RawTheorySet::from_texts(crate::catena::runtime_sources(&source))?;
        let module = crate::catena::compile(raw, catena_lang::runtime::GpuDialect::Hip)?;
        assert_eq!(module.functions.len(), 3);
        use catena_lang::runtime::ValueKind;
        for (name, extra_inputs, outputs) in [
            ("tts.prepare", 0, vec![ValueKind::MemOwn; 5]),
            ("tts.synthesize", 5, vec![ValueKind::MemOwn, ValueKind::U64]),
            (
                "tts.synthesize-debug",
                5,
                vec![
                    ValueKind::MemOwn,
                    ValueKind::U64,
                    ValueKind::MemOwn,
                    ValueKind::MemOwn,
                    ValueKind::MemOwn,
                    ValueKind::MemOwn,
                    ValueKind::MemOwn,
                ],
            ),
        ] {
            let entry = module
                .functions
                .iter()
                .find(|f| f.source_name == name)
                .unwrap();
            assert_eq!(
                entry.inputs.len(),
                7 + super::super::bindings().len() + extra_inputs,
                "{name}"
            );
            assert_eq!(entry.outputs, outputs, "{name}");
        }
        Ok(())
    }
    #[test]
    #[ignore = "requires a HIP GPU"]
    fn synthesis_stages_with_zero_weights() -> Result<()> {
        use catena_lang::runtime::{GpuDialect, Value};
        let config = Config(Config::expected());
        let source = bundle(&config)?;
        let (rt, artifact) =
            crate::catena::load_runtime(crate::catena::runtime_sources(&source), GpuDialect::Hip)?;
        let weights = super::super::bindings()
            .into_iter()
            .map(|(_, binding)| rt.mem_u16(&vec![0; binding.shape.iter().product()]))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let tokens = rt.mem_u64(&[0, 1])?;
        let inputs = || {
            let mut args = vec![
                tokens.as_ref().into(),
                2u64.into(),
                0u32.into(),
                0u32.into(),
                0f32.into(),
                0f32.into(),
                1f32.into(),
            ];
            args.extend(weights.iter().map(|weight| weight.as_ref().into()));
            args
        };
        let mut prepared = artifact.exec_values("tts.prepare", inputs())?;
        assert_eq!(prepared.len(), 5);
        let Some(Value::MemOwn(header)) = prepared.pop() else {
            anyhow::bail!("missing length header")
        };
        let frames = super::super::checked_frame_count(&header.try_to_f32_vec()?)?;
        let mut args = inputs();
        args.extend(prepared);
        args.push((frames as u64).into());
        let outputs = artifact.exec_values("tts.synthesize-debug", args)?;
        assert_eq!(outputs.len(), 7);
        let mut outputs = outputs.into_iter();
        let Some(Value::MemOwn(waveform)) = outputs.next() else {
            anyhow::bail!("missing waveform")
        };
        assert_eq!(waveform.try_to_f32_vec()?, vec![0.; frames * 256]);
        assert!(matches!(outputs.next(), Some(Value::U64(0))));
        Ok(())
    }

    #[test]
    #[ignore = "requires a HIP GPU"]
    fn kernels_match_cpu_references() -> Result<()> {
        use catena_lang::runtime::{GpuDialect, Value};
        let mut source = with_exports(&[
            "tts.noise",
            "tts.softmax",
            "tts.cumulative",
            "tts.xor",
            "tts.length-header",
            "tts.linear-scalar",
        ]);
        source.push_str(r#"
(def program test.copy : {((cap.own mem) val) ((cap.own mem) val) (u64 val) (u64 val)} -> ((cap.own mem) val) =
 ([input output offset count.]
  ({[.input] [.offset] [.count] [.output] [.count]} stdlib.tts.copy.into [copied.])
  ({[.input] [.copied]} stdlib.qwen3.free-bf16-before)))
"#);
        let (rt, artifact) =
            crate::catena::load_runtime(crate::catena::runtime_sources(&source), GpuDialect::Hip)?;
        let memory = |name: &str, args| -> Result<catena_lang::runtime::MemOwn> {
            let mut outputs = artifact.exec_values(name, args)?;
            match outputs.pop().unwrap() {
                Value::MemOwn(value) => Ok(value),
                _ => anyhow::bail!("expected memory"),
            }
        };
        for (a, b) in [(0u32, 0), (u32::MAX, 0x12345678), (0xDEADBEEF, 0x13579BDF)] {
            let [Value::U32(value)] = artifact.exec("tts.xor", [a.into(), b.into()])? else {
                anyhow::bail!("expected U32")
            };
            assert_eq!(value, a ^ b);
        }
        let bf16 = |xs: &[f32]| {
            xs.iter()
                .map(|x| (x.to_bits() >> 16) as u16)
                .collect::<Vec<_>>()
        };
        let x = rt.mem_u16(&bf16(&[1., 2., 3., 4., 5., 6.]))?;
        let weight = rt.mem_u16(&bf16(&[-1., 0., 1., 2., 1., 0.]))?;
        let projected = memory(
            "tts.linear-scalar",
            vec![
                x.into(),
                weight.as_ref().into(),
                2u64.into(),
                3u64.into(),
                2u64.into(),
            ],
        )?;
        assert_eq!(projected.try_to_u16_vec()?, bf16(&[2., 4., 2., 13.]));
        let input = vec![1000.0f32, 1001.0, 999.0, -1000.0, -1000.0, -1000.0];
        let actual = memory("tts.softmax", vec![rt.mem_f32(&input)?.into(), 3u64.into()])?
            .try_to_f32_vec()?;
        for (row, actual) in input.chunks(3).zip(actual.chunks(3)) {
            let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps = row.iter().map(|x| (x - maximum).exp()).collect::<Vec<_>>();
            let sum: f32 = exps.iter().sum();
            for (value, expected) in actual.iter().zip(exps.iter().map(|x| x / sum)) {
                assert!((value - expected).abs() < 1e-5);
            }
        }
        let cumulative = memory(
            "tts.cumulative",
            vec![
                rt.mem_f32(&[1., 2., 3., 4.])?.into(),
                4u64.into(),
                4u64.into(),
            ],
        )?
        .try_to_f32_vec()?;
        assert_eq!(cumulative, [1., 3., 6., 10.]);
        let header = memory(
            "tts.length-header",
            vec![rt.mem_f32(&cumulative)?.into(), 4u64.into(), 1u64.into()],
        )?
        .try_to_f32_vec()?;
        assert_eq!(header, [10.]);
        let copied = memory(
            "test.copy",
            vec![
                rt.mem_u16(&[0x3F80, 0x4000])?.into(),
                rt.mem_u16(&[0; 5])?.into(),
                2u64.into(),
                2u64.into(),
            ],
        )?;
        assert_eq!(copied.try_to_u16_vec()?, [0, 0, 0x3F80, 0x4000, 0]);
        let noise = |seed: u64, stream: u32| -> Result<Vec<f32>> {
            memory(
                "tts.noise",
                vec![
                    (seed as u32).into(),
                    ((seed >> 32) as u32).into(),
                    stream.into(),
                    16u64.into(),
                ],
            )?
            .try_to_f32_vec()
            .map_err(Into::into)
        };
        let first = noise(0xFEDCBA9876543210, 3)?;
        assert_eq!(first, noise(0xFEDCBA9876543210, 3)?);
        assert_ne!(first, noise(0xFEDCBA9876543211, 3)?);
        assert_ne!(first, noise(0xFEDCBA9876543210, 4)?);
        for (index, actual) in first.iter().enumerate() {
            let mut x = [index as u32, 3, 0, 0];
            let mut key = [0x76543210u32, 0xFEDCBA98];
            for _ in 0..10 {
                let a = 0xD2511F53u64 * x[0] as u64;
                let b = 0xCD9E8D57u64 * x[2] as u64;
                x = [
                    (b >> 32) as u32 ^ x[1] ^ key[0],
                    b as u32,
                    (a >> 32) as u32 ^ x[3] ^ key[1],
                    a as u32,
                ];
                key[0] = key[0].wrapping_add(0x9E3779B9);
                key[1] = key[1].wrapping_add(0xBB67AE85);
            }
            let u = ((x[0] >> 8) as f32 + 0.5) * 2f32.powi(-24);
            let v = ((x[1] >> 8) as f32 + 0.5) * 2f32.powi(-24);
            let expected = (-2.0 * u.ln()).sqrt() * (v * std::f32::consts::TAU).cos();
            assert!(
                (actual - expected).abs() < 0.003,
                "noise {index}: {actual} vs {expected}"
            );
        }
        Ok(())
    }
}
