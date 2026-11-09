use anyhow::{Result, ensure};
use catena_lang::runtime::{GpuDialect, Runtime, Value};
use catena_models::{
    catena::{compile_sources, core_sources, runtime_sources},
    tts,
    weights::{Weights, bf16},
};
use std::{collections::BTreeMap, path::Path};

fn length_source() -> Vec<String> {
    let model = tts::graph::SOURCE
        .replace("tts.", "stdlib.tts.")
        .replace("stdlib.stdlib.tts.", "stdlib.tts.");
    [
        include_str!("../src/catena/memory.hex"),
        include_str!("../src/catena/numeric.hex"),
        include_str!("../src/catena/resources.hex"),
        include_str!("../src/catena/cooperative.hex"),
        &model,
        r#"
(def program check-length : {((cap.own mem) val)} -> ((cap.own mem) val) =
 ([header.] ((const.u64.0x0000000000000002) stdlib.models.memory.alloc-u64 [result.])
 {[.header] (const.u64.0x0000000000000001) [.result]} stdlib.tts.prealloc.metadata))
"#,
    ]
    .into_iter()
    .map(|source| source.replace(" bf16x2.dot-f32 ", " stdlib.models.math.dot-pair-f32 "))
    .collect()
}

#[test]
fn preallocated_metadata_compiles() -> Result<()> {
    let source = length_source();
    for dialect in [GpuDialect::Hip, GpuDialect::Cuda] {
        let module = compile_sources(
            core_sources()
                .map(|s| -> &str { s })
                .chain(source.iter().map(String::as_str)),
            dialect,
        )?;
        ensure!(
            !module.source.contains("catena_host_read"),
            "unexpected HostRead dependency"
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires HIP GPU"]
fn device_length_metadata() -> Result<()> {
    let source = length_source();
    let mut rt = Runtime::new(GpuDialect::Hip)?;
    let artifact = rt.load(compile_sources(
        core_sources()
            .map(|s| -> &str { s })
            .chain(source.iter().map(String::as_str)),
        GpuDialect::Hip,
    )?)?;
    for (value, frames) in [
        (1., 1),
        (1875., 1875),
        (0., 0),
        (-1., 0),
        (1.5, 0),
        (1876., 0),
        (f32::NAN, 0),
        (f32::INFINITY, 0),
        (f32::NEG_INFINITY, 0),
    ] {
        let header = rt.mem_f32(&[value])?;
        let [Value::MemOwn(result)] = artifact.exec("check-length", [header.into()])? else {
            anyhow::bail!("wrong metadata ABI")
        };
        let actual = result.try_to_u64_vec()?;
        ensure!(
            actual == [frames * 256, u64::from(frames == 0)],
            "{value}: unexpected metadata {actual:?}"
        );
    }
    Ok(())
}

fn assert_dump(dir: &Path, name: &str, values: &[f32]) -> Result<()> {
    let actual: Vec<u8> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
    ensure!(
        actual == std::fs::read(dir.join(format!("{name}.f32")))?,
        "baseline differs: {name}"
    );
    Ok(())
}

#[test]
#[ignore = "requires HIP GPU, cached MMS weights, and /tmp/mms-single-baseline"]
fn exported_single_call_matches_baseline() -> Result<()> {
    let dir = tts::snapshot("facebook/mms-tts-eng")?;
    let source = tts::graph::bundle(&tts::Config::read(&dir)?)?;
    let module = compile_sources(runtime_sources(&source), GpuDialect::Hip)?;
    let mut rt = Runtime::new(GpuDialect::Hip)?;
    let artifact = rt.load(module)?;
    let mut checkpoint = Weights::open(&dir.join("model.safetensors"))?;
    let bindings = tts::bindings();
    let ordinary = bindings
        .iter()
        .filter(|(_, b)| !b.normalized && !b.transpose)
        .map(|(n, _)| n.clone())
        .collect::<Vec<_>>();
    checkpoint.prepare(&rt, &ordinary, false)?;
    let mut prepared = BTreeMap::new();
    for (name, binding) in &bindings {
        if binding.normalized || binding.transpose {
            prepared.insert(
                name.clone(),
                rt.mem_u16(&bf16(&tts::materialize(&checkpoint, name, binding)?))?,
            );
        }
    }
    let ids = tts::tokenizer::Tokenizer::read(&dir)?.encode("Hi.")?;
    let tokens = rt.mem_u64(&ids)?;
    let args = |rate: f32| -> Result<Vec<Value<'_>>> {
        let mut values = vec![
            tokens.as_ref().into(),
            (ids.len() as u64).into(),
            0u32.into(),
            0u32.into(),
            0.8f32.into(),
            0.667f32.into(),
            rate.into(),
        ];
        for (name, _) in &bindings {
            values.push(if let Some(w) = prepared.get(name) {
                w.as_ref().into()
            } else {
                checkpoint.gpu_ref(name)?.into()
            });
        }
        Ok(values)
    };
    let baseline = Path::new("/tmp/mms-single-baseline");
    let mut values = artifact
        .exec_values("tts.synthesize-debug", args(1.)?)?
        .into_iter();
    let Some(Value::MemOwn(waveform)) = values.next() else {
        anyhow::bail!("missing waveform")
    };
    let Some(Value::MemOwn(result)) = values.next() else {
        anyhow::bail!("missing metadata")
    };
    let result = result.try_to_u64_vec()?;
    ensure!(result[1] == 0, "synthesis failed");
    let mut wave = waveform.try_to_f32_vec()?;
    wave.truncate(result[0] as usize);
    assert_dump(baseline, "waveform", &wave)?;
    for (name, is_bf16) in [
        ("durations", false),
        ("stats", true),
        ("duration_noise", false),
        ("latent_noise", false),
        ("flow_output", true),
    ] {
        let Some(Value::MemOwn(mem)) = values.next() else {
            anyhow::bail!("missing capture {name}")
        };
        let data: Vec<f32> = if is_bf16 {
            mem.try_to_u16_vec()?
                .into_iter()
                .map(|v| f32::from_bits((v as u32) << 16))
                .collect()
        } else {
            mem.try_to_f32_vec()?
        };
        let count = match name {
            "durations" => ids.len(),
            "stats" => ids.len() * 384,
            "duration_noise" => ids.len() * 2,
            _ => 192 * wave.len() / 256,
        };
        assert_dump(baseline, name, &data[..count])?;
    }
    ensure!(values.next().is_none(), "extra debug outputs");
    let mut values = artifact
        .exec_values("tts.synthesize", args(1.)?)?
        .into_iter();
    let Some(Value::MemOwn(mem)) = values.next() else {
        anyhow::bail!("missing waveform")
    };
    let Some(Value::MemOwn(result)) = values.next() else {
        anyhow::bail!("missing metadata")
    };
    let result = result.try_to_u64_vec()?;
    ensure!(result[1] == 0, "normal synthesis failed");
    let mut normal = mem.try_to_f32_vec()?;
    normal.truncate(result[0] as usize);
    ensure!(
        normal
            .iter()
            .map(|v| v.to_bits())
            .eq(wave.iter().map(|v| v.to_bits())),
        "normal/debug waveforms differ"
    );
    let mut invalid = artifact
        .exec_values("tts.synthesize-debug", args(1e-20)?)?
        .into_iter();
    let Some(Value::MemOwn(_capacity)) = invalid.next() else {
        anyhow::bail!("missing output capacity")
    };
    let Some(Value::MemOwn(result)) = invalid.next() else {
        anyhow::bail!("missing failure metadata")
    };
    ensure!(
        result.try_to_u64_vec()? == [0, 1],
        "invalid duration accepted"
    );
    Ok(())
}

#[test]
#[ignore = "requires HIP GPU; bounded convolution tails and poisoned padding"]
fn bounded_convolutions_match_original() -> Result<()> {
    let mut source = length_source();
    for helper in [
        include_str!("../src/catena/state_buffers.hex"),
        include_str!("../src/catena/qwen35_batch_projection.hex"),
        include_str!("../src/catena/whisper.hex"),
    ] {
        source.push(helper.replace(" bf16x2.dot-f32 ", " stdlib.models.math.dot-pair-f32 "));
    }
    source.push(r#"
(def program bounded-conv : {((cap.own mem) val) ((cap.ref mem) val) ((cap.ref mem) val) ((cap.own mem) val) ((cap.own mem) val) (u64 val) (u64 val) (u64 val) (u64 val) (u64 val) (u64 val) (u64 val)} -> ((cap.own mem) val) =
 ([x w b metadata out capacity out-capacity ks dilation stride pad transpose.]
 {[.x] [.w] [.b] [.capacity] [.out-capacity] (const.u64.0x0000000000000002) (const.u64.0x0000000000000003) [.ks] [.dilation] [.stride] [.pad] [.transpose] (const.u64.0x0000000000000001) [.metadata] [.out]} stdlib.tts.prealloc.conv))
(def program reference-conv : {((cap.own mem) val) ((cap.ref mem) val) ((cap.ref mem) val) (u64 val) (u64 val) (u64 val) (u64 val) (u64 val) (u64 val) (u64 val)} -> ((cap.own mem) val) =
 ([x w b frames out-frames ks dilation stride pad transpose.]
 {[.x] [.w] [.b] [.frames] [.out-frames] (const.u64.0x0000000000000002) (const.u64.0x0000000000000003) [.ks] [.dilation] [.stride] [.pad] [.transpose] (const.u64.0x0000000000000001)} stdlib.tts.conv))
"#.into());
    let mut rt = Runtime::new(GpuDialect::Hip)?;
    let artifact = rt.load(compile_sources(
        core_sources()
            .map(|s| -> &str { s })
            .chain(source.iter().map(String::as_str)),
        GpuDialect::Hip,
    )?)?;
    for (frames, ks, dilation, stride, pad, transpose) in
        [(257, 3, 3, 1, 3, 0), (33, 4, 1, 2, 1, 1)]
    {
        let x: Vec<f32> = (0..frames * 2)
            .map(|i| (i % 17) as f32 / 32. - 0.25)
            .collect();
        let mut capacity = vec![0x7fc0; 1875 * 2]; // BF16 NaNs must never enter live convolution bounds.
        capacity[..x.len()].copy_from_slice(&bf16(&x));
        let w: Vec<f32> = (0..3 * 2 * ks)
            .map(|i| (i % 11) as f32 / 64. - 0.1)
            .collect();
        let w = rt.mem_u16(&bf16(&w))?;
        let bias = rt.mem_u16(&bf16(&[0.125, -0.25, 0.375]))?;
        let metadata = rt.mem_u64(&[(frames * 256) as u64, 0])?;
        let out = rt.mem_u16(&vec![0x40e8; 1875 * stride * 3])?;
        let [Value::MemOwn(actual)] = artifact.exec(
            "bounded-conv",
            [
                rt.mem_u16(&capacity)?.into(),
                w.as_ref().into(),
                bias.as_ref().into(),
                metadata.into(),
                out.into(),
                1875u64.into(),
                (1875 * stride as u64).into(),
                (ks as u64).into(),
                (dilation as u64).into(),
                (stride as u64).into(),
                (pad as u64).into(),
                (transpose as u64).into(),
            ],
        )?
        else {
            anyhow::bail!("bad convolution output")
        };
        let [Value::MemOwn(reference)] = artifact.exec(
            "reference-conv",
            [
                rt.mem_u16(&bf16(&x))?.into(),
                w.as_ref().into(),
                bias.as_ref().into(),
                (frames as u64).into(),
                ((frames * stride) as u64).into(),
                (ks as u64).into(),
                (dilation as u64).into(),
                (stride as u64).into(),
                (pad as u64).into(),
                (transpose as u64).into(),
            ],
        )?
        else {
            anyhow::bail!("bad reference output")
        };
        let actual = actual.try_to_u16_vec()?;
        let reference = reference.try_to_u16_vec()?;
        ensure!(
            actual[..reference.len()] == reference,
            "convolution differs at {frames} frames"
        );
        ensure!(
            actual[reference.len()..].iter().all(|x| *x == 0x40e8),
            "inactive output modified"
        );
    }
    Ok(())
}
