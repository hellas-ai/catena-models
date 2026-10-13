//! Opt-in device checks against independent scalar references.
use super::*;
use catena_lang::runtime::{Artifact, MemOwn, Runtime, Value};

const WRAPPERS: &str = r#"
(def program test.project : {((cap.own mem) val) ((cap.ref mem) val) (u64 val) (u64 val)} -> {((cap.own mem) val) ((cap.own mem) val)} =
 ([input weight features outputs.] ({[.input] [.weight] [.features] [.outputs]} qwen3.project [result.]) {[.input] [.result]}))
(def program test.normalize : {((cap.own mem) val) ((cap.ref mem) val) (u64 val) (f32 val) (u64 val)} -> {((cap.own mem) val) ((cap.own mem) val)} =
 ([input weight cols cols-f rows.] ({[.input] [.weight] [.cols] [.cols-f] [.rows]} qwen3.normalize [result.]) {[.input] [.result]}))
(def program test.rotate : {((cap.own mem) val) ((cap.own mem) val)} -> {((cap.own mem) val) ((cap.own mem) val) ((cap.own mem) val)} =
 ([input rope.] ({[.input] [.rope]} qwen3.rotate [result.]) {[.input] [.rope] [.result]}))
(def program test.append : {((cap.own mem) val) ((cap.own mem) val) (u64 val)} -> {((cap.own mem) val) ((cap.own mem) val)} =
 ([input cache offset.] ({[.input] [.cache] [.offset]} qwen3.append [result.]) {[.input] [.result]}))
(def program test.scores : {((cap.own mem) val) ((cap.own mem) val) (u64 val) (u64 val)} -> {((cap.own mem) val) ((cap.own mem) val) ((cap.own mem) val)} =
 ([query cache keys offset.] ({[.query] [.cache] [.keys] [.offset]} qwen3.scores [result.]) {[.query] [.cache] [.result]}))
(def program test.softmax : {((cap.own mem) val) (u64 val)} -> {((cap.own mem) val) ((cap.own mem) val)} =
 ([input keys.] ({[.input] [.keys]} qwen3.softmax [result.]) {[.input] [.result]}))
(def program test.context : {((cap.own mem) val) ((cap.own mem) val) (u64 val) (u64 val)} -> {((cap.own mem) val) ((cap.own mem) val) ((cap.own mem) val)} =
 ([prob cache keys offset.] ({[.prob] [.cache] [.keys] [.offset]} qwen3.context [result.]) {[.prob] [.cache] [.result]}))
(def program test.argmax : ((cap.own mem) val) -> {((cap.own mem) val) ((cap.own mem) val)} =
 ([logits.] ([.logits] qwen3.argmax [token.]) {[.logits] [.token]}))
"#;
fn bf16(x: f32) -> u16 {
    let b = x.to_bits();
    ((b.wrapping_add(0x7fff + ((b >> 16) & 1))) >> 16) as u16
}
fn rounded(x: f32) -> f32 {
    f32::from_bits(u32::from(bf16(x)) << 16)
}
fn upload(rt: &Runtime, xs: &[f32]) -> Result<MemOwn> {
    Ok(rt.mem_u16(&xs.iter().copied().map(bf16).collect::<Vec<_>>())?)
}
fn values(m: &MemOwn) -> Result<Vec<f32>> {
    Ok(m.try_to_u16_vec()?
        .into_iter()
        .map(|b| f32::from_bits(u32::from(b) << 16))
        .collect())
}
fn memories(a: &Artifact, name: &str, args: Vec<Value<'_>>) -> Result<Vec<MemOwn>> {
    a.exec_values(name, args)?
        .into_iter()
        .map(|v| match v {
            Value::MemOwn(m) => Ok(m),
            _ => anyhow::bail!("unexpected scalar"),
        })
        .collect()
}
fn last(mut ms: Vec<MemOwn>) -> MemOwn {
    ms.pop().unwrap()
}
fn close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= tolerance * e.abs().max(1.0),
            "index {i}: {a} != {e}"
        );
    }
}

#[test]
fn experimental_kernels_match_independent_references_on_gpu() -> Result<()> {
    if std::env::var_os("CATENA_QWEN3_EXPERIMENTAL_GPU_TESTS").is_none() {
        return Ok(());
    }
    let c = tests::tiny_config();
    let options =
        InferenceOptions::new(InferencePrecision::Bf16).with_codegen(CodegenKind::Experimental);
    let model = source(&c, options)?;
    let module = compile(
        crate::catena::runtime_sources_for_options(&model, options)
            .into_iter()
            .chain([WRAPPERS]),
        GpuDialect::Hip,
    )?;
    let mut rt = Runtime::new(GpuDialect::Hip)?;
    let a = rt.load(module)?;
    // Output count is not a multiple of the 256-thread launch width.
    let x = (0..32)
        .map(|i| rounded((i as f32 - 15.0) / 16.0))
        .collect::<Vec<_>>();
    let weights = (0..257 * 32)
        .map(|i| rounded(((i * 7 % 23) as f32 - 11.0) / 32.0))
        .collect::<Vec<_>>();
    let w = upload(&rt, &weights)?;
    let result = last(memories(
        &a,
        "test.project",
        vec![
            upload(&rt, &x)?.into(),
            w.as_ref().into(),
            32u64.into(),
            257u64.into(),
        ],
    )?);
    let expected = weights
        .chunks(32)
        .map(|row| rounded(row.iter().zip(&x).map(|(&w, &v)| w * v).sum()))
        .collect::<Vec<_>>();
    close(&values(&result)?, &expected, 0.01);
    // Per-head normalization, learned weights stored as F32, BF16 rounding twice.
    let norm = (0..16).map(|i| 0.7 + i as f32 / 37.0).collect::<Vec<_>>();
    let n = rt.mem_f32(&norm)?;
    let result = last(memories(
        &a,
        "test.normalize",
        vec![
            upload(&rt, &x)?.into(),
            n.as_ref().into(),
            16u64.into(),
            16f32.into(),
            2u64.into(),
        ],
    )?);
    let expected = x
        .chunks(16)
        .flat_map(|row| {
            let inv = (row.iter().map(|v| v * v).sum::<f32>() / 16.0 + c.rms_norm_eps)
                .sqrt()
                .recip();
            row.iter()
                .zip(&norm)
                .map(move |(&v, &w)| rounded(rounded(v * inv) * rounded(w)))
        })
        .collect::<Vec<_>>();
    close(&values(&result)?, &expected, 0.001);
    for position in [0u64, 3, 127] {
        let [Value::MemOwn(rope)] = a.exec("qwen3.rope-one", [position.into()])? else {
            anyhow::bail!("rope output");
        };
        let actual_rope = rope.try_to_f32_vec()?;
        let mut expected_rope = vec![0.0; 16];
        for pair in 0..8 {
            let angle = position as f32 * c.rope_theta.powf(-((2 * pair) as f32) / 16.0);
            expected_rope[pair] = angle.cos();
            expected_rope[pair + 8] = angle.sin();
        }
        close(&actual_rope, &expected_rope, 1e-5);
        let result = last(memories(
            &a,
            "test.rotate",
            vec![upload(&rt, &x)?.into(), rope.into()],
        )?);
        let expected = x
            .chunks(16)
            .flat_map(|head| {
                (0..16)
                    .map(|channel| {
                        let pair = channel % 8;
                        let other = if channel < 8 {
                            channel + 8
                        } else {
                            channel - 8
                        };
                        rounded(
                            head[channel] * expected_rope[pair]
                                + (if channel < 8 { -1.0 } else { 1.0 })
                                    * head[other]
                                    * expected_rope[pair + 8],
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        close(&values(&result)?, &expected, 0.01);
    }
    let keys = 3usize;
    let capacity = 5usize;
    let kv = 16usize;
    let value_offset = capacity * kv;
    let key_rows = (0..capacity * kv)
        .map(|i| rounded(((i * 3 % 17) as f32 - 8.0) / 16.0))
        .collect::<Vec<_>>();
    let value_rows = (0..capacity * kv)
        .map(|i| rounded(((i * 5 % 19) as f32 - 9.0) / 16.0))
        .collect::<Vec<_>>();
    let cache = upload(&rt, &[key_rows.clone(), value_rows.clone()].concat())?;
    let result = last(memories(
        &a,
        "test.scores",
        vec![
            upload(&rt, &x)?.into(),
            cache.into(),
            (keys as u64).into(),
            0u64.into(),
        ],
    )?);
    let expected = (0..2)
        .flat_map(|head| {
            (0..keys)
                .map(|key| {
                    rounded(
                        x[head * 16..(head + 1) * 16]
                            .iter()
                            .zip(&key_rows[key * 16..(key + 1) * 16])
                            .map(|(&q, &k)| q * k)
                            .sum::<f32>()
                            / 4.0,
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    close(&result.try_to_f32_vec()?, &expected, 1e-5);
    let probabilities = last(memories(
        &a,
        "test.softmax",
        vec![result.into(), (keys as u64).into()],
    )?);
    let expected_prob = expected
        .chunks(keys)
        .flat_map(|row| {
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exp = row.iter().map(|v| (v - max).exp()).collect::<Vec<_>>();
            let total = exp.iter().sum::<f32>();
            exp.into_iter()
                .map(|v| rounded(v / total))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    close(&values(&probabilities)?, &expected_prob, 0.001);
    let result = last(memories(
        &a,
        "test.context",
        vec![
            probabilities.into(),
            upload(&rt, &[key_rows, value_rows.clone()].concat())?.into(),
            (keys as u64).into(),
            (value_offset as u64).into(),
        ],
    )?);
    let expected = (0..32)
        .map(|index| {
            rounded(
                (0..keys)
                    .map(|key| {
                        expected_prob[(index / 16) * keys + key] * value_rows[key * 16 + index % 16]
                    })
                    .sum(),
            )
        })
        .collect::<Vec<_>>();
    close(&values(&result)?, &expected, 0.001);
    let row = vec![0.75; 16];
    let zero = vec![0.0; 2 * capacity * kv];
    let cache = last(memories(
        &a,
        "test.append",
        vec![
            upload(&rt, &row)?.into(),
            upload(&rt, &zero)?.into(),
            ((value_offset + 2 * kv) as u64).into(),
        ],
    )?);
    let actual = values(&cache)?;
    let mut expected = zero;
    expected[value_offset + 2 * kv..value_offset + 3 * kv].copy_from_slice(&row);
    assert_eq!(actual, expected);
    for xs in [vec![2.0, 2.0, -1.0], vec![-2.0, -1.0, -3.0]] {
        let token = last(memories(&a, "test.argmax", vec![upload(&rt, &xs)?.into()])?);
        assert_eq!(
            token.try_to_u64_vec()?,
            vec![if xs[0] == 2.0 { 0 } else { 1 }]
        );
    }
    let mut wide_logits = vec![-1.0; 151_936];
    wide_logits[100_001] = 3.0;
    let token = last(memories(
        &a,
        "test.argmax",
        vec![upload(&rt, &wide_logits)?.into()],
    )?);
    assert_eq!(token.try_to_u64_vec()?, vec![100_001]);
    Ok(())
}

#[test]
fn experimental_checkpoint_parity_and_cache_sessions_on_gpu() -> Result<()> {
    let Some(path) = std::env::var_os("CATENA_QWEN3_EXPERIMENTAL_GGUF") else {
        return Ok(());
    };
    use crate::{model::Model, tokenizer::QwenTokenizer};
    let tokenizer = QwenTokenizer::from_gguf(&path)?
        .ok_or_else(|| anyhow::anyhow!("missing GGUF tokenizer"))?;
    let reference = Model::load_gguf_with_precision(&path, InferencePrecision::Bf16)?;
    let experimental = Model::load_gguf_with_options(
        &path,
        InferenceOptions::new(InferencePrecision::Bf16).with_codegen(CodegenKind::Experimental),
    )?;
    let mut tokens = tokenizer.encode("Gravity is")?;
    let capacity = tokens.len() + 20;
    let mut session = experimental.session(capacity)?;
    assert!(session.decode_logits(tokens[0]).is_err());
    assert!(session.prefill_logits(&[]).is_err());
    let mut actual = session.prefill_logits(&tokens)?;
    assert!(session.prefill_logits(&tokens).is_err());
    let argmax = |xs: &[f32]| {
        xs.iter()
            .enumerate()
            .max_by(|(ia, a), (ib, b)| a.total_cmp(b).then_with(|| ib.cmp(ia)))
            .unwrap()
            .0 as u64
    };
    for step in 0..20 {
        let expected = reference.logits(&tokens)?;
        let error = actual
            .iter()
            .zip(&expected)
            .map(|(&a, &e)| f64::from(a - e).powi(2))
            .sum::<f64>();
        let energy = expected.iter().map(|&e| f64::from(e).powi(2)).sum::<f64>();
        assert!(
            (error / energy.max(1e-30)).sqrt() <= 0.05,
            "logit RMS error exceeds 5% at step {step}"
        );
        let token = argmax(&expected);
        assert_eq!(argmax(&actual), token, "greedy token at step {step}");
        if step < 3 {
            assert_eq!(
                actual,
                experimental.logits(&tokens)?,
                "cache vs full-prefix"
            );
        }
        tokens.push(token);
        actual = session.decode_logits(token)?;
    }
    assert_eq!(session.position(), capacity);
    assert!(session.decode_logits(tokens[0]).is_err());
    let mut independent = experimental.session(tokens.len())?;
    assert_eq!(independent.prefill_logits(&tokens)?, actual);
    Ok(())
}
