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
const FUSED_WRAPPERS: &str = r#"
(def program test.residual : {((cap.own mem) val) ((cap.ref mem) val) (u64 val) (u64 val) ((cap.own mem) val)} -> {((cap.own mem) val) ((cap.own mem) val) ((cap.own mem) val)} =
 ([input weight features outputs residual.] ({[.input] [.weight] [.features] [.outputs] [.residual]} qwen3.fused.residual [result.]) {[.input] [.residual] [.result]}))
(def program test.gate : {((cap.own mem) val) ((cap.ref mem) val) (u64 val) (u64 val) ((cap.ref mem) val)} -> {((cap.own mem) val) ((cap.own mem) val)} =
 ([input weight features outputs up.] ({[.input] [.weight] [.features] [.outputs] [.up]} qwen3.fused.gate [result.]) {[.input] [.result]}))
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
    check_kernels_on_gpu(ExperimentalKernelKind::Reference)
}

#[test]
fn experimental_fused_kernels_match_independent_references_on_gpu() -> Result<()> {
    check_kernels_on_gpu(ExperimentalKernelKind::Fused)
}

fn check_kernels_on_gpu(kind: ExperimentalKernelKind) -> Result<()> {
    if std::env::var_os("CATENA_QWEN3_EXPERIMENTAL_GPU_TESTS").is_none() {
        return Ok(());
    }
    let c = tests::tiny_config();
    let options = InferenceOptions::new(InferencePrecision::Bf16)
        .with_codegen(CodegenKind::Experimental)
        .with_experimental_kernels(kind);
    let model = source(&c, options)?;
    let wrappers = if kind == ExperimentalKernelKind::Fused {
        WRAPPERS
            .replace("} qwen3.project [", "} qwen3.fused.project [")
            .replace("} qwen3.normalize [", "} qwen3.fused.normalize [")
            .replace("qwen3.argmax [", "qwen3.fused.argmax [")
            + FUSED_WRAPPERS
    } else {
        WRAPPERS.to_owned()
    };
    let module = compile(
        crate::catena::runtime_sources_for_options(&model, options)
            .into_iter()
            .chain([wrappers.as_str()]),
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
    if kind == ExperimentalKernelKind::Fused {
        let residual = (0..257)
            .map(|i| rounded((i as f32 - 128.0) / 32.0))
            .collect::<Vec<_>>();
        let result = last(memories(
            &a,
            "test.residual",
            vec![
                upload(&rt, &x)?.into(),
                w.as_ref().into(),
                32u64.into(),
                257u64.into(),
                upload(&rt, &residual)?.into(),
            ],
        )?);
        let target = expected
            .iter()
            .zip(&residual)
            .map(|(&p, &r)| rounded(p + r))
            .collect::<Vec<_>>();
        close(&values(&result)?, &target, 0.01);
        let up = weights
            .iter()
            .map(|&w| rounded(w * 0.75))
            .collect::<Vec<_>>();
        let uw = upload(&rt, &up)?;
        let result = last(memories(
            &a,
            "test.gate",
            vec![
                upload(&rt, &x)?.into(),
                w.as_ref().into(),
                32u64.into(),
                257u64.into(),
                uw.as_ref().into(),
            ],
        )?);
        let target = expected
            .iter()
            .zip(up.chunks(32))
            .map(|(&g, row)| {
                let u = rounded(row.iter().zip(&x).map(|(&w, &v)| w * v).sum());
                rounded(rounded(g / (1.0 + (-g).exp())) * u)
            })
            .collect::<Vec<_>>();
        close(&values(&result)?, &target, 0.01);
    }
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
    check_checkpoint_on_gpu(ExperimentalKernelKind::Reference)
}

#[test]
fn experimental_fused_checkpoint_parity_and_cache_sessions_on_gpu() -> Result<()> {
    check_checkpoint_on_gpu(ExperimentalKernelKind::Fused)
}

fn check_checkpoint_on_gpu(kind: ExperimentalKernelKind) -> Result<()> {
    let Some(path) = std::env::var_os("CATENA_QWEN3_EXPERIMENTAL_GGUF") else {
        return Ok(());
    };
    use crate::{model::Model, tokenizer::QwenTokenizer};
    let tokenizer = QwenTokenizer::from_gguf(&path)?
        .ok_or_else(|| anyhow::anyhow!("missing GGUF tokenizer"))?;
    let reference = Model::load_gguf_with_precision(&path, InferencePrecision::Bf16)?;
    let experimental = Model::load_gguf_with_options(
        &path,
        InferenceOptions::new(InferencePrecision::Bf16)
            .with_codegen(CodegenKind::Experimental)
            .with_experimental_kernels(kind),
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

#[test]
fn experimental_qwen35_checkpoint_parity_on_gpu() -> Result<()> {
    let Some(path) = std::env::var_os("CATENA_QWEN35_EXPERIMENTAL_GGUF") else {
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
    let argmax = |xs: &[f32]| {
        xs.iter()
            .enumerate()
            .max_by(|(ia, a), (ib, b)| a.total_cmp(b).then_with(|| ib.cmp(ia)))
            .unwrap()
            .0 as u64
    };
    for prompt in [
        "Gravity is".to_owned(),
        "The quick brown fox jumps over the lazy dog. ".repeat(12),
    ] {
        let tokens = tokenizer.encode(&prompt)?;
        let capacity = tokens.len() + 20;
        let mut expected = reference.session(capacity)?;
        let mut actual = experimental.session(capacity)?;
        let mut e = expected.prefill_logits(&tokens)?;
        let mut a = actual.prefill_logits(&tokens)?;
        let expected_cache = expected.cache_values_for_test()?;
        let actual_cache = actual.cache_values_for_test()?;
        assert_eq!(expected_cache.len(), actual_cache.len());
        assert_eq!(
            expected_cache
                .iter()
                .zip(&actual_cache)
                .position(|(e, a)| e != a),
            None,
            "Qwen3.5 cache after {}-token prefill",
            tokens.len()
        );
        for step in 0..20 {
            let error = a
                .iter()
                .zip(&e)
                .map(|(&a, &e)| f64::from(a - e).powi(2))
                .sum::<f64>();
            let energy = e.iter().map(|&e| f64::from(e).powi(2)).sum::<f64>();
            assert!(
                (error / energy.max(1e-30)).sqrt() <= 0.05,
                "Qwen3.5 logit RMS for {} prompt tokens at step {step}: {}",
                tokens.len(),
                (error / energy).sqrt()
            );
            let token = argmax(&e);
            assert_eq!(
                argmax(&a),
                token,
                "Qwen3.5 greedy token after {} prompt tokens at step {step}",
                tokens.len()
            );
            e = expected.decode_logits(token)?;
            a = actual.decode_logits(token)?;
        }
        assert_eq!(actual.position(), capacity);
    }
    Ok(())
}

#[test]
fn experimental_qwen35_delta_retains_f32_state_on_gpu() -> Result<()> {
    if std::env::var_os("CATENA_QWEN3_EXPERIMENTAL_GPU_TESTS").is_none() {
        return Ok(());
    }
    let mut c = ModelConfig::qwen35_0_8b();
    c.hidden_size = 32;
    c.intermediate_size = 64;
    c.num_layers = 4;
    c.num_attention_heads = 2;
    c.num_key_value_heads = 1;
    c.head_dim = 16;
    c.rope_dimension_count = 8;
    c.vocabulary_size = 33;
    c.linear_state_size = Some(16);
    c.linear_group_count = Some(1);
    c.linear_time_step_rank = Some(2);
    c.linear_inner_size = Some(32);
    let options =
        InferenceOptions::new(InferencePrecision::Bf16).with_codegen(CodegenKind::Experimental);
    let source = source(&c, options)?;
    let wrapper = r#"
(def program test.delta : {((cap.own mem) val) ((cap.own mem) val) ((cap.own mem) val) ((cap.own mem) val) ((cap.own mem) val) ((cap.ref mem) val) ((cap.ref mem) val) ((cap.own mem) val)} -> {((cap.own mem) val) ((cap.own mem) val)} =
 ([conv query key alpha beta dt a shadow.] ({[.conv] [.query] [.key] [.alpha] [.beta] [.dt] [.a] [.shadow] (const.u64.0x0000000000000000)} qwen35.delta [output.]) {[.shadow] [.output]}))
(def program test.qwen35-conv : {((cap.own mem) val) ((cap.own mem) val) ((cap.ref mem) val) (u64 val)} -> {((cap.own mem) val) ((cap.own mem) val)} =
 ([qkv cache weight position.] ({[.qkv] [.cache] [.weight] (const.u64.0x0000000000000000) [.position]} qwen35.conv [output.]) {[.cache] [.output]}))
(def program test.qwen35-scores : {((cap.own mem) val) ((cap.own mem) val) (u64 val) (u64 val)} -> {((cap.own mem) val) ((cap.own mem) val) ((cap.own mem) val)} =
 ([query cache keys offset.] ({[.query] [.cache] [.keys] [.offset]} qwen3.scores [output.]) {[.query] [.cache] [.output]}))
(def program test.qwen35-softmax : {((cap.own mem) val) (u64 val)} -> {((cap.own mem) val) ((cap.own mem) val)} =
 ([scores keys.] ({[.scores] [.keys]} qwen35.softmax [output.]) {[.scores] [.output]}))
(def program test.qwen35-context : {((cap.own mem) val) ((cap.own mem) val) (u64 val) (u64 val)} -> {((cap.own mem) val) ((cap.own mem) val) ((cap.own mem) val)} =
 ([prob cache keys offset.] ({[.prob] [.cache] [.keys] [.offset]} qwen35.context [output.]) {[.prob] [.cache] [.output]}))
(def program test.qwen35-normalize : {((cap.own mem) val) ((cap.ref mem) val) (u64 val) (f32 val) (u64 val) (u64 val)} -> {((cap.own mem) val) ((cap.own mem) val)} =
 ([input weight columns cf rows stages.] ({[.input] [.weight] [.columns] [.cf] [.rows] [.stages]} qwen35.normalize [output.]) {[.input] [.output]}))
"#;
    let module = compile(
        crate::catena::runtime_sources_for_options(&source, options)
            .into_iter()
            .chain([wrapper]),
        GpuDialect::Hip,
    )?;
    let mut rt = Runtime::new(GpuDialect::Hip)?;
    let artifact = rt.load(module)?;
    let conv_weight = (0..256)
        .map(|i| ((i * 17 % 23) as f32 - 11.0) / 31.0)
        .collect::<Vec<_>>();
    let cw = rt.mem_f32(&conv_weight)?;
    let mut history = upload(&rt, &vec![0.0; 192])?;
    let mut inputs = Vec::<Vec<f32>>::new();
    for position in 0..7 {
        let qkv = (0..64)
            .map(|i| rounded(((i * 7 + position * 13) % 41) as f32 / 23.0 - 0.7))
            .collect::<Vec<_>>();
        inputs.push(qkv.clone());
        let mut out = memories(
            &artifact,
            "test.qwen35-conv",
            vec![
                upload(&rt, &qkv)?.into(),
                history.into(),
                cw.as_ref().into(),
                (position as u64).into(),
            ],
        )?;
        let result = out.pop().unwrap();
        history = out.pop().unwrap();
        let gold = (0..64)
            .map(|channel| {
                let mut taps = [0.0; 4];
                for tap in 0..4 {
                    if position + tap >= 3 {
                        taps[tap] =
                            inputs[position + tap - 3][channel] * conv_weight[channel * 4 + tap];
                    }
                }
                let raw = rounded((taps[0] + taps[2]) + (taps[1] + taps[3]));
                rounded(raw / (1.0 + (-raw).exp()))
            })
            .collect::<Vec<_>>();
        close(&values(&result)?, &gold, 0.01);
        let raw = values(&history)?;
        assert_eq!(
            &raw[position % 3 * 64..(position % 3 + 1) * 64],
            qkv.as_slice(),
            "convolution cache stores raw QKV"
        );
    }
    for width in [16usize, 32, 1024] {
        let input = (0..width * 2)
            .map(|i| rounded((i * 137 % 2029) as f32 / 201.0 - 4.0))
            .collect::<Vec<_>>();
        let norm = (0..width)
            .map(|i| 0.7 + (i % 37) as f32 / 113.0)
            .collect::<Vec<_>>();
        let weight = rt.mem_f32(&norm)?;
        let out = last(memories(
            &artifact,
            "test.qwen35-normalize",
            vec![
                upload(&rt, &input)?.into(),
                weight.as_ref().into(),
                (width as u64).into(),
                (width as f32).into(),
                2u64.into(),
                (width.trailing_zeros() as u64).into(),
            ],
        )?);
        let gold = input
            .chunks(width)
            .flat_map(|row| {
                let mut squared = row.iter().map(|&v| v * v).collect::<Vec<_>>();
                let mut stride = 1;
                while stride < width {
                    for left in (0..width).step_by(stride * 2) {
                        if left + stride < width {
                            squared[left] += squared[left + stride];
                        }
                    }
                    stride *= 2;
                }
                let inverse = (squared[0] / width as f32 + c.rms_norm_eps).sqrt().recip();
                row.iter()
                    .zip(&norm)
                    .map(move |(&v, &w)| rounded(rounded(v * inverse) * rounded(w)))
            })
            .collect::<Vec<_>>();
        close(&values(&out)?, &gold, 0.001);
    }
    for keys in [1usize, 3, 33, 129, 257] {
        let query = (0..32)
            .map(|i| rounded((i * 7 % 31) as f32 / 23.0 - 0.6))
            .collect::<Vec<_>>();
        let cache = (0..keys * 16)
            .map(|i| rounded((i * 13 % 53) as f32 / 37.0 - 0.7))
            .collect::<Vec<_>>();
        let scores = last(memories(
            &artifact,
            "test.qwen35-scores",
            vec![
                upload(&rt, &query)?.into(),
                upload(&rt, &cache)?.into(),
                (keys as u64).into(),
                0u64.into(),
            ],
        )?);
        let gold = (0..2)
            .flat_map(|h| {
                let query = &query;
                let cache = &cache;
                (0..keys).map(move |k| {
                    rounded(
                        query[h * 16..h * 16 + 16]
                            .iter()
                            .zip(&cache[k * 16..k * 16 + 16])
                            .map(|(&q, &k)| q * k)
                            .sum::<f32>()
                            / 4.0,
                    )
                })
            })
            .collect::<Vec<_>>();
        close(&scores.try_to_f32_vec()?, &gold, 0.001);
        let probabilities = last(memories(
            &artifact,
            "test.qwen35-softmax",
            vec![scores.into(), (keys as u64).into()],
        )?);
        let pgold = gold
            .chunks(keys)
            .flat_map(|row| {
                let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let numerators = row.iter().map(|&v| (v - maximum).exp()).collect::<Vec<_>>();
                let mut terms = numerators.clone();
                let mut stride = 1;
                while stride < keys {
                    for left in (0..keys).step_by(2 * stride) {
                        if left + stride < keys {
                            terms[left] += terms[left + stride];
                        }
                    }
                    stride *= 2;
                }
                numerators.into_iter().map(move |v| rounded(v / terms[0]))
            })
            .collect::<Vec<_>>();
        close(&values(&probabilities)?, &pgold, 0.01);
        let actual_prob = values(&probabilities)?;
        let result = last(memories(
            &artifact,
            "test.qwen35-context",
            vec![
                probabilities.into(),
                upload(&rt, &cache)?.into(),
                (keys as u64).into(),
                0u64.into(),
            ],
        )?);
        let gold = (0..32)
            .map(|out| {
                let head = out / 16;
                let channel = out % 16;
                let mut parts = [0.0; 8];
                for lane in 0..8 {
                    for k in (lane..keys).step_by(8) {
                        parts[lane] += actual_prob[head * keys + k] * cache[k * 16 + channel];
                    }
                }
                for i in 0..4 {
                    parts[i] += parts[i + 4];
                }
                parts[0] += parts[2];
                parts[1] += parts[3];
                rounded(parts[0] + parts[1])
            })
            .collect::<Vec<_>>();
        close(&values(&result)?, &gold, 0.001);
    }
    let dt = [0.1f32, -0.2];
    let decay_weights = [-0.7f32, -0.9];
    let d = rt.mem_f32(&dt)?;
    let a = rt.mem_f32(&decay_weights)?;
    let mut state = (0..512)
        .map(|i| ((i * 17 % 23) as f32 - 11.0) / 137.0)
        .collect::<Vec<_>>();
    let mut shadow = rt.mem_f32(&state)?;
    for step in 0..3 {
        let q = (0..16)
            .map(|i| rounded(((i * 7 + step) % 19) as f32 / 41.0 - 0.2))
            .collect::<Vec<_>>();
        let k = (0..16)
            .map(|i| rounded(((i * 3 + step) % 17) as f32 / 39.0 - 0.15))
            .collect::<Vec<_>>();
        let v = (0..32)
            .map(|i| rounded(((i * 5 + step) % 29) as f32 / 53.0 - 0.25))
            .collect::<Vec<_>>();
        let alpha = [
            rounded(-0.2 + step as f32 / 10.0),
            rounded(0.3 - step as f32 / 11.0),
        ];
        let beta = [rounded(0.4), rounded(-0.3)];
        let mut conv = vec![0.0; 32];
        conv.extend_from_slice(&v);
        let mut expected = vec![0.0; 32];
        for col in 0..32 {
            let head = col / 16;
            let decay =
                rounded((1.0 + (alpha[head] + dt[head]).exp()).ln() * decay_weights[head]).exp();
            let rate = rounded(1.0 / (1.0 + (-beta[head]).exp()));
            let start = col * 16;
            let prediction = state[start..start + 16]
                .iter()
                .zip(&k)
                .map(|(&s, &k)| s * k)
                .sum::<f32>();
            let delta = (v[col] - decay * prediction) * rate;
            let mut output = 0.0;
            for channel in 0..16 {
                state[start + channel] = decay * state[start + channel] + k[channel] * delta;
                output += state[start + channel] * q[channel];
            }
            expected[col] = rounded(output / 4.0);
        }
        let mut out = memories(
            &artifact,
            "test.delta",
            vec![
                upload(&rt, &conv)?.into(),
                upload(&rt, &q)?.into(),
                upload(&rt, &k)?.into(),
                upload(&rt, &alpha)?.into(),
                upload(&rt, &beta)?.into(),
                d.as_ref().into(),
                a.as_ref().into(),
                shadow.into(),
            ],
        )?;
        let result = out.pop().unwrap();
        shadow = out.pop().unwrap();
        close(&values(&result)?, &expected, 0.01);
        close(&shadow.try_to_f32_vec()?, &state, 0.002);
    }
    assert!(
        shadow.try_to_f32_vec()?.iter().any(|&v| v != rounded(v)),
        "active recurrent state must retain F32 precision"
    );
    Ok(())
}
