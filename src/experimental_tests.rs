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
    let Some(selector) = std::env::var_os("CATENA_QWEN3_EXPERIMENTAL_MODEL") else {
        return Ok(());
    };
    let path = crate::hfhub::download_gguf(&selector.to_string_lossy())?.primary;
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
    qwen35_checkpoint_parity("CATENA_QWEN35_EXPERIMENTAL_MODEL")
}
#[test]
fn experimental_qwen35_moe_checkpoint_parity_on_gpu() -> Result<()> {
    qwen35_checkpoint_parity("CATENA_QWEN36_EXPERIMENTAL_MODEL")
}
fn qwen35_checkpoint_parity(variable: &str) -> Result<()> {
    let Some(selector) = std::env::var_os(variable) else {
        return Ok(());
    };
    let path = crate::hfhub::download_gguf(&selector.to_string_lossy())?.primary;
    use crate::{model::Model, tokenizer::QwenTokenizer};
    let tokenizer = QwenTokenizer::from_gguf(&path)?
        .ok_or_else(|| anyhow::anyhow!("missing GGUF tokenizer"))?;
    let argmax = |xs: &[f32]| {
        xs.iter()
            .enumerate()
            .max_by(|(ia, a), (ib, b)| a.total_cmp(b).then_with(|| ib.cmp(ia)))
            .unwrap()
            .0 as u64
    };
    // Register one checkpoint at a time: two simultaneous 65-GiB HIP host
    // registrations exceed the physical-memory pinning limit on this machine.
    let mut prefills = Vec::new();
    {
        let reference = Model::load_gguf_with_precision(&path, InferencePrecision::Bf16)?;
        for prompt in [
            "Gravity is".to_owned(),
            "The quick brown fox jumps over the lazy dog. ".repeat(12),
        ] {
            let tokens = tokenizer.encode(&prompt)?;
            let mut session = reference.session(tokens.len() + 20)?;
            let logits = session.prefill_logits(&tokens)?;
            prefills.push((tokens, session.cache_values_for_test()?, logits));
        }
    }
    let mut golden = Vec::new();
    {
        // Both compilers decode from the same validated staged-prefill cache.
        // The default fused SIMT vector kernel is the appropriate reference
        // for online decode; its tiled multi-token prefill has a different
        // accumulation order and is not part of this decode-only optimization.
        let reference = Model::load_gguf_with_options(
            &path,
            InferenceOptions::new(InferencePrecision::Bf16)
                .with_fused_attention(variable == "CATENA_QWEN36_EXPERIMENTAL_MODEL"),
        )?;
        for (tokens, cache, mut logits) in prefills {
            let mut session =
                reference.session_from_cache_for_test(tokens.len() + 20, tokens.len(), &cache)?;
            let mut steps = Vec::new();
            for _ in 0..20 {
                let token = argmax(&logits);
                steps.push((logits, token));
                logits = session.decode_logits(token)?;
            }
            golden.push((tokens, cache, steps));
        }
    }
    let experimental = Model::load_gguf_with_options(
        &path,
        InferenceOptions::new(InferencePrecision::Bf16).with_codegen(CodegenKind::Experimental),
    )?;
    for (tokens, expected_cache, steps) in golden {
        let capacity = tokens.len() + 20;
        let mut actual = experimental.session(capacity)?;
        let mut logits = actual.prefill_logits(&tokens)?;
        let actual_cache = actual.cache_values_for_test()?;
        assert_eq!(expected_cache.len(), actual_cache.len());
        assert_eq!(expected_cache, actual_cache, "prefill cache");
        let greedy = steps.iter().map(|(_, token)| *token).collect::<Vec<_>>();
        for (step, (expected, token)) in steps.into_iter().enumerate() {
            let error = logits
                .iter()
                .zip(&expected)
                .map(|(&a, &e)| f64::from(a - e).powi(2))
                .sum::<f64>();
            let energy = expected.iter().map(|&e| f64::from(e).powi(2)).sum::<f64>();
            assert!(
                (error / energy.max(1e-30)).sqrt() <= 0.05,
                "Qwen3.5 logit RMS for {} prompt tokens at step {step}: {}",
                tokens.len(),
                (error / energy).sqrt()
            );
            assert_eq!(
                argmax(&logits),
                token,
                "Qwen3.5 greedy token after {} prompt tokens at step {step}",
                tokens.len()
            );
            logits = actual.decode_logits(token)?;
        }
        assert_eq!(actual.position(), capacity);
        let mut independent = experimental.session(capacity)?;
        let mut token = independent.prefill_next_token(&tokens)?;
        assert_eq!(token, greedy[0], "device argmax after prefill");
        for &expected in &greedy[1..] {
            token = independent.decode_next_token(token)?;
            assert_eq!(token, expected, "device argmax after decode");
        }
        eprintln!(
            "passed prefill cache comparison and 20 greedy tokens for {} prompt tokens",
            tokens.len()
        );
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
(def program test.qwen35-normalize : {((cap.own mem) val) ((cap.ref mem) val) (u64 val) (f32 val) (u64 val) (u64 val) (u64 val)} -> {((cap.own mem) val) ((cap.own mem) val)} =
 ([input weight columns cf rows stages padded.] ({[.input] [.weight] [.columns] [.cf] [.rows] [.stages] [.padded]} qwen35.normalize [output.]) {[.input] [.output]}))
"#;
    let fused_wrapper = wrapper
        .split("(def program test.qwen35-conv")
        .next()
        .unwrap()
        .replace("test.delta", "test.fused-delta")
        .replace("qwen35.delta [", "qwen35.fused-delta [");
    let module = compile(
        crate::catena::runtime_sources_for_options(&source, options)
            .into_iter()
            .chain([wrapper, fused_wrapper.as_str()]),
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
    for width in [16usize, 32, 48, 192, 1024, 2560] {
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
                (width.next_power_of_two().trailing_zeros() as u64).into(),
                (width.next_power_of_two() as u64).into(),
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
        let mut fused = memories(
            &artifact,
            "test.fused-delta",
            vec![
                upload(&rt, &conv)?.into(),
                upload(&rt, &q)?.into(),
                upload(&rt, &k)?.into(),
                upload(&rt, &alpha)?.into(),
                upload(&rt, &beta)?.into(),
                d.as_ref().into(),
                a.as_ref().into(),
                rt.mem_f32(&shadow.try_to_f32_vec()?)?.into(),
            ],
        )?;
        let fused_output = values(&fused.pop().unwrap())?;
        let fused_state = fused.pop().unwrap().try_to_f32_vec()?;
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
        assert_eq!(values(&result)?, fused_output, "fused DeltaNet output");
        assert_eq!(
            shadow.try_to_f32_vec()?,
            fused_state,
            "fused DeltaNet F32 state"
        );
        close(&values(&result)?, &expected, 0.01);
        close(&shadow.try_to_f32_vec()?, &state, 0.002);
    }
    assert!(
        shadow.try_to_f32_vec()?.iter().any(|&v| v != rounded(v)),
        "active recurrent state must retain F32 precision"
    );
    Ok(())
}

fn tiny_moe_config() -> ModelConfig {
    let mut c = ModelConfig::qwen36_35b_a3b();
    c.hidden_size = 32;
    c.intermediate_size = 16;
    c.shared_expert_intermediate_size = Some(16);
    c.num_experts = 256;
    c.experts_per_token = 8;
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
    c
}

const MOE_ROUTING_REFERENCE: &str = r#"
(def program test.moe.route : {(cap.own mem)} -> {(cap.own mem) (cap.own mem) (cap.own mem)} =
 ([router.] ([.router] qwen3.router-select-experts [router-after ids.])
  ({[.router-after] [.ids]} qwen3.routing-weights [probabilities.])
  {[.router-after] [.ids] [.probabilities]}))
"#;
const MOE_WRAPPERS: &str = r#"
(def program test.moe.router : {((cap.own mem) val) ((cap.ref mem) val) ((cap.ref mem) val) ((cap.own mem) val)} -> {((cap.own mem) val) ((cap.own mem) val)} =
 ([input router shared scratch.] ({[.input] [.router] [.shared] [.scratch]} qwen35.moe.router [result.]) {[.input] [.result]}))
(def program test.moe.select : {((cap.own mem) val)} -> ((cap.own mem) val) =
 ([scratch.] ({[.scratch]} qwen35.moe.select [result.]) [.result]))
(def program test.moe.ffn : {((cap.own mem) val) ((cap.own mem) val) ((cap.own mem) val) ((cap.ref mem) val) ((cap.ref mem) val) ((cap.ref mem) val) ((cap.ref mem) val) ((cap.ref mem) val) ((cap.ref mem) val) ((cap.ref mem) val) ((cap.ref mem) val)} -> {((cap.own mem) val) ((cap.own mem) val)} =
 ([input residual scratch router gate up down shared-router shared-gate shared-up shared-down.]
  ({[.input] [.residual] [.scratch] [.router] [.gate] [.up] [.down] [.shared-router] [.shared-gate] [.shared-up] [.shared-down]} qwen35.moe.ffn [result.])
  ([.result] *.elim [workspace output.]) {[.workspace] [.output]}))
"#;

#[test]
fn experimental_qwen35_moe_fused_ffn_on_gpu() -> Result<()> {
    if std::env::var_os("CATENA_QWEN3_EXPERIMENTAL_GPU_TESTS").is_none() {
        return Ok(());
    }
    for (experts, k) in [(256, 8), (4, 3)] {
        let mut c = tiny_moe_config();
        c.num_experts = experts;
        c.experts_per_token = k;
        let options =
            InferenceOptions::new(InferencePrecision::Bf16).with_codegen(CodegenKind::Experimental);
        let model = source(&c, options)?;
        let module = compile(
            crate::catena::runtime_sources_for_options(&model, options)
                .into_iter()
                .chain([MOE_WRAPPERS]),
            GpuDialect::Hip,
        )?;
        let mut rt = Runtime::new(GpuDialect::Hip)?;
        let artifact = rt.load(module)?;
        let default_options = InferenceOptions::new(InferencePrecision::Bf16);
        let default_source =
            crate::catena::generated_model_source_for_options(&c, default_options)?;
        let (default_rt, default_artifact) = crate::catena::load_runtime(
            crate::catena::runtime_sources_for_options(&default_source, default_options)
                .into_iter()
                .chain([MOE_ROUTING_REFERENCE]),
            GpuDialect::Hip,
        )?;
        let shadow = c.num_layers
            * c.linear_time_step_rank.unwrap()
            * c.linear_state_size.unwrap()
            * c.linear_value_head_size().unwrap()
            * 4;
        let layout = qwen35::moe_workspace(&c, shadow)?;
        let router_offset = (shadow + 7) & !7;
        let ids_offset = (router_offset + (experts + 1) * 4 + 7) & !7;
        let probs_offset = ids_offset + k * 8;
        let scratch = || rt.mem_f32(&vec![0.125; layout.bytes / 4]);
        // Ties, signed zero, negative values, and the highest expert ID.
        let mut logits = vec![0.0f32; layout.bytes / 4];
        for i in 0..experts {
            logits[router_offset / 4 + i] = if i % 2 == 0 { 0.0 } else { -0.0 };
        }
        logits[router_offset / 4 + experts - 1] = 2.0;
        let selected = last(memories(
            &artifact,
            "test.moe.select",
            vec![rt.mem_f32(&logits)?.into()],
        )?);
        let ids = selected.try_to_u64_vec()?;
        let actual = &ids[ids_offset / 8..ids_offset / 8 + k];
        let mut expected = vec![(experts - 1) as u64];
        expected.extend((0..k - 1).map(|i| i as u64));
        assert_eq!(actual, expected);
        let probabilities = selected.try_to_f32_vec()?;
        let target = std::iter::once(2.0f32.exp())
            .chain(std::iter::repeat_n(1.0, k - 1))
            .collect::<Vec<_>>();
        let denominator: f32 = target.iter().sum();
        close(
            &probabilities[probs_offset / 4..probs_offset / 4 + k],
            &target.iter().map(|p| p / denominator).collect::<Vec<_>>(),
            0.0001,
        );
        // Uneven selected probabilities expose left-to-right versus tree-sum
        // differences that can cross a downstream BF16 rounding boundary.
        let mut adversarial = vec![-20.0f32; experts];
        for (i, value) in [0.0, -0.013, -0.19, -0.72, -1.41, -2.3, -4.7, -8.2]
            .into_iter()
            .take(k)
            .enumerate()
        {
            adversarial[experts - 1 - i] = value;
        }
        let mut scratch_logits = vec![0.0; layout.bytes / 4];
        scratch_logits[router_offset / 4..router_offset / 4 + experts]
            .copy_from_slice(&adversarial);
        let selected = last(memories(
            &artifact,
            "test.moe.select",
            vec![rt.mem_f32(&scratch_logits)?.into()],
        )?);
        let mut reference = memories(
            &default_artifact,
            "test.moe.route",
            vec![default_rt.mem_f32(&adversarial)?.into()],
        )?;
        let expected_probabilities = reference.pop().unwrap().try_to_f32_vec()?;
        let expected_ids = reference.pop().unwrap().try_to_u64_vec()?;
        assert_eq!(
            &selected.try_to_u64_vec()?[ids_offset / 8..ids_offset / 8 + k],
            expected_ids
        );
        assert_eq!(
            &selected.try_to_f32_vec()?[probs_offset / 4..probs_offset / 4 + k],
            expected_probabilities,
            "routing accumulation order"
        );
        let h = c.hidden_size;
        let inter = c.intermediate_size;
        let shared = c.shared_expert_intermediate_size.unwrap();
        let input = (0..h)
            .map(|i| rounded((i as f32 - 15.0) / 16.0))
            .collect::<Vec<_>>();
        let residual = (0..h)
            .map(|i| rounded((i as f32 - 16.0) / 64.0))
            .collect::<Vec<_>>();
        let weights = |len: usize, seed: usize| {
            (0..len)
                .map(|i| rounded(((i * seed + i / 17) % 23) as f32 / 32.0 - 0.34375))
                .collect::<Vec<_>>()
        };
        let router = weights(experts * h, 7);
        let sr = weights(h, 11);
        let gate = weights(experts * inter * h, 3);
        let up = weights(experts * inter * h, 5);
        let down = weights(experts * h * inter, 7);
        let sg = weights(shared * h, 9);
        let su = weights(shared * h, 13);
        let sd = weights(h * shared, 17);
        let owned = [
            rt.mem_f32(&router)?,
            upload(&rt, &gate)?,
            upload(&rt, &up)?,
            upload(&rt, &down)?,
            rt.mem_f32(&sr)?,
            upload(&rt, &sg)?,
            upload(&rt, &su)?,
            upload(&rt, &sd)?,
        ];
        let mut args = vec![
            upload(&rt, &input)?.into(),
            upload(&rt, &residual)?.into(),
            scratch()?.into(),
        ];
        args.extend(owned.iter().map(|w| w.as_ref().into()));
        let mut results = memories(&artifact, "test.moe.ffn", args)?;
        let output = values(&results.pop().unwrap())?;
        let work = results.pop().unwrap();
        assert!(
            work.try_to_f32_vec()?[..shadow / 4]
                .iter()
                .all(|&x| x == 0.125),
            "FFN overwrote recurrent shadow"
        );
        let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(a, b)| a * b).sum::<f32>();
        let scores = router
            .chunks(h)
            .map(|row| dot(&input, row))
            .collect::<Vec<_>>();
        close(
            &work.try_to_f32_vec()?[router_offset / 4..router_offset / 4 + experts],
            &scores,
            0.00001,
        );
        let mut order = (0..experts).collect::<Vec<_>>();
        order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
        order.truncate(k);
        assert_eq!(
            &work.try_to_u64_vec()?[ids_offset / 8..ids_offset / 8 + k],
            &order.iter().map(|&i| i as u64).collect::<Vec<_>>()
        );
        let exp = order
            .iter()
            .map(|&e| (scores[e] - scores[order[0]]).exp())
            .collect::<Vec<_>>();
        let sum: f32 = exp.iter().sum();
        let silu = |x: f32| x / (1.0 + (-x).exp());
        let mut active = Vec::new();
        for (slot, &expert) in order.iter().enumerate() {
            for j in 0..inter {
                let base = (expert * inter + j) * h;
                let g = rounded(dot(&input, &gate[base..base + h]));
                let u = rounded(dot(&input, &up[base..base + h]));
                active.push(rounded(silu(g) * u * (exp[slot] / sum)));
            }
        }
        let shared_scale = 1.0 / (1.0 + (-dot(&input, &sr)).exp());
        let shared_active = (0..shared)
            .map(|j| {
                let g = rounded(dot(&input, &sg[j * h..(j + 1) * h]));
                let u = rounded(dot(&input, &su[j * h..(j + 1) * h]));
                rounded(rounded(rounded(silu(g)) * u) * shared_scale)
            })
            .collect::<Vec<_>>();
        let active_offset = (probs_offset + k * 4 + 7) & !7;
        let mut expected_active = active.clone();
        expected_active.extend(&shared_active);
        close(
            &values(&work)?[active_offset / 2..active_offset / 2 + expected_active.len()],
            &expected_active,
            0.005,
        );
        let target = (0..h)
            .map(|j| {
                let routed = order
                    .iter()
                    .enumerate()
                    .map(|(slot, &e)| {
                        dot(
                            &active[slot * inter..(slot + 1) * inter],
                            &down[(e * h + j) * inter..(e * h + j + 1) * inter],
                        )
                    })
                    .sum::<f32>();
                rounded(
                    rounded(routed + residual[j])
                        + rounded(dot(&shared_active, &sd[j * shared..(j + 1) * shared])),
                )
            })
            .collect::<Vec<_>>();
        close(&output, &target, 0.01);
    }
    Ok(())
}

#[test]
fn experimental_qwen35_moe_hybrid_sessions_on_gpu() -> Result<()> {
    if std::env::var_os("CATENA_QWEN3_EXPERIMENTAL_GPU_TESTS").is_none() {
        return Ok(());
    }
    use crate::spec::WeightDtype;
    let mut c = tiny_moe_config();
    c.num_experts = 4;
    c.experts_per_token = 2;
    c.mrope_sections = [2, 1, 1];
    let specs = weight_specs(&c)?;
    let make = |codegen| -> Result<(Runtime, Artifact, Vec<MemOwn>)> {
        let options = InferenceOptions::new(InferencePrecision::Bf16)
            .with_codegen(codegen)
            .with_fused_attention(false);
        let source = crate::catena::generated_multimodal_source(&c, options)?;
        let (rt, artifact) = crate::catena::load_runtime_with_codegen(
            crate::catena::runtime_sources_for_options(&source, options),
            GpuDialect::Hip,
            codegen,
        )?;
        let mut weights = Vec::new();
        for (wi, spec) in specs.iter().enumerate() {
            let xs = (0..spec.element_count())
                .map(|i| {
                    if spec.name.contains("norm") {
                        1.0
                    } else if spec.name.contains("A_log") {
                        -0.5
                    } else {
                        rounded((((i * 7 + wi * 11) % 31) as f32 - 15.0) / 128.0)
                    }
                })
                .collect::<Vec<_>>();
            weights.push(match spec.dtype {
                WeightDtype::F32 => rt.mem_f32(&xs)?,
                WeightDtype::Bf16 => upload(&rt, &xs)?,
            });
        }
        Ok((rt, artifact, weights))
    };
    let (reference_rt, reference, rw) = make(CodegenKind::Default)?;
    let (rt, experimental, ew) = make(CodegenKind::Experimental)?;
    let capacity = 16usize;
    let workspace_len = workspace_bytes(
        &c,
        InferenceOptions::new(InferencePrecision::Bf16).with_codegen(CodegenKind::Experimental),
    )? / 4;
    let invoke = |runtime: &Runtime,
                  artifact: &Artifact,
                  weights: &[MemOwn],
                  tokens: &[u64],
                  position: usize,
                  cache: MemOwn,
                  scratch: Option<MemOwn>|
     -> Result<(MemOwn, Option<MemOwn>, Vec<f32>)> {
        let token_mem = runtime.mem_u64(tokens)?;
        let mut args = vec![token_mem.as_ref().into(), cache.into()];
        if let Some(scratch) = scratch {
            args.push(scratch.into());
        }
        args.extend([(capacity as u64).into(), (position as u64).into()]);
        args.extend(weights.iter().map(|w| w.as_ref().into()));
        let mut outputs = memories(artifact, "qwen3", args)?;
        let workspace = if outputs.len() == 4 {
            Some(outputs.remove(1))
        } else {
            None
        };
        let cache = outputs.remove(0);
        let logits = values(&outputs.remove(0))?;
        Ok((cache, workspace, logits))
    };
    for prompt in [vec![1u64, 2], vec![1, 2, 3, 4, 5, 6, 7]] {
        let cache = || rt.mem_u16(&vec![0; c.cache_elements(capacity).unwrap()]);
        let (mut ec, mut scratch, mut actual) = invoke(
            &rt,
            &experimental,
            &ew,
            &prompt,
            0,
            cache()?,
            Some(rt.mem_f32(&vec![0.0; workspace_len])?),
        )?;
        let (mut rc, _, mut expected) = invoke(
            &reference_rt,
            &reference,
            &rw,
            &prompt,
            0,
            reference_rt.mem_u16(&vec![0; c.cache_elements(capacity)?])?,
            None,
        )?;
        let snapshot = ec.try_to_u16_vec()?;
        assert_eq!(snapshot, rc.try_to_u16_vec()?, "hybrid cache after prefill");
        // A new session must reproduce prefill after another session has run.
        let (independent, _, repeated) = invoke(
            &rt,
            &experimental,
            &ew,
            &prompt,
            0,
            cache()?,
            Some(rt.mem_f32(&vec![0.0; workspace_len])?),
        )?;
        assert_eq!(independent.try_to_u16_vec()?, snapshot);
        assert_eq!(repeated, actual);
        for position in prompt.len()..capacity {
            close(&actual, &expected, 0.01);
            let argmax = |xs: &[f32]| {
                xs.iter()
                    .enumerate()
                    .max_by(|(ia, a), (ib, b)| a.total_cmp(b).then_with(|| ib.cmp(ia)))
                    .unwrap()
                    .0 as u64
            };
            let token = argmax(&expected);
            assert_eq!(argmax(&actual), token);
            (ec, scratch, actual) =
                invoke(&rt, &experimental, &ew, &[token], position, ec, scratch)?;
            (rc, _, expected) =
                invoke(&reference_rt, &reference, &rw, &[token], position, rc, None)?;
            assert_eq!(
                ec.try_to_u16_vec()?,
                rc.try_to_u16_vec()?,
                "hybrid cache after decode {position}"
            );
        }
    }

    // Image rows and token rows use the same public entrypoints as Session.
    let image_values = (0..4 * c.hidden_size)
        .map(|i| ((i * 17 % 59) as f32 - 29.0) / 87.0)
        .collect::<Vec<_>>();
    let image = rt.mem_f32(&image_values)?;
    let reference_image = reference_rt.mem_f32(&image_values)?;
    for row in 0..4u64 {
        let actual = last(memories(
            &experimental,
            "qwen3.mm-image-row",
            vec![image.as_ref().into(), row.into()],
        )?);
        let expected = last(memories(
            &reference,
            "qwen3.mm-image-row",
            vec![reference_image.as_ref().into(), row.into()],
        )?);
        assert_eq!(
            actual.try_to_u16_vec()?,
            expected.try_to_u16_vec()?,
            "image row {row}"
        );
    }
    for coordinates in [[3u64, 8, 5], [5, 5, 5]] {
        let positions = rt.mem_u64(&coordinates)?;
        let reference_positions = reference_rt.mem_u64(&coordinates)?;
        let actual = last(memories(
            &experimental,
            "qwen3.mm-rope-table",
            vec![positions.as_ref().into()],
        )?);
        let expected = last(memories(
            &reference,
            "qwen3.mm-rope-table",
            vec![reference_positions.as_ref().into()],
        )?);
        close(&actual.try_to_f32_vec()?, &expected.try_to_f32_vec()?, 1e-6);
    }
    for _ in 0..2 {
        let mut ec = rt.mem_u16(&vec![0; c.cache_elements(capacity)?])?;
        let mut rc = reference_rt.mem_u16(&vec![0; c.cache_elements(capacity)?])?;
        let mut scratch = rt.mem_f32(&vec![0.0; workspace_len])?;
        for position in 0..capacity {
            let coordinates = match position {
                2..=5 => [
                    2,
                    2 + (position as u64 - 2) / 2,
                    2 + (position as u64 - 2) % 2,
                ],
                6.. => [position as u64 - 2; 3],
                _ => [position as u64; 3],
            };
            let invoke_mm = |runtime: &Runtime,
                             artifact: &Artifact,
                             weights: &[MemOwn],
                             image: &MemOwn,
                             cache: MemOwn,
                             workspace: Option<MemOwn>|
             -> Result<Vec<MemOwn>> {
                let hidden = if (2..=5).contains(&position) {
                    last(memories(
                        artifact,
                        "qwen3.mm-image-row",
                        vec![image.as_ref().into(), ((position - 2) as u64).into()],
                    )?)
                } else {
                    let tokens = runtime.mem_u64(&[1 + (position as u64 % 7)])?;
                    last(memories(
                        artifact,
                        "qwen3.embedding",
                        vec![tokens.as_ref().into(), weights[0].as_ref().into()],
                    )?)
                };
                let positions = runtime.mem_u64(&coordinates)?;
                let experimental = workspace.is_some();
                let mut inputs = vec![hidden.into(), positions.as_ref().into(), cache.into()];
                if let Some(workspace) = workspace {
                    inputs.push(workspace.into());
                }
                inputs.extend([(capacity as u64).into(), (position as u64).into()]);
                if experimental {
                    inputs.push((if position < 8 { 8u64 } else { 1u64 }).into());
                }
                inputs.extend(weights.iter().map(|w| w.as_ref().into()));
                memories(artifact, "qwen3.multimodal", inputs)
            };
            let mut actual = invoke_mm(&rt, &experimental, &ew, &image, ec, Some(scratch))?;
            let mut expected =
                invoke_mm(&reference_rt, &reference, &rw, &reference_image, rc, None)?;
            ec = actual.remove(0);
            scratch = actual.remove(0);
            rc = expected.remove(0);
            assert_eq!(
                ec.try_to_u16_vec()?,
                rc.try_to_u16_vec()?,
                "image cache at {position}"
            );
            let actual_logits = values(&actual.remove(0))?;
            let expected_logits = values(&expected.remove(0))?;
            close(&actual_logits, &expected_logits, 0.01);
            let first_max = |xs: &[f32]| {
                xs.iter()
                    .enumerate()
                    .max_by(|(ia, a), (ib, b)| a.total_cmp(b).then_with(|| ib.cmp(ia)))
                    .unwrap()
                    .0 as u64
            };
            let last_max = |xs: &[f32]| {
                xs.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.total_cmp(b))
                    .unwrap()
                    .0 as u64
            };
            // Synthetic LM rows 0 and 31 are identical. Preserve the existing
            // first-token experimental and last-token default tie policies.
            assert_eq!(
                first_max(&actual_logits),
                first_max(&expected_logits),
                "image greedy choice at {position}"
            );
            assert_eq!(
                actual.remove(0).try_to_u64_vec()?,
                vec![first_max(&actual_logits)]
            );
            assert_eq!(
                expected.remove(0).try_to_u64_vec()?,
                vec![last_max(&expected_logits)]
            );
        }
    }
    Ok(())
}

#[test]
fn experimental_qwen35_online_and_cooperative_norm_on_gpu() -> Result<()> {
    if std::env::var_os("CATENA_QWEN3_EXPERIMENTAL_GPU_TESTS").is_none() {
        return Ok(());
    }
    let mut c = tiny_moe_config();
    c.architecture = ModelArchitecture::Qwen35Dense;
    c.num_experts = 0;
    c.experts_per_token = 0;
    c.shared_expert_intermediate_size = None;
    let options =
        InferenceOptions::new(InferencePrecision::Bf16).with_codegen(CodegenKind::Experimental);
    let source = source(&c, options)?;
    let wrappers = r#"
(def program test.online : {((cap.own mem) val) ((cap.own mem) val) ((cap.own mem) val) (u64 val)} -> {((cap.own mem) val) ((cap.own mem) val) ((cap.own mem) val) ((cap.own mem) val)} =
 ([q cache qg keys.]
  ({[.keys] (const.u64.0x0000000000000010)} stdlib.qwen3.memory.u64.mul [vb.])
  ({[.q] [.cache] [.qg] [.keys] (const.u64.0x0000000000000000) [.vb]} qwen35.online [output.])
  {[.q] [.cache] [.qg] [.output]}))
(def program test.coop-l2 : {((cap.own mem) val)} -> {((cap.own mem) val) ((cap.own mem) val) ((cap.own mem) val)} =
 ([input.] ({[.input] (const.u64.0x0000000000000000) (const.u64.0x0000000000000010)} qwen35.l2 [legacy.])
  ({[.input] (const.u64.0x0000000000000000) (const.u64.0x0000000000000010)} qwen35.coop-l2 [cooperative.]) {[.input] [.legacy] [.cooperative]}))
(def program test.coop-gated : {((cap.own mem) val) ((cap.own mem) val) ((cap.ref mem) val)} -> {((cap.own mem) val) ((cap.own mem) val) ((cap.own mem) val) ((cap.own mem) val)} =
 ([input z w.] ({[.input] [.z] [.w]} qwen35.gated-norm [legacy.]) ({[.input] [.z] [.w]} qwen35.coop-gated [cooperative.]) {[.input] [.z] [.legacy] [.cooperative]}))
"#;
    let module = compile(
        crate::catena::runtime_sources_for_options(&source, options)
            .into_iter()
            .chain([wrappers]),
        GpuDialect::Hip,
    )?;
    let mut rt = Runtime::new(GpuDialect::Hip)?;
    let a = rt.load(module)?;
    let q = (0..32)
        .map(|i| rounded(((i * 7 % 19) as f32 - 9.0) / 16.0))
        .collect::<Vec<_>>();
    let qg = (0..64)
        .map(|i| rounded(((i * 11 % 29) as f32 - 14.0) / 8.0))
        .collect::<Vec<_>>();
    for keys in [1usize, 3, 8, 9, 33, 257, 4097] {
        let cache = (0..keys * 32)
            .map(|i| rounded(((i * 13 % 31) as f32 - 15.0) / 16.0))
            .collect::<Vec<_>>();
        let output = last(memories(
            &a,
            "test.online",
            vec![
                upload(&rt, &q)?.into(),
                upload(&rt, &cache)?.into(),
                upload(&rt, &qg)?.into(),
                (keys as u64).into(),
            ],
        )?);
        let expected = (0..32)
            .map(|ix| {
                let head = ix / 16;
                let ch = ix % 16;
                let scores = (0..keys)
                    .map(|k| {
                        (0..16)
                            .map(|j| q[head * 16 + j] * cache[k * 16 + j])
                            .sum::<f32>()
                            / 4.0
                    })
                    .collect::<Vec<_>>();
                let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let weights = scores
                    .iter()
                    .map(|s| (s - maximum).exp())
                    .collect::<Vec<_>>();
                let den = weights.iter().sum::<f32>();
                let value = weights
                    .iter()
                    .enumerate()
                    .map(|(k, p)| p * cache[keys * 16 + k * 16 + ch])
                    .sum::<f32>()
                    / den;
                let gate = qg[head * 32 + 16 + ch];
                rounded(rounded(value) / (1.0 + (-gate).exp()))
            })
            .collect::<Vec<_>>();
        eprintln!("online attention history {keys}");
        close(&values(&output)?, &expected, 0.008);
    }
    let mut out = memories(&a, "test.coop-l2", vec![upload(&rt, &q)?.into()])?;
    let coop = values(&out.pop().unwrap())?;
    assert_eq!(coop, values(&out.pop().unwrap())?);
    let w = rt.mem_f32(&vec![0.75; 16])?;
    let mut out = memories(
        &a,
        "test.coop-gated",
        vec![
            upload(&rt, &q)?.into(),
            upload(&rt, &q)?.into(),
            w.as_ref().into(),
        ],
    )?;
    let coop = values(&out.pop().unwrap())?;
    assert_eq!(coop, values(&out.pop().unwrap())?);
    Ok(())
}
