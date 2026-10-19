//! Batch kernels are checked against the existing per-token kernels.
use super::*;

#[test]
fn qwen35_batch_kernels_on_gpu() -> Result<()> {
    if std::env::var_os("CATENA_QWEN3_GPU_TESTS").is_none() {
        return Ok(());
    }
    let mut c = tiny_moe_config();
    c.num_experts = 4;
    c.experts_per_token = 2;
    c.linear_state_size = Some(128);
    let options = InferenceOptions::default();
    let mut src = source(&c, options)?.replace(
        "(def program qwen3 :",
        "(def program stdlib.test.unused-model :",
    );
    let project = r#"
(def program test.batch-project : {((cap.own mem) val) ((cap.ref mem) val)} -> {((cap.own mem) val) ((cap.own mem) val)} =
 ([input weight.] ({[.input] [.weight] (const.u64.0x0000000000000020) (const.u64.0x000000000000000A)} stdlib.qwen35.batch.project [out.]) {[.input] [.out]}))
"#;
    src.push_str(project);
    src.push_str(
        &project
            .replace("test.batch-project", "test.row-project")
            .replace("stdlib.qwen35.batch.project", "qwen3.fused.project"),
    );
    let prep = r#"
(def program test.batch-prepare : {((cap.own mem) val) ((cap.own mem) val) ((cap.ref mem) val) (u64 val) (u64 val)} -> {((cap.own mem) val) ((cap.own mem) val) ((cap.own mem) val) ((cap.own mem) val) ((cap.own mem) val)} =
 ([input cache weight base start.]
  ({[.input] [.cache] [.weight] [.base] [.start]} stdlib.qwen35.batch.prepare [conv query key.])
  ([.conv] meta.master.complete-own [ready phase.])
  ({[.cache] [.phase]} meta.master.after-own [after-read.])
  ({[.input] [.after-read] [.base] [.start]} stdlib.qwen35.batch.history.run [updated.])
  {[.input] [.updated] [.ready] [.query] [.key]}))
"#;
    src.push_str(prep);
    let rowprep = prep.replace("test.batch-prepare", "test.row-prepare").replace("stdlib.qwen35.batch.prepare", "qwen35.linear-prepare")
        .replace("  ({[.input] [.after-read] [.base] [.start]} stdlib.qwen35.batch.history.run [updated.])", "  ([.after-read] [updated.])");
    src.push_str(&rowprep);
    let scan = r#"
(def program test.batch-scan : {((cap.own mem) val) ((cap.own mem) val) ((cap.own mem) val) ((cap.own mem) val) ((cap.own mem) val) ((cap.ref mem) val) ((cap.ref mem) val) ((cap.own mem) val) (u64 val)} -> {((cap.own mem) val) ((cap.own mem) val)} =
 ([conv query key alpha beta dt a shadow base.]
  ({[.conv] [.query] [.key] [.alpha] [.beta] [.dt] [.a] [.shadow] [.base]} stdlib.qwen35.batch.scan [out.]) {[.shadow] [.out]}))
"#;
    src.push_str(scan);
    src.push_str(
        &scan
            .replace("test.batch-scan", "test.row-scan")
            .replace("stdlib.qwen35.batch.scan", "qwen35.fused-delta"),
    );
    let gate = r#"
(def program test.batch-gated : {((cap.own mem) val) ((cap.own mem) val) ((cap.ref mem) val)} -> {((cap.own mem) val)} =
 ([input z weight.] ({[.input] [.z] [.weight]} stdlib.qwen35.batch.gated)))
"#;
    src.push_str(gate);
    src.push_str(
        &gate
            .replace("test.batch-gated", "test.row-gated")
            .replace("stdlib.qwen35.batch.gated", "qwen35.coop-gated"),
    );
    let module = compile(crate::catena::runtime_sources(&src), GpuDialect::Hip)?;
    let mut rt = Runtime::new(GpuDialect::Hip)?;
    let artifact = rt.load(module)?;
    let data = |n: usize, seed: usize| {
        (0..n)
            .map(|i| rounded(((i * 7 + seed * 13) % 59) as f32 / 151.0 - 0.15))
            .collect::<Vec<_>>()
    };
    let weights = upload(&rt, &data(32 * 10, 0))?;
    let conv_weights = rt.mem_f32(&data(288 * 4, 1))?;
    let dt = rt.mem_f32(&[0.1, -0.2])?;
    let a = rt.mem_f32(&[-0.7, -0.9])?;
    let norm = rt.mem_f32(&vec![1.0; 16])?;
    for tokens in [1usize, 3, 7, 33] {
        let input = data(tokens * 32, tokens);
        let mut expected = Vec::new();
        for row in input.chunks(32) {
            expected.extend(
                last(memories(
                    &artifact,
                    "test.row-project",
                    vec![upload(&rt, row)?.into(), weights.as_ref().into()],
                )?)
                .try_to_u16_vec()?,
            );
        }
        let actual = last(memories(
            &artifact,
            "test.batch-project",
            vec![upload(&rt, &input)?.into(), weights.as_ref().into()],
        )?);
        assert_eq!(
            actual.try_to_u16_vec()?,
            expected,
            "batch projection T={tokens}"
        );
        for start in [0usize, 5] {
            let qkv = data(tokens * 288, tokens + start);
            let base = 17usize;
            let cache_values = data(base + 288 * 3 + 19, 3);
            let mut cache = upload(&rt, &cache_values)?;
            let mut conv = Vec::new();
            let mut queries = Vec::new();
            let mut keys = Vec::new();
            for (t, row) in qkv.chunks(288).enumerate() {
                let mut out = memories(
                    &artifact,
                    "test.row-prepare",
                    vec![
                        upload(&rt, row)?.into(),
                        cache.into(),
                        conv_weights.as_ref().into(),
                        (base as u64).into(),
                        ((start + t) as u64).into(),
                    ],
                )?;
                keys.extend(values(&out.pop().unwrap())?);
                queries.extend(values(&out.pop().unwrap())?);
                conv.extend(values(&out.pop().unwrap())?);
                cache = out.pop().unwrap();
            }
            let mut out = memories(
                &artifact,
                "test.batch-prepare",
                vec![
                    upload(&rt, &qkv)?.into(),
                    upload(&rt, &cache_values)?.into(),
                    conv_weights.as_ref().into(),
                    (base as u64).into(),
                    (start as u64).into(),
                ],
            )?;
            assert_eq!(
                values(&out.pop().unwrap())?,
                keys,
                "batch K T={tokens} start={start}"
            );
            assert_eq!(
                values(&out.pop().unwrap())?,
                queries,
                "batch Q T={tokens} start={start}"
            );
            assert_eq!(
                values(&out.pop().unwrap())?,
                conv,
                "batch conv T={tokens} start={start}"
            );
            assert_eq!(
                out.pop().unwrap().try_to_u16_vec()?,
                cache.try_to_u16_vec()?,
                "batch history T={tokens} start={start}"
            );
            let state_values = data(base + 32 * 128 + 19, 5);
            let mut state = rt.mem_f32(&state_values)?;
            let alpha = data(tokens * 2, 6);
            let beta = data(tokens * 2, 7);
            let mut expected = Vec::new();
            for t in 0..tokens {
                let mut out = memories(
                    &artifact,
                    "test.row-scan",
                    vec![
                        upload(&rt, &conv[t * 288..(t + 1) * 288])?.into(),
                        upload(&rt, &queries[t * 128..(t + 1) * 128])?.into(),
                        upload(&rt, &keys[t * 128..(t + 1) * 128])?.into(),
                        upload(&rt, &alpha[t * 2..(t + 1) * 2])?.into(),
                        upload(&rt, &beta[t * 2..(t + 1) * 2])?.into(),
                        dt.as_ref().into(),
                        a.as_ref().into(),
                        state.into(),
                        (base as u64).into(),
                    ],
                )?;
                expected.extend(values(&out.pop().unwrap())?);
                state = out.pop().unwrap();
            }
            let mut out = memories(
                &artifact,
                "test.batch-scan",
                vec![
                    upload(&rt, &conv)?.into(),
                    upload(&rt, &queries)?.into(),
                    upload(&rt, &keys)?.into(),
                    upload(&rt, &alpha)?.into(),
                    upload(&rt, &beta)?.into(),
                    dt.as_ref().into(),
                    a.as_ref().into(),
                    rt.mem_f32(&state_values)?.into(),
                    (base as u64).into(),
                ],
            )?;
            assert_eq!(
                values(&out.pop().unwrap())?,
                expected,
                "scan outputs T={tokens} start={start}"
            );
            assert_eq!(
                out.pop().unwrap().try_to_f32_vec()?,
                state.try_to_f32_vec()?,
                "scan F32 state T={tokens} start={start}"
            );
            let z = data(tokens * 32, 8);
            let mut gated = Vec::new();
            for t in 0..tokens {
                gated.extend(values(&last(memories(
                    &artifact,
                    "test.row-gated",
                    vec![
                        upload(&rt, &expected[t * 32..(t + 1) * 32])?.into(),
                        upload(&rt, &z[t * 32..(t + 1) * 32])?.into(),
                        norm.as_ref().into(),
                    ],
                )?))?);
            }
            let out = last(memories(
                &artifact,
                "test.batch-gated",
                vec![
                    upload(&rt, &expected)?.into(),
                    upload(&rt, &z)?.into(),
                    norm.as_ref().into(),
                ],
            )?);
            assert_eq!(values(&out)?, gated, "batched gated norm T={tokens}");
        }
    }
    Ok(())
}

#[test]
fn qwen35_batch_graph_compiles() -> Result<()> {
    let mut c = tiny_moe_config();
    c.num_experts = 4;
    c.experts_per_token = 2;
    c.linear_state_size = Some(128);
    let options = InferenceOptions::default();
    let src = source(&c, options)?;
    let raw = metacat::theory::RawTheorySet::from_texts(crate::catena::runtime_sources(&src))?;
    let raw = crate::catena::prepare_sources(raw)?;

    let result = crate::catena::compile(raw, GpuDialect::Hip);
    let module = result?;
    assert_eq!(
        module
            .functions
            .iter()
            .find(|f| f.source_name == "qwen3")
            .unwrap()
            .outputs
            .len(),
        11
    );
    Ok(())
}

#[test]
fn qwen35_batch_moe_on_gpu() -> Result<()> {
    if std::env::var_os("CATENA_QWEN3_GPU_TESTS").is_none() {
        return Ok(());
    }
    let mut c = tiny_moe_config();
    c.num_experts = 4;
    c.experts_per_token = 3;
    let options = InferenceOptions::default();
    let mut src = source(&c, options)?.replace(
        "(def program qwen3 :",
        "(def program stdlib.test.unused-model :",
    );
    src.push_str(MOE_WRAPPERS);
    src.push_str(r#"
(def program test.batch-ffn : {((cap.own mem) val) ((cap.own mem) val) ((cap.ref mem) val) ((cap.ref mem) val) ((cap.ref mem) val) ((cap.ref mem) val) ((cap.ref mem) val) ((cap.ref mem) val) ((cap.ref mem) val) ((cap.ref mem) val)} -> ((cap.own mem) val) =
 ([input residual router gate up down shared-router shared-gate shared-up shared-down.]
 ({[.input] [.residual] [.router] [.gate] [.up] [.down] [.shared-router] [.shared-gate] [.shared-up] [.shared-down]} stdlib.qwen35.batch.moe.ffn)))
"#);
    let module = compile(crate::catena::runtime_sources(&src), GpuDialect::Hip)?;
    let mut rt = Runtime::new(GpuDialect::Hip)?;
    let artifact = rt.load(module)?;
    let data = |n: usize, seed: usize| {
        (0..n)
            .map(|i| rounded(((i * seed + i / 17) % 23) as f32 / 32.0 - 0.34375))
            .collect::<Vec<_>>()
    };
    let h = c.hidden_size;
    let inter = c.intermediate_size;
    let shared = c.shared_expert_intermediate_size.unwrap();
    let experts = c.num_experts;
    let owned = [
        rt.mem_f32(&data(experts * h, 7))?,
        upload(&rt, &data(experts * inter * h, 3))?,
        upload(&rt, &data(experts * inter * h, 5))?,
        upload(&rt, &data(experts * h * inter, 7))?,
        rt.mem_f32(&data(h, 11))?,
        upload(&rt, &data(shared * h, 9))?,
        upload(&rt, &data(shared * h, 13))?,
        upload(&rt, &data(h * shared, 17))?,
    ];
    let shadow = c.num_layers
        * c.linear_time_step_rank.unwrap()
        * c.linear_state_size.unwrap()
        * c.linear_value_head_size().unwrap()
        * 4;
    let router = (shadow + 7) & !7;
    let ids = (router + (experts + 1) * 4 + 7) & !7;
    let active = (ids + c.experts_per_token * 12 + 7) & !7;
    let workspace = (active + (c.experts_per_token * inter + shared) * 2 + 7) & !7;
    for tokens in [1, 3, 7, 33] {
        let input = data(tokens * h, 19);
        let residual = data(tokens * h, 23);
        let mut expected = Vec::new();
        for t in 0..tokens {
            let mut args = vec![
                upload(&rt, &input[t * h..(t + 1) * h])?.into(),
                upload(&rt, &residual[t * h..(t + 1) * h])?.into(),
                rt.mem_f32(&vec![0.125; workspace / 4])?.into(),
            ];
            args.extend(owned.iter().map(|w| w.as_ref().into()));
            expected.extend(last(memories(&artifact, "test.moe.ffn", args)?).try_to_u16_vec()?);
        }
        let mut args = vec![upload(&rt, &input)?.into(), upload(&rt, &residual)?.into()];
        args.extend(owned.iter().map(|w| w.as_ref().into()));
        let actual = last(memories(&artifact, "test.batch-ffn", args)?).try_to_u16_vec()?;
        assert_eq!(actual, expected, "batch MoE T={tokens}");
    }
    Ok(())
}

#[test]
fn qwen35_tiled_projection_on_gpu() -> Result<()> {
    if std::env::var_os("CATENA_QWEN3_GPU_TESTS").is_none() {
        return Ok(());
    }
    let c = tiny_moe_config();
    let options = InferenceOptions::default();
    let mut src = source(&c, options)?.replace(
        "(def program qwen3 :",
        "(def program stdlib.test.unused-model :",
    );
    src.push_str(
        &include_str!("../catena/qwen35_batch_projection.hex")
            .replace("stdlib.qwen35.batch.", "stdlib.qwen35.tile."),
    );
    let wrapper = r#"
(def program test.project-batch : {((cap.own mem) val) ((cap.ref mem) val) (u64 val) (u64 val)} -> {((cap.own mem) val) ((cap.own mem) val)} =
 ([input weight features outputs.] ({[.input] [.weight] [.features] [.outputs]} stdlib.qwen35.batch.project [out.]) {[.input] [.out]}))
"#;
    src.push_str(
        &include_str!("../catena/testdata/qwen35_projection_reference.hex")
            .replace("stdlib.qwen35.batch.", "stdlib.qwen35.reference."),
    );
    src.push_str(&wrapper.replace("stdlib.qwen35.batch.", "stdlib.qwen35.reference."));
    src.push_str(
        &wrapper
            .replace("test.project-batch", "test.project-tile")
            .replace("stdlib.qwen35.batch.", "stdlib.qwen35.tile."),
    );

    let extra = r#"
(def program test.extra-residual : {((cap.own mem) val) ((cap.ref mem) val) ((cap.own mem) val) (u64 val) (u64 val)} -> ((cap.own mem) val) =
 ([input weight residual features outputs.] ({[.input] [.weight] [.features] [.outputs] [.residual]} stdlib.qwen35.reference.residual)))
(def program test.extra-gate : {((cap.own mem) val) ((cap.ref mem) val) ((cap.ref mem) val) (u64 val) (u64 val)} -> ((cap.own mem) val) =
 ([input weight up features outputs.] ({[.input] [.weight] [.features] [.outputs] [.up]} stdlib.qwen35.reference.gate)))
"#;
    src.push_str(extra);
    src.push_str(
        &extra
            .replace("test.extra-", "test.tile-")
            .replace("stdlib.qwen35.reference.", "stdlib.qwen35.tile."),
    );
    let module = compile(crate::catena::runtime_sources(&src), GpuDialect::Hip)?;
    let mut rt = Runtime::new(GpuDialect::Hip)?;
    let artifact = rt.load(module)?;
    for (features, outputs, tokens) in [
        (32usize, 10usize, 1usize),
        (32, 10, 3),
        (32, 10, 7),
        (2048, 2048, 128),
        (5120, 2048, 128),
    ] {
        let data = |n: usize, seed: usize| {
            (0..n)
                .map(|i| rounded(((i * seed + i / 17) % 23) as f32 / 32.0 - 0.34375))
                .collect::<Vec<_>>()
        };
        let mut input = upload(&rt, &data(features * tokens, 7))?;
        let weight = upload(&rt, &data(features * outputs, 11))?;
        let mut expected = Vec::new();
        for name in ["test.project-batch", "test.project-tile"] {
            let mut out = memories(
                &artifact,
                name,
                vec![
                    input.into(),
                    weight.as_ref().into(),
                    (features as u64).into(),
                    (outputs as u64).into(),
                ],
            )?;
            let result = out.pop().unwrap().try_to_u16_vec()?;
            input = out.pop().unwrap();
            if expected.is_empty() {
                expected = result;
            } else {
                assert_eq!(result, expected, "tile K={features} N={outputs} T={tokens}");
            }
            if features >= 2048 {
                let start = std::time::Instant::now();
                for _ in 0..12 {
                    let mut out = memories(
                        &artifact,
                        name,
                        vec![
                            input.into(),
                            weight.as_ref().into(),
                            (features as u64).into(),
                            (outputs as u64).into(),
                        ],
                    )?;
                    out.pop();
                    input = out.pop().unwrap();
                }
                eprintln!(
                    "{name} K={features} N={outputs} T={tokens} us={:.3}",
                    start.elapsed().as_secs_f64() * 1e6 / 12.0
                );
            }
        }
    }

    for tokens in [1usize, 3, 7, 33] {
        let input = (0..tokens * 32)
            .map(|i| rounded((i % 37) as f32 / 64.0 - 0.2))
            .collect::<Vec<_>>();
        let residual = (0..tokens * 10)
            .map(|i| rounded((i % 23) as f32 / 32.0 - 0.2))
            .collect::<Vec<_>>();
        let weight = upload(
            &rt,
            &(0..320)
                .map(|i| rounded((i % 31) as f32 / 64.0 - 0.1))
                .collect::<Vec<_>>(),
        )?;
        let up = upload(
            &rt,
            &(0..320)
                .map(|i| rounded((i % 29) as f32 / 64.0 - 0.1))
                .collect::<Vec<_>>(),
        )?;
        for kind in ["residual", "gate"] {
            let mut expected = Vec::new();
            for prefix in ["test.extra-", "test.tile-"] {
                let third = if kind == "gate" {
                    up.as_ref().into()
                } else {
                    upload(&rt, &residual)?.into()
                };
                let actual = last(memories(
                    &artifact,
                    &format!("{prefix}{kind}"),
                    vec![
                        upload(&rt, &input)?.into(),
                        weight.as_ref().into(),
                        third,
                        32u64.into(),
                        10u64.into(),
                    ],
                )?)
                .try_to_u16_vec()?;
                if expected.is_empty() {
                    expected = actual;
                } else {
                    assert_eq!(actual, expected, "tile {kind} T={tokens}");
                }
            }
        }
    }
    Ok(())
}

#[test]
fn qwen35_batch_full_attention_on_gpu() -> Result<()> {
    if std::env::var_os("CATENA_QWEN3_GPU_TESTS").is_none() {
        return Ok(());
    }
    let mut c = tiny_moe_config();
    c.head_dim = 256;
    c.rope_dimension_count = 64;
    let options = InferenceOptions::default();
    let mut src = source(&c, options)?.replace(
        "(def program qwen3 :",
        "(def program stdlib.test.unused-model :",
    );
    let prepare = r#"
(def program test.full-batch : {((cap.own mem) val) ((cap.own mem) val) ((cap.own mem) val) ((cap.own mem) val) ((cap.ref mem) val) ((cap.ref mem) val) (u64 val) (u64 val) (u64 val)} -> {((cap.own mem) val) ((cap.own mem) val)} =
 ([qg key value cache qw kw key-target value-target start.]
 ({[.qg] [.key] [.value] [.start] [.cache] [.qw] [.kw] [.key-target] [.value-target]} stdlib.qwen35.batch.full-prepare)))
"#;
    src.push_str(prepare);
    src.push_str(
        &prepare
            .replace("test.full-batch", "test.full-row")
            .replace(
                " ({[.qg] [.key] [.value] [.start]",
                " ([.start] qwen3.rope-one [rope.])\n ({[.qg] [.key] [.value] [.rope]",
            )
            .replace("stdlib.qwen35.batch.full-prepare", "qwen35.full-prepare"),
    );
    let staged = r#"
(def program test.staged-batch : {((cap.own mem) val) ((cap.own mem) val) ((cap.own mem) val) (u64 val) (u64 val) (u64 val)} -> {((cap.own mem) val) ((cap.own mem) val)} =
 ([query cache qg start key-base value-base.]
 ({[.query] [.cache] [.qg] [.start] [.key-base] [.value-base]} stdlib.qwen35.batch.staged [out.]) {[.cache] [.out]}))
"#;
    src.push_str(staged);
    src.push_str(
        &include_str!("../catena/qwen35_prefill.hex")
            .replace("{{HIDDEN_SIZE}}", &format!("{:016X}", c.hidden_size)),
    );
    src.push_str(r#"
(def program test.staged-row : {((cap.own mem) val) ((cap.own mem) val) ((cap.own mem) val) (u64 val) (u64 val) (u64 val)} -> {((cap.own mem) val) ((cap.own mem) val)} =
 ([query cache qg keys key-base value-base.]
 ({[.query] [.cache] [.keys] [.key-base]} qwen3.scores [scores.])
 ({[.scores] [.keys]} qwen35.softmax [probs.])
 ({[.probs] [.cache] [.keys] [.value-base]} qwen35.context [context.])
 ({[.qg] [.context]} qwen35.gate-context [out.]) {[.cache] [.out]}))
"#);
    let module = compile(crate::catena::runtime_sources(&src), GpuDialect::Hip)?;
    let mut rt = Runtime::new(GpuDialect::Hip)?;
    let artifact = rt.load(module)?;
    let data = |n: usize, seed: usize| {
        (0..n)
            .map(|i| rounded(((i * seed + i / 17) % 31) as f32 / 128.0 - 0.1))
            .collect::<Vec<_>>()
    };
    let qw = rt.mem_f32(&vec![1.0; 256])?;
    let kw = rt.mem_f32(&vec![1.0; 256])?;
    for tokens in [1usize, 3, 7, 33] {
        for start in [0usize, 5] {
            let qg = data(tokens * 1024, 7);
            let key = data(tokens * 256, 11);
            let val = data(tokens * 256, 13);
            let keybase = 17usize;
            let valuebase = keybase + 64 * 256;
            let initial = data(valuebase + 64 * 256 + 19, 19);
            let mut cache = upload(&rt, &initial)?;
            let mut queries = Vec::new();
            for t in 0..tokens {
                let mut out = memories(
                    &artifact,
                    "test.full-row",
                    vec![
                        upload(&rt, &qg[t * 1024..(t + 1) * 1024])?.into(),
                        upload(&rt, &key[t * 256..(t + 1) * 256])?.into(),
                        upload(&rt, &val[t * 256..(t + 1) * 256])?.into(),
                        cache.into(),
                        qw.as_ref().into(),
                        kw.as_ref().into(),
                        ((keybase + (start + t) * 256) as u64).into(),
                        ((valuebase + (start + t) * 256) as u64).into(),
                        ((start + t) as u64).into(),
                    ],
                )?;
                cache = out.pop().unwrap();
                queries.extend(values(&out.pop().unwrap())?);
            }
            let mut out = memories(
                &artifact,
                "test.full-batch",
                vec![
                    upload(&rt, &qg)?.into(),
                    upload(&rt, &key)?.into(),
                    upload(&rt, &val)?.into(),
                    upload(&rt, &initial)?.into(),
                    qw.as_ref().into(),
                    kw.as_ref().into(),
                    ((keybase + start * 256) as u64).into(),
                    ((valuebase + start * 256) as u64).into(),
                    (start as u64).into(),
                ],
            )?;
            let actualcache = out.pop().unwrap();
            let actualqueries = values(&out.pop().unwrap())?;
            assert_eq!(
                actualqueries, queries,
                "full query T={tokens} start={start}"
            );
            assert_eq!(
                actualcache.try_to_u16_vec()?,
                cache.try_to_u16_vec()?,
                "full cache T={tokens} start={start}"
            );
            let mut staged_expected = Vec::new();
            for t in 0..tokens {
                let mut out = memories(
                    &artifact,
                    "test.staged-row",
                    vec![
                        upload(&rt, &queries[t * 512..(t + 1) * 512])?.into(),
                        cache.into(),
                        upload(&rt, &qg[t * 1024..(t + 1) * 1024])?.into(),
                        ((start + t + 1) as u64).into(),
                        (keybase as u64).into(),
                        (valuebase as u64).into(),
                    ],
                )?;
                staged_expected.extend(out.pop().unwrap().try_to_u16_vec()?);
                cache = out.pop().unwrap();
            }
            let staged = last(memories(
                &artifact,
                "test.staged-batch",
                vec![
                    upload(&rt, &queries)?.into(),
                    actualcache.into(),
                    upload(&rt, &qg)?.into(),
                    (start as u64).into(),
                    (keybase as u64).into(),
                    (valuebase as u64).into(),
                ],
            )?)
            .try_to_u16_vec()?;
            assert_eq!(
                staged, staged_expected,
                "staged BF16 attention T={tokens} start={start}"
            );
        }
    }
    Ok(())
}
