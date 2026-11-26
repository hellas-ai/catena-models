use anyhow::Result;
use catena_lang::runtime::{GpuDialect, Runtime, Value};
use catena_models::{
    catena::{
        InferenceOptions, PrefillDtype, compile_sources, generated_model_source_for_options,
        runtime_sources,
    },
    spec::ModelConfig,
    weights::{bf16_to_f32, f32_to_bf16},
};

fn config() -> ModelConfig {
    let mut c = ModelConfig::qwen36_35b_a3b();
    c.num_layers = 4;
    c.hidden_size = 8;
    c.intermediate_size = 32;
    c.shared_expert_intermediate_size = Some(32);
    c.head_dim = 32;
    c.rope_dimension_count = 8;
    c.mrope_sections = [1, 1, 2];
    c.num_attention_heads = 8;
    c.num_key_value_heads = 4;
    c
}
fn options() -> InferenceOptions {
    InferenceOptions::default()
        .with_wmma_prefill(true)
        .with_prefill_dtype(PrefillDtype::Fp8)
}

#[test]
fn tiny_moe_graph_reuses_dense_attention_and_deltanet() -> Result<()> {
    for c in [config(), ModelConfig::qwen36_35b_a3b()] {
        let source = generated_model_source_for_options(&c, options())?;
        assert!(source.contains("stdlib.qwen35.fp8.moe.ffn"));
        assert!(source.contains("stdlib.qwen3.fp8.attention.run [context.]"));
        assert!(source.contains("stdlib.qwen35.dense.scan-l8"));
        assert!(source.contains("stdlib.qwen35.fp8.tail.project"));
        assert!(!source.contains("{{EXPERT_COUNT}}"));
    }
    Ok(())
}

fn kernel_module(mut c: ModelConfig) -> Result<catena_lang::runtime::RuntimeModule> {
    c.num_experts = 3;
    c.experts_per_token = 2;
    let mut source = generated_model_source_for_options(&c, options())?;
    source.push_str(r#"
(def program tail-test : {((cap.own mem) val) ((cap.ref mem) val) ((cap.ref mem) val) (u64 val) (u64 val)} -> ((cap.own mem) val) =
 ([input weight ws k n.]
  ({[.input]} stdlib.models.memory.own-count-bf16 [elements.])
  ({[.elements] [.k]} stdlib.models.memory.u64.div [rows.])
  ({[.input] [.k]} stdlib.qwen35.fp8.quantize-matrix *.elim [bits scales.])
  ({[.bits] [.weight] [.scales] [.ws] [.rows] [.k] [.n]} stdlib.qwen35.fp8.tail.project [out.])
  ({[.bits] [.out]} stdlib.models.resources.free-f32-before [a.])
  ({[.scales] [.a]} stdlib.models.resources.free-f32-before [b.])
  ({[.input] [.b]} stdlib.models.resources.free-bf16-before)
 ))
(def program expert-test : {((cap.own mem) val) ((cap.own mem) val) ((cap.ref mem) val) ((cap.ref mem) val) ((cap.ref mem) val) ((cap.ref mem) val) ((cap.ref mem) val) ((cap.ref mem) val) ((cap.ref mem) val) ((cap.ref mem) val) ((cap.ref mem) val) ((cap.ref mem) val) ((cap.ref mem) val) ((cap.ref mem) val) ((cap.ref mem) val) ((cap.ref mem) val)} -> ((cap.own mem) val) =
 ([x residual router sr gate gs up us down ds sg sgs su sus sd sds.]
  ({[.x] [.residual] [.router] [.gate] [.up] [.down] [.sr] [.sg] [.su] [.sd] [.gate] [.gs] [.up] [.us] [.down] [.ds] [.sg] [.sgs] [.su] [.sus] [.sd] [.sds]} stdlib.qwen35.fp8.moe.ffn)
 ))
"#);
    let source = source
        .replace("stdlib.qwen3.", "stdlib.fixture-internal.")
        .replace("qwen3.", "stdlib.fixture-public.")
        .replace("qwen3 :", "stdlib.fixture-root :")
        .replace("qwen35.", "stdlib.fixture35.");
    let module = compile_sources(runtime_sources(&source), GpuDialect::Cuda)?;
    assert!(
        module
            .source
            .contains("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32")
    );
    Ok(module)
}

#[test]
#[ignore = "requires CUDA SM89+ and nvcc"]
fn padded_and_grouped_kernels_match_cpu_reference() -> Result<()> {
    let mut runtime = Runtime::new(GpuDialect::Cuda)?;
    let artifact = runtime.load(kernel_module(config())?)?;
    for (rows, k, n) in [(2usize, 8usize, 32usize), (17, 136, 72), (65, 256, 128)] {
        let x: Vec<_> = (0..rows * k)
            .map(|i| f32_to_bf16([1.0, 0.5, -1.0, -0.5][i % 4]))
            .collect();
        let pk = k.next_multiple_of(128);
        let pn = n.next_multiple_of(128);
        let mut weight = vec![0; pk * pn];
        for row in 0..n {
            for col in 0..k {
                weight[row * pk + col] = [0x38, 0x30, 0xb8][(row + col) % 3];
            }
        }
        let scales: Vec<f32> = (0..pn / 128 * (pk / 128))
            .map(|i| (i + 1) as f32 / 8.0)
            .collect();
        let w = runtime.mem_u8(&weight)?;
        let ws = runtime.mem_f32(&scales)?;
        let [Value::MemOwn(out)] = artifact.exec(
            "tail-test",
            [
                runtime.mem_u16(&x)?.into(),
                w.as_ref().into(),
                ws.as_ref().into(),
                (k as u64).into(),
                (n as u64).into(),
            ],
        )?
        else {
            anyhow::bail!("expected output");
        };
        let actual = out.try_to_u16_vec()?;
        for row in 0..rows {
            for col in 0..n {
                let expected: f32 = (0..k)
                    .map(|i| {
                        bf16_to_f32(x[row * k + i])
                            * [1.0, 0.5, -1.0][(col + i) % 3]
                            * scales[col / 128 * (pk / 128) + i / 128]
                    })
                    .sum();
                assert_eq!(
                    actual[row * n + col],
                    f32_to_bf16(expected),
                    "tail {rows}x{k}x{n} ({row},{col})"
                );
            }
        }
    }
    check_experts(&runtime, &artifact, 8, 32, 32)?;
    Ok(())
}

fn check_experts(
    runtime: &Runtime,
    artifact: &catena_lang::runtime::Artifact,
    h: usize,
    i: usize,
    shared: usize,
) -> Result<()> {
    // Constant representable matrices give a simple independent routed/shared oracle.
    let mut router = vec![0.0; 3 * h];
    router[..h].fill(8.0 / h as f32);
    router[2 * h..].fill(-8.0 / h as f32);
    let router = runtime.mem_f32(&router)?;
    let sr = runtime.mem_f32(&vec![0.0; h])?;
    let matrix = |experts: usize, rows: usize, cols: usize| -> Result<_> {
        let pr = rows.next_multiple_of(128);
        let pc = cols.next_multiple_of(128);
        let mut bytes = vec![0; experts * pr * pc];
        for e in 0..experts {
            for r in 0..rows {
                bytes[e * pr * pc + r * pc..e * pr * pc + r * pc + cols].fill(0x38);
            }
        }
        Ok(runtime.mem_u8(&bytes)?)
    };
    let scales = |experts: usize, rows: usize, cols: usize, factor: f32| -> Result<_> {
        let blocks = rows.div_ceil(128) * cols.div_ceil(128);
        let values: Vec<_> = (0..experts)
            .flat_map(|e| vec![(e + 1) as f32 * factor; blocks])
            .collect();
        Ok(runtime.mem_f32(&values)?)
    };
    let gate = matrix(3, i, h)?;
    let up = matrix(3, i, h)?;
    let down = matrix(3, h, i)?;
    let sg = matrix(1, shared, h)?;
    let su = matrix(1, shared, h)?;
    let sd = matrix(1, h, shared)?;
    let gs = scales(3, i, h, 1.0 / h as f32)?;
    let us = scales(3, i, h, 0.5 / h as f32)?;
    let ds = scales(3, h, i, 1.0 / i as f32)?;
    let sgs = scales(1, shared, h, 1.0 / h as f32)?;
    let sus = scales(1, shared, h, 2.0 / h as f32)?;
    let sds = scales(1, h, shared, 2.0 / shared as f32)?;
    for rows in [2, 17, 65] {
        let x: Vec<_> = (0..rows * h)
            .map(|i| f32_to_bf16(if i / h % 2 == 0 { 1.0 } else { -1.0 }))
            .collect();
        let input = runtime.mem_u16(&x)?;
        let residual = runtime.mem_u16(&vec![f32_to_bf16(0.5); rows * h])?;
        let mut result = artifact.exec_values(
            "expert-test",
            vec![
                input.into(),
                residual.into(),
                router.as_ref().into(),
                sr.as_ref().into(),
                gate.as_ref().into(),
                gs.as_ref().into(),
                up.as_ref().into(),
                us.as_ref().into(),
                down.as_ref().into(),
                ds.as_ref().into(),
                sg.as_ref().into(),
                sgs.as_ref().into(),
                su.as_ref().into(),
                sus.as_ref().into(),
                sd.as_ref().into(),
                sds.as_ref().into(),
            ],
        )?;
        let Value::MemOwn(out) = result.remove(0) else {
            anyhow::bail!("expected output");
        };
        let actual = out.try_to_u16_vec()?;
        let round = |x| bf16_to_f32(f32_to_bf16(x));
        for row in 0..rows {
            let x = if row % 2 == 0 { 1.0 } else { -1.0 };
            let ids = if x > 0.0 { [0, 1] } else { [2, 1] };
            let prob = 1.0 / (1.0 + (-8.0_f32).exp());
            let mut routed = 0.0;
            for (slot, e) in ids.into_iter().enumerate() {
                let g = x * (e + 1) as f32;
                let up = x * (e + 1) as f32 * 0.5;
                let active = round(
                    (g / (1.0 + (-g).exp())) * up * if slot == 0 { prob } else { 1.0 - prob },
                );
                routed += active * (e + 1) as f32;
            }
            let g = x;
            let up = x * 2.0;
            let shared = round(round(round(g / (1.0 + (-g).exp())) * up) * 0.5) * 2.0;
            let expected = round(round(routed + 0.5) + round(shared));
            for col in 0..h {
                let got = bf16_to_f32(actual[row * h + col]);
                assert!(
                    (got - expected).abs() < 0.02 * expected.abs() + 0.01,
                    "expert row{row} {got} != {expected}"
                );
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA SM89+ and nvcc"]
fn grouped_kernels_handle_multiple_blocks_and_wider_shared_expert() -> Result<()> {
    let mut c = config();
    c.hidden_size = 256;
    c.intermediate_size = 128;
    c.shared_expert_intermediate_size = Some(256);
    let mut runtime = Runtime::new(GpuDialect::Cuda)?;
    let artifact = runtime.load(kernel_module(c)?)?;
    check_experts(&runtime, &artifact, 256, 128, 256)
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(&a, &b)| a as f64 * b as f64).sum();
    let aa: f64 = a.iter().map(|&x| (x as f64).powi(2)).sum();
    let bb: f64 = b.iter().map(|&x| (x as f64).powi(2)).sum();
    dot / (aa * bb).sqrt()
}

#[test]
#[ignore = "requires CUDA; downloads tiny-random/qwen3.6-moe"]
fn tiny_random_moe_fp8_prefill_and_decode() -> Result<()> {
    use catena_models::{hfhub::download_model, model::Model};
    let checkpoint = download_model("tiny-random/qwen3.6-moe")?;
    let baseline = Model::load_checkpoint_with_options(&checkpoint, InferenceOptions::default())?;
    let fp8 = Model::load_checkpoint_with_options(&checkpoint, options())?;
    for n in [1, 2, 17, 65, 128] {
        let tokens: Vec<u64> = (0..n).map(|i| 100 + i as u64 * 17 % 5000).collect();
        let mut reference = baseline.session(n + 4)?;
        let expected = reference.prefill_logits(&tokens)?;
        let mut session = fp8.session(n + 4)?;
        let actual = session.prefill_logits(&tokens)?;
        assert!(actual.iter().all(|v| v.is_finite()));
        let cos = cosine(&expected, &actual);
        eprintln!("tiny-random MoE pp{n} cosine={cos:.8}");
        // Allow weight/activation quantization error in this random 8-wide model;
        // numerical kernel correctness is checked against CPU oracles above.
        assert!(cos > 0.95, "FP8 logits cosine {cos}");
        if n == 1 {
            assert_eq!(expected, actual);
        }
        for token in [100, 200, 300] {
            let expected = reference.decode_logits(token)?;
            let actual = session.decode_logits(token)?;
            assert!(actual.iter().all(|v| v.is_finite()));
            let cos = cosine(&expected, &actual);
            eprintln!("decode pp{n} token{token} cosine={cos:.8}");
            if n == 1 {
                assert_eq!(expected, actual);
            }
        }
    }
    let tokens: Vec<u64> = (0..128).map(|i| 100 + i * 17 % 5000).collect();
    let mut full_session = fp8.session(132)?;
    let whole = full_session.prefill_logits(&tokens)?;
    let mut session = fp8.session(132)?;
    session.append_logits(&tokens[..17])?;
    let chunked = session.append_logits(&tokens[17..])?;
    let cos = cosine(&whole, &chunked);
    eprintln!("tiny-random MoE chunked cosine={cos:.8}");
    assert!(cos > 0.99);
    for token in [100, 200, 300] {
        let whole = full_session.decode_logits(token)?;
        let chunked = session.decode_logits(token)?;
        let cos = cosine(&whole, &chunked);
        eprintln!("chunked decode token{token} cosine={cos:.8}");
        assert!(cos > 0.98);
    }
    Ok(())
}
