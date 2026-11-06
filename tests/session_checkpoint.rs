use catena_lang::runtime::{GpuDialect, Runtime, RuntimeModule, Value};
use catena_models::{
    catena::{InferenceOptions, compile_sources, core_sources},
    model::Model,
};

fn copy_module(dialect: GpuDialect) -> anyhow::Result<RuntimeModule> {
    compile_sources(
        core_sources().chain([
            include_str!("../src/catena/memory.hex"),
            include_str!("../src/catena/session_copy.hex"),
        ]),
        dialect,
    )
}

#[test]
fn checkpoint_copies_compile_for_both_backends() -> anyhow::Result<()> {
    for dialect in [GpuDialect::Cuda, GpuDialect::Hip] {
        let module = copy_module(dialect)?;
        let mut names: Vec<_> = module
            .functions
            .iter()
            .map(|f| f.source_name.as_str())
            .collect();
        names.sort_unstable();
        assert_eq!(names, ["qwen3.copy-buffer", "qwen3.copy-buffer-16"]);
        // Helpers stay private, and the kernels only move bits. SDK helper
        // definitions may contain floating point, so examine emitted kernels.
        for kernel in module.source.split("__global__").skip(1) {
            let body = kernel.split("extern \"C\"").next().unwrap();
            assert!(!body.contains("catena_bf16_to_f32("));
            assert!(!body.contains("catena_bf16_from_f32("));
        }
    }
    Ok(())
}

#[test]
fn checkpoint_entrypoints_are_in_generated_model_sources() -> anyhow::Result<()> {
    use catena_models::{catena::generated_model_source_for_options, spec::ModelConfig};
    for (config, options) in [
        (ModelConfig::qwen3_0_6b(), InferenceOptions::default()),
        (
            ModelConfig::qwen35_0_8b(),
            InferenceOptions::default().with_wmma_prefill(true),
        ),
        (
            ModelConfig::qwen36_35b_a3b(),
            InferenceOptions::default().with_wmma_prefill(true),
        ),
    ] {
        let source = generated_model_source_for_options(&config, options)?;
        assert_eq!(
            source.matches("(def program qwen3.copy-buffer :").count(),
            1
        );
        assert_eq!(
            source
                .matches("(def program qwen3.copy-buffer-16 :")
                .count(),
            1
        );
        // Interpret the complete source, including all callback types and model
        // insertion order, without requiring native compilation or GPU access.
        metacat::theory::RawTheorySet::from_texts(catena_models::catena::runtime_sources(&source))?;
    }
    Ok(())
}

#[test]
#[ignore = "requires a HIP GPU"]
fn checkpoint_device_copy_preserves_all_bits() -> anyhow::Result<()> {
    let mut runtime = Runtime::new(GpuDialect::Hip)?;
    let artifact = runtime.load(copy_module(GpuDialect::Hip)?)?;
    // Includes zeros, infinities, signaling/quiet NaNs and arbitrary payloads.
    let halves: Vec<u16> = (0..=u16::MAX).collect();
    for len in [0, 1, 2, 3, 4, 255, 256, 257, halves.len()] {
        let input = runtime.mem_u16(&halves[..len])?;
        let [Value::MemOwn(copy)] = artifact.exec(
            "qwen3.copy-buffer-16",
            [input.as_ref().into(), (len as u64).into()],
        )?
        else {
            panic!("copy ABI mismatch")
        };
        assert_eq!(copy.try_to_u16_vec()?, halves[..len]);
    }
    let words: Vec<u64> = halves
        .chunks_exact(4)
        .map(|x| {
            u64::from(x[0])
                | (u64::from(x[1]) << 16)
                | (u64::from(x[2]) << 32)
                | (u64::from(x[3]) << 48)
        })
        .collect();
    for len in [0, 1, 255, 256, 257, words.len()] {
        let input = runtime.mem_u64(&words[..len])?;
        let [Value::MemOwn(copy)] = artifact.exec(
            "qwen3.copy-buffer",
            [input.as_ref().into(), (len as u64).into()],
        )?
        else {
            panic!("copy ABI mismatch")
        };
        assert_eq!(copy.try_to_u64_vec()?, words[..len]);
    }
    Ok(())
}

#[test]
#[ignore = "requires a GPU and CATENA_PREFIX_MODEL pointing to a BF16 GGUF or HF selector"]
fn checkpoint_resume_is_bitwise_exact_and_independent() -> anyhow::Result<()> {
    let selector = std::env::var("CATENA_PREFIX_MODEL")?;
    let options = InferenceOptions::default()
        .with_wmma_prefill(std::env::var("CATENA_PREFIX_WMMA").as_deref() != Ok("0"));
    let path = if std::path::Path::new(&selector).exists() {
        std::path::PathBuf::from(&selector)
    } else {
        catena_models::hfhub::download_gguf(&selector)?.primary
    };
    let model = Model::load_gguf_with_options(&path, options)?;
    let tokens: Vec<_> = (0..2051).map(|i| 100 + i % 97).collect();
    let mut session = model.session(4096)?;
    session.append_next_token(&tokens[..1024])?;
    let checkpoint = session.into_checkpoint()?;
    assert_eq!(checkpoint.position(), 1024);
    assert_eq!(checkpoint.capacity(), 4096);
    assert!(checkpoint.byte_len() > 0);
    let mut warm = model.resume_checkpoint(&checkpoint)?;
    warm.append_next_token(&tokens[1024..2048])?;
    let warm_logits = warm.append_logits(&tokens[2048..])?;
    let mut warm_next = warm.decode_next_token(101)?;
    let mut generated = vec![warm_next];
    for _ in 0..3 {
        warm_next = warm.decode_next_token(warm_next)?;
        generated.push(warm_next);
    }
    drop(warm);
    for label in ["cold", "restored again after decode"] {
        let mut session = if label == "cold" {
            let mut session = model.session(4096)?;
            session.append_next_token(&tokens[..1024])?;
            session
        } else {
            model.resume_checkpoint(&checkpoint)?
        };
        session.append_next_token(&tokens[1024..2048])?;
        let logits = session.append_logits(&tokens[2048..])?;
        assert_eq!(
            logits.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            warm_logits.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            "{label}"
        );
        let mut next = session.decode_next_token(101)?;
        assert_eq!(next, generated[0], "{label}");
        for &expected in &generated[1..] {
            next = session.decode_next_token(next)?;
            assert_eq!(next, expected, "{label}");
        }
    }
    // A same-config model is still a different owner; reject before copying.
    drop(model);
    let other = Model::load_gguf_with_options(&path, options)?;
    assert!(other.resume_checkpoint(&checkpoint).is_err());
    Ok(())
}
