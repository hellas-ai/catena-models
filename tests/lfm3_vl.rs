use catena_models::{
    catena::InferenceOptions,
    model::Model,
    tokenizer::QwenTokenizer,
    vision::{VisionConfig, VisionKind, VisionModel, image::ImagePatches},
};
use std::path::PathBuf;
fn cached(name: &str) -> anyhow::Result<PathBuf> {
    let root = std::env::var_os("HF_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var_os("HOME").unwrap()).join(".cache/huggingface")
        });
    for entry in std::fs::read_dir(root.join("hub/models--LiquidAI--LFM3-VL-2B-GGUF/snapshots"))? {
        let path = entry?.path().join(name);
        if path.exists() {
            return Ok(path);
        }
    }
    anyhow::bail!("missing cached {name}")
}
#[test]
#[ignore = "requires cached LFM3-VL BF16 checkpoint and projector"]
fn projector_metadata_and_template() -> anyhow::Result<()> {
    let config = VisionConfig::from_gguf(&cached("mmproj-LFM3-VL-2B-BF16.gguf")?)?;
    assert_eq!(config.kind, VisionKind::Lfm3);
    assert_eq!(
        (
            config.hidden_size,
            config.output_size,
            config.layers,
            config.heads,
            config.kv_heads
        ),
        (1024, 2048, 14, 16, 8)
    );
    assert_eq!(config.conv_layers, vec![0, 1, 3, 5, 7, 9, 11, 13]);
    let tokenizer = QwenTokenizer::from_gguf(cached("LFM3-VL-2B-BF16.gguf")?)?.unwrap();
    assert_eq!(tokenizer.special_token_id("<|image|>")?, 125030);
    Ok(())
}
#[test]
#[ignore = "requires HIP GPU and cached LFM3-VL BF16 projector"]
fn image_embeddings_match_reference() -> anyhow::Result<()> {
    let vision = VisionModel::load_gguf(
        cached("mmproj-LFM3-VL-2B-BF16.gguf")?,
        catena_lang::runtime::GpuDialect::Hip,
        None,
    )?;
    for (w, h) in [(256, 256), (320, 256), (257, 193)] {
        let rgb = (0..w * h)
            .flat_map(|i| {
                let x = i % w;
                let y = i / w;
                [
                    (x * 3 + y * 5) as u8,
                    (x * 7 + y * 2) as u8,
                    (x + y * 11) as u8,
                ]
            })
            .collect::<Vec<_>>();
        let patches = ImagePatches::from_rgb(&rgb, w, h, vision.config())?;
        let image = vision.encode_patches(&patches)?;
        assert_eq!(image.token_count(), patches.grid[1] * patches.grid[2] / 4);
        let actual = image.to_f32_vec()?;
        assert!(actual.iter().all(|v| v.is_finite()));
        if let Ok(prefix) = std::env::var("LFM3_VL_REFERENCE") {
            let normalized = std::fs::read(format!("{prefix}-{w}x{h}-pixels.f32"))?;
            let normalized = normalized
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect::<Vec<_>>();
            assert_eq!(normalized.len(), patches.resized_rgb.len());
            for (&x, &u) in normalized.iter().zip(&patches.resized_rgb) {
                assert!(
                    (x - (f32::from(u) / 255. - 0.5) * 2.).abs() < 1e-6,
                    "resized pixels differ from torchvision reference"
                );
            }
            let bytes = std::fs::read(format!("{prefix}-{w}x{h}.f32"))?;
            let reference = bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect::<Vec<_>>();
            assert_eq!(actual.len(), reference.len());
            let error = actual
                .iter()
                .zip(&reference)
                .map(|(a, b)| f64::from(a - b).powi(2))
                .sum::<f64>();
            let scale = reference.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>();
            let relative = (error / scale).sqrt();
            eprintln!("{w}x{h} vision relative RMS {relative}");
            assert!(relative < 0.005, "vision embedding mismatch");
        }
    }
    Ok(())
}
#[test]
#[ignore = "requires HIP GPU and cached LFM3-VL BF16 model/projector"]
fn image_prefill_decode_and_resume() -> anyhow::Result<()> {
    let vision = VisionModel::load_gguf(
        cached("mmproj-LFM3-VL-2B-BF16.gguf")?,
        catena_lang::runtime::GpuDialect::Hip,
        None,
    )?;
    let path = cached("LFM3-VL-2B-BF16.gguf")?;
    let tokenizer = QwenTokenizer::from_gguf(&path)?.unwrap();
    let model = Model::load_gguf_with_multimodal_options(
        &path,
        InferenceOptions::default().with_wmma_prefill(true),
    )?;
    let rgb = [255, 0, 0].repeat(256 * 256);
    let patches = ImagePatches::from_rgb(&rgb, 256, 256, vision.config())?;
    let prompt = catena_models::vision::PreparedMultimodalPrompt::new(
        &tokenizer,
        "What color is the image?",
        vec![vision.encode_patches(&patches)?],
    )?;
    assert_eq!(prompt.rope_delta(), 0);
    assert_eq!(
        prompt.positions().last(),
        Some(&[(prompt.token_count() - 1) as u64; 3])
    );
    let mut session = model.session(prompt.token_count() + 32)?;
    let mut next = session.prefill_multimodal_next_token(&prompt)?;
    let checkpoint = session.into_checkpoint()?;
    let mut session = model.resume_checkpoint(&checkpoint)?;
    let mut generated = vec![next];
    for _ in 0..12 {
        if Some(next) == model.config().eos_token_id {
            break;
        }
        next = session.decode_next_token(next)?;
        generated.push(next);
    }
    let decoded = tokenizer.decode(&generated, true)?;
    eprintln!("red image answer: {decoded}");
    assert!(decoded.to_lowercase().contains("red"));
    Ok(())
}
