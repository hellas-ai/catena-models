use anyhow::{Context, Result};
use catena_models::{
    catena::{InferenceOptions, PrefillDtype, generated_model_source_for_options},
    hfhub::{HubCheckpoint, download_model_with_layer_limit},
    model::Model,
    spec::{ModelConfig, weight_specs},
};
use std::{collections::BTreeSet, fs, path::PathBuf};

#[test]
fn prefix_graphs_keep_globals_and_handle_single_attention_types() -> Result<()> {
    let original = ModelConfig::qwen35_0_8b();
    let defaults = InferenceOptions::default().with_wmma_prefill(true);
    let full = generated_model_source_for_options(&original, defaults)?;
    for n in [1, 3, 4, 5, 24] {
        let mut config = original.clone();
        config.num_layers = n;
        let specs = weight_specs(&config)?;
        assert_eq!(specs.first().unwrap().name, "model.embed_tokens.weight");
        assert_eq!(specs[specs.len() - 2].name, "model.norm.weight");
        assert_eq!(specs.last().unwrap().name, "lm_head.weight");
        assert!(
            !specs
                .iter()
                .any(|s| s.name.starts_with(&format!("model.layers.{n}.")))
        );
        for dtype in [PrefillDtype::Bf16, PrefillDtype::Fp8] {
            let source =
                generated_model_source_for_options(&config, defaults.with_prefill_dtype(dtype))?;
            let shadow = source
                .split("(def program stdlib.qwen3.state-shadow :")
                .nth(1)
                .context("state layout")?
                .split("(def program")
                .next()
                .unwrap();
            assert!(shadow.contains(&format!("(const.u64.0x{n:016X})")));
            if n < 4 {
                assert!(source.contains("stdlib.qwen35.apply-linear"));
                assert!(!source.contains("(def program stdlib.qwen35.full-weights"));
                assert!(!source.contains("(def program stdlib.qwen35.branch-full"));
                assert!(!source.contains("(def program stdlib.qwen35.batch.branch-full"));
            }
            if n == 24 && dtype == PrefillDtype::Bf16 {
                assert_eq!(source, full);
            }
        }
    }
    // The same empty-group handling also works for a full-attention-only config.
    let mut config = original;
    config.num_layers = 1;
    config.full_attention_interval = Some(1);
    let source = generated_model_source_for_options(&config, defaults)?;
    assert!(source.contains("stdlib.qwen35.apply-full"));
    assert!(!source.contains("(def program stdlib.qwen35.linear-weights"));
    Ok(())
}

struct Temp(PathBuf);

#[test]
fn moe_prefix_graphs_handle_linear_only_and_mixed_attention() -> Result<()> {
    for n in [1, 3, 4, 5] {
        let mut config = ModelConfig::qwen36_35b_a3b();
        config.num_layers = n;
        for wmma in [false, true] {
            for dtype in [PrefillDtype::Bf16, PrefillDtype::Fp8] {
                if !wmma && dtype == PrefillDtype::Fp8 {
                    continue;
                }
                let source = generated_model_source_for_options(
                    &config,
                    InferenceOptions::default()
                        .with_wmma_prefill(wmma)
                        .with_prefill_dtype(dtype),
                )?;
                assert!(source.contains("qwen35.moe.ffn"));
                if n < 4 {
                    assert!(!source.contains("(def program stdlib.qwen35.full-weights"));
                    assert!(!source.contains("(def program stdlib.qwen35.branch-full"));
                }
            }
        }
    }
    Ok(())
}

impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// Independently remove the tail tensors from the physical safetensors files.
/// This reference has no omitted layers available for the loader to accidentally use.
fn cropped(checkpoint: &HubCheckpoint, tag: &str) -> Result<(HubCheckpoint, Temp)> {
    use memmap2::MmapOptions;
    use safetensors::{SafeTensors, tensor::serialize_to_file};
    let HubCheckpoint::Safetensors { path, config } = checkpoint else {
        anyhow::bail!("expected safetensors");
    };
    let directory = Temp(
        std::env::temp_dir().join(format!("catena-layer-prefix-{tag}-{}", std::process::id())),
    );
    fs::create_dir_all(&directory.0)?;
    let paths = if path.to_string_lossy().ends_with(".index.json") {
        let index: serde_json::Value = serde_json::from_slice(&fs::read(path)?)?;
        let selected = |name: &str| {
            if let Some((_, rest)) = name.split_once(".layers.") {
                rest.split('.')
                    .next()
                    .and_then(|n| n.parse::<usize>().ok())
                    .is_some_and(|n| n < config.num_layers)
            } else {
                !name.contains(".visual.") && !name.starts_with("mtp.")
            }
        };
        index["weight_map"]
            .as_object()
            .context("index weight_map")?
            .iter()
            .filter(|(name, _)| selected(name))
            .map(|(_, shard)| {
                Ok(path
                    .parent()
                    .unwrap()
                    .join(shard.as_str().context("shard name")?))
            })
            .collect::<Result<BTreeSet<_>>>()?
    } else {
        BTreeSet::from([path.clone()])
    };
    let mut output = serde_json::Map::new();
    for (i, shard) in paths.iter().enumerate() {
        let file = fs::File::open(shard)?;
        // SAFETY: immutable downloaded checkpoint; mapping lives through serialization.
        let bytes = unsafe { MmapOptions::new().map(&file)? };
        let tensors = SafeTensors::deserialize(&bytes)?;
        let views: Vec<_> = tensors
            .tensors()
            .into_iter()
            .filter(|(name, _)| {
                if let Some((_, rest)) = name.split_once(".layers.") {
                    rest.split('.')
                        .next()
                        .and_then(|n| n.parse::<usize>().ok())
                        .is_some_and(|n| n < config.num_layers)
                } else {
                    !name.contains(".visual.") && !name.starts_with("mtp.")
                }
            })
            .collect();
        let filename = format!("prefix-{i}.safetensors");
        for (name, _) in &views {
            output.insert(name.clone(), filename.clone().into());
        }
        serialize_to_file(views, None, &directory.0.join(filename))?;
    }
    let index = directory.0.join("model.safetensors.index.json");
    fs::write(
        &index,
        serde_json::to_vec(&serde_json::json!({"weight_map": output}))?,
    )?;
    let mut metadata: serde_json::Value =
        serde_json::from_slice(&fs::read(path.parent().unwrap().join("config.json"))?)?;
    let text = if metadata.get("text_config").is_some() {
        &mut metadata["text_config"]
    } else {
        &mut metadata
    };
    text["num_hidden_layers"] = config.num_layers.into();
    if let Some(layers) = text["layer_types"].as_array_mut() {
        layers.truncate(config.num_layers);
    }
    fs::write(
        directory.0.join("config.json"),
        serde_json::to_vec(&metadata)?,
    )?;
    Ok((
        HubCheckpoint::Safetensors {
            path: index,
            config: config.clone(),
        },
        directory,
    ))
}

fn parity(repo: &str, dtype: PrefillDtype, wmma: bool) -> Result<()> {
    let options = InferenceOptions::default()
        .with_wmma_prefill(wmma)
        .with_prefill_dtype(dtype);
    for n in [1, 4] {
        let checkpoint = download_model_with_layer_limit(repo, Some(n))?;
        let (reference_checkpoint, _temp) = cropped(&checkpoint, &format!("{n}-{dtype:?}-{wmma}"))?;
        let model = Model::load_checkpoint_with_options(&checkpoint, options)?;
        let reference = Model::load_checkpoint_with_options(&reference_checkpoint, options)?;
        assert_eq!(model.config().num_layers, n);
        for len in [1, 2, 17, 65] {
            let tokens: Vec<_> = (0..len).map(|i| 100 + i * 17).collect();
            let mut actual = model.session(len as usize + 4)?;
            let mut expected = reference.session(len as usize + 4)?;
            let a = actual.prefill_logits(&tokens)?;
            assert!(a.iter().all(|v| v.is_finite()));
            assert_eq!(a, expected.prefill_logits(&tokens)?, "prefix={n} pp={len}");
            for token in [100, 200, 300] {
                assert_eq!(actual.decode_logits(token)?, expected.decode_logits(token)?);
            }
        }
        let tokens: Vec<_> = (0..17).map(|i| 100 + i * 17).collect();
        let mut actual = model.session(21)?;
        let mut expected = reference.session(21)?;
        actual.append_logits(&tokens[..2])?;
        expected.append_logits(&tokens[..2])?;
        assert_eq!(
            actual.append_logits(&tokens[2..])?,
            expected.append_logits(&tokens[2..])?
        );
        eprintln!("{repo}: first {n} layers {dtype:?}, cropped-checkpoint parity passed");
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA; downloads Qwen/Qwen3.5-0.8B"]
fn dense_prefix_matches_physically_cropped_checkpoint() -> Result<()> {
    parity("Qwen/Qwen3.5-0.8B", PrefillDtype::Bf16, true)
}

#[test]
#[ignore = "requires CUDA SM89+; downloads surogate/Qwen3.5-0.8B-FP8"]
fn fp8_prefix_matches_physically_cropped_checkpoint() -> Result<()> {
    parity("surogate/Qwen3.5-0.8B-FP8", PrefillDtype::Fp8, true)
}

#[test]
#[ignore = "requires CUDA; downloads Qwen/Qwen3.5-0.8B"]
fn simt_prefix_matches_physically_cropped_checkpoint() -> Result<()> {
    parity("Qwen/Qwen3.5-0.8B", PrefillDtype::Bf16, false)
}
