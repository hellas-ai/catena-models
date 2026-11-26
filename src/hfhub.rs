use std::{collections::BTreeMap, path::PathBuf};

use anyhow::{Context, Result, bail, ensure};
use hf_hub::{HFClientSync, repository::RepoTreeEntry, split_id};

use crate::{checkpoint::read_gguf_model_config, spec::ModelConfig, tokenizer::QwenTokenizer};

/// Local GGUF files selected and cached for a Hub model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HubGguf {
    /// A file accepted by the existing GGUF shard resolver. For a sharded model
    /// this is shard one; all remaining shards are downloaded beside it.
    pub primary: PathBuf,
    pub files: Vec<PathBuf>,
}

/// A text checkpoint and its model-owned metadata in the Hub cache.
#[derive(Debug, Clone)]
pub enum HubCheckpoint {
    Gguf(HubGguf),
    Safetensors {
        path: PathBuf,
        config: Box<ModelConfig>,
    },
}

impl HubCheckpoint {
    pub fn gguf_path(&self) -> Option<&std::path::Path> {
        match self {
            Self::Gguf(gguf) => Some(&gguf.primary),
            Self::Safetensors { .. } => None,
        }
    }

    pub fn config(&self) -> Result<ModelConfig> {
        match self {
            Self::Gguf(gguf) => read_gguf_model_config(&gguf.primary),
            Self::Safetensors { config, .. } => Ok(config.as_ref().clone()),
        }
    }

    pub fn tokenizer(&self) -> Result<QwenTokenizer> {
        match self {
            Self::Gguf(gguf) => QwenTokenizer::from_gguf(&gguf.primary)?
                .context("HF checkpoint has no embedded tokenizer"),
            Self::Safetensors { path, .. } => {
                QwenTokenizer::from_hf_files(path.parent().context("missing checkpoint directory")?)
            }
        }
    }
}

/// Resolve `owner/model[:variant]`, list its GGUF files, select one checkpoint,
/// and download every shard into the standard Hugging Face cache.
pub fn download_gguf(model: &str) -> Result<HubGguf> {
    match download_checkpoint(model, false, None)? {
        HubCheckpoint::Gguf(gguf) => Ok(gguf),
        HubCheckpoint::Safetensors { .. } => unreachable!(),
    }
}

/// Prefer a model GGUF, otherwise resolve a Qwen safetensors checkpoint.
pub fn download_model(model: &str) -> Result<HubCheckpoint> {
    download_model_with_layer_limit(model, None)
}

/// Resolve a prefix of a Qwen3.5 safetensors model without fetching unused shards.
pub fn download_model_with_layer_limit(
    model: &str,
    num_layers: Option<usize>,
) -> Result<HubCheckpoint> {
    download_checkpoint(model, true, num_layers)
}

fn download_checkpoint(
    model: &str,
    allow_safetensors: bool,
    num_layers: Option<usize>,
) -> Result<HubCheckpoint> {
    let (repo_id, variant) = parse_model_selector(model)?;
    let (owner, name) = split_id(repo_id);
    ensure!(
        !owner.is_empty() && !name.is_empty(),
        "Hugging Face model must be written as owner/name, optionally followed by :variant"
    );

    let client = HFClientSync::new().context("initializing Hugging Face Hub client")?;
    let repo = client.model(owner, name);
    let entries = repo
        .list_tree()
        .recursive(true)
        .send()
        .with_context(|| format!("listing files for Hugging Face model {repo_id}"))?;
    let paths = entries
        .into_iter()
        .filter_map(|entry| match entry {
            RepoTreeEntry::File { path, .. } => Some(path),
            _ => None,
        })
        .collect::<Vec<_>>();
    let fetch = |path: &str| {
        repo.download_file()
            .filename(path)
            .send()
            .with_context(|| format!("downloading {repo_id}/{path}"))
    };
    let selectable = if num_layers.is_some() {
        ensure!(
            variant.is_none(),
            "--num-layers requires safetensors without a GGUF variant"
        );
        ensure!(
            paths
                .iter()
                .any(|p| p == "model.safetensors" || p == "model.safetensors.index.json"),
            "--num-layers requires a safetensors checkpoint"
        );
        paths
            .iter()
            .filter(|p| !is_model_gguf(p))
            .cloned()
            .collect()
    } else {
        paths.clone()
    };
    match select_model_files(&selectable, variant, allow_safetensors)
        .with_context(|| format!("selecting checkpoint from {repo_id}"))?
    {
        ModelFiles::Gguf(selected) => {
            download_gguf_files(&selected, fetch).map(HubCheckpoint::Gguf)
        }
        ModelFiles::Safetensors(checkpoint) => {
            let config_path = fetch("config.json")?;
            let mut config = crate::qwen_checkpoint::read_config(&config_path, repo_id)?;
            if let Some(count) = num_layers {
                limit_layers(&mut config, count)?;
            }
            let selection = num_layers
                .map(|_| crate::qwen_checkpoint::TensorSelection::new(&config))
                .transpose()?;
            let mut filenames = vec![checkpoint, "tokenizer.json", "config.json"];
            for name in ["tokenizer_config.json", "chat_template.jinja"] {
                if paths.iter().any(|path| path == name) {
                    filenames.push(name);
                }
            }
            let selected = |name: &str| selection.as_ref().is_none_or(|s| s.contains(name));
            let root = resolve_files_selected(
                &filenames,
                |name| {
                    if name == "config.json" {
                        Ok(config_path.clone())
                    } else {
                        fetch(name)
                    }
                },
                &selected,
            )?;
            Ok(HubCheckpoint::Safetensors {
                path: root.join(checkpoint),
                config: Box::new(config),
            })
        }
    }
}

fn limit_layers(config: &mut ModelConfig, count: usize) -> Result<()> {
    ensure!(
        config.is_qwen35(),
        "--num-layers currently requires Qwen3.5 dense or MoE text safetensors"
    );
    ensure!(
        count > 0 && count <= config.num_layers,
        "--num-layers must be between 1 and {} (got {count})",
        config.num_layers
    );
    eprintln!(
        "selected {count} of {} transformer layers",
        config.num_layers
    );
    config.num_layers = count;
    config.validate()
}

#[derive(Debug, PartialEq, Eq)]
enum ModelFiles {
    Gguf(Vec<String>),
    Safetensors(&'static str),
}

fn select_model_files(
    paths: &[String],
    variant: Option<&str>,
    allow_safetensors: bool,
) -> Result<ModelFiles> {
    let ggufs = paths
        .iter()
        .filter(|p| is_model_gguf(p))
        .cloned()
        .collect::<Vec<_>>();
    if !ggufs.is_empty() || !allow_safetensors || variant.is_some() {
        return select_checkpoint(&ggufs, variant).map(ModelFiles::Gguf);
    }
    ["model.safetensors", "model.safetensors.index.json"]
        .into_iter()
        .find(|name| paths.iter().any(|p| p == name))
        .map(ModelFiles::Safetensors)
        .context("repository contains no model GGUF or safetensors checkpoint")
}

fn download_gguf_files(
    selected: &[String],
    fetch: impl Fn(&str) -> Result<PathBuf>,
) -> Result<HubGguf> {
    let mut files = selected
        .iter()
        .map(|path| fetch(path))
        .collect::<Result<Vec<_>>>()?;
    files.sort();
    ensure!(
        !files.is_empty(),
        "selected GGUF checkpoint contained no files"
    );
    if files.len() > 1 {
        let parent = files[0].parent();
        ensure!(
            files.iter().all(|path| path.parent() == parent),
            "downloaded GGUF shards were not placed in one cache directory"
        );
    }

    Ok(HubGguf {
        primary: files[0].clone(),
        files,
    })
}

/// Select an unambiguous unquantized companion projector from the same repository.
pub fn download_mmproj(model: &str) -> Result<PathBuf> {
    let (repo_id, _) = parse_model_selector(model)?;
    let (owner, name) = split_id(repo_id);
    let client = HFClientSync::new().context("initializing Hugging Face Hub client")?;
    let repo = client.model(owner, name);
    let paths = repo
        .list_tree()
        .recursive(true)
        .send()?
        .into_iter()
        .filter_map(|e| match e {
            RepoTreeEntry::File { path, .. } if is_mmproj_gguf(&path) => Some(path),
            _ => None,
        })
        .collect::<Vec<_>>();
    let selected = select_mmproj(&paths)?;
    repo.download_file()
        .filename(selected)
        .send()
        .with_context(|| format!("downloading vision projector from {repo_id}"))
}
fn is_mmproj_gguf(path: &str) -> bool {
    has_gguf_extension(path) && !is_model_gguf(path)
}
fn select_mmproj(paths: &[String]) -> Result<String> {
    ensure!(
        !paths.is_empty(),
        "repository has no mmproj GGUF; provide --mmproj-model owner/repository"
    );
    let bf16: Vec<_> = paths
        .iter()
        .filter(|p| p.to_ascii_lowercase().contains("bf16"))
        .collect();
    let f32: Vec<_> = paths
        .iter()
        .filter(|p| p.to_ascii_lowercase().contains("f32"))
        .collect();
    let candidates = if !bf16.is_empty() {
        bf16
    } else if !f32.is_empty() {
        f32
    } else {
        paths.iter().collect()
    };
    ensure!(
        candidates.len() == 1,
        "projector selection is ambiguous: {}; provide --mmproj-model owner/repository",
        candidates
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );
    Ok(candidates[0].clone())
}

fn validate_model_name(model: &str) -> Result<()> {
    let parts = model.split('/').collect::<Vec<_>>();
    ensure!(
        parts.len() == 2
            && parts.iter().all(|part| {
                !part.is_empty()
                    && *part != "."
                    && *part != ".."
                    && part
                        .bytes()
                        .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
            }),
        "model must be an HF name owner/repository, not a filesystem path"
    );
    Ok(())
}

fn parse_model_selector(value: &str) -> Result<(&str, Option<&str>)> {
    let (repo, variant) = match value.rsplit_once(':') {
        Some((repo, variant)) => (repo, Some(variant)),
        None => (value, None),
    };
    validate_model_name(repo)?;
    if let Some(variant) = variant {
        ensure!(!variant.is_empty(), "Hugging Face GGUF variant is empty");
    }
    Ok((repo, variant))
}

fn has_gguf_extension(path: &str) -> bool {
    path.rsplit_once('.')
        .is_some_and(|(_, extension)| extension.eq_ignore_ascii_case("gguf"))
}

fn is_model_gguf(path: &str) -> bool {
    if !has_gguf_extension(path) {
        return false;
    }
    let filename = path.rsplit('/').next().unwrap_or(path).to_ascii_lowercase();
    !filename.starts_with("mmproj-") && filename != "mmproj.gguf"
}

fn select_checkpoint(paths: &[String], variant: Option<&str>) -> Result<Vec<String>> {
    ensure!(!paths.is_empty(), "repository contains no GGUF files");
    let candidates = if let Some(variant) = variant {
        let exact_directory = paths
            .iter()
            .filter(|path| {
                path.split('/')
                    .any(|part| part.eq_ignore_ascii_case(variant))
            })
            .cloned()
            .collect::<Vec<_>>();
        if !exact_directory.is_empty() {
            exact_directory
        } else {
            let needle = variant.to_ascii_lowercase();
            paths
                .iter()
                .filter(|path| path.to_ascii_lowercase().contains(&needle))
                .cloned()
                .collect()
        }
    } else {
        paths.to_vec()
    };
    if candidates.is_empty() {
        bail!(
            "no GGUF files match variant {:?}; available checkpoints: {}",
            variant,
            checkpoint_names(paths).join(", ")
        );
    }

    let groups = checkpoint_groups(&candidates)?;
    if groups.len() != 1 {
        bail!(
            "GGUF selection is ambiguous ({}); append :variant to the model name",
            groups.keys().cloned().collect::<Vec<_>>().join(", ")
        );
    }
    Ok(groups.into_values().next().expect("one group was checked"))
}

fn checkpoint_names(paths: &[String]) -> Vec<String> {
    checkpoint_groups(paths)
        .map(|groups| groups.into_keys().collect())
        .unwrap_or_default()
}

fn checkpoint_groups(paths: &[String]) -> Result<BTreeMap<String, Vec<String>>> {
    let mut groups = BTreeMap::<String, Vec<String>>::new();
    for path in paths {
        let key = shard_parts(path)
            .map(|(key, _, total)| format!("{key} ({total} shards)"))
            .unwrap_or_else(|| path.clone());
        groups.entry(key).or_default().push(path.clone());
    }
    for (name, files) in &mut groups {
        files.sort();
        if let Some((_, _, total)) = shard_parts(&files[0]) {
            ensure!(
                files.len() == total,
                "incomplete GGUF checkpoint {name}: found {} of {total} shards",
                files.len()
            );
            for (expected, file) in (1..=total).zip(files.iter()) {
                let (_, actual, actual_total) =
                    shard_parts(file).expect("all members of a shard group have shard suffixes");
                ensure!(
                    actual == expected && actual_total == total,
                    "incomplete or inconsistently named GGUF checkpoint {name}"
                );
            }
        }
    }
    Ok(groups)
}

/// Return `(path prefix, one-based shard index, shard count)` for the standard
/// `name-00001-of-00004.gguf` convention.
fn shard_parts(path: &str) -> Option<(String, usize, usize)> {
    let stem = path.get(..path.len().checked_sub(5)?)?;
    let (left, total) = stem.rsplit_once("-of-")?;
    let (prefix, index) = left.rsplit_once('-')?;
    if index.is_empty()
        || total.is_empty()
        || !index.bytes().all(|byte| byte.is_ascii_digit())
        || !total.bytes().all(|byte| byte.is_ascii_digit())
    {
        return None;
    }
    Some((prefix.to_owned(), index.parse().ok()?, total.parse().ok()?))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn paths(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    #[test]
    fn projector_selection_prefers_bf16_without_affecting_language_selection() {
        let paths = vec!["mmproj-F16.gguf".into(), "mmproj-BF16.gguf".into()];
        assert_eq!(select_mmproj(&paths).unwrap(), "mmproj-BF16.gguf");
        assert!(select_mmproj(&[]).is_err());
        assert!(
            select_mmproj(&["a/mmproj-BF16.gguf".into(), "b/mmproj-BF16.gguf".into()]).is_err()
        );
        assert!(is_mmproj_gguf("vision/mmproj.gguf"));
        assert!(!is_mmproj_gguf("model-BF16.gguf"));
    }

    #[test]
    fn parses_repo_and_optional_variant() -> Result<()> {
        assert_eq!(
            parse_model_selector("unsloth/Qwen3.5-0.8B-GGUF:BF16")?,
            ("unsloth/Qwen3.5-0.8B-GGUF", Some("BF16"))
        );
        assert_eq!(parse_model_selector("owner/model")?, ("owner/model", None));
        for path in [
            "/tmp/model.gguf",
            "../model",
            "./model",
            "~/model",
            "model.gguf",
            "owner/repo/nested",
        ] {
            assert!(parse_model_selector(path).is_err(), "accepted {path}");
        }
        Ok(())
    }

    #[test]
    fn selects_all_shards_in_variant_directory() -> Result<()> {
        let files = paths(&[
            "BF16/model-BF16-00002-of-00002.gguf",
            "Q8_0/model-Q8_0.gguf",
            "BF16/model-BF16-00001-of-00002.gguf",
        ]);
        assert_eq!(
            select_checkpoint(&files, Some("bf16"))?,
            paths(&[
                "BF16/model-BF16-00001-of-00002.gguf",
                "BF16/model-BF16-00002-of-00002.gguf",
            ])
        );
        Ok(())
    }

    #[test]
    fn rejects_ambiguous_quantizations() {
        let files = paths(&["model-BF16.gguf", "model-Q8_0.gguf"]);
        let error = select_checkpoint(&files, None).unwrap_err().to_string();
        assert!(error.contains("append :variant"));
    }

    #[test]
    fn rejects_incomplete_shard_set() {
        let files = paths(&["BF16/model-00001-of-00002.gguf"]);
        let error = select_checkpoint(&files, Some("BF16"))
            .unwrap_err()
            .to_string();
        assert!(error.contains("found 1 of 2 shards"));
    }

    #[test]
    fn excludes_multimodal_projector_ggufs() {
        assert!(is_model_gguf("Qwen3.5-0.8B-BF16.gguf"));
        assert!(!is_model_gguf("mmproj-BF16.gguf"));
        assert!(!is_model_gguf("vision/mmproj.gguf"));
    }
}

/// A safetensors index names its shards relative to the index file.
#[derive(serde::Deserialize)]
pub(crate) struct SafetensorsIndex {
    pub weight_map: BTreeMap<String, String>,
}
impl SafetensorsIndex {
    pub fn shards(&self) -> Result<Vec<String>> {
        ensure!(!self.weight_map.is_empty(), "empty safetensors index");
        let shards: std::collections::BTreeSet<_> = self.weight_map.values().cloned().collect();
        for name in &shards {
            relative_path(name)?;
        }
        Ok(shards.into_iter().collect())
    }
}
pub fn read_json<T: serde::de::DeserializeOwned>(path: impl AsRef<std::path::Path>) -> Result<T> {
    let path = path.as_ref();
    serde_json::from_reader(
        std::fs::File::open(path).with_context(|| format!("reading {}", path.display()))?,
    )
    .with_context(|| format!("parsing {}", path.display()))
}
fn relative_path(name: &str) -> Result<&std::path::Path> {
    use std::path::{Component, Path};
    let path = Path::new(name);
    ensure!(
        !name.is_empty() && path.components().all(|c| matches!(c, Component::Normal(_))),
        "unsafe model filename {name}"
    );
    Ok(path)
}

/// Resolve model-owned filenames using an HF name. Cache paths and revisions
/// are owned by hf-hub; safetensors indices bring their referenced shards.
pub fn resolve_model_files(model: &str, filenames: &[&str]) -> Result<PathBuf> {
    let client = HFClientSync::new().context("initializing Hugging Face Hub client")?;
    resolve_hub_files(&client, model, filenames)
}
#[cfg(test)]
fn resolve_local_files(
    _model: &str,
    filenames: &[&str],
    directory: Option<&std::path::Path>,
) -> Result<PathBuf> {
    let directory = directory.ok_or_else(|| anyhow::anyhow!("missing test directory"))?;
    resolve_files(filenames, |name| {
        let path = directory.join(name);
        ensure!(path.is_file(), "missing fixture {}", path.display());
        Ok(path)
    })
}
fn resolve_hub_files(client: &HFClientSync, model: &str, filenames: &[&str]) -> Result<PathBuf> {
    validate_model_name(model)?;
    let (owner, name) = split_id(model);
    ensure!(
        !owner.is_empty() && !name.is_empty(),
        "model must be owner/name"
    );
    let repo = client.model(owner, name);
    resolve_files(filenames, |filename| {
        repo.download_file()
            .filename(filename)
            .send()
            .with_context(|| format!("resolving {model}/{filename}"))
    })
}
fn resolve_files(
    filenames: &[&str],
    fetch: impl FnMut(&str) -> Result<PathBuf>,
) -> Result<PathBuf> {
    resolve_files_selected(filenames, fetch, &|_| true)
}
fn resolve_files_selected(
    filenames: &[&str],
    mut fetch: impl FnMut(&str) -> Result<PathBuf>,
    selected: &dyn Fn(&str) -> bool,
) -> Result<PathBuf> {
    use std::collections::BTreeSet;
    let mut pending: Vec<String> = filenames.iter().map(|s| s.to_string()).collect();
    let mut seen = BTreeSet::new();
    let mut root = None;
    while let Some(name) = pending.pop() {
        let relative = relative_path(&name)?;
        if !seen.insert(name.clone()) {
            continue;
        }
        let path = fetch(&name)?;
        let mut directory = path.as_path();
        for _ in relative.components() {
            directory = directory
                .parent()
                .context("model file has no snapshot root")?;
        }
        if let Some(existing) = &root {
            ensure!(
                existing == directory,
                "model files resolved to different snapshots; retry resolution"
            );
        } else {
            root = Some(directory.to_owned());
        }
        if name.ends_with(".safetensors.index.json") {
            let mut index: SafetensorsIndex = read_json(&path)?;
            index.shards()?;
            index.weight_map.retain(|name, _| selected(name));
            for shard in index.shards()? {
                pending.push(
                    relative
                        .parent()
                        .unwrap()
                        .join(shard)
                        .to_string_lossy()
                        .into_owned(),
                );
            }
        }
    }
    root.context("model file list is empty")
}

#[cfg(test)]
mod model_files_tests {
    use super::*;
    use std::fs;
    struct Temp(PathBuf);
    impl Temp {
        fn new(name: &str) -> Self {
            let p = std::env::temp_dir().join(format!("hub-{name}-{}", std::process::id()));
            fs::create_dir_all(&p).unwrap();
            Self(p)
        }
    }
    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    #[test]
    fn layer_limit_validates_and_keeps_the_original_schedule() -> Result<()> {
        for original in [ModelConfig::qwen35_0_8b(), ModelConfig::qwen36_35b_a3b()] {
            for n in [1, 3, 4, 5, original.num_layers] {
                let mut config = original.clone();
                limit_layers(&mut config, n)?;
                assert_eq!(config.num_layers, n);
                for l in 0..n {
                    assert_eq!(config.layer_type(l), original.layer_type(l));
                }
            }
            for n in [0, original.num_layers + 1] {
                assert!(limit_layers(&mut original.clone(), n).is_err());
            }
        }
        assert!(limit_layers(&mut ModelConfig::qwen3_0_6b(), 4).is_err());
        Ok(())
    }

    #[test]
    fn moe_selection_includes_experts_scales_and_fused_layouts() -> Result<()> {
        let mut config = ModelConfig::qwen36_35b_a3b();
        limit_layers(&mut config, 4)?;
        let selection = crate::qwen_checkpoint::TensorSelection::new(&config)?;
        for suffix in [
            "experts.0.gate_proj.weight",
            "experts.255.up_proj.weight_scale_inv",
            "experts.255.down_proj.weight_scale_inv",
            "experts.gate_up_proj",
            "experts.down_proj",
            "gate.weight",
            "shared_expert.gate_proj.weight_scale_inv",
        ] {
            assert!(
                selection.contains(&format!("model.language_model.layers.3.mlp.{suffix}")),
                "{suffix}"
            );
            assert!(
                !selection.contains(&format!("model.language_model.layers.4.mlp.{suffix}")),
                "{suffix}"
            );
        }
        assert!(selection.contains("model.language_model.embed_tokens.weight"));
        assert!(selection.contains("model.language_model.norm.weight"));
        assert!(selection.contains("lm_head.weight"));
        assert!(!selection.contains("model.visual.weight"));
        Ok(())
    }

    #[test]
    fn selected_index_fetches_only_required_shards_without_rewriting_it() -> Result<()> {
        let t = Temp::new("layer-selection");
        let mut config = ModelConfig::qwen35_0_8b();
        limit_layers(&mut config, 1)?;
        let selection = crate::qwen_checkpoint::TensorSelection::new(&config)?;
        assert!(selection.contains(
            "model.language_model.language_model.layers.0.linear_attn.in_proj_qkv.weight_scale_inv"
        ));
        assert!(!selection.contains("model.language_model.layers.1.input_layernorm.weight"));
        assert!(!selection.contains("model.visual.weight"));
        let index = serde_json::json!({"weight_map": {
            "model.language_model.embed_tokens.weight": "globals.safetensors",
            "model.language_model.norm.weight": "globals.safetensors",
            "lm_head.weight": "globals.safetensors",
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight": "shared.safetensors",
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight_scale_inv": "shared.safetensors",
            "model.language_model.layers.1.input_layernorm.weight": "shared.safetensors",
            "model.language_model.layers.23.input_layernorm.weight": "absent.safetensors",
            "model.visual.weight": "vision-absent.safetensors"
        }});
        let path = t.0.join("model.safetensors.index.json");
        fs::write(&path, serde_json::to_vec(&index)?)?;
        for name in ["globals.safetensors", "shared.safetensors"] {
            fs::write(t.0.join(name), b"fixture")?;
        }
        let mut fetched = Vec::new();
        resolve_files_selected(
            &["model.safetensors.index.json"],
            |name| {
                fetched.push(name.to_owned());
                let path = t.0.join(name);
                ensure!(path.is_file(), "unexpected fetch {name}");
                Ok(path)
            },
            &|name| selection.contains(name),
        )?;
        fetched.sort();
        assert_eq!(
            fetched,
            [
                "globals.safetensors",
                "model.safetensors.index.json",
                "shared.safetensors"
            ]
        );
        assert_eq!(read_json::<serde_json::Value>(&path)?, index);
        assert!(
            resolve_local_files(
                "unused/model",
                &["model.safetensors.index.json"],
                Some(&t.0)
            )
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn selected_loader_does_not_require_omitted_shards_or_expose_omitted_tensors() -> Result<()> {
        use safetensors::{
            Dtype,
            tensor::{TensorView, serialize_to_file},
        };
        let t = Temp::new("selected-loader");
        let path = t.0.join("shared.safetensors");
        serialize_to_file(
            [
                ("keep", TensorView::new(Dtype::BF16, vec![1], &[0, 0x3f])?),
                ("skip", TensorView::new(Dtype::I64, vec![1], &[255; 8])?),
            ],
            None,
            &path,
        )?;
        let index = t.0.join("model.safetensors.index.json");
        fs::write(
            &index,
            serde_json::to_vec(&serde_json::json!({"weight_map": {
                "keep": "shared.safetensors", "skip": "shared.safetensors", "absent": "absent.safetensors"
            }}))?,
        )?;
        for path in [&path, &index] {
            let weights = crate::weights::Weights::open_selected(path, &|name| name == "keep")?;
            assert_eq!(
                weights
                    .tensors
                    .keys()
                    .map(String::as_str)
                    .collect::<Vec<_>>(),
                ["keep"]
            );
            assert_eq!(weights.host_bf16("keep")?, [0x3f00]);
            assert!(weights.host_bf16("skip").is_err());
        }
        assert!(crate::weights::Weights::open(&index).is_err());
        let bad = t.0.join("bad.index.json");
        fs::write(
            &bad,
            r#"{"weight_map":{"keep":"shared.safetensors","missing":"shared.safetensors"}}"#,
        )?;
        assert!(crate::weights::Weights::open_selected(&bad, &|_| true).is_err());
        Ok(())
    }

    #[test]
    fn prefers_gguf_and_keeps_its_selection_errors() -> Result<()> {
        let mut paths = vec![
            "mmproj-BF16.gguf".into(),
            "model.safetensors.index.json".into(),
        ];
        assert_eq!(
            select_model_files(&paths, None, true)?,
            ModelFiles::Safetensors("model.safetensors.index.json")
        );
        paths.push("model.safetensors".into());
        assert_eq!(
            select_model_files(&paths, None, true)?,
            ModelFiles::Safetensors("model.safetensors")
        );
        assert!(select_model_files(&paths, Some("BF16"), true).is_err());
        paths.push("model-BF16.gguf".into());
        assert_eq!(
            select_model_files(&paths, None, true)?,
            ModelFiles::Gguf(vec!["model-BF16.gguf".into()])
        );
        paths.push("model-F32-00001-of-00002.gguf".into());
        assert!(select_model_files(&paths, Some("F32"), true).is_err());
        assert!(select_model_files(&paths, Some("missing"), true).is_err());
        Ok(())
    }
    #[test]
    fn local_files_expand_indexes_and_report_missing_files() -> Result<()> {
        let t = Temp::new("local");
        fs::create_dir_all(t.0.join("encoder"))?;
        fs::write(
            t.0.join("encoder/model.safetensors.index.json"),
            r#"{"weight_map":{"a":"one.safetensors","b":"two.safetensors"}}"#,
        )?;
        fs::write(t.0.join("encoder/one.safetensors"), b"one")?;
        fs::write(t.0.join("encoder/two.safetensors"), b"two")?;
        assert_eq!(
            resolve_local_files(
                "unused/model",
                &["encoder/model.safetensors.index.json"],
                Some(&t.0)
            )?,
            t.0
        );
        fs::remove_file(t.0.join("encoder/two.safetensors"))?;
        assert!(
            resolve_local_files(
                "unused/model",
                &["encoder/model.safetensors.index.json"],
                Some(&t.0)
            )
            .is_err()
        );
        assert!(resolve_local_files("unused/model", &["../secret"], Some(&t.0)).is_err());
        assert!(resolve_local_files("unused/model", &[], Some(&t.0)).is_err());
        Ok(())
    }
    #[test]
    fn snapshot_mismatch_and_bad_index_are_errors() -> Result<()> {
        assert!(
            resolve_files(&["a.json", "b.json"], |name| Ok(PathBuf::from(format!(
                "/{name}/{name}"
            ))))
            .is_err()
        );
        let t = Temp::new("bad-index");
        for text in [
            "not json",
            r#"{"weight_map":{}}"#,
            r#"{"weight_map":{"a":"../bad.safetensors"}}"#,
        ] {
            fs::write(t.0.join("model.safetensors.index.json"), text)?;
            assert!(
                resolve_local_files("owner/model", &["model.safetensors.index.json"], Some(&t.0))
                    .is_err()
            );
        }
        Ok(())
    }
}
