use std::{collections::BTreeMap, path::PathBuf};

use anyhow::{Context, Result, bail, ensure};
use hf_hub::{HFClientSync, repository::RepoTreeEntry, split_id};

/// Local GGUF files selected and cached for a Hub model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HubGguf {
    /// A file accepted by the existing GGUF shard resolver. For a sharded model
    /// this is shard one; all remaining shards are downloaded beside it.
    pub primary: PathBuf,
    pub files: Vec<PathBuf>,
}

/// Resolve `owner/model[:variant]`, list its GGUF files, select one checkpoint,
/// and download every shard into the standard Hugging Face cache.
pub fn download_gguf(model: &str) -> Result<HubGguf> {
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
    let ggufs = entries
        .into_iter()
        .filter_map(|entry| match entry {
            RepoTreeEntry::File { path, .. } if is_model_gguf(&path) => Some(path),
            _ => None,
        })
        .collect::<Vec<_>>();
    let selected = select_checkpoint(&ggufs, variant)
        .with_context(|| format!("selecting GGUF checkpoint from {repo_id}"))?;

    let mut files = Vec::with_capacity(selected.len());
    for remote_path in selected {
        let local_path = repo
            .download_file()
            .filename(remote_path.clone())
            .send()
            .with_context(|| format!("downloading {repo_id}/{remote_path}"))?;
        files.push(local_path);
    }
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

fn parse_model_selector(value: &str) -> Result<(&str, Option<&str>)> {
    let (repo, variant) = match value.rsplit_once(':') {
        Some((repo, variant)) => (repo, Some(variant)),
        None => (value, None),
    };
    ensure!(!repo.is_empty(), "Hugging Face model name is empty");
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
    fn parses_repo_and_optional_variant() -> Result<()> {
        assert_eq!(
            parse_model_selector("unsloth/Qwen3.5-0.8B-GGUF:BF16")?,
            ("unsloth/Qwen3.5-0.8B-GGUF", Some("BF16"))
        );
        assert_eq!(parse_model_selector("owner/model")?, ("owner/model", None));
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
