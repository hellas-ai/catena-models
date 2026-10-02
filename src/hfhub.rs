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

/// Resolve model-owned filenames on the Hub's default branch, or in an explicit
/// local directory. Hub caching, authentication and offline fallback remain
/// owned by hf-hub. Index files automatically bring their referenced shards.
pub fn resolve_model_files(
    model: &str,
    filenames: &[&str],
    model_dir: Option<&std::path::Path>,
) -> Result<PathBuf> {
    if let Some(dir) = model_dir {
        return resolve_files(filenames, |name| {
            let path = dir.join(name);
            ensure!(path.is_file(), "missing model file {}", path.display());
            Ok(path)
        });
    }
    let client = HFClientSync::new().context("initializing Hugging Face Hub client")?;
    resolve_hub_files(&client, model, filenames)
}
fn resolve_hub_files(client: &HFClientSync, model: &str, filenames: &[&str]) -> Result<PathBuf> {
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
    mut fetch: impl FnMut(&str) -> Result<PathBuf>,
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
            let index: SafetensorsIndex = read_json(&path)?;
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
    use std::{
        fs,
        io::{Read, Write},
        net::TcpListener,
        sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        },
        thread,
        time::Duration,
    };
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
            resolve_model_files(
                "unused/model",
                &["encoder/model.safetensors.index.json"],
                Some(&t.0)
            )?,
            t.0
        );
        fs::remove_file(t.0.join("encoder/two.safetensors"))?;
        assert!(
            resolve_model_files(
                "unused/model",
                &["encoder/model.safetensors.index.json"],
                Some(&t.0)
            )
            .is_err()
        );
        assert!(resolve_model_files("unused/model", &["../secret"], Some(&t.0)).is_err());
        assert!(resolve_model_files("unused/model", &[], Some(&t.0)).is_err());
        Ok(())
    }
    #[test]
    fn named_downloads_use_main_then_reuse_cache_without_public_network() -> Result<()> {
        let cache = Temp::new("download");
        let listener = TcpListener::bind("127.0.0.1:0")?;
        listener.set_nonblocking(true)?;
        let endpoint = format!("http://{}", listener.local_addr()?);
        let stop = Arc::new(AtomicBool::new(false));
        let worker_stop = stop.clone();
        let server = thread::spawn(move || -> Result<()> {
            while !worker_stop.load(Ordering::Relaxed) {
                let (mut socket, _) = match listener.accept() {
                    Ok(pair) => pair,
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(2));
                        continue;
                    }
                    Err(e) => return Err(e.into()),
                };
                socket.set_read_timeout(Some(Duration::from_secs(2)))?;
                let mut request = Vec::new();
                let mut buffer = [0; 1024];
                while !request.windows(4).any(|w| w == b"\r\n\r\n") {
                    let n = socket.read(&mut buffer)?;
                    if n == 0 {
                        break;
                    }
                    request.extend_from_slice(&buffer[..n]);
                }
                let request = String::from_utf8(request)?;
                let first = request.lines().next().unwrap();
                assert!(
                    first.contains("/owner/model/resolve/main/"),
                    "unexpected request {first}"
                );
                let path = first.split_whitespace().nth(1).unwrap();
                let (etag, body) = if path.ends_with("config.json") {
                    ("config", b"{}".as_slice())
                } else if path.ends_with("encoder/model.safetensors.index.json") {
                    (
                        "index",
                        br#"{"weight_map":{"a":"one.safetensors"}}"#.as_slice(),
                    )
                } else if path.ends_with("encoder/one.safetensors") {
                    ("one", b"tensor".as_slice())
                } else {
                    panic!("unexpected model file {path}")
                };
                let head = first.starts_with("HEAD ");
                write!(
                    socket,
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nETag: \"{etag}\"\r\nX-Repo-Commit: 0123456789012345678901234567890123456789\r\nConnection: close\r\n\r\n",
                    body.len()
                )?;
                if !head {
                    socket.write_all(body)?;
                }
            }
            Ok(())
        });
        let client = hf_hub::HFClient::builder()
            .cache_dir(&cache.0)
            .endpoint(&endpoint)
            .retry_max_attempts(1)
            .build_sync()?;
        let files = ["config.json", "encoder/model.safetensors.index.json"];
        let result = resolve_hub_files(&client, "owner/model", &files);
        stop.store(true, Ordering::Relaxed);
        server.join().unwrap()?;
        let snapshot = result?;
        assert_eq!(
            fs::read(snapshot.join("encoder/one.safetensors"))?,
            b"tensor"
        );
        // The same endpoint is now unavailable. hf-hub owns the cache fallback.
        assert_eq!(resolve_hub_files(&client, "owner/model", &files)?, snapshot);
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
                resolve_model_files("owner/model", &["model.safetensors.index.json"], Some(&t.0))
                    .is_err()
            );
        }
        Ok(())
    }
}
