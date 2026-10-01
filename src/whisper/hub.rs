//! Pinned Whisper snapshot in the standard Hugging Face cache.
use anyhow::{Context, Result, ensure};
use hf_hub::HFClientSync;
use std::path::{Path, PathBuf};

pub const MODEL_ID: &str = "openai/whisper-small";
pub const REVISION: &str = "973afd24965f72e36ca33b3055d56a652f456b4d";
const FILES: &[&str] = &[
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "model.safetensors",
];

/// An explicit directory bypasses the Hub. Otherwise cached files at the pinned
/// commit are reused without network requests, and missing files are downloaded.
pub fn snapshot(explicit: Option<&Path>) -> Result<PathBuf> {
    if let Some(path) = explicit {
        return Ok(path.to_owned());
    }
    let client = HFClientSync::new().context("initializing Hugging Face Hub client")?;
    download_snapshot(&client)
}

fn download_snapshot(client: &HFClientSync) -> Result<PathBuf> {
    let repo = client.model("openai", "whisper-small");
    let mut directory = None;
    for &filename in FILES {
        let path = repo
            .download_file()
            .filename(filename)
            .revision(REVISION)
            .send()
            .with_context(|| format!("resolving {MODEL_ID}/{filename} at {REVISION}"))?;
        let parent = path
            .parent()
            .context("cached file has no snapshot directory")?;
        if let Some(dir) = &directory {
            ensure!(
                dir == parent,
                "Whisper files resolved to different snapshots"
            );
        } else {
            directory = Some(parent.to_owned());
        }
    }
    directory.context("Whisper snapshot has no files")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pinned_snapshot_reuses_hub_cache_without_network() -> Result<()> {
        let cache = std::env::temp_dir().join(format!("whisper-hub-test-{}", std::process::id()));
        let snapshot = cache
            .join("models--openai--whisper-small/snapshots")
            .join(REVISION);
        std::fs::create_dir_all(&snapshot)?;
        for &filename in FILES {
            std::fs::write(snapshot.join(filename), filename)?;
        }
        let client = hf_hub::HFClient::builder()
            .cache_dir(&cache)
            .endpoint("http://127.0.0.1:1")
            .build_sync()?;
        let result = download_snapshot(&client);
        assert_eq!(result?, snapshot);
        for &filename in FILES {
            assert_eq!(std::fs::read_to_string(snapshot.join(filename))?, filename);
        }
        std::fs::remove_dir_all(cache)?;
        Ok(())
    }
    #[test]
    fn explicit_directory_bypasses_hub() -> Result<()> {
        assert_eq!(
            snapshot(Some(Path::new("custom-whisper")))?,
            PathBuf::from("custom-whisper")
        );
        Ok(())
    }
}
