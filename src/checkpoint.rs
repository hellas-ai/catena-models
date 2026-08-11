use std::{
    collections::{BTreeSet, HashMap},
    fs::File,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail, ensure};
use memmap2::{Mmap, MmapOptions};
use safetensors::{Dtype, SafeTensors, tensor::Metadata};
use serde::Deserialize;

use crate::spec::{TensorSource, TensorSpec, weight_specs};

#[derive(Deserialize)]
struct CheckpointIndex {
    weight_map: HashMap<String, String>,
}

struct Shard {
    path: PathBuf,
    bytes: Mmap,
    data_start: usize,
    metadata: Metadata,
}

/// Memory-mapped sharded Qwen checkpoint. Metadata is parsed once per shard;
/// BF16 payload bytes are copied directly into device-upload vectors.
pub struct Checkpoint {
    model_dir: PathBuf,
    weight_map: HashMap<String, String>,
    shards: HashMap<String, Shard>,
}

impl Checkpoint {
    pub fn open(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref().to_path_buf();
        let index_path = model_dir.join("model.safetensors.index.json");
        let index_bytes = std::fs::read(&index_path)
            .with_context(|| format!("failed to read {}", index_path.display()))?;
        let index: CheckpointIndex = serde_json::from_slice(&index_bytes)
            .with_context(|| format!("failed to parse {}", index_path.display()))?;

        let shard_names = index
            .weight_map
            .values()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        let mut shards = HashMap::with_capacity(shard_names.len());
        for shard_name in shard_names {
            let path = model_dir.join(shard_name);
            let file = File::open(&path)
                .with_context(|| format!("failed to open checkpoint shard {}", path.display()))?;
            // SAFETY: mappings are read-only and checkpoint files must remain immutable while loaded.
            let bytes = unsafe { MmapOptions::new().map(&file) }
                .with_context(|| format!("failed to map checkpoint shard {}", path.display()))?;
            let (header_len, metadata) = SafeTensors::read_metadata(&bytes)
                .with_context(|| format!("failed to parse checkpoint shard {}", path.display()))?;
            shards.insert(
                shard_name.to_owned(),
                Shard {
                    path,
                    bytes,
                    data_start: 8 + header_len,
                    metadata,
                },
            );
        }

        let checkpoint = Self {
            model_dir,
            weight_map: index.weight_map,
            shards,
        };
        checkpoint.validate()?;
        Ok(checkpoint)
    }

    pub fn validate(&self) -> Result<()> {
        let specs = weight_specs();
        let expected_names = specs
            .iter()
            .flat_map(|spec| spec.sources.iter().map(|source| source.name.as_str()))
            .collect::<BTreeSet<_>>();
        let actual_names = self
            .weight_map
            .keys()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        if expected_names != actual_names {
            let missing = expected_names
                .difference(&actual_names)
                .take(16)
                .copied()
                .collect::<Vec<_>>();
            let unexpected = actual_names
                .difference(&expected_names)
                .take(16)
                .copied()
                .collect::<Vec<_>>();
            bail!(
                "checkpoint tensor inventory mismatch; missing(first 16)={missing:?}, unexpected(first 16)={unexpected:?}"
            );
        }

        for spec in &specs {
            for source in &spec.sources {
                self.validate_source(source)?;
            }
        }
        Ok(())
    }

    /// Load one device ABI buffer as raw little-endian BF16 words. Grouped
    /// expert specs concatenate experts in numeric expert order.
    pub fn load_bf16(&self, spec: &TensorSpec) -> Result<Vec<u16>> {
        let mut values = Vec::with_capacity(spec.element_count());
        for source in &spec.sources {
            let (shard, info) = self.source(source)?;
            let start = shard.data_start + info.data_offsets.0;
            let end = shard.data_start + info.data_offsets.1;
            let bytes = shard.bytes.get(start..end).with_context(|| {
                format!(
                    "tensor {} has invalid offsets in {}",
                    source.name,
                    shard.path.display()
                )
            })?;
            ensure!(
                bytes.len() == source.shape.iter().product::<usize>() * 2,
                "tensor {} has an unexpected byte count",
                source.name
            );
            values.extend(
                bytes
                    .chunks_exact(2)
                    .map(|word| u16::from_le_bytes([word[0], word[1]])),
            );
        }
        ensure!(
            values.len() == spec.element_count(),
            "packed tensor {} has {} elements, expected {}",
            spec.name,
            values.len(),
            spec.element_count()
        );
        Ok(values)
    }

    fn validate_source(&self, source: &TensorSource) -> Result<()> {
        let (_shard, info) = self.source(source)?;
        ensure!(
            info.dtype == Dtype::BF16,
            "tensor {} has dtype {:?}, expected BF16",
            source.name,
            info.dtype
        );
        ensure!(
            info.shape == source.shape,
            "tensor {} has shape {:?}, expected {:?}",
            source.name,
            info.shape,
            source.shape
        );
        Ok(())
    }

    fn source(&self, source: &TensorSource) -> Result<(&Shard, &safetensors::tensor::TensorInfo)> {
        let shard_name = self
            .weight_map
            .get(&source.name)
            .with_context(|| format!("checkpoint index is missing {}", source.name))?;
        let shard = self.shards.get(shard_name).with_context(|| {
            format!(
                "checkpoint index points {} at unopened shard {} in {}",
                source.name,
                shard_name,
                self.model_dir.display()
            )
        })?;
        let info = shard
            .metadata
            .info(&source.name)
            .with_context(|| format!("{} is missing from {}", source.name, shard.path.display()))?;
        Ok((shard, info))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_downloaded_checkpoint_when_present() -> Result<()> {
        let path = Path::new("models/Qwen3-30B-A3B-Base");
        if path.exists() {
            Checkpoint::open(path)?;
        }
        Ok(())
    }

    #[test]
    fn loads_raw_bf16_and_grouped_experts_when_present() -> Result<()> {
        let path = Path::new("models/Qwen3-30B-A3B-Base");
        if path.exists() {
            let checkpoint = Checkpoint::open(path)?;
            let specs = weight_specs();
            let norm = checkpoint.load_bf16(&specs[1])?;
            assert_eq!(norm.len(), 2_048);
            let experts = checkpoint.load_bf16(&specs[10])?;
            assert_eq!(experts.len(), 128 * 768 * 2_048);
        }
        Ok(())
    }
}
