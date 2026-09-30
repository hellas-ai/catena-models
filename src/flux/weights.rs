//! Validated, mmap-backed safetensors with GPU borrows tied to their storage.
use crate::checkpoint::GpuRegistration;
use anyhow::{Context, Result, bail, ensure};
use catena_lang::{
    codegen::GpuDialect,
    runtime::{MemOwn, MemRef, Runtime},
};
use memmap2::{Mmap, MmapOptions};
use safetensors::{Dtype, SafeTensors};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    path::{Path, PathBuf},
};

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub shape: Vec<usize>,
    pub dtype: Dtype,
    file: usize,
    offset: usize,
    len: usize,
}
struct Shard {
    registration: Option<GpuRegistration>,
    bytes: Mmap,
}
pub struct Weights {
    shards: Vec<Shard>,
    pub tensors: BTreeMap<String, TensorInfo>,
    converted: BTreeMap<String, MemOwn>,
}
#[derive(serde::Deserialize)]
struct Index {
    weight_map: BTreeMap<String, String>,
}
impl Weights {
    pub fn open(component: &Path) -> Result<Self> {
        let start = std::time::Instant::now();
        let index = component.join("model.safetensors.index.json");
        let indexed: Option<Index> = if index.exists() {
            Some(serde_json::from_reader(File::open(index)?)?)
        } else {
            None
        };
        let paths: Vec<PathBuf> = if let Some(i) = &indexed {
            let names: BTreeSet<_> = i.weight_map.values().collect();
            names
                .into_iter()
                .map(|n| {
                    ensure!(
                        Path::new(n).components().count() == 1,
                        "unsafe shard path {n}"
                    );
                    Ok(component.join(n))
                })
                .collect::<Result<_>>()?
        } else {
            let model = component.join("model.safetensors");
            vec![if model.exists() {
                model
            } else {
                component.join("diffusion_pytorch_model.safetensors")
            }]
        };
        let mut shards = Vec::new();
        let mut tensors = BTreeMap::new();
        for path in &paths {
            let file = File::open(path)
                .with_context(|| format!("missing checkpoint {}", path.display()))?;
            // SAFETY: model files are immutable for the lifetime of this loader.
            let bytes = unsafe { MmapOptions::new().map(&file)? };
            let (header, metadata) = SafeTensors::read_metadata(&bytes)
                .with_context(|| format!("invalid safetensors {}", path.display()))?;
            for (name, t) in metadata.tensors() {
                ensure!(!tensors.contains_key(&name), "duplicate tensor {name}");
                if let Some(i) = &indexed {
                    let declared = i
                        .weight_map
                        .get(&name)
                        .with_context(|| format!("unindexed tensor {name}"))?;
                    ensure!(
                        Path::new(declared).file_name() == path.file_name(),
                        "shard index mismatch for {name}"
                    );
                }
                tensors.insert(
                    name,
                    TensorInfo {
                        shape: t.shape.clone(),
                        dtype: t.dtype,
                        file: shards.len(),
                        offset: header + 8 + t.data_offsets.0,
                        len: t.data_offsets.1 - t.data_offsets.0,
                    },
                );
            }
            shards.push(Shard {
                registration: None,
                bytes,
            });
        }
        if let Some(i) = indexed {
            ensure!(
                i.weight_map.len() == tensors.len()
                    && i.weight_map.keys().all(|k| tensors.contains_key(k)),
                "incomplete shard index"
            );
        }
        super::ops::startup_timing(
            &format!("{} checkpoint metadata/mmap", component.display()),
            start,
        );
        Ok(Self {
            shards,
            tensors,
            converted: BTreeMap::new(),
        })
    }
    pub fn validate(&self, name: &str, shape: &[usize]) -> Result<()> {
        let t = self
            .tensors
            .get(name)
            .with_context(|| format!("missing tensor {name}"))?;
        ensure!(
            t.shape == shape,
            "{name}: expected shape {shape:?}, got {:?}",
            t.shape
        );
        ensure!(
            matches!(t.dtype, Dtype::BF16 | Dtype::F32),
            "{name}: unsupported dtype {:?}",
            t.dtype
        );
        Ok(())
    }
    pub fn host_f32(&self, name: &str) -> Result<Vec<f32>> {
        let t = self
            .tensors
            .get(name)
            .with_context(|| format!("missing tensor {name}"))?;
        let bytes = &self.shards[t.file].bytes[t.offset..t.offset + t.len];
        match t.dtype {
            Dtype::BF16 => Ok(bytes
                .chunks_exact(2)
                .map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16))
                .collect()),
            Dtype::F32 => Ok(bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect()),
            d => bail!("cannot convert {name} from {d:?}"),
        }
    }
    pub fn prepare(
        &mut self,
        runtime: &Runtime,
        names: &[String],
        f32_storage: bool,
    ) -> Result<()> {
        let start = std::time::Instant::now();
        for name in names {
            let t = self
                .tensors
                .get(name)
                .with_context(|| format!("missing tensor {name}"))?;
            if f32_storage {
                self.converted
                    .insert(name.clone(), runtime.mem_f32(&self.host_f32(name)?)?);
            } else {
                ensure!(t.dtype == Dtype::BF16, "{name}: expected BF16");
                // Matrix kernels require naturally aligned scalars, not a GGUF-specific tensor alignment.
                if t.offset % 2 != 0 {
                    let bytes = &self.shards[t.file].bytes[t.offset..t.offset + t.len];
                    let values: Vec<_> = bytes
                        .chunks_exact(2)
                        .map(|b| u16::from_le_bytes([b[0], b[1]]))
                        .collect();
                    self.converted
                        .insert(name.clone(), runtime.mem_u16(&values)?);
                } else if self.shards[t.file].registration.is_none() {
                    self.shards[t.file].registration = Some(GpuRegistration::new(
                        &self.shards[t.file].bytes,
                        GpuDialect::Hip,
                    )?);
                }
            }
        }
        super::ops::startup_timing(
            if f32_storage {
                "weights F32 conversion/upload"
            } else {
                "weights BF16 registration/upload"
            },
            start,
        );
        Ok(())
    }
    pub fn gpu_ref(&self, name: &str) -> Result<MemRef<'_>> {
        if let Some(m) = self.converted.get(name) {
            return Ok(m.as_ref());
        }
        let t = self
            .tensors
            .get(name)
            .with_context(|| format!("missing tensor {name}"))?;
        self.shards[t.file]
            .registration
            .as_ref()
            .context("weights not prepared")?
            .reference(t.offset, t.len)
    }
    pub fn byte_len(&self) -> usize {
        self.shards.iter().map(|s| s.bytes.len()).sum()
    }
}
