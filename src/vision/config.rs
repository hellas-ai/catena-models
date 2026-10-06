use crate::{
    checkpoint::{self, GgufDtype},
    spec::{TensorSpec, WeightDtype},
};
use anyhow::{Context, Result, ensure};
use serde_json::Value;
use std::{collections::HashMap, path::Path};

#[derive(Clone, Debug)]
pub struct VisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub output_size: usize,
    pub layers: usize,
    pub heads: usize,
    pub patch_size: usize,
    pub merge_size: usize,
    pub position_side: usize,
    pub eps: f32,
    pub min_pixels: usize,
    pub max_pixels: usize,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}
impl VisionConfig {
    pub fn from_gguf(path: &Path) -> Result<Self> {
        Self::from_metadata(&checkpoint::read_metadata(path).context("expected BF16/F32 vision GGUF; convert F16/quantized projectors with --mmproj --outtype bf16")?)
    }
    fn from_metadata(m: &HashMap<String, Value>) -> Result<Self> {
        let projector = m
            .get("clip.vision.projector_type")
            .or_else(|| m.get("clip.projector_type"));
        ensure!(
            projector.and_then(Value::as_str) == Some("qwen3vl_merger"),
            "expected Qwen3.5 qwen3vl_merger projector"
        );
        ensure!(
            m.get("clip.vision.is_deepstack_layers")
                .and_then(Value::as_array)
                .is_none_or(|a| a.iter().all(|v| v.as_bool() == Some(false))),
            "DeepStack projectors are unsupported"
        );
        let u = |key: &str| -> Result<usize> {
            usize::try_from(
                m.get(key)
                    .and_then(Value::as_u64)
                    .with_context(|| format!("missing integer {key}"))?,
            )
            .map_err(Into::into)
        };
        let optional = |key: &str, default| -> Result<usize> {
            if m.contains_key(key) {
                u(key)
            } else {
                Ok(default)
            }
        };
        let triple = |key: &str| -> Result<[f32; 3]> {
            let a = m
                .get(key)
                .and_then(Value::as_array)
                .with_context(|| format!("missing {key}"))?;
            ensure!(a.len() == 3, "{key} requires three channels");
            let mut out = [0.; 3];
            for i in 0..3 {
                out[i] = a[i].as_f64().context("invalid channel value")? as f32;
            }
            Ok(out)
        };
        let patch_size = u("clip.vision.patch_size")?;
        let image_size = u("clip.vision.image_size")?;
        let config = Self {
            hidden_size: u("clip.vision.embedding_length")?,
            intermediate_size: u("clip.vision.feed_forward_length")?,
            output_size: u("clip.vision.projection_dim")?,
            layers: u("clip.vision.block_count")?,
            heads: u("clip.vision.attention.head_count")?,
            patch_size,
            merge_size: optional("clip.vision.spatial_merge_size", 2)?,
            position_side: image_size
                .checked_div(patch_size)
                .context("zero patch size")?,
            eps: m
                .get("clip.vision.attention.layer_norm_epsilon")
                .and_then(Value::as_f64)
                .unwrap_or(1e-6) as f32,
            min_pixels: optional("clip.vision.image_min_pixels", 65536)?,
            max_pixels: optional("clip.vision.image_max_pixels", 16777216)?,
            mean: triple("clip.vision.image_mean")?,
            std: triple("clip.vision.image_std")?,
        };
        config.validate()?;
        ensure!(
            image_size.is_multiple_of(patch_size),
            "invalid learned position grid"
        );
        Ok(config)
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(
            [
                self.hidden_size,
                self.intermediate_size,
                self.output_size,
                self.layers,
                self.heads,
                self.patch_size,
                self.position_side
            ]
            .iter()
            .all(|&n| n > 0),
            "zero vision dimension"
        );
        ensure!(
            self.merge_size == 2,
            "only 2x2 spatial merging is supported"
        );
        ensure!(
            self.hidden_size.is_multiple_of(self.heads)
                && (self.hidden_size / self.heads).is_multiple_of(4),
            "invalid vision head dimensions"
        );
        ensure!(
            self.min_pixels > 0 && self.min_pixels <= self.max_pixels,
            "invalid image pixel limits"
        );
        ensure!(
            self.eps.is_finite()
                && self.eps > 0.
                && self.mean.iter().all(|v| v.is_finite())
                && self.std.iter().all(|v| v.is_finite() && *v > 0.),
            "invalid image normalization"
        );
        Ok(())
    }
    pub(crate) fn weight_specs(&self, path: &Path) -> Result<Vec<TensorSpec>> {
        let inventory = checkpoint::tensor_inventory(path)?;
        let h = self.hidden_size;
        let f = self.intermediate_size;
        let merged = h * self.merge_size * self.merge_size;
        let mut expected = vec![
            (
                "v.patch_embd.weight".into(),
                vec![h, 3, self.patch_size, self.patch_size],
            ),
            (
                "v.patch_embd.weight.1".into(),
                vec![h, 3, self.patch_size, self.patch_size],
            ),
            ("v.patch_embd.bias".into(), vec![h]),
            (
                "v.position_embd.weight".into(),
                vec![self.position_side * self.position_side, h],
            ),
            ("v.post_ln.weight".into(), vec![h]),
            ("v.post_ln.bias".into(), vec![h]),
            ("mm.0.weight".into(), vec![merged, merged]),
            ("mm.0.bias".into(), vec![merged]),
            ("mm.2.weight".into(), vec![self.output_size, merged]),
            ("mm.2.bias".into(), vec![self.output_size]),
        ];
        for l in 0..self.layers {
            for (name, shape) in [
                ("ln1", vec![h]),
                ("ln2", vec![h]),
                ("attn_qkv", vec![3 * h, h]),
                ("attn_out", vec![h, h]),
                ("ffn_up", vec![f, h]),
                ("ffn_down", vec![h, f]),
            ] {
                let bias = vec![shape[0]];
                expected.push((format!("v.blk.{l}.{name}.weight"), shape));
                expected.push((format!("v.blk.{l}.{name}.bias"), bias));
            }
        }
        expected
            .into_iter()
            .map(|(name, shape)| {
                let (actual, dtype) = inventory
                    .get(&name)
                    .with_context(|| format!("missing vision tensor {name}"))?;
                ensure!(
                    *actual == shape,
                    "{name}: expected {shape:?}, found {actual:?}"
                );
                Ok(TensorSpec {
                    name: name.clone(),
                    gguf_name: name,
                    dtype: match dtype {
                        GgufDtype::F32 => WeightDtype::F32,
                        GgufDtype::Bf16 => WeightDtype::Bf16,
                    },
                    shape,
                    sources: vec![],
                })
            })
            .collect()
    }
}
