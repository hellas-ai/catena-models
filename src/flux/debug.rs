//! Optional safetensors activation dumps for component-level reference comparisons.
use anyhow::{Result, ensure};
use catena_lang::runtime::MemOwn;
use safetensors::{
    Dtype,
    tensor::{TensorView, serialize_to_file},
};
use std::{fs, path::Path};
pub fn dump(dir: Option<&Path>, name: &str, m: &MemOwn, shape: &[usize], bf16: bool) -> Result<()> {
    if let Some(dir) = dir {
        fs::create_dir_all(dir)?;
        let bytes: Vec<u8> = if bf16 {
            m.try_to_u16_vec()?
                .into_iter()
                .flat_map(u16::to_le_bytes)
                .collect()
        } else {
            m.try_to_f32_vec()?
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect()
        };
        let t = TensorView::new(
            if bf16 { Dtype::BF16 } else { Dtype::F32 },
            shape.to_vec(),
            &bytes,
        )?;
        serialize_to_file(
            [("value", t)],
            None,
            &dir.join(format!("{name}.safetensors")),
        )?;
    }
    Ok(())
}
pub fn noise(path: &Path, count: usize) -> Result<Vec<f32>> {
    let bytes = fs::read(path)?;
    ensure!(
        bytes.len() == count.checked_mul(4).unwrap_or(0),
        "noise file must contain {count} little-endian F32 values"
    );
    let values: Vec<_> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    ensure!(values.iter().all(|v| v.is_finite()), "non-finite noise");
    Ok(values)
}
