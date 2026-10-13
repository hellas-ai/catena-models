//! Model-owned files, resolved by the common Hub loader.
use anyhow::Result;
use std::path::PathBuf;
const FILES: &[&str] = &[
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "model.safetensors",
];
pub fn snapshot(model: &str) -> Result<PathBuf> {
    crate::hfhub::resolve_model_files(model, FILES)
}
