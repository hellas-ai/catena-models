//! Model-owned files, resolved by the common Hub loader.
use anyhow::Result;
use std::path::{Path, PathBuf};
pub const MODEL_ID: &str = "openai/whisper-small";
const FILES: &[&str] = &[
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "model.safetensors",
];
pub fn snapshot(explicit: Option<&Path>) -> Result<PathBuf> {
    crate::hfhub::resolve_model_files(MODEL_ID, FILES, explicit)
}
