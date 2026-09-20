//! Export a causal-LM entrypoint and its immutable tensor layout.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use serde::Serialize;

use crate::{
    catena::{InferenceOptions, InferencePrecision, causal_lm_source_for_options},
    checkpoint::{read_gguf_model_config, tensor_layout},
};

#[derive(Serialize)]
struct Settings {
    entrypoint: &'static str,
    static_objects: Vec<PathBuf>,
    static_inputs: Vec<StaticInput>,
    state_bytes_per_capacity: Vec<usize>,
    vocabulary_size: usize,
    maximum_capacity: usize,
    generation: Generation,
}

#[derive(Serialize)]
struct StaticInput {
    object: usize,
    offset: usize,
    bytes: usize,
}

#[derive(Serialize)]
struct Generation {
    fixed_capacity: usize,
    prefill_chunk_tokens: usize,
}

/// Write `model.hex` and `model.toml` for a fixed-schedule causal-LM host.
/// Paths in the settings locate the GGUF shards; hosts content-address them.
/// The exported ABI returns F32 logits and owns all persistent cache bytes.
/// A BF16 graph widens only its final logits after selecting the next token.
pub fn write_environment(
    gguf: &Path,
    directory: &Path,
    capacity: usize,
    chunk: usize,
    options: InferenceOptions,
) -> Result<()> {
    let config = read_gguf_model_config(gguf)?;
    ensure!(
        capacity > 0 && capacity <= config.max_position_embeddings,
        "capacity exceeds model context"
    );
    ensure!(
        chunk > 0 && chunk <= capacity,
        "prefill chunk must fit the context"
    );
    let (files, views) = tensor_layout(gguf, &config)?;
    // Round to four-byte units. Hybrid recurrent state is independent of
    // capacity, so a committed schedule may leave a small amount of padding.
    let element_bytes = match options.precision {
        InferencePrecision::F32 => 4,
        InferencePrecision::Bf16 => 2,
    };
    let cache_bytes = config
        .cache_elements(capacity)?
        .checked_mul(element_bytes)
        .context("cache byte count overflowed")?;
    let stride = cache_bytes
        .div_ceil(capacity)
        .div_ceil(4)
        .checked_mul(4)
        .context("cache byte stride overflowed")?;
    let settings = Settings {
        entrypoint: "qwen3",
        static_objects: files
            .iter()
            .map(std::fs::canonicalize)
            .collect::<std::io::Result<_>>()?,
        static_inputs: views
            .iter()
            .map(|view| StaticInput {
                object: view.file_index,
                offset: view.file_offset,
                bytes: view.byte_len,
            })
            .collect(),
        state_bytes_per_capacity: vec![stride],
        vocabulary_size: config.vocabulary_size,
        maximum_capacity: capacity,
        generation: Generation {
            fixed_capacity: capacity,
            prefill_chunk_tokens: chunk,
        },
    };
    let source = causal_lm_source_for_options(&config, options)?;
    let settings = toml::to_string_pretty(&settings)?;
    std::fs::create_dir(directory).context("export directory must not already exist")?;
    std::fs::write(directory.join("model.hex"), source)?;
    std::fs::write(directory.join("model.toml"), settings)?;
    Ok(())
}
