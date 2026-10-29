//! One default-path 256x256 generation; no prompt/seed/reference matrix.
use anyhow::Result;
use catena_models::flux::{
    config,
    pipeline::{FluxPipeline, GenerationOptions},
};

#[test]
#[ignore = "requires gfx11 GPU and cached Flux Klein 4B weights; renders only 256x256"]
fn default_pipeline_256() -> Result<()> {
    let snapshot = config::snapshot("black-forest-labs/FLUX.2-klein-4B")?;
    let output = std::env::var_os("FLUX_WMMA_OUTPUT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| "/tmp/flux-wmma-default256.png".into());
    FluxPipeline::new(snapshot)?.generate(&GenerationOptions {
        prompt: "A red fox sitting in a snowy forest".into(),
        output,
        seed: 42,
        width: 256,
        height: 256,
        steps: 4,
        dump_dir: None,
        noise: None,
    })
}
