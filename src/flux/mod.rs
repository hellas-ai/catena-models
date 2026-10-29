pub mod config;
pub mod debug;
pub mod graph;
pub mod ops;
pub mod pipeline;
pub mod text;
pub mod transformer;
pub mod vae;

#[cfg(test)]
mod kernel_tests;
mod kernels;

pub(crate) fn validate_backend() -> anyhow::Result<()> {
    anyhow::ensure!(
        crate::model::configured_gpu_dialect()? == catena_lang::runtime::GpuDialect::Hip,
        "Flux WMMA requires HIP on gfx11"
    );
    Ok(())
}
