//! Text-only serving. HTTP schemas and inference are deliberately separate.
mod api;
mod engine;
mod output;
mod prompt;
mod protocol;
mod responses;
mod responses_output;
#[cfg(test)]
mod responses_tests;
mod structured;
#[cfg(test)]
mod tests;
mod tools;

use anyhow::{Context, Result};
use catena_models::catena::InferenceOptions;
use clap::Args;
use std::{
    net::IpAddr,
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

#[derive(Debug, Args)]
#[group(conflicts_with_all = ["prompt", "image", "bench", "generate_entrypoint"])]
pub struct ServeOptions {
    /// Listening address (use 0.0.0.0 for remote access).
    #[arg(long, default_value = "127.0.0.1", requires = "serve")]
    host: IpAddr,
    #[arg(long, default_value_t = 8000, requires = "serve")]
    port: u16,
    /// Maximum prompt plus output tokens for each request.
    #[arg(long, default_value_t=128*1024, value_parser=super::parse_positive_usize, requires="serve")]
    ctx_size: usize,
    /// Optional bearer credential for all endpoints.
    #[arg(
        long,
        env = "CATENA_API_KEY",
        hide_env_values = true,
        requires = "serve"
    )]
    api_key: Option<String>,
}

pub fn run(
    config: ServeOptions,
    model: String,
    path: PathBuf,
    options: InferenceOptions,
) -> Result<()> {
    anyhow::ensure!(
        config.api_key.as_ref().is_none_or(|k| !k.is_empty()),
        "API key must not be empty"
    );
    let shutdown = Arc::new(AtomicBool::new(false));
    eprintln!("loading model and chat template for serving");
    let worker = engine::start(
        move || engine::QwenBackend::load(path, options, config.ctx_size),
        shutdown.clone(),
    )?;
    let state = api::ServerState {
        model: Arc::from(model),
        key: config.api_key.map(Arc::from),
        jobs: worker.sender,
    };
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    let result = runtime.block_on(async move {
        let listener = tokio::net::TcpListener::bind((config.host, config.port))
            .await
            .context("binding HTTP listener")?;
        eprintln!(
            "Qwen chat server ready at http://{}/v1",
            listener.local_addr()?
        );
        let signal = shutdown.clone();
        axum::serve(listener, api::router(state))
            .with_graceful_shutdown(async move {
                if let Err(e) = tokio::signal::ctrl_c().await {
                    eprintln!("shutdown signal: {e}");
                }
                signal.store(true, Ordering::Relaxed);
            })
            .await
            .context("serving HTTP")
    });
    worker
        .thread
        .join()
        .map_err(|_| anyhow::anyhow!("inference worker panicked"))?;
    result
}
