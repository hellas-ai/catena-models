//! HTTP serving for text generation, System One decisions, and Phonon speech.
mod api;
mod engine;
mod output;
mod prefix_cache;
mod prompt;
mod protocol;
mod responses;
mod responses_output;
#[cfg(test)]
mod responses_tests;
mod speech;
mod structured;
#[cfg(test)]
mod tests;
mod tools;

use anyhow::{Context, Result};
use catena_models::{catena::InferenceOptions, hfhub::HubCheckpoint};
use clap::Args;
use std::{
    net::IpAddr,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

#[derive(Debug, Args)]
#[group(conflicts_with_all = ["prompt", "image", "bench", "generate_entrypoint"])]
pub struct ServeOptions {
    /// Serve Phonon speech only, loading its model on the first request.
    #[arg(long, requires = "serve", conflicts_with = "model")]
    tts_only: bool,
    /// Listening address (use 0.0.0.0 for remote access).
    #[arg(long, default_value = "127.0.0.1", requires = "serve")]
    host: IpAddr,
    #[arg(long, default_value_t = 8000, requires = "serve")]
    port: u16,
    /// Maximum prompt plus output tokens for each request.
    #[arg(long, default_value_t=128*1024, value_parser=super::parse_positive_usize, requires="serve")]
    ctx_size: usize,
    /// Disable GPU prompt checkpoints (canonical prefill batching is retained).
    #[arg(long, requires = "serve")]
    no_prefix_cache: bool,
    /// Optional bearer credential for all endpoints.
    #[arg(
        long,
        env = "CATENA_API_KEY",
        hide_env_values = true,
        requires = "serve"
    )]
    api_key: Option<String>,
}

impl ServeOptions {
    pub fn tts_only(&self) -> bool {
        self.tts_only
    }
}

pub fn run(
    config: ServeOptions,
    model: Option<(String, HubCheckpoint)>,
    options: InferenceOptions,
) -> Result<()> {
    anyhow::ensure!(
        config.api_key.as_ref().is_none_or(|k| !k.is_empty()),
        "API key must not be empty"
    );
    let shutdown = Arc::new(AtomicBool::new(false));
    let model_id = model.as_ref().map(|(model, _)| Arc::from(model.as_str()));
    let worker = engine::start(
        move || {
            let Some((model, checkpoint)) = model else {
                return Ok(engine::ModelBackend::Speech);
            };
            eprintln!("loading model for serving");
            if let Some(path) = checkpoint.gguf_path()
                && catena_models::systemone::Metadata::read(path)?.is_some()
            {
                let projector = catena_models::hfhub::download_mmproj(&model)?;
                Ok(engine::ModelBackend::Decision(
                    catena_models::systemone::Runner::load(
                        path,
                        Some(&projector),
                        options,
                        config.ctx_size,
                        model,
                    )?,
                ))
            } else {
                Ok(engine::ModelBackend::Chat(engine::QwenBackend::load(
                    &checkpoint,
                    options,
                    config.ctx_size,
                    !config.no_prefix_cache,
                )?))
            }
        },
        shutdown.clone(),
    )?;
    let state = api::ServerState {
        model: model_id,
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
            "Inference server ready at http://{}/v1",
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
