use rag_server::{
    config::AppConfig,
    inference::{MockEmbedder, OnnxClipEmbedder},
    rag::{EmbedderTextAdapter, OpenAiCompatibleLlmClient, RagPipeline},
    routes::build_router,
    state::AppState,
    storage::{InMemoryVectorStore, traits::VectorStore},
    telemetry::init_telemetry,
};
use std::{net::SocketAddr, sync::Arc, time::Duration};
use tokio::net::TcpListener;
async fn shutdown_signal() {
    #[cfg(unix)]
    {
        let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("Failed to install SIGTERM signal handler");
        tokio::select! {
            res = tokio::signal::ctrl_c() => {
                if let Err(err) = res {
                    tracing::error!("Failed to listen for Ctrl+C: {err}");
                } else {
                    tracing::info!("Shutdown signal received (Ctrl+C)");
                }
            }
            _ = sigterm.recv() => {
                tracing::info!("Shutdown signal received (SIGTERM)");
            }
        }
    }
    #[cfg(windows)]
    {
        let ctrl_c = tokio::signal::ctrl_c();
        let mut ctrl_close = tokio::signal::windows::ctrl_close().ok();
        let mut ctrl_shutdown = tokio::signal::windows::ctrl_shutdown().ok();
        tokio::select! {
            res = ctrl_c => {
                if let Err(err) = res {
                    tracing::error!("Failed to listen for Ctrl+C: {err}");
                } else {
                    tracing::info!("Shutdown signal received (Ctrl+C)");
                }
            }
            _ = async {
                match &mut ctrl_close {
                    Some(h) => {
                        let _ = h.recv().await;
                    }
                    None => std::future::pending::<()>().await,
                }
            } => {
                tracing::info!("Shutdown signal received (Windows CTRL_CLOSE)");
            }
            _ = async {
                match &mut ctrl_shutdown {
                    Some(h) => {
                        let _ = h.recv().await;
                    }
                    None => std::future::pending::<()>().await,
                }
            } => {
                tracing::info!("Shutdown signal received (Windows CTRL_SHUTDOWN)");
            }
        }
    }
    #[cfg(not(any(unix, windows)))]
    {
        if let Err(err) = tokio::signal::ctrl_c().await {
            tracing::error!("Failed to install or listen for Ctrl+C signal: {err}");
        } else {
            tracing::info!("Shutdown signal received (Ctrl+C)");
        }
    }
}
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = AppConfig::load()?;
    let _log_guard = init_telemetry(&config.logging);
    tracing::info!(
        host = %config.server.host,
        port = config.server.port,
        body_limit_mb = config.server.max_body_limit_mb,
        snapshot_path = ?config.server.snapshot_path,
        snapshot_interval_secs = config.server.snapshot_interval_secs,
        log_file = ?config.logging.file_path,
        llm_endpoint = %config.llm.endpoint,
        llm_model = %config.llm.model,
        "Server configuration successfully loaded"
    );
    let embedder = match OnnxClipEmbedder::new(
        &config.model.text_model_path,
        &config.model.vision_model_path,
        &config.model.tokenizer_path,
        config.model.max_concurrency,
        config.model.dimension,
    ) {
        Ok(onnx) => {
            tracing::info!(
                dimension = config.model.dimension,
                concurrency = config.model.max_concurrency,
                "Initialized local ONNX CLIP runtime session pool"
            );
            onnx.into_shared()
        }
        Err(e) => {
            tracing::warn!(
                warning = %e,
                "ONNX model files not found or could not be loaded. Initializing MockEmbedder fallback"
            );
            Arc::new(MockEmbedder::new(config.model.dimension))
        }
    };
    let vector_store = Arc::new(InMemoryVectorStore::new(config.model.dimension));
    let snapshot_path = config.server.snapshot_path.clone();
    match vector_store.load_from_disk(&snapshot_path).await {
        Ok(count) if count > 0 => {
            tracing::info!(
                loaded = count,
                path = ?snapshot_path,
                "Restored vector store snapshot from disk"
            );
        }
        Ok(_) => {
            tracing::info!("No existing snapshot found, starting with empty vector store");
        }
        Err(e) => {
            tracing::error!(
                error = %e,
                path = ?snapshot_path,
                "Failed to load snapshot from disk. Aborting server startup to prevent accidental data wipeout"
            );
            return Err(Box::<dyn std::error::Error>::from(e.to_string()));
        }
    }
    let periodic_store = vector_store.clone();
    let periodic_snapshot_path = snapshot_path.clone();
    let interval_secs = config.server.snapshot_interval_secs;
    let flush_task = if interval_secs > 0 {
        Some(tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(interval_secs));
            interval.tick().await;
            loop {
                interval.tick().await;
                if !periodic_store.is_dirty() {
                    tracing::debug!("Vector store clean, skipping periodic snapshot checkpoint");
                    continue;
                }
                tracing::debug!(path = ?periodic_snapshot_path, "Running periodic vector store snapshot checkpoint...");
                if let Err(e) = periodic_store.save_to_disk(&periodic_snapshot_path).await {
                    tracing::warn!(error = %e, "Periodic snapshot checkpoint failed");
                } else {
                    tracing::debug!("Periodic snapshot checkpoint completed successfully");
                }
            }
        }))
    } else {
        None
    };
    let llm_client = Arc::new(
        OpenAiCompatibleLlmClient::new(
            &config.llm.endpoint,
            &config.llm.model,
            config.llm.api_key.clone(),
        )
        .with_timeout(Duration::from_secs(config.llm.timeout_secs)),
    );
    let pipeline = Arc::new(RagPipeline::new(
        vector_store.clone(),
        Arc::new(EmbedderTextAdapter(embedder.clone())),
        llm_client,
    ));
    let state = AppState::new(config, embedder, vector_store.clone(), pipeline);
    let app = build_router(state.clone());
    let addr: SocketAddr = format!(
        "{}:{}",
        state.config().server.host,
        state.config().server.port
    )
    .parse()?;
    let listener = TcpListener::bind(addr).await?;
    tracing::info!("Server actively listening on http://{}", addr);
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    if let Some(task) = flush_task {
        task.abort();
        let _ = task.await;
    }
    tracing::info!(path = ?snapshot_path, "Persisting vector store snapshot before exit...");
    if let Err(e) = vector_store.save_to_disk(&snapshot_path).await {
        tracing::error!(error = %e, "Failed to persist vector store snapshot");
    } else {
        tracing::info!("Vector store snapshot successfully persisted");
    }
    Ok(())
}
