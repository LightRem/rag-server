use crate::{
    config::{AppConfig, ModelConfig, ServerConfig},
    inference::{Embedder, MockEmbedder, OnnxClipEmbedder},
    rag::{EmbedderTextAdapter, OpenAiCompatibleLlmClient, RagPipeline},
    routes::build_router,
    state::AppState,
    storage::{InMemoryVectorStore, traits::VectorStore},
};
use std::{path::Path, sync::Arc, time::Duration};
use tokio::{net::TcpListener, task::JoinHandle};

fn init_embedder(config: &ModelConfig) -> Arc<dyn Embedder> {
    match OnnxClipEmbedder::new(
        &config.text_model_path,
        &config.vision_model_path,
        &config.tokenizer_path,
        config.max_concurrency,
        config.dimension,
    ) {
        Ok(onnx) => {
            tracing::info!(
                dimension = config.dimension,
                concurrency = config.max_concurrency,
                "Initialized local ONNX CLIP runtime session pool"
            );
            onnx.into_shared()
        }
        Err(e) => {
            tracing::warn!(
                warning = %e,
                "ONNX model files not found or could not be loaded. Initializing MockEmbedder fallback"
            );
            Arc::new(MockEmbedder::new(config.dimension))
        }
    }
}

async fn init_vector_store(
    config: &AppConfig,
) -> Result<Arc<InMemoryVectorStore>, Box<dyn std::error::Error>> {
    let vector_store = Arc::new(InMemoryVectorStore::new(config.model.dimension));
    let snapshot_path = &config.server.snapshot_path;

    match vector_store.load_from_disk(snapshot_path).await {
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
                "Failed to load snapshot from disk. Aborting startup to prevent accidental data loss"
            );
            return Err(Box::<dyn std::error::Error>::from(e.to_string()));
        }
    }

    Ok(vector_store)
}

fn spawn_periodic_flush_task(
    store: Arc<InMemoryVectorStore>,
    config: &ServerConfig,
) -> Option<JoinHandle<()>> {
    let interval_secs = config.snapshot_interval_secs;
    if interval_secs == 0 {
        return None;
    }

    let snapshot_path = config.snapshot_path.clone();
    Some(tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(interval_secs));
        interval.tick().await;

        loop {
            interval.tick().await;
            if !store.is_dirty() {
                tracing::debug!("Vector store clean, skipping periodic snapshot checkpoint");
                continue;
            }

            tracing::debug!(path = ?snapshot_path, "Running periodic snapshot checkpoint...");
            if let Err(e) = store.save_to_disk(&snapshot_path).await {
                tracing::warn!(error = %e, "Periodic snapshot checkpoint failed");
            } else {
                tracing::debug!("Periodic snapshot checkpoint completed successfully");
            }
        }
    }))
}

async fn persist_final_snapshot(store: &InMemoryVectorStore, path: &Path) {
    tracing::info!(path = ?path, "Persisting vector store snapshot before shutdown...");
    if let Err(e) = store.save_to_disk(path).await {
        tracing::error!(error = %e, "Failed to persist vector store snapshot");
    } else {
        tracing::info!("Vector store snapshot successfully persisted");
    }
}

async fn shutdown_signal() {
    #[cfg(unix)]
    {
        let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("Failed to install SIGTERM handler");
        tokio::select! {
            res = tokio::signal::ctrl_c() => {
                if let Err(e) = res {
                    tracing::error!("Failed to listen for Ctrl+C: {e}");
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
                if let Err(e) = res {
                    tracing::error!("Failed to listen for Ctrl+C: {e}");
                } else {
                    tracing::info!("Shutdown signal received (Ctrl+C)");
                }
            }
            _ = async {
                match &mut ctrl_close {
                    Some(h) => { let _ = h.recv().await; }
                    None => std::future::pending::<()>().await,
                }
            } => {
                tracing::info!("Shutdown signal received (Windows CTRL_CLOSE)");
            }
            _ = async {
                match &mut ctrl_shutdown {
                    Some(h) => { let _ = h.recv().await; }
                    None => std::future::pending::<()>().await,
                }
            } => {
                tracing::info!("Shutdown signal received (Windows CTRL_SHUTDOWN)");
            }
        }
    }

    #[cfg(not(any(unix, windows)))]
    {
        if let Err(e) = tokio::signal::ctrl_c().await {
            tracing::error!("Failed to install or listen for Ctrl+C: {e}");
        } else {
            tracing::info!("Shutdown signal received (Ctrl+C)");
        }
    }
}

pub async fn run(config: AppConfig) -> Result<(), Box<dyn std::error::Error>> {
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

    let embedder = init_embedder(&config.model);
    let vector_store = init_vector_store(&config).await?;
    let flush_task = spawn_periodic_flush_task(vector_store.clone(), &config.server);

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

    let state = AppState::new(config.clone(), embedder, vector_store.clone(), pipeline);
    let app = build_router(state);

    let listener = TcpListener::bind((config.server.host.as_str(), config.server.port)).await?;
    let local_addr = listener.local_addr()?;
    tracing::info!("Server actively listening on http://{}", local_addr);

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    if let Some(task) = flush_task {
        task.abort();
        let _ = task.await;
    }

    persist_final_snapshot(&vector_store, &config.server.snapshot_path).await;
    Ok(())
}
