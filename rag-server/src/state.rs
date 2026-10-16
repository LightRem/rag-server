use crate::config::AppConfig;
use crate::inference::Embedder;
use crate::rag::RagPipeline;
use crate::storage::VectorStore;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

#[derive(Clone)]
pub struct AppState {
    config: Arc<AppConfig>,
    embedder: Arc<dyn Embedder>,
    vector_store: Arc<dyn VectorStore>,
    pipeline: Arc<RagPipeline>,
    http_client: reqwest::Client,
    uploads_dir: Arc<PathBuf>,
}

impl AppState {
    pub fn new(
        config: AppConfig,
        embedder: Arc<dyn Embedder>,
        vector_store: Arc<dyn VectorStore>,
        pipeline: Arc<RagPipeline>,
    ) -> Self {
        let uploads_dir = std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join("uploads");

        let _ = std::fs::create_dir_all(&uploads_dir);
        let canonical_uploads_dir = uploads_dir.canonicalize().unwrap_or(uploads_dir);
        let http_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .unwrap_or_default();

        Self {
            config: Arc::new(config),
            embedder,
            vector_store,
            pipeline,
            http_client,
            uploads_dir: Arc::new(canonical_uploads_dir),
        }
    }

    pub fn config(&self) -> &AppConfig {
        &self.config
    }

    pub fn embedder(&self) -> Arc<dyn Embedder> {
        self.embedder.clone()
    }

    pub fn vector_store(&self) -> Arc<dyn VectorStore> {
        self.vector_store.clone()
    }

    pub fn pipeline(&self) -> Arc<RagPipeline> {
        self.pipeline.clone()
    }

    pub fn http_client(&self) -> &reqwest::Client {
        &self.http_client
    }

    pub fn uploads_dir(&self) -> &Path {
        &self.uploads_dir
    }
}
