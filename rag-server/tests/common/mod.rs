#![allow(dead_code)]

use rag_server::{
    config::AppConfig,
    inference::MockEmbedder,
    rag::{MockLlmClient, RagPipeline},
    state::AppState,
    storage::InMemoryVectorStore,
};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT_DIR: AtomicUsize = AtomicUsize::new(0);

/// Уникальная папка uploads во временной директории ОС, а не в cwd.
pub fn test_uploads_dir() -> PathBuf {
    let n = NEXT_DIR.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "rag_server_test_uploads_{}_{n}",
        std::process::id()
    ))
}

/// Состояние приложения с mock-эмбеддером, in-memory хранилищем и заданным mock-LLM.
/// Папка uploads всегда уникальна и лежит во временной директории.
pub fn state_with_llm(mut config: AppConfig, llm: MockLlmClient) -> AppState {
    config.server.uploads_dir = test_uploads_dir();
    let embedder = Arc::new(MockEmbedder::new(config.model.dimension));
    let vector_store = Arc::new(InMemoryVectorStore::new(config.model.dimension));
    let pipeline = Arc::new(RagPipeline::new(
        vector_store.clone(),
        embedder.clone(),
        Arc::new(llm),
    ));
    AppState::new(config, embedder, vector_store, pipeline).expect("AppState must initialize")
}
