use axum::{
    body::Body,
    http::{Request, StatusCode, header},
};
use http_body_util::BodyExt;
use rag_server::{
    config::AppConfig,
    inference::MockEmbedder,
    rag::{EmbedderTextAdapter, OpenAiCompatibleLlmClient, RagPipeline},
    routes::build_router,
    state::AppState,
    storage::InMemoryVectorStore,
};
use std::sync::Arc;
use std::time::Duration;
use tower::ServiceExt;
fn create_test_app() -> axum::Router {
    let config = AppConfig::default();
    let embedder = Arc::new(MockEmbedder::new(config.model.dimension));
    let vector_store = Arc::new(InMemoryVectorStore::new(config.model.dimension));
    let llm_client = Arc::new(
        OpenAiCompatibleLlmClient::new(&config.llm.endpoint, &config.llm.model, None)
            .with_timeout(Duration::from_secs(5)),
    );
    let pipeline = Arc::new(RagPipeline::new(
        vector_store.clone(),
        Arc::new(EmbedderTextAdapter(embedder.clone())),
        llm_client,
    ));
    let state = AppState::new(config, embedder, vector_store, pipeline);
    build_router(state)
}
#[tokio::test]
async fn test_root_serves_html_dashboard() {
    let app = create_test_app();
    let req = Request::builder()
        .method("GET")
        .uri("/")
        .body(Body::empty())
        .unwrap();
    let res = app.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let content_type = res
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default();
    assert!(
        content_type.contains("text/html"),
        "Expected text/html content type, got: {content_type}"
    );
    let body = res.into_body().collect().await.unwrap().to_bytes();
    let html = String::from_utf8_lossy(&body);
    assert!(html.contains("Multimodal RAG Server — Minimal Control Panel"));
    assert!(html.contains("Index Document"));
    assert!(html.contains("Vector Search"));
    assert!(html.contains("RAG Query"));
}
