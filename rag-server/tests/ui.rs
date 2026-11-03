use axum::{
    body::Body,
    http::{Request, StatusCode, header},
};
use http_body_util::BodyExt;
use rag_server::{config::AppConfig, rag::MockLlmClient, routes::build_router};
use tower::ServiceExt;

mod common;

fn create_test_app() -> axum::Router {
    let config = AppConfig::default();
    build_router(common::state_with_llm(config, MockLlmClient::new()))
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
