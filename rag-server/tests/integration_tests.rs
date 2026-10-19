use axum::{
    body::Body,
    http::{Request, StatusCode, header},
};
use http_body_util::BodyExt;
use rag_server::{
    config::{AppConfig, ServerConfig},
    domain::ImageSource,
    rag::MockLlmClient,
    routes::build_router,
    state::AppState,
    telemetry::init_telemetry,
};
use serde_json::{Value, json};
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};
use tower::ServiceExt;

mod common;

fn create_test_state(config: AppConfig) -> AppState {
    common::state_with_llm(config, MockLlmClient::new())
}

#[tokio::test]
async fn test_health_check_end_to_end() {
    let config = AppConfig::load().expect("Config must load");
    let _log_guard = init_telemetry(&config.logging);
    let state = create_test_state(config);
    let app = build_router(state);
    let response = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = response.into_body().collect().await.unwrap().to_bytes();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["status"], "degraded");
    assert_eq!(json["embedder"], "mock");
    assert_eq!(json["service"], "multimodal-rag-server");
}

#[tokio::test]
async fn test_payload_validation_http_success() {
    let config = AppConfig::load().expect("Config must load");
    let state = create_test_state(config);
    let app = build_router(state);
    let payload = json!({
        "text": "sample search query",
        "image": null
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/v1/payload/validate")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_vec(&payload).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = response.into_body().collect().await.unwrap().to_bytes();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["status"], "valid");
    assert_eq!(json["has_text"], true);
}

#[tokio::test]
async fn test_payload_validation_http_malformed_json_returns_json_error() {
    let config = AppConfig::load().expect("Config must load");
    let state = create_test_state(config);
    let app = build_router(state);
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/v1/payload/validate")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(b"{ malformed json string }".to_vec()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = response.into_body().collect().await.unwrap().to_bytes();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert!(json.get("error").is_some());
    assert_eq!(json["error"]["status_code"], 400);
}

#[tokio::test]
async fn test_payload_validation_http_unsupported_mime_type() {
    let config = AppConfig::load().expect("Config must load");
    let state = create_test_state(config);
    let app = build_router(state);
    let payload = json!({
        "text": null,
        "image": {
            "type": "Raw",
            "data": {
                "base64": "ZmFrZV9ieXRlcw==",
                "mime_type": "application/x-executable"
            }
        }
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/v1/payload/validate")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_vec(&payload).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = response.into_body().collect().await.unwrap().to_bytes();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert!(
        json["error"]["message"]
            .as_str()
            .unwrap()
            .contains("Unsupported MIME type")
    );
}

#[tokio::test]
async fn test_config_validation_rejects_zero_port() {
    let mut config = AppConfig::default();
    config.server.port = 0;
    assert!(config.validate().is_err());
    let valid_config = AppConfig {
        server: ServerConfig {
            host: "127.0.0.1".into(),
            port: 8080,
            max_body_limit_mb: 16,
            snapshot_path: std::path::PathBuf::from("data/vector_store.json"),
            ..Default::default()
        },
        ..Default::default()
    };
    assert!(valid_config.validate().is_ok());
}

#[tokio::test]
async fn test_path_traversal_protection_comprehensive() {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let pid = std::process::id();
    let unique_name = format!("rag_server_test_sandbox_{}_{}", pid, timestamp);
    let temp_root = std::env::temp_dir().join(unique_name);
    let allowed_data = temp_root.join("uploads");
    fs::create_dir_all(&allowed_data).expect("Must create test directory");
    let empty_source = ImageSource::Path(PathBuf::from(""));
    assert!(empty_source.validate_safe_path(&allowed_data).is_err());
    let traversal = ImageSource::Path(PathBuf::from("../outside.txt"));
    assert!(traversal.validate_safe_path(&allowed_data).is_err());
    let nested_new_file = ImageSource::Path(PathBuf::from("2026/10/nested_photo.jpg"));
    let result = nested_new_file.validate_safe_path(&allowed_data);
    assert!(result.is_ok());
    let _ = fs::remove_dir_all(&temp_root);
}
