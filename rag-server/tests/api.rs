use axum::{
    body::Body,
    http::{Request, StatusCode, header},
};
use http_body_util::BodyExt;
use rag_server::{
    config::AppConfig,
    rag::MockLlmClient,
    routes::{MAX_BATCH_SIZE, build_router},
};
use serde_json::{Value, json};
use tower::ServiceExt;

mod common;

fn create_test_app(llm_response: Option<&str>) -> axum::Router {
    let mut config = AppConfig::default();
    config.model.dimension = 4;
    let llm = match llm_response {
        Some(resp) => MockLlmClient::with_response(resp),
        None => MockLlmClient::new(),
    };
    build_router(common::state_with_llm(config, llm))
}

#[tokio::test]
async fn test_api_stats_and_health() {
    let app = create_test_app(None);
    let health_req = Request::builder()
        .uri("/health")
        .body(Body::empty())
        .unwrap();
    let res = app.clone().oneshot(health_req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let body = res.into_body().collect().await.unwrap().to_bytes();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["status"], "degraded");
    assert_eq!(json["embedder"], "mock");
    let stats_req = Request::builder()
        .uri("/api/v1/stats")
        .body(Body::empty())
        .unwrap();
    let res = app.oneshot(stats_req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let body = res.into_body().collect().await.unwrap().to_bytes();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["total_documents"], 0);
    assert_eq!(json["vector_dimension"], 4);
}

#[tokio::test]
async fn test_api_multimodal_indexing_and_rag_fusion() {
    let mock_response = "Information found in [Source ID: doc_multimodal_1] with view [Image ID: doc_multimodal_1].";
    let app = create_test_app(Some(mock_response));
    let payload = json!({
        "id": "doc_multimodal_1",
        "text": "Ferrari F8 Tributo technical specification",
        "image": {
            "type": "Raw",
            "data": {
                "base64": "ZmFrZV9pbWFnZV9ieXRlcw==",
                "mime_type": "image/png"
            }
        },
        "title": "Ferrari Model",
        "metadata": {
            "caption": "Red sports car"
        }
    });
    let index_req = Request::builder()
        .method("POST")
        .uri("/api/v1/index")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();
    let res = app.clone().oneshot(index_req).await.unwrap();
    assert_eq!(res.status(), StatusCode::CREATED);
    let rag_req = Request::builder()
        .method("POST")
        .uri("/api/v1/rag")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(
            json!({
                "query": "Ferrari specs",
                "top_k": 2,
                "filter_uncited": false
            })
            .to_string(),
        ))
        .unwrap();
    let res = app.oneshot(rag_req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let body = res.into_body().collect().await.unwrap().to_bytes();
    let json: Value = serde_json::from_slice(&body).unwrap();
    let sources = json["sources"].as_array().unwrap();
    assert_eq!(sources.len(), 1);
    assert_eq!(sources[0]["id"], "doc_multimodal_1");
    let visuals = json["visuals"].as_array().unwrap();
    assert_eq!(visuals.len(), 1);
    assert_eq!(visuals[0]["id"], "doc_multimodal_1");
    assert_eq!(visuals[0]["url"], "/uploads/doc_multimodal_1.png");
}

#[tokio::test]
async fn test_api_visual_search_with_image_query() {
    let app = create_test_app(None);
    let index_payload = json!({
        "id": "doc_visual_match",
        "text": "Sports car side view"
    });
    let index_req = Request::builder()
        .method("POST")
        .uri("/api/v1/index")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_vec(&index_payload).unwrap()))
        .unwrap();
    let _ = app.clone().oneshot(index_req).await.unwrap();
    let search_payload = json!({
        "image": {
            "type": "Raw",
            "data": {
                "base64": "ZmFrZV9zZWFyY2hfaW1hZ2U=",
                "mime_type": "image/jpeg"
            }
        },
        "top_k": 2
    });
    let search_req = Request::builder()
        .method("POST")
        .uri("/api/v1/search")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_vec(&search_payload).unwrap()))
        .unwrap();
    let res = app.oneshot(search_req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let body = res.into_body().collect().await.unwrap().to_bytes();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["has_image_query"], true);
    let results = json["results"].as_array().unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0]["id"], "doc_visual_match");
}

#[tokio::test]
async fn test_api_non_object_metadata_preserves_text() {
    let app = create_test_app(None);
    let payload = json!({
        "id": "doc_flat_meta",
        "text": "Text content must not be lost",
        "metadata": "general_category_string"
    });
    let index_req = Request::builder()
        .method("POST")
        .uri("/api/v1/index")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();
    let res = app.clone().oneshot(index_req).await.unwrap();
    assert_eq!(res.status(), StatusCode::CREATED);
    let search_req = Request::builder()
        .method("POST")
        .uri("/api/v1/search")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(
            json!({
                "query": "content",
                "top_k": 1
            })
            .to_string(),
        ))
        .unwrap();
    let res = app.oneshot(search_req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let body = res.into_body().collect().await.unwrap().to_bytes();
    let json: Value = serde_json::from_slice(&body).unwrap();
    let doc_res = &json["results"][0];
    assert_eq!(doc_res["id"], "doc_flat_meta");
    assert_eq!(doc_res["payload"]["text"], "Text content must not be lost");
    assert_eq!(
        doc_res["payload"]["custom_metadata"],
        "general_category_string"
    );
}

#[tokio::test]
async fn test_api_ssrf_protection_blocks_loopback_and_metadata_ip() {
    let app = create_test_app(None);
    let loopback_payload = json!({
        "id": "ssrf_loopback",
        "image": {
            "type": "Url",
            "data": "http://127.0.0.1:8080/secret"
        }
    });
    let req1 = Request::builder()
        .method("POST")
        .uri("/api/v1/index")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_vec(&loopback_payload).unwrap()))
        .unwrap();
    let res1 = app.clone().oneshot(req1).await.unwrap();
    assert_eq!(res1.status(), StatusCode::BAD_REQUEST);
    let body1 = res1.into_body().collect().await.unwrap().to_bytes();
    let json1: Value = serde_json::from_slice(&body1).unwrap();
    assert!(
        json1["error"]["message"]
            .as_str()
            .unwrap()
            .contains("SSRF protection"),
        "Expected SSRF error, got: {}",
        json1["error"]["message"]
    );
    let aws_payload = json!({
        "id": "ssrf_aws",
        "image": {
            "type": "Url",
            "data": "http://169.254.169.254/latest/meta-data/"
        }
    });
    let req2 = Request::builder()
        .method("POST")
        .uri("/api/v1/index")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_vec(&aws_payload).unwrap()))
        .unwrap();
    let res2 = app.oneshot(req2).await.unwrap();
    assert_eq!(res2.status(), StatusCode::BAD_REQUEST);
    let body2 = res2.into_body().collect().await.unwrap().to_bytes();
    let json2: Value = serde_json::from_slice(&body2).unwrap();
    assert!(
        json2["error"]["message"]
            .as_str()
            .unwrap()
            .contains("SSRF protection"),
        "Expected SSRF error, got: {}",
        json2["error"]["message"]
    );
}

#[tokio::test]
async fn test_api_batch_index_exceeds_limit() {
    let app = create_test_app(None);
    let mut docs = Vec::new();
    for i in 0..(MAX_BATCH_SIZE + 1) {
        docs.push(json!({
            "id": format!("doc_{i}"),
            "text": "Sample text"
        }));
    }
    let batch_payload = json!({ "documents": docs });
    let req = Request::builder()
        .method("POST")
        .uri("/api/v1/index/batch")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_vec(&batch_payload).unwrap()))
        .unwrap();
    let res = app.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_api_delete_document() {
    let app = create_test_app(None);
    let index_payload = json!({
        "id": "doc_temp",
        "text": "Temporary document that will be purged."
    });
    let index_req = Request::builder()
        .method("POST")
        .uri("/api/v1/index")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_vec(&index_payload).unwrap()))
        .unwrap();
    let _ = app.clone().oneshot(index_req).await.unwrap();
    let del_req = Request::builder()
        .method("DELETE")
        .uri("/api/v1/documents/doc_temp")
        .body(Body::empty())
        .unwrap();
    let del_res = app.clone().oneshot(del_req).await.unwrap();
    assert_eq!(del_res.status(), StatusCode::OK);
    let stats_req = Request::builder()
        .uri("/api/v1/stats")
        .body(Body::empty())
        .unwrap();
    let stats_res = app.oneshot(stats_req).await.unwrap();
    let body = stats_res.into_body().collect().await.unwrap().to_bytes();
    let stats_json: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(stats_json["total_documents"], 0);
}

#[tokio::test]
async fn test_api_ssrf_protection_blocks_ipv4_mapped_ipv6() {
    let app = create_test_app(None);
    let mapped_payload = json!({
        "id": "ssrf_mapped_ipv6",
        "image": {
            "type": "Url",
            "data": "http://[::ffff:127.0.0.1]:8080/secret"
        }
    });
    let req = Request::builder()
        .method("POST")
        .uri("/api/v1/index")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_vec(&mapped_payload).unwrap()))
        .unwrap();
    let res = app.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);
    let body = res.into_body().collect().await.unwrap().to_bytes();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert!(
        json["error"]["message"]
            .as_str()
            .unwrap()
            .contains("SSRF protection")
    );
}
