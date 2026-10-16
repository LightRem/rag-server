use axum::{
    body::Body,
    http::{Request, StatusCode, header},
};
use bytes::Bytes;
use http_body_util::BodyExt;
use rag_server::{
    config::AppConfig,
    inference::{Embedder, MockEmbedder, l2_normalize, preprocessor::ImagePreprocessor},
    rag::{EmbedderTextAdapter, MockLlmClient, RagPipeline},
    routes::build_router,
    state::AppState,
    storage::InMemoryVectorStore,
};
use serde_json::{Value, json};
use std::sync::Arc;
use tower::ServiceExt;

fn create_test_state(config: AppConfig) -> AppState {
    let dimension = config.model.dimension;
    let embedder = Arc::new(MockEmbedder::new(dimension));
    let vector_store = Arc::new(InMemoryVectorStore::new(dimension));
    let mock_llm = Arc::new(MockLlmClient::new());
    let pipeline = Arc::new(RagPipeline::new(
        vector_store.clone(),
        Arc::new(EmbedderTextAdapter(embedder.clone())),
        mock_llm,
    ));
    AppState::new(config, embedder, vector_store, pipeline)
}

#[tokio::test]
async fn test_text_embedding_generation_and_l2_norm() {
    let config = AppConfig::load().expect("Config must load");
    let dimension = config.model.dimension;
    let state = create_test_state(config);
    let app = build_router(state);
    let payload = json!({
        "text": "multimodal vector retrieval in rust",
        "image": null
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/v1/embeddings")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_vec(&payload).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = response.into_body().collect().await.unwrap().to_bytes();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["dimension"], dimension);
    assert!(json["image_embedding"].is_null());
    let text_vec: Vec<f32> = serde_json::from_value(json["text_embedding"].clone()).unwrap();
    assert_eq!(text_vec.len(), dimension);
    let norm: f32 = text_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-5, "Norm must be ~1.0, got {norm}");
}

#[tokio::test]
async fn test_image_embedding_generation_from_raw_base64() {
    let config = AppConfig::load().expect("Config must load");
    let dimension = config.model.dimension;
    let state = create_test_state(config);
    let app = build_router(state);
    let payload = json!({
        "text": null,
        "image": {
            "type": "Raw",
            "data": {
                "base64": "ZmFrZV9wbmdfaW1hZ2VfcGF5bG9hZA==",
                "mime_type": "image/png"
            }
        }
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/v1/embeddings")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_vec(&payload).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = response.into_body().collect().await.unwrap().to_bytes();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert!(json["text_embedding"].is_null());
    let image_vec: Vec<f32> = serde_json::from_value(json["image_embedding"].clone()).unwrap();
    assert_eq!(image_vec.len(), dimension);
    let norm: f32 = image_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-5);
}

#[tokio::test]
async fn test_batch_api_endpoints() {
    let config = AppConfig::load().expect("Config must load");
    let state = create_test_state(config);
    let app = build_router(state);
    let payload = json!({
        "texts": ["first sentence", "second sentence"],
        "images": [
            {
                "type": "Raw",
                "data": {
                    "base64": "aW1hZ2Vfb25l",
                    "mime_type": "image/jpeg"
                }
            },
            {
                "type": "Raw",
                "data": {
                    "base64": "aW1hZ2VfdHdv",
                    "mime_type": "image/jpeg"
                }
            }
        ]
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/v1/embeddings")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_vec(&payload).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = response.into_body().collect().await.unwrap().to_bytes();
    let json: Value = serde_json::from_slice(&body).unwrap();
    let text_batch: Vec<Vec<f32>> =
        serde_json::from_value(json["text_embeddings"].clone()).unwrap();
    assert_eq!(text_batch.len(), 2);
    let img_batch: Vec<Vec<f32>> =
        serde_json::from_value(json["image_embeddings"].clone()).unwrap();
    assert_eq!(img_batch.len(), 2);
}

#[tokio::test]
async fn test_mock_embedder_batch_consistency() {
    let embedder = MockEmbedder::new(128);
    let queries = &["query alpha", "query beta", "query gamma"];
    let batch_results = embedder
        .embed_text_batch(queries)
        .await
        .expect("Batch embedding must succeed");
    assert_eq!(batch_results.len(), 3);
    for vec in &batch_results {
        assert_eq!(vec.len(), 128);
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }
    let single_result = embedder
        .embed_text("query alpha")
        .await
        .expect("Single embedding must succeed");
    assert_eq!(batch_results[0], single_result);
    let dummy_images = &[
        Bytes::from_static(b"image_one"),
        Bytes::from_static(b"image_two_bytes"),
    ];
    let img_batch = embedder
        .embed_image_batch(dummy_images)
        .await
        .expect("Image batch embedding must succeed");
    assert_eq!(img_batch.len(), 2);
    assert_eq!(img_batch[0].len(), 128);
}

#[tokio::test]
async fn test_l2_normalization_strict_math_and_nan_rejection() {
    let mut vec = vec![3.0, 4.0];
    l2_normalize(&mut vec).expect("Normalization should succeed");
    assert!((vec[0] - 0.6).abs() < 1e-6);
    assert!((vec[1] - 0.8).abs() < 1e-6);
    let mut zero_vec = vec![0.0, 0.0, 0.0];
    assert!(l2_normalize(&mut zero_vec).is_err());
    let mut nan_vec = vec![f32::NAN, 1.0, 2.0];
    assert!(l2_normalize(&mut nan_vec).is_err());
    let mut inf_vec = vec![f32::INFINITY, 1.0];
    assert!(l2_normalize(&mut inf_vec).is_err());
}

#[tokio::test]
async fn test_image_preprocessing_non_square_aspect_ratio() {
    let mut img = image::RgbImage::new(400, 200);
    for pixel in img.pixels_mut() {
        *pixel = image::Rgb([100, 150, 200]);
    }
    let mut png_bytes = Vec::new();
    image::DynamicImage::ImageRgb8(img)
        .write_to(
            &mut std::io::Cursor::new(&mut png_bytes),
            image::ImageFormat::Png,
        )
        .unwrap();
    let tensor = ImagePreprocessor::preprocess(&png_bytes).expect("Preprocessing must succeed");
    assert_eq!(tensor.shape(), &[1, 3, 224, 224]);
}

#[tokio::test]
async fn test_image_preprocessing_corrupted_payload_fails() {
    let corrupted = b"corrupted_binary_header_not_an_image";
    assert!(ImagePreprocessor::preprocess(corrupted).is_err());
}
