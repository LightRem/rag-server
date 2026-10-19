use super::media::load_image_bytes;
use super::models::{EmbeddingsRequest, EmbeddingsResponse, MAX_BATCH_SIZE};
use crate::{error::ApiError, extractors::AppJson, state::AppState};
use axum::{Json, extract::State, http::StatusCode, response::IntoResponse};

pub async fn process_embeddings_request(
    state: &AppState,
    payload: EmbeddingsRequest,
) -> Result<EmbeddingsResponse, ApiError> {
    let embedder = state.embedder();
    let mut response = EmbeddingsResponse {
        dimension: embedder.dimension(),
        text_embedding: None,
        text_embeddings: None,
        image_embedding: None,
        image_embeddings: None,
    };

    let mut processed_anything = false;

    if let Some(text) = payload.text {
        let vec = embedder.embed_text(&text).await?;
        response.text_embedding = Some(vec);
        processed_anything = true;
    }

    if let Some(texts) = payload.texts.filter(|t| !t.is_empty()) {
        if texts.len() > MAX_BATCH_SIZE {
            return Err(ApiError::ValidationError(format!(
                "Batch size {} exceeds maximum allowed limit of {} items",
                texts.len(),
                MAX_BATCH_SIZE
            )));
        }
        let str_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let batch = embedder.embed_text_batch(&str_refs).await?;
        response.text_embeddings = Some(batch);
        processed_anything = true;
    }

    let allowed_root = state.uploads_dir();
    let max_image_bytes = state.config().server.max_body_limit_bytes();

    if let Some(img_source) = payload.image {
        let bytes = load_image_bytes(&img_source, allowed_root, max_image_bytes).await?;
        let vec = embedder.embed_image(&bytes).await?;
        response.image_embedding = Some(vec);
        processed_anything = true;
    }

    if let Some(images) = payload.images.filter(|i| !i.is_empty()) {
        if images.len() > MAX_BATCH_SIZE {
            return Err(ApiError::ValidationError(format!(
                "Batch size {} exceeds maximum allowed limit of {} items",
                images.len(),
                MAX_BATCH_SIZE
            )));
        }
        let mut byte_buffers = Vec::with_capacity(images.len());
        for source in &images {
            byte_buffers.push(load_image_bytes(source, allowed_root, max_image_bytes).await?);
        }
        let batch = embedder.embed_image_batch(&byte_buffers).await?;
        response.image_embeddings = Some(batch);
        processed_anything = true;
    }

    if !processed_anything {
        return Err(ApiError::ValidationError(
            "Request must contain at least one non-empty text or image input".into(),
        ));
    }

    Ok(response)
}

pub async fn generate_embeddings_handler(
    State(state): State<AppState>,
    AppJson(payload): AppJson<EmbeddingsRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let response = process_embeddings_request(&state, payload).await?;
    Ok((StatusCode::OK, Json(response)))
}
