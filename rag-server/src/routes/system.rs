use super::models::{EmbeddingsRequest, MAX_BATCH_SIZE, StatsResponse};
use crate::{domain::ImageSource, error::ApiError, extractors::AppJson, state::AppState};
use axum::{Json, extract::State, http::StatusCode, response::IntoResponse};
use serde_json::json;

pub async fn health_check_handler() -> impl IntoResponse {
    (
        StatusCode::OK,
        Json(json!({
            "status": "healthy",
            "service": "multimodal-rag-server"
        })),
    )
}

pub async fn validate_payload_handler(
    State(state): State<AppState>,
    AppJson(payload): AppJson<EmbeddingsRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let has_text = payload.text.is_some() || payload.texts.as_ref().is_some_and(|t| !t.is_empty());
    let has_image =
        payload.image.is_some() || payload.images.as_ref().is_some_and(|i| !i.is_empty());
    if !has_text && !has_image {
        return Err(ApiError::ValidationError(
            "Payload must contain at least one text or image field".into(),
        ));
    }

    if let Some(texts) = payload.texts.as_ref().filter(|t| t.len() > MAX_BATCH_SIZE) {
        return Err(ApiError::ValidationError(format!(
            "Batch size {} exceeds maximum allowed limit of {} items",
            texts.len(),
            MAX_BATCH_SIZE
        )));
    }

    if let Some(images) = payload.images.as_ref().filter(|i| i.len() > MAX_BATCH_SIZE) {
        return Err(ApiError::ValidationError(format!(
            "Batch size {} exceeds maximum allowed limit of {} items",
            images.len(),
            MAX_BATCH_SIZE
        )));
    }

    let allowed_root = state.uploads_dir();
    if let Some(img @ ImageSource::Path(_)) = &payload.image {
        img.validate_safe_path(allowed_root)
            .map_err(ApiError::ValidationError)?;
    }

    if let Some(ref images) = payload.images {
        for img in images {
            if matches!(img, ImageSource::Path(_)) {
                img.validate_safe_path(allowed_root)
                    .map_err(ApiError::ValidationError)?;
            }
        }
    }

    Ok((
        StatusCode::OK,
        Json(json!({
            "status": "valid",
            "has_text": has_text,
            "has_image": has_image
        })),
    ))
}

pub async fn get_stats_handler(
    State(state): State<AppState>,
) -> Result<impl IntoResponse, ApiError> {
    let count = state.vector_store().count().await?;
    let dimension = state.vector_store().dimension();
    Ok((
        StatusCode::OK,
        Json(StatsResponse {
            total_documents: count,
            vector_dimension: dimension,
            status: "operational".to_string(),
        }),
    ))
}
