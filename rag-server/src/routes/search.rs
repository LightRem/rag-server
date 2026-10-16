use super::media::load_image_bytes;
use super::models::{SearchRequest, SearchResponse};
use crate::{
    domain::Modality, error::ApiError, extractors::AppJson, rag::RagQueryRequest, state::AppState,
    storage::SearchFilter,
};
use axum::{Json, extract::State, http::StatusCode, response::IntoResponse};

pub const MIN_SEARCH_TOP_K: usize = 1;
pub const MAX_SEARCH_TOP_K: usize = 100;

fn fuse_query_vectors(text_vec: &[f32], img_vec: &[f32]) -> Result<Vec<f32>, ApiError> {
    let mut fused_vec = Vec::with_capacity(text_vec.len());
    for (t, i) in text_vec.iter().zip(img_vec.iter()) {
        fused_vec.push(0.5 * t + 0.5 * i);
    }
    crate::inference::l2_normalize(&mut fused_vec).map_err(ApiError::Inference)?;
    Ok(fused_vec)
}

pub async fn search_documents_handler(
    State(state): State<AppState>,
    AppJson(payload): AppJson<SearchRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let query_trimmed = payload
        .query
        .as_ref()
        .map(|q| q.trim())
        .filter(|q| !q.is_empty());
    let has_image = payload.image.is_some();

    if query_trimmed.is_none() && !has_image {
        return Err(ApiError::ValidationError(
            "Search request must contain at least a non-empty text 'query' or an 'image'".into(),
        ));
    }

    let allowed_root = state.uploads_dir();
    let max_image_bytes = state.config().server.max_body_limit_mb * 1024 * 1024;

    let (query_vector, query_modality) = match (query_trimmed, payload.image.as_ref()) {
        (Some(text), Some(img_source)) => {
            let bytes = load_image_bytes(
                img_source,
                allowed_root,
                max_image_bytes,
                state.http_client(),
            )
            .await?;
            let text_vec = state.embedder().embed_text(text).await?;
            let img_vec = state.embedder().embed_image(&bytes).await?;
            let fused = fuse_query_vectors(&text_vec, &img_vec)?;
            (fused, Modality::Multimodal)
        }
        (Some(text), None) => (state.embedder().embed_text(text).await?, Modality::Text),
        (None, Some(img_source)) => {
            let bytes = load_image_bytes(
                img_source,
                allowed_root,
                max_image_bytes,
                state.http_client(),
            )
            .await?;
            (state.embedder().embed_image(&bytes).await?, Modality::Image)
        }
        (None, None) => unreachable!(),
    };

    let mut filter = SearchFilter::new()
        .with_min_score(payload.min_score)
        .with_query_modality(query_modality);

    if let Some(text) = query_trimmed {
        filter = filter.with_query_text(text);
    }
    if let Some(m) = payload.modality {
        filter = filter.with_modality(m);
    }

    let bounded_top_k = payload.top_k.clamp(MIN_SEARCH_TOP_K, MAX_SEARCH_TOP_K);
    let results = state
        .vector_store()
        .search(&query_vector, bounded_top_k, Some(filter))
        .await?;

    Ok((
        StatusCode::OK,
        Json(SearchResponse {
            query: query_trimmed.map(|s| s.to_string()),
            has_image_query: has_image,
            results,
        }),
    ))
}

pub async fn rag_query_handler(
    State(state): State<AppState>,
    AppJson(payload): AppJson<RagQueryRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let response = state.pipeline().answer(payload).await?;
    Ok((StatusCode::OK, Json(response)))
}
