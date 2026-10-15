use super::media::{prepare_single_point, process_single_document};
use super::models::{
    BatchIndexRequest, BatchIndexResponse, DeleteResponse, IndexDocumentRequest, MAX_BATCH_SIZE,
};
use crate::{error::ApiError, extractors::AppJson, state::AppState};
use axum::{
    Json,
    extract::{Path as AxumPath, State},
    http::StatusCode,
    response::IntoResponse,
};
use serde_json::json;
use std::sync::Arc;
pub async fn index_document_handler(
    State(state): State<AppState>,
    AppJson(doc): AppJson<IndexDocumentRequest>,
) -> Result<impl IntoResponse, ApiError> {
    process_single_document(&state, doc).await?;
    Ok((StatusCode::CREATED, Json(json!({ "status": "indexed" }))))
}
pub async fn index_batch_handler(
    State(state): State<AppState>,
    AppJson(batch): AppJson<BatchIndexRequest>,
) -> Result<impl IntoResponse, ApiError> {
    if batch.documents.len() > MAX_BATCH_SIZE {
        return Err(ApiError::ValidationError(format!(
            "Batch size {} exceeds maximum allowed limit of {} items",
            batch.documents.len(),
            MAX_BATCH_SIZE
        )));
    }
    let max_concurrency = state.config().model.max_concurrency.max(1) * 2;
    let semaphore = Arc::new(tokio::sync::Semaphore::new(max_concurrency));
    let mut set = tokio::task::JoinSet::new();
    for doc in batch.documents {
        let state_clone = state.clone();
        let sem = semaphore.clone();
        let doc_id = doc.id.clone();
        set.spawn(async move {
            let _permit = match sem.acquire().await {
                Ok(permit) => permit,
                Err(e) => {
                    return (
                        doc_id,
                        Err(ApiError::Internal(format!(
                            "Concurrency semaphore closed: {e}"
                        ))),
                    );
                }
            };
            let res = prepare_single_point(&state_clone, doc).await;
            (doc_id, res)
        });
    }
    let mut points = Vec::new();
    let mut failed_ids = Vec::new();
    while let Some(join_res) = set.join_next().await {
        match join_res {
            Ok((_, Ok(point))) => {
                points.push(point);
            }
            Ok((id, Err(e))) => {
                tracing::warn!(id = %id, error = %e, "Failed to prepare document in batch");
                failed_ids.push(id);
            }
            Err(e) => {
                tracing::error!(error = %e, "Batch worker task panicked or failed");
            }
        }
    }
    let indexed_count = points.len();
    if !points.is_empty() {
        state.vector_store().insert_batch(points).await?;
    }
    Ok((
        StatusCode::OK,
        Json(BatchIndexResponse {
            indexed_count,
            failed_ids,
        }),
    ))
}
pub async fn list_documents_handler(
    State(state): State<AppState>,
) -> Result<impl IntoResponse, ApiError> {
    let documents = state.vector_store().list_documents().await?;
    let count = documents.len();
    Ok((
        StatusCode::OK,
        Json(json!({
            "count": count,
            "documents": documents
        })),
    ))
}
pub async fn delete_document_handler(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<String>,
) -> Result<impl IntoResponse, ApiError> {
    let deleted = state.vector_store().delete(&id).await?;
    Ok((StatusCode::OK, Json(DeleteResponse { id, deleted })))
}
