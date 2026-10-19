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
    let doc_ids: Vec<String> = batch.documents.iter().map(|d| d.id.clone()).collect();
    let mut completed = vec![false; batch.documents.len()];
    let max_concurrency = state.config().model.max_concurrency.max(1) * 2;
    let semaphore = Arc::new(tokio::sync::Semaphore::new(max_concurrency));
    let mut set = tokio::task::JoinSet::new();
    for (idx, doc) in batch.documents.into_iter().enumerate() {
        let state_clone = state.clone();
        let sem = semaphore.clone();
        let doc_id = doc.id.clone();
        set.spawn(async move {
            let _permit = match sem.acquire().await {
                Ok(permit) => permit,
                Err(e) => {
                    return (
                        idx,
                        doc_id,
                        Err(ApiError::Internal(format!(
                            "Concurrency semaphore closed: {e}"
                        ))),
                    );
                }
            };
            let res = prepare_single_point(&state_clone, doc).await;
            (idx, doc_id, res)
        });
    }
    let mut points = Vec::new();
    let mut failed_ids = Vec::new();
    while let Some(join_res) = set.join_next().await {
        match join_res {
            Ok((idx, _id, Ok(point))) => {
                completed[idx] = true;
                points.push(point);
            }
            Ok((idx, id, Err(e))) => {
                completed[idx] = true;
                tracing::warn!(id = %id, error = %e, "Failed to prepare document in batch");
                failed_ids.push(id);
            }
            Err(e) => {
                tracing::error!(error = %e, "Batch worker task panicked or failed");
            }
        }
    }
    for (id, is_done) in doc_ids.into_iter().zip(completed) {
        if !is_done {
            failed_ids.push(id);
        }
    }
    let indexed_count = points.len();
    let mut uploaded_files = Vec::new();
    for pt in &points {
        if let Some(filename) = pt
            .payload
            .get("url")
            .and_then(|u| u.as_str())
            .and_then(|u| u.strip_prefix("/uploads/"))
        {
            uploaded_files.push(filename.to_string());
        }
    }
    if !points.is_empty()
        && let Err(e) = state.vector_store().insert_batch(points).await
    {
        for filename in uploaded_files {
            let file_path = state.uploads_dir().join(filename);
            let _ = tokio::fs::remove_file(&file_path).await;
        }
        return Err(e.into());
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
    let maybe_point = state.vector_store().get(&id).await?;
    let uploaded_filename = maybe_point.as_ref().and_then(|pt| {
        pt.payload
            .get("url")
            .and_then(|u| u.as_str())
            .and_then(|u| u.strip_prefix("/uploads/"))
            .map(|f| f.to_string())
    });
    let deleted = state.vector_store().delete(&id).await?;
    if deleted && let Some(filename) = uploaded_filename {
        let file_path = state.uploads_dir().join(filename);
        let _ = tokio::fs::remove_file(&file_path).await;
    }
    Ok((StatusCode::OK, Json(DeleteResponse { id, deleted })))
}
