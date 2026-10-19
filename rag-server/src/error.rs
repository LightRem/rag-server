use crate::domain::DomainError;
use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde_json::json;
use std::path::PathBuf;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ConfigError {
    #[error("Configuration loading failed: {0}")]
    Load(String),
    #[error("Invalid configuration parameter: field '{field}' - {reason}")]
    InvalidParameter { field: &'static str, reason: String },
}

#[derive(Error, Debug)]
pub enum InferenceError {
    #[error("Model file not found at path: {path}")]
    ModelNotFound { path: PathBuf },

    #[error("Failed to load model: {0}")]
    ModelLoad(String),

    #[error("Preprocessing failed: {0}")]
    Preprocessing(String),

    #[error("Failed to decode image: {reason} (mime: {mime_type:?})")]
    ImageDecode {
        reason: String,
        mime_type: Option<String>,
    },

    #[error("Tensor shape mismatch: expected {expected:?}, got {actual:?}")]
    TensorShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("Inference execution timed out: {0}")]
    Timeout(String),

    #[error("Runtime inference failure: {0}")]
    Execution(String),
}

#[derive(Error, Debug)]
pub enum StorageError {
    #[error("Vector dimension mismatch: store expects {expected}, vector is {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Invalid vector data: {0}")]
    InvalidVector(String),

    #[error("Invalid search filter: {0}")]
    InvalidFilter(String),

    #[error("Storage capacity exceeded: current {current}, max allowed {max}")]
    CapacityExceeded { current: usize, max: usize },

    #[error("Metadata payload serialization failed: {0}")]
    Serialization(String),

    #[error("Storage operation timed out: {0}")]
    Timeout(String),

    #[error("Database operation failed: {0}")]
    Internal(String),
}

#[derive(Error, Debug)]
pub enum GenerationError {
    #[error("LLM upstream request failed: {0}")]
    RequestFailed(String),

    #[error("LLM provider timed out: {0}")]
    Timeout(String),

    #[error("Failed to parse LLM response: {0}")]
    ParseError(String),

    #[error("Prompt length {length} exceeds maximum supported tokens {max}")]
    PromptTooLong { length: usize, max: usize },

    #[error("LLM returned an empty response")]
    EmptyResponse,
}

#[derive(Error, Debug)]
pub enum RagError {
    #[error("Validation error: {0}")]
    ValidationError(String),

    #[error(transparent)]
    Inference(#[from] InferenceError),

    #[error(transparent)]
    Storage(#[from] StorageError),

    #[error(transparent)]
    Generation(#[from] GenerationError),
}

#[derive(Error, Debug)]
pub enum ApiError {
    #[error("Internal server error: {0}")]
    Internal(String),

    #[error(transparent)]
    Config(#[from] ConfigError),

    #[error(transparent)]
    Domain(#[from] DomainError),

    #[error(transparent)]
    Inference(#[from] InferenceError),

    #[error(transparent)]
    Storage(#[from] StorageError),

    #[error(transparent)]
    Generation(#[from] GenerationError),

    #[error(transparent)]
    Rag(#[from] RagError),

    #[error("Payload validation failed: {0}")]
    ValidationError(String),

    #[error("Payload size {size} exceeds allowed limit of {max_allowed} bytes")]
    PayloadTooLarge { size: usize, max_allowed: usize },

    #[error("Payload limit exceeded: {0}")]
    PayloadLimitExceeded(String),

    #[error("Unsupported media type: {0}")]
    UnsupportedMediaType(String),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, client_message) = match self {
            ApiError::Rag(rag_err) => match rag_err {
                RagError::ValidationError(msg) => (StatusCode::BAD_REQUEST, msg),
                RagError::Inference(e) => return ApiError::Inference(e).into_response(),
                RagError::Storage(e) => return ApiError::Storage(e).into_response(),
                RagError::Generation(e) => return ApiError::Generation(e).into_response(),
            },

            ApiError::Domain(DomainError::InvalidAllowedRoot { path, source }) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Invalid allowed root '{path:?}': {source}"),
            ),
            ApiError::Domain(e) => (StatusCode::BAD_REQUEST, e.to_string()),

            ApiError::Internal(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
            ApiError::ValidationError(msg) => (StatusCode::BAD_REQUEST, msg),
            ApiError::PayloadTooLarge { size, max_allowed } => (
                StatusCode::PAYLOAD_TOO_LARGE,
                format!("Payload size {size} bytes exceeds limit of {max_allowed} bytes"),
            ),
            ApiError::PayloadLimitExceeded(msg) => (StatusCode::PAYLOAD_TOO_LARGE, msg),
            ApiError::UnsupportedMediaType(msg) => (StatusCode::UNSUPPORTED_MEDIA_TYPE, msg),

            ApiError::Inference(InferenceError::Preprocessing(reason)) => (
                StatusCode::BAD_REQUEST,
                format!("Input preprocessing error: {reason}"),
            ),

            ApiError::Inference(InferenceError::ImageDecode { reason, .. }) => (
                StatusCode::UNPROCESSABLE_ENTITY,
                format!("Invalid image format: {reason}"),
            ),

            ApiError::Inference(InferenceError::TensorShapeMismatch { .. }) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Internal model inference failure: tensor shape mismatch".to_string(),
            ),

            ApiError::Inference(InferenceError::Timeout(reason)) => (
                StatusCode::GATEWAY_TIMEOUT,
                format!("Inference execution timed out: {reason}"),
            ),

            ApiError::Storage(StorageError::DimensionMismatch { expected, actual }) => (
                StatusCode::BAD_REQUEST,
                format!("Vector dimension mismatch: expected {expected}, got {actual}"),
            ),

            ApiError::Storage(StorageError::InvalidVector(reason)) => {
                (StatusCode::BAD_REQUEST, format!("Invalid vector: {reason}"))
            }

            ApiError::Storage(StorageError::InvalidFilter(reason)) => (
                StatusCode::BAD_REQUEST,
                format!("Invalid search filter: {reason}"),
            ),

            ApiError::Storage(StorageError::Serialization(reason)) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Document serialization failed: {reason}"),
            ),

            ApiError::Storage(StorageError::CapacityExceeded { current, max }) => (
                StatusCode::INSUFFICIENT_STORAGE,
                format!("Storage capacity limit reached ({current}/{max})"),
            ),

            ApiError::Storage(StorageError::Timeout(reason)) => (
                StatusCode::GATEWAY_TIMEOUT,
                format!("Storage operation timed out: {reason}"),
            ),

            ApiError::Generation(GenerationError::RequestFailed(reason)) => (
                StatusCode::BAD_GATEWAY,
                format!("LLM upstream communication error: {reason}"),
            ),

            ApiError::Generation(GenerationError::Timeout(reason)) => (
                StatusCode::GATEWAY_TIMEOUT,
                format!("LLM provider timed out: {reason}"),
            ),

            ApiError::Generation(GenerationError::ParseError(reason)) => (
                StatusCode::BAD_GATEWAY,
                format!("Malformed completion payload from LLM provider: {reason}"),
            ),

            ApiError::Generation(GenerationError::PromptTooLong { length, max }) => (
                StatusCode::BAD_REQUEST,
                format!("Prompt tokens {length} exceed maximum window limit {max}"),
            ),

            ApiError::Generation(GenerationError::EmptyResponse) => (
                StatusCode::BAD_GATEWAY,
                "LLM provider returned an empty completion".to_string(),
            ),

            err => {
                tracing::error!(error = %err, "Unhandled internal error processing request");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "An unexpected internal error occurred".to_string(),
                )
            }
        };

        let body = Json(json!({
            "error": {
                "message": client_message,
                "status_code": status.as_u16(),
            }
        }));
        (status, body).into_response()
    }
}
