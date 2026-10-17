pub mod documents;
pub mod embeddings;
pub mod media;
pub mod models;
pub mod search;
pub mod security;
pub mod system;

use crate::state::AppState;
use axum::{
    Router,
    response::Html,
    routing::{delete, get, post},
};
pub use models::MAX_BATCH_SIZE;
use tower_http::{limit::RequestBodyLimitLayer, services::ServeDir, trace::TraceLayer};

async fn index_page_handler() -> Html<&'static str> {
    Html(include_str!("../../static/index.html"))
}

pub fn build_router(state: AppState) -> Router {
    let max_body_bytes = state.config().server.max_body_limit_bytes();
    let uploads_path = state.uploads_dir().to_path_buf();
    let _ = std::fs::create_dir_all(&uploads_path);

    Router::new()
        .route("/", get(index_page_handler))
        .nest_service("/uploads", ServeDir::new(uploads_path))
        .route("/health", get(system::health_check_handler))
        .route(
            "/api/v1/payload/validate",
            post(system::validate_payload_handler),
        )
        .route("/api/v1/stats", get(system::get_stats_handler))
        .route(
            "/api/v1/embeddings",
            post(embeddings::generate_embeddings_handler),
        )
        .route("/api/v1/search", post(search::search_documents_handler))
        .route("/api/v1/rag", post(search::rag_query_handler))
        .route("/api/v1/index", post(documents::index_document_handler))
        .route("/api/v1/index/batch", post(documents::index_batch_handler))
        .route("/api/v1/documents", get(documents::list_documents_handler))
        .route(
            "/api/v1/documents/:id",
            delete(documents::delete_document_handler),
        )
        .layer(RequestBodyLimitLayer::new(max_body_bytes))
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}
