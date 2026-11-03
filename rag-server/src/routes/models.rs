use crate::domain::{ImageSource, Modality};
use crate::storage::SearchResult;
use serde::{Deserialize, Serialize};
pub const MAX_BATCH_SIZE: usize = 64;

#[derive(Debug, Deserialize)]
pub struct EmbeddingsRequest {
    pub text: Option<String>,
    pub texts: Option<Vec<String>>,
    pub image: Option<ImageSource>,
    pub images: Option<Vec<ImageSource>>,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingsResponse {
    pub dimension: usize,
    pub text_embedding: Option<Vec<f32>>,
    pub text_embeddings: Option<Vec<Vec<f32>>>,
    pub image_embedding: Option<Vec<f32>>,
    pub image_embeddings: Option<Vec<Vec<f32>>>,
}

#[derive(Debug, Deserialize)]
pub struct IndexDocumentRequest {
    pub id: String,
    pub text: Option<String>,
    pub image: Option<ImageSource>,
    pub title: Option<String>,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub struct BatchIndexRequest {
    pub documents: Vec<IndexDocumentRequest>,
}

#[derive(Debug, Serialize)]
pub struct BatchIndexResponse {
    pub indexed_count: usize,
    pub failed_ids: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct DeleteResponse {
    pub id: String,
    pub deleted: bool,
}

#[derive(Debug, Deserialize)]
pub struct SearchRequest {
    pub query: Option<String>,
    pub image: Option<ImageSource>,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    #[serde(default = "default_min_score")]
    pub min_score: f32,
    pub modality: Option<Modality>,
}

fn default_top_k() -> usize {
    5
}

fn default_min_score() -> f32 {
    0.20
}

#[derive(Debug, Serialize)]
pub struct SearchResponse {
    pub query: Option<String>,
    pub has_image_query: bool,
    pub results: Vec<SearchResult>,
}

#[derive(Debug, Serialize)]
pub struct StatsResponse {
    pub total_documents: usize,
    pub vector_dimension: usize,
    pub status: String,
}
