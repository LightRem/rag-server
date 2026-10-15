use super::models::{SearchFilter, SearchResult, VectorPoint};
use crate::domain::{DocumentId, Modality};
use crate::error::StorageError;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::Path;
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentSummary {
    pub id: DocumentId,
    pub modality: Modality,
    pub payload: Value,
}
#[async_trait]
pub trait VectorStore: Send + Sync {
    async fn insert(&self, point: VectorPoint) -> Result<(), StorageError>;
    async fn insert_batch(&self, points: Vec<VectorPoint>) -> Result<(), StorageError>;
    async fn search(
        &self,
        query_vector: &[f32],
        limit: usize,
        filter: Option<SearchFilter>,
    ) -> Result<Vec<SearchResult>, StorageError>;
    async fn get(&self, id: &DocumentId) -> Result<Option<VectorPoint>, StorageError>;
    async fn delete(&self, id: &DocumentId) -> Result<bool, StorageError>;
    async fn list_documents(&self) -> Result<Vec<DocumentSummary>, StorageError> {
        Ok(Vec::new())
    }
    async fn count(&self) -> Result<usize, StorageError>;
    fn dimension(&self) -> usize;
    async fn save_to_disk(&self, path: &Path) -> Result<(), StorageError> {
        let _ = path;
        Ok(())
    }
    async fn load_from_disk(&self, path: &Path) -> Result<usize, StorageError> {
        let _ = path;
        Ok(0)
    }
}
