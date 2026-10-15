use crate::domain::{DocumentId, Modality};
use serde::{Deserialize, Serialize};
use serde_json::Value;
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VectorPoint {
    pub id: DocumentId,
    pub vector: Vec<f32>,
    pub payload: Value,
    pub modality: Modality,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_vector: Option<Vec<f32>>,
}
impl VectorPoint {
    pub fn new(
        id: impl Into<DocumentId>,
        vector: Vec<f32>,
        payload: Value,
        modality: Modality,
    ) -> Self {
        Self {
            id: id.into(),
            vector,
            payload,
            modality,
            image_vector: None,
        }
    }
    pub fn with_image_vector(mut self, image_vector: Vec<f32>) -> Self {
        self.image_vector = Some(image_vector);
        self
    }
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SearchResult {
    pub id: DocumentId,
    pub score: f32,
    pub payload: Value,
    pub modality: Modality,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub matched_by: Option<Modality>,
}
#[derive(Debug, Clone, Default)]
pub struct SearchFilter {
    pub min_score: Option<f32>,
    pub modality: Option<Modality>,
    pub query_text: Option<String>,
    pub query_modality: Option<Modality>,
}
impl SearchFilter {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_min_score(mut self, min_score: f32) -> Self {
        self.min_score = Some(min_score);
        self
    }
    pub fn with_modality(mut self, modality: Modality) -> Self {
        self.modality = Some(modality);
        self
    }
    pub fn with_query_text(mut self, query_text: impl Into<String>) -> Self {
        self.query_text = Some(query_text.into());
        self
    }
    pub fn with_query_modality(mut self, modality: Modality) -> Self {
        self.query_modality = Some(modality);
        self
    }
}
