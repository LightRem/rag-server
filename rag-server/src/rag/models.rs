use crate::domain::ImageSource;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagQueryRequest {
    pub query: String,
    #[serde(default)]
    pub image: Option<ImageSource>,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    #[serde(default = "default_min_score")]
    pub min_score: f32,
    pub system_prompt: Option<String>,
    #[serde(default)]
    pub filter_uncited: bool,
    #[serde(default)]
    pub allow_empty_context: bool,
    pub temperature: Option<f32>,
    pub max_tokens: Option<usize>,
}

fn default_top_k() -> usize {
    5
}

fn default_min_score() -> f32 {
    0.20
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SourceReference {
    pub id: String,
    pub title: Option<String>,
    pub snippet: String,
    pub score: f32,
    #[serde(default)]
    pub is_cited: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VisualReference {
    pub id: String,
    pub url: String,
    pub caption: Option<String>,
    pub score: f32,
    #[serde(default)]
    pub is_cited: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RagQueryResponse {
    pub query: String,
    pub answer: String,
    pub sources: Vec<SourceReference>,
    pub visuals: Vec<VisualReference>,
}
