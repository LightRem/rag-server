use super::{
    context::ContextBuilder,
    llm::LlmClient,
    models::{RagQueryRequest, RagQueryResponse},
};
use crate::domain::Modality;
use crate::error::{ApiError, GenerationError, InferenceError};
use crate::storage::{SearchFilter, VectorStore};
use async_trait::async_trait;
use std::sync::Arc;
pub const DEFAULT_SYSTEM_PROMPT: &str = r#"You are an authoritative, grounded Multimodal Assistant.
Your mission is to answer user queries truthfully relying EXCLUSIVELY on the provided context documents and visual evidence.
Follow these rules strictly:
1. Always cite source identifiers using [Source ID: <id>] format when making factual claims.
2. If relevant images are provided, explicitly reference them using [Image ID: <id>] and briefly explain how the image complements the answer.
3. If the context does not contain sufficient information to answer the question with high confidence, state clearly that the knowledge base does not have this information. Do not hallucinate."#;
pub const CITATION_INSTRUCTIONS: &str = r#"### MANDATORY CITATION AND EVIDENCE RULES:
1. Always cite source identifiers using [Source ID: <id>] format when making factual claims.
2. If relevant images are provided, explicitly reference them using [Image ID: <id>] and briefly explain how the image complements the answer.
3. If the context does not contain sufficient information to answer the question with high confidence, state clearly that the knowledge base does not have this information. Do not hallucinate."#;
pub const MAX_PROMPT_CHARS: usize = 16_000;
const PROMPT_TEMPLATE_OVERHEAD_CHARS: usize = 200;
pub struct EmbedderTextAdapter(pub Arc<dyn crate::inference::Embedder>);
#[async_trait]
pub trait TextEmbedder: Send + Sync {
    async fn embed_text(&self, text: &str) -> Result<Vec<f32>, InferenceError>;
}
fn extract_bracketed_tokens(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut in_bracket = false;
    let mut current = String::new();
    for ch in text.chars() {
        match ch {
            '[' => {
                in_bracket = true;
                current.clear();
            }
            ']' => {
                if in_bracket {
                    in_bracket = false;
                    for part in current.split([',', ';']) {
                        let trimmed = part.trim();
                        if !trimmed.is_empty() {
                            tokens.push(trimmed.to_lowercase());
                        }
                    }
                    current.clear();
                }
            }
            _ => {
                if in_bracket {
                    current.push(ch);
                }
            }
        }
    }
    tokens
}
fn is_item_cited(tokens: &[String], id: &str, prefixes: &[&str]) -> bool {
    let id_lower = id.trim().to_lowercase();
    if id_lower.is_empty() {
        return false;
    }
    let is_pure_numeric = id_lower.chars().all(|c| c.is_ascii_digit());
    for token in tokens {
        if token == &id_lower {
            if !is_pure_numeric || id_lower.len() >= 4 {
                return true;
            }
        }
        for &prefix in prefixes {
            let p_lower = prefix.to_lowercase();
            if token.starts_with(&p_lower) {
                let remainder = token[p_lower.len()..].trim();
                if remainder == id_lower {
                    return true;
                }
            }
        }
    }
    false
}
pub struct RagPipeline {
    store: Arc<dyn VectorStore>,
    embedder: Arc<dyn TextEmbedder>,
    llm: Arc<dyn LlmClient>,
}
impl RagPipeline {
    pub fn new(
        store: Arc<dyn VectorStore>,
        embedder: Arc<dyn TextEmbedder>,
        llm: Arc<dyn LlmClient>,
    ) -> Self {
        Self {
            store,
            embedder,
            llm,
        }
    }
    pub async fn answer(&self, request: RagQueryRequest) -> Result<RagQueryResponse, ApiError> {
        let query_trimmed = request.query.trim();
        if query_trimmed.is_empty() {
            return Err(ApiError::ValidationError(
                "Search query cannot be empty".into(),
            ));
        }
        let system_prompt = match &request.system_prompt {
            Some(custom) => format!("{custom}\n\n{CITATION_INSTRUCTIONS}"),
            None => DEFAULT_SYSTEM_PROMPT.to_string(),
        };
        let initial_prompt_chars = system_prompt.chars().count() + query_trimmed.chars().count();
        if initial_prompt_chars > MAX_PROMPT_CHARS {
            return Err(ApiError::Generation(GenerationError::PromptTooLong {
                length: initial_prompt_chars,
                max: MAX_PROMPT_CHARS,
            }));
        }
        let query_vector = self.embedder.embed_text(query_trimmed).await?;
        let filter = SearchFilter::new()
            .with_min_score(request.min_score)
            .with_query_text(query_trimmed)
            .with_query_modality(Modality::Text);
        let search_results = self
            .store
            .search(&query_vector, request.top_k, Some(filter))
            .await?;
        if search_results.is_empty() && !request.allow_empty_context {
            return Ok(RagQueryResponse {
                query: request.query,
                answer: "Information matching the query was not found in the knowledge base."
                    .to_string(),
                sources: Vec::new(),
                visuals: Vec::new(),
            });
        }
        let max_context_chars = MAX_PROMPT_CHARS
            .saturating_sub(initial_prompt_chars)
            .saturating_sub(PROMPT_TEMPLATE_OVERHEAD_CHARS);
        let augmented = ContextBuilder::build(search_results, max_context_chars);
        let user_prompt = format!(
            "{}\n\n### USER QUESTION:\n{}\n\n### FINAL ANSWER:",
            augmented.context_prompt, query_trimmed
        );
        let total_prompt_chars = system_prompt.chars().count() + user_prompt.chars().count();
        if total_prompt_chars > MAX_PROMPT_CHARS {
            return Err(ApiError::Generation(GenerationError::PromptTooLong {
                length: total_prompt_chars,
                max: MAX_PROMPT_CHARS,
            }));
        }
        let answer = self
            .llm
            .generate(
                &system_prompt,
                &user_prompt,
                request.temperature,
                request.max_tokens,
            )
            .await?;
        let tokens = extract_bracketed_tokens(&answer);
        const SOURCE_PREFIXES: &[&str] = &["source id:", "source:", "doc id:", "doc:", "id:"];
        const VISUAL_PREFIXES: &[&str] = &["image id:", "image:", "img id:", "img:"];
        let mut sources = augmented.sources;
        for s in &mut sources {
            s.is_cited = is_item_cited(&tokens, &s.id, SOURCE_PREFIXES);
        }
        let mut visuals = augmented.visuals;
        for v in &mut visuals {
            v.is_cited = is_item_cited(&tokens, &v.id, VISUAL_PREFIXES);
        }
        if request.filter_uncited {
            sources.retain(|s| s.is_cited);
            visuals.retain(|v| v.is_cited);
        }
        Ok(RagQueryResponse {
            query: request.query,
            answer,
            sources,
            visuals,
        })
    }
}
#[async_trait]
impl TextEmbedder for EmbedderTextAdapter {
    async fn embed_text(&self, text: &str) -> Result<Vec<f32>, InferenceError> {
        self.0.embed_text(text).await
    }
}
