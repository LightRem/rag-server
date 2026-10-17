use super::{
    citations::{CITATION_INSTRUCTIONS, verify_citations},
    context::ContextBuilder,
    llm::LlmClient,
    models::{RagQueryRequest, RagQueryResponse},
};
use crate::domain::Modality;
use crate::error::{GenerationError, InferenceError, RagError};
use crate::storage::{SearchFilter, VectorStore};
use async_trait::async_trait;
use std::sync::Arc;

pub const DEFAULT_SYSTEM_PROMPT: &str = concat!(
    "You are an authoritative, grounded Multimodal Assistant.\n",
    "Your mission is to answer user queries truthfully relying EXCLUSIVELY on the provided context documents and visual evidence.\n",
    "Follow these rules strictly:\n",
    "1. Always cite source identifiers using [Source ID: <id>] format when making factual claims.\n",
    "2. If relevant images are provided, explicitly reference them using [Image ID: <id>] and briefly explain how the image complements the answer.\n",
    "3. Address ONLY what the user asked. Do NOT include tangential advice or cite documents that are irrelevant to the core question, even if they appear in the context.\n",
    "4. If the context does not contain sufficient information to answer the question with high confidence, state clearly that the knowledge base does not have this information. Do not hallucinate."
);

pub const MAX_PROMPT_CHARS: usize = 16_000;
const PROMPT_TEMPLATE_OVERHEAD_CHARS: usize = 200;

fn canonicalize_clip_query(query: &str) -> String {
    let lower = query.to_lowercase();
    const PREFIXES: &[&str] = &[
        "tell me about ",
        "can you tell me about ",
        "what do you know about ",
        "give me a recipe for ",
        "give me recipe for ",
        "how do i cook ",
        "how to cook ",
        "recipe for ",
        "show me ",
        "find me ",
        "explain to me ",
    ];

    for prefix in PREFIXES {
        if lower.starts_with(prefix) {
            return query[prefix.len()..].trim().to_string();
        }
    }
    query.trim().to_string()
}

pub struct EmbedderTextAdapter(pub Arc<dyn crate::inference::Embedder>);

#[async_trait]
pub trait TextEmbedder: Send + Sync {
    async fn embed_text(&self, text: &str) -> Result<Vec<f32>, InferenceError>;
}

#[async_trait]
impl TextEmbedder for EmbedderTextAdapter {
    async fn embed_text(&self, text: &str) -> Result<Vec<f32>, InferenceError> {
        self.0.embed_text(text).await
    }
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

    pub async fn answer(&self, request: RagQueryRequest) -> Result<RagQueryResponse, RagError> {
        let query_trimmed = request.query.trim();
        if query_trimmed.is_empty() {
            return Err(RagError::ValidationError(
                "Search query cannot be empty".into(),
            ));
        }

        let system_prompt = match &request.system_prompt {
            Some(custom) => format!("{custom}\n\n{CITATION_INSTRUCTIONS}"),
            None => DEFAULT_SYSTEM_PROMPT.to_string(),
        };

        let initial_prompt_chars = system_prompt.chars().count() + query_trimmed.chars().count();
        if initial_prompt_chars > MAX_PROMPT_CHARS {
            return Err(RagError::Generation(GenerationError::PromptTooLong {
                length: initial_prompt_chars,
                max: MAX_PROMPT_CHARS,
            }));
        }

        let search_text = canonicalize_clip_query(query_trimmed);
        let query_vector = self.embedder.embed_text(&search_text).await?;

        let filter = SearchFilter::new()
            .with_min_score(request.min_score)
            .with_query_text(&search_text)
            .with_query_modality(Modality::Text);

        let retrieval_k = request.top_k.max(7);
        let search_results = self
            .store
            .search(&query_vector, retrieval_k, Some(filter))
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
            return Err(RagError::Generation(GenerationError::PromptTooLong {
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

        let mut sources = augmented.sources;
        let mut visuals = augmented.visuals;
        verify_citations(&answer, &mut sources, &mut visuals);

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
