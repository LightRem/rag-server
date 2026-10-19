use super::{
    citations::{CITATION_INSTRUCTIONS, verify_citations},
    context::ContextBuilder,
    llm::LlmClient,
    models::{RagQueryRequest, RagQueryResponse},
};
use crate::domain::Modality;
use crate::error::{GenerationError, RagError};
use crate::inference::Embedder;
use crate::storage::{SearchFilter, VectorStore};
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
pub const MAX_TOP_K: usize = 100;
pub const MAX_TOKENS_LIMIT: usize = 8192;
const PROMPT_TEMPLATE_OVERHEAD_CHARS: usize = 200;

fn strip_prefix_ignore_case<'a>(s: &'a str, prefix: &str) -> Option<&'a str> {
    let mut s_chars = s.chars();
    let mut p_chars = prefix.chars();

    while let Some(p) = p_chars.next() {
        let s_char = s_chars.next()?;
        let mut s_lower = s_char.to_lowercase();

        if s_lower.next() != Some(p) {
            return None;
        }

        for extra_p in s_lower {
            if p_chars.next() != Some(extra_p) {
                return None;
            }
        }
    }

    Some(s_chars.as_str())
}

fn canonicalize_clip_query(query: &str) -> String {
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
        if let Some(stripped) = strip_prefix_ignore_case(query, prefix) {
            return stripped.trim().to_string();
        }
    }

    query.trim().to_string()
}

pub struct RagPipeline {
    store: Arc<dyn VectorStore>,
    embedder: Arc<dyn Embedder>,
    llm: Arc<dyn LlmClient>,
}

impl RagPipeline {
    pub fn new(
        store: Arc<dyn VectorStore>,
        embedder: Arc<dyn Embedder>,
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

        if request.top_k == 0 || request.top_k > MAX_TOP_K {
            return Err(RagError::ValidationError(format!(
                "top_k must be between 1 and {MAX_TOP_K}"
            )));
        }

        if !request.min_score.is_finite() || !(-1.0..=1.0).contains(&request.min_score) {
            return Err(RagError::ValidationError(
                "min_score must be between -1.0 and 1.0".into(),
            ));
        }

        if let Some(temp) = request.temperature
            && (!temp.is_finite() || !(0.0..=2.0).contains(&temp))
        {
            return Err(RagError::ValidationError(
                "temperature must be between 0.0 and 2.0".into(),
            ));
        }

        if let Some(tokens) = request.max_tokens
            && (tokens == 0 || tokens > MAX_TOKENS_LIMIT)
        {
            return Err(RagError::ValidationError(format!(
                "max_tokens must be between 1 and {MAX_TOKENS_LIMIT}"
            )));
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
