pub mod context;
pub mod llm;
pub mod models;
pub mod pipeline;
pub use context::{AugmentedContext, ContextBuilder, MAX_CHUNK_CHARS};
pub use llm::{LlmClient, MockLlmClient, OpenAiCompatibleLlmClient};
pub use models::{RagQueryRequest, RagQueryResponse, SourceReference, VisualReference};
pub use pipeline::{
    CITATION_INSTRUCTIONS, DEFAULT_SYSTEM_PROMPT, EmbedderTextAdapter, MAX_PROMPT_CHARS,
    RagPipeline, TextEmbedder,
};
