pub mod citations;
pub mod context;
pub mod llm;
pub mod models;
pub mod pipeline;

pub use citations::{CITATION_INSTRUCTIONS, verify_citations};
pub use context::ContextBuilder;
#[cfg(any(test, feature = "test-utils"))]
pub use llm::MockLlmClient;
pub use llm::{LlmClient, OpenAiCompatibleLlmClient};
pub use models::{RagQueryRequest, RagQueryResponse};
pub use pipeline::{DEFAULT_SYSTEM_PROMPT, MAX_PROMPT_CHARS, MAX_TOP_K, RagPipeline};
