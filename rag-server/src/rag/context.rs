use super::models::{SourceReference, VisualReference};
use crate::domain::Modality;
use crate::storage::SearchResult;
pub const MAX_CHUNK_CHARS: usize = 2000;
pub struct AugmentedContext {
    pub context_prompt: String,
    pub sources: Vec<SourceReference>,
    pub visuals: Vec<VisualReference>,
}
pub struct ContextBuilder;
impl ContextBuilder {
    pub fn build(results: Vec<SearchResult>, max_context_chars: usize) -> AugmentedContext {
        let mut text_blocks = Vec::new();
        let mut image_blocks = Vec::new();
        let mut sources = Vec::new();
        let mut visuals = Vec::new();
        let mut used_chars = 0;
        for item in results {
            let has_text = matches!(item.modality, Modality::Text | Modality::Multimodal);
            let has_image = matches!(item.modality, Modality::Image | Modality::Multimodal);
            if has_text {
                let raw_text = item
                    .payload
                    .get("text")
                    .or_else(|| item.payload.get("content"))
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| {
                        if item.modality == Modality::Multimodal && !item.payload.is_object() {
                            String::new()
                        } else {
                            item.payload.to_string()
                        }
                    });
                if !raw_text.trim().is_empty() {
                    let title = item
                        .payload
                        .get("title")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());
                    let trimmed = raw_text.trim();
                    let bounded_text = if trimmed.chars().count() > MAX_CHUNK_CHARS {
                        let prefix: String = trimmed.chars().take(MAX_CHUNK_CHARS).collect();
                        format!("{prefix}... [content truncated]")
                    } else {
                        trimmed.to_string()
                    };
                    let block = format!(
                        "--- DOCUMENT [Source ID: {}] (Score: {:.2}) ---\nTitle: {}\nContent:\n{}\n\n",
                        item.id,
                        item.score,
                        title.as_deref().unwrap_or("Untitled"),
                        bounded_text
                    );
                    let block_chars = block.chars().count();
                    if used_chars + block_chars <= max_context_chars {
                        used_chars += block_chars;
                        text_blocks.push(block);
                        sources.push(SourceReference {
                            id: item.id.clone(),
                            title,
                            snippet: raw_text.chars().take(240).collect(),
                            score: item.score,
                            is_cited: false,
                        });
                    }
                }
            }
            if has_image {
                let url_opt = item
                    .payload
                    .get("url")
                    .or_else(|| item.payload.get("image_url"))
                    .and_then(|v| v.as_str());
                let caption = item
                    .payload
                    .get("caption")
                    .or_else(|| item.payload.get("description"))
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                let url = url_opt
                    .map(|u| u.to_string())
                    .unwrap_or_else(|| format!("embedded://image/{}", item.id));
                let block = format!(
                    "--- ATTACHED IMAGE [Image ID: {}] (Score: {:.2}) ---\nCaption: {}\nURL: {}\n\n",
                    item.id,
                    item.score,
                    caption.as_deref().unwrap_or("No description available"),
                    url
                );
                let block_chars = block.chars().count();
                if used_chars + block_chars <= max_context_chars {
                    used_chars += block_chars;
                    image_blocks.push(block);
                    visuals.push(VisualReference {
                        id: item.id.clone(),
                        url,
                        caption,
                        score: item.score,
                        is_cited: false,
                    });
                }
            }
        }
        let mut prompt = String::new();
        if !text_blocks.is_empty() {
            prompt.push_str("### RETRIEVED TEXT CONTEXT:\n");
            for block in text_blocks {
                prompt.push_str(&block);
            }
        }
        if !image_blocks.is_empty() {
            prompt.push_str("### RELEVANT VISUAL EVIDENCE:\n");
            for block in image_blocks {
                prompt.push_str(&block);
            }
        }
        AugmentedContext {
            context_prompt: prompt,
            sources,
            visuals,
        }
    }
}
