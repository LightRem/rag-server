use super::models::{SourceReference, VisualReference};

pub const CITATION_INSTRUCTIONS: &str = r#"### MANDATORY CITATION AND EVIDENCE RULES:
1. Always cite source identifiers using [Source ID: <id>] format when making factual claims.
2. If relevant images are provided, explicitly reference them using [Image ID: <id>] and briefly explain how the image complements the answer.
3. If the context does not contain sufficient information to answer the question with high confidence, state clearly that the knowledge base does not have this information. Do not hallucinate."#;

const SOURCE_PREFIXES: &[&str] = &["source id:", "source:", "doc id:", "doc:", "id:"];
const VISUAL_PREFIXES: &[&str] = &["image id:", "image:", "img id:", "img:"];

pub fn extract_bracketed_tokens(text: &str) -> Vec<String> {
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

pub fn is_item_cited(tokens: &[String], id: &str, prefixes: &[&str]) -> bool {
    let id_lower = id.trim().to_lowercase();
    if id_lower.is_empty() {
        return false;
    }
    let is_pure_numeric = id_lower.chars().all(|c| c.is_ascii_digit());

    for token in tokens {
        if token == &id_lower && (!is_pure_numeric || id_lower.len() >= 4) {
            return true;
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

pub fn verify_citations(
    answer: &str,
    sources: &mut [SourceReference],
    visuals: &mut [VisualReference],
) {
    let tokens = extract_bracketed_tokens(answer);

    for s in sources.iter_mut() {
        s.is_cited = is_item_cited(&tokens, &s.id, SOURCE_PREFIXES);
    }

    for v in visuals.iter_mut() {
        v.is_cited = is_item_cited(&tokens, &v.id, VISUAL_PREFIXES);
    }
}
