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

pub fn extract_citation_id<'a>(token: &'a str, prefixes: &[&str]) -> Option<&'a str> {
    for &prefix in prefixes {
        if let Some(rest) = token.strip_prefix(prefix) {
            let id = rest.trim();
            if !id.is_empty() {
                return Some(id);
            }
        }
    }
    None
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
        if let Some(remainder) = extract_citation_id(token, prefixes)
            && remainder == id_lower
        {
            return true;
        }
    }
    false
}

pub fn detect_hallucinated_citations(
    tokens: &[String],
    sources: &[SourceReference],
    visuals: &[VisualReference],
) -> Vec<String> {
    let mut hallucinations: Vec<String> = Vec::new();

    for token in tokens {
        if let Some(id) = extract_citation_id(token, SOURCE_PREFIXES) {
            let exists = sources.iter().any(|s| s.id.trim().eq_ignore_ascii_case(id));
            if !exists
                && !hallucinations
                    .iter()
                    .any(|h: &String| h.eq_ignore_ascii_case(id))
            {
                hallucinations.push(id.to_string());
            }
        } else if let Some(id) = extract_citation_id(token, VISUAL_PREFIXES) {
            let exists = visuals.iter().any(|v| v.id.trim().eq_ignore_ascii_case(id));
            if !exists
                && !hallucinations
                    .iter()
                    .any(|h: &String| h.eq_ignore_ascii_case(id))
            {
                hallucinations.push(id.to_string());
            }
        }
    }

    hallucinations
}

pub fn verify_citations(
    answer: &str,
    sources: &mut [SourceReference],
    visuals: &mut [VisualReference],
) -> Vec<String> {
    let tokens = extract_bracketed_tokens(answer);

    for s in sources.iter_mut() {
        s.is_cited = is_item_cited(&tokens, &s.id, SOURCE_PREFIXES);
    }

    for v in visuals.iter_mut() {
        v.is_cited = is_item_cited(&tokens, &v.id, VISUAL_PREFIXES);
    }

    detect_hallucinated_citations(&tokens, sources, visuals)
}
