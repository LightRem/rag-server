use crate::domain::Modality;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationConfig {
    pub text_noise_floor: f32,
    pub text_match_ceiling: f32,
    pub cross_modal_min: f32,
    pub cross_modal_max: f32,
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            text_noise_floor: 0.65,
            text_match_ceiling: 0.85,
            cross_modal_min: 0.16,
            cross_modal_max: 0.30,
        }
    }
}

#[inline]
pub fn calibrate_text_score(raw_score: f32, cfg: &CalibrationConfig) -> f32 {
    let denom = (cfg.text_match_ceiling - cfg.text_noise_floor).max(f32::EPSILON);
    ((raw_score - cfg.text_noise_floor) / denom).clamp(0.0, 1.0)
}

#[inline]
pub fn calibrate_image_score(
    raw_score: f32,
    _has_image_intent: bool,
    cfg: &CalibrationConfig,
) -> f32 {
    let denom = (cfg.cross_modal_max - cfg.cross_modal_min).max(f32::EPSILON);
    ((raw_score - cfg.cross_modal_min) / denom).clamp(0.0, 1.0)
}

fn stem(word: &str) -> &str {
    if let Some(s) = word.strip_suffix("ing")
        && s.len() >= 3
    {
        return s;
    }
    if let Some(s) = word.strip_suffix("ed")
        && s.len() >= 3
    {
        return s;
    }
    if let Some(s) = word.strip_suffix("es")
        && s.len() >= 3
    {
        return s;
    }
    if let Some(s) = word.strip_suffix('s')
        && s.len() >= 3
    {
        return s;
    }
    if let Some(s) = word.strip_suffix("er")
        && s.len() >= 3
    {
        return s;
    }
    word
}

/// Вычисляет взвешенное лексическое пересечение терминов (заголовок: 2.5, текст: 1.0)
pub fn calculate_keyword_score(query_text: &str, payload: &Value) -> f32 {
    let title = payload.get("title").and_then(|v| v.as_str()).unwrap_or("");
    let text = payload.get("text").and_then(|v| v.as_str()).unwrap_or("");

    if title.is_empty() && text.is_empty() {
        return 0.0;
    }

    const STOP_WORDS: &[&str] = &[
        "a", "an", "the", "in", "on", "at", "to", "for", "of", "with", "by", "from", "is", "are",
        "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "and",
        "or", "but", "if", "then", "else", "when", "here", "there", "why", "how", "all", "any",
        "both", "each", "tell", "me", "about", "what", "which", "who", "whom", "this", "that",
        "it", "its",
    ];

    let title_words: Vec<&str> = title
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .collect();

    let text_words: Vec<&str> = text
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .collect();

    let mut query_terms = 0;
    let mut match_weight = 0.0f32;

    for raw_token in query_text.split(|c: char| !c.is_alphanumeric()) {
        let token = raw_token.trim().to_lowercase();
        if token.len() < 3 || STOP_WORDS.contains(&token.as_str()) {
            continue;
        }
        query_terms += 1;
        let token_stem = stem(&token);

        let in_title = title_words.iter().any(|&tw| {
            let tw_l = tw.to_lowercase();
            tw_l == token
                || stem(&tw_l) == token_stem
                || (token_stem.len() >= 4 && tw_l.contains(token_stem))
        });

        let in_text = text_words.iter().any(|&dw| {
            let dw_l = dw.to_lowercase();
            dw_l == token
                || stem(&dw_l) == token_stem
                || (token_stem.len() >= 4 && dw_l.contains(token_stem))
        });

        if in_title {
            match_weight += 2.5;
        } else if in_text {
            match_weight += 1.0;
        }
    }

    if query_terms > 0 && match_weight > 0.0 {
        (match_weight / (query_terms as f32 * 2.5)).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Слияние Dense и Sparse оценок по алгоритму RRF (Reciprocal Rank Fusion)
pub fn compute_rrf_hybrid_score(dense_score: f32, sparse_score: f32) -> f32 {
    const K: f32 = 20.0;
    let rank_dense = (1.0 - dense_score).max(0.0) * 10.0;
    let rank_sparse = (1.0 - sparse_score).max(0.0) * 10.0;

    let rrf_dense = 1.0 / (K + rank_dense);
    let rrf_sparse = if sparse_score > 0.0 {
        1.5 / (K + rank_sparse)
    } else {
        0.0
    };

    let max_possible = (1.0 / K) + (1.5 / K);
    ((rrf_dense + rrf_sparse) / max_possible).clamp(0.0, 1.0)
}

pub fn fuse_scores(
    query_modality: Option<&Modality>,
    img_score: f32,
    text_score: f32,
) -> (f32, Modality) {
    match query_modality {
        Some(Modality::Text) => {
            let fused = 0.85 * text_score + 0.15 * img_score;
            let matched_by = if img_score > text_score + 0.20 {
                Modality::Image
            } else {
                Modality::Text
            };
            (fused.clamp(0.0, 1.0), matched_by)
        }
        Some(Modality::Image) => {
            let fused = 0.85 * img_score + 0.15 * text_score;
            (fused.clamp(0.0, 1.0), Modality::Image)
        }
        _ => {
            let max_s = text_score.max(img_score);
            let min_s = text_score.min(img_score);
            let fused = 0.75 * max_s + 0.25 * min_s;
            let matched_by = if img_score > text_score {
                Modality::Image
            } else {
                Modality::Text
            };
            (fused.clamp(0.0, 1.0), matched_by)
        }
    }
}
