use crate::domain::Modality;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationConfig {
    pub cross_modal_min: f32,
    pub cross_modal_max: f32,
    pub text_noise_floor: f32,
    pub text_match_ceiling: f32,
    pub text_modal_min: f32,
    pub text_modal_max: f32,
    pub image_trigger_boost: f32,
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            cross_modal_min: 0.14,
            cross_modal_max: 0.28,
            text_noise_floor: 0.67,
            text_match_ceiling: 0.78,
            text_modal_min: 0.55,
            text_modal_max: 0.95,
            image_trigger_boost: 0.15,
        }
    }
}

#[inline]
pub fn calibrate_text_score(raw_score: f32, cfg: &CalibrationConfig) -> f32 {
    if raw_score <= cfg.text_noise_floor {
        let floor = cfg.text_noise_floor.max(f32::EPSILON);
        (raw_score / floor).powi(4) * 0.15
    } else {
        let denom = (cfg.text_match_ceiling - cfg.text_noise_floor).max(f32::EPSILON);
        let norm = ((raw_score - cfg.text_noise_floor) / denom).clamp(0.0, 1.0);
        0.20 + norm * 0.75
    }
}

#[inline]
pub fn calibrate_image_score(
    raw_score: f32,
    has_image_intent: bool,
    cfg: &CalibrationConfig,
) -> f32 {
    let calibrated = if raw_score <= 0.0 {
        0.0
    } else if raw_score < cfg.cross_modal_min {
        let cross_min = cfg.cross_modal_min.max(f32::EPSILON);
        (raw_score / cross_min) * 0.25
    } else if raw_score <= cfg.cross_modal_max {
        let denom = (cfg.cross_modal_max - cfg.cross_modal_min).max(f32::EPSILON);
        let normalized = (raw_score - cfg.cross_modal_min) / denom;
        cfg.text_modal_min + normalized * (cfg.text_modal_max - cfg.text_modal_min)
    } else {
        let denom = (1.0 - cfg.cross_modal_max).max(f32::EPSILON);
        let extra = ((raw_score - cfg.cross_modal_max) / denom).clamp(0.0, 1.0);
        cfg.text_modal_max + extra * (0.98 - cfg.text_modal_max)
    };

    if has_image_intent && raw_score >= cfg.cross_modal_min {
        (calibrated + cfg.image_trigger_boost).min(1.0)
    } else {
        calibrated
    }
}

pub fn calculate_keyword_boost(query_text: &str, payload: &Value) -> f32 {
    let text = match payload.get("text").and_then(|v| v.as_str()) {
        Some(t) => t,
        None => return 0.0,
    };
    let title = payload.get("title").and_then(|v| v.as_str()).unwrap_or("");
    let lower_doc = format!("{} {}", text, title).to_lowercase();

    const STOP_WORDS: &[&str] = &[
        "a", "an", "the", "in", "on", "at", "to", "for", "of", "with", "by", "from", "is", "are",
        "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "and",
        "or", "but", "if", "then", "else", "when", "here", "there", "why", "how", "all", "any",
        "both", "each", "tell", "me", "about", "what", "which", "who", "whom", "this", "that",
        "it", "its",
    ];

    let mut matches = 0;
    let mut query_terms = 0;

    for raw_token in query_text.split(|c: char| !c.is_alphanumeric()) {
        let token = raw_token.trim().to_lowercase();
        if token.len() < 3 || STOP_WORDS.contains(&token.as_str()) {
            continue;
        }
        query_terms += 1;
        let stem = token.strip_suffix('s').unwrap_or(&token);
        if lower_doc.contains(&token) || lower_doc.contains(stem) {
            matches += 1;
        }
    }

    if query_terms > 0 && matches > 0 {
        let ratio = matches as f32 / query_terms as f32;
        0.18 * ratio
    } else {
        0.0
    }
}

#[inline]
pub fn has_image_keywords(text: &str) -> bool {
    let lower = text.to_lowercase();
    const EXACT_KEYWORDS: &[&str] = &[
        "photo",
        "photos",
        "image",
        "images",
        "picture",
        "pictures",
        "pic",
        "pics",
        "snapshot",
        "snapshots",
        "drawing",
        "drawings",
        "illustration",
        "illustrations",
        "фото",
        "зображення",
        "картинка",
        "картин",
    ];

    lower
        .split(|c: char| !c.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .any(|token| {
            EXACT_KEYWORDS.contains(&token)
                || token.starts_with("фотк")
                || token.starts_with("фотограф")
                || token.starts_with("картинк")
                || token.starts_with("рисун")
                || token.starts_with("изображен")
                || token.starts_with("світлин")
                || token.starts_with("малюн")
        })
}

pub fn fuse_scores(
    query_modality: Option<&Modality>,
    img_score: f32,
    text_score: f32,
) -> (f32, Modality) {
    match query_modality {
        Some(Modality::Text) => {
            let fused = (0.55 * img_score + 0.45 * text_score).min(1.0);
            let matched_by = if img_score > text_score {
                Modality::Image
            } else {
                Modality::Text
            };
            (fused, matched_by)
        }
        Some(Modality::Image) => {
            let fused = (0.70 * img_score + 0.30 * text_score).min(1.0);
            (fused, Modality::Image)
        }
        _ => {
            if img_score > text_score {
                (img_score, Modality::Image)
            } else {
                (text_score, Modality::Text)
            }
        }
    }
}
