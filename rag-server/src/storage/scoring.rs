use crate::domain::Modality;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

pub const DEFAULT_RRF_K: usize = 60;

pub const STOP_WORDS: &[&str] = &[
    "a", "an", "the", "in", "on", "at", "to", "for", "of", "with", "by", "from", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "and", "or",
    "but", "if", "then", "else", "when", "here", "there", "why", "how", "all", "any", "both",
    "each", "tell", "me", "about", "what", "which", "who", "whom", "this", "that", "it", "its",
    "can", "you",
];

// --- Tokenization and stemming ---

/// Tokens shorter than this (in UTF-8 bytes) are dropped.
const MIN_TOKEN_LEN: usize = 2;
/// The stem left after suffix removal must be at least this long (in UTF-8 bytes).
const MIN_STEM_LEN: usize = 3;
/// Suffixes tried in order; the first matching suffix is stripped.
const STEM_SUFFIXES: &[&str] = &["ing", "ed", "es", "er", "s"];

// --- Default values ---

/// Raw text scores below this threshold are treated as noise.
const DEFAULT_TEXT_NOISE_FLOOR: f32 = 0.705;
/// Raw text score at which the main curve reaches its maximum. Above it, a bonus is added.
const DEFAULT_TEXT_MATCH_CEILING: f32 = 0.810;
/// Raw image scores below this threshold are treated as noise.
const DEFAULT_CROSS_MODAL_MIN: f32 = 0.180;
/// Raw image score at which the main curve reaches its maximum. Above it, a bonus is added.
const DEFAULT_CROSS_MODAL_MAX: f32 = 0.280;

/// BM25 term frequency saturation parameter.
const DEFAULT_BM25_K1: f32 = 1.2;
/// BM25 document length normalization parameter.
const DEFAULT_BM25_B: f32 = 0.75;
/// Weight of title terms relative to body text terms.
const DEFAULT_TITLE_WEIGHT: f32 = 2.5;

// --- Text score calibration ---

/// Maximum calibrated score for raw values at or below the noise floor.
const TEXT_BELOW_FLOOR_SCALE: f32 = 0.02;
/// Minimum calibrated score for values above the noise floor.
const TEXT_BASE: f32 = 0.12;
/// Exponent applied to the normalized score (below 1 means fast growth at the start).
const TEXT_CURVE_EXPONENT: f32 = 0.80;
/// Maximum contribution of the main curve on top of the base score.
const TEXT_CURVE_SCALE: f32 = 0.70;
/// Width of the zone above the match ceiling over which the bonus grows.
const TEXT_OVERSHOOT_RANGE: f32 = 0.080;
/// Maximum bonus for exceeding the match ceiling.
const TEXT_OVERSHOOT_BONUS: f32 = 0.16;

// --- Image score calibration (cross-modal) ---

/// Maximum calibrated score for raw values at or below the noise floor.
const IMAGE_BELOW_FLOOR_SCALE: f32 = 0.01;
/// Minimum calibrated score for values above the noise floor.
const IMAGE_BASE: f32 = 0.12;
/// Exponent applied to the normalized score (below 1 means fast growth at the start).
const IMAGE_CURVE_EXPONENT: f32 = 0.85;
/// Maximum contribution of the main curve on top of the base score.
const IMAGE_CURVE_SCALE: f32 = 0.72;
/// Width of the zone above the match ceiling over which the bonus grows.
const IMAGE_OVERSHOOT_RANGE: f32 = 0.080;
/// Maximum bonus for exceeding the match ceiling.
const IMAGE_OVERSHOOT_BONUS: f32 = 0.14;

// --- IDF ---

/// Smoothing term in the idf formula (analogous to the 0.5 constant in BM25).
const IDF_SMOOTHING: f32 = 0.5;
/// Lower bound for idf, so that very common terms never drop below it.
const IDF_MIN_WEIGHT: f32 = 0.1;

// --- Modality fusion ---

/// If the other modality's score is below this threshold, the fused score is penalized.
const MODALITY_CONSISTENCY_THRESHOLD: f32 = 0.10;
/// Exponent of the penalty for a weak second modality.
const CONSISTENCY_EXPONENT: f32 = 2.0;

/// Weights for text queries.
const TEXT_QUERY_TEXT_WEIGHT: f32 = 0.40;
const TEXT_QUERY_IMAGE_WEIGHT: f32 = 0.60;

/// Weights for image queries.
const IMAGE_QUERY_IMAGE_WEIGHT: f32 = 0.60;
const IMAGE_QUERY_TEXT_WEIGHT: f32 = 0.40;

/// Weights for queries without a modality, or mixed cases.
const MIXED_TEXT_WEIGHT: f32 = 0.50;
const MIXED_IMAGE_WEIGHT: f32 = 0.50;
/// Scale of the mixed-query penalty: if min(text, image) reaches this value, there is no penalty.
const MIXED_CONSISTENCY_SCALE: f32 = 0.10;
/// Lower bound of the penalty for mixed queries.
const MIXED_CONSISTENCY_FLOOR: f32 = 0.05;

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
            text_noise_floor: DEFAULT_TEXT_NOISE_FLOOR,
            text_match_ceiling: DEFAULT_TEXT_MATCH_CEILING,
            cross_modal_min: DEFAULT_CROSS_MODAL_MIN,
            cross_modal_max: DEFAULT_CROSS_MODAL_MAX,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bm25Params {
    pub k1: f32,
    pub b: f32,
    pub title_weight: f32,
}

impl Default for Bm25Params {
    fn default() -> Self {
        Self {
            k1: DEFAULT_BM25_K1,
            b: DEFAULT_BM25_B,
            title_weight: DEFAULT_TITLE_WEIGHT,
        }
    }
}

#[inline]
pub fn calibrate_text_score(raw_score: f32, cfg: &CalibrationConfig) -> f32 {
    if !raw_score.is_finite() || raw_score <= cfg.text_noise_floor {
        return TEXT_BELOW_FLOOR_SCALE
            * (raw_score.max(0.0) / cfg.text_noise_floor.max(f32::EPSILON));
    }
    let denom = (cfg.text_match_ceiling - cfg.text_noise_floor).max(f32::EPSILON);
    let norm = ((raw_score - cfg.text_noise_floor) / denom).clamp(0.0, 1.0);
    TEXT_BASE
        + norm.powf(TEXT_CURVE_EXPONENT) * TEXT_CURVE_SCALE
        + if raw_score > cfg.text_match_ceiling {
            ((raw_score - cfg.text_match_ceiling) / TEXT_OVERSHOOT_RANGE).clamp(0.0, 1.0)
                * TEXT_OVERSHOOT_BONUS
        } else {
            0.0
        }
}

#[inline]
pub fn calibrate_image_score(raw_score: f32, cfg: &CalibrationConfig) -> f32 {
    if !raw_score.is_finite() || raw_score <= cfg.cross_modal_min {
        return IMAGE_BELOW_FLOOR_SCALE
            * (raw_score.max(0.0) / cfg.cross_modal_min.max(f32::EPSILON));
    }
    let denom = (cfg.cross_modal_max - cfg.cross_modal_min).max(f32::EPSILON);
    let norm = ((raw_score - cfg.cross_modal_min) / denom).clamp(0.0, 1.0);
    IMAGE_BASE
        + norm.powf(IMAGE_CURVE_EXPONENT) * IMAGE_CURVE_SCALE
        + if raw_score > cfg.cross_modal_max {
            ((raw_score - cfg.cross_modal_max) / IMAGE_OVERSHOOT_RANGE).clamp(0.0, 1.0)
                * IMAGE_OVERSHOOT_BONUS
        } else {
            0.0
        }
}

pub fn stem(word: &str) -> &str {
    for suffix in STEM_SUFFIXES {
        if let Some(s) = word.strip_suffix(suffix)
            && s.len() >= MIN_STEM_LEN
        {
            return s;
        }
    }
    word
}

pub fn tokenize_and_stem(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .map(|t| t.trim().to_lowercase())
        .filter(|t| t.len() >= MIN_TOKEN_LEN && !STOP_WORDS.contains(&t.as_str()))
        .map(|t| stem(&t).to_string())
        .collect()
}

#[inline]
pub fn idf(doc_freq: usize, total_docs: usize) -> f32 {
    let n = doc_freq as f32;
    let total = (total_docs as f32).max(1.0);
    ((total - n + IDF_SMOOTHING) / (n + IDF_SMOOTHING))
        .ln_1p()
        .max(IDF_MIN_WEIGHT)
}

pub fn calculate_bm25_score(
    query_text: &str,
    payload: &Value,
    avg_doc_len: f32,
    total_docs: usize,
    doc_freqs: Option<&HashMap<String, usize>>,
    params: &Bm25Params,
) -> f32 {
    let title = payload.get("title").and_then(|v| v.as_str()).unwrap_or("");
    let text = payload.get("text").and_then(|v| v.as_str()).unwrap_or("");

    let q_stems = tokenize_and_stem(query_text);
    if q_stems.is_empty() {
        return 0.0;
    }

    let title_stems = tokenize_and_stem(title);
    let text_stems = tokenize_and_stem(text);

    let doc_len =
        (title_stems.len() as f32 * params.title_weight + text_stems.len() as f32).max(1.0);
    let avgdl = avg_doc_len.max(1.0);
    let len_norm = 1.0 - params.b + params.b * (doc_len / avgdl);

    let mut title_tf_map: HashMap<&str, f32> = HashMap::with_capacity(title_stems.len());
    for s in &title_stems {
        *title_tf_map.entry(s.as_str()).or_default() += 1.0;
    }

    let mut text_tf_map: HashMap<&str, f32> = HashMap::with_capacity(text_stems.len());
    for s in &text_stems {
        *text_tf_map.entry(s.as_str()).or_default() += 1.0;
    }

    let mut total_score = 0.0f32;

    for q in &q_stems {
        let q_str = q.as_str();
        let t_tf = title_tf_map.get(q_str).copied().unwrap_or(0.0);
        let b_tf = text_tf_map.get(q_str).copied().unwrap_or(0.0);
        let tf = t_tf * params.title_weight + b_tf;

        if tf <= 0.0 {
            continue;
        }

        let term_idf = if let Some(dfs) = doc_freqs {
            let df = dfs.get(q_str).copied().unwrap_or(1);
            idf(df, total_docs)
        } else {
            1.0
        };

        let term_bm25 = term_idf * (tf * (params.k1 + 1.0)) / (tf + params.k1 * len_norm);
        total_score += term_bm25;
    }

    total_score
}

pub fn reciprocal_rank_fusion<T: Clone + Eq + std::hash::Hash>(
    ranked_runs: &[(&[T], f32)],
    k: usize,
) -> Vec<(T, f32)> {
    let mut scores: HashMap<T, f32> = HashMap::new();

    for (run, weight) in ranked_runs {
        for (rank_idx, item) in run.iter().enumerate() {
            let rank = (rank_idx + 1) as f32;
            let contrib = *weight / (k as f32 + rank);
            *scores.entry(item.clone()).or_insert(0.0) += contrib;
        }
    }

    let mut fused: Vec<(T, f32)> = scores.into_iter().collect();
    fused.sort_by(|a, b| b.1.total_cmp(&a.1));
    fused
}

#[inline]
pub fn compute_rrf_score_from_ranks(
    dense_rank: Option<usize>,
    sparse_rank: Option<usize>,
    k: usize,
    dense_weight: f32,
    sparse_weight: f32,
) -> f32 {
    let mut score = 0.0f32;
    if let Some(r) = dense_rank {
        score += dense_weight / ((k + r) as f32);
    }
    if let Some(r) = sparse_rank {
        score += sparse_weight / ((k + r) as f32);
    }
    let max_possible = (dense_weight + sparse_weight) / ((k + 1) as f32);
    if max_possible > 0.0 {
        (score / max_possible).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

pub fn fuse_scores(
    query_modality: Option<&Modality>,
    img_score: f32,
    text_score: f32,
) -> (f32, Modality) {
    match query_modality {
        Some(Modality::Text) => {
            let consistency = if img_score < MODALITY_CONSISTENCY_THRESHOLD {
                (img_score / MODALITY_CONSISTENCY_THRESHOLD).powf(CONSISTENCY_EXPONENT)
            } else {
                1.0
            };
            let base = TEXT_QUERY_TEXT_WEIGHT * text_score + TEXT_QUERY_IMAGE_WEIGHT * img_score;
            let fused = (base * consistency).min(1.0);
            let matched_by = if img_score > text_score {
                Modality::Image
            } else {
                Modality::Text
            };
            (fused, matched_by)
        }
        Some(Modality::Image) => {
            let consistency = if text_score < MODALITY_CONSISTENCY_THRESHOLD {
                (text_score / MODALITY_CONSISTENCY_THRESHOLD).powf(CONSISTENCY_EXPONENT)
            } else {
                1.0
            };
            let fused = ((IMAGE_QUERY_IMAGE_WEIGHT * img_score
                + IMAGE_QUERY_TEXT_WEIGHT * text_score)
                * consistency)
                .min(1.0);
            (fused, Modality::Image)
        }
        _ => {
            let consistency = (text_score.min(img_score) / MIXED_CONSISTENCY_SCALE)
                .clamp(MIXED_CONSISTENCY_FLOOR, 1.0);
            let fused = ((MIXED_TEXT_WEIGHT * text_score + MIXED_IMAGE_WEIGHT * img_score)
                * consistency)
                .min(1.0);
            let matched_by = if img_score > text_score {
                Modality::Image
            } else {
                Modality::Text
            };
            (fused, matched_by)
        }
    }
}
