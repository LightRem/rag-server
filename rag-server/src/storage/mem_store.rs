use super::{
    dot_product,
    hnsw::{HnswConfig, HnswIndex, VectorView},
    models::{SearchFilter, SearchResult, VectorPoint},
    partition::StorageState,
    persistence::{load_snapshot, save_snapshot},
    scoring::{
        CalibrationConfig, DEFAULT_RRF_K, STOP_WORDS, calibrate_image_score, calibrate_text_score,
        stem,
    },
    traits::{DocumentSummary, VectorStore},
};
use crate::domain::{DocumentId, Modality};
use crate::error::StorageError;
use async_trait::async_trait;
use rayon::prelude::*;
use rayon::slice::ParallelSlice;
use serde_json::Value;
use std::{
    cmp::Ordering,
    collections::{BinaryHeap, HashMap, HashSet},
    path::Path,
    sync::{
        Arc, Mutex as StdMutex, RwLock, RwLockReadGuard, RwLockWriteGuard,
        atomic::{AtomicU64, AtomicUsize, Ordering as AtomicOrdering},
    },
};

pub const DEFAULT_MAX_CAPACITY: usize = 1_000_000;
const L2_NORM_SQ_TOLERANCE: f32 = 1e-2;
const HNSW_MIN_POINTS_THRESHOLD: usize = 128;

// Rank-fusion list weights. A document at position `p` in a list receives
// `weight / (DEFAULT_RRF_K + p)` from that list. Lists: text vectors, image vectors, lexical.
const RRF_TEXT_WEIGHT: f32 = 1.0;
const RRF_IMAGE_WEIGHT: f32 = 1.0;
const RRF_LEXICAL_WEIGHT: f32 = 1.0;

type PartitionMatch = (Arc<str>, f32, Modality, Arc<Value>);

#[derive(Clone, Copy)]
struct ScoredCandidate<'a> {
    score: f32,
    slot_idx: usize,
    id: &'a str,
}

impl<'a> PartialEq for ScoredCandidate<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.score.total_cmp(&other.score) == Ordering::Equal && self.id == other.id
    }
}
impl<'a> Eq for ScoredCandidate<'a> {}
impl<'a> Ord for ScoredCandidate<'a> {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .score
            .total_cmp(&self.score)
            .then_with(|| self.id.cmp(other.id))
    }
}
impl<'a> PartialOrd for ScoredCandidate<'a> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[inline]
fn push_candidate<'a>(
    heap: &mut BinaryHeap<ScoredCandidate<'a>>,
    candidate: ScoredCandidate<'a>,
    limit: usize,
) {
    if heap.len() < limit {
        heap.push(candidate);
    } else if heap
        .peek()
        .is_some_and(|&worst_in_top| candidate < worst_in_top)
    {
        heap.pop();
        heap.push(candidate);
    }
}

/// Whole-word match on stems only, so "cat" does not match "category" or "education".
fn is_term_match(q_stem: &str, doc_word: &str) -> bool {
    stem(doc_word) == q_stem
}

fn tokenize_query(query_lower: &str) -> Vec<&str> {
    query_lower
        .split(|c: char| !c.is_alphanumeric())
        .map(str::trim)
        .filter(|t| !t.is_empty() && t.len() >= 2 && !STOP_WORDS.contains(t))
        .collect()
}

struct LexicalMatch {
    ratio: f32,
    in_title: bool,
    has_exact_bigram: bool,
    has_close_pair: bool,
}

fn compute_lexical_match(query: &str, payload: &Value) -> LexicalMatch {
    let query_lower = query.to_lowercase();
    let query_tokens = tokenize_query(&query_lower);

    if query_tokens.is_empty() {
        return LexicalMatch {
            ratio: 0.0,
            in_title: false,
            has_exact_bigram: false,
            has_close_pair: false,
        };
    }

    let title = payload.get("title").and_then(|v| v.as_str()).unwrap_or("");
    let text = payload.get("text").and_then(|v| v.as_str()).unwrap_or("");
    let caption = payload
        .get("caption")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let full = format!("{title} {text} {caption}").to_lowercase();

    let words: Vec<&str> = full
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .collect();

    let title_lower = title.to_lowercase();
    let title_words: Vec<&str> = title_lower
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .collect();
    let mut matched_count = 0;
    let mut in_title = false;

    for q in &query_tokens {
        let q_stem = stem(q);
        let hit_title = title_words.iter().any(|w| is_term_match(q_stem, w));
        let hit_text = words.iter().any(|w| is_term_match(q_stem, w));

        if hit_title || hit_text {
            matched_count += 1;
            if hit_title {
                in_title = true;
            }
        }
    }

    let mut has_exact_bigram = false;
    let mut has_close_pair = false;

    if query_tokens.len() >= 2 {
        for i in 0..query_tokens.len() {
            for j in (i + 1)..query_tokens.len() {
                let q1 = query_tokens[i];
                let q2 = query_tokens[j];
                let q1_stem = stem(q1);
                let q2_stem = stem(q2);

                let mut pos1 = Vec::new();
                let mut pos2 = Vec::new();
                for (idx, w) in words.iter().copied().enumerate() {
                    if is_term_match(q1_stem, w) {
                        pos1.push(idx);
                    }
                    if is_term_match(q2_stem, w) {
                        pos2.push(idx);
                    }
                }

                for &p1 in &pos1 {
                    for &p2 in &pos2 {
                        let diff = p1.abs_diff(p2);
                        if diff == 1 {
                            has_exact_bigram = true;
                        } else if diff <= 3 {
                            has_close_pair = true;
                        }
                    }
                }
            }
        }
    }

    let ratio = matched_count as f32 / query_tokens.len() as f32;
    LexicalMatch {
        ratio,
        in_title,
        has_exact_bigram,
        has_close_pair,
    }
}

fn collect_all_document_ids(t_state: &StorageState, i_state: &StorageState) -> HashSet<DocumentId> {
    let mut all_ids = HashSet::with_capacity(t_state.len() + i_state.len());
    all_ids.extend(t_state.index_map.keys().cloned());
    all_ids.extend(i_state.index_map.keys().cloned());
    all_ids
}

fn extract_point_from_slots(
    id: &DocumentId,
    t_slot: Option<usize>,
    i_slot: Option<usize>,
    t_lock: &StorageState,
    i_lock: &StorageState,
    dim: usize,
) -> Option<VectorPoint> {
    match (t_slot, i_slot) {
        (Some(ts), Some(is)) => {
            let text_vec = t_lock.vectors[ts * dim..(ts + 1) * dim].to_vec();
            let img_vec = i_lock.vectors[is * dim..(is + 1) * dim].to_vec();
            Some(
                VectorPoint::new(
                    id.clone(),
                    text_vec,
                    (*t_lock.payloads[ts]).clone(),
                    Modality::Multimodal,
                )
                .with_image_vector(img_vec),
            )
        }
        (Some(ts), None) => {
            let text_vec = t_lock.vectors[ts * dim..(ts + 1) * dim].to_vec();
            Some(VectorPoint::new(
                id.clone(),
                text_vec,
                (*t_lock.payloads[ts]).clone(),
                Modality::Text,
            ))
        }
        (None, Some(is)) => {
            let img_vec = i_lock.vectors[is * dim..(is + 1) * dim].to_vec();
            Some(VectorPoint::new(
                id.clone(),
                img_vec,
                (*i_lock.payloads[is]).clone(),
                Modality::Image,
            ))
        }
        (None, None) => None,
    }
}

#[derive(Clone)]
pub struct InMemoryVectorStore {
    dimension: usize,
    max_capacity: usize,
    calibration: CalibrationConfig,
    doc_count: Arc<AtomicUsize>,
    text_state: Arc<RwLock<StorageState>>,
    image_state: Arc<RwLock<StorageState>>,
    hnsw_text: Arc<RwLock<HnswIndex>>,
    hnsw_image: Arc<RwLock<HnswIndex>>,
    disk_lock: Arc<StdMutex<()>>,
    version: Arc<AtomicU64>,
    persisted_version: Arc<AtomicU64>,
}

impl InMemoryVectorStore {
    pub fn new(dimension: usize) -> Self {
        Self::with_config(
            dimension,
            DEFAULT_MAX_CAPACITY,
            CalibrationConfig::default(),
        )
    }

    pub fn with_capacity(dimension: usize, max_capacity: usize) -> Self {
        Self::with_config(dimension, max_capacity, CalibrationConfig::default())
    }

    pub fn with_config(
        dimension: usize,
        max_capacity: usize,
        calibration: CalibrationConfig,
    ) -> Self {
        Self {
            dimension,
            max_capacity,
            calibration,
            doc_count: Arc::new(AtomicUsize::new(0)),
            text_state: Arc::new(RwLock::new(StorageState::default())),
            image_state: Arc::new(RwLock::new(StorageState::default())),
            hnsw_text: Arc::new(RwLock::new(HnswIndex::new(HnswConfig::default()))),
            hnsw_image: Arc::new(RwLock::new(HnswIndex::new(HnswConfig::default()))),
            disk_lock: Arc::new(StdMutex::new(())),
            version: Arc::new(AtomicU64::new(0)),
            persisted_version: Arc::new(AtomicU64::new(0)),
        }
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn is_dirty(&self) -> bool {
        self.version.load(AtomicOrdering::Acquire)
            > self.persisted_version.load(AtomicOrdering::Acquire)
    }

    pub fn max_capacity(&self) -> usize {
        self.max_capacity
    }

    pub fn calibration(&self) -> &CalibrationConfig {
        &self.calibration
    }

    fn read_states(
        &self,
    ) -> Result<
        (
            RwLockReadGuard<'_, StorageState>,
            RwLockReadGuard<'_, StorageState>,
        ),
        StorageError,
    > {
        let t = self.text_state.read().map_err(|_| {
            StorageError::Internal("Vector store text_state read lock poisoned".into())
        })?;
        let i = self.image_state.read().map_err(|_| {
            StorageError::Internal("Vector store image_state read lock poisoned".into())
        })?;
        Ok((t, i))
    }

    fn write_states(
        &self,
    ) -> Result<
        (
            RwLockWriteGuard<'_, StorageState>,
            RwLockWriteGuard<'_, StorageState>,
        ),
        StorageError,
    > {
        let t = self.text_state.write().map_err(|_| {
            StorageError::Internal("Vector store text_state write lock poisoned".into())
        })?;
        let i = self.image_state.write().map_err(|_| {
            StorageError::Internal("Vector store image_state write lock poisoned".into())
        })?;
        Ok((t, i))
    }

    pub(crate) fn validate_vector(&self, vector: &[f32]) -> Result<(), StorageError> {
        if self.dimension == 0 {
            return Err(StorageError::InvalidVector(
                "Vector store configured with invalid dimension 0".into(),
            ));
        }
        if vector.len() != self.dimension {
            return Err(StorageError::DimensionMismatch {
                expected: self.dimension,
                actual: vector.len(),
            });
        }
        let mut norm_sq = 0.0f32;
        for &x in vector {
            if !x.is_finite() {
                return Err(StorageError::InvalidVector(
                    "Vector contains non-finite values (NaN or Infinity)".into(),
                ));
            }
            norm_sq += x * x;
        }
        if !norm_sq.is_finite() || norm_sq <= f32::EPSILON {
            return Err(StorageError::InvalidVector(
                "Zero-norm or overflowing vector is not permitted".into(),
            ));
        }
        if (norm_sq - 1.0).abs() > L2_NORM_SQ_TOLERANCE {
            return Err(StorageError::InvalidVector(format!(
                "Vector must be L2-normalized (expected norm ~1.0, got {:.6})",
                norm_sq.sqrt()
            )));
        }
        Ok(())
    }

    fn validate_point_vectors(&self, point: &VectorPoint) -> Result<(), StorageError> {
        if self.dimension == 0 {
            return Err(StorageError::InvalidVector(
                "Vector store configured with invalid dimension 0".into(),
            ));
        }
        if point.id.trim().is_empty() {
            return Err(StorageError::InvalidVector(
                "Document ID cannot be empty".into(),
            ));
        }
        self.validate_vector(&point.vector)?;
        if let Some(ref img_v) = point.image_vector {
            self.validate_vector(img_v)?;
        }
        Ok(())
    }

    fn validate_filter(filter: &Option<SearchFilter>) -> Result<(), StorageError> {
        if let Some(min_score) = filter
            .as_ref()
            .and_then(|f| f.min_score)
            .filter(|&score| !(-1.0 - 1e-4..=1.0 + 1e-4).contains(&score))
        {
            return Err(StorageError::InvalidFilter(format!(
                "Filter min_score must be in range [-1.0, 1.0], got {min_score}"
            )));
        }
        Ok(())
    }

    fn insert_into_state_and_hnsw(
        state: &mut StorageState,
        hnsw: &RwLock<HnswIndex>,
        id: &DocumentId,
        vector: &[f32],
        payload: Arc<Value>,
        modality: Modality,
        dimension: usize,
    ) -> Result<(), StorageError> {
        let (slot_idx, was_new) = state.upsert(id, vector, payload, modality, dimension);
        let mut hnsw_lock = hnsw
            .write()
            .map_err(|_| StorageError::Internal("HNSW index write lock poisoned".into()))?;

        if !was_new {
            hnsw_lock.remove_node_edges(slot_idx);
        }
        let start = slot_idx * dimension;
        let pt_slice = &state.vectors[start..start + dimension];
        hnsw_lock.insert(slot_idx, pt_slice, &state.vectors, dimension);
        Ok(())
    }

    fn remove_from_state_and_hnsw(
        state: &mut StorageState,
        hnsw: &RwLock<HnswIndex>,
        id: &DocumentId,
        dimension: usize,
    ) -> Result<bool, StorageError> {
        if let Some((slot_idx, last_idx)) = state.remove(id, dimension) {
            let mut hnsw_lock = hnsw
                .write()
                .map_err(|_| StorageError::Internal("HNSW index write lock poisoned".into()))?;
            hnsw_lock.remove(slot_idx, last_idx);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn upsert_point_into_partitions(
        &self,
        t_state: &mut StorageState,
        i_state: &mut StorageState,
        point: &VectorPoint,
        payload_arc: Arc<Value>,
    ) -> Result<(), StorageError> {
        match point.modality {
            Modality::Text => {
                Self::insert_into_state_and_hnsw(
                    t_state,
                    &self.hnsw_text,
                    &point.id,
                    &point.vector,
                    payload_arc,
                    Modality::Text,
                    self.dimension,
                )?;
                Self::remove_from_state_and_hnsw(
                    i_state,
                    &self.hnsw_image,
                    &point.id,
                    self.dimension,
                )?;
            }
            Modality::Image => {
                Self::insert_into_state_and_hnsw(
                    i_state,
                    &self.hnsw_image,
                    &point.id,
                    &point.vector,
                    payload_arc,
                    Modality::Image,
                    self.dimension,
                )?;
                Self::remove_from_state_and_hnsw(
                    t_state,
                    &self.hnsw_text,
                    &point.id,
                    self.dimension,
                )?;
            }
            Modality::Multimodal => {
                let img_vec = point.image_vector.as_deref().unwrap_or(&point.vector);
                Self::insert_into_state_and_hnsw(
                    t_state,
                    &self.hnsw_text,
                    &point.id,
                    &point.vector,
                    Arc::clone(&payload_arc),
                    Modality::Multimodal,
                    self.dimension,
                )?;
                Self::insert_into_state_and_hnsw(
                    i_state,
                    &self.hnsw_image,
                    &point.id,
                    img_vec,
                    payload_arc,
                    Modality::Multimodal,
                    self.dimension,
                )?;
            }
        }
        Ok(())
    }

    pub(crate) fn insert_blocking(&self, point: VectorPoint) -> Result<(), StorageError> {
        self.validate_point_vectors(&point)?;
        let (mut t_state, mut i_state) = self.write_states()?;

        let is_existing =
            t_state.index_map.contains_key(&point.id) || i_state.index_map.contains_key(&point.id);
        let current_count = self.doc_count.load(AtomicOrdering::Acquire);
        if !is_existing && current_count >= self.max_capacity {
            return Err(StorageError::CapacityExceeded {
                current: current_count,
                max: self.max_capacity,
            });
        }

        let payload_arc = Arc::new(point.payload.clone());
        self.upsert_point_into_partitions(&mut t_state, &mut i_state, &point, payload_arc)?;

        if !is_existing {
            self.doc_count.fetch_add(1, AtomicOrdering::Release);
        }
        self.version.fetch_add(1, AtomicOrdering::Release);
        Ok(())
    }

    pub(crate) fn insert_batch_blocking(
        &self,
        points: Vec<VectorPoint>,
    ) -> Result<(), StorageError> {
        for point in &points {
            self.validate_point_vectors(point)?;
        }

        let (mut t_state, mut i_state) = self.write_states()?;
        let current_count = self.doc_count.load(AtomicOrdering::Acquire);
        let mut new_unique_count = 0;
        let mut seen_in_batch = HashSet::with_capacity(points.len());

        for pt in &points {
            let is_existing =
                t_state.index_map.contains_key(&pt.id) || i_state.index_map.contains_key(&pt.id);
            if !is_existing && seen_in_batch.insert(&pt.id) {
                new_unique_count += 1;
            }
        }

        if current_count + new_unique_count > self.max_capacity {
            return Err(StorageError::CapacityExceeded {
                current: current_count,
                max: self.max_capacity,
            });
        }

        for point in points {
            let is_existing = t_state.index_map.contains_key(&point.id)
                || i_state.index_map.contains_key(&point.id);
            let payload_arc = Arc::new(point.payload.clone());

            self.upsert_point_into_partitions(&mut t_state, &mut i_state, &point, payload_arc)?;

            if !is_existing {
                self.doc_count.fetch_add(1, AtomicOrdering::Release);
            }
        }

        self.version.fetch_add(1, AtomicOrdering::Release);
        Ok(())
    }

    fn search_partition_zero_copy(
        hnsw: &RwLock<HnswIndex>,
        state: &StorageState,
        query_vector: &[f32],
        limit: usize,
        dimension: usize,
        filter: &Option<SearchFilter>,
    ) -> Result<Vec<PartitionMatch>, StorageError> {
        let total_points = state.len();
        if total_points == 0 {
            return Ok(Vec::new());
        }

        let effective_limit = limit.min(total_points);
        let candidates: Vec<(f32, usize)> = if total_points >= HNSW_MIN_POINTS_THRESHOLD {
            let hnsw_lock = hnsw
                .read()
                .map_err(|_| StorageError::Internal("HNSW index read lock poisoned".into()))?;

            let view = VectorView {
                vectors: &state.vectors,
                ids: &state.ids,
                modalities: &state.modalities,
                dim: dimension,
            };

            let hnsw_res = hnsw_lock.search(query_vector, effective_limit, &view, filter.as_ref());
            if filter.is_some() && hnsw_res.len() < effective_limit {
                drop(hnsw_lock);
                Self::exact_flat_scan(state, query_vector, effective_limit, filter, dimension)
            } else {
                hnsw_res
            }
        } else {
            Self::exact_flat_scan(state, query_vector, effective_limit, filter, dimension)
        };

        let mut results = Vec::with_capacity(candidates.len());
        for (score, slot) in candidates {
            results.push((
                Arc::clone(&state.ids[slot]),
                score,
                state.modalities[slot].clone(),
                Arc::clone(&state.payloads[slot]),
            ));
        }
        Ok(results)
    }

    fn exact_flat_scan(
        state: &StorageState,
        query: &[f32],
        limit: usize,
        filter: &Option<SearchFilter>,
        dim: usize,
    ) -> Vec<(f32, usize)> {
        if limit == 0 || state.is_empty() {
            return Vec::new();
        }

        let min_score = filter.as_ref().and_then(|f| f.min_score);
        let target_mod = filter.as_ref().and_then(|f| f.modality.as_ref());

        let heap = if state.len() < 256 {
            let mut heap = BinaryHeap::with_capacity(limit);
            for (slot_idx, point_vec) in state.vectors.chunks_exact(dim).enumerate() {
                if let Some(modality) = target_mod {
                    let doc_mod = &state.modalities[slot_idx];
                    let matches = doc_mod == modality || *doc_mod == Modality::Multimodal;
                    if !matches {
                        continue;
                    }
                }
                let score = dot_product(query, point_vec);
                if !score.is_finite() || min_score.is_some_and(|min| score < min) {
                    continue;
                }
                push_candidate(
                    &mut heap,
                    ScoredCandidate {
                        score,
                        slot_idx,
                        id: &state.ids[slot_idx],
                    },
                    limit,
                );
            }
            heap
        } else {
            state
                .vectors
                .as_slice()
                .par_chunks_exact(dim)
                .enumerate()
                .fold(
                    || BinaryHeap::with_capacity(limit),
                    |mut local_heap, (slot_idx, point_vec)| {
                        if let Some(modality) = target_mod {
                            let doc_mod = &state.modalities[slot_idx];
                            let matches = doc_mod == modality || *doc_mod == Modality::Multimodal;
                            if !matches {
                                return local_heap;
                            }
                        }
                        let score = dot_product(query, point_vec);
                        if !score.is_finite() || min_score.is_some_and(|min| score < min) {
                            return local_heap;
                        }
                        push_candidate(
                            &mut local_heap,
                            ScoredCandidate {
                                score,
                                slot_idx,
                                id: &state.ids[slot_idx],
                            },
                            limit,
                        );
                        local_heap
                    },
                )
                .reduce(
                    || BinaryHeap::with_capacity(limit),
                    |mut heap_a, heap_b| {
                        for cand in heap_b {
                            push_candidate(&mut heap_a, cand, limit);
                        }
                        heap_a
                    },
                )
        };

        let mut res = Vec::with_capacity(heap.len());
        let mut sorted_heap = heap;
        while let Some(c) = sorted_heap.pop() {
            res.push((c.score, c.slot_idx));
        }
        res.reverse();
        res
    }

    fn search_blocking(
        &self,
        query_vector: &[f32],
        limit: usize,
        filter: Option<SearchFilter>,
    ) -> Result<Vec<SearchResult>, StorageError> {
        let (t_state, i_state) = self.read_states()?;

        let search_k = limit.saturating_mul(4).max(32);
        let explicit_modality_filter = filter.as_ref().and_then(|f| f.modality.as_ref());

        let mut partition_filter = filter.clone();
        if let Some(ref mut pf) = partition_filter {
            pf.min_score = None;
        }

        let (text_matches, image_matches) = match explicit_modality_filter {
            Some(Modality::Text) => {
                let t_res = Self::search_partition_zero_copy(
                    &self.hnsw_text,
                    &t_state,
                    query_vector,
                    search_k,
                    self.dimension,
                    &partition_filter,
                )?;
                (t_res, Vec::new())
            }
            Some(Modality::Image) => {
                let i_res = Self::search_partition_zero_copy(
                    &self.hnsw_image,
                    &i_state,
                    query_vector,
                    search_k,
                    self.dimension,
                    &partition_filter,
                )?;
                (Vec::new(), i_res)
            }
            _ => {
                let t_res = Self::search_partition_zero_copy(
                    &self.hnsw_text,
                    &t_state,
                    query_vector,
                    search_k,
                    self.dimension,
                    &partition_filter,
                )?;
                let i_res = Self::search_partition_zero_copy(
                    &self.hnsw_image,
                    &i_state,
                    query_vector,
                    search_k,
                    self.dimension,
                    &partition_filter,
                )?;
                (t_res, i_res)
            }
        };

        let query_text = filter.as_ref().and_then(|f| f.query_text.as_deref());
        let query_modality = filter.as_ref().and_then(|f| f.query_modality.as_ref());

        fn rank_ids(matches: &[PartitionMatch], min_similarity: f32) -> Vec<Arc<str>> {
            let mut order: Vec<&PartitionMatch> =
                matches.iter().filter(|m| m.1 > min_similarity).collect();
            order.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            order.into_iter().map(|m| Arc::clone(&m.0)).collect()
        }

        fn positions(ranked: &[Arc<str>]) -> HashMap<Arc<str>, usize> {
            ranked
                .iter()
                .enumerate()
                .map(|(i, id)| (Arc::clone(id), i + 1))
                .collect()
        }

        let calibrate_text_similarity = |raw: f32| calibrate_text_score(raw, &self.calibration);
        let calibrate_cross_modal = |raw: f32| calibrate_image_score(raw, &self.calibration);

        // Raw similarity per document: (text partition, image partition).
        let mut dots: HashMap<Arc<str>, (Option<f32>, Option<f32>)> = HashMap::new();
        for (id, dot, _, _) in &text_matches {
            dots.entry(Arc::clone(id)).or_default().0 = Some(*dot);
        }
        for (id, dot, _, _) in &image_matches {
            dots.entry(Arc::clone(id)).or_default().1 = Some(*dot);
        }

        let mut payloads: HashMap<Arc<str>, (Arc<Value>, Modality)> =
            HashMap::with_capacity(dots.len());
        for (id, _, modality, payload) in text_matches.iter().chain(image_matches.iter()) {
            payloads
                .entry(Arc::clone(id))
                .or_insert_with(|| (Arc::clone(payload), modality.clone()));
        }

        struct ScoredDoc {
            id: Arc<str>,
            score: f32,
            modality: Modality,
            matched_by: Modality,
            payload: Arc<Value>,
        }

        let mut scored_docs: Vec<ScoredDoc> = match query_text {
            Some(query) => {
                // Text query: rank-fuse the dense lists with the lexical list.
                // A document enters a dense list only if it is above that signal's
                // noise floor, so RRF cannot reward unrelated documents.
                // Text vs text uses the text floor; cross-modal uses the cross-modal floor.
                // An image query flips which partition is cross-modal.
                let (text_floor, image_floor) = if query_modality == Some(&Modality::Image) {
                    (
                        self.calibration.cross_modal_min,
                        self.calibration.text_noise_floor,
                    )
                } else {
                    (
                        self.calibration.text_noise_floor,
                        self.calibration.cross_modal_min,
                    )
                };
                // Documents with an image are ranked by their image vector only.
                // Text-only documents are ranked by their text vector and by lexical match.
                let text_only: Vec<PartitionMatch> = text_matches
                    .iter()
                    .filter(|m| m.2 == Modality::Text)
                    .cloned()
                    .collect();
                let text_pos = positions(&rank_ids(&text_only, text_floor));
                let image_pos = positions(&rank_ids(&image_matches, image_floor));

                // Lexical list: text-only documents with at least one query-term hit.
                let mut hits: Vec<(LexicalMatch, Arc<str>)> = payloads
                    .iter()
                    .filter_map(|(id, (payload, modality))| {
                        if *modality != Modality::Text {
                            return None;
                        }
                        let lex = compute_lexical_match(query, payload);
                        (lex.ratio > 0.0).then(|| (lex, Arc::clone(id)))
                    })
                    .collect();
                hits.sort_by(|(a, a_id), (b, b_id)| {
                    b.ratio
                        .total_cmp(&a.ratio)
                        .then_with(|| b.has_exact_bigram.cmp(&a.has_exact_bigram))
                        .then_with(|| b.has_close_pair.cmp(&a.has_close_pair))
                        .then_with(|| b.in_title.cmp(&a.in_title))
                        .then_with(|| a_id.cmp(b_id))
                });
                let lexical_order: Vec<Arc<str>> = hits.into_iter().map(|(_, id)| id).collect();
                let lexical_pos = positions(&lexical_order);

                let k = DEFAULT_RRF_K as f32;
                let contribution = |pos: Option<&usize>, weight: f32| {
                    pos.map_or(0.0, |&rank| weight / (k + rank as f32))
                };

                let fused: Vec<(Arc<str>, f32)> = payloads
                    .keys()
                    .map(|id| {
                        let rrf = contribution(text_pos.get(id), RRF_TEXT_WEIGHT)
                            + contribution(image_pos.get(id), RRF_IMAGE_WEIGHT)
                            + contribution(lexical_pos.get(id), RRF_LEXICAL_WEIGHT);
                        (Arc::clone(id), rrf)
                    })
                    .collect();
                let best = fused.iter().map(|(_, rrf)| *rrf).fold(0.0f32, f32::max);

                // Documents above no noise floor and with no lexical hit are not candidates.
                fused
                    .into_iter()
                    .filter(|(_, rrf)| *rrf > 0.0)
                    .map(|(id, rrf)| {
                        let (payload, modality) = payloads[&id].clone();
                        let matched_by = match (text_pos.get(&id), image_pos.get(&id)) {
                            (Some(t), Some(i)) if i < t => Modality::Image,
                            (None, Some(_)) => Modality::Image,
                            _ => Modality::Text,
                        };
                        tracing::debug!(
                            target: "search_scoring",
                            doc_id = %id,
                            text_rank = ?text_pos.get(&id),
                            image_rank = ?image_pos.get(&id),
                            lexical_rank = ?lexical_pos.get(&id),
                            rrf,
                            "rank fusion"
                        );
                        ScoredDoc {
                            id,
                            score: rrf / best,
                            modality,
                            matched_by,
                            payload,
                        }
                    })
                    .collect()
            }
            None => {
                // No text query: same scoring as before the RRF change.
                // Raw similarity when there is no modality; calibrated similarity otherwise.
                dots.into_iter()
                    .map(|(id, (text_dot, img_dot))| {
                        let (payload, modality) = payloads[&id].clone();
                        let (score, matched_by) = match query_modality {
                            None => match (text_dot, img_dot) {
                                (Some(t), Some(i)) => {
                                    if t >= i {
                                        (t, Modality::Text)
                                    } else {
                                        (i, Modality::Image)
                                    }
                                }
                                (Some(t), None) => (t, Modality::Text),
                                (None, Some(i)) => (i, Modality::Image),
                                (None, None) => (0.0, Modality::Text),
                            },
                            Some(Modality::Image) => match (img_dot, text_dot) {
                                (Some(i), Some(t)) => {
                                    let img_s = calibrate_text_similarity(i);
                                    let txt_s = calibrate_cross_modal(t);
                                    let consistency = if txt_s < 0.08 {
                                        (txt_s / 0.08).powf(2.0)
                                    } else {
                                        1.0
                                    };
                                    let combined = (0.50 * img_s + 0.50 * txt_s) * consistency;
                                    (combined.min(1.0), Modality::Image)
                                }
                                (Some(i), None) => (calibrate_text_similarity(i), Modality::Image),
                                (None, Some(t)) => (calibrate_cross_modal(t), Modality::Text),
                                (None, None) => (0.0, Modality::Image),
                            },
                            Some(_) => match (text_dot, img_dot) {
                                (Some(t), Some(i)) => {
                                    let txt_s = calibrate_text_similarity(t);
                                    let img_s = calibrate_cross_modal(i);
                                    let consistency = if img_s < 0.08 {
                                        (img_s / 0.08).powf(2.0)
                                    } else {
                                        1.0
                                    };
                                    let base = 0.55 * txt_s + 0.45 * img_s;
                                    let matched = if img_s > txt_s {
                                        Modality::Image
                                    } else {
                                        Modality::Text
                                    };
                                    ((base * consistency).min(1.0), matched)
                                }
                                (Some(t), None) => (calibrate_text_similarity(t), Modality::Text),
                                (None, Some(i)) => (calibrate_cross_modal(i), Modality::Image),
                                (None, None) => (0.0, Modality::Text),
                            },
                        };
                        ScoredDoc {
                            id,
                            score,
                            modality,
                            matched_by,
                            payload,
                        }
                    })
                    .collect()
            }
        };

        if let Some(ref f) = filter {
            if let Some(min_s) = f.min_score {
                scored_docs.retain(|r| r.score >= min_s);
            }
            if let Some(ref target_m) = f.modality {
                scored_docs
                    .retain(|r| &r.modality == target_m || r.modality == Modality::Multimodal);
            }
        }

        scored_docs.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.id.cmp(&b.id)));
        scored_docs.truncate(limit);

        let mut final_results = Vec::with_capacity(scored_docs.len());
        for item in scored_docs {
            final_results.push(SearchResult {
                id: item.id.to_string(),
                score: item.score,
                payload: (*item.payload).clone(),
                modality: item.modality,
                matched_by: Some(item.matched_by),
            });
        }

        Ok(final_results)
    }

    pub(crate) fn get_blocking(
        &self,
        id: &DocumentId,
    ) -> Result<Option<VectorPoint>, StorageError> {
        let (t_lock, i_lock) = self.read_states()?;
        let t_slot = t_lock.index_map.get(id).copied();
        let i_slot = i_lock.index_map.get(id).copied();
        Ok(extract_point_from_slots(
            id,
            t_slot,
            i_slot,
            &t_lock,
            &i_lock,
            self.dimension,
        ))
    }

    pub(crate) fn delete_blocking(&self, id: &DocumentId) -> Result<bool, StorageError> {
        let (mut t_state, mut i_state) = self.write_states()?;

        let del_text =
            Self::remove_from_state_and_hnsw(&mut t_state, &self.hnsw_text, id, self.dimension)?;
        let del_image =
            Self::remove_from_state_and_hnsw(&mut i_state, &self.hnsw_image, id, self.dimension)?;
        let deleted = del_text || del_image;

        if deleted {
            self.doc_count.fetch_sub(1, AtomicOrdering::Release);
            self.version.fetch_add(1, AtomicOrdering::Release);
        }
        Ok(deleted)
    }

    pub(crate) fn list_documents_blocking(&self) -> Result<Vec<DocumentSummary>, StorageError> {
        let (t_lock, i_lock) = self.read_states()?;
        let all_ids = collect_all_document_ids(&t_lock, &i_lock);

        let mut summaries = Vec::with_capacity(all_ids.len());
        for id in all_ids {
            let in_text = t_lock.index_map.get(&id).copied();
            let in_image = i_lock.index_map.get(&id).copied();
            let (payload, modality) = match (in_text, in_image) {
                (Some(ts), Some(_)) => ((*t_lock.payloads[ts]).clone(), Modality::Multimodal),
                (Some(ts), None) => ((*t_lock.payloads[ts]).clone(), Modality::Text),
                (None, Some(is)) => ((*i_lock.payloads[is]).clone(), Modality::Image),
                (None, None) => continue,
            };
            summaries.push(DocumentSummary {
                id,
                modality,
                payload,
            });
        }
        summaries.sort_by(|a, b| a.id.cmp(&b.id));
        Ok(summaries)
    }

    pub(crate) fn collect_points_for_snapshot(
        &self,
        chunk: &[DocumentId],
        dim: usize,
    ) -> Result<Vec<VectorPoint>, StorageError> {
        let (t_lock, i_lock) = self.read_states()?;
        let mut pts = Vec::with_capacity(chunk.len());
        for id in chunk {
            let t_slot = t_lock.index_map.get(id).copied();
            let i_slot = i_lock.index_map.get(id).copied();
            if let Some(pt) = extract_point_from_slots(id, t_slot, i_slot, &t_lock, &i_lock, dim) {
                pts.push(pt);
            }
        }
        Ok(pts)
    }

    fn save_to_disk_blocking(&self, path: &Path) -> Result<(), StorageError> {
        let _guard = self
            .disk_lock
            .lock()
            .map_err(|_| StorageError::Internal("Disk lock poisoned".into()))?;

        let current_version = self.version.load(AtomicOrdering::Acquire);
        let saved_version = self.persisted_version.load(AtomicOrdering::Acquire);
        if current_version <= saved_version && path.exists() {
            return Ok(());
        }

        let all_ids: Vec<DocumentId> = {
            let (t_lock, i_lock) = self.read_states()?;
            collect_all_document_ids(&t_lock, &i_lock)
                .into_iter()
                .collect()
        };

        save_snapshot(self, path, &all_ids)?;

        self.persisted_version
            .store(current_version, AtomicOrdering::Release);
        Ok(())
    }

    fn load_from_disk_blocking(&self, path: &Path) -> Result<usize, StorageError> {
        let _guard = self
            .disk_lock
            .lock()
            .map_err(|_| StorageError::Internal("Disk lock poisoned".into()))?;

        let Some(temp_store) = load_snapshot(self, path)? else {
            return Ok(0);
        };

        let loaded_count;
        {
            let (mut t_state, mut i_state) = self.write_states()?;
            let mut hnsw_t = self
                .hnsw_text
                .write()
                .map_err(|_| StorageError::Internal("HNSW text write lock poisoned".into()))?;
            let mut hnsw_i = self
                .hnsw_image
                .write()
                .map_err(|_| StorageError::Internal("HNSW image write lock poisoned".into()))?;

            let (mut temp_t_state, mut temp_i_state) = temp_store.write_states()?;
            let mut temp_hnsw_t = temp_store
                .hnsw_text
                .write()
                .map_err(|_| StorageError::Internal("Temp HNSW text lock poisoned".into()))?;
            let mut temp_hnsw_i = temp_store
                .hnsw_image
                .write()
                .map_err(|_| StorageError::Internal("Temp HNSW image lock poisoned".into()))?;

            let all_ids = collect_all_document_ids(&temp_t_state, &temp_i_state);
            loaded_count = all_ids.len();

            *t_state = std::mem::take(&mut *temp_t_state);
            *i_state = std::mem::take(&mut *temp_i_state);
            *hnsw_t = std::mem::replace(&mut *temp_hnsw_t, HnswIndex::new(HnswConfig::default()));
            *hnsw_i = std::mem::replace(&mut *temp_hnsw_i, HnswIndex::new(HnswConfig::default()));
        }

        self.doc_count.store(loaded_count, AtomicOrdering::Release);
        self.version.store(0, AtomicOrdering::Release);
        self.persisted_version.store(0, AtomicOrdering::Release);
        Ok(loaded_count)
    }
}

#[async_trait]
impl VectorStore for InMemoryVectorStore {
    async fn insert(&self, point: VectorPoint) -> Result<(), StorageError> {
        self.validate_point_vectors(&point)?;

        let store = self.clone();
        tokio::task::spawn_blocking(move || store.insert_blocking(point))
            .await
            .map_err(|e| StorageError::Internal(format!("Insert task failed to join: {e}")))?
    }

    async fn insert_batch(&self, points: Vec<VectorPoint>) -> Result<(), StorageError> {
        for point in &points {
            self.validate_point_vectors(point)?;
        }

        let store = self.clone();
        tokio::task::spawn_blocking(move || store.insert_batch_blocking(points))
            .await
            .map_err(|e| StorageError::Internal(format!("Insert batch task failed to join: {e}")))?
    }

    async fn search(
        &self,
        query_vector: &[f32],
        limit: usize,
        filter: Option<SearchFilter>,
    ) -> Result<Vec<SearchResult>, StorageError> {
        self.validate_vector(query_vector)?;
        Self::validate_filter(&filter)?;
        if limit == 0 {
            return Ok(Vec::new());
        }

        let store = self.clone();
        let q = query_vector.to_vec();
        tokio::task::spawn_blocking(move || store.search_blocking(&q, limit, filter))
            .await
            .map_err(|e| StorageError::Internal(format!("Search task failed to join: {e}")))?
    }

    async fn get(&self, id: &DocumentId) -> Result<Option<VectorPoint>, StorageError> {
        let store = self.clone();
        let doc_id = id.clone();
        tokio::task::spawn_blocking(move || store.get_blocking(&doc_id))
            .await
            .map_err(|e| StorageError::Internal(format!("Get task failed to join: {e}")))?
    }

    async fn list_documents(&self) -> Result<Vec<DocumentSummary>, StorageError> {
        let store = self.clone();
        tokio::task::spawn_blocking(move || store.list_documents_blocking())
            .await
            .map_err(|e| {
                StorageError::Internal(format!("List documents task failed to join: {e}"))
            })?
    }

    async fn delete(&self, id: &DocumentId) -> Result<bool, StorageError> {
        let store = self.clone();
        let doc_id = id.clone();
        tokio::task::spawn_blocking(move || store.delete_blocking(&doc_id))
            .await
            .map_err(|e| StorageError::Internal(format!("Delete task failed to join: {e}")))?
    }

    async fn count(&self) -> Result<usize, StorageError> {
        Ok(self.doc_count.load(AtomicOrdering::Acquire))
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    async fn save_to_disk(&self, path: &Path) -> Result<(), StorageError> {
        if !self.is_dirty() && path.exists() {
            return Ok(());
        }

        let store = self.clone();
        let target_path = path.to_path_buf();
        tokio::task::spawn_blocking(move || store.save_to_disk_blocking(&target_path))
            .await
            .map_err(|e| StorageError::Internal(format!("Save snapshot task panicked: {e}")))?
    }

    async fn load_from_disk(&self, path: &Path) -> Result<usize, StorageError> {
        let store = self.clone();
        let target_path = path.to_path_buf();
        tokio::task::spawn_blocking(move || store.load_from_disk_blocking(&target_path))
            .await
            .map_err(|e| StorageError::Internal(format!("Load snapshot task panicked: {e}")))?
    }
}
