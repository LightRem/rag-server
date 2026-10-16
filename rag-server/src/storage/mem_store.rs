use super::{
    dot_product,
    hnsw::{HnswConfig, HnswIndex},
    models::{SearchFilter, SearchResult, VectorPoint},
    partition::StorageState,
    persistence::{load_snapshot, save_snapshot},
    scoring::{
        CalibrationConfig, calculate_keyword_boost, calibrate_image_score, calibrate_text_score,
        fuse_scores, has_image_keywords,
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
        Arc, Mutex as StdMutex, RwLock,
        atomic::{AtomicU64, AtomicUsize, Ordering as AtomicOrdering},
    },
};

pub const DEFAULT_MAX_CAPACITY: usize = 1_000_000;
const L2_NORM_SQ_TOLERANCE: f32 = 1e-2;
const HNSW_MIN_POINTS_THRESHOLD: usize = 128;

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

    pub(crate) fn insert_blocking(&self, point: VectorPoint) -> Result<(), StorageError> {
        let mut t_state = self.text_state.write().map_err(|_| {
            StorageError::Internal("Vector store text_state write lock poisoned".into())
        })?;
        let mut i_state = self.image_state.write().map_err(|_| {
            StorageError::Internal("Vector store image_state write lock poisoned".into())
        })?;

        let is_existing =
            t_state.index_map.contains_key(&point.id) || i_state.index_map.contains_key(&point.id);
        let current_count = self.doc_count.load(AtomicOrdering::Acquire);
        if !is_existing && current_count >= self.max_capacity {
            return Err(StorageError::CapacityExceeded {
                current: current_count,
                max: self.max_capacity,
            });
        }

        let payload_arc = Arc::new(point.payload);
        match point.modality {
            Modality::Text => {
                self.validate_vector(&point.vector)?;
                Self::insert_into_state_and_hnsw(
                    &mut t_state,
                    &self.hnsw_text,
                    &point.id,
                    &point.vector,
                    payload_arc,
                    Modality::Text,
                    self.dimension,
                )?;
                Self::remove_from_state_and_hnsw(
                    &mut i_state,
                    &self.hnsw_image,
                    &point.id,
                    self.dimension,
                )?;
            }
            Modality::Image => {
                self.validate_vector(&point.vector)?;
                Self::insert_into_state_and_hnsw(
                    &mut i_state,
                    &self.hnsw_image,
                    &point.id,
                    &point.vector,
                    payload_arc,
                    Modality::Image,
                    self.dimension,
                )?;
                Self::remove_from_state_and_hnsw(
                    &mut t_state,
                    &self.hnsw_text,
                    &point.id,
                    self.dimension,
                )?;
            }
            Modality::Multimodal => {
                self.validate_vector(&point.vector)?;
                let img_vec = point.image_vector.as_deref().unwrap_or(&point.vector);
                self.validate_vector(img_vec)?;

                Self::insert_into_state_and_hnsw(
                    &mut t_state,
                    &self.hnsw_text,
                    &point.id,
                    &point.vector,
                    Arc::clone(&payload_arc),
                    Modality::Multimodal,
                    self.dimension,
                )?;
                Self::insert_into_state_and_hnsw(
                    &mut i_state,
                    &self.hnsw_image,
                    &point.id,
                    img_vec,
                    payload_arc,
                    Modality::Multimodal,
                    self.dimension,
                )?;
            }
        }

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
        let mut t_state = self.text_state.write().map_err(|_| {
            StorageError::Internal("Vector store text_state write lock poisoned".into())
        })?;
        let mut i_state = self.image_state.write().map_err(|_| {
            StorageError::Internal("Vector store image_state write lock poisoned".into())
        })?;

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
            let payload_arc = Arc::new(point.payload);

            match point.modality {
                Modality::Text => {
                    Self::insert_into_state_and_hnsw(
                        &mut t_state,
                        &self.hnsw_text,
                        &point.id,
                        &point.vector,
                        payload_arc,
                        Modality::Text,
                        self.dimension,
                    )?;
                    Self::remove_from_state_and_hnsw(
                        &mut i_state,
                        &self.hnsw_image,
                        &point.id,
                        self.dimension,
                    )?;
                }
                Modality::Image => {
                    Self::insert_into_state_and_hnsw(
                        &mut i_state,
                        &self.hnsw_image,
                        &point.id,
                        &point.vector,
                        payload_arc,
                        Modality::Image,
                        self.dimension,
                    )?;
                    Self::remove_from_state_and_hnsw(
                        &mut t_state,
                        &self.hnsw_text,
                        &point.id,
                        self.dimension,
                    )?;
                }
                Modality::Multimodal => {
                    let img_vec = point.image_vector.as_deref().unwrap_or(&point.vector);
                    Self::insert_into_state_and_hnsw(
                        &mut t_state,
                        &self.hnsw_text,
                        &point.id,
                        &point.vector,
                        Arc::clone(&payload_arc),
                        Modality::Multimodal,
                        self.dimension,
                    )?;
                    Self::insert_into_state_and_hnsw(
                        &mut i_state,
                        &self.hnsw_image,
                        &point.id,
                        img_vec,
                        payload_arc,
                        Modality::Multimodal,
                        self.dimension,
                    )?;
                }
            }

            if !is_existing {
                self.doc_count.fetch_add(1, AtomicOrdering::Release);
            }
        }

        self.version.fetch_add(1, AtomicOrdering::Release);
        Ok(())
    }

    fn search_partition_zero_copy(
        hnsw: &RwLock<HnswIndex>,
        state: &RwLock<StorageState>,
        query_vector: &[f32],
        limit: usize,
        dimension: usize,
        filter: &Option<SearchFilter>,
    ) -> Result<Vec<(Arc<str>, f32, Modality)>, StorageError> {
        let state_lock = state.read().map_err(|_| {
            StorageError::Internal("Vector store partition read lock poisoned".into())
        })?;
        let total_points = state_lock.len();
        if total_points == 0 {
            return Ok(Vec::new());
        }

        let effective_limit = limit.min(total_points);
        let candidates: Vec<(f32, usize)> = if total_points >= HNSW_MIN_POINTS_THRESHOLD {
            let hnsw_lock = hnsw
                .read()
                .map_err(|_| StorageError::Internal("HNSW index read lock poisoned".into()))?;
            let hnsw_res = hnsw_lock.search(
                query_vector,
                effective_limit,
                &state_lock.vectors,
                &state_lock.modalities,
                &state_lock.ids,
                dimension,
                filter,
            );
            if filter.is_some() && hnsw_res.len() < effective_limit {
                drop(hnsw_lock);
                Self::exact_flat_scan(
                    &state_lock,
                    query_vector,
                    effective_limit,
                    filter,
                    dimension,
                )
            } else {
                hnsw_res
            }
        } else {
            Self::exact_flat_scan(
                &state_lock,
                query_vector,
                effective_limit,
                filter,
                dimension,
            )
        };

        let mut results = Vec::with_capacity(candidates.len());
        for (score, slot) in candidates {
            results.push((
                Arc::clone(&state_lock.ids[slot]),
                score,
                state_lock.modalities[slot].clone(),
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

        // Для коллекций < 256 векторов последовательный проход быстрее,
        // так как отсутствует оверхед на координацию рабочих потоков Rayon.
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
        let search_k = limit.saturating_mul(2).max(10);
        let explicit_modality_filter = filter.as_ref().and_then(|f| f.modality.as_ref());

        let mut partition_filter = filter.clone();
        if let Some(ref mut pf) = partition_filter {
            pf.min_score = None;
        }

        let (text_matches, image_matches) = match explicit_modality_filter {
            Some(Modality::Text) => {
                let t_res = Self::search_partition_zero_copy(
                    &self.hnsw_text,
                    &self.text_state,
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
                    &self.image_state,
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
                    &self.text_state,
                    query_vector,
                    search_k,
                    self.dimension,
                    &partition_filter,
                )?;
                let i_res = Self::search_partition_zero_copy(
                    &self.hnsw_image,
                    &self.image_state,
                    query_vector,
                    search_k,
                    self.dimension,
                    &partition_filter,
                )?;
                (t_res, i_res)
            }
        };

        let effective_query_modality = filter
            .as_ref()
            .and_then(|f| f.query_modality.clone())
            .or_else(|| {
                if filter
                    .as_ref()
                    .and_then(|f| f.query_text.as_deref())
                    .is_some()
                    || self.dimension >= 128
                {
                    Some(Modality::Text)
                } else {
                    None
                }
            });

        let has_image_intent = filter
            .as_ref()
            .and_then(|f| f.query_text.as_deref())
            .is_some_and(has_image_keywords);

        struct FastMatch {
            id: Arc<str>,
            score: f32,
            text_score: Option<f32>,
            img_score: Option<f32>,
            modality: Modality,
            matched_by: Modality,
        }

        let mut merged: HashMap<Arc<str>, FastMatch> =
            HashMap::with_capacity(text_matches.len() + image_matches.len());

        for (id, score, modality) in text_matches {
            let final_score = match effective_query_modality {
                Some(Modality::Image) => calibrate_image_score(score, false, &self.calibration),
                Some(Modality::Text) | Some(Modality::Multimodal) => {
                    calibrate_text_score(score, &self.calibration)
                }
                None => score,
            };
            merged.insert(
                Arc::clone(&id),
                FastMatch {
                    id,
                    score: final_score,
                    text_score: Some(final_score),
                    img_score: None,
                    modality,
                    matched_by: Modality::Text,
                },
            );
        }

        for (id, raw_score, modality) in image_matches {
            let img_final = match effective_query_modality {
                Some(Modality::Text) | Some(Modality::Multimodal) => {
                    calibrate_image_score(raw_score, has_image_intent, &self.calibration)
                }
                _ => raw_score,
            };

            if let Some(existing) = merged.get_mut(&id) {
                existing.img_score = Some(img_final);
                let text_s = existing.text_score.unwrap_or(existing.score);
                let (fused, matched_by) =
                    fuse_scores(effective_query_modality.as_ref(), img_final, text_s);
                existing.score = fused;
                existing.matched_by = matched_by;
            } else {
                merged.insert(
                    Arc::clone(&id),
                    FastMatch {
                        id,
                        score: img_final,
                        text_score: None,
                        img_score: Some(img_final),
                        modality,
                        matched_by: Modality::Image,
                    },
                );
            }
        }

        let t_lock = self.text_state.read().map_err(|_| {
            StorageError::Internal("Vector store text_state read lock poisoned".into())
        })?;
        let i_lock = self.image_state.read().map_err(|_| {
            StorageError::Internal("Vector store image_state read lock poisoned".into())
        })?;

        if let Some(query_str) = filter.as_ref().and_then(|f| f.query_text.as_deref()) {
            for m in merged.values_mut() {
                let doc_id = m.id.to_string();
                let payload_opt = if let Some(&slot) = t_lock.index_map.get(&doc_id) {
                    Some(&t_lock.payloads[slot])
                } else {
                    i_lock
                        .index_map
                        .get(&doc_id)
                        .map(|&slot| &i_lock.payloads[slot])
                };
                if let Some(payload) = payload_opt {
                    let boost = calculate_keyword_boost(query_str, payload);
                    m.score = (m.score + boost).min(1.0);
                }
            }
        }

        let mut ranked: Vec<FastMatch> = merged.into_values().collect();
        if let Some(ref f) = filter {
            if let Some(min_s) = f.min_score {
                ranked.retain(|r| r.score >= min_s);
            }
            if let Some(ref target_m) = f.modality {
                ranked.retain(|r| &r.modality == target_m || r.modality == Modality::Multimodal);
            }
        }
        ranked.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.id.cmp(&b.id)));
        ranked.truncate(limit);

        let mut final_results = Vec::with_capacity(ranked.len());
        for item in ranked {
            let doc_id = item.id.to_string();
            let payload = if let Some(&slot) = t_lock.index_map.get(&doc_id) {
                (*t_lock.payloads[slot]).clone()
            } else if let Some(&slot) = i_lock.index_map.get(&doc_id) {
                (*i_lock.payloads[slot]).clone()
            } else {
                Value::Null
            };
            final_results.push(SearchResult {
                id: doc_id,
                score: item.score,
                payload,
                modality: item.modality,
                matched_by: Some(item.matched_by),
            });
        }

        Ok(final_results)
    }

    pub(crate) fn collect_points_for_snapshot(
        &self,
        chunk: &[DocumentId],
        dim: usize,
    ) -> Result<Vec<VectorPoint>, StorageError> {
        let t_lock = self.text_state.read().map_err(|_| {
            StorageError::Internal("Vector store text_state read lock poisoned".into())
        })?;
        let i_lock = self.image_state.read().map_err(|_| {
            StorageError::Internal("Vector store image_state read lock poisoned".into())
        })?;

        let mut pts = Vec::with_capacity(chunk.len());
        for id in chunk {
            let t_slot = t_lock.index_map.get(id).copied();
            let i_slot = i_lock.index_map.get(id).copied();
            match (t_slot, i_slot) {
                (Some(ts), Some(is)) => {
                    let text_vec = t_lock.vectors[ts * dim..(ts + 1) * dim].to_vec();
                    let img_vec = i_lock.vectors[is * dim..(is + 1) * dim].to_vec();
                    pts.push(
                        VectorPoint::new(
                            id.clone(),
                            text_vec,
                            (*t_lock.payloads[ts]).clone(),
                            Modality::Multimodal,
                        )
                        .with_image_vector(img_vec),
                    );
                }
                (Some(ts), None) => {
                    let text_vec = t_lock.vectors[ts * dim..(ts + 1) * dim].to_vec();
                    pts.push(VectorPoint::new(
                        id.clone(),
                        text_vec,
                        (*t_lock.payloads[ts]).clone(),
                        Modality::Text,
                    ));
                }
                (None, Some(is)) => {
                    let img_vec = i_lock.vectors[is * dim..(is + 1) * dim].to_vec();
                    pts.push(VectorPoint::new(
                        id.clone(),
                        img_vec,
                        (*i_lock.payloads[is]).clone(),
                        Modality::Image,
                    ));
                }
                (None, None) => {}
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
            let t_lock = self.text_state.read().map_err(|_| {
                StorageError::Internal("Vector store text_state read lock poisoned".into())
            })?;
            let i_lock = self.image_state.read().map_err(|_| {
                StorageError::Internal("Vector store image_state read lock poisoned".into())
            })?;
            let mut ids = HashSet::with_capacity(t_lock.len() + i_lock.len());
            ids.extend(t_lock.index_map.keys().cloned());
            ids.extend(i_lock.index_map.keys().cloned());
            ids.into_iter().collect()
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
            let mut t_state = self.text_state.write().map_err(|_| {
                StorageError::Internal("Vector store text_state write lock poisoned".into())
            })?;
            let mut i_state = self.image_state.write().map_err(|_| {
                StorageError::Internal("Vector store image_state write lock poisoned".into())
            })?;
            let mut hnsw_t = self
                .hnsw_text
                .write()
                .map_err(|_| StorageError::Internal("HNSW text write lock poisoned".into()))?;
            let mut hnsw_i = self
                .hnsw_image
                .write()
                .map_err(|_| StorageError::Internal("HNSW image write lock poisoned".into()))?;

            let mut temp_t_state = temp_store
                .text_state
                .write()
                .map_err(|_| StorageError::Internal("Temp text_state lock poisoned".into()))?;
            let mut temp_i_state = temp_store
                .image_state
                .write()
                .map_err(|_| StorageError::Internal("Temp image_state lock poisoned".into()))?;
            let mut temp_hnsw_t = temp_store
                .hnsw_text
                .write()
                .map_err(|_| StorageError::Internal("Temp HNSW text lock poisoned".into()))?;
            let mut temp_hnsw_i = temp_store
                .hnsw_image
                .write()
                .map_err(|_| StorageError::Internal("Temp HNSW image lock poisoned".into()))?;

            let mut all_ids = HashSet::new();
            all_ids.extend(temp_t_state.index_map.keys().cloned());
            all_ids.extend(temp_i_state.index_map.keys().cloned());
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

        let store = self.clone();
        tokio::task::spawn_blocking(move || store.insert_blocking(point))
            .await
            .map_err(|e| StorageError::Internal(format!("Insert task failed to join: {e}")))?
    }

    async fn insert_batch(&self, points: Vec<VectorPoint>) -> Result<(), StorageError> {
        if self.dimension == 0 {
            return Err(StorageError::InvalidVector(
                "Vector store configured with invalid dimension 0".into(),
            ));
        }
        for point in &points {
            if point.id.trim().is_empty() {
                return Err(StorageError::InvalidVector(
                    "Document ID cannot be empty".into(),
                ));
            }
            self.validate_vector(&point.vector)?;
            if let Some(ref img_v) = point.image_vector {
                self.validate_vector(img_v)?;
            }
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
        let t_lock = self.text_state.read().map_err(|_| {
            StorageError::Internal("Vector store text_state read lock poisoned".into())
        })?;
        let i_lock = self.image_state.read().map_err(|_| {
            StorageError::Internal("Vector store image_state read lock poisoned".into())
        })?;

        let t_slot = t_lock.index_map.get(id).copied();
        let i_slot = i_lock.index_map.get(id).copied();
        if t_slot.is_none() && i_slot.is_none() {
            return Ok(None);
        }

        let dim = self.dimension;
        match (t_slot, i_slot) {
            (Some(ts), Some(is)) => {
                let text_vec = t_lock.vectors[ts * dim..(ts + 1) * dim].to_vec();
                let img_vec = i_lock.vectors[is * dim..(is + 1) * dim].to_vec();
                Ok(Some(
                    VectorPoint::new(
                        id.clone(),
                        text_vec,
                        (*t_lock.payloads[ts]).clone(),
                        Modality::Multimodal,
                    )
                    .with_image_vector(img_vec),
                ))
            }
            (Some(ts), None) => {
                let text_vec = t_lock.vectors[ts * dim..(ts + 1) * dim].to_vec();
                Ok(Some(VectorPoint::new(
                    id.clone(),
                    text_vec,
                    (*t_lock.payloads[ts]).clone(),
                    Modality::Text,
                )))
            }
            (None, Some(is)) => {
                let img_vec = i_lock.vectors[is * dim..(is + 1) * dim].to_vec();
                Ok(Some(VectorPoint::new(
                    id.clone(),
                    img_vec,
                    (*i_lock.payloads[is]).clone(),
                    Modality::Image,
                )))
            }
            (None, None) => Ok(None),
        }
    }

    async fn list_documents(&self) -> Result<Vec<DocumentSummary>, StorageError> {
        let t_lock = self.text_state.read().map_err(|_| {
            StorageError::Internal("Vector store text_state read lock poisoned".into())
        })?;
        let i_lock = self.image_state.read().map_err(|_| {
            StorageError::Internal("Vector store image_state read lock poisoned".into())
        })?;

        let mut all_ids: HashSet<DocumentId> = HashSet::with_capacity(t_lock.len() + i_lock.len());
        all_ids.extend(t_lock.index_map.keys().cloned());
        all_ids.extend(i_lock.index_map.keys().cloned());

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

    async fn delete(&self, id: &DocumentId) -> Result<bool, StorageError> {
        let mut t_state = self.text_state.write().map_err(|_| {
            StorageError::Internal("Vector store text_state write lock poisoned".into())
        })?;
        let mut i_state = self.image_state.write().map_err(|_| {
            StorageError::Internal("Vector store image_state write lock poisoned".into())
        })?;

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
