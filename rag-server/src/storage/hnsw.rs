use super::{dot_product, models::SearchFilter};
use crate::domain::Modality;
use std::{cell::RefCell, cmp::Ordering, collections::BinaryHeap, sync::Arc};

thread_local! {
    static VISITED_TRACKER: RefCell<(Vec<u32>, u32)> = const { RefCell::new((Vec::new(), 0)) };
}

pub struct VectorView<'a> {
    pub vectors: &'a [f32],
    pub ids: &'a [Arc<str>],
    pub modalities: &'a [Modality],
    pub dim: usize,
}

#[derive(Debug, Clone)]
pub struct HnswConfig {
    pub m: usize,
    pub m0: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: 16,
            m0: 32,
            ef_construction: 128,
            ef_search: 64,
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
struct Candidate {
    score: f32,
    slot: usize,
}

impl Eq for Candidate {}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .total_cmp(&other.score)
            .then_with(|| other.slot.cmp(&self.slot))
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Copy)]
struct SearchCandidate<'a> {
    score: f32,
    slot: usize,
    id: &'a str,
}

impl<'a> PartialEq for SearchCandidate<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.score.total_cmp(&other.score) == Ordering::Equal && self.id == other.id
    }
}

impl<'a> Eq for SearchCandidate<'a> {}

impl<'a> Ord for SearchCandidate<'a> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .total_cmp(&other.score)
            .then_with(|| other.id.cmp(self.id))
    }
}

impl<'a> PartialOrd for SearchCandidate<'a> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Debug, Default)]
struct HnswNode {
    layers: Vec<Vec<usize>>,
    in_layers: Vec<Vec<usize>>,
}

#[inline]
fn get_vector(all_vectors: &[f32], slot: usize, dim: usize) -> &[f32] {
    let start = slot * dim;
    &all_vectors[start..start + dim]
}

fn select_neighbors(
    base_vec: &[f32],
    candidate_slots: &[usize],
    max_edges: usize,
    all_vectors: &[f32],
    dim: usize,
) -> Vec<usize> {
    let mut scored: Vec<(f32, usize)> = candidate_slots
        .iter()
        .copied()
        .map(|slot| {
            (
                dot_product(base_vec, get_vector(all_vectors, slot, dim)),
                slot,
            )
        })
        .collect();

    scored.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
    let mut result: Vec<usize> = Vec::with_capacity(max_edges);
    let mut discarded: Vec<usize> = Vec::new();

    for (_, slot) in scored {
        if result.contains(&slot) {
            continue;
        }
        let cand_vec = get_vector(all_vectors, slot, dim);
        let mut keep = true;
        for &r in &result {
            let r_vec = get_vector(all_vectors, r, dim);
            let sim_to_r = dot_product(cand_vec, r_vec);
            if sim_to_r >= 0.9999 {
                keep = false;
                break;
            }
        }
        if keep {
            result.push(slot);
            if result.len() >= max_edges {
                break;
            }
        } else {
            discarded.push(slot);
        }
    }

    for slot in discarded {
        if result.len() >= max_edges {
            break;
        }
        if !result.contains(&slot) {
            result.push(slot);
        }
    }
    result
}

pub struct HnswIndex {
    config: HnswConfig,
    nodes: Vec<HnswNode>,
    entry_point: Option<usize>,
    max_layer: usize,
    rng_state: u64,
}

impl HnswIndex {
    pub fn new(config: HnswConfig) -> Self {
        Self {
            config,
            nodes: Vec::new(),
            entry_point: None,
            max_layer: 0,
            rng_state: 0x8541_5289_4798_1253,
        }
    }

    fn random_level(&mut self) -> usize {
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 7;
        self.rng_state ^= self.rng_state << 17;
        let r = ((self.rng_state as f64) / (u64::MAX as f64)).max(1e-15);
        let mult = 1.0 / (self.config.m as f64).ln().max(1e-5);
        let lvl = (-r.ln() * mult).floor() as usize;
        lvl.min(16)
    }

    fn set_neighbors(&mut self, from: usize, layer: usize, new_neighbors: Vec<usize>) {
        if from >= self.nodes.len() || layer >= self.nodes[from].layers.len() {
            return;
        }
        let old_neighbors =
            std::mem::replace(&mut self.nodes[from].layers[layer], new_neighbors.clone());
        for &old in &old_neighbors {
            if !new_neighbors.contains(&old)
                && old < self.nodes.len()
                && layer < self.nodes[old].in_layers.len()
            {
                self.nodes[old].in_layers[layer].retain(|&x| x != from);
            }
        }
        for &new in &new_neighbors {
            if !old_neighbors.contains(&new)
                && new < self.nodes.len()
                && layer < self.nodes[new].in_layers.len()
                && !self.nodes[new].in_layers[layer].contains(&from)
            {
                self.nodes[new].in_layers[layer].push(from);
            }
        }
    }

    pub fn remove_node_edges(&mut self, slot_idx: usize) {
        if slot_idx >= self.nodes.len() {
            return;
        }
        let num_layers = self.nodes[slot_idx].layers.len();
        for l in 0..num_layers {
            let outgoing = std::mem::take(&mut self.nodes[slot_idx].layers[l]);
            for nb in outgoing {
                if nb < self.nodes.len() && l < self.nodes[nb].in_layers.len() {
                    self.nodes[nb].in_layers[l].retain(|&x| x != slot_idx);
                }
            }
            let incoming = std::mem::take(&mut self.nodes[slot_idx].in_layers[l]);
            for nb in incoming {
                if nb < self.nodes.len() && l < self.nodes[nb].layers.len() {
                    self.nodes[nb].layers[l].retain(|&x| x != slot_idx);
                }
            }
        }
        if self.entry_point == Some(slot_idx) {
            let mut best_entry = None;
            let mut best_layer = 0;
            for (idx, node) in self.nodes.iter().enumerate() {
                if idx != slot_idx && !node.layers.is_empty() && node.layers.len() > best_layer {
                    best_layer = node.layers.len();
                    best_entry = Some(idx);
                }
            }
            self.entry_point = best_entry;
            self.max_layer = best_layer.saturating_sub(1);
        }
    }

    pub fn insert(&mut self, slot_idx: usize, vector: &[f32], all_vectors: &[f32], dim: usize) {
        let node_level = self.random_level();
        let new_node = HnswNode {
            layers: vec![Vec::new(); node_level + 1],
            in_layers: vec![Vec::new(); node_level + 1],
        };

        while self.nodes.len() <= slot_idx {
            self.nodes.push(HnswNode::default());
        }

        self.nodes[slot_idx] = new_node;
        let Some(mut curr_obj) = self.entry_point else {
            self.entry_point = Some(slot_idx);
            self.max_layer = node_level;
            return;
        };

        let mut curr_dist = dot_product(vector, get_vector(all_vectors, curr_obj, dim));

        for layer in (node_level + 1..=self.max_layer).rev() {
            let mut changed = true;
            while changed {
                changed = false;
                if layer < self.nodes[curr_obj].layers.len() {
                    for &nb in &self.nodes[curr_obj].layers[layer] {
                        let d = dot_product(vector, get_vector(all_vectors, nb, dim));
                        if d > curr_dist {
                            curr_dist = d;
                            curr_obj = nb;
                            changed = true;
                        }
                    }
                }
            }
        }

        let min_layer = node_level.min(self.max_layer);

        for layer in (0..=min_layer).rev() {
            let total_nodes = self.nodes.len();

            let (cand_slots, next_entry) = VISITED_TRACKER.with(|tracker| {
                let (ref mut visited, ref mut cur_version) = *tracker.borrow_mut();
                *cur_version = cur_version.wrapping_add(1);
                if *cur_version == 0 {
                    visited.fill(0);
                    *cur_version = 1;
                }

                if visited.len() < total_nodes {
                    visited.resize(total_nodes, 0);
                }

                let v = *cur_version;
                visited[curr_obj] = v;
                let mut candidates = BinaryHeap::new();
                let mut w: BinaryHeap<std::cmp::Reverse<Candidate>> = BinaryHeap::new();
                candidates.push(Candidate {
                    score: curr_dist,
                    slot: curr_obj,
                });

                w.push(std::cmp::Reverse(Candidate {
                    score: curr_dist,
                    slot: curr_obj,
                }));

                while let Some(c) = candidates.pop() {
                    let worst_score = w.peek().unwrap().0.score;
                    if c.score < worst_score && w.len() >= self.config.ef_construction {
                        break;
                    }
                    if layer < self.nodes[c.slot].layers.len() {
                        for &nb in &self.nodes[c.slot].layers[layer] {
                            if visited[nb] != v {
                                visited[nb] = v;
                                let score = dot_product(vector, get_vector(all_vectors, nb, dim));
                                let worst_score = w.peek().unwrap().0.score;
                                if score >= worst_score || w.len() < self.config.ef_construction {
                                    candidates.push(Candidate { score, slot: nb });
                                    w.push(std::cmp::Reverse(Candidate { score, slot: nb }));
                                    if w.len() > self.config.ef_construction {
                                        w.pop();
                                    }
                                }
                            }
                        }
                    }
                }

                let slots: Vec<usize> = w
                    .into_iter()
                    .map(|rev| rev.0.slot)
                    .filter(|&s| s != slot_idx)
                    .collect();

                (slots, curr_obj)
            });

            curr_obj = next_entry;
            let max_edges = if layer == 0 {
                self.config.m0
            } else {
                self.config.m
            };

            let chosen_neighbors =
                select_neighbors(vector, &cand_slots, max_edges, all_vectors, dim);

            for &nb in &chosen_neighbors {
                if layer < self.nodes[nb].layers.len() {
                    let mut nb_neighbors = self.nodes[nb].layers[layer].clone();
                    if !nb_neighbors.contains(&slot_idx) {
                        nb_neighbors.push(slot_idx);
                    }
                    if nb_neighbors.len() > max_edges {
                        let nb_vec = get_vector(all_vectors, nb, dim);
                        let mut pruned =
                            select_neighbors(nb_vec, &nb_neighbors, max_edges, all_vectors, dim);
                        if !pruned.contains(&slot_idx) {
                            if pruned.len() >= max_edges {
                                pruned.pop();
                            }
                            pruned.push(slot_idx);
                        }
                        self.set_neighbors(nb, layer, pruned);
                    } else {
                        self.set_neighbors(nb, layer, nb_neighbors);
                    }
                }
            }

            self.set_neighbors(slot_idx, layer, chosen_neighbors);
            if let Some(&best) = self.nodes[slot_idx].layers[layer].first() {
                curr_obj = best;
                curr_dist = dot_product(vector, get_vector(all_vectors, curr_obj, dim));
            }
        }

        if node_level > self.max_layer {
            self.max_layer = node_level;
            self.entry_point = Some(slot_idx);
        }
    }

    pub fn remove(&mut self, slot_idx: usize, last_idx: usize) {
        if slot_idx >= self.nodes.len() {
            return;
        }

        let entry_was_deleted = self.entry_point == Some(slot_idx);
        let entry_was_last = self.entry_point == Some(last_idx);
        self.remove_node_edges(slot_idx);
        if slot_idx != last_idx && last_idx < self.nodes.len() {
            let num_layers = self.nodes[last_idx].layers.len();
            for l in 0..num_layers {
                let in_neighbors: Vec<usize> = self.nodes[last_idx].in_layers[l].clone();
                for nb in in_neighbors {
                    if nb < self.nodes.len() && l < self.nodes[nb].layers.len() {
                        for target in &mut self.nodes[nb].layers[l] {
                            if *target == last_idx {
                                *target = slot_idx;
                            }
                        }
                    }
                }

                let out_neighbors: Vec<usize> = self.nodes[last_idx].layers[l].clone();
                for nb in out_neighbors {
                    if nb < self.nodes.len() && l < self.nodes[nb].in_layers.len() {
                        for target in &mut self.nodes[nb].in_layers[l] {
                            if *target == last_idx {
                                *target = slot_idx;
                            }
                        }
                    }
                }
            }
            self.nodes.swap(slot_idx, last_idx);
        }

        self.nodes.pop();
        if self.nodes.is_empty() {
            self.entry_point = None;
            self.max_layer = 0;
        } else if entry_was_deleted {
            let mut best_entry = 0;
            let mut best_layer = 0;
            for (idx, node) in self.nodes.iter().enumerate() {
                if node.layers.len() > best_layer {
                    best_layer = node.layers.len();
                    best_entry = idx;
                }
            }

            if best_layer > 0 {
                self.entry_point = Some(best_entry);
                self.max_layer = best_layer.saturating_sub(1);
            } else {
                self.entry_point = None;
                self.max_layer = 0;
            }
        } else if entry_was_last {
            self.entry_point = Some(slot_idx);
        }
    }

    pub fn search(
        &self,
        query: &[f32],
        limit: usize,
        view: &VectorView<'_>,
        filter: Option<&SearchFilter>,
    ) -> Vec<(f32, usize)> {
        let Some(mut curr_obj) = self.entry_point else {
            return Vec::new();
        };

        let mut curr_dist = dot_product(query, get_vector(view.vectors, curr_obj, view.dim));
        for layer in (1..=self.max_layer).rev() {
            let mut changed = true;
            while changed {
                changed = false;
                if layer < self.nodes[curr_obj].layers.len() {
                    for &nb in &self.nodes[curr_obj].layers[layer] {
                        let d = dot_product(query, get_vector(view.vectors, nb, view.dim));
                        if d > curr_dist {
                            curr_dist = d;
                            curr_obj = nb;
                            changed = true;
                        }
                    }
                }
            }
        }

        let ef = self.config.ef_search.max(limit * 4);
        let min_score = filter.and_then(|f| f.min_score);
        let target_mod = filter.and_then(|f| f.modality.as_ref());
        let total_nodes = self.nodes.len();
        let matches_filter = |slot: usize, score: f32| -> bool {
            if target_mod.is_some_and(|m| &view.modalities[slot] != m) {
                return false;
            }
            if min_score.is_some_and(|min| score < min) {
                return false;
            }
            true
        };

        let mut results: BinaryHeap<std::cmp::Reverse<SearchCandidate>> =
            BinaryHeap::with_capacity(limit + 1);

        VISITED_TRACKER.with(|tracker| {
            let (ref mut visited, ref mut cur_version) = *tracker.borrow_mut();
            *cur_version = cur_version.wrapping_add(1);
            if *cur_version == 0 {
                visited.fill(0);
                *cur_version = 1;
            }
            if visited.len() < total_nodes {
                visited.resize(total_nodes, 0);
            }
            let v = *cur_version;
            visited[curr_obj] = v;
            let mut candidates = BinaryHeap::new();
            candidates.push(Candidate {
                score: curr_dist,
                slot: curr_obj,
            });
            let mut frontier: BinaryHeap<std::cmp::Reverse<Candidate>> =
                BinaryHeap::with_capacity(ef + 1);
            frontier.push(std::cmp::Reverse(Candidate {
                score: curr_dist,
                slot: curr_obj,
            }));
            if matches_filter(curr_obj, curr_dist) {
                results.push(std::cmp::Reverse(SearchCandidate {
                    score: curr_dist,
                    slot: curr_obj,
                    id: &view.ids[curr_obj],
                }));
            }

            while let Some(c) = candidates.pop() {
                if frontier.len() >= ef && frontier.peek().is_some_and(|w| c.score < w.0.score) {
                    break;
                }
                if !self.nodes[c.slot].layers.is_empty() {
                    for &nb in &self.nodes[c.slot].layers[0] {
                        if visited[nb] != v {
                            visited[nb] = v;
                            let score = dot_product(query, get_vector(view.vectors, nb, view.dim));
                            let worst_frontier_score = frontier
                                .peek()
                                .map(|x| x.0.score)
                                .unwrap_or(f32::NEG_INFINITY);
                            if score > worst_frontier_score || frontier.len() < ef {
                                candidates.push(Candidate { score, slot: nb });
                                frontier.push(std::cmp::Reverse(Candidate { score, slot: nb }));
                                if frontier.len() > ef {
                                    frontier.pop();
                                }
                            }
                            if matches_filter(nb, score) {
                                let cand = SearchCandidate {
                                    score,
                                    slot: nb,
                                    id: &view.ids[nb],
                                };
                                if results.len() < limit {
                                    results.push(std::cmp::Reverse(cand));
                                } else if results
                                    .peek()
                                    .is_some_and(|&worst| std::cmp::Reverse(cand) < worst)
                                {
                                    results.pop();
                                    results.push(std::cmp::Reverse(cand));
                                }
                            }
                        }
                    }
                }
            }
        });

        let mut filtered: Vec<(f32, usize)> = results
            .into_iter()
            .map(|rev| (rev.0.score, rev.0.slot))
            .collect();
        filtered.sort_unstable_by(|a, b| {
            b.0.total_cmp(&a.0)
                .then_with(|| view.ids[a.1].cmp(&view.ids[b.1]))
        });
        filtered
    }
}
