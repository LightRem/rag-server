use crate::domain::{DocumentId, Modality};
use serde_json::Value;
use std::{collections::HashMap, sync::Arc};

#[derive(Default)]
pub struct StorageState {
    pub index_map: HashMap<DocumentId, usize>,
    pub vectors: Vec<f32>,
    pub ids: Vec<Arc<str>>,
    pub modalities: Vec<Modality>,
    pub payloads: Vec<Arc<Value>>,
}

impl StorageState {
    #[inline]
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    pub fn upsert(
        &mut self,
        id: &DocumentId,
        vector: &[f32],
        payload: Arc<Value>,
        modality: Modality,
        dimension: usize,
    ) -> (usize, bool) {
        if let Some(&slot_idx) = self.index_map.get(id) {
            let start = slot_idx * dimension;
            self.vectors[start..start + dimension].copy_from_slice(vector);
            self.modalities[slot_idx] = modality;
            self.payloads[slot_idx] = payload;
            (slot_idx, false)
        } else {
            let slot_idx = self.ids.len();
            self.index_map.insert(id.clone(), slot_idx);
            self.vectors.extend_from_slice(vector);
            self.ids.push(Arc::from(id.as_str()));
            self.modalities.push(modality);
            self.payloads.push(payload);
            (slot_idx, true)
        }
    }

    pub fn remove(&mut self, id: &DocumentId, dimension: usize) -> Option<(usize, usize)> {
        let slot_idx = self.index_map.remove(id)?;
        let last_idx = self.ids.len() - 1;

        if slot_idx != last_idx {
            let src_start = last_idx * dimension;
            let dst_start = slot_idx * dimension;
            self.vectors
                .copy_within(src_start..src_start + dimension, dst_start);
            self.ids.swap_remove(slot_idx);
            self.modalities.swap_remove(slot_idx);
            self.payloads.swap_remove(slot_idx);

            let moved_id = self.ids[slot_idx].to_string();
            if let Some(mapped_slot) = self.index_map.get_mut(&moved_id) {
                *mapped_slot = slot_idx;
            }
        } else {
            self.ids.pop();
            self.modalities.pop();
            self.payloads.pop();
        }

        self.vectors.truncate(last_idx * dimension);
        Some((slot_idx, last_idx))
    }
}
