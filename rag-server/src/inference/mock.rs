use super::traits::{Embedder, l2_normalize};
use crate::error::InferenceError;
use async_trait::async_trait;
use bytes::Bytes;
use std::hash::{DefaultHasher, Hasher};
pub struct MockEmbedder {
    dimension: usize,
}

impl MockEmbedder {
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }

    fn hash_bytes(data: &[u8]) -> u64 {
        let mut hasher = DefaultHasher::new();
        hasher.write(data);
        hasher.finish()
    }

    fn generate_vector_from_hash(&self, hash: u64) -> Vec<f32> {
        let mut vec = Vec::with_capacity(self.dimension);
        for i in 0..self.dimension {
            let val = ((hash.wrapping_mul((i + 1) as u64) % 1000) as f32 / 1000.0) + 0.1;
            vec.push(val);
        }
        let _ = l2_normalize(&mut vec);
        vec
    }
}

#[async_trait]
impl Embedder for MockEmbedder {
    async fn embed_text(&self, text: &str) -> Result<Vec<f32>, InferenceError> {
        let hash = Self::hash_bytes(text.as_bytes());
        Ok(self.generate_vector_from_hash(hash))
    }

    async fn embed_text_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, InferenceError> {
        Ok(texts
            .iter()
            .map(|t| {
                let hash = Self::hash_bytes(t.as_bytes());
                self.generate_vector_from_hash(hash)
            })
            .collect())
    }

    async fn embed_image(&self, image_bytes: &[u8]) -> Result<Vec<f32>, InferenceError> {
        let hash = Self::hash_bytes(image_bytes);
        Ok(self.generate_vector_from_hash(hash))
    }

    async fn embed_image_batch(&self, images: &[Bytes]) -> Result<Vec<Vec<f32>>, InferenceError> {
        Ok(images
            .iter()
            .map(|b| {
                let hash = Self::hash_bytes(b.as_ref());
                self.generate_vector_from_hash(hash)
            })
            .collect())
    }
    fn dimension(&self) -> usize {
        self.dimension
    }
}
