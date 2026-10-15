use crate::error::InferenceError;
use async_trait::async_trait;
use bytes::Bytes;
#[async_trait]
pub trait Embedder: Send + Sync {
    async fn embed_text(&self, text: &str) -> Result<Vec<f32>, InferenceError>;
    async fn embed_text_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, InferenceError>;
    async fn embed_image(&self, image_bytes: &[u8]) -> Result<Vec<f32>, InferenceError>;
    async fn embed_image_batch(&self, images: &[Bytes]) -> Result<Vec<Vec<f32>>, InferenceError>;
    fn dimension(&self) -> usize;
}
pub fn l2_normalize(vector: &mut [f32]) -> Result<(), InferenceError> {
    let sum_sq: f32 = vector.iter().map(|x| x * x).sum();
    let norm = sum_sq.sqrt();
    if !norm.is_finite() || norm <= f32::EPSILON {
        return Err(InferenceError::Execution(
            "Non-finite or zero-norm vector detected: normalization failed".into(),
        ));
    }
    let inv_norm = 1.0 / norm;
    for x in vector.iter_mut() {
        *x *= inv_norm;
    }
    Ok(())
}
