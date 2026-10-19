use super::{
    pool::{SessionGuard, SessionPool},
    preprocessor::{
        CLIP_EOT_TOKEN_ID, CLIP_IMAGE_SIZE, CLIP_TEXT_SEQ_LEN, ImagePreprocessor, TextPreprocessor,
    },
    traits::{Embedder, l2_normalize},
};
use crate::error::InferenceError;
use async_trait::async_trait;
use bytes::Bytes;
use ort::{environment::Environment, session::SessionOutputs, value::TensorRef};
use std::{path::Path, sync::Arc};
use tokenizers::{PaddingStrategy, Tokenizer, TruncationDirection, TruncationStrategy};

const TEXT_PRIORITY_NAMES: &[&str] = &[
    "text_embeds",
    "embeddings",
    "output",
    "pooler_output",
    "last_hidden_state_pooled",
];

const VISION_PRIORITY_NAMES: &[&str] = &[
    "image_embeds",
    "embeddings",
    "output",
    "pooler_output",
    "last_hidden_state_pooled",
];

pub struct OnnxClipEmbedder {
    _environment: Arc<Environment>,
    text_pool: SessionPool,
    vision_pool: SessionPool,
    tokenizer: Arc<Tokenizer>,
    dimension: usize,
}

impl OnnxClipEmbedder {
    pub fn new(
        text_model_path: &Path,
        vision_model_path: &Path,
        tokenizer_path: &Path,
        max_concurrency: usize,
        dimension: usize,
    ) -> Result<Self, InferenceError> {
        if !text_model_path.exists() {
            return Err(InferenceError::ModelNotFound {
                path: text_model_path.to_path_buf(),
            });
        }

        if !vision_model_path.exists() {
            return Err(InferenceError::ModelNotFound {
                path: vision_model_path.to_path_buf(),
            });
        }

        let environment = Arc::new(
            ort::init()
                .with_name("multimodal-rag")
                .build()
                .map_err(|e| InferenceError::ModelLoad(e.to_string()))?,
        );

        let text_pool = SessionPool::new(&environment, text_model_path, max_concurrency)?;
        let vision_pool = SessionPool::new(&environment, vision_model_path, max_concurrency)?;

        let mut tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| InferenceError::Preprocessing(format!("Failed to load tokenizer: {e}")))?;

        let mut padding_params = tokenizer.get_padding().cloned().unwrap_or_default();
        padding_params.strategy = PaddingStrategy::Fixed(CLIP_TEXT_SEQ_LEN);
        padding_params.pad_id = CLIP_EOT_TOKEN_ID as u32;
        padding_params.pad_token = "<|endoftext|>".to_string();

        let mut truncation_params = tokenizer.get_truncation().cloned().unwrap_or_default();
        truncation_params.max_length = CLIP_TEXT_SEQ_LEN;
        truncation_params.strategy = TruncationStrategy::LongestFirst;
        truncation_params.direction = TruncationDirection::Right;

        tokenizer
            .with_padding(Some(padding_params))
            .with_truncation(Some(truncation_params))
            .map_err(|e| {
                InferenceError::Preprocessing(format!("Failed to configure tokenizer: {e}"))
            })?;

        Ok(Self {
            _environment: environment,
            text_pool,
            vision_pool,
            tokenizer: Arc::new(tokenizer),
            dimension,
        })
    }

    pub fn into_shared(self) -> Arc<dyn Embedder> {
        Arc::new(self)
    }
}

fn extract_raw_embeddings(
    outputs: SessionOutputs<'_>,
    priority_names: &[&str],
    expected_count: usize,
    expected_dim: usize,
) -> Result<Vec<f32>, InferenceError> {
    let outputs_vec: Vec<_> = outputs.into_iter().collect();
    let mut best_priority_idx = usize::MAX;
    let mut best_match_idx = None;
    let mut fallback_match_idx = None;

    for (idx, (name, tensor)) in outputs_vec.iter().enumerate() {
        let name_str: &str = name;

        if let Ok((shape, _)) = tensor.try_extract_tensor::<f32>() {
            let shape_ref: &[i64] = shape.as_ref();
            let matches_shape = (shape_ref.len() == 2
                && shape_ref[0] == expected_count as i64
                && shape_ref[1] == expected_dim as i64)
                || (shape_ref.len() == 1 && shape_ref[0] == (expected_count * expected_dim) as i64);

            if matches_shape {
                if let Some(pos) = priority_names.iter().position(|&p| p == name_str) {
                    if pos < best_priority_idx {
                        best_priority_idx = pos;
                        best_match_idx = Some(idx);
                        if pos == 0 {
                            break;
                        }
                    }
                } else if fallback_match_idx.is_none() {
                    fallback_match_idx = Some(idx);
                }
            }
        }
    }

    let target_idx = best_match_idx.or(fallback_match_idx).ok_or_else(|| {
        let available: Vec<String> = outputs_vec.iter().map(|(k, _)| k.to_string()).collect();
        InferenceError::Execution(format!(
            "Target embedding tensor [{}, {}] not found. Available output keys: {:?}",
            expected_count, expected_dim, available
        ))
    })?;

    let (_shape, data) = outputs_vec[target_idx]
        .1
        .try_extract_tensor::<f32>()
        .map_err(|e| InferenceError::Execution(e.to_string()))?;

    let expected_total_elements = expected_count * expected_dim;
    if data.len() != expected_total_elements {
        return Err(InferenceError::TensorShapeMismatch {
            expected: vec![expected_count, expected_dim],
            actual: vec![
                data.len() / expected_dim.max(1),
                data.len() % expected_dim.max(1),
            ],
        });
    }

    // Copy out while the session is still borrowed by `outputs`.
    Ok(data.to_vec())
}

/// Splits a flat `[count, dim]` buffer into L2-normalized rows.
/// Called after the session is back in the pool, so a failure here never reloads the model.
fn normalize_rows(data: Vec<f32>, expected_dim: usize) -> Result<Vec<Vec<f32>>, InferenceError> {
    data.chunks_exact(expected_dim)
        .map(|chunk| {
            let mut vec = chunk.to_vec();
            l2_normalize(&mut vec)?;
            Ok(vec)
        })
        .collect::<Result<Vec<_>, InferenceError>>()
}

#[async_trait]
impl Embedder for OnnxClipEmbedder {
    async fn embed_text(&self, text: &str) -> Result<Vec<f32>, InferenceError> {
        let mut results = self.embed_text_batch(&[text]).await?;
        results
            .pop()
            .ok_or_else(|| InferenceError::Execution("Empty inference result returned".into()))
    }

    async fn embed_text_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, InferenceError> {
        let input_tensor = TextPreprocessor::preprocess_batch(&self.tokenizer, texts)?;
        let permit = self.text_pool.acquire_permit().await?;
        let pool = self.text_pool.clone();
        let expected_dim = self.dimension;
        let expected_count = texts.len();

        tokio::task::spawn_blocking(move || {
            let mut input_tensor = input_tensor;
            let session = pool.pop_session()?;
            let mut guard = SessionGuard::new(pool, session, permit);

            let shape = [expected_count, CLIP_TEXT_SEQ_LEN];
            let slice = input_tensor.as_slice_mut().ok_or_else(|| {
                InferenceError::Execution("Text tensor buffer is not contiguous".into())
            })?;

            let eot_id: i64 = CLIP_EOT_TOKEN_ID;
            for chunk in slice.chunks_exact_mut(CLIP_TEXT_SEQ_LEN) {
                if !chunk.contains(&eot_id) {
                    chunk[CLIP_TEXT_SEQ_LEN - 1] = eot_id;
                }
            }

            let input_ref = TensorRef::from_array_view((shape, &*slice))
                .map_err(|e| InferenceError::Execution(e.to_string()))?;

            let outputs = guard
                .session_mut()
                .run(ort::inputs![input_ref])
                .map_err(|e| InferenceError::Execution(e.to_string()))?;

            // `run` succeeded, so the session is healthy even if output parsing fails.
            // Return it to the pool first; only a failed `run` should trigger a model reload.
            let raw =
                extract_raw_embeddings(outputs, TEXT_PRIORITY_NAMES, expected_count, expected_dim);
            guard.success();
            normalize_rows(raw?, expected_dim)
        })
        .await
        .map_err(|e| InferenceError::Execution(e.to_string()))?
    }

    async fn embed_image(&self, image_bytes: &[u8]) -> Result<Vec<f32>, InferenceError> {
        let bytes = Bytes::copy_from_slice(image_bytes);
        let mut results = self.embed_image_batch(&[bytes]).await?;
        results
            .pop()
            .ok_or_else(|| InferenceError::Execution("Empty inference result returned".into()))
    }

    async fn embed_image_batch(&self, images: &[Bytes]) -> Result<Vec<Vec<f32>>, InferenceError> {
        let permit = self.vision_pool.acquire_permit().await?;
        let pool = self.vision_pool.clone();
        let expected_dim = self.dimension;
        let expected_count = images.len();
        let owned_images: Vec<Bytes> = images.to_vec();

        tokio::task::spawn_blocking(move || {
            let image_slices: Vec<&[u8]> = owned_images.iter().map(|b| b.as_ref()).collect();
            let input_tensor = ImagePreprocessor::preprocess_batch(&image_slices)?;

            let session = pool.pop_session()?;
            let mut guard = SessionGuard::new(pool, session, permit);

            let shape = [expected_count, 3, CLIP_IMAGE_SIZE, CLIP_IMAGE_SIZE];
            let slice = input_tensor.as_slice().ok_or_else(|| {
                InferenceError::Execution("Image tensor buffer is not contiguous".into())
            })?;
            let input_ref = TensorRef::from_array_view((shape, slice))
                .map_err(|e| InferenceError::Execution(e.to_string()))?;

            let outputs = guard
                .session_mut()
                .run(ort::inputs![input_ref])
                .map_err(|e| InferenceError::Execution(e.to_string()))?;

            // `run` succeeded, so the session is healthy even if output parsing fails.
            // Return it to the pool first; only a failed `run` should trigger a model reload.
            let raw = extract_raw_embeddings(
                outputs,
                VISION_PRIORITY_NAMES,
                expected_count,
                expected_dim,
            );
            guard.success();
            normalize_rows(raw?, expected_dim)
        })
        .await
        .map_err(|e| InferenceError::Execution(e.to_string()))?
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}
