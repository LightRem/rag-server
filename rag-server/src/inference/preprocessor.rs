use crate::error::InferenceError;
use image::imageops::FilterType;
use ndarray::{Array2, Array4};
use rayon::prelude::*;
use tokenizers::Tokenizer;

pub const CLIP_IMAGE_SIZE: usize = 224;
pub const CLIP_TEXT_SEQ_LEN: usize = 77;
pub const CLIP_EOT_TOKEN_ID: i64 = 49407;
const CLIP_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
#[allow(clippy::excessive_precision)]
const CLIP_STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];

pub struct ImagePreprocessor;

impl ImagePreprocessor {
    pub fn preprocess(bytes: &[u8]) -> Result<Array4<f32>, InferenceError> {
        Self::preprocess_batch(&[bytes])
    }
    fn process_single_image(bytes: &[u8], out_slice: &mut [f32]) -> Result<(), InferenceError> {
        let img = image::load_from_memory(bytes).map_err(|e| InferenceError::ImageDecode {
            reason: e.to_string(),
            mime_type: None,
        })?;

        let processed = img.resize_to_fill(
            CLIP_IMAGE_SIZE as u32,
            CLIP_IMAGE_SIZE as u32,
            FilterType::CatmullRom,
        );

        let rgb_img = processed.to_rgb8();
        let raw_bytes = rgb_img.as_raw();
        let num_pixels = CLIP_IMAGE_SIZE * CLIP_IMAGE_SIZE;
        let (r_plane, rest) = out_slice.split_at_mut(num_pixels);
        let (g_plane, b_plane) = rest.split_at_mut(num_pixels);
        for (i, chunk) in raw_bytes.chunks_exact(3).enumerate() {
            r_plane[i] = ((chunk[0] as f32 / 255.0) - CLIP_MEAN[0]) / CLIP_STD[0];
            g_plane[i] = ((chunk[1] as f32 / 255.0) - CLIP_MEAN[1]) / CLIP_STD[1];
            b_plane[i] = ((chunk[2] as f32 / 255.0) - CLIP_MEAN[2]) / CLIP_STD[2];
        }
        Ok(())
    }

    pub fn preprocess_batch(images: &[&[u8]]) -> Result<Array4<f32>, InferenceError> {
        let batch_size = images.len();
        if batch_size == 0 {
            return Err(InferenceError::Execution(
                "Empty image batch provided".into(),
            ));
        }

        let num_pixels = CLIP_IMAGE_SIZE * CLIP_IMAGE_SIZE;
        let image_element_count = 3 * num_pixels;
        let mut flat_data = vec![0.0f32; batch_size * image_element_count];
        flat_data
            .par_chunks_exact_mut(image_element_count)
            .zip(images.par_iter())
            .try_for_each(|(slice, &image_bytes)| Self::process_single_image(image_bytes, slice))?;
        Array4::from_shape_vec((batch_size, 3, CLIP_IMAGE_SIZE, CLIP_IMAGE_SIZE), flat_data)
            .map_err(|e| InferenceError::Preprocessing(e.to_string()))
    }
}

pub struct TextPreprocessor;

impl TextPreprocessor {
    pub fn preprocess_single(
        tokenizer: &Tokenizer,
        text: &str,
    ) -> Result<Array2<i64>, InferenceError> {
        Self::preprocess_batch(tokenizer, &[text])
    }

    pub fn preprocess_batch(
        tokenizer: &Tokenizer,
        texts: &[&str],
    ) -> Result<Array2<i64>, InferenceError> {
        if texts.is_empty() {
            return Err(InferenceError::Execution(
                "Empty text batch provided".into(),
            ));
        }
        let encodings = tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| InferenceError::Preprocessing(e.to_string()))?;
        let batch_size = texts.len();
        let mut flat_data = vec![CLIP_EOT_TOKEN_ID; batch_size * CLIP_TEXT_SEQ_LEN];
        for (b, encoding) in encodings.iter().enumerate() {
            let ids = encoding.get_ids();
            let dest_offset = b * CLIP_TEXT_SEQ_LEN;
            let copy_len = ids.len().min(CLIP_TEXT_SEQ_LEN);
            for (i, &id) in ids[..copy_len].iter().enumerate() {
                flat_data[dest_offset + i] = id as i64;
            }
        }
        Array2::from_shape_vec((batch_size, CLIP_TEXT_SEQ_LEN), flat_data)
            .map_err(|e| InferenceError::Preprocessing(e.to_string()))
    }
}
