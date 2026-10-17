use super::models::IndexDocumentRequest;
use super::security::fetch_safe_remote_image;
use crate::{
    domain::{ImageSource, Modality, RawImageData},
    error::ApiError,
    state::AppState,
    storage::VectorPoint,
};
use bytes::Bytes;
use serde_json::{Value, json};
use std::path::Path;
use tokio::io::AsyncReadExt;

fn build_document_payload(
    raw_metadata: Option<Value>,
    title: Option<String>,
    text_content: Option<String>,
    image_url: Option<String>,
) -> Value {
    let mut payload = match raw_metadata {
        Some(Value::Object(map)) => Value::Object(map),
        Some(primitive) => json!({ "custom_metadata": primitive }),
        None => json!({}),
    };

    if let Some(obj) = payload.as_object_mut() {
        if let Some(text) = text_content {
            obj.insert("text".to_string(), json!(text));
        }
        if let Some(u) = image_url {
            obj.insert("url".to_string(), json!(u));
        }
        if let Some(t) = title {
            obj.insert("title".to_string(), json!(t));
        }
    }
    payload
}

fn extract_clip_caption(title: Option<&str>, text: &str) -> String {
    let first_sentence = text
        .split(['\n', '.', '!', '?'])
        .map(|s| s.trim())
        .find(|s| s.len() >= 10)
        .unwrap_or(text);

    match title {
        Some(t) if !t.trim().is_empty() => {
            let clean_t = t.trim();
            if first_sentence
                .to_lowercase()
                .starts_with(&clean_t.to_lowercase())
            {
                first_sentence.to_string()
            } else {
                format!("{clean_t}: {first_sentence}")
            }
        }
        _ => first_sentence.to_string(),
    }
}

fn ext_from_mime(mime: &str) -> &'static str {
    match mime {
        "image/png" => "png",
        "image/webp" => "webp",
        "image/gif" => "gif",
        _ => "jpg",
    }
}

async fn save_raw_image_to_disk(
    raw: &RawImageData,
    doc_id: &str,
    allowed_root: &Path,
) -> Result<String, ApiError> {
    tokio::fs::create_dir_all(allowed_root)
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to create uploads directory: {e}")))?;

    let ext = ext_from_mime(&raw.mime_type);
    let safe_id: String = doc_id
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let filename = format!("{safe_id}.{ext}");
    let file_path = allowed_root.join(&filename);

    tokio::fs::write(&file_path, &raw.bytes)
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to persist uploaded image: {e}")))?;

    Ok(format!("/uploads/{filename}"))
}

async fn resolve_image_url(
    image_source: &ImageSource,
    doc_id: &str,
    allowed_root: &Path,
) -> Result<Option<String>, ApiError> {
    match image_source {
        ImageSource::Url(u) => Ok(Some(u.clone())),
        ImageSource::Path(p) => Ok(Some(format!("file://{}", p.display()))),
        ImageSource::Raw(raw) => {
            let web_url = save_raw_image_to_disk(raw, doc_id, allowed_root).await?;
            Ok(Some(web_url))
        }
    }
}

async fn load_local_image_file(
    source: &ImageSource,
    path: &Path,
    allowed_root: &Path,
    max_allowed_bytes: usize,
) -> Result<Bytes, ApiError> {
    let safe_path = source
        .validate_safe_path(allowed_root)
        .map_err(ApiError::ValidationError)?;

    let mut file = tokio::fs::File::open(&safe_path).await.map_err(|e| {
        ApiError::ValidationError(format!("Failed to open image at '{path:?}': {e}"))
    })?;

    let metadata = file.metadata().await.map_err(|e| {
        ApiError::ValidationError(format!("Failed to read metadata for '{path:?}': {e}"))
    })?;

    if metadata.len() > max_allowed_bytes as u64 {
        return Err(ApiError::PayloadTooLarge {
            size: metadata.len() as usize,
            max_allowed: max_allowed_bytes,
        });
    }

    let mut buffer = Vec::with_capacity(metadata.len().min(max_allowed_bytes as u64) as usize);
    let mut handle = (&mut file).take((max_allowed_bytes + 1) as u64);
    handle.read_to_end(&mut buffer).await.map_err(|e| {
        ApiError::ValidationError(format!("Failed to read image at '{path:?}': {e}"))
    })?;

    if buffer.len() > max_allowed_bytes {
        return Err(ApiError::PayloadTooLarge {
            size: buffer.len(),
            max_allowed: max_allowed_bytes,
        });
    }

    Ok(Bytes::from(buffer))
}

pub async fn load_image_bytes(
    source: &ImageSource,
    allowed_root: &Path,
    max_allowed_bytes: usize,
) -> Result<Bytes, ApiError> {
    match source {
        ImageSource::Raw(raw) => Ok(raw.bytes.clone()),
        ImageSource::Path(path) => {
            load_local_image_file(source, path, allowed_root, max_allowed_bytes).await
        }
        ImageSource::Url(url) => fetch_safe_remote_image(url, max_allowed_bytes).await,
    }
}

pub async fn prepare_single_point(
    state: &AppState,
    doc: IndexDocumentRequest,
) -> Result<VectorPoint, ApiError> {
    let doc_id = doc.id.trim();
    if doc_id.is_empty() {
        return Err(ApiError::ValidationError(
            "Document id cannot be empty".into(),
        ));
    }

    let has_text = doc.text.as_ref().is_some_and(|t| !t.trim().is_empty());
    let has_image = doc.image.is_some();
    if !has_text && !has_image {
        return Err(ApiError::ValidationError(
            "Document must contain at least text or image".into(),
        ));
    }

    let allowed_root = state.uploads_dir();
    let max_image_bytes = state.config().server.max_body_limit_bytes();

    match (doc.text, doc.image) {
        (Some(text), Some(image_source)) => {
            let image_url = resolve_image_url(&image_source, doc_id, allowed_root).await?;
            let bytes = load_image_bytes(&image_source, allowed_root, max_image_bytes).await?;

            let clip_caption = extract_clip_caption(doc.title.as_deref(), &text);
            let text_vec = state.embedder().embed_text(&clip_caption).await?;
            let img_vec = state.embedder().embed_image(&bytes).await?;
            let payload = build_document_payload(doc.metadata, doc.title, Some(text), image_url);

            Ok(
                VectorPoint::new(doc_id, text_vec, payload, Modality::Multimodal)
                    .with_image_vector(img_vec),
            )
        }
        (Some(text), None) => {
            let clip_caption = extract_clip_caption(doc.title.as_deref(), &text);
            let vec = state.embedder().embed_text(&clip_caption).await?;
            let payload = build_document_payload(doc.metadata, doc.title, Some(text), None);
            Ok(VectorPoint::new(doc_id, vec, payload, Modality::Text))
        }
        (None, Some(image_source)) => {
            let image_url = resolve_image_url(&image_source, doc_id, allowed_root).await?;
            let bytes = load_image_bytes(&image_source, allowed_root, max_image_bytes).await?;

            let vec = state.embedder().embed_image(&bytes).await?;
            let fallback_text = doc.title.as_deref().unwrap_or(doc_id);
            let text_hint = format!("[Image: {fallback_text}]");
            let payload =
                build_document_payload(doc.metadata, doc.title, Some(text_hint), image_url);

            Ok(VectorPoint::new(doc_id, vec, payload, Modality::Image))
        }
        (None, None) => unreachable!(),
    }
}

pub async fn process_single_document(
    state: &AppState,
    doc: IndexDocumentRequest,
) -> Result<(), ApiError> {
    let point = prepare_single_point(state, doc).await?;
    state.vector_store().insert(point).await?;
    Ok(())
}
