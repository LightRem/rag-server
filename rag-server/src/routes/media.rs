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
use std::{path::Path, sync::Arc};
use tokio::io::AsyncReadExt;

pub struct PreparedBatch {
    pub points: Vec<VectorPoint>,
    pub failed_ids: Vec<String>,
    pub uploaded_files: Vec<String>,
}

struct PendingDoc {
    original_idx: usize,
    id: String,
    title: Option<String>,
    text: Option<String>,
    metadata: Option<Value>,
    image_source: Option<ImageSource>,
}

struct LoadedDoc {
    id: String,
    title: Option<String>,
    text: Option<String>,
    metadata: Option<Value>,
    image_source: Option<ImageSource>,
    image_bytes: Option<Bytes>,
    chunks: Vec<String>,
    modality: Modality,
}

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

fn chunk_text(title: Option<&str>, text: &str) -> Vec<String> {
    let clean_title = title.map(str::trim).filter(|t| !t.is_empty());
    let prefix = clean_title.map(|t| format!("{t}: "));

    let mut raw_units = Vec::new();
    let mut current = String::new();
    for c in text.chars() {
        current.push(c);
        if c == '.' || c == '!' || c == '?' || c == '\n' {
            let t = current.trim();
            if !t.is_empty() {
                raw_units.push(t.to_string());
            }
            current.clear();
        }
    }
    let t = current.trim();
    if !t.is_empty() {
        raw_units.push(t.to_string());
    }
    if raw_units.is_empty() {
        let trimmed = text.trim();
        if !trimmed.is_empty() {
            raw_units.push(trimmed.to_string());
        }
    }

    let mut units = Vec::new();
    for unit in raw_units {
        if unit.len() <= 280 {
            units.push(unit);
        } else {
            let chars: Vec<char> = unit.chars().collect();
            for chunk_slice in chars.chunks(280) {
                let sub: String = chunk_slice.iter().collect();
                let trimmed = sub.trim().to_string();
                if !trimmed.is_empty() {
                    units.push(trimmed);
                }
            }
        }
    }

    let mut chunks = Vec::new();
    let mut cur_chunk = String::new();
    for u in units {
        if cur_chunk.is_empty() {
            cur_chunk.push_str(&u);
        } else if cur_chunk.len() + 1 + u.len() <= 280 {
            cur_chunk.push(' ');
            cur_chunk.push_str(&u);
        } else {
            chunks.push(cur_chunk);
            cur_chunk = u;
        }
    }
    if !cur_chunk.is_empty() {
        chunks.push(cur_chunk);
    }

    if let Some(p) = prefix {
        chunks
            .into_iter()
            .map(|c| {
                if c.to_lowercase().starts_with(&p.to_lowercase()) {
                    c
                } else {
                    format!("{p}{c}")
                }
            })
            .collect()
    } else {
        chunks
    }
}

fn normalize_vector(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

async fn embed_document_text(
    state: &AppState,
    title: Option<&str>,
    text: &str,
) -> Result<Vec<f32>, ApiError> {
    let chunks = chunk_text(title, text);
    if chunks.is_empty() {
        let mut vec = state.embedder().embed_text(text).await?;
        normalize_vector(&mut vec);
        return Ok(vec);
    }
    if chunks.len() == 1 {
        let mut vec = state.embedder().embed_text(&chunks[0]).await?;
        normalize_vector(&mut vec);
        return Ok(vec);
    }

    let chunk_refs: Vec<&str> = chunks.iter().map(|s| s.as_str()).collect();
    let batch_vecs = state.embedder().embed_text_batch(&chunk_refs).await?;
    let mut pooled = Vec::new();
    for vec in batch_vecs {
        if pooled.is_empty() {
            pooled = vec;
        } else {
            for (p, v) in pooled.iter_mut().zip(vec.iter()) {
                *p += v;
            }
        }
    }
    normalize_vector(&mut pooled);
    Ok(pooled)
}

fn ext_from_mime(mime: &str) -> &'static str {
    match mime {
        "image/png" => "png",
        "image/webp" => "webp",
        "image/gif" => "gif",
        _ => "jpg",
    }
}

fn hash_doc_id(id: &str) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in id.as_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
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
    let filename = if doc_id
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
        && !doc_id.is_empty()
    {
        format!("{doc_id}.{ext}")
    } else {
        let id_hash = hash_doc_id(doc_id);
        let safe_id: String = doc_id
            .chars()
            .take(48)
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        format!("{safe_id}_{id_hash:016x}.{ext}")
    };
    let file_path = allowed_root.join(&filename);

    if let Err(e) = tokio::fs::write(&file_path, &raw.bytes).await {
        let _ = tokio::fs::remove_file(&file_path).await;
        return Err(ApiError::Internal(format!(
            "Failed to persist uploaded image: {e}"
        )));
    }

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
    let safe_path = source.validate_safe_path(allowed_root)?;

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

    let text = doc.text.filter(|t| !t.trim().is_empty());
    let has_text = text.is_some();
    let has_image = doc.image.is_some();
    if !has_text && !has_image {
        return Err(ApiError::ValidationError(
            "Document must contain at least text or image".into(),
        ));
    }

    let allowed_root = state.uploads_dir();
    let max_image_bytes = state.config().server.max_body_limit_bytes();

    match (text, doc.image) {
        (Some(text), Some(image_source)) => {
            let bytes = load_image_bytes(&image_source, allowed_root, max_image_bytes).await?;
            let img_vec = state.embedder().embed_image(&bytes).await?;
            let text_vec = embed_document_text(state, doc.title.as_deref(), &text).await?;
            let image_url = resolve_image_url(&image_source, doc_id, allowed_root).await?;
            let payload = build_document_payload(doc.metadata, doc.title, Some(text), image_url);

            Ok(
                VectorPoint::new(doc_id, text_vec, payload, Modality::Multimodal)
                    .with_image_vector(img_vec),
            )
        }
        (Some(text), None) => {
            let vec = embed_document_text(state, doc.title.as_deref(), &text).await?;
            let payload = build_document_payload(doc.metadata, doc.title, Some(text), None);
            Ok(VectorPoint::new(doc_id, vec, payload, Modality::Text))
        }
        (None, Some(image_source)) => {
            let bytes = load_image_bytes(&image_source, allowed_root, max_image_bytes).await?;
            let vec = state.embedder().embed_image(&bytes).await?;
            let image_url = resolve_image_url(&image_source, doc_id, allowed_root).await?;
            let fallback_text = doc.title.as_deref().unwrap_or(doc_id);
            let text_hint = format!("[Image: {fallback_text}]");
            let payload =
                build_document_payload(doc.metadata, doc.title, Some(text_hint), image_url);

            Ok(VectorPoint::new(doc_id, vec, payload, Modality::Image))
        }
        (None, None) => unreachable!(),
    }
}

pub async fn prepare_batch_points(
    state: &AppState,
    docs: Vec<IndexDocumentRequest>,
) -> Result<PreparedBatch, ApiError> {
    let total_docs = docs.len();
    let doc_ids: Vec<String> = docs.iter().map(|d| d.id.clone()).collect();
    let mut completed = vec![false; total_docs];
    let mut failed_ids = Vec::new();
    let mut pending = Vec::with_capacity(total_docs);

    for (idx, doc) in docs.into_iter().enumerate() {
        let doc_id = doc.id.trim().to_string();
        if doc_id.is_empty() {
            completed[idx] = true;
            failed_ids.push(doc.id);
            continue;
        }
        let text = doc.text.filter(|t| !t.trim().is_empty());
        let has_text = text.is_some();
        let has_image = doc.image.is_some();
        if !has_text && !has_image {
            completed[idx] = true;
            failed_ids.push(doc_id);
            continue;
        }
        pending.push(PendingDoc {
            original_idx: idx,
            id: doc_id,
            title: doc.title,
            text,
            metadata: doc.metadata,
            image_source: doc.image,
        });
    }

    let max_concurrency = state.config().model.max_concurrency.max(1) * 2;
    let semaphore = Arc::new(tokio::sync::Semaphore::new(max_concurrency));
    let allowed_root = state.uploads_dir();
    let max_image_bytes = state.config().server.max_body_limit_bytes();
    let mut set = tokio::task::JoinSet::new();

    for p in pending {
        let sem = semaphore.clone();
        let root = allowed_root.to_path_buf();
        set.spawn(async move {
            let _permit = match sem.acquire().await {
                Ok(permit) => permit,
                Err(e) => {
                    return (
                        p.original_idx,
                        p.id,
                        Err(ApiError::Internal(format!(
                            "Concurrency semaphore closed: {e}"
                        ))),
                    );
                }
            };

            let image_bytes = if let Some(ref src) = p.image_source {
                match load_image_bytes(src, &root, max_image_bytes).await {
                    Ok(b) => Some(b),
                    Err(e) => return (p.original_idx, p.id, Err(e)),
                }
            } else {
                None
            };

            let (chunks, has_text) = if let Some(ref text) = p.text {
                (chunk_text(p.title.as_deref(), text), true)
            } else {
                (Vec::new(), false)
            };

            let modality = match (has_text, p.image_source.is_some()) {
                (true, true) => Modality::Multimodal,
                (true, false) => Modality::Text,
                (false, true) => Modality::Image,
                (false, false) => unreachable!(),
            };

            (
                p.original_idx,
                p.id.clone(),
                Ok(LoadedDoc {
                    id: p.id,
                    title: p.title,
                    text: p.text,
                    metadata: p.metadata,
                    image_source: p.image_source,
                    image_bytes,
                    chunks,
                    modality,
                }),
            )
        });
    }

    let mut loaded_docs = Vec::new();
    while let Some(join_res) = set.join_next().await {
        match join_res {
            Ok((orig_idx, _id, Ok(loaded))) => {
                completed[orig_idx] = true;
                loaded_docs.push(loaded);
            }
            Ok((orig_idx, id, Err(e))) => {
                completed[orig_idx] = true;
                tracing::warn!(id = %id, error = %e, "Failed to load document media in batch");
                failed_ids.push(id);
            }
            Err(e) => {
                tracing::error!(error = %e, "Batch worker task panicked or failed");
            }
        }
    }

    for (id, is_done) in doc_ids.into_iter().zip(completed) {
        if !is_done && !failed_ids.contains(&id) {
            failed_ids.push(id);
        }
    }

    let mut chunk_map = Vec::new();
    for (doc_i, doc) in loaded_docs.iter().enumerate() {
        for chunk in &doc.chunks {
            chunk_map.push((doc_i, chunk.clone()));
        }
    }

    let mut doc_text_vecs: Vec<Option<Vec<f32>>> = vec![None; loaded_docs.len()];
    if !chunk_map.is_empty() {
        let chunk_refs: Vec<&str> = chunk_map.iter().map(|(_, c)| c.as_str()).collect();
        let embeddings = state.embedder().embed_text_batch(&chunk_refs).await?;
        let mut pooled_by_doc: Vec<Vec<f32>> = vec![Vec::new(); loaded_docs.len()];
        for ((doc_i, _), vec) in chunk_map.into_iter().zip(embeddings) {
            if pooled_by_doc[doc_i].is_empty() {
                pooled_by_doc[doc_i] = vec;
            } else {
                for (p, v) in pooled_by_doc[doc_i].iter_mut().zip(vec.iter()) {
                    *p += v;
                }
            }
        }
        for (doc_i, mut pooled) in pooled_by_doc.into_iter().enumerate() {
            if !pooled.is_empty() {
                normalize_vector(&mut pooled);
                doc_text_vecs[doc_i] = Some(pooled);
            }
        }
    }

    let mut img_map = Vec::new();
    let mut img_bytes_list: Vec<Bytes> = Vec::new();
    for (doc_i, doc) in loaded_docs.iter().enumerate() {
        if let Some(ref bytes) = doc.image_bytes {
            img_map.push(doc_i);
            img_bytes_list.push(bytes.clone());
        }
    }

    let mut doc_img_vecs: Vec<Option<Vec<f32>>> = vec![None; loaded_docs.len()];
    if !img_bytes_list.is_empty() {
        let embeddings = state.embedder().embed_image_batch(&img_bytes_list).await?;
        for (doc_i, vec) in img_map.into_iter().zip(embeddings) {
            doc_img_vecs[doc_i] = Some(vec);
        }
    }

    let mut points = Vec::new();
    let mut uploaded_files = Vec::new();

    for (i, doc) in loaded_docs.into_iter().enumerate() {
        let text_vec = doc_text_vecs[i].take();
        let img_vec = doc_img_vecs[i].take();

        let image_url = if let Some(ref src) = doc.image_source {
            match resolve_image_url(src, &doc.id, allowed_root).await {
                Ok(url) => {
                    if let Some(ref u) = url
                        && let Some(filename) = u.strip_prefix("/uploads/")
                    {
                        uploaded_files.push(filename.to_string());
                    }
                    url
                }
                Err(e) => {
                    tracing::warn!(id = %doc.id, error = %e, "Failed to resolve image URL in batch");
                    failed_ids.push(doc.id);
                    continue;
                }
            }
        } else {
            None
        };

        let payload = match doc.modality {
            Modality::Multimodal => {
                build_document_payload(doc.metadata, doc.title, doc.text, image_url)
            }
            Modality::Text => build_document_payload(doc.metadata, doc.title, doc.text, None),
            Modality::Image => {
                let fallback_text = doc.title.as_deref().unwrap_or(&doc.id);
                let text_hint = format!("[Image: {fallback_text}]");
                build_document_payload(doc.metadata, doc.title, Some(text_hint), image_url)
            }
        };

        match doc.modality {
            Modality::Multimodal => {
                if let (Some(tv), Some(iv)) = (text_vec, img_vec) {
                    points.push(
                        VectorPoint::new(doc.id, tv, payload, Modality::Multimodal)
                            .with_image_vector(iv),
                    );
                } else {
                    failed_ids.push(doc.id);
                }
            }
            Modality::Text => {
                if let Some(tv) = text_vec {
                    points.push(VectorPoint::new(doc.id, tv, payload, Modality::Text));
                } else {
                    failed_ids.push(doc.id);
                }
            }
            Modality::Image => {
                if let Some(iv) = img_vec {
                    points.push(VectorPoint::new(doc.id, iv, payload, Modality::Image));
                } else {
                    failed_ids.push(doc.id);
                }
            }
        }
    }

    Ok(PreparedBatch {
        points,
        failed_ids,
        uploaded_files,
    })
}

pub async fn process_single_document(
    state: &AppState,
    doc: IndexDocumentRequest,
) -> Result<(), ApiError> {
    let point = prepare_single_point(state, doc).await?;
    let uploaded_filename = point
        .payload
        .get("url")
        .and_then(|u| u.as_str())
        .and_then(|u| u.strip_prefix("/uploads/"))
        .map(|f| f.to_string());

    if let Err(e) = state.vector_store().insert(point).await {
        if let Some(filename) = uploaded_filename {
            let file_path = state.uploads_dir().join(filename);
            let _ = tokio::fs::remove_file(&file_path).await;
        }
        return Err(e.into());
    }
    Ok(())
}
