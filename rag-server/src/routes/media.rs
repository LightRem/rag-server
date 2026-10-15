use super::models::IndexDocumentRequest;
use crate::{
    domain::{ImageSource, Modality},
    error::ApiError,
    state::AppState,
    storage::VectorPoint,
};
use bytes::Bytes;
use serde_json::json;
use std::net::{IpAddr, SocketAddr};
use std::path::Path;
use tokio::io::AsyncReadExt;
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
    let max_image_bytes = state.config().server.max_body_limit_mb * 1024 * 1024;
    match (doc.text, doc.image) {
        (Some(text), Some(image_source)) => {
            let image_url = match &image_source {
                ImageSource::Url(u) => Some(u.clone()),
                ImageSource::Path(p) => Some(format!("file://{}", p.display())),
                ImageSource::Raw(_) => Some(format!("embedded://image/{}", doc_id)),
            };
            let bytes = load_image_bytes(
                &image_source,
                allowed_root,
                max_image_bytes,
                state.http_client(),
            )
            .await?;
            let text_vec = state.embedder().embed_text(&text).await?;
            let img_vec = state.embedder().embed_image(&bytes).await?;
            let mut payload = match doc.metadata {
                Some(serde_json::Value::Object(map)) => serde_json::Value::Object(map),
                Some(primitive) => json!({ "custom_metadata": primitive }),
                None => json!({}),
            };
            let obj = payload
                .as_object_mut()
                .expect("Payload is guaranteed to be an Object");
            obj.insert("text".to_string(), json!(text));
            if let Some(u) = image_url {
                obj.insert("url".to_string(), json!(u));
            }
            if let Some(title) = doc.title {
                obj.insert("title".to_string(), json!(title));
            }
            Ok(
                VectorPoint::new(doc_id, text_vec, payload, Modality::Multimodal)
                    .with_image_vector(img_vec),
            )
        }
        (Some(text), None) => {
            let vec = state.embedder().embed_text(&text).await?;
            let mut payload = match doc.metadata {
                Some(serde_json::Value::Object(map)) => serde_json::Value::Object(map),
                Some(primitive) => json!({ "custom_metadata": primitive }),
                None => json!({}),
            };
            let obj = payload
                .as_object_mut()
                .expect("Payload is guaranteed to be an Object");
            obj.insert("text".to_string(), json!(text));
            if let Some(title) = doc.title {
                obj.insert("title".to_string(), json!(title));
            }
            Ok(VectorPoint::new(doc_id, vec, payload, Modality::Text))
        }
        (None, Some(image_source)) => {
            let image_url = match &image_source {
                ImageSource::Url(u) => Some(u.clone()),
                ImageSource::Path(p) => Some(format!("file://{}", p.display())),
                ImageSource::Raw(_) => Some(format!("embedded://image/{}", doc_id)),
            };
            let bytes = load_image_bytes(
                &image_source,
                allowed_root,
                max_image_bytes,
                state.http_client(),
            )
            .await?;
            let vec = state.embedder().embed_image(&bytes).await?;
            let mut payload = match doc.metadata {
                Some(serde_json::Value::Object(map)) => serde_json::Value::Object(map),
                Some(primitive) => json!({ "custom_metadata": primitive }),
                None => json!({}),
            };
            let obj = payload
                .as_object_mut()
                .expect("Payload is guaranteed to be an Object");
            let fallback_text = doc.title.as_deref().unwrap_or(doc_id);
            obj.insert(
                "text".to_string(),
                json!(format!("[Image: {fallback_text}]")),
            );
            if let Some(u) = image_url {
                obj.insert("url".to_string(), json!(u));
            }
            if let Some(title) = doc.title {
                obj.insert("title".to_string(), json!(title));
            }
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
pub async fn load_image_bytes(
    source: &ImageSource,
    allowed_root: &Path,
    max_allowed_bytes: usize,
    _http_client: &reqwest::Client,
) -> Result<Bytes, ApiError> {
    match source {
        ImageSource::Raw(raw) => Ok(raw.bytes.clone()),
        ImageSource::Path(path) => {
            let safe_path = source
                .validate_safe_path(allowed_root)
                .map_err(ApiError::ValidationError)?;
            let mut file = tokio::fs::File::open(&safe_path).await.map_err(|e| {
                ApiError::ValidationError(format!("Failed to open image at '{:?}': {e}", path))
            })?;
            let metadata = file.metadata().await.map_err(|e| {
                ApiError::ValidationError(format!("Failed to read metadata for '{:?}': {e}", path))
            })?;
            if metadata.len() > max_allowed_bytes as u64 {
                return Err(ApiError::PayloadTooLarge {
                    size: metadata.len() as usize,
                    max_allowed: max_allowed_bytes,
                });
            }
            let mut buffer =
                Vec::with_capacity(metadata.len().min(max_allowed_bytes as u64) as usize);
            let mut handle = (&mut file).take((max_allowed_bytes + 1) as u64);
            handle.read_to_end(&mut buffer).await.map_err(|e| {
                ApiError::ValidationError(format!("Failed to read image at '{:?}': {e}", path))
            })?;
            if buffer.len() > max_allowed_bytes {
                return Err(ApiError::PayloadTooLarge {
                    size: buffer.len(),
                    max_allowed: max_allowed_bytes,
                });
            }
            Ok(Bytes::from(buffer))
        }
        ImageSource::Url(url) => {
            let mut current_url = url.clone();
            let mut hops = 0;
            const MAX_HOPS: usize = 5;
            let mut cached_client: Option<(String, SocketAddr, reqwest::Client)> = None;
            let (resp, content_length) = loop {
                let (parsed_url, host_str, safe_ip, port) =
                    resolve_and_validate_safe_url(&current_url).await?;
                let target_addr = SocketAddr::new(safe_ip, port);
                let client = match &cached_client {
                    Some((h, addr, c)) if h == &host_str && addr == &target_addr => c.clone(),
                    _ => {
                        let new_c = reqwest::Client::builder()
                            .resolve(&host_str, target_addr)
                            .redirect(reqwest::redirect::Policy::none())
                            .timeout(std::time::Duration::from_secs(15))
                            .build()
                            .map_err(|e| {
                                ApiError::Internal(format!(
                                    "Failed to build secure HTTP client: {e}"
                                ))
                            })?;
                        cached_client = Some((host_str.clone(), target_addr, new_c.clone()));
                        new_c
                    }
                };
                let response = client.get(parsed_url.clone()).send().await.map_err(|e| {
                    ApiError::ValidationError(format!("Failed to fetch image URL: {e}"))
                })?;
                if response.status().is_redirection() {
                    hops += 1;
                    if hops > MAX_HOPS {
                        return Err(ApiError::ValidationError(
                            "Too many redirects (max 5)".into(),
                        ));
                    }
                    let loc_header = response
                        .headers()
                        .get(reqwest::header::LOCATION)
                        .ok_or_else(|| {
                            ApiError::ValidationError(
                                "Redirect response missing Location header".into(),
                            )
                        })?;
                    let loc_str = loc_header.to_str().map_err(|_| {
                        ApiError::ValidationError("Invalid Location header encoding".into())
                    })?;
                    let next_url = parsed_url.join(loc_str).map_err(|e| {
                        ApiError::ValidationError(format!(
                            "Invalid redirect target '{loc_str}': {e}"
                        ))
                    })?;
                    current_url = next_url.to_string();
                    continue;
                }
                if !response.status().is_success() {
                    return Err(ApiError::ValidationError(format!(
                        "Image URL returned HTTP {}",
                        response.status()
                    )));
                }
                let cl = response.content_length();
                break (response, cl);
            };
            if let Some(cl) = content_length.filter(|&cl| cl > max_allowed_bytes as u64) {
                return Err(ApiError::PayloadTooLarge {
                    size: cl as usize,
                    max_allowed: max_allowed_bytes,
                });
            }
            let mut buffer = Vec::new();
            let mut stream = resp;
            while let Some(chunk) = stream.chunk().await.map_err(|e| {
                ApiError::ValidationError(format!("Failed to read image stream: {e}"))
            })? {
                if buffer.len() + chunk.len() > max_allowed_bytes {
                    return Err(ApiError::PayloadTooLarge {
                        size: buffer.len() + chunk.len(),
                        max_allowed: max_allowed_bytes,
                    });
                }
                buffer.extend_from_slice(&chunk);
            }
            Ok(Bytes::from(buffer))
        }
    }
}
fn is_allowed_port(port: u16) -> bool {
    matches!(port, 80 | 443 | 8080 | 8443)
}
fn is_forbidden_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ipv4) => {
            let octets = ipv4.octets();
            ipv4.is_loopback()
                || ipv4.is_unspecified()
                || (octets[0] == 0)
                || (octets[0] == 10)
                || (octets[0] == 100 && (64..=127).contains(&octets[1]))
                || (octets[0] == 169 && octets[1] == 254)
                || (octets[0] == 172 && (16..=31).contains(&octets[1]))
                || (octets[0] == 192 && octets[1] == 0 && octets[2] == 0)
                || (octets[0] == 192 && octets[1] == 0 && octets[2] == 2)
                || (octets[0] == 192 && octets[1] == 168)
                || (octets[0] == 198 && (18..=19).contains(&octets[1]))
                || (octets[0] == 198 && octets[1] == 51 && octets[2] == 100)
                || (octets[0] == 203 && octets[1] == 0 && octets[2] == 113)
                || (octets[0] >= 224)
        }
        IpAddr::V6(ipv6) => {
            let segs = ipv6.segments();
            if segs[0] == 0
                && segs[1] == 0
                && segs[2] == 0
                && segs[3] == 0
                && segs[4] == 0
                && (segs[5] == 0 || segs[5] == 0xffff)
            {
                let mapped_ipv4 = std::net::Ipv4Addr::new(
                    (segs[6] >> 8) as u8,
                    (segs[6] & 0xff) as u8,
                    (segs[7] >> 8) as u8,
                    (segs[7] & 0xff) as u8,
                );
                return is_forbidden_ip(IpAddr::V4(mapped_ipv4));
            }
            ipv6.is_loopback()
                || ipv6.is_unspecified()
                || (segs[0] & 0xfe00) == 0xfc00
                || (segs[0] & 0xffc0) == 0xfe80
                || (segs[0] & 0xff00) == 0xff00
                || (segs[0] == 0x2001 && segs[1] == 0x0db8)
                || (segs[0] == 0x0100 && segs[1] == 0 && segs[2] == 0 && segs[3] == 0)
        }
    }
}
async fn resolve_and_validate_safe_url(
    url_str: &str,
) -> Result<(reqwest::Url, String, IpAddr, u16), ApiError> {
    let parsed = reqwest::Url::parse(url_str)
        .map_err(|e| ApiError::ValidationError(format!("Invalid image URL '{url_str}': {e}")))?;
    if parsed.scheme() != "http" && parsed.scheme() != "https" {
        return Err(ApiError::ValidationError(format!(
            "Forbidden URL scheme '{}': only http and https are allowed",
            parsed.scheme()
        )));
    }
    let host_str = parsed
        .host_str()
        .ok_or_else(|| ApiError::ValidationError("Missing host in image URL".into()))?
        .to_string();
    let port = parsed.port_or_known_default().unwrap_or(80);
    if !is_allowed_port(port) {
        return Err(ApiError::ValidationError(format!(
            "Port {port} is forbidden for media fetching (allowed standard web ports: 80, 443, 8080, 8443)"
        )));
    }
    let addrs = tokio::net::lookup_host(format!("{host_str}:{port}"))
        .await
        .map_err(|e| {
            ApiError::ValidationError(format!("DNS resolution failed for '{host_str}': {e}"))
        })?;
    let mut safe_ip = None;
    for addr in addrs {
        if is_forbidden_ip(addr.ip()) {
            return Err(ApiError::ValidationError(format!(
                "Resolved IP '{}' for host '{}' is forbidden (SSRF protection)",
                addr.ip(),
                host_str
            )));
        }
        if safe_ip.is_none() {
            safe_ip = Some(addr.ip());
        }
    }
    let ip = safe_ip.ok_or_else(|| {
        ApiError::ValidationError(format!("No IP addresses resolved for '{host_str}'"))
    })?;
    Ok((parsed, host_str, ip, port))
}
