use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use bytes::Bytes;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::path::{Component, Path, PathBuf};

pub type DocumentId = String;

const MAX_BASE64_STRING_LEN: usize = 32 * 1024 * 1024;
pub const SUPPORTED_IMAGE_MIME_TYPES: &[&str] =
    &["image/jpeg", "image/png", "image/webp", "image/gif"];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum Modality {
    Text,
    Image,
    Multimodal,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawImageData {
    pub bytes: Bytes,
    pub mime_type: String,
}

impl RawImageData {
    pub fn new(bytes: Bytes, mime_type: impl Into<String>) -> Result<Self, String> {
        let mime = mime_type.into();
        if !SUPPORTED_IMAGE_MIME_TYPES.contains(&mime.as_str()) {
            return Err(format!(
                "Unsupported MIME type '{mime}'. Allowed: {SUPPORTED_IMAGE_MIME_TYPES:?}"
            ));
        }
        Ok(Self {
            bytes,
            mime_type: mime,
        })
    }
}

impl Serialize for RawImageData {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("RawImageData", 2)?;
        state.serialize_field("base64", &BASE64.encode(&self.bytes))?;
        state.serialize_field("mime_type", &self.mime_type)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for RawImageData {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct RawHelper {
            base64: String,
            mime_type: String,
        }

        let helper = RawHelper::deserialize(deserializer)?;
        if !SUPPORTED_IMAGE_MIME_TYPES.contains(&helper.mime_type.as_str()) {
            return Err(serde::de::Error::custom(format!(
                "Unsupported MIME type '{}'. Permitted formats: {:?}",
                helper.mime_type, SUPPORTED_IMAGE_MIME_TYPES
            )));
        }

        if helper.base64.len() > MAX_BASE64_STRING_LEN {
            return Err(serde::de::Error::custom(format!(
                "Base64 payload length {} exceeds maximum allowed limit of {} bytes",
                helper.base64.len(),
                MAX_BASE64_STRING_LEN
            )));
        }

        let decoded = BASE64
            .decode(helper.base64.as_bytes())
            .map_err(serde::de::Error::custom)?;

        Ok(Self {
            bytes: Bytes::from(decoded),
            mime_type: helper.mime_type,
        })
    }
}

fn validate_relative_path(path: &Path, allowed_root: &Path) -> Result<PathBuf, String> {
    if path.as_os_str().is_empty() {
        return Err("Path cannot be empty".to_string());
    }
    if path.is_absolute() {
        return Err(format!("Absolute paths are not permitted: '{path:?}'"));
    }

    for comp in path.components() {
        match comp {
            Component::ParentDir => {
                return Err(format!(
                    "Path traversal detected: forbidden '..' in '{path:?}'"
                ));
            }
            Component::RootDir | Component::Prefix(_) => {
                return Err(format!(
                    "Root directories or drive prefixes are forbidden: '{path:?}'"
                ));
            }
            Component::Normal(_) | Component::CurDir => {}
        }
    }

    let canonical_root = allowed_root
        .canonicalize()
        .map_err(|e| format!("Invalid allowed root '{allowed_root:?}': {e}"))?;

    let target = canonical_root.join(path);
    if let Ok(canonical_target) = target.canonicalize() {
        if !canonical_target.starts_with(&canonical_root) {
            return Err(format!(
                "Path '{path:?}' escapes allowed root '{canonical_root:?}'"
            ));
        }
        return Ok(canonical_target);
    }

    let mut current = target.as_path();
    while let Some(parent) = current.parent() {
        if let Ok(canonical_parent) = parent.canonicalize() {
            if !canonical_parent.starts_with(&canonical_root) {
                return Err(format!(
                    "Parent directory of '{path:?}' escapes allowed root '{canonical_root:?}'"
                ));
            }
            break;
        }
        current = parent;
    }

    Ok(target)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", content = "data")]
pub enum ImageSource {
    Path(PathBuf),
    Url(String),
    Raw(RawImageData),
}

impl ImageSource {
    pub fn validate_safe_path(&self, allowed_root: &Path) -> Result<PathBuf, String> {
        match self {
            ImageSource::Path(path) => validate_relative_path(path, allowed_root),
            _ => Err("ImageSource is not a local file path".to_string()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultimodalPayload {
    pub text: Option<String>,
    pub image: Option<ImageSource>,
}

impl MultimodalPayload {
    pub fn new_text(text: impl Into<String>) -> Self {
        Self {
            text: Some(text.into()),
            image: None,
        }
    }

    pub fn new_image_path(path: impl Into<PathBuf>) -> Self {
        Self {
            text: None,
            image: Some(ImageSource::Path(path.into())),
        }
    }

    pub fn new_image_bytes(bytes: Bytes, mime_type: impl Into<String>) -> Result<Self, String> {
        let raw = RawImageData::new(bytes, mime_type)?;
        Ok(Self {
            text: None,
            image: Some(ImageSource::Raw(raw)),
        })
    }

    pub fn is_empty(&self) -> bool {
        self.text.is_none() && self.image.is_none()
    }
}
