use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use bytes::Bytes;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::path::{Component, Path, PathBuf};
use thiserror::Error;

pub type DocumentId = String;

const MAX_BASE64_STRING_LEN: usize = 32 * 1024 * 1024;
pub const SUPPORTED_IMAGE_MIME_TYPES: &[&str] =
    &["image/jpeg", "image/png", "image/webp", "image/gif"];

#[derive(Error, Debug)]
pub enum DomainError {
    #[error("Unsupported MIME type '{0}'. Permitted formats: {SUPPORTED_IMAGE_MIME_TYPES:?}")]
    UnsupportedMimeType(String),

    #[error("Base64 payload length {length} exceeds maximum allowed limit of {max} bytes")]
    Base64TooLong { length: usize, max: usize },

    #[error("Failed to decode base64 payload: {0}")]
    Base64Decode(#[from] base64::DecodeError),

    #[error("Path cannot be empty")]
    EmptyPath,

    #[error("Absolute paths are not permitted: '{0:?}'")]
    AbsolutePath(PathBuf),

    #[error("Path traversal detected: forbidden '..' in '{0:?}'")]
    PathTraversal(PathBuf),

    #[error("Root directories or drive prefixes are forbidden: '{0:?}'")]
    ForbiddenPrefix(PathBuf),

    #[error("Invalid allowed root '{path:?}': {source}")]
    InvalidAllowedRoot {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("Path '{path:?}' escapes allowed root '{root:?}'")]
    PathEscapesRoot { path: PathBuf, root: PathBuf },

    #[error("Parent directory of '{path:?}' escapes allowed root '{root:?}'")]
    ParentEscapesRoot { path: PathBuf, root: PathBuf },

    #[error("ImageSource is not a local file path")]
    NotALocalPath,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum Modality {
    Text,
    Image,
    Multimodal,
}

impl Modality {
    /// Whether a document with this modality passes a modality filter for `requested`.
    /// A multimodal document holds both text and image, so it passes either filter.
    /// Every search path (flat scan, HNSW, post-filter) must use this one rule.
    #[inline]
    pub fn matches_filter(&self, requested: &Modality) -> bool {
        match self {
            Modality::Multimodal => true,
            Modality::Text | Modality::Image => self == requested,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawImageData {
    pub bytes: Bytes,
    pub mime_type: String,
}

impl RawImageData {
    pub fn new(bytes: Bytes, mime_type: impl Into<String>) -> Result<Self, DomainError> {
        let mime = mime_type.into();
        if !SUPPORTED_IMAGE_MIME_TYPES.contains(&mime.as_str()) {
            return Err(DomainError::UnsupportedMimeType(mime));
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
            return Err(serde::de::Error::custom(DomainError::UnsupportedMimeType(
                helper.mime_type,
            )));
        }

        if helper.base64.len() > MAX_BASE64_STRING_LEN {
            return Err(serde::de::Error::custom(DomainError::Base64TooLong {
                length: helper.base64.len(),
                max: MAX_BASE64_STRING_LEN,
            }));
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

fn validate_relative_path(path: &Path, allowed_root: &Path) -> Result<PathBuf, DomainError> {
    if path.as_os_str().is_empty() {
        return Err(DomainError::EmptyPath);
    }
    if path.is_absolute() {
        return Err(DomainError::AbsolutePath(path.to_path_buf()));
    }

    for comp in path.components() {
        match comp {
            Component::ParentDir => {
                return Err(DomainError::PathTraversal(path.to_path_buf()));
            }
            Component::RootDir | Component::Prefix(_) => {
                return Err(DomainError::ForbiddenPrefix(path.to_path_buf()));
            }
            Component::Normal(_) | Component::CurDir => {}
        }
    }

    let canonical_root =
        allowed_root
            .canonicalize()
            .map_err(|source| DomainError::InvalidAllowedRoot {
                path: allowed_root.to_path_buf(),
                source,
            })?;

    let target = canonical_root.join(path);
    if let Ok(canonical_target) = target.canonicalize() {
        if !canonical_target.starts_with(&canonical_root) {
            return Err(DomainError::PathEscapesRoot {
                path: path.to_path_buf(),
                root: canonical_root,
            });
        }
        return Ok(canonical_target);
    }

    let mut current = target.as_path();
    while let Some(parent) = current.parent() {
        if let Ok(canonical_parent) = parent.canonicalize() {
            if !canonical_parent.starts_with(&canonical_root) {
                return Err(DomainError::ParentEscapesRoot {
                    path: path.to_path_buf(),
                    root: canonical_root,
                });
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
    pub fn validate_safe_path(&self, allowed_root: &Path) -> Result<PathBuf, DomainError> {
        match self {
            ImageSource::Path(path) => validate_relative_path(path, allowed_root),
            _ => Err(DomainError::NotALocalPath),
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

    pub fn new_image_bytes(
        bytes: Bytes,
        mime_type: impl Into<String>,
    ) -> Result<Self, DomainError> {
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
