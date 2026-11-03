use crate::domain::DocumentId;
use crate::error::StorageError;
use crate::storage::{InMemoryVectorStore, VectorPoint};
use serde::de::{Deserializer, SeqAccess, Visitor};
use std::{
    fmt,
    fs::File,
    io::{BufReader, BufWriter, ErrorKind, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering as AtomicOrdering},
};

pub const LOAD_BATCH_CHUNK_SIZE: usize = 1_000;
pub const SNAPSHOT_SERIALIZE_CHUNK_SIZE: usize = 500;

static TMP_FILE_COUNTER: AtomicU64 = AtomicU64::new(0);

pub struct TempFileGuard<'a> {
    path: &'a Path,
    committed: bool,
}

impl<'a> TempFileGuard<'a> {
    pub fn new(path: &'a Path) -> Self {
        Self {
            path,
            committed: false,
        }
    }

    pub fn commit(mut self) {
        self.committed = true;
    }
}

impl<'a> Drop for TempFileGuard<'a> {
    fn drop(&mut self) {
        if !self.committed {
            let _ = std::fs::remove_file(self.path);
        }
    }
}

pub struct BatchLoaderVisitor<'a> {
    pub store: &'a InMemoryVectorStore,
}

impl<'de, 'a> Visitor<'de> for BatchLoaderVisitor<'a> {
    type Value = ();

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("a JSON array of VectorPoints")
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut batch = Vec::with_capacity(LOAD_BATCH_CHUNK_SIZE);
        while let Some(point) = seq.next_element::<VectorPoint>()? {
            if point.id.trim().is_empty() {
                return Err(serde::de::Error::custom("Document ID cannot be empty"));
            }
            self.store
                .validate_vector(&point.vector)
                .map_err(serde::de::Error::custom)?;
            if let Some(ref img_v) = point.image_vector {
                self.store
                    .validate_vector(img_v)
                    .map_err(serde::de::Error::custom)?;
            }
            batch.push(point);
            if batch.len() >= LOAD_BATCH_CHUNK_SIZE {
                self.store
                    .insert_batch_blocking(std::mem::take(&mut batch))
                    .map_err(serde::de::Error::custom)?;
                batch.reserve(LOAD_BATCH_CHUNK_SIZE);
            }
        }
        if !batch.is_empty() {
            self.store
                .insert_batch_blocking(batch)
                .map_err(serde::de::Error::custom)?;
        }
        Ok(())
    }
}

pub fn save_snapshot(
    store: &InMemoryVectorStore,
    path: &Path,
    all_ids: &[DocumentId],
) -> Result<(), StorageError> {
    let counter = TMP_FILE_COUNTER.fetch_add(1, AtomicOrdering::Relaxed);
    let pid = std::process::id();
    let tmp_path = PathBuf::from(format!("{}.tmp.{}.{}", path.display(), pid, counter));
    let backup_path = PathBuf::from(format!("{}.bak.{}.{}", path.display(), pid, counter));

    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        let _ = std::fs::create_dir_all(parent);
    }

    let tmp_guard = TempFileGuard::new(&tmp_path);
    {
        let file = File::create(&tmp_path)
            .map_err(|e| StorageError::Internal(format!("Failed to create temp snapshot: {e}")))?;
        let mut writer = BufWriter::new(file);
        writer
            .write_all(b"[\n")
            .map_err(|e| StorageError::Internal(format!("IO error: {e}")))?;

        let mut first = true;
        let dim = store.dimension();

        for chunk in all_ids.chunks(SNAPSHOT_SERIALIZE_CHUNK_SIZE) {
            let chunk_points = store.collect_points_for_snapshot(chunk, dim)?;
            for pt in chunk_points {
                if !first {
                    writer
                        .write_all(b",\n")
                        .map_err(|e| StorageError::Internal(format!("IO error: {e}")))?;
                }
                first = false;
                serde_json::to_writer(&mut writer, &pt)
                    .map_err(|e| StorageError::Internal(format!("Serialization error: {e}")))?;
            }
        }

        writer
            .write_all(b"\n]")
            .map_err(|e| StorageError::Internal(format!("IO error: {e}")))?;
        writer
            .flush()
            .map_err(|e| StorageError::Internal(format!("Failed to flush snapshot: {e}")))?;

        let underlying_file = writer
            .into_inner()
            .map_err(|e| StorageError::Internal(e.to_string()))?;
        underlying_file
            .sync_all()
            .map_err(|e| StorageError::Internal(format!("Failed to fsync snapshot: {e}")))?;
    }

    let mut has_backup = false;

    if let Err(rename_err) = std::fs::rename(&tmp_path, path) {
        if path.exists() {
            if let Err(backup_err) = std::fs::rename(path, &backup_path) {
                return Err(StorageError::Internal(format!(
                    "Failed to backup existing snapshot ({backup_err}), original rename error: {rename_err}"
                )));
            }
            has_backup = true;

            if let Err(replace_err) = std::fs::rename(&tmp_path, path) {
                let _ = std::fs::rename(&backup_path, path);
                return Err(StorageError::Internal(format!(
                    "Failed to commit snapshot ({replace_err}), existing snapshot successfully restored"
                )));
            }
        } else {
            return Err(StorageError::Internal(format!(
                "Failed to commit snapshot: {rename_err}"
            )));
        }
    }

    tmp_guard.commit();

    if has_backup {
        let _ = std::fs::remove_file(&backup_path);
    }

    Ok(())
}

pub fn load_snapshot(
    store: &InMemoryVectorStore,
    path: &Path,
) -> Result<Option<InMemoryVectorStore>, StorageError> {
    let file = match File::open(path) {
        Ok(f) => f,
        Err(err) if err.kind() == ErrorKind::NotFound => return Ok(None),
        Err(err) => {
            return Err(StorageError::Internal(format!(
                "Failed to open snapshot: {err}"
            )));
        }
    };

    let reader = BufReader::new(file);
    let mut deserializer = serde_json::Deserializer::from_reader(reader);
    let temp_store = InMemoryVectorStore::with_config(
        store.dimension(),
        store.max_capacity(),
        store.calibration().clone(),
    );

    deserializer
        .deserialize_seq(BatchLoaderVisitor { store: &temp_store })
        .map_err(|e| StorageError::Serialization(format!("Corrupt snapshot file: {e}")))?;

    Ok(Some(temp_store))
}
