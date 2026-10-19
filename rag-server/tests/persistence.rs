use rag_server::{
    domain::Modality,
    storage::{InMemoryVectorStore, VectorPoint, VectorStore},
};
use serde_json::json;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

static TEST_ID_GEN: AtomicU64 = AtomicU64::new(0);

fn generate_temp_path(prefix: &str) -> PathBuf {
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let unique_counter = TEST_ID_GEN.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    std::env::temp_dir().join(format!("{prefix}_{pid}_{timestamp}_{unique_counter}.json"))
}

#[tokio::test]
async fn test_snapshot_roundtrip_restores_hnsw_search() {
    let dim = 4;
    let store = InMemoryVectorStore::new(dim);
    let p1 = VectorPoint::new(
        "doc_alpha",
        vec![1.0, 0.0, 0.0, 0.0],
        json!({"title": "Alpha"}),
        Modality::Text,
    );
    let p2 = VectorPoint::new(
        "doc_beta",
        vec![0.0, 1.0, 0.0, 0.0],
        json!({"title": "Beta"}),
        Modality::Text,
    );
    store.insert(p1).await.unwrap();
    store.insert(p2).await.unwrap();
    assert_eq!(store.count().await.unwrap(), 2);
    let snapshot_file = generate_temp_path("test_snapshot_roundtrip");
    store.save_to_disk(&snapshot_file).await.unwrap();
    assert!(snapshot_file.exists());
    let fresh_store = InMemoryVectorStore::new(dim);
    let loaded_count = fresh_store.load_from_disk(&snapshot_file).await.unwrap();
    assert_eq!(loaded_count, 2);
    assert_eq!(fresh_store.count().await.unwrap(), 2);
    let search_res = fresh_store
        .search(&[1.0, 0.0, 0.0, 0.0], 1, None)
        .await
        .unwrap();
    assert_eq!(search_res.len(), 1);
    assert_eq!(search_res[0].id, "doc_alpha");
    let _ = tokio::fs::remove_file(&snapshot_file).await;
}

#[tokio::test]
async fn test_snapshot_missing_file_handled_gracefully() {
    let store = InMemoryVectorStore::new(4);
    let missing_path = PathBuf::from("non_existent_snapshot_never_created_98765.json");
    let res = store.load_from_disk(&missing_path).await;
    assert!(res.is_ok());
    assert_eq!(res.unwrap(), 0);
}

#[tokio::test]
async fn test_snapshot_corrupted_data_returns_error() {
    let store = InMemoryVectorStore::new(4);
    let corrupt_file = generate_temp_path("test_corrupted_snapshot");
    tokio::fs::write(&corrupt_file, b"NOT_A_VALID_JSON_STRUCTURE")
        .await
        .unwrap();
    let res = store.load_from_disk(&corrupt_file).await;
    assert!(res.is_err());
    let _ = tokio::fs::remove_file(&corrupt_file).await;
}

#[tokio::test]
async fn test_snapshot_atomic_write_cleans_up_tmp_file() {
    let store = InMemoryVectorStore::new(4);
    let p = VectorPoint::new("p1", vec![1.0, 0.0, 0.0, 0.0], json!({}), Modality::Text);
    store.insert(p).await.unwrap();
    let target = generate_temp_path("test_atomic");
    store.save_to_disk(&target).await.unwrap();
    assert!(target.exists(), "Target file must exist after rename");
    let parent = target.parent().unwrap();
    let file_stem = target.file_stem().unwrap().to_str().unwrap();
    let mut dir_entries = tokio::fs::read_dir(parent).await.unwrap();
    while let Ok(Some(entry)) = dir_entries.next_entry().await {
        let name = entry.file_name();
        let name_str = name.to_str().unwrap_or_default();
        if name_str.starts_with(file_stem) && name_str.contains(".tmp.") {
            panic!("Found uncleaned temp file: {name_str}");
        }
    }
    let _ = tokio::fs::remove_file(&target).await;
}
