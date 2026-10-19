use rag_server::{
    domain::Modality,
    error::StorageError,
    storage::{InMemoryVectorStore, SearchFilter, VectorPoint, VectorStore},
};
use serde_json::json;
use std::f32::consts::FRAC_1_SQRT_2;

#[tokio::test]
async fn test_insert_and_get() {
    let store = InMemoryVectorStore::new(4);
    let point = VectorPoint::new(
        "doc1",
        vec![1.0, 0.0, 0.0, 0.0],
        json!({"title": "multimodal article"}),
        Modality::Text,
    );
    store.insert(point).await.expect("Insert must succeed");
    assert_eq!(store.count().await.unwrap(), 1);
    let retrieved = store.get(&"doc1".to_string()).await.unwrap();
    assert!(retrieved.is_some());
    let unwrapped = retrieved.unwrap();
    assert_eq!(unwrapped.id, "doc1");
    assert_eq!(unwrapped.payload["title"], "multimodal article");
}

#[tokio::test]
async fn test_dimension_mismatch_and_zero_dimension_rejected() {
    let store = InMemoryVectorStore::new(4);
    let point = VectorPoint::new("doc1", vec![1.0, 0.0], json!({}), Modality::Text);
    let result = store.insert(point).await;
    assert!(matches!(
        result,
        Err(StorageError::DimensionMismatch {
            expected: 4,
            actual: 2
        })
    ));
    let search_result = store.search(&[1.0, 0.0, 0.0], 5, None).await;
    assert!(matches!(
        search_result,
        Err(StorageError::DimensionMismatch {
            expected: 4,
            actual: 3
        })
    ));
    let zero_store = InMemoryVectorStore::new(0);
    let zero_res = zero_store
        .insert(VectorPoint::new("d", vec![], json!({}), Modality::Text))
        .await;
    assert!(matches!(zero_res, Err(StorageError::InvalidVector(_))));
}

#[tokio::test]
async fn test_nan_infinity_zero_and_unnormalized_vectors_rejected() {
    let store = InMemoryVectorStore::new(3);
    let nan_point = VectorPoint::new(
        "nan_doc",
        vec![1.0, f32::NAN, 0.0],
        json!({}),
        Modality::Text,
    );
    assert!(matches!(
        store.insert(nan_point).await,
        Err(StorageError::InvalidVector(_))
    ));
    let inf_point = VectorPoint::new(
        "inf_doc",
        vec![1.0, f32::INFINITY, 0.0],
        json!({}),
        Modality::Text,
    );
    assert!(matches!(
        store.insert(inf_point).await,
        Err(StorageError::InvalidVector(_))
    ));
    let zero_point = VectorPoint::new("zero_doc", vec![0.0, 0.0, 0.0], json!({}), Modality::Text);
    assert!(matches!(
        store.insert(zero_point).await,
        Err(StorageError::InvalidVector(_))
    ));
    let unnorm_point = VectorPoint::new(
        "unnorm_doc",
        vec![10.0, 0.0, 0.0],
        json!({}),
        Modality::Text,
    );
    assert!(matches!(
        store.insert(unnorm_point).await,
        Err(StorageError::InvalidVector(_))
    ));
    let batch = vec![
        VectorPoint::new("valid_doc", vec![1.0, 0.0, 0.0], json!({}), Modality::Text),
        VectorPoint::new("bad_doc", vec![5.0, 5.0, 0.0], json!({}), Modality::Text),
    ];
    assert!(matches!(
        store.insert_batch(batch).await,
        Err(StorageError::InvalidVector(_))
    ));
    assert_eq!(store.count().await.unwrap(), 0);
    assert!(matches!(
        store.search(&[0.0, 0.0, 0.0], 5, None).await,
        Err(StorageError::InvalidVector(_))
    ));
}

#[tokio::test]
async fn test_filter_min_score_range_validation() {
    let store = InMemoryVectorStore::new(2);
    let filter_too_high = SearchFilter::new().with_min_score(1.5);
    assert!(matches!(
        store.search(&[1.0, 0.0], 5, Some(filter_too_high)).await,
        Err(StorageError::InvalidFilter(_))
    ));
    let filter_too_low = SearchFilter::new().with_min_score(-2.0);
    assert!(matches!(
        store.search(&[1.0, 0.0], 5, Some(filter_too_low)).await,
        Err(StorageError::InvalidFilter(_))
    ));
    let filter_nan = SearchFilter::new().with_min_score(f32::NAN);
    assert!(matches!(
        store.search(&[1.0, 0.0], 5, Some(filter_nan)).await,
        Err(StorageError::InvalidFilter(_))
    ));
}

#[tokio::test]
async fn test_capacity_exceeded_error_returned() {
    let store = InMemoryVectorStore::with_capacity(2, 2);
    store
        .insert(VectorPoint::new(
            "p1",
            vec![1.0, 0.0],
            json!({}),
            Modality::Text,
        ))
        .await
        .unwrap();
    store
        .insert(VectorPoint::new(
            "p2",
            vec![0.0, 1.0],
            json!({}),
            Modality::Text,
        ))
        .await
        .unwrap();
    let overflow_res = store
        .insert(VectorPoint::new(
            "p3",
            vec![1.0, 0.0],
            json!({}),
            Modality::Text,
        ))
        .await;
    assert!(matches!(
        overflow_res,
        Err(StorageError::CapacityExceeded { current: 2, max: 2 })
    ));
    let batch = vec![VectorPoint::new(
        "p4",
        vec![1.0, 0.0],
        json!({}),
        Modality::Text,
    )];
    let batch_res = store.insert_batch(batch).await;
    assert!(matches!(
        batch_res,
        Err(StorageError::CapacityExceeded { current: 2, max: 2 })
    ));
    let update_res = store
        .insert(VectorPoint::new(
            "p1",
            vec![0.0, 1.0],
            json!({"updated": true}),
            Modality::Text,
        ))
        .await;
    assert!(update_res.is_ok());
    assert_eq!(store.count().await.unwrap(), 2);
}

#[tokio::test]
async fn test_top_k_cosine_search_ranking_and_filtering() {
    let store = InMemoryVectorStore::new(2);
    let points = vec![
        VectorPoint::new(
            "p_exact",
            vec![1.0, 0.0],
            json!({"name": "exact"}),
            Modality::Text,
        ),
        VectorPoint::new(
            "p_ortho",
            vec![0.0, 1.0],
            json!({"name": "ortho"}),
            Modality::Image,
        ),
        VectorPoint::new(
            "p_diag",
            vec![FRAC_1_SQRT_2, FRAC_1_SQRT_2],
            json!({"name": "diag"}),
            Modality::Text,
        ),
    ];
    store
        .insert_batch(points)
        .await
        .expect("Batch insert must succeed");
    let query = vec![1.0, 0.0];
    let results = store.search(&query, 2, None).await.unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].id, "p_exact");
    assert!((results[0].score - 1.0).abs() < 1e-4);
    assert_eq!(results[1].id, "p_diag");
    assert!((results[1].score - FRAC_1_SQRT_2).abs() < 1e-4);
    let filter_image = SearchFilter::new().with_modality(Modality::Image);
    let img_results = store.search(&query, 10, Some(filter_image)).await.unwrap();
    assert_eq!(img_results.len(), 1);
    assert_eq!(img_results[0].id, "p_ortho");
    let filter_score = SearchFilter::new().with_min_score(0.8);
    let high_score_results = store.search(&query, 10, Some(filter_score)).await.unwrap();
    assert_eq!(high_score_results.len(), 1);
    assert_eq!(high_score_results[0].id, "p_exact");
}

#[tokio::test]
async fn test_deterministic_tie_breaking_on_equal_scores() {
    let store = InMemoryVectorStore::new(2);
    let points = vec![
        VectorPoint::new("doc_e", vec![1.0, 0.0], json!({}), Modality::Text),
        VectorPoint::new("doc_b", vec![1.0, 0.0], json!({}), Modality::Text),
        VectorPoint::new("doc_a", vec![1.0, 0.0], json!({}), Modality::Text),
        VectorPoint::new("doc_d", vec![1.0, 0.0], json!({}), Modality::Text),
        VectorPoint::new("doc_c", vec![1.0, 0.0], json!({}), Modality::Text),
    ];
    store.insert_batch(points).await.unwrap();
    let results = store.search(&[1.0, 0.0], 3, None).await.unwrap();
    assert_eq!(results.len(), 3);
    assert_eq!(results[0].id, "doc_a");
    assert_eq!(results[1].id, "doc_b");
    assert_eq!(results[2].id, "doc_c");
}

#[tokio::test]
async fn test_hnsw_scale_search_accuracy() {
    let dim = 16;
    let store = InMemoryVectorStore::new(dim);
    let mut batch = Vec::with_capacity(500);
    for i in 0..500 {
        let mut vec = vec![0.0f32; dim];
        let mut norm_sq = 0.0f32;
        for (d, item) in vec.iter_mut().enumerate() {
            let seed = ((i as u64 + 1) * 2654435761) ^ ((d as u64 + 1) * 2246822519);
            let val = ((seed % 2000) as f32 / 1000.0) - 1.0;
            *item = val;
            norm_sq += val * val;
        }
        let inv_norm = 1.0 / norm_sq.sqrt();
        for item in &mut vec {
            *item *= inv_norm;
        }
        if vec[0] > 0.4 {
            vec[0] = 0.2;
            let mut nsq = 0.0f32;
            for &x in &vec {
                nsq += x * x;
            }
            let inorm = 1.0 / nsq.sqrt();
            for x in &mut vec {
                *x *= inorm;
            }
        }
        batch.push(VectorPoint::new(
            format!("bg_{i:04}"),
            vec,
            json!({"index": i}),
            Modality::Text,
        ));
    }
    let mut target_1 = vec![0.0f32; dim];
    target_1[0] = 1.0;
    batch.push(VectorPoint::new(
        "winner_1",
        target_1.clone(),
        json!({"rank": 1}),
        Modality::Image,
    ));
    let mut target_2 = vec![0.0f32; dim];
    target_2[0] = 0.8;
    target_2[1] = 0.6;
    batch.push(VectorPoint::new(
        "winner_2",
        target_2,
        json!({"rank": 2}),
        Modality::Text,
    ));
    store.insert_batch(batch).await.unwrap();
    assert_eq!(store.count().await.unwrap(), 502);
    let results = store.search(&target_1, 2, None).await.unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].id, "winner_1");
    assert!((results[0].score - 1.0).abs() < 1e-4);
    assert_eq!(results[1].id, "winner_2");
    assert!((results[1].score - 0.8).abs() < 1e-4);
}

#[tokio::test]
async fn test_hnsw_upsert_entry_point_preserves_graph_connectivity() {
    let dim = 16;
    let store = InMemoryVectorStore::new(dim);
    let mut initial_vec = vec![0.0f32; dim];
    initial_vec[0] = 1.0;
    store
        .insert(VectorPoint::new(
            "entry_doc",
            initial_vec.clone(),
            json!({"v": 1}),
            Modality::Text,
        ))
        .await
        .unwrap();
    let mut batch = Vec::with_capacity(140);
    for i in 1..=140 {
        let mut vec = vec![0.0f32; dim];
        let mut norm_sq = 0.0f32;
        for (d, item) in vec.iter_mut().enumerate() {
            let seed = ((i as u64 + 42) * 2654435761) ^ ((d as u64 + 7) * 2246822519);
            let val = ((seed % 2000) as f32 / 1000.0) - 1.0;
            *item = val;
            norm_sq += val * val;
        }
        let inv_norm = 1.0 / norm_sq.sqrt();
        for item in &mut vec {
            *item *= inv_norm;
        }
        if vec[0].abs() > 0.3 || vec[3].abs() > 0.3 {
            vec[0] = 0.1;
            vec[3] = 0.1;
            let mut nsq = 0.0f32;
            for &x in &vec {
                nsq += x * x;
            }
            let inorm = 1.0 / nsq.sqrt();
            for x in &mut vec {
                *x *= inorm;
            }
        }
        batch.push(VectorPoint::new(
            format!("doc_{i}"),
            vec,
            json!({"v": i}),
            Modality::Text,
        ));
    }
    store.insert_batch(batch).await.unwrap();
    assert_eq!(store.count().await.unwrap(), 141);
    let mut updated_vec = vec![0.0f32; dim];
    updated_vec[3] = 1.0;
    store
        .insert(VectorPoint::new(
            "entry_doc",
            updated_vec.clone(),
            json!({"v": 999}),
            Modality::Text,
        ))
        .await
        .unwrap();
    assert_eq!(store.count().await.unwrap(), 141);
    let results = store.search(&updated_vec, 3, None).await.unwrap();
    assert!(!results.is_empty());
    assert_eq!(results[0].id, "entry_doc");
    assert!((results[0].score - 1.0).abs() < 1e-4);
    let results_other = store.search(&initial_vec, 3, None).await.unwrap();
    assert!(!results_other.is_empty());
}

#[tokio::test]
async fn test_hnsw_search_after_deletions_no_panic() {
    let dim = 8;
    let store = InMemoryVectorStore::new(dim);
    let mut batch = Vec::with_capacity(150);
    for i in 0..150 {
        let mut vec = vec![0.0f32; dim];
        vec[i % dim] = 1.0;
        batch.push(VectorPoint::new(
            format!("node_{i:03}"),
            vec,
            json!({}),
            Modality::Text,
        ));
    }
    store.insert_batch(batch).await.unwrap();
    assert!(store.delete(&"node_000".to_string()).await.unwrap());
    assert!(store.delete(&"node_075".to_string()).await.unwrap());
    assert!(store.delete(&"node_149".to_string()).await.unwrap());
    assert_eq!(store.count().await.unwrap(), 147);
    let mut query = vec![0.0f32; dim];
    query[1] = 1.0;
    let results = store.search(&query, 5, None).await.unwrap();
    assert!(!results.is_empty());
    assert!(
        results
            .iter()
            .all(|r| r.id != "node_000" && r.id != "node_075" && r.id != "node_149")
    );
}

#[tokio::test]
async fn test_hnsw_in_search_filtering_rare_modality() {
    let dim = 8;
    let store = InMemoryVectorStore::new(dim);
    let mut batch = Vec::with_capacity(200);
    for i in 0..198 {
        let mut vec = vec![0.0f32; dim];
        vec[0] = 1.0;
        batch.push(VectorPoint::new(
            format!("text_{i}"),
            vec,
            json!({}),
            Modality::Text,
        ));
    }
    let mut img_vec = vec![0.0f32; dim];
    img_vec[1] = 1.0;
    batch.push(VectorPoint::new(
        "rare_img_1",
        img_vec.clone(),
        json!({}),
        Modality::Image,
    ));
    batch.push(VectorPoint::new(
        "rare_img_2",
        img_vec.clone(),
        json!({}),
        Modality::Image,
    ));
    store.insert_batch(batch).await.unwrap();
    assert_eq!(store.count().await.unwrap(), 200);
    let filter = SearchFilter::new().with_modality(Modality::Image);
    let results = store.search(&img_vec, 2, Some(filter)).await.unwrap();
    assert_eq!(results.len(), 2);
    assert!(results.iter().all(|r| r.modality == Modality::Image));
}

#[tokio::test]
async fn test_delete_swap_remove_and_upsert_integrity() {
    let store = InMemoryVectorStore::new(3);
    store
        .insert_batch(vec![
            VectorPoint::new(
                "first",
                vec![1.0, 0.0, 0.0],
                json!({"v": 1}),
                Modality::Text,
            ),
            VectorPoint::new(
                "middle",
                vec![0.0, 1.0, 0.0],
                json!({"v": 2}),
                Modality::Image,
            ),
            VectorPoint::new("last", vec![0.0, 0.0, 1.0], json!({"v": 3}), Modality::Text),
        ])
        .await
        .unwrap();
    assert!(store.delete(&"first".to_string()).await.unwrap());
    assert_eq!(store.count().await.unwrap(), 2);
    assert!(store.get(&"first".to_string()).await.unwrap().is_none());
    let last_pt = store.get(&"last".to_string()).await.unwrap().unwrap();
    assert_eq!(last_pt.vector, vec![0.0, 0.0, 1.0]);
    assert_eq!(last_pt.payload["v"], 3);
    store
        .insert(VectorPoint::new(
            "middle",
            vec![1.0, 0.0, 0.0],
            json!({"v": 22}),
            Modality::Multimodal,
        ))
        .await
        .unwrap();
    assert_eq!(store.count().await.unwrap(), 2);
    let updated_mid = store.get(&"middle".to_string()).await.unwrap().unwrap();
    assert_eq!(updated_mid.vector, vec![1.0, 0.0, 0.0]);
    assert_eq!(updated_mid.payload["v"], 22);
    assert_eq!(updated_mid.modality, Modality::Multimodal);
}

#[tokio::test]
async fn test_huge_limit_does_not_overflow_or_oom() {
    let store = InMemoryVectorStore::new(2);
    store
        .insert(VectorPoint::new(
            "p1",
            vec![1.0, 0.0],
            json!({}),
            Modality::Text,
        ))
        .await
        .unwrap();
    let results = store.search(&[1.0, 0.0], usize::MAX, None).await.unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, "p1");
}
