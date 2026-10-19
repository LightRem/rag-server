use async_trait::async_trait;
use bytes::Bytes;
use rag_server::{
    domain::Modality,
    error::{GenerationError, InferenceError, RagError},
    inference::Embedder,
    rag::{MAX_PROMPT_CHARS, MockLlmClient, RagPipeline, RagQueryRequest},
    storage::{InMemoryVectorStore, VectorPoint, VectorStore},
};
use serde_json::json;
use std::sync::Arc;

struct MockTextEmbedder;
#[async_trait]
impl Embedder for MockTextEmbedder {
    fn dimension(&self) -> usize {
        4
    }

    async fn embed_text(&self, text: &str) -> Result<Vec<f32>, InferenceError> {
        let mut vec = vec![0.0f32; 4];
        if text.contains("car") || text.contains("Ferrari") || text.contains("авто") {
            vec[0] = 1.0;
        } else {
            vec[1] = 1.0;
        }
        Ok(vec)
    }

    async fn embed_image(&self, _image: &[u8]) -> Result<Vec<f32>, InferenceError> {
        Ok(vec![0.0; 4])
    }

    async fn embed_text_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, InferenceError> {
        let mut res = Vec::with_capacity(texts.len());
        for t in texts {
            res.push(self.embed_text(t).await?);
        }
        Ok(res)
    }

    async fn embed_image_batch(&self, images: &[Bytes]) -> Result<Vec<Vec<f32>>, InferenceError> {
        Ok(vec![vec![0.0; 4]; images.len()])
    }
}

#[tokio::test]
async fn test_multimodal_rag_end_to_end_pipeline() {
    let dim = 4;
    let store = Arc::new(InMemoryVectorStore::new(dim));
    store
        .insert(VectorPoint::new(
            "doc_ferrari_spec",
            vec![1.0, 0.0, 0.0, 0.0],
            json!({
                "title": "Ferrari F8 Tributo Specs",
                "text": "The Ferrari F8 Tributo reaches a top speed of 340 km/h with a 720 hp twin-turbo V8 engine."
            }),
            Modality::Text,
        ))
        .await
        .unwrap();
    store
        .insert(VectorPoint::new(
            "img_ferrari_red",
            vec![1.0, 0.0, 0.0, 0.0],
            json!({
                "url": "https://cdn.example.com/ferrari_f8_red.jpg",
                "caption": "Side profile of a red Ferrari F8 Tributo parked on a track"
            }),
            Modality::Image,
        ))
        .await
        .unwrap();
    store
        .insert(VectorPoint::new(
            "doc_coffee_guide",
            vec![0.0, 1.0, 0.0, 0.0],
            json!({
                "title": "Espresso Brewing",
                "text": "Espresso extraction requires 9 bars of pressure."
            }),
            Modality::Text,
        ))
        .await
        .unwrap();
    assert_eq!(store.count().await.unwrap(), 3);
    let embedder = Arc::new(MockTextEmbedder);
    let mock_llm = Arc::new(MockLlmClient::with_response(
        "Based on [source id: doc_ferrari_spec], the Ferrari F8 Tributo reaches 340 km/h. \
         Visual reference is provided in [image id: img_ferrari_red].",
    ));
    let pipeline = RagPipeline::new(store, embedder, mock_llm);
    let request = RagQueryRequest {
        query: "What is the top speed of the Ferrari car?".to_string(),
        image: None,
        top_k: 2,
        min_score: 0.5,
        system_prompt: None,
        filter_uncited: false,
        allow_empty_context: false,
        temperature: Some(0.1),
        max_tokens: Some(512),
    };
    let response = pipeline.answer(request).await.expect("RAG pipeline failed");
    assert!(response.answer.contains("doc_ferrari_spec"));
    assert!(response.answer.contains("img_ferrari_red"));
    assert_eq!(response.sources.len(), 1);
    assert_eq!(response.sources[0].id, "doc_ferrari_spec");
    assert!(response.sources[0].is_cited);
    assert_eq!(response.visuals.len(), 1);
    assert_eq!(response.visuals[0].id, "img_ferrari_red");
    assert!(response.visuals[0].is_cited);
}

#[tokio::test]
async fn test_grouped_and_unspaced_citations() {
    let dim = 4;
    let store = Arc::new(InMemoryVectorStore::new(dim));
    store
        .insert_batch(vec![
            VectorPoint::new(
                "doc_1",
                vec![1.0, 0.0, 0.0, 0.0],
                json!({"text": "Fact 1"}),
                Modality::Text,
            ),
            VectorPoint::new(
                "doc_2",
                vec![1.0, 0.0, 0.0, 0.0],
                json!({"text": "Fact 2"}),
                Modality::Text,
            ),
            VectorPoint::new(
                "doc_3",
                vec![1.0, 0.0, 0.0, 0.0],
                json!({"text": "Fact 3"}),
                Modality::Text,
            ),
        ])
        .await
        .unwrap();
    let embedder = Arc::new(MockTextEmbedder);
    let mock_llm = Arc::new(MockLlmClient::with_response(
        "First fact from [doc:doc_1]. Second and third facts from [doc_2, doc_3].",
    ));
    let pipeline = RagPipeline::new(store, embedder, mock_llm);
    let request = RagQueryRequest {
        query: "Ferrari car facts".to_string(),
        image: None,
        top_k: 5,
        min_score: 0.5,
        system_prompt: None,
        filter_uncited: true,
        allow_empty_context: false,
        temperature: None,
        max_tokens: None,
    };
    let response = pipeline.answer(request).await.unwrap();
    assert_eq!(response.sources.len(), 3);
    assert!(response.sources.iter().all(|s| s.is_cited));
}

#[tokio::test]
async fn test_numeric_footnote_disambiguation() {
    let dim = 4;
    let store = Arc::new(InMemoryVectorStore::new(dim));
    store
        .insert(VectorPoint::new(
            "1",
            vec![1.0, 0.0, 0.0, 0.0],
            json!({"text": "Short numeric ID doc"}),
            Modality::Text,
        ))
        .await
        .unwrap();
    let embedder = Arc::new(MockTextEmbedder);
    let mock_llm = Arc::new(MockLlmClient::with_response(
        "Here is a regular list: 1. Step one [1].",
    ));
    let pipeline = RagPipeline::new(store.clone(), embedder.clone(), mock_llm);
    let req_uncited = RagQueryRequest {
        query: "Ferrari car info".to_string(),
        image: None,
        top_k: 2,
        min_score: 0.5,
        system_prompt: None,
        filter_uncited: true,
        allow_empty_context: false,
        temperature: None,
        max_tokens: None,
    };
    let resp_uncited = pipeline.answer(req_uncited).await.unwrap();
    assert_eq!(resp_uncited.sources.len(), 0);
    let mock_llm_explicit = Arc::new(MockLlmClient::with_response(
        "According to [doc: 1], the specs are verified.",
    ));
    let pipeline_explicit = RagPipeline::new(store, embedder, mock_llm_explicit);
    let resp_explicit = pipeline_explicit
        .answer(RagQueryRequest {
            query: "Ferrari car info".to_string(),
            image: None,
            top_k: 2,
            min_score: 0.5,
            system_prompt: None,
            filter_uncited: true,
            allow_empty_context: false,
            temperature: None,
            max_tokens: None,
        })
        .await
        .unwrap();
    assert_eq!(resp_explicit.sources.len(), 1);
    assert!(resp_explicit.sources[0].is_cited);
}

#[tokio::test]
async fn test_context_budgeting_does_not_crash_on_large_search_results() {
    let dim = 4;
    let store = Arc::new(InMemoryVectorStore::new(dim));
    let mut batch = Vec::new();
    for i in 0..12 {
        batch.push(VectorPoint::new(
            format!("doc_large_{i}"),
            vec![1.0, 0.0, 0.0, 0.0],
            json!({"text": "A".repeat(1800)}),
            Modality::Text,
        ));
    }
    store.insert_batch(batch).await.unwrap();
    let embedder = Arc::new(MockTextEmbedder);
    let mock_llm = Arc::new(MockLlmClient::with_response("Summarized answer."));
    let pipeline = RagPipeline::new(store, embedder, mock_llm);
    let request = RagQueryRequest {
        query: "Ferrari car large query".to_string(),
        image: None,
        top_k: 12,
        min_score: 0.5,
        system_prompt: None,
        filter_uncited: false,
        allow_empty_context: false,
        temperature: None,
        max_tokens: None,
    };
    let response = pipeline.answer(request).await;
    assert!(response.is_ok());
    let unwrapped = response.unwrap();
    assert!(!unwrapped.sources.is_empty());
    assert!(unwrapped.sources.len() < 12);
}

#[tokio::test]
async fn test_custom_system_prompt_preserves_citation_instructions() {
    let dim = 4;
    let store = Arc::new(InMemoryVectorStore::new(dim));
    store
        .insert(VectorPoint::new(
            "doc_cited",
            vec![1.0, 0.0, 0.0, 0.0],
            json!({"text": "Fact"}),
            Modality::Text,
        ))
        .await
        .unwrap();
    let embedder = Arc::new(MockTextEmbedder);
    let mock_llm = Arc::new(MockLlmClient::with_response(
        "Відповідь на основі [Source ID: doc_cited].",
    ));
    let pipeline = RagPipeline::new(store, embedder, mock_llm);
    let request = RagQueryRequest {
        query: "Ferrari car info".to_string(),
        image: None,
        top_k: 2,
        min_score: 0.5,
        system_prompt: Some("Відповідай коротко українською мовою.".to_string()),
        filter_uncited: true,
        allow_empty_context: false,
        temperature: None,
        max_tokens: None,
    };
    let response = pipeline.answer(request).await.unwrap();
    assert_eq!(response.sources.len(), 1);
    assert_eq!(response.sources[0].id, "doc_cited");
    assert!(response.sources[0].is_cited);
}

#[tokio::test]
async fn test_multimodal_document_extracts_both_text_and_image() {
    let dim = 4;
    let store = Arc::new(InMemoryVectorStore::new(dim));
    store
        .insert(VectorPoint::new(
            "product_page",
            vec![1.0, 0.0, 0.0, 0.0],
            json!({
                "title": "Ferrari Model",
                "text": "Full product description and technical specs.",
                "url": "https://cdn.example.com/product_photo.jpg",
                "caption": "Product packaging and car"
            }),
            Modality::Multimodal,
        ))
        .await
        .unwrap();
    let embedder = Arc::new(MockTextEmbedder);
    let mock_llm = Arc::new(MockLlmClient::with_response(
        "According to [Source ID: product_page], view photo [Image ID: product_page].",
    ));
    let pipeline = RagPipeline::new(store, embedder, mock_llm);
    let request = RagQueryRequest {
        query: "Ferrari car info".to_string(),
        image: None,
        top_k: 2,
        min_score: 0.5,
        system_prompt: None,
        filter_uncited: false,
        allow_empty_context: false,
        temperature: None,
        max_tokens: None,
    };
    let response = pipeline.answer(request).await.unwrap();
    assert_eq!(response.sources.len(), 1);
    assert_eq!(response.sources[0].id, "product_page");
    assert!(response.sources[0].is_cited);
    assert_eq!(response.visuals.len(), 1);
    assert_eq!(response.visuals[0].id, "product_page");
    assert_eq!(
        response.visuals[0].url,
        "https://cdn.example.com/product_photo.jpg"
    );
    assert!(response.visuals[0].is_cited);
}

#[tokio::test]
async fn test_cyrillic_unicode_prompt_length_accepted() {
    let dim = 4;
    let store = Arc::new(InMemoryVectorStore::new(dim));
    let embedder = Arc::new(MockTextEmbedder);
    let mock_llm = Arc::new(MockLlmClient::new());
    let pipeline = RagPipeline::new(store, embedder, mock_llm);
    let cyrillic_prompt = "текст ".repeat(1500);
    assert!(cyrillic_prompt.len() > 16_000);
    assert!(cyrillic_prompt.chars().count() < MAX_PROMPT_CHARS);
    let request = RagQueryRequest {
        query: "авто".to_string(),
        image: None,
        top_k: 5,
        min_score: 0.1,
        system_prompt: Some(cyrillic_prompt),
        filter_uncited: false,
        allow_empty_context: true,
        temperature: None,
        max_tokens: None,
    };
    let result = pipeline.answer(request).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_filter_uncited_sources() {
    let dim = 4;
    let store = Arc::new(InMemoryVectorStore::new(dim));
    store
        .insert(VectorPoint::new(
            "doc_cited",
            vec![1.0, 0.0, 0.0, 0.0],
            json!({"text": "Used in answer"}),
            Modality::Text,
        ))
        .await
        .unwrap();
    store
        .insert(VectorPoint::new(
            "doc_ignored",
            vec![1.0, 0.0, 0.0, 0.0],
            json!({"text": "Irrelevant details"}),
            Modality::Text,
        ))
        .await
        .unwrap();
    let embedder = Arc::new(MockTextEmbedder);
    let mock_llm = Arc::new(MockLlmClient::with_response(
        "Answer relies only on [Source ID: doc_cited].",
    ));
    let pipeline = RagPipeline::new(store, embedder, mock_llm);
    let request = RagQueryRequest {
        query: "Ferrari car info".to_string(),
        image: None,
        top_k: 5,
        min_score: 0.5,
        system_prompt: None,
        filter_uncited: true,
        allow_empty_context: false,
        temperature: None,
        max_tokens: None,
    };
    let response = pipeline.answer(request).await.unwrap();
    assert_eq!(response.sources.len(), 1);
    assert_eq!(response.sources[0].id, "doc_cited");
}

#[tokio::test]
async fn test_rag_pipeline_empty_context_early_return() {
    let dim = 4;
    let store = Arc::new(InMemoryVectorStore::new(dim));
    let embedder = Arc::new(MockTextEmbedder);
    let mock_llm = Arc::new(MockLlmClient::with_response("Should not be called"));
    let pipeline = RagPipeline::new(store, embedder, mock_llm);
    let request = RagQueryRequest {
        query: "Unknown topic".to_string(),
        image: None,
        top_k: 5,
        min_score: 0.9,
        system_prompt: None,
        filter_uncited: false,
        allow_empty_context: false,
        temperature: None,
        max_tokens: None,
    };
    let response = pipeline.answer(request).await.unwrap();
    assert!(response.answer.contains("not found in the knowledge base"));
    assert!(response.sources.is_empty());
    assert!(response.visuals.is_empty());
}

#[tokio::test]
async fn test_rag_pipeline_prompt_too_long() {
    let dim = 4;
    let store = Arc::new(InMemoryVectorStore::new(dim));
    let embedder = Arc::new(MockTextEmbedder);
    let mock_llm = Arc::new(MockLlmClient::new());
    let pipeline = RagPipeline::new(store, embedder, mock_llm);
    let huge_prompt = "x".repeat(MAX_PROMPT_CHARS + 100);
    let request = RagQueryRequest {
        query: "car".to_string(),
        image: None,
        top_k: 5,
        min_score: 0.1,
        system_prompt: Some(huge_prompt),
        filter_uncited: false,
        allow_empty_context: true,
        temperature: None,
        max_tokens: None,
    };
    let result = pipeline.answer(request).await;
    assert!(matches!(
        result,
        Err(RagError::Generation(GenerationError::PromptTooLong { .. }))
    ));
}

#[tokio::test]
async fn test_rag_pipeline_rejects_empty_query() {
    let dim = 4;
    let store = Arc::new(InMemoryVectorStore::new(dim));
    let embedder = Arc::new(MockTextEmbedder);
    let mock_llm = Arc::new(MockLlmClient::new());
    let pipeline = RagPipeline::new(store, embedder, mock_llm);
    let request = RagQueryRequest {
        query: "   ".to_string(),
        image: None,
        top_k: 5,
        min_score: 0.2,
        system_prompt: None,
        filter_uncited: false,
        allow_empty_context: false,
        temperature: None,
        max_tokens: None,
    };
    let result = pipeline.answer(request).await;
    assert!(matches!(result, Err(RagError::ValidationError(_))));
}
