use vector_store::{
    CollectionConfig, PagePoint, PageVectors, QdrantConfig, SearchQuery,
    QueryVectors, VectorStore, VectorWeights, quantize_vector, dequantize_vector,
    calculate_binary_similarity,
};
use std::sync::Arc;
use uuid::Uuid;

#[tokio::test]
#[ignore] // Requires running Qdrant instance
async fn test_vector_store_creation() {
    let config = QdrantConfig::default();
    let collection_config = CollectionConfig::default();
    
    let store = VectorStore::new(config, collection_config).await;
    assert!(store.is_ok());
}

#[test]
fn test_binary_quantization() {
    let vector = vec![0.5, -0.3, 0.8, -0.1, 0.2, -0.7, 0.9, -0.4];
    
    // Quantize
    let quantized = quantize_vector(&vector);
    assert_eq!(quantized.len(), 1); // 8 values fit in 1 byte
    
    // Dequantize
    let dequantized = dequantize_vector(&quantized, 8);
    assert_eq!(dequantized.len(), 8);
    
    // Check signs match
    for (orig, deq) in vector.iter().zip(dequantized.iter()) {
        assert_eq!(orig.signum(), deq.signum());
    }
}

#[test]
fn test_binary_similarity() {
    let a = vec![0b11110000]; // First 4 bits set
    let b = vec![0b11000000]; // First 2 bits set
    
    let similarity = calculate_binary_similarity(&a, &b);
    
    // 6 matching bits out of 8 = 0.75
    assert!((similarity - 0.75).abs() < 0.01);
}

#[test]
fn test_page_point_creation() {
    let point = PagePoint::new(
        "https://example.com".to_string(),
        "Example Page".to_string(),
        "example.com".to_string(),
        "This is an example page content preview".to_string(),
    )
    .with_metadata("category".to_string(), serde_json::json!("tech"))
    .with_metadata("tags".to_string(), serde_json::json!(["rust", "web"]));
    
    assert_eq!(point.url, "https://example.com");
    assert_eq!(point.title, "Example Page");
    assert_eq!(point.domain, "example.com");
    assert_eq!(point.metadata.len(), 2);
}

#[test]
fn test_vector_weights() {
    let weights = VectorWeights::default();
    assert_eq!(weights.title, 0.3);
    assert_eq!(weights.content, 0.5);
    assert_eq!(weights.code, 0.2);
    
    // Sum should be 1.0
    assert!((weights.title + weights.content + weights.code - 1.0).abs() < 0.01);
}

#[tokio::test]
#[ignore] // Requires running Qdrant instance
async fn test_upsert_and_search() {
    let config = QdrantConfig::default();
    let collection_config = CollectionConfig {
        name: "test_collection".to_string(),
        ..Default::default()
    };
    
    let store = Arc::new(
        VectorStore::new(config, collection_config)
            .await
            .unwrap()
    );
    
    // Create test data
    let point = PagePoint::new(
        "https://test.com/page1".to_string(),
        "Test Page 1".to_string(),
        "test.com".to_string(),
        "This is test content for page 1".to_string(),
    );
    
    let vectors = PageVectors {
        title_vector: vec![0.1; 768],
        content_vector: vec![0.2; 768],
        code_vector: None,
    };
    
    // Upsert
    store.upsert_page(point.clone(), vectors.clone()).await.unwrap();
    
    // Search
    let query = SearchQuery {
        query_text: "test".to_string(),
        query_vectors: QueryVectors {
            title_vector: Some(vec![0.1; 768]),
            content_vector: Some(vec![0.2; 768]),
            code_vector: None,
            weights: VectorWeights::default(),
        },
        filter: None,
        limit: 10,
        offset: None,
        with_payload: true,
        with_vectors: false,
        score_threshold: None,
    };
    
    let results = store.search(query).await.unwrap();
    assert!(!results.is_empty());
    assert_eq!(results[0].id, point.id);
}

#[test]
fn test_large_vector_quantization() {
    let dimension = 768;
    let vector: Vec<f32> = (0..dimension)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    
    let quantized = quantize_vector(&vector);
    assert_eq!(quantized.len(), (dimension + 7) / 8); // Ceiling division
    
    let dequantized = dequantize_vector(&quantized, dimension);
    assert_eq!(dequantized.len(), dimension);
    
    // Verify pattern preserved
    for (i, val) in dequantized.iter().enumerate() {
        if i % 2 == 0 {
            assert_eq!(*val, 1.0);
        } else {
            assert_eq!(*val, -1.0);
        }
    }
}

#[test]
fn test_binary_similarity_edge_cases() {
    // Same vectors
    let a = vec![0b11111111];
    let similarity = calculate_binary_similarity(&a, &a);
    assert_eq!(similarity, 1.0);
    
    // Opposite vectors
    let b = vec![0b00000000];
    let similarity = calculate_binary_similarity(&a, &b);
    assert_eq!(similarity, 0.0);
    
    // Empty vectors
    let empty: Vec<u8> = vec![];
    let similarity = calculate_binary_similarity(&empty, &empty);
    assert!(similarity.is_nan() || similarity == 1.0);
}