use embeddings::{
    EmbeddingConfig, EmbeddingService, ModelPresets, TextChunker,
    TextPreprocessor, format_for_e5, format_for_bge,
};

#[tokio::test]
#[ignore] // Requires model download
async fn test_embedding_service_creation() {
    let config = ModelPresets::minilm_l6_v2()
        .with_cache_dir("./test_models");
    
    let service = EmbeddingService::with_config(config).await;
    assert!(service.is_ok());
}

#[tokio::test]
#[ignore] // Requires model download
async fn test_single_embedding() {
    let config = ModelPresets::minilm_l6_v2()
        .with_cache_dir("./test_models")
        .with_batch_size(1);
    
    let service = EmbeddingService::with_config(config).await.unwrap();
    
    let text = "This is a test sentence for embedding generation.".to_string();
    let embedding = service.embed_text(text).await.unwrap();
    
    // MiniLM-L6-v2 has 384 dimensions
    assert_eq!(embedding.len(), 384);
    
    // Check if normalized
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 0.01);
}

#[tokio::test]
#[ignore] // Requires model download
async fn test_batch_embedding() {
    let config = ModelPresets::minilm_l6_v2()
        .with_cache_dir("./test_models");
    
    let service = EmbeddingService::with_config(config).await.unwrap();
    
    let texts = vec![
        "First test sentence".to_string(),
        "Second test sentence".to_string(),
        "Third test sentence".to_string(),
    ];
    
    let embeddings = service.embed_texts(texts).await.unwrap();
    
    assert_eq!(embeddings.len(), 3);
    for embedding in embeddings {
        assert_eq!(embedding.len(), 384);
    }
}

#[test]
fn test_text_chunker_basic() {
    let chunker = TextChunker::new(50, 10);
    let text = "The quick brown fox jumps over the lazy dog. This is a test sentence that should be chunked properly.";
    
    let chunks = chunker.chunk_text(text);
    
    // Should create multiple chunks
    assert!(chunks.len() > 1);
    
    // Each chunk should be around the chunk size
    for chunk in &chunks {
        assert!(chunk.len() <= 60); // Some buffer for word boundaries
    }
}

#[test]
fn test_text_preprocessor() {
    let preprocessor = TextPreprocessor::new()
        .with_lowercase(true)
        .with_strip_accents(true);
    
    let text = "  HELLO Wörld! Café  ";
    let processed = preprocessor.process(text);
    
    // Should lowercase and strip accents
    assert_eq!(processed, "hello world! cafe");
}

#[test]
fn test_model_presets() {
    let minilm = ModelPresets::minilm_l6_v2();
    assert_eq!(minilm.model_id, "sentence-transformers/all-MiniLM-L6-v2");
    assert_eq!(minilm.max_sequence_length, 256);
    
    let mpnet = ModelPresets::mpnet_base_v2();
    assert_eq!(mpnet.model_id, "sentence-transformers/all-mpnet-base-v2");
    assert_eq!(mpnet.max_sequence_length, 384);
    
    let e5 = ModelPresets::e5_small_v2();
    assert_eq!(e5.model_id, "intfloat/e5-small-v2");
    assert_eq!(e5.max_sequence_length, 512);
}

#[test]
fn test_instruction_formatting() {
    let query = "machine learning algorithms";
    let passage = "Introduction to neural networks and deep learning";
    
    // E5 formatting
    let e5_query = format_for_e5(query, true);
    assert_eq!(e5_query, "query: machine learning algorithms");
    
    let e5_passage = format_for_e5(passage, false);
    assert_eq!(e5_passage, "passage: Introduction to neural networks and deep learning");
    
    // BGE formatting
    let bge_query = format_for_bge(query, true);
    assert!(bge_query.starts_with("Represent this sentence"));
    
    let bge_passage = format_for_bge(passage, false);
    assert_eq!(bge_passage, passage);
}

#[tokio::test]
async fn test_model_info() {
    let config = EmbeddingConfig::new("test-model")
        .with_batch_size(64)
        .with_device(embeddings::DeviceConfig::Auto);
    
    // Skip actual service creation for this test
    // Just verify config values
    assert_eq!(config.model_id, "test-model");
    assert_eq!(config.max_batch_size, 64);
}

#[test]
fn test_text_chunker_overlap() {
    let chunker = TextChunker::new(40, 15);
    let text = "one two three four five six seven eight nine ten eleven twelve";
    
    let chunks = chunker.chunk_text(text);
    
    // Verify overlap exists
    if chunks.len() > 1 {
        for i in 1..chunks.len() {
            let prev_words: Vec<&str> = chunks[i-1].split(' ').collect();
            let curr_words: Vec<&str> = chunks[i].split(' ').collect();
            
            // Count overlapping words
            let overlap_count = prev_words.iter()
                .rev()
                .zip(curr_words.iter())
                .filter(|(a, b)| a == b)
                .count();
            
            assert!(overlap_count > 0);
        }
    }
}