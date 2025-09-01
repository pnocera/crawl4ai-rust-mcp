use std::sync::Arc;
use vector_store::{
    CollectionConfig, HybridSearchEngine, IndexingPipeline, PagePoint,
    PageVectors, QdrantConfig, QueryVectors, SearchQuery, VectorStore,
    VectorWeights, RecipRankFusion, ResultDiversifier,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    // Configure Qdrant with binary quantization
    let qdrant_config = QdrantConfig {
        url: "http://localhost:6333".to_string(),
        api_key: None,
        use_binary_quantization: true,
        binary_quantization_threshold: Some(0.95),
        oversampling: 3.0,
        rescore: true,
    };
    
    // Configure collection with multi-vector schema
    let collection_config = CollectionConfig {
        name: "example_pages".to_string(),
        ..Default::default()
    };
    
    // Create vector store
    let vector_store = Arc::new(
        VectorStore::new(qdrant_config, collection_config).await?
    );
    
    println!("Vector store initialized successfully!");
    
    // Example 1: Index some pages
    index_example_pages(vector_store.clone()).await?;
    
    // Example 2: Perform hybrid search
    search_example(vector_store.clone()).await?;
    
    // Example 3: Demonstrate binary quantization benefits
    demonstrate_quantization().await?;
    
    Ok(())
}

async fn index_example_pages(vector_store: Arc<VectorStore>) -> anyhow::Result<()> {
    println!("\n=== Indexing Example Pages ===");
    
    let pipeline = IndexingPipeline::new(vector_store)
        .with_batch_size(50)
        .with_max_concurrent(4);
    
    // Create sample pages
    let pages = vec![
        (
            PagePoint::new(
                "https://rust-lang.org".to_string(),
                "The Rust Programming Language".to_string(),
                "rust-lang.org".to_string(),
                "A language empowering everyone to build reliable software".to_string(),
            )
            .with_metadata("language".to_string(), serde_json::json!("en"))
            .with_metadata("category".to_string(), serde_json::json!("programming")),
            PageVectors {
                title_vector: generate_mock_embedding(768, 1),
                content_vector: generate_mock_embedding(768, 2),
                code_vector: Some(generate_mock_embedding(768, 3)),
            },
        ),
        (
            PagePoint::new(
                "https://docs.rs".to_string(),
                "Docs.rs - Rust Package Documentation".to_string(),
                "docs.rs".to_string(),
                "Documentation for all Rust crates published to crates.io".to_string(),
            )
            .with_metadata("language".to_string(), serde_json::json!("en"))
            .with_metadata("category".to_string(), serde_json::json!("documentation")),
            PageVectors {
                title_vector: generate_mock_embedding(768, 4),
                content_vector: generate_mock_embedding(768, 5),
                code_vector: None,
            },
        ),
        (
            PagePoint::new(
                "https://crates.io".to_string(),
                "crates.io: Rust Package Registry".to_string(),
                "crates.io".to_string(),
                "The Rust community's crate registry".to_string(),
            )
            .with_metadata("language".to_string(), serde_json::json!("en"))
            .with_metadata("category".to_string(), serde_json::json!("registry")),
            PageVectors {
                title_vector: generate_mock_embedding(768, 6),
                content_vector: generate_mock_embedding(768, 7),
                code_vector: None,
            },
        ),
    ];
    
    let stats = pipeline.index_pages(pages).await?;
    
    println!("Indexed {} pages successfully", stats.successful);
    println!("Failed: {}", stats.failed);
    
    Ok(())
}

async fn search_example(vector_store: Arc<VectorStore>) -> anyhow::Result<()> {
    println!("\n=== Search Example ===");
    
    // Create hybrid search engine
    let search_engine = HybridSearchEngine::new(vector_store.clone())
        .with_weights(0.3, 0.7) // 30% BM25, 70% vector
        .with_reranking(true);
    
    // Create search query
    let query = SearchQuery {
        query_text: "rust documentation".to_string(),
        query_vectors: QueryVectors {
            title_vector: Some(generate_mock_embedding(768, 10)),
            content_vector: Some(generate_mock_embedding(768, 11)),
            code_vector: None,
            weights: VectorWeights {
                title: 0.4,
                content: 0.6,
                code: 0.0,
            },
        },
        filter: None,
        limit: 10,
        offset: None,
        with_payload: true,
        with_vectors: false,
        score_threshold: Some(0.5),
    };
    
    let results = search_engine.search(query).await?;
    
    println!("Found {} results:", results.len());
    for (i, result) in results.iter().enumerate() {
        if let Some(payload) = &result.payload {
            println!(
                "{}. {} (score: {:.3})",
                i + 1,
                payload.title,
                result.score
            );
            println!("   URL: {}", payload.url);
            println!("   Preview: {}", payload.content_preview);
        }
    }
    
    // Demonstrate result diversification
    let diversifier = ResultDiversifier::new();
    let diverse_results = diversifier.diversify(results);
    
    println!("\nDiversified results: {} items", diverse_results.len());
    
    Ok(())
}

async fn demonstrate_quantization() -> anyhow::Result<()> {
    use vector_store::{quantize_vector, calculate_binary_similarity};
    
    println!("\n=== Binary Quantization Demo ===");
    
    let dimension = 768;
    let vector: Vec<f32> = (0..dimension)
        .map(|i| ((i as f32 / dimension as f32) * 2.0 - 1.0).sin())
        .collect();
    
    // Original size
    let original_size = vector.len() * std::mem::size_of::<f32>();
    println!("Original vector size: {} bytes", original_size);
    
    // Quantized size
    let quantized = quantize_vector(&vector);
    let quantized_size = quantized.len();
    println!("Quantized vector size: {} bytes", quantized_size);
    
    // Compression ratio
    let compression_ratio = original_size as f32 / quantized_size as f32;
    println!("Compression ratio: {:.1}x", compression_ratio);
    
    // Compare similarity calculation speed
    let vector2: Vec<f32> = (0..dimension)
        .map(|i| ((i as f32 / dimension as f32) * 2.0 - 1.0).cos())
        .collect();
    let quantized2 = quantize_vector(&vector2);
    
    // Measure binary similarity calculation
    let start = std::time::Instant::now();
    for _ in 0..1000 {
        let _ = calculate_binary_similarity(&quantized, &quantized2);
    }
    let binary_time = start.elapsed();
    
    // Measure cosine similarity calculation (for comparison)
    let start = std::time::Instant::now();
    for _ in 0..1000 {
        let _ = cosine_similarity(&vector, &vector2);
    }
    let cosine_time = start.elapsed();
    
    println!(
        "Binary similarity: {:.2}μs per calculation",
        binary_time.as_micros() as f32 / 1000.0
    );
    println!(
        "Cosine similarity: {:.2}μs per calculation",
        cosine_time.as_micros() as f32 / 1000.0
    );
    println!(
        "Speedup: {:.1}x",
        cosine_time.as_micros() as f32 / binary_time.as_micros() as f32
    );
    
    Ok(())
}

fn generate_mock_embedding(dimension: usize, seed: u64) -> Vec<f32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let mut hasher = DefaultHasher::new();
    seed.hash(&mut hasher);
    let hash = hasher.finish();
    
    (0..dimension)
        .map(|i| {
            let val = ((hash.wrapping_add(i as u64) % 1000) as f32 / 1000.0) * 2.0 - 1.0;
            val / (val * val + 1.0).sqrt() // Normalize
        })
        .collect()
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    dot / (norm_a * norm_b)
}