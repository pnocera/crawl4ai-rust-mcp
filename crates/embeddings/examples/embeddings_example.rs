use embeddings::{
    EmbeddingConfig, EmbeddingService, ModelPresets, TextChunker,
    DeviceConfig, PoolingStrategy, format_for_e5,
};
use std::time::Instant;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    println!("=== Candle Embeddings Example ===\n");
    
    // Example 1: Basic embedding generation
    basic_embedding_example().await?;
    
    // Example 2: Different models comparison
    model_comparison_example().await?;
    
    // Example 3: Chunking and batch processing
    chunking_example().await?;
    
    // Example 4: Performance benchmarking
    performance_example().await?;
    
    // Example 5: GPU vs CPU comparison
    device_comparison_example().await?;
    
    Ok(())
}

async fn basic_embedding_example() -> anyhow::Result<()> {
    println!("1. Basic Embedding Generation");
    println!("-----------------------------");
    
    // Use MiniLM model (fast and efficient)
    let config = ModelPresets::minilm_l6_v2()
        .with_cache_dir("./models");
    
    let service = EmbeddingService::with_config(config).await?;
    
    let text = "Rust is a systems programming language focused on safety, speed, and concurrency.";
    println!("Input text: {}", text);
    
    let start = Instant::now();
    let embedding = service.embed_text(text.to_string()).await?;
    let duration = start.elapsed();
    
    println!("Embedding dimensions: {}", embedding.len());
    println!("Generation time: {:?}", duration);
    println!("First 5 values: {:?}", &embedding[..5]);
    
    // Calculate norm to verify normalization
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!("L2 norm: {:.6} (should be ~1.0 if normalized)", norm);
    
    println!();
    Ok(())
}

async fn model_comparison_example() -> anyhow::Result<()> {
    println!("2. Model Comparison");
    println!("-------------------");
    
    let models = vec![
        ("MiniLM-L6-v2", ModelPresets::minilm_l6_v2()),
        ("MiniLM-L12-v2", ModelPresets::minilm_l12_v2()),
        ("E5-small-v2", ModelPresets::e5_small_v2()),
    ];
    
    let test_text = "Natural language processing with transformer models";
    
    for (name, config) in models {
        println!("\nModel: {}", name);
        println!("Hidden size: {}", match name {
            "MiniLM-L6-v2" => 384,
            "MiniLM-L12-v2" => 384,
            "E5-small-v2" => 384,
            _ => 768,
        });
        
        // For E5, format the input appropriately
        let input = if name.contains("E5") {
            format_for_e5(test_text, true)
        } else {
            test_text.to_string()
        };
        
        // Skip actual model loading in example
        println!("Input: {}", input);
        println!("Max sequence length: {}", config.max_sequence_length);
        println!("Pooling strategy: {:?}", config.pooling_strategy);
    }
    
    Ok(())
}

async fn chunking_example() -> anyhow::Result<()> {
    println!("\n3. Text Chunking and Batch Processing");
    println!("-------------------------------------");
    
    let long_text = "Rust provides zero-cost abstractions, move semantics, guaranteed memory safety, \
                     threads without data races, trait-based generics, pattern matching, type inference, \
                     minimal runtime, and efficient C bindings. The Rust compiler enforces memory safety \
                     and thread safety, enabling you to eliminate many classes of bugs at compile-time. \
                     Rust has great documentation, a friendly compiler with useful error messages, \
                     and top-notch tooling including an integrated package manager and build tool, \
                     smart multi-editor support with auto-completion and type inspections.";
    
    let chunker = TextChunker::new(100, 20);
    let chunks = chunker.chunk_text(long_text);
    
    println!("Original text length: {} chars", long_text.len());
    println!("Number of chunks: {}", chunks.len());
    
    for (i, chunk) in chunks.iter().enumerate() {
        println!("\nChunk {}: {} chars", i + 1, chunk.len());
        println!("Content: {}...", &chunk[..chunk.len().min(50)]);
    }
    
    // Simulate batch embedding
    println!("\nSimulating batch embedding of {} chunks...", chunks.len());
    
    // In real usage:
    // let service = EmbeddingService::new().await?;
    // let embeddings = service.embed_texts(chunks).await?;
    
    Ok(())
}

async fn performance_example() -> anyhow::Result<()> {
    println!("\n4. Performance Benchmarking");
    println!("---------------------------");
    
    let batch_sizes = vec![1, 8, 16, 32];
    let text = "Benchmark text for performance testing";
    
    for batch_size in batch_sizes {
        let texts: Vec<String> = vec![text.to_string(); batch_size];
        
        println!("\nBatch size: {}", batch_size);
        
        // Calculate theoretical memory usage
        let input_memory = batch_size * 512 * 4; // max_seq_len * sizeof(f32)
        let output_memory = batch_size * 384 * 4; // hidden_size * sizeof(f32)
        let total_memory = (input_memory + output_memory) / 1024;
        
        println!("Estimated memory usage: {} KB", total_memory);
        println!("Throughput: ~{} texts/second (estimated)", batch_size * 10);
    }
    
    Ok(())
}

async fn device_comparison_example() -> anyhow::Result<()> {
    println!("\n5. Device Comparison");
    println!("--------------------");
    
    let devices = vec![
        ("CPU", DeviceConfig::Cpu),
        ("CUDA GPU 0", DeviceConfig::Cuda(0)),
        ("Metal (Apple Silicon)", DeviceConfig::Metal),
        ("Auto-select", DeviceConfig::Auto),
    ];
    
    for (name, device) in devices {
        println!("\nDevice: {}", name);
        
        let config = ModelPresets::minilm_l6_v2()
            .with_device(device.clone());
        
        println!("Configuration: {:?}", device);
        
        // Show expected performance characteristics
        match device {
            DeviceConfig::Cpu => {
                println!("Expected: Slower but works everywhere");
                println!("Best for: Small batches, CPU-only systems");
            }
            DeviceConfig::Cuda(_) => {
                println!("Expected: Fast inference with NVIDIA GPU");
                println!("Best for: Large batches, production workloads");
            }
            DeviceConfig::Metal => {
                println!("Expected: Hardware acceleration on Apple Silicon");
                println!("Best for: Mac development and deployment");
            }
            DeviceConfig::Auto => {
                println!("Expected: Automatically selects best available device");
                println!("Best for: Portable code across different systems");
            }
        }
    }
    
    // Show memory savings with quantization
    println!("\n6. Memory Optimization");
    println!("----------------------");
    
    let embedding_dim = 384;
    let num_embeddings = 1_000_000;
    
    let f32_memory = (embedding_dim * num_embeddings * 4) / (1024 * 1024);
    let binary_memory = (embedding_dim * num_embeddings) / 8 / (1024 * 1024);
    
    println!("Storage for {} embeddings:", num_embeddings);
    println!("- Float32: {} MB", f32_memory);
    println!("- Binary quantized: {} MB", binary_memory);
    println!("- Compression ratio: {:.1}x", f32_memory as f32 / binary_memory as f32);
    
    Ok(())
}