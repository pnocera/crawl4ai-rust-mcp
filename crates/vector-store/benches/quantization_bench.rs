use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use vector_store::{quantize_vector, dequantize_vector, calculate_binary_similarity};

fn bench_quantization(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantization");
    
    for dimension in [128, 768, 1536, 3072].iter() {
        let vector: Vec<f32> = (0..*dimension)
            .map(|i| (i as f32 / *dimension as f32) * 2.0 - 1.0)
            .collect();
        
        group.bench_with_input(
            BenchmarkId::new("quantize", dimension),
            dimension,
            |b, _| {
                b.iter(|| {
                    quantize_vector(black_box(&vector))
                });
            },
        );
        
        let quantized = quantize_vector(&vector);
        
        group.bench_with_input(
            BenchmarkId::new("dequantize", dimension),
            dimension,
            |b, &dim| {
                b.iter(|| {
                    dequantize_vector(black_box(&quantized), black_box(dim))
                });
            },
        );
    }
    
    group.finish();
}

fn bench_binary_similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("binary_similarity");
    
    for bytes in [16, 96, 192, 384].iter() {
        let a: Vec<u8> = (0..*bytes).map(|i| i as u8).collect();
        let b: Vec<u8> = (0..*bytes).map(|i| (i * 2) as u8).collect();
        
        group.bench_with_input(
            BenchmarkId::new("hamming", bytes),
            bytes,
            |bench, _| {
                bench.iter(|| {
                    calculate_binary_similarity(black_box(&a), black_box(&b))
                });
            },
        );
    }
    
    group.finish();
}

fn bench_memory_usage(c: &mut Criterion) {
    c.bench_function("memory_comparison_768d", |b| {
        b.iter(|| {
            // Original vector: 768 * 4 bytes = 3072 bytes
            let original = vec![0.0f32; 768];
            let original_size = std::mem::size_of_val(&original[..]);
            
            // Quantized vector: 768 / 8 = 96 bytes
            let quantized = quantize_vector(&original);
            let quantized_size = std::mem::size_of_val(&quantized[..]);
            
            // Return compression ratio
            black_box(original_size as f32 / quantized_size as f32)
        });
    });
}

criterion_group!(benches, bench_quantization, bench_binary_similarity, bench_memory_usage);
criterion_main!(benches);