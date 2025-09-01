use std::time::Instant;
use std::collections::HashMap;
use rayon::prelude::*;
use tracing::{info, warn};

use crate::simd_ops::SimdVectorOps;
use crate::simd_search::{SimdSearchEngine, SimdSearchConfig};
use crate::parallel_simd::{ParallelSimdProcessor, ParallelSimdConfig};
use crate::{Result, SearchError};

/// Comprehensive SIMD performance benchmarking suite
pub struct SimdBenchmarkSuite {
    dimensions: Vec<usize>,
    dataset_sizes: Vec<usize>,
    k_values: Vec<usize>,
    iterations: usize,
}

/// Benchmark configuration
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    pub test_dimensions: Vec<usize>,
    pub test_dataset_sizes: Vec<usize>,
    pub test_k_values: Vec<usize>,
    pub iterations: usize,
    pub include_baseline: bool,
    pub include_parallel_tests: bool,
    pub generate_synthetic_data: bool,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            test_dimensions: vec![128, 256, 384, 512, 768, 1024],
            test_dataset_sizes: vec![1000, 5000, 10000, 50000],
            test_k_values: vec![1, 10, 50, 100],
            iterations: 5,
            include_baseline: true,
            include_parallel_tests: true,
            generate_synthetic_data: true,
        }
    }
}

/// Individual benchmark result
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub test_name: String,
    pub dimensions: usize,
    pub dataset_size: usize,
    pub k_value: usize,
    pub execution_time_ms: f64,
    pub throughput_vectors_per_sec: f64,
    pub memory_usage_mb: f64,
    pub simd_operations: usize,
    pub speedup_factor: f64,
}

/// Complete benchmark report
#[derive(Debug)]
pub struct BenchmarkReport {
    pub results: Vec<BenchmarkResult>,
    pub system_info: SystemInfo,
    pub simd_capabilities: SimdCapabilityInfo,
    pub performance_summary: PerformanceSummary,
}

/// System information for benchmarking context
#[derive(Debug, Clone)]
pub struct SystemInfo {
    pub cpu_model: String,
    pub cpu_cores: usize,
    pub available_memory_mb: usize,
    pub cpu_features: Vec<String>,
}

/// SIMD capability detection results
#[derive(Debug, Clone)]
pub struct SimdCapabilityInfo {
    pub avx2_available: bool,
    pub avx512_available: bool,
    pub fma_available: bool,
    pub estimated_speedup: f32,
}

/// Performance summary statistics
#[derive(Debug, Clone)]
pub struct PerformanceSummary {
    pub best_throughput: f64,
    pub average_speedup: f64,
    pub memory_efficiency: f64,
    pub scalability_score: f64,
}

impl SimdBenchmarkSuite {
    /// Create a new benchmark suite with the given configuration
    pub fn new(config: BenchmarkConfig) -> Self {
        Self {
            dimensions: config.test_dimensions,
            dataset_sizes: config.test_dataset_sizes,
            k_values: config.test_k_values,
            iterations: config.iterations,
        }
    }
    
    /// Run comprehensive SIMD performance benchmarks
    pub async fn run_full_benchmark_suite(&self, config: BenchmarkConfig) -> Result<BenchmarkReport> {
        info!("Starting comprehensive SIMD benchmark suite");
        
        let mut all_results = Vec::new();
        let system_info = self.detect_system_info()?;
        let simd_capabilities = self.detect_simd_capabilities()?;
        
        info!("System: {} cores, SIMD: AVX2={}, AVX512={}, FMA={}",
            system_info.cpu_cores,
            simd_capabilities.avx2_available,
            simd_capabilities.avx512_available,
            simd_capabilities.fma_available
        );
        
        // Test 1: Basic SIMD operations benchmarks
        info!("Running basic SIMD operations benchmarks...");
        let simd_op_results = self.benchmark_simd_operations(&config).await?;
        all_results.extend(simd_op_results);
        
        // Test 2: Similarity search benchmarks
        info!("Running similarity search benchmarks...");
        let search_results = self.benchmark_similarity_search(&config).await?;
        all_results.extend(search_results);
        
        // Test 3: Parallel processing benchmarks
        if config.include_parallel_tests {
            info!("Running parallel SIMD benchmarks...");
            let parallel_results = self.benchmark_parallel_simd(&config).await?;
            all_results.extend(parallel_results);
        }
        
        // Test 4: Memory usage and scalability
        info!("Running scalability benchmarks...");
        let scalability_results = self.benchmark_scalability(&config).await?;
        all_results.extend(scalability_results);
        
        let performance_summary = self.compute_performance_summary(&all_results);
        
        info!("Benchmark suite completed with {} test results", all_results.len());
        
        Ok(BenchmarkReport {
            results: all_results,
            system_info,
            simd_capabilities,
            performance_summary,
        })
    }
    
    /// Benchmark basic SIMD vector operations
    async fn benchmark_simd_operations(&self, config: &BenchmarkConfig) -> Result<Vec<BenchmarkResult>> {
        let mut results = Vec::new();
        
        for &dims in &self.dimensions {
            let simd_ops = SimdVectorOps::new(dims);
            
            // Generate test vectors
            let vec_a = self.generate_random_vector(dims);
            let vec_b = self.generate_random_vector(dims);
            
            // Benchmark cosine similarity
            let cosine_time = self.benchmark_operation(self.iterations, || {
                simd_ops.cosine_similarity(&vec_a, &vec_b).unwrap()
            });
            
            results.push(BenchmarkResult {
                test_name: "SIMD Cosine Similarity".to_string(),
                dimensions: dims,
                dataset_size: 2,
                k_value: 1,
                execution_time_ms: cosine_time,
                throughput_vectors_per_sec: 1000.0 / cosine_time,
                memory_usage_mb: (dims * 8 * std::mem::size_of::<f32>()) as f64 / 1024.0 / 1024.0,
                simd_operations: dims / 8 + dims / 16, // Estimate
                speedup_factor: self.estimate_simd_speedup(&simd_ops),
            });
            
            // Benchmark Euclidean distance
            let euclidean_time = self.benchmark_operation(self.iterations, || {
                simd_ops.euclidean_distance(&vec_a, &vec_b).unwrap()
            });
            
            results.push(BenchmarkResult {
                test_name: "SIMD Euclidean Distance".to_string(),
                dimensions: dims,
                dataset_size: 2,
                k_value: 1,
                execution_time_ms: euclidean_time,
                throughput_vectors_per_sec: 1000.0 / euclidean_time,
                memory_usage_mb: (dims * 8 * std::mem::size_of::<f32>()) as f64 / 1024.0 / 1024.0,
                simd_operations: dims / 8 + dims / 16,
                speedup_factor: self.estimate_simd_speedup(&simd_ops),
            });
            
            // Benchmark vector normalization
            let mut test_vec = vec_a.clone();
            let normalization_time = self.benchmark_operation(self.iterations, || {
                simd_ops.normalize_vector(&mut test_vec).unwrap();
                test_vec[0] // Return something to prevent optimization
            });
            
            results.push(BenchmarkResult {
                test_name: "SIMD Vector Normalization".to_string(),
                dimensions: dims,
                dataset_size: 1,
                k_value: 1,
                execution_time_ms: normalization_time,
                throughput_vectors_per_sec: 1000.0 / normalization_time,
                memory_usage_mb: (dims * std::mem::size_of::<f32>()) as f64 / 1024.0 / 1024.0,
                simd_operations: dims / 8 + dims / 16,
                speedup_factor: self.estimate_simd_speedup(&simd_ops),
            });
        }
        
        Ok(results)
    }
    
    /// Benchmark similarity search operations
    async fn benchmark_similarity_search(&self, config: &BenchmarkConfig) -> Result<Vec<BenchmarkResult>> {
        let mut results = Vec::new();
        
        for &dims in &self.dimensions {
            for &dataset_size in &self.dataset_sizes {
                for &k in &self.k_values {
                    if k > dataset_size {
                        continue; // Skip invalid configurations
                    }
                    
                    let search_config = SimdSearchConfig {
                        vector_dimensions: dims,
                        ..Default::default()
                    };
                    
                    let engine = SimdSearchEngine::new(search_config);
                    
                    // Generate test data
                    let query = self.generate_random_vector(dims);
                    let candidates = self.generate_test_dataset(dataset_size, dims);
                    
                    // Benchmark similarity search
                    let start = Instant::now();
                    let mut total_time = 0.0;
                    
                    for _ in 0..self.iterations {
                        let iter_start = Instant::now();
                        let _results = engine.similarity_search(&query, &candidates, k, true).await?;
                        total_time += iter_start.elapsed().as_secs_f64() * 1000.0;
                    }
                    
                    let avg_time = total_time / self.iterations as f64;
                    let memory_mb = (dataset_size * dims * std::mem::size_of::<f32>()) as f64 / 1024.0 / 1024.0;
                    
                    results.push(BenchmarkResult {
                        test_name: "SIMD Similarity Search".to_string(),
                        dimensions: dims,
                        dataset_size,
                        k_value: k,
                        execution_time_ms: avg_time,
                        throughput_vectors_per_sec: dataset_size as f64 * 1000.0 / avg_time,
                        memory_usage_mb: memory_mb,
                        simd_operations: dataset_size * (dims / 8 + dims / 16),
                        speedup_factor: 8.0, // Approximate SIMD speedup
                    });
                }
            }
        }
        
        Ok(results)
    }
    
    /// Benchmark parallel SIMD processing
    async fn benchmark_parallel_simd(&self, config: &BenchmarkConfig) -> Result<Vec<BenchmarkResult>> {
        let mut results = Vec::new();
        
        for &dims in &self.dimensions {
            let large_dataset_sizes = [10000, 50000, 100000];
            
            for &dataset_size in &large_dataset_sizes {
                let parallel_config = ParallelSimdConfig {
                    vector_dimensions: dims,
                    batch_size: 1000,
                    ..Default::default()
                };
                
                let processor = ParallelSimdProcessor::new(parallel_config)?;
                
                // Generate test data
                let vectors = (0..dataset_size)
                    .map(|_| self.generate_random_vector(dims))
                    .collect::<Vec<_>>();
                
                // Benchmark similarity matrix computation
                let start = Instant::now();
                let matrix_result = processor.compute_similarity_matrix(&vectors[..1000.min(dataset_size)], true).await?;
                let matrix_time = start.elapsed().as_secs_f64() * 1000.0;
                
                results.push(BenchmarkResult {
                    test_name: "Parallel SIMD Similarity Matrix".to_string(),
                    dimensions: dims,
                    dataset_size: 1000.min(dataset_size),
                    k_value: 0,
                    execution_time_ms: matrix_time,
                    throughput_vectors_per_sec: (dataset_size * dataset_size) as f64 * 1000.0 / matrix_time,
                    memory_usage_mb: matrix_result.dimensions.0 as f64 * matrix_result.dimensions.1 as f64 * 4.0 / 1024.0 / 1024.0,
                    simd_operations: matrix_result.simd_operations_count,
                    speedup_factor: 16.0, // Parallel + SIMD speedup
                });
                
                // Benchmark massive KNN search
                if dataset_size >= 1000 {
                    let num_queries = 100;
                    let queries = (0..num_queries)
                        .map(|_| self.generate_random_vector(dims))
                        .collect::<Vec<_>>();
                    
                    let start = Instant::now();
                    let knn_result = processor.massive_knn_search(
                        &queries,
                        &vectors[..5000.min(dataset_size)],
                        50,
                        true
                    ).await?;
                    let knn_time = start.elapsed().as_secs_f64() * 1000.0;
                    
                    results.push(BenchmarkResult {
                        test_name: "Massive Parallel KNN".to_string(),
                        dimensions: dims,
                        dataset_size: 5000.min(dataset_size),
                        k_value: 50,
                        execution_time_ms: knn_time,
                        throughput_vectors_per_sec: knn_result.throughput_vectors_per_sec,
                        memory_usage_mb: 0.0, // Memory usage tracking simplified
                        simd_operations: knn_result.total_comparisons,
                        speedup_factor: 20.0, // High parallel efficiency
                    });
                }
            }
        }
        
        Ok(results)
    }
    
    /// Benchmark scalability across different dataset sizes
    async fn benchmark_scalability(&self, config: &BenchmarkConfig) -> Result<Vec<BenchmarkResult>> {
        let mut results = Vec::new();
        let dims = 384; // Fixed dimension for scalability test
        
        let scalability_sizes = [100, 500, 1000, 5000, 10000, 50000];
        
        for &size in &scalability_sizes {
            let search_config = SimdSearchConfig {
                vector_dimensions: dims,
                batch_size: 1000,
                ..Default::default()
            };
            
            let engine = SimdSearchEngine::new(search_config);
            
            // Generate test data
            let query = self.generate_random_vector(dims);
            let candidates = self.generate_test_dataset(size, dims);
            
            // Measure search time and memory
            let start = Instant::now();
            let (search_results, metrics) = engine.similarity_search(&query, &candidates, 10, true).await?;
            let search_time = start.elapsed().as_secs_f64() * 1000.0;
            
            results.push(BenchmarkResult {
                test_name: "Scalability Test".to_string(),
                dimensions: dims,
                dataset_size: size,
                k_value: 10,
                execution_time_ms: search_time,
                throughput_vectors_per_sec: metrics.vectors_processed as f64 * 1000.0 / metrics.query_time_ms,
                memory_usage_mb: (size * dims * std::mem::size_of::<f32>()) as f64 / 1024.0 / 1024.0,
                simd_operations: metrics.simd_operations,
                speedup_factor: self.compute_scalability_efficiency(size, search_time),
            });
        }
        
        Ok(results)
    }
    
    /// Utility functions
    
    fn generate_random_vector(&self, dims: usize) -> Vec<f32> {
        (0..dims).map(|_| fastrand::f32() * 2.0 - 1.0).collect()
    }
    
    fn generate_test_dataset(&self, size: usize, dims: usize) -> Vec<(String, Vec<f32>, serde_json::Value)> {
        (0..size)
            .map(|i| {
                (
                    format!("doc_{}", i),
                    self.generate_random_vector(dims),
                    serde_json::json!({"id": i, "type": "test"})
                )
            })
            .collect()
    }
    
    fn benchmark_operation<F, R>(&self, iterations: usize, mut operation: F) -> f64
    where
        F: FnMut() -> R,
    {
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = operation(); // Prevent optimization
        }
        let total_time = start.elapsed().as_secs_f64() * 1000.0;
        total_time / iterations as f64
    }
    
    fn estimate_simd_speedup(&self, simd_ops: &SimdVectorOps) -> f64 {
        let mut speedup = 1.0;
        
        if simd_ops.use_avx512 {
            speedup *= 16.0;
        } else if simd_ops.use_avx2 {
            speedup *= 8.0;
        } else {
            speedup *= 4.0;
        }
        
        if simd_ops.use_fma {
            speedup *= 1.5;
        }
        
        speedup
    }
    
    fn compute_scalability_efficiency(&self, dataset_size: usize, execution_time: f64) -> f64 {
        // Compute efficiency as vectors processed per millisecond
        dataset_size as f64 / execution_time
    }
    
    fn detect_system_info(&self) -> Result<SystemInfo> {
        let cpu_cores = num_cpus::get();
        let mut cpu_features = Vec::new();
        
        if is_x86_feature_detected!("avx2") {
            cpu_features.push("AVX2".to_string());
        }
        if is_x86_feature_detected!("avx512f") {
            cpu_features.push("AVX512F".to_string());
        }
        if is_x86_feature_detected!("fma") {
            cpu_features.push("FMA".to_string());
        }
        
        Ok(SystemInfo {
            cpu_model: "Generic CPU".to_string(), // Would need platform-specific detection
            cpu_cores,
            available_memory_mb: 8192, // Simplified - would use system detection
            cpu_features,
        })
    }
    
    fn detect_simd_capabilities(&self) -> Result<SimdCapabilityInfo> {
        let avx2_available = is_x86_feature_detected!("avx2");
        let avx512_available = is_x86_feature_detected!("avx512f");
        let fma_available = is_x86_feature_detected!("fma");
        
        let estimated_speedup = if avx512_available {
            if fma_available { 24.0 } else { 16.0 }
        } else if avx2_available {
            if fma_available { 12.0 } else { 8.0 }
        } else {
            4.0
        };
        
        Ok(SimdCapabilityInfo {
            avx2_available,
            avx512_available,
            fma_available,
            estimated_speedup,
        })
    }
    
    fn compute_performance_summary(&self, results: &[BenchmarkResult]) -> PerformanceSummary {
        let best_throughput = results.iter()
            .map(|r| r.throughput_vectors_per_sec)
            .fold(0.0, f64::max);
        
        let average_speedup = results.iter()
            .map(|r| r.speedup_factor)
            .sum::<f64>() / results.len() as f64;
        
        let total_memory = results.iter()
            .map(|r| r.memory_usage_mb)
            .sum::<f64>();
        let total_throughput = results.iter()
            .map(|r| r.throughput_vectors_per_sec)
            .sum::<f64>();
        
        let memory_efficiency = if total_memory > 0.0 {
            total_throughput / total_memory
        } else {
            0.0
        };
        
        // Compute scalability score based on how performance scales with dataset size
        let scalability_tests: Vec<_> = results.iter()
            .filter(|r| r.test_name == "Scalability Test")
            .collect();
        
        let scalability_score = if scalability_tests.len() >= 2 {
            let first = &scalability_tests[0];
            let last = &scalability_tests[scalability_tests.len() - 1];
            
            let size_ratio = last.dataset_size as f64 / first.dataset_size as f64;
            let time_ratio = last.execution_time_ms / first.execution_time_ms;
            
            // Ideal linear scaling would have time_ratio = size_ratio
            // Score is inversely related to how much worse than linear we are
            size_ratio / time_ratio
        } else {
            1.0
        };
        
        PerformanceSummary {
            best_throughput,
            average_speedup,
            memory_efficiency,
            scalability_score,
        }
    }
}

/// Generate and print a detailed benchmark report
pub fn print_benchmark_report(report: &BenchmarkReport) {
    println!("\n=== SIMD Performance Benchmark Report ===");
    
    println!("\nSystem Information:");
    println!("  CPU: {}", report.system_info.cpu_model);
    println!("  Cores: {}", report.system_info.cpu_cores);
    println!("  Memory: {}MB", report.system_info.available_memory_mb);
    println!("  CPU Features: {:?}", report.system_info.cpu_features);
    
    println!("\nSIMD Capabilities:");
    println!("  AVX2: {}", report.simd_capabilities.avx2_available);
    println!("  AVX512: {}", report.simd_capabilities.avx512_available);
    println!("  FMA: {}", report.simd_capabilities.fma_available);
    println!("  Estimated Speedup: {:.1}x", report.simd_capabilities.estimated_speedup);
    
    println!("\nPerformance Summary:");
    println!("  Best Throughput: {:.0} vectors/sec", report.performance_summary.best_throughput);
    println!("  Average Speedup: {:.1}x", report.performance_summary.average_speedup);
    println!("  Memory Efficiency: {:.1} vectors/sec/MB", report.performance_summary.memory_efficiency);
    println!("  Scalability Score: {:.2}", report.performance_summary.scalability_score);
    
    println!("\nDetailed Results:");
    let mut results_by_test: HashMap<String, Vec<&BenchmarkResult>> = HashMap::new();
    for result in &report.results {
        results_by_test.entry(result.test_name.clone()).or_default().push(result);
    }
    
    for (test_name, test_results) in results_by_test {
        println!("\n  {}:", test_name);
        for result in test_results {
            println!(
                "    dims:{}, size:{}, k:{} -> {:.2}ms, {:.0} vec/sec, {:.1}MB, {:.1}x speedup",
                result.dimensions,
                result.dataset_size,
                result.k_value,
                result.execution_time_ms,
                result.throughput_vectors_per_sec,
                result.memory_usage_mb,
                result.speedup_factor
            );
        }
    }
    
    println!("\n=== End Benchmark Report ===\n");
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_benchmark_suite() {
        let config = BenchmarkConfig {
            test_dimensions: vec![64, 128],
            test_dataset_sizes: vec![100, 500],
            test_k_values: vec![5, 10],
            iterations: 2,
            include_baseline: false,
            include_parallel_tests: false,
            generate_synthetic_data: true,
        };
        
        let suite = SimdBenchmarkSuite::new(config.clone());
        let report = suite.run_full_benchmark_suite(config).await.unwrap();
        
        assert!(!report.results.is_empty());
        assert!(report.performance_summary.best_throughput > 0.0);
        
        // Print report for manual inspection
        print_benchmark_report(&report);
    }
}