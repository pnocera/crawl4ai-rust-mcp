use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use rayon::prelude::*;
use tokio::sync::{Semaphore, RwLock};
use tracing::{info, debug, warn};

use crate::simd_ops::SimdVectorOps;
use crate::{Result, SearchError};

/// Parallel SIMD processing system for massive-scale vector operations
pub struct ParallelSimdProcessor {
    simd_ops: SimdVectorOps,
    thread_pool_size: usize,
    batch_size: usize,
    memory_limit: usize,
    semaphore: Arc<Semaphore>,
    processed_count: Arc<AtomicUsize>,
}

/// Configuration for parallel SIMD processing
#[derive(Debug, Clone)]
pub struct ParallelSimdConfig {
    pub vector_dimensions: usize,
    pub thread_pool_size: Option<usize>, // None = use all available cores
    pub batch_size: usize,
    pub memory_limit_mb: usize,
    pub max_concurrent_operations: usize,
}

impl Default for ParallelSimdConfig {
    fn default() -> Self {
        Self {
            vector_dimensions: 384,
            thread_pool_size: None,
            batch_size: 10000,
            memory_limit_mb: 2048, // 2GB
            max_concurrent_operations: num_cpus::get() * 2,
        }
    }
}

/// Parallel similarity matrix computation result
#[derive(Debug, Clone)]
pub struct SimilarityMatrix {
    pub dimensions: (usize, usize),
    pub similarities: Vec<Vec<f32>>,
    pub computation_time_ms: f64,
    pub simd_operations_count: usize,
}

/// Large-scale nearest neighbors search result
#[derive(Debug, Clone)]
pub struct MassiveKnnResult {
    pub query_results: Vec<Vec<(usize, f32)>>,
    pub total_comparisons: usize,
    pub processing_time_ms: f64,
    pub peak_memory_usage_mb: usize,
    pub throughput_vectors_per_sec: f64,
}

/// Memory-mapped vector batch for processing large datasets
#[derive(Debug)]
pub struct VectorBatch {
    pub vectors: Vec<Vec<f32>>,
    pub start_index: usize,
    pub end_index: usize,
    pub batch_id: usize,
}

impl ParallelSimdProcessor {
    /// Create a new parallel SIMD processor
    pub fn new(config: ParallelSimdConfig) -> Result<Self> {
        let thread_pool_size = config.thread_pool_size.unwrap_or_else(num_cpus::get);
        let memory_limit = config.memory_limit_mb * 1024 * 1024; // Convert to bytes
        
        // Configure Rayon thread pool
        if let Some(pool_size) = config.thread_pool_size {
            rayon::ThreadPoolBuilder::new()
                .num_threads(pool_size)
                .build_global()
                .map_err(|e| SearchError::SimdError(format!("Failed to initialize thread pool: {}", e)))?;
        }
        
        info!(
            "Initialized parallel SIMD processor - threads: {}, batch_size: {}, memory_limit: {}MB",
            thread_pool_size, config.batch_size, config.memory_limit_mb
        );
        
        Ok(Self {
            simd_ops: SimdVectorOps::new(config.vector_dimensions),
            thread_pool_size,
            batch_size: config.batch_size,
            memory_limit,
            semaphore: Arc::new(Semaphore::new(config.max_concurrent_operations)),
            processed_count: Arc::new(AtomicUsize::new(0)),
        })
    }
    
    /// Compute similarity matrix for all pairs of vectors using parallel SIMD
    pub async fn compute_similarity_matrix(
        &self,
        vectors: &[Vec<f32>],
        use_cosine: bool,
    ) -> Result<SimilarityMatrix> {
        let start_time = Instant::now();
        let n = vectors.len();
        
        if n == 0 {
            return Ok(SimilarityMatrix {
                dimensions: (0, 0),
                similarities: Vec::new(),
                computation_time_ms: 0.0,
                simd_operations_count: 0,
            });
        }
        
        // Estimate memory usage and validate
        let estimated_memory = n * n * std::mem::size_of::<f32>();
        if estimated_memory > self.memory_limit {
            warn!(
                "Similarity matrix would require {}MB, exceeding limit of {}MB",
                estimated_memory / 1024 / 1024,
                self.memory_limit / 1024 / 1024
            );
            return self.compute_chunked_similarity_matrix(vectors, use_cosine).await;
        }
        
        // Initialize result matrix
        let mut similarities = vec![vec![0.0f32; n]; n];
        let simd_ops_count = Arc::new(AtomicUsize::new(0));
        
        // Parallel computation of upper triangle (matrix is symmetric)
        similarities.par_iter_mut().enumerate().for_each(|(i, row)| {
            for j in i..n {
                let similarity = if i == j {
                    1.0 // Self-similarity
                } else {
                    let sim = if use_cosine {
                        self.simd_ops.cosine_similarity(&vectors[i], &vectors[j]).unwrap_or(0.0)
                    } else {
                        let dist = self.simd_ops.euclidean_distance(&vectors[i], &vectors[j]).unwrap_or(f32::INFINITY);
                        1.0 / (1.0 + dist)
                    };
                    simd_ops_count.fetch_add(1, Ordering::Relaxed);
                    sim
                };
                
                row[j] = similarity;
                
                // Fill symmetric position
                if i != j {
                    // Note: This is not thread-safe for the general case, but works for upper triangle computation
                    // In a production implementation, you'd want to handle this differently
                }
            }
        });
        
        // Fill lower triangle (matrix is symmetric)
        for i in 0..n {
            for j in 0..i {
                similarities[i][j] = similarities[j][i];
            }
        }
        
        let elapsed = start_time.elapsed();
        let total_ops = simd_ops_count.load(Ordering::Relaxed);
        
        info!(
            "Computed {}x{} similarity matrix in {:.2}ms using {} SIMD operations",
            n, n, elapsed.as_secs_f64() * 1000.0, total_ops
        );
        
        Ok(SimilarityMatrix {
            dimensions: (n, n),
            similarities,
            computation_time_ms: elapsed.as_secs_f64() * 1000.0,
            simd_operations_count: total_ops,
        })
    }
    
    /// Compute similarity matrix in chunks to handle large datasets
    async fn compute_chunked_similarity_matrix(
        &self,
        vectors: &[Vec<f32>],
        use_cosine: bool,
    ) -> Result<SimilarityMatrix> {
        let start_time = Instant::now();
        let n = vectors.len();
        let chunk_size = (self.memory_limit / (n * std::mem::size_of::<f32>())).min(1000);
        
        info!("Computing chunked similarity matrix with chunk_size: {}", chunk_size);
        
        let mut similarities = vec![vec![0.0f32; n]; n];
        let simd_ops_count = Arc::new(AtomicUsize::new(0));
        
        // Process matrix in chunks
        for chunk_start in (0..n).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(n);
            
            // Acquire semaphore to limit concurrent operations
            let _permit = self.semaphore.acquire().await
                .map_err(|e| SearchError::SimdError(format!("Semaphore error: {}", e)))?;
            
            // Process chunk in parallel
            let chunk_results: Vec<_> = (chunk_start..chunk_end).into_par_iter()
                .map(|i| {
                    let mut row = vec![0.0f32; n];
                    let mut ops_count = 0;
                    
                    for j in 0..n {
                        let similarity = if i == j {
                            1.0
                        } else {
                            let sim = if use_cosine {
                                self.simd_ops.cosine_similarity(&vectors[i], &vectors[j]).unwrap_or(0.0)
                            } else {
                                let dist = self.simd_ops.euclidean_distance(&vectors[i], &vectors[j]).unwrap_or(f32::INFINITY);
                                1.0 / (1.0 + dist)
                            };
                            ops_count += 1;
                            sim
                        };
                        row[j] = similarity;
                    }
                    
                    (i, row, ops_count)
                })
                .collect();
            
            // Store results
            for (i, row, ops_count) in chunk_results {
                similarities[i] = row;
                simd_ops_count.fetch_add(ops_count, Ordering::Relaxed);
            }
            
            debug!("Completed chunk {}-{}/{}", chunk_start, chunk_end, n);
        }
        
        let elapsed = start_time.elapsed();
        let total_ops = simd_ops_count.load(Ordering::Relaxed);
        
        Ok(SimilarityMatrix {
            dimensions: (n, n),
            similarities,
            computation_time_ms: elapsed.as_secs_f64() * 1000.0,
            simd_operations_count: total_ops,
        })
    }
    
    /// Massive-scale k-nearest neighbors search using parallel SIMD processing
    pub async fn massive_knn_search(
        &self,
        query_vectors: &[Vec<f32>],
        database_vectors: &[Vec<f32>],
        k: usize,
        use_cosine: bool,
    ) -> Result<MassiveKnnResult> {
        let start_time = Instant::now();
        let num_queries = query_vectors.len();
        let num_candidates = database_vectors.len();
        
        if num_queries == 0 || num_candidates == 0 || k == 0 {
            return Ok(MassiveKnnResult {
                query_results: Vec::new(),
                total_comparisons: 0,
                processing_time_ms: 0.0,
                peak_memory_usage_mb: 0,
                throughput_vectors_per_sec: 0.0,
            });
        }
        
        info!(
            "Starting massive KNN search: {} queries × {} candidates, k={}",
            num_queries, num_candidates, k
        );
        
        // Convert database vectors to slices for efficient access
        let db_vector_refs: Vec<&[f32]> = database_vectors.iter().map(|v| v.as_slice()).collect();
        
        // Process queries in parallel batches
        let batch_size = self.batch_size.min(num_queries);
        let mut all_results = Vec::with_capacity(num_queries);
        
        for batch_start in (0..num_queries).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(num_queries);
            let query_batch = &query_vectors[batch_start..batch_end];
            
            // Parallel processing within batch
            let batch_results: Result<Vec<Vec<(usize, f32)>>> = query_batch.par_iter()
                .map(|query| {
                    // Use SIMD-accelerated similarity calculations
                    let similarities: Result<Vec<(usize, f32)>> = db_vector_refs.par_iter()
                        .enumerate()
                        .map(|(idx, candidate)| {
                            let similarity = if use_cosine {
                                self.simd_ops.cosine_similarity(query, candidate)?
                            } else {
                                let dist = self.simd_ops.euclidean_distance(query, candidate)?;
                                1.0 / (1.0 + dist)
                            };
                            Ok((idx, similarity))
                        })
                        .collect();
                    
                    let mut sims = similarities?;
                    
                    // Partial sort to get top-k
                    if k >= sims.len() {
                        sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    } else {
                        sims.select_nth_unstable_by(k, |a, b| {
                            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                        });
                        sims.truncate(k);
                        sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    }
                    
                    self.processed_count.fetch_add(num_candidates, Ordering::Relaxed);
                    Ok(sims)
                })
                .collect();
            
            all_results.extend(batch_results?);
            
            debug!(
                "Completed batch {}-{}/{} queries",
                batch_start, batch_end, num_queries
            );
        }
        
        let elapsed = start_time.elapsed();
        let total_comparisons = num_queries * num_candidates;
        let throughput = total_comparisons as f64 / elapsed.as_secs_f64();
        
        // Estimate peak memory usage
        let peak_memory_mb = (
            num_queries * num_candidates * std::mem::size_of::<f32>() + 
            num_queries * k * std::mem::size_of::<(usize, f32)>()
        ) / 1024 / 1024;
        
        info!(
            "Massive KNN search completed: {} comparisons in {:.2}ms, {:.2} vectors/sec",
            total_comparisons, elapsed.as_secs_f64() * 1000.0, throughput
        );
        
        Ok(MassiveKnnResult {
            query_results: all_results,
            total_comparisons,
            processing_time_ms: elapsed.as_secs_f64() * 1000.0,
            peak_memory_usage_mb: peak_memory_mb,
            throughput_vectors_per_sec: throughput,
        })
    }
    
    /// Parallel vector quantization using SIMD-accelerated k-means
    pub async fn parallel_vector_quantization(
        &self,
        vectors: &[Vec<f32>],
        codebook_size: usize,
        subvector_dimensions: usize,
    ) -> Result<ProductQuantizationResult> {
        let start_time = Instant::now();
        let vector_dims = vectors[0].len();
        
        if vector_dims % subvector_dimensions != 0 {
            return Err(SearchError::InvalidDimensions {
                expected: vector_dims,
                actual: subvector_dimensions,
            });
        }
        
        let num_subvectors = vector_dims / subvector_dimensions;
        info!(
            "Starting product quantization: {} vectors, {} subvectors of {} dimensions, codebook size {}",
            vectors.len(), num_subvectors, subvector_dimensions, codebook_size
        );
        
        // Split vectors into subvectors and process in parallel
        let subvector_codebooks: Result<Vec<Vec<Vec<f32>>>> = (0..num_subvectors).into_par_iter()
            .map(|subvector_idx| {
                let start_dim = subvector_idx * subvector_dimensions;
                let end_dim = start_dim + subvector_dimensions;
                
                // Extract subvectors
                let subvectors: Vec<Vec<f32>> = vectors.iter()
                    .map(|v| v[start_dim..end_dim].to_vec())
                    .collect();
                
                // Perform k-means clustering on subvectors
                let config = crate::simd_search::SimdSearchConfig {
                    vector_dimensions: subvector_dimensions,
                    ..Default::default()
                };
                let engine = crate::simd_search::SimdSearchEngine::new(config);
                
                // Use SIMD k-means clustering
                let rt = tokio::runtime::Handle::current();
                let (centroids, _assignments) = rt.block_on(async {
                    engine.simd_kmeans_clustering(&subvectors, codebook_size, 100, 0.001).await
                })?;
                
                Ok(centroids)
            })
            .collect();
        
        let codebooks = subvector_codebooks?;
        
        // Quantize all vectors using the learned codebooks
        let quantized_vectors: Vec<Vec<u8>> = vectors.par_iter()
            .map(|vector| {
                let mut codes = Vec::with_capacity(num_subvectors);
                
                for (subvector_idx, codebook) in codebooks.iter().enumerate() {
                    let start_dim = subvector_idx * subvector_dimensions;
                    let end_dim = start_dim + subvector_dimensions;
                    let subvector = &vector[start_dim..end_dim];
                    
                    // Find closest centroid
                    let mut best_code = 0u8;
                    let mut best_distance = f32::INFINITY;
                    
                    for (code, centroid) in codebook.iter().enumerate() {
                        let distance = self.simd_ops.euclidean_distance(subvector, centroid)
                            .unwrap_or(f32::INFINITY);
                        if distance < best_distance {
                            best_distance = distance;
                            best_code = code as u8;
                        }
                    }
                    
                    codes.push(best_code);
                }
                
                codes
            })
            .collect();
        
        let elapsed = start_time.elapsed();
        
        let vector_count = quantized_vectors.len();
        Ok(ProductQuantizationResult {
            codebooks,
            quantized_vectors,
            subvector_dimensions,
            codebook_size,
            compression_ratio: (vectors.len() * vector_dims * 4) as f32 / 
                              (vector_count * num_subvectors) as f32,
            quantization_time_ms: elapsed.as_secs_f64() * 1000.0,
        })
    }
    
    /// Get processing statistics
    pub fn get_processing_stats(&self) -> ProcessingStats {
        ProcessingStats {
            total_vectors_processed: self.processed_count.load(Ordering::Relaxed),
            thread_pool_size: self.thread_pool_size,
            batch_size: self.batch_size,
            memory_limit_mb: self.memory_limit / 1024 / 1024,
            simd_capabilities: self.simd_ops.use_avx512 || self.simd_ops.use_avx2,
        }
    }
}

/// Product quantization result
#[derive(Debug, Clone)]
pub struct ProductQuantizationResult {
    pub codebooks: Vec<Vec<Vec<f32>>>, // [subvector_id][centroid_id][dimension]
    pub quantized_vectors: Vec<Vec<u8>>, // [vector_id][subvector_code]
    pub subvector_dimensions: usize,
    pub codebook_size: usize,
    pub compression_ratio: f32,
    pub quantization_time_ms: f64,
}

/// Processing statistics
#[derive(Debug, Clone)]
pub struct ProcessingStats {
    pub total_vectors_processed: usize,
    pub thread_pool_size: usize,
    pub batch_size: usize,
    pub memory_limit_mb: usize,
    pub simd_capabilities: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_similarity_matrix_computation() {
        let config = ParallelSimdConfig {
            vector_dimensions: 4,
            batch_size: 2,
            ..Default::default()
        };
        
        let processor = ParallelSimdProcessor::new(config).unwrap();
        
        let vectors = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![1.0, 1.0, 0.0, 0.0],
        ];
        
        let matrix = processor.compute_similarity_matrix(&vectors, true).await.unwrap();
        
        assert_eq!(matrix.dimensions, (3, 3));
        assert_eq!(matrix.similarities[0][0], 1.0); // Self-similarity
        assert_eq!(matrix.similarities[1][1], 1.0);
        assert_eq!(matrix.similarities[2][2], 1.0);
        
        // Check symmetry
        for i in 0..3 {
            for j in 0..3 {
                assert!((matrix.similarities[i][j] - matrix.similarities[j][i]).abs() < 1e-6);
            }
        }
    }
    
    #[tokio::test]
    async fn test_massive_knn_search() {
        let config = ParallelSimdConfig {
            vector_dimensions: 3,
            batch_size: 2,
            ..Default::default()
        };
        
        let processor = ParallelSimdProcessor::new(config).unwrap();
        
        let queries = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
        ];
        
        let database = vec![
            vec![1.0, 0.0, 0.0], // Should match first query
            vec![0.0, 1.0, 0.0], // Should match second query
            vec![0.0, 0.0, 1.0], // Different from both
        ];
        
        let result = processor.massive_knn_search(&queries, &database, 2, true).await.unwrap();
        
        assert_eq!(result.query_results.len(), 2);
        assert_eq!(result.query_results[0].len(), 2); // k=2
        assert_eq!(result.query_results[1].len(), 2);
        assert_eq!(result.total_comparisons, 6); // 2 queries × 3 database vectors
        
        // First query should have database[0] as top match
        assert_eq!(result.query_results[0][0].0, 0);
        // Second query should have database[1] as top match
        assert_eq!(result.query_results[1][0].0, 1);
    }
}