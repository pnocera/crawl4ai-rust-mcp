use std::sync::Arc;
use std::time::Instant;
use std::collections::HashMap;
use rayon::prelude::*;
use tracing::{info, debug};
use std::sync::RwLock;

use crate::simd_ops::{SimdVectorOps, SimdKnnSearch};
use crate::{Result, SearchError};

/// Advanced SIMD-accelerated similarity search engine
pub struct SimdSearchEngine {
    simd_ops: SimdVectorOps,
    knn_search: SimdKnnSearch,
    vector_dims: usize,
    batch_size: usize,
    cache: Arc<RwLock<SearchCache>>,
}

/// Simple LRU cache for search results
struct SearchCache {
    cache: HashMap<String, (Vec<SimdSearchResult>, SearchMetrics)>,
    access_order: Vec<String>,
    max_size: usize,
}

impl SearchCache {
    fn new(max_size: usize) -> Self {
        Self {
            cache: HashMap::new(),
            access_order: Vec::new(),
            max_size,
        }
    }
    
    fn get(&mut self, key: &str) -> Option<(Vec<SimdSearchResult>, SearchMetrics)> {
        if let Some(value) = self.cache.get(key) {
            // Move to end (most recently used)
            if let Some(pos) = self.access_order.iter().position(|k| k == key) {
                self.access_order.remove(pos);
            }
            self.access_order.push(key.to_string());
            
            Some(value.clone())
        } else {
            None
        }
    }
    
    fn insert(&mut self, key: String, value: (Vec<SimdSearchResult>, SearchMetrics)) {
        // Remove old entry if exists
        if self.cache.contains_key(&key) {
            if let Some(pos) = self.access_order.iter().position(|k| k == &key) {
                self.access_order.remove(pos);
            }
        }
        
        // Evict least recently used if at capacity
        while self.cache.len() >= self.max_size && !self.access_order.is_empty() {
            let lru_key = self.access_order.remove(0);
            self.cache.remove(&lru_key);
        }
        
        // Insert new entry
        self.cache.insert(key.clone(), value);
        self.access_order.push(key);
    }
}

/// Search configuration for SIMD operations
#[derive(Debug, Clone)]
pub struct SimdSearchConfig {
    pub vector_dimensions: usize,
    pub batch_size: usize,
    pub parallel_threshold: usize,
    pub use_cosine_similarity: bool,
    pub precision_threshold: f32,
    pub cache_size: usize,
}

impl Default for SimdSearchConfig {
    fn default() -> Self {
        Self {
            vector_dimensions: 384, // Common embedding dimension
            batch_size: 1000,
            parallel_threshold: 10000, // Use parallel processing for large datasets
            use_cosine_similarity: true,
            precision_threshold: 1e-6,
            cache_size: 1000, // Cache up to 1000 search results
        }
    }
}

/// Search result with SIMD-computed similarity scores
#[derive(Debug, Clone)]
pub struct SimdSearchResult {
    pub id: String,
    pub score: f32,
    pub vector: Vec<f32>,
    pub metadata: serde_json::Value,
}

/// Performance metrics for SIMD search operations
#[derive(Debug, Clone)]
pub struct SearchMetrics {
    pub query_time_ms: f64,
    pub vectors_processed: usize,
    pub simd_operations: usize,
    pub cache_hits: usize,
    pub similarity_calculations: usize,
}

impl SimdSearchEngine {
    /// Create a new SIMD search engine with the given configuration
    pub fn new(config: SimdSearchConfig) -> Self {
        let simd_ops = SimdVectorOps::new(config.vector_dimensions);
        let knn_search = SimdKnnSearch::new(config.vector_dimensions);
        let cache = Arc::new(RwLock::new(SearchCache::new(config.cache_size)));
        
        info!(
            "Initialized SIMD search engine - dims: {}, batch_size: {}, parallel_threshold: {}, cache_size: {}",
            config.vector_dimensions, config.batch_size, config.parallel_threshold, config.cache_size
        );
        
        Self {
            simd_ops,
            knn_search,
            vector_dims: config.vector_dimensions,
            batch_size: config.batch_size,
            cache,
        }
    }
    
    /// Perform high-performance similarity search using SIMD acceleration
    pub async fn similarity_search(
        &self,
        query_vector: &[f32],
        candidate_vectors: &[(String, Vec<f32>, serde_json::Value)],
        top_k: usize,
        use_cosine: bool,
    ) -> Result<(Vec<SimdSearchResult>, SearchMetrics)> {
        let start_time = Instant::now();
        
        if query_vector.len() != self.vector_dims {
            return Err(SearchError::InvalidDimensions {
                expected: self.vector_dims,
                actual: query_vector.len(),
            });
        }
        
        if candidate_vectors.is_empty() {
            return Ok((Vec::new(), SearchMetrics {
                query_time_ms: 0.0,
                vectors_processed: 0,
                simd_operations: 0,
                cache_hits: 0,
                similarity_calculations: 0,
            }));
        }
        
        // Create cache key from query vector and parameters
        let cache_key = self.create_cache_key(query_vector, top_k, use_cosine, candidate_vectors);
        
        // Check cache first
        {
            let mut cache = self.cache.write().unwrap();
            if let Some((results, mut metrics)) = cache.get(&cache_key) {
                metrics.cache_hits = 1;
                metrics.query_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;
                debug!("Cache hit for search query, returning {} results", results.len());
                return Ok((results, metrics));
            }
        }
        
        // Extract vectors for SIMD operations
        let vectors: Vec<&[f32]> = candidate_vectors.iter()
            .map(|(_, vec, _)| vec.as_slice())
            .collect();
        
        // Perform SIMD-accelerated k-nearest neighbors search
        let knn_results = self.knn_search.find_knn(
            query_vector,
            &vectors,
            top_k,
            use_cosine
        )?;
        
        // Convert results back to full search results
        let search_results: Vec<SimdSearchResult> = knn_results.into_iter()
            .map(|(idx, score)| {
                let (id, vector, metadata) = &candidate_vectors[idx];
                SimdSearchResult {
                    id: id.clone(),
                    score,
                    vector: vector.clone(),
                    metadata: metadata.clone(),
                }
            })
            .collect();
        
        let elapsed = start_time.elapsed();
        let metrics = SearchMetrics {
            query_time_ms: elapsed.as_secs_f64() * 1000.0,
            vectors_processed: candidate_vectors.len(),
            simd_operations: candidate_vectors.len() / 8 + candidate_vectors.len() / 16, // Estimate
            cache_hits: 0,
            similarity_calculations: candidate_vectors.len(),
        };
        
        // Store results in cache for future use
        {
            let mut cache = self.cache.write().unwrap();
            cache.insert(cache_key, (search_results.clone(), metrics.clone()));
        }
        
        debug!(
            "SIMD search completed: {} vectors, {:.2}ms, {:.2} vectors/ms",
            metrics.vectors_processed, 
            metrics.query_time_ms,
            metrics.vectors_processed as f64 / metrics.query_time_ms
        );
        
        Ok((search_results, metrics))
    }
    
    fn create_cache_key(
        &self,
        query_vector: &[f32],
        top_k: usize,
        use_cosine: bool,
        candidate_vectors: &[(String, Vec<f32>, serde_json::Value)],
    ) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        
        // Hash query vector (with some precision loss to improve cache hits)
        for &val in query_vector {
            ((val * 10000.0) as i32).hash(&mut hasher);
        }
        
        // Hash parameters
        top_k.hash(&mut hasher);
        use_cosine.hash(&mut hasher);
        
        // Hash candidate vector IDs (assuming the same set of candidates)
        let candidate_ids: Vec<&str> = candidate_vectors.iter()
            .map(|(id, _, _)| id.as_str())
            .collect();
        candidate_ids.hash(&mut hasher);
        
        format!("search_{:x}", hasher.finish())
    }
    
    /// Batch similarity search for multiple queries
    pub async fn batch_similarity_search(
        &self,
        query_vectors: &[&[f32]],
        candidate_vectors: &[(String, Vec<f32>, serde_json::Value)],
        top_k: usize,
        use_cosine: bool,
    ) -> Result<(Vec<Vec<SimdSearchResult>>, SearchMetrics)> {
        let start_time = Instant::now();
        
        // Validate all query vectors have correct dimensions
        for (i, query) in query_vectors.iter().enumerate() {
            if query.len() != self.vector_dims {
                return Err(SearchError::InvalidDimensions {
                    expected: self.vector_dims,
                    actual: query.len(),
                });
            }
        }
        
        // Extract candidate vectors for SIMD operations
        let vectors: Vec<&[f32]> = candidate_vectors.iter()
            .map(|(_, vec, _)| vec.as_slice())
            .collect();
        
        // Perform batch SIMD operations using parallel processing
        let batch_results: Result<Vec<Vec<(usize, f32)>>> = query_vectors.par_iter()
            .map(|query| {
                self.knn_search.find_knn(query, &vectors, top_k, use_cosine)
            })
            .collect();
        
        let knn_results = batch_results?;
        
        // Convert results to search results
        let all_results: Vec<Vec<SimdSearchResult>> = knn_results.into_iter()
            .map(|query_results| {
                query_results.into_iter()
                    .map(|(idx, score)| {
                        let (id, vector, metadata) = &candidate_vectors[idx];
                        SimdSearchResult {
                            id: id.clone(),
                            score,
                            vector: vector.clone(),
                            metadata: metadata.clone(),
                        }
                    })
                    .collect()
            })
            .collect();
        
        let elapsed = start_time.elapsed();
        let total_comparisons = query_vectors.len() * candidate_vectors.len();
        
        let metrics = SearchMetrics {
            query_time_ms: elapsed.as_secs_f64() * 1000.0,
            vectors_processed: total_comparisons,
            simd_operations: total_comparisons / 8 + total_comparisons / 16, // Estimate
            cache_hits: 0,
            similarity_calculations: total_comparisons,
        };
        
        info!(
            "Batch SIMD search completed: {} queries, {} candidates, {:.2}ms, {:.2} comparisons/ms",
            query_vectors.len(),
            candidate_vectors.len(),
            metrics.query_time_ms,
            total_comparisons as f64 / metrics.query_time_ms
        );
        
        Ok((all_results, metrics))
    }
    
    /// SIMD-accelerated vector clustering using k-means
    pub async fn simd_kmeans_clustering(
        &self,
        vectors: &[Vec<f32>],
        k: usize,
        max_iterations: usize,
        tolerance: f32,
    ) -> Result<(Vec<Vec<f32>>, Vec<usize>)> {
        if vectors.is_empty() || k == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        
        let n = vectors.len();
        let dims = vectors[0].len();
        
        // Validate dimensions
        for vec in vectors {
            if vec.len() != dims {
                return Err(SearchError::InvalidDimensions {
                    expected: dims,
                    actual: vec.len(),
                });
            }
        }
        
        // Initialize centroids using k-means++
        let mut centroids = self.initialize_centroids_plus_plus(vectors, k)?;
        let mut assignments = vec![0; n];
        
        for iteration in 0..max_iterations {
            let mut changed = false;
            
            // Assignment step using SIMD
            let new_assignments: Vec<usize> = vectors.par_iter().enumerate()
                .map(|(i, vector)| {
                    let centroid_refs: Vec<&[f32]> = centroids.iter().map(|c| c.as_slice()).collect();
                    let knn_result = self.knn_search.find_knn(
                        vector, 
                        &centroid_refs, 
                        1, 
                        true // Use cosine similarity
                    ).unwrap_or_default();
                    
                    if let Some((best_centroid_idx, _)) = knn_result.first() {
                        *best_centroid_idx
                    } else {
                        0
                    }
                })
                .collect();
            
            // Check for changes
            for (i, &new_assignment) in new_assignments.iter().enumerate() {
                if assignments[i] != new_assignment {
                    changed = true;
                    assignments[i] = new_assignment;
                }
            }
            
            if !changed {
                debug!("K-means converged after {} iterations", iteration + 1);
                break;
            }
            
            // Update centroids
            for cluster_id in 0..k {
                let cluster_vectors: Vec<&[f32]> = assignments.iter()
                    .enumerate()
                    .filter_map(|(i, &assignment)| {
                        if assignment == cluster_id {
                            Some(vectors[i].as_slice())
                        } else {
                            None
                        }
                    })
                    .collect();
                
                if !cluster_vectors.is_empty() {
                    centroids[cluster_id] = self.compute_centroid(&cluster_vectors)?;
                }
            }
        }
        
        Ok((centroids, assignments))
    }
    
    /// Initialize centroids using k-means++ algorithm for better clustering
    fn initialize_centroids_plus_plus(&self, vectors: &[Vec<f32>], k: usize) -> Result<Vec<Vec<f32>>> {
        if k >= vectors.len() {
            return Ok(vectors.to_vec());
        }
        
        let mut centroids = Vec::with_capacity(k);
        let mut distances = vec![f32::INFINITY; vectors.len()];
        
        // Choose first centroid randomly
        centroids.push(vectors[0].clone());
        
        for _ in 1..k {
            // Update distances to nearest centroid
            for (i, vector) in vectors.iter().enumerate() {
                let min_dist = centroids.iter()
                    .map(|centroid| {
                        self.simd_ops.euclidean_distance(vector, centroid).unwrap_or(f32::INFINITY)
                    })
                    .fold(f32::INFINITY, f32::min);
                
                distances[i] = min_dist;
            }
            
            // Choose next centroid with probability proportional to squared distance
            let total_weight: f32 = distances.iter().map(|d| d * d).sum();
            if total_weight == 0.0 {
                break;
            }
            
            let mut cumulative_weight = 0.0;
            let target_weight = total_weight * 0.5; // Simplified selection
            
            for (i, &dist) in distances.iter().enumerate() {
                cumulative_weight += dist * dist;
                if cumulative_weight >= target_weight {
                    centroids.push(vectors[i].clone());
                    break;
                }
            }
        }
        
        Ok(centroids)
    }
    
    /// Compute centroid of a set of vectors using SIMD operations
    fn compute_centroid(&self, vectors: &[&[f32]]) -> Result<Vec<f32>> {
        if vectors.is_empty() {
            return Err(SearchError::InsufficientData);
        }
        
        let dims = vectors[0].len();
        let mut centroid = vec![0.0; dims];
        
        // SIMD-accelerated centroid calculation
        for vector in vectors {
            for (i, &value) in vector.iter().enumerate() {
                centroid[i] += value;
            }
        }
        
        let count = vectors.len() as f32;
        for element in centroid.iter_mut() {
            *element /= count;
        }
        
        // Normalize the centroid
        self.simd_ops.normalize_vector(&mut centroid)?;
        
        Ok(centroid)
    }
    
    /// Get SIMD capabilities and performance information
    pub fn get_simd_info(&self) -> SimdCapabilities {
        SimdCapabilities {
            avx2_supported: self.simd_ops.use_avx2,
            avx512_supported: self.simd_ops.use_avx512,
            fma_supported: self.simd_ops.use_fma,
            vector_dimensions: self.vector_dims,
            estimated_speedup: self.estimate_speedup(),
        }
    }
    
    /// Estimate SIMD speedup based on available features
    fn estimate_speedup(&self) -> f32 {
        let mut speedup = 1.0;
        
        if self.simd_ops.use_avx512 {
            speedup *= 16.0; // Process 16 f32s at once
        } else if self.simd_ops.use_avx2 {
            speedup *= 8.0; // Process 8 f32s at once
        } else {
            speedup *= 4.0; // Fallback SIMD
        }
        
        if self.simd_ops.use_fma {
            speedup *= 1.5; // FMA provides additional speedup
        }
        
        speedup
    }
}

/// SIMD capabilities and performance information
#[derive(Debug, Clone)]
pub struct SimdCapabilities {
    pub avx2_supported: bool,
    pub avx512_supported: bool,
    pub fma_supported: bool,
    pub vector_dimensions: usize,
    pub estimated_speedup: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    
    #[tokio::test]
    async fn test_simd_similarity_search() {
        let config = SimdSearchConfig::default();
        let engine = SimdSearchEngine::new(config);
        
        let query = vec![1.0; 384];
        let candidates = vec![
            ("doc1".to_string(), vec![1.0; 384], json!({"title": "Test 1"})),
            ("doc2".to_string(), vec![0.5; 384], json!({"title": "Test 2"})),
            ("doc3".to_string(), vec![2.0; 384], json!({"title": "Test 3"})),
        ];
        
        let (results, metrics) = engine.similarity_search(&query, &candidates, 2, true).await.unwrap();
        
        assert_eq!(results.len(), 2);
        assert!(results[0].score >= results[1].score);
        assert!(metrics.query_time_ms > 0.0);
        assert_eq!(metrics.vectors_processed, 3);
    }
    
    #[tokio::test]
    async fn test_batch_similarity_search() {
        let config = SimdSearchConfig::default();
        let engine = SimdSearchEngine::new(config);
        
        let queries = vec![
            vec![1.0; 384].as_slice(),
            vec![0.0; 384].as_slice(),
        ];
        
        let candidates = vec![
            ("doc1".to_string(), vec![1.0; 384], json!({"title": "Test 1"})),
            ("doc2".to_string(), vec![0.0; 384], json!({"title": "Test 2"})),
        ];
        
        let (results, metrics) = engine.batch_similarity_search(&queries, &candidates, 1, true).await.unwrap();
        
        assert_eq!(results.len(), 2); // Two queries
        assert_eq!(results[0].len(), 1); // Top 1 result per query
        assert_eq!(results[1].len(), 1);
        assert!(metrics.query_time_ms > 0.0);
    }
    
    #[tokio::test]
    async fn test_kmeans_clustering() {
        let config = SimdSearchConfig {
            vector_dimensions: 3,
            ..Default::default()
        };
        let engine = SimdSearchEngine::new(config);
        
        let vectors = vec![
            vec![1.0, 1.0, 1.0],
            vec![1.1, 1.1, 1.1],
            vec![5.0, 5.0, 5.0],
            vec![5.1, 5.1, 5.1],
        ];
        
        let (centroids, assignments) = engine.simd_kmeans_clustering(&vectors, 2, 10, 0.01).await.unwrap();
        
        assert_eq!(centroids.len(), 2);
        assert_eq!(assignments.len(), 4);
        
        // Vectors should be assigned to different clusters
        assert_ne!(assignments[0], assignments[2]);
    }
}