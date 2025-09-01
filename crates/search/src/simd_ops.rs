use std::arch::x86_64::*;
use rayon::prelude::*;

use crate::{Result, SearchError};

/// SIMD-accelerated vector operations for high-performance similarity search
pub struct SimdVectorOps {
    pub use_avx2: bool,
    pub use_avx512: bool,
    pub use_fma: bool,
    pub vector_dims: usize,
}

impl SimdVectorOps {
    /// Create a new SIMD vector operations instance with feature detection
    pub fn new(vector_dims: usize) -> Self {
        let use_avx2 = is_x86_feature_detected!("avx2");
        let use_avx512 = false; // AVX512 is unstable, disable for now
        let use_fma = is_x86_feature_detected!("fma");
        
        Self {
            use_avx2,
            use_avx512,
            use_fma,
            vector_dims,
        }
    }
    
    /// Compute cosine similarity between two vectors using optimal SIMD
    pub fn cosine_similarity(&self, vec_a: &[f32], vec_b: &[f32]) -> Result<f32> {
        if vec_a.len() != vec_b.len() {
            return Err(SearchError::InvalidDimensions {
                expected: vec_a.len(),
                actual: vec_b.len(),
            });
        }
        
        if self.use_avx2 {
            unsafe { self.cosine_similarity_avx2(vec_a, vec_b) }
        } else {
            self.cosine_similarity_fallback(vec_a, vec_b)
        }
    }
    
    /// Compute Euclidean distance between two vectors
    pub fn euclidean_distance(&self, vec_a: &[f32], vec_b: &[f32]) -> Result<f32> {
        if vec_a.len() != vec_b.len() {
            return Err(SearchError::InvalidDimensions {
                expected: vec_a.len(),
                actual: vec_b.len(),
            });
        }
        
        if self.use_avx2 {
            unsafe { self.euclidean_distance_avx2(vec_a, vec_b) }
        } else {
            self.euclidean_distance_fallback(vec_a, vec_b)
        }
    }
    
    /// Normalize a vector in-place using SIMD
    pub fn normalize_vector(&self, vector: &mut [f32]) -> Result<()> {
        let mut norm = 0.0;
        
        // Calculate norm using SIMD
        if self.use_avx2 {
            unsafe {
                norm = self.vector_norm_avx2(vector)?;
            }
        } else {
            norm = vector.iter().map(|&x| x * x).sum::<f32>().sqrt();
        }
        
        if norm == 0.0 {
            return Ok(());
        }
        
        // Normalize using SIMD
        if self.use_avx2 {
            unsafe {
                self.vector_scale_avx2(vector, 1.0 / norm)?;
            }
        } else {
            for element in vector {
                *element /= norm;
            }
        }
        
        Ok(())
    }
    
    // AVX512 implementations removed due to unstable feature requirements
    
    // AVX2 implementations
    
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn cosine_similarity_avx2(&self, vec_a: &[f32], vec_b: &[f32]) -> Result<f32> {
        let mut dot_product = 0.0;
        let mut norm_a = 0.0;
        let mut norm_b = 0.0;
        
        let len = vec_a.len();
        let chunks = len / 8;
        
        let mut dot_acc = _mm256_setzero_ps();
        let mut norm_a_acc = _mm256_setzero_ps();
        let mut norm_b_acc = _mm256_setzero_ps();
        
        for i in 0..chunks {
            let idx = i * 8;
            let a_vec = _mm256_loadu_ps(vec_a.as_ptr().add(idx));
            let b_vec = _mm256_loadu_ps(vec_b.as_ptr().add(idx));
            
            if self.use_fma {
                dot_acc = _mm256_fmadd_ps(a_vec, b_vec, dot_acc);
                norm_a_acc = _mm256_fmadd_ps(a_vec, a_vec, norm_a_acc);
                norm_b_acc = _mm256_fmadd_ps(b_vec, b_vec, norm_b_acc);
            } else {
                dot_acc = _mm256_add_ps(dot_acc, _mm256_mul_ps(a_vec, b_vec));
                norm_a_acc = _mm256_add_ps(norm_a_acc, _mm256_mul_ps(a_vec, a_vec));
                norm_b_acc = _mm256_add_ps(norm_b_acc, _mm256_mul_ps(b_vec, b_vec));
            }
        }
        
        // Sum up the accumulated values
        dot_product = self.sum_avx2(dot_acc);
        norm_a = self.sum_avx2(norm_a_acc);
        norm_b = self.sum_avx2(norm_b_acc);
        
        // Process remaining elements
        for i in (chunks * 8)..len {
            dot_product += vec_a[i] * vec_b[i];
            norm_a += vec_a[i] * vec_a[i];
            norm_b += vec_b[i] * vec_b[i];
        }
        
        let norm_product = norm_a.sqrt() * norm_b.sqrt();
        
        if norm_product == 0.0 {
            Ok(0.0)
        } else {
            Ok(dot_product / norm_product)
        }
    }
    
    #[target_feature(enable = "avx2")]
    unsafe fn euclidean_distance_avx2(&self, vec_a: &[f32], vec_b: &[f32]) -> Result<f32> {
        let mut sum_sq_diff = 0.0;
        
        let len = vec_a.len();
        let chunks = len / 8;
        
        let mut acc = _mm256_setzero_ps();
        
        for i in 0..chunks {
            let idx = i * 8;
            let a_vec = _mm256_loadu_ps(vec_a.as_ptr().add(idx));
            let b_vec = _mm256_loadu_ps(vec_b.as_ptr().add(idx));
            let diff = _mm256_sub_ps(a_vec, b_vec);
            
            if self.use_fma {
                acc = _mm256_fmadd_ps(diff, diff, acc);
            } else {
                acc = _mm256_add_ps(acc, _mm256_mul_ps(diff, diff));
            }
        }
        
        sum_sq_diff = self.sum_avx2(acc);
        
        // Process remaining elements
        for i in (chunks * 8)..len {
            let diff = vec_a[i] - vec_b[i];
            sum_sq_diff += diff * diff;
        }
        
        Ok(sum_sq_diff.sqrt())
    }
    
    #[target_feature(enable = "avx2")]
    unsafe fn vector_norm_avx2(&self, vector: &[f32]) -> Result<f32> {
        let mut norm_sq = 0.0;
        
        let len = vector.len();
        let chunks = len / 8;
        
        let mut acc = _mm256_setzero_ps();
        
        for i in 0..chunks {
            let idx = i * 8;
            let vec = _mm256_loadu_ps(vector.as_ptr().add(idx));
            
            if self.use_fma {
                acc = _mm256_fmadd_ps(vec, vec, acc);
            } else {
                acc = _mm256_add_ps(acc, _mm256_mul_ps(vec, vec));
            }
        }
        
        norm_sq = self.sum_avx2(acc);
        
        // Process remaining elements
        for i in (chunks * 8)..len {
            norm_sq += vector[i] * vector[i];
        }
        
        Ok(norm_sq.sqrt())
    }
    
    #[target_feature(enable = "avx2")]
    unsafe fn vector_scale_avx2(&self, vector: &mut [f32], scale: f32) -> Result<()> {
        let len = vector.len();
        let chunks = len / 8;
        
        let scale_vec = _mm256_set1_ps(scale);
        
        for i in 0..chunks {
            let idx = i * 8;
            let vec = _mm256_loadu_ps(vector.as_ptr().add(idx));
            let scaled = _mm256_mul_ps(vec, scale_vec);
            _mm256_storeu_ps(vector.as_mut_ptr().add(idx), scaled);
        }
        
        // Process remaining elements
        for i in (chunks * 8)..len {
            vector[i] *= scale;
        }
        
        Ok(())
    }
    
    // Helper functions for summing SIMD registers
    
    #[target_feature(enable = "avx2")]
    unsafe fn sum_avx2(&self, vec: __m256) -> f32 {
        let sum128 = _mm_add_ps(_mm256_extractf128_ps(vec, 0), _mm256_extractf128_ps(vec, 1));
        let sum = _mm_hadd_ps(sum128, sum128);
        let sum2 = _mm_hadd_ps(sum, sum);
        _mm_cvtss_f32(sum2)
    }
    
    // Fallback implementations using scalar operations
    
    fn cosine_similarity_fallback(&self, vec_a: &[f32], vec_b: &[f32]) -> Result<f32> {
        let mut dot_product = 0.0;
        let mut norm_a = 0.0;
        let mut norm_b = 0.0;
        
        for i in 0..vec_a.len() {
            dot_product += vec_a[i] * vec_b[i];
            norm_a += vec_a[i] * vec_a[i];
            norm_b += vec_b[i] * vec_b[i];
        }
        
        let norm_product = norm_a.sqrt() * norm_b.sqrt();
        
        if norm_product == 0.0 {
            Ok(0.0)
        } else {
            Ok(dot_product / norm_product)
        }
    }
    
    fn euclidean_distance_fallback(&self, vec_a: &[f32], vec_b: &[f32]) -> Result<f32> {
        let mut sum_sq_diff = 0.0;
        
        for i in 0..vec_a.len() {
            let diff = vec_a[i] - vec_b[i];
            sum_sq_diff += diff * diff;
        }
        
        Ok(sum_sq_diff.sqrt())
    }
}

/// K-nearest neighbors search using SIMD-accelerated distance calculations
pub struct SimdKnnSearch {
    simd_ops: SimdVectorOps,
    vector_dims: usize,
}

impl SimdKnnSearch {
    pub fn new(vector_dims: usize) -> Self {
        Self {
            simd_ops: SimdVectorOps::new(vector_dims),
            vector_dims,
        }
    }
    
    /// Find k-nearest neighbors using SIMD-accelerated similarity calculations
    pub fn find_knn(
        &self,
        query: &[f32],
        candidates: &[&[f32]],
        k: usize,
        use_cosine: bool,
    ) -> Result<Vec<(usize, f32)>> {
        if candidates.is_empty() {
            return Ok(Vec::new());
        }
        
        // Compute similarities in parallel using SIMD
        let similarities: Result<Vec<(usize, f32)>> = candidates
            .par_iter()
            .enumerate()
            .map(|(idx, candidate)| {
                let similarity = if use_cosine {
                    self.simd_ops.cosine_similarity(query, candidate)?
                } else {
                    let dist = self.simd_ops.euclidean_distance(query, candidate)?;
                    1.0 / (1.0 + dist) // Convert distance to similarity
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
        
        Ok(sims)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_simd_cosine_similarity() {
        let ops = SimdVectorOps::new(4);
        
        let vec_a = vec![1.0, 0.0, 0.0, 0.0];
        let vec_b = vec![1.0, 0.0, 0.0, 0.0];
        
        let similarity = ops.cosine_similarity(&vec_a, &vec_b).unwrap();
        assert!((similarity - 1.0).abs() < 1e-6);
        
        let vec_c = vec![0.0, 1.0, 0.0, 0.0];
        let similarity2 = ops.cosine_similarity(&vec_a, &vec_c).unwrap();
        assert!((similarity2 - 0.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_simd_euclidean_distance() {
        let ops = SimdVectorOps::new(3);
        
        let vec_a = vec![0.0, 0.0, 0.0];
        let vec_b = vec![3.0, 4.0, 0.0];
        
        let distance = ops.euclidean_distance(&vec_a, &vec_b).unwrap();
        assert!((distance - 5.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_vector_normalization() {
        let ops = SimdVectorOps::new(3);
        
        let mut vector = vec![3.0, 4.0, 0.0];
        ops.normalize_vector(&mut vector).unwrap();
        
        let magnitude = (vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]).sqrt();
        assert!((magnitude - 1.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_knn_search() {
        let knn = SimdKnnSearch::new(2);
        
        let query = vec![1.0, 0.0];
        let candidates = vec![
            vec![1.0, 0.0].as_slice(),
            vec![0.0, 1.0].as_slice(),
            vec![0.5, 0.5].as_slice(),
        ];
        
        let results = knn.find_knn(&query, &candidates, 2, true).unwrap();
        
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0); // First candidate should be most similar
        assert!(results[0].1 > results[1].1); // Scores should be ordered
    }
}