use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PagePoint {
    pub id: Uuid,
    pub url: String,
    pub title: String,
    pub domain: String,
    pub content_preview: String,
    pub crawled_at: DateTime<Utc>,
    pub metadata: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageVectors {
    pub title_vector: Vec<f32>,
    pub content_vector: Vec<f32>,
    pub code_vector: Option<Vec<f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchQuery {
    pub query_text: String,
    pub query_vectors: QueryVectors,
    pub filter: Option<SearchFilter>,
    pub limit: usize,
    pub offset: Option<usize>,
    pub with_payload: bool,
    pub with_vectors: bool,
    pub score_threshold: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryVectors {
    pub title_vector: Option<Vec<f32>>,
    pub content_vector: Option<Vec<f32>>,
    pub code_vector: Option<Vec<f32>>,
    pub weights: VectorWeights,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorWeights {
    pub title: f32,
    pub content: f32,
    pub code: f32,
}

impl Default for VectorWeights {
    fn default() -> Self {
        Self {
            title: 0.3,
            content: 0.5,
            code: 0.2,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchFilter {
    pub domain: Option<String>,
    pub date_from: Option<DateTime<Utc>>,
    pub date_to: Option<DateTime<Utc>>,
    pub has_code: Option<bool>,
    pub metadata_filters: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub id: Uuid,
    pub score: f32,
    pub payload: Option<PagePoint>,
    pub vectors: Option<PageVectors>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinaryQuantizationStats {
    pub original_size_bytes: usize,
    pub quantized_size_bytes: usize,
    pub compression_ratio: f32,
    pub recall_estimate: f32,
}

impl PagePoint {
    pub fn new(
        url: String,
        title: String,
        domain: String,
        content_preview: String,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            url,
            title,
            domain,
            content_preview,
            crawled_at: Utc::now(),
            metadata: HashMap::new(),
        }
    }
    
    pub fn with_metadata(mut self, key: String, value: serde_json::Value) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

// Binary quantization helpers
pub fn quantize_vector(vector: &[f32]) -> Vec<u8> {
    let mut quantized = vec![0u8; (vector.len() + 7) / 8];
    
    for (i, &value) in vector.iter().enumerate() {
        if value > 0.0 {
            let byte_idx = i / 8;
            let bit_idx = i % 8;
            quantized[byte_idx] |= 1 << bit_idx;
        }
    }
    
    quantized
}

pub fn dequantize_vector(quantized: &[u8], dimension: usize) -> Vec<f32> {
    let mut vector = vec![0.0f32; dimension];
    
    for i in 0..dimension {
        let byte_idx = i / 8;
        let bit_idx = i % 8;
        
        if byte_idx < quantized.len() && (quantized[byte_idx] & (1 << bit_idx)) != 0 {
            vector[i] = 1.0;
        } else {
            vector[i] = -1.0;
        }
    }
    
    vector
}

pub fn calculate_binary_similarity(a: &[u8], b: &[u8]) -> f32 {
    let mut hamming_distance = 0u32;
    
    for i in 0..a.len().min(b.len()) {
        hamming_distance += (a[i] ^ b[i]).count_ones();
    }
    
    let total_bits = a.len() * 8;
    1.0 - (hamming_distance as f32 / total_bits as f32)
}