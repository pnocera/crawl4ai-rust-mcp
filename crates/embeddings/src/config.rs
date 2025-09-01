use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    pub model_id: String,
    pub cache_dir: PathBuf,
    pub max_batch_size: usize,
    pub max_sequence_length: usize,
    pub device: DeviceConfig,
    pub pooling_strategy: PoolingStrategy,
    pub normalize: bool,
    pub use_half_precision: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DeviceConfig {
    Cpu,
    Cuda(usize),  // GPU index
    Metal,        // Apple Silicon
    Auto,         // Automatically select best available
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PoolingStrategy {
    Mean,
    Max,
    Cls,  // Use [CLS] token
    MeanSqrt,  // Mean with sqrt length normalization
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            model_id: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            cache_dir: PathBuf::from("./models"),
            max_batch_size: 32,
            max_sequence_length: 512,
            device: DeviceConfig::Auto,
            pooling_strategy: PoolingStrategy::Mean,
            normalize: true,
            use_half_precision: false,
        }
    }
}

impl EmbeddingConfig {
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            ..Default::default()
        }
    }
    
    pub fn with_device(mut self, device: DeviceConfig) -> Self {
        self.device = device;
        self
    }
    
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.max_batch_size = size;
        self
    }
    
    pub fn with_cache_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.cache_dir = dir.into();
        self
    }
}

// Predefined model configurations
pub struct ModelPresets;

impl ModelPresets {
    pub fn minilm_l6_v2() -> EmbeddingConfig {
        EmbeddingConfig {
            model_id: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            max_sequence_length: 256,
            ..Default::default()
        }
    }
    
    pub fn minilm_l12_v2() -> EmbeddingConfig {
        EmbeddingConfig {
            model_id: "sentence-transformers/all-MiniLM-L12-v2".to_string(),
            max_sequence_length: 256,
            ..Default::default()
        }
    }
    
    pub fn mpnet_base_v2() -> EmbeddingConfig {
        EmbeddingConfig {
            model_id: "sentence-transformers/all-mpnet-base-v2".to_string(),
            max_sequence_length: 384,
            ..Default::default()
        }
    }
    
    pub fn e5_small_v2() -> EmbeddingConfig {
        EmbeddingConfig {
            model_id: "intfloat/e5-small-v2".to_string(),
            max_sequence_length: 512,
            pooling_strategy: PoolingStrategy::MeanSqrt,
            ..Default::default()
        }
    }
    
    pub fn bge_small_en() -> EmbeddingConfig {
        EmbeddingConfig {
            model_id: "BAAI/bge-small-en-v1.5".to_string(),
            max_sequence_length: 512,
            pooling_strategy: PoolingStrategy::Cls,
            ..Default::default()
        }
    }
}