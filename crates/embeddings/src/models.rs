use candle_core::{Device, Tensor, D, IndexOp};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use std::path::Path;
use tracing::{debug, info};

use crate::{DeviceConfig, EmbeddingsError, PoolingStrategy, Result};

// Trait for embedding models
pub trait EmbeddingModel: Send + Sync {
    fn embed(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor>;
    fn hidden_size(&self) -> usize;
    fn device(&self) -> &Device;
}

// BERT-based embedding model
pub struct BertEmbedder {
    model: BertModel,
    config: BertConfig,
    device: Device,
    pooling_strategy: PoolingStrategy,
}

impl BertEmbedder {
    pub fn new(
        model_path: &Path,
        device_config: &DeviceConfig,
        pooling_strategy: PoolingStrategy,
    ) -> Result<Self> {
        let device = create_device(device_config)?;
        
        // Load model config
        let config_path = model_path.join("config.json");
        let config: BertConfig = serde_json::from_reader(
            std::fs::File::open(&config_path)
                .map_err(|e| EmbeddingsError::ModelLoading(
                    format!("Failed to open config: {}", e)
                ))?
        ).map_err(|e| EmbeddingsError::ModelLoading(
            format!("Failed to parse config: {}", e)
        ))?;
        
        // Load model weights
        let weights_path = model_path.join("pytorch_model.safetensors");
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], candle_core::DType::F32, &device)?
        };
        
        let model = BertModel::load(vb, &config)?;
        
        info!("Loaded BERT model with hidden size: {}", config.hidden_size);
        
        Ok(Self {
            model,
            config,
            device,
            pooling_strategy,
        })
    }
    
    fn apply_pooling(
        &self,
        hidden_states: &Tensor,
        attention_mask: &Tensor,
    ) -> Result<Tensor> {
        match self.pooling_strategy {
            PoolingStrategy::Mean => mean_pooling(hidden_states, attention_mask),
            PoolingStrategy::Max => max_pooling(hidden_states, attention_mask),
            PoolingStrategy::Cls => cls_pooling(hidden_states),
            PoolingStrategy::MeanSqrt => mean_sqrt_pooling(hidden_states, attention_mask),
        }
    }
}

impl EmbeddingModel for BertEmbedder {
    fn embed(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        // Get token type IDs (all zeros for single sequence)
        let token_type_ids = Tensor::zeros_like(input_ids)?;
        
        // Forward pass through BERT
        let outputs = self.model.forward(input_ids, &token_type_ids, Some(attention_mask))?;
        
        // Apply pooling strategy
        let pooled = self.apply_pooling(&outputs, attention_mask)?;
        
        Ok(pooled)
    }
    
    fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }
    
    fn device(&self) -> &Device {
        &self.device
    }
}

// Device creation helper
fn create_device(config: &DeviceConfig) -> Result<Device> {
    match config {
        DeviceConfig::Cpu => Ok(Device::Cpu),
        DeviceConfig::Cuda(idx) => {
            Device::new_cuda(*idx).map_err(|_| EmbeddingsError::GpuNotAvailable)
        }
        DeviceConfig::Metal => {
            Device::new_metal(0).map_err(|_| EmbeddingsError::GpuNotAvailable)
        }
        DeviceConfig::Auto => {
            // Try CUDA first
            if let Ok(device) = Device::new_cuda(0) {
                debug!("Using CUDA device");
                return Ok(device);
            }
            
            // Try Metal for Apple Silicon
            #[cfg(target_os = "macos")]
            if let Ok(device) = Device::new_metal(0) {
                debug!("Using Metal device");
                return Ok(device);
            }
            
            // Fall back to CPU
            debug!("Using CPU device");
            Ok(Device::Cpu)
        }
    }
}

// Pooling implementations
fn mean_pooling(hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
    let mask_expanded = attention_mask.unsqueeze(D::Minus1)?.expand(hidden_states.shape())?;
    let sum = hidden_states.mul(&mask_expanded)?.sum_keepdim(1)?;
    let count = mask_expanded.sum_keepdim(1)?.clamp(1e-9, f64::INFINITY)?;
    Ok((sum / count)?)
}

fn max_pooling(hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
    let mask_expanded = attention_mask.unsqueeze(D::Minus1)?.expand(hidden_states.shape())?;
    let masked = hidden_states.where_cond(
        &mask_expanded.eq(&Tensor::ones_like(&mask_expanded)?)?,
        &Tensor::new(&[f32::NEG_INFINITY], hidden_states.device())?.broadcast_as(hidden_states.shape())?
    )?;
    Ok(masked.max_keepdim(1)?)
}

fn cls_pooling(hidden_states: &Tensor) -> Result<Tensor> {
    // Take the first token (CLS token) embeddings
    Ok(hidden_states.i((.., 0, ..))?)
}

fn mean_sqrt_pooling(hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
    let pooled = mean_pooling(hidden_states, attention_mask)?;
    let seq_len = attention_mask.sum_keepdim(1)?;
    let sqrt_len = seq_len.sqrt()?;
    Ok((pooled * sqrt_len)?)
}

// Model cache for reusing loaded models
use std::sync::{Arc, RwLock};
use std::collections::HashMap;

pub struct ModelCache {
    cache: Arc<RwLock<HashMap<String, Arc<dyn EmbeddingModel>>>>,
}

impl ModelCache {
    pub fn new() -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    pub fn get(&self, model_id: &str) -> Option<Arc<dyn EmbeddingModel>> {
        self.cache.read().unwrap().get(model_id).cloned()
    }
    
    pub fn insert(&self, model_id: String, model: Arc<dyn EmbeddingModel>) {
        self.cache.write().unwrap().insert(model_id, model);
    }
    
    pub fn clear(&self) {
        self.cache.write().unwrap().clear();
    }
}