use std::path::{Path, PathBuf};
use std::sync::Arc;
use hf_hub::api::tokio::Api;
use hf_hub::{Repo, RepoType};
use tokio::sync::RwLock;
use tracing::{info, warn};

use crate::{
    BatchProcessor, BertEmbedder, EmbeddingConfig, EmbeddingModel,
    EmbeddingsError, ModelCache, Result, TokenizerWrapper,
};

pub struct EmbeddingService {
    config: EmbeddingConfig,
    model_cache: Arc<ModelCache>,
    current_model: Arc<RwLock<Option<Arc<dyn EmbeddingModel>>>>,
    current_tokenizer: Arc<RwLock<Option<Arc<TokenizerWrapper>>>>,
    api: Api,
}

impl EmbeddingService {
    pub async fn new() -> Result<Self> {
        Self::with_config(EmbeddingConfig::default()).await
    }
    
    pub async fn with_config(config: EmbeddingConfig) -> Result<Self> {
        // Create cache directory if it doesn't exist
        std::fs::create_dir_all(&config.cache_dir)?;
        
        let api = Api::new()
            .map_err(|e| EmbeddingsError::HubApi(e.to_string()))?;
        let model_cache = Arc::new(ModelCache::new());
        
        let mut service = Self {
            config,
            model_cache,
            current_model: Arc::new(RwLock::new(None)),
            current_tokenizer: Arc::new(RwLock::new(None)),
            api,
        };
        
        // Load default model
        service.load_model(&service.config.model_id.clone()).await?;
        
        Ok(service)
    }
    
    pub async fn load_model(&mut self, model_id: &str) -> Result<()> {
        info!("Loading model: {}", model_id);
        
        // Check cache first
        if let Some(model) = self.model_cache.get(model_id) {
            info!("Model found in cache");
            *self.current_model.write().await = Some(model);
            return Ok(());
        }
        
        // Download model if needed
        let model_path = self.ensure_model_downloaded(model_id).await?;
        
        // Load tokenizer
        let tokenizer_path = model_path.join("tokenizer.json");
        let tokenizer = Arc::new(TokenizerWrapper::new(
            &tokenizer_path,
            self.config.max_sequence_length,
        )?);
        
        // Load model
        let model: Arc<dyn EmbeddingModel> = Arc::new(BertEmbedder::new(
            &model_path,
            &self.config.device,
            self.config.pooling_strategy,
        )?);
        
        // Update cache
        self.model_cache.insert(model_id.to_string(), model.clone());
        
        // Update current model
        *self.current_model.write().await = Some(model);
        *self.current_tokenizer.write().await = Some(tokenizer);
        
        info!("Model loaded successfully");
        
        Ok(())
    }
    
    async fn ensure_model_downloaded(&self, model_id: &str) -> Result<PathBuf> {
        let model_path = self.config.cache_dir.join(model_id.replace('/', "--"));
        
        if model_path.exists() {
            info!("Model already downloaded at: {:?}", model_path);
            return Ok(model_path);
        }
        
        info!("Downloading model from Hugging Face Hub...");
        
        let repo = Repo::new(model_id.to_string(), RepoType::Model);
        let api_repo = self.api.repo(repo);
        
        // Download required files
        let files = vec![
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "pytorch_model.safetensors",
            "model.safetensors", // Alternative name
        ];
        
        std::fs::create_dir_all(&model_path)?;
        
        for file in files {
            match api_repo.get(file).await {
                Ok(file_path) => {
                    let dest = model_path.join(file);
                    if !dest.exists() {
                        std::fs::copy(file_path, dest)?;
                        info!("Downloaded: {}", file);
                    }
                }
                Err(e) => {
                    if file == "pytorch_model.safetensors" || file == "model.safetensors" {
                        // One of these must exist
                        warn!("Could not download {}: {}", file, e);
                    } else if file != "model.safetensors" {
                        // Other files are required
                        return Err(EmbeddingsError::ModelLoading(
                            format!("Failed to download {}: {}", file, e)
                        ));
                    }
                }
            }
        }
        
        Ok(model_path)
    }
    
    pub async fn create_processor(&self) -> Result<BatchProcessor> {
        let model = self.current_model.read().await
            .as_ref()
            .ok_or_else(|| EmbeddingsError::ModelNotFound("No model loaded".to_string()))?
            .clone();
        
        let tokenizer = self.current_tokenizer.read().await
            .as_ref()
            .ok_or_else(|| EmbeddingsError::ModelNotFound("No tokenizer loaded".to_string()))?
            .clone();
        
        Ok(BatchProcessor::new(
            model,
            tokenizer,
            self.config.max_batch_size,
            4, // max concurrent batches
        ).with_normalization(self.config.normalize))
    }
    
    pub async fn embed_texts(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let processor = self.create_processor().await?;
        processor.process_batch(texts).await
    }
    
    pub async fn embed_text(&self, text: String) -> Result<Vec<f32>> {
        let processor = self.create_processor().await?;
        processor.process_single(text).await
    }
    
    pub fn list_cached_models(&self) -> Vec<String> {
        std::fs::read_dir(&self.config.cache_dir)
            .ok()
            .map(|entries| {
                entries
                    .filter_map(|entry| {
                        entry.ok().and_then(|e| {
                            e.file_name()
                                .to_str()
                                .map(|s| s.replace("--", "/"))
                        })
                    })
                    .collect()
            })
            .unwrap_or_default()
    }
    
    pub async fn clear_model_cache(&self) {
        self.model_cache.clear();
        *self.current_model.write().await = None;
        *self.current_tokenizer.write().await = None;
    }
    
    pub fn get_model_info(&self) -> ModelInfo {
        ModelInfo {
            model_id: self.config.model_id.clone(),
            cache_dir: self.config.cache_dir.clone(),
            device: format!("{:?}", self.config.device),
            max_batch_size: self.config.max_batch_size,
            max_sequence_length: self.config.max_sequence_length,
            pooling_strategy: format!("{:?}", self.config.pooling_strategy),
            normalize: self.config.normalize,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub model_id: String,
    pub cache_dir: PathBuf,
    pub device: String,
    pub max_batch_size: usize,
    pub max_sequence_length: usize,
    pub pooling_strategy: String,
    pub normalize: bool,
}

// Model download manager for offline use
pub struct ModelDownloadManager {
    api: Api,
    cache_dir: PathBuf,
}

impl ModelDownloadManager {
    pub fn new(cache_dir: PathBuf) -> Result<Self> {
        std::fs::create_dir_all(&cache_dir)?;
        Ok(Self {
            api: Api::new()
                .map_err(|e| EmbeddingsError::HubApi(e.to_string()))?,
            cache_dir,
        })
    }
    
    pub async fn download_models(&self, model_ids: Vec<String>) -> Result<()> {
        for model_id in model_ids {
            info!("Downloading model: {}", model_id);
            
            let model_path = self.cache_dir.join(model_id.replace('/', "--"));
            if model_path.exists() {
                info!("Model already exists, skipping");
                continue;
            }
            
            let service = EmbeddingService::with_config(
                EmbeddingConfig::new(model_id.clone())
                    .with_cache_dir(self.cache_dir.clone())
            ).await?;
            
            // This triggers download
            drop(service);
            
            info!("Model downloaded successfully");
        }
        
        Ok(())
    }
    
    pub fn list_downloaded_models(&self) -> Vec<String> {
        std::fs::read_dir(&self.cache_dir)
            .ok()
            .map(|entries| {
                entries
                    .filter_map(|entry| {
                        entry.ok().and_then(|e| {
                            e.file_name()
                                .to_str()
                                .map(|s| s.replace("--", "/"))
                        })
                    })
                    .collect()
            })
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_model_info() {
        let config = EmbeddingConfig::default();
        let expected_model_id = config.model_id.clone();
        
        // Don't actually download in tests
        let service = EmbeddingService {
            config,
            model_cache: Arc::new(ModelCache::new()),
            current_model: Arc::new(RwLock::new(None)),
            current_tokenizer: Arc::new(RwLock::new(None)),
            api: Api::new().unwrap(),
        };
        
        let info = service.get_model_info();
        assert_eq!(info.model_id, expected_model_id);
        assert_eq!(info.max_batch_size, 32);
        assert!(info.normalize);
    }
}