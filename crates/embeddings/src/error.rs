use thiserror::Error;

#[derive(Error, Debug)]
pub enum EmbeddingsError {
    #[error("Model error: {0}")]
    Model(String),
    
    #[error("Tokenizer error: {0}")]
    Tokenizer(#[from] tokenizers::Error),
    
    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),
    
    #[error("Model not found: {0}")]
    ModelNotFound(String),
    
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    
    #[error("GPU not available")]
    GpuNotAvailable,
    
    #[error("Model loading error: {0}")]
    ModelLoading(String),
    
    #[error("Batch size exceeded: max {max}, got {actual}")]
    BatchSizeExceeded { max: usize, actual: usize },
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Hub API error: {0}")]
    HubApi(String),
}

pub type Result<T> = std::result::Result<T, EmbeddingsError>;