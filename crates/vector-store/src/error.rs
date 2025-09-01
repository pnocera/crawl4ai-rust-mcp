use thiserror::Error;

#[derive(Error, Debug)]
pub enum VectorStoreError {
    #[error("Qdrant error: {0}")]
    Qdrant(#[from] qdrant_client::QdrantError),
    
    #[error("Collection not found: {0}")]
    CollectionNotFound(String),
    
    #[error("Invalid vector dimension: expected {expected}, got {actual}")]
    InvalidDimension { expected: usize, actual: usize },
    
    #[error("Binary quantization error: {0}")]
    BinaryQuantization(String),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    #[error("Configuration error: {0}")]
    Configuration(String),
    
    #[error("Search error: {0}")]
    Search(String),
    
    #[error("Indexing error: {0}")]
    Indexing(String),
}

pub type Result<T> = std::result::Result<T, VectorStoreError>;