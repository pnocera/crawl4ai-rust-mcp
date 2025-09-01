use thiserror::Error;

#[derive(Error, Debug)]
pub enum ProtocolError {
    #[error("Invalid message format: {0}")]
    InvalidFormat(String),
    
    #[error("JSON serialization error: {0}")]
    JsonSerialization(#[from] serde_json::Error),
    
    #[error("Rkyv serialization error: {0}")]
    RkyvSerialization(String),
    
    #[error("Archive validation error: {0}")]
    ArchiveValidation(String),
    
    #[error("Zero-copy conversion error: {0}")]
    ZeroCopyConversion(String),
    
    #[error("Invalid tool: {0}")]
    InvalidTool(String),
    
    #[error("Missing required field: {0}")]
    MissingField(String),
}

pub type Result<T> = std::result::Result<T, ProtocolError>;