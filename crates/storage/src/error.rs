use thiserror::Error;
use uuid::Uuid;

#[derive(Error, Debug)]
pub enum StorageError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("RocksDB error: {0}")]
    RocksDb(#[from] rocksdb::Error),
    
    #[error("DuckDB error: {0}")]
    DuckDb(#[from] duckdb::Error),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] bincode::Error),
    
    #[error("Memory map error: {0}")]
    MemoryMap(String),
    
    #[error("Storage not initialized")]
    NotInitialized,
    
    #[error("Invalid storage path: {0}")]
    InvalidPath(String),
    
    #[error("Compression error: {0}")]
    Compression(String),
    
    #[error("Transaction error: {0}")]
    TransactionError(String),
    
    #[error("Transaction not found: {0}")]
    TransactionNotFound(Uuid),
    
    #[error("Deadlock detected")]
    Deadlock,
    
    #[error("Data not found: {0}")]
    NotFound(String),
    
    #[error("Other error: {0}")]
    Other(String),
    
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    
    #[error("Backup operation failed: {0}")]
    BackupFailed(String),
    
    #[error("Corrupted data: {0}")]
    CorruptedData(String),
}

pub type Result<T> = std::result::Result<T, StorageError>;