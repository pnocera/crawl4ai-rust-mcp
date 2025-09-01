use thiserror::Error;

#[derive(Error, Debug)]
pub enum GraphStoreError {
    #[error("Graph store error: {0}")]
    Internal(String),
    
    #[error("Connection error: {0}")]
    Connection(String),
    
    #[error("Neo4j/Memgraph error: {0}")]
    Database(#[from] neo4rs::Error),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Repository clone error: {0}")]
    RepositoryClone(String),
    
    #[error("Regex error: {0}")]
    Regex(#[from] regex::Error),
    
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}