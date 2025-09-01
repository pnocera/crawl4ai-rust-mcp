use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ServerError {
    #[error("Protocol error: {0}")]
    Protocol(#[from] mcp_protocol::ProtocolError),
    
    #[error("Storage error: {0}")]
    Storage(#[from] storage::StorageError),
    
    #[error("Vector store error: {0}")]
    VectorStore(#[from] vector_store::VectorStoreError),
    
    #[error("Graph store error: {0}")]
    GraphStore(#[from] graph_store::GraphStoreError),
    
    #[error("Crawler error: {0}")]
    Crawler(#[from] crawler::CrawlerError),
    
    #[error("Embeddings error: {0}")]
    Embeddings(#[from] embeddings::EmbeddingsError),
    
    #[error("Search error: {0}")]
    Search(#[from] search::SearchError),
    
    #[error("Invalid request: {0}")]
    InvalidRequest(String),
    
    #[error("Not found: {0}")]
    NotFound(String),
    
    #[error("Internal server error: {0}")]
    Internal(String),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

impl IntoResponse for ServerError {
    fn into_response(self) -> Response {
        let (status, error_message) = match self {
            ServerError::InvalidRequest(ref msg) => (StatusCode::BAD_REQUEST, msg.clone()),
            ServerError::NotFound(ref msg) => (StatusCode::NOT_FOUND, msg.clone()),
            ServerError::Protocol(ref e) => (StatusCode::BAD_REQUEST, e.to_string()),
            _ => (StatusCode::INTERNAL_SERVER_ERROR, self.to_string()),
        };

        let body = Json(json!({
            "error": error_message,
            "status": status.as_u16(),
        }));

        (status, body).into_response()
    }
}

pub type Result<T> = std::result::Result<T, ServerError>;