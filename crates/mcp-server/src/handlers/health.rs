use axum::{http::StatusCode, Json};
use serde_json::json;

pub async fn health() -> (StatusCode, Json<serde_json::Value>) {
    (
        StatusCode::OK,
        Json(json!({
            "status": "healthy",
            "version": env!("CARGO_PKG_VERSION"),
            "service": "mcp-crawl4ai-rag",
        })),
    )
}