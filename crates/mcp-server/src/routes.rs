use axum::{
    routing::{get, post},
    Router,
};

use crate::{handlers, AppState};

pub fn create_routes(state: AppState) -> Router {
    Router::new()
        // Health check
        .route("/health", get(handlers::health))
        
        // MCP protocol endpoint
        .route("/mcp", post(handlers::mcp_handler))
        
        // SSE endpoint for streaming responses
        .route("/mcp/stream", post(handlers::mcp_stream_handler))
        
        // Tool-specific routes (optional REST API)
        .route("/api/crawl", post(handlers::crawl_handler))
        .route("/api/search", post(handlers::search_handler))
        .route("/api/sources", get(handlers::sources_handler))
        
        // WebSocket endpoint for bidirectional communication
        .route("/ws", get(handlers::websocket_handler))
        
        .with_state(state)
}