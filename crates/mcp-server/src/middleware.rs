use axum::{
    body::Body,
    http::{Request, Response, StatusCode},
    middleware::Next,
};
use std::time::Instant;
use tower_http::trace::MakeSpan;
use tracing::{error, info, Span};
use uuid::Uuid;

// Request ID middleware
pub async fn request_id_middleware(
    mut req: Request<Body>,
    next: Next,
) -> Response<Body> {
    let request_id = Uuid::new_v4().to_string();
    req.extensions_mut().insert(request_id.clone());
    
    let mut response = next.run(req).await;
    response.headers_mut().insert(
        "x-request-id",
        request_id.parse().unwrap(),
    );
    
    response
}

// Metrics middleware
pub async fn metrics_middleware(
    req: Request<Body>,
    next: Next,
) -> Response<Body> {
    let start = Instant::now();
    let method = req.method().clone();
    let uri = req.uri().clone();
    
    let response = next.run(req).await;
    
    let duration = start.elapsed();
    let status = response.status();
    
    if status.is_server_error() {
        error!(
            method = %method,
            uri = %uri,
            status = %status,
            duration_ms = %duration.as_millis(),
            "Request failed"
        );
    } else {
        info!(
            method = %method,
            uri = %uri,
            status = %status,
            duration_ms = %duration.as_millis(),
            "Request completed"
        );
    }
    
    response
}

// Custom span maker for tracing
#[derive(Clone)]
pub struct McpMakeSpan;

impl<B> MakeSpan<B> for McpMakeSpan {
    fn make_span(&mut self, request: &Request<B>) -> Span {
        let request_id = request
            .extensions()
            .get::<String>()
            .cloned()
            .unwrap_or_else(|| Uuid::new_v4().to_string());
        
        tracing::info_span!(
            "http_request",
            method = %request.method(),
            uri = %request.uri(),
            request_id = %request_id,
        )
    }
}