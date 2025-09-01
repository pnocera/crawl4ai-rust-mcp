use axum::{
    middleware,
    Router,
};
use std::net::SocketAddr;
use tower::ServiceBuilder;
use tower_http::{
    cors::CorsLayer,
    limit::RequestBodyLimitLayer,
    timeout::TimeoutLayer,
    trace::TraceLayer,
};
use tracing::info;

use crate::{
    middleware::{metrics_middleware, request_id_middleware, McpMakeSpan},
    routes::create_routes,
    AppState, ServerConfig,
};

pub struct McpServer {
    pub app: Router,
    pub addr: SocketAddr,
}

impl McpServer {
    pub async fn new(
        config: ServerConfig,
        qdrant_config: vector_store::QdrantConfig,
        memgraph_config: graph_store::MemgraphConfig,
    ) -> anyhow::Result<Self> {
        // Initialize application state
        let state = AppState::new(config.clone(), qdrant_config, memgraph_config).await?;
        
        // Create CORS layer
        let cors = CorsLayer::new()
            .allow_origin(
                config
                    .cors_origins
                    .iter()
                    .map(|o| o.parse().unwrap())
                    .collect::<Vec<_>>(),
            )
            .allow_methods(tower_http::cors::Any)
            .allow_headers(tower_http::cors::Any);
        
        // Build application with middleware stack
        let app = create_routes(state)
            .layer(cors)
            .layer(middleware::from_fn(request_id_middleware))
            .layer(middleware::from_fn(metrics_middleware))
            .layer(TraceLayer::new_for_http().make_span_with(McpMakeSpan))
            .layer(TimeoutLayer::new(
                std::time::Duration::from_secs(config.request_timeout),
            ))
            .layer(RequestBodyLimitLayer::new(config.max_request_size));
        
        Ok(Self {
            app,
            addr: config.socket_addr(),
        })
    }
    
    pub async fn run(self) -> anyhow::Result<()> {
        info!("Starting MCP server on {}", self.addr);
        
        let listener = tokio::net::TcpListener::bind(self.addr).await?;
        
        axum::serve(listener, self.app)
            .await
            .map_err(Into::into)
    }
}

// Graceful shutdown handler
pub async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }

    info!("Received shutdown signal");
}