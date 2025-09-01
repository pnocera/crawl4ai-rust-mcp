use mcp_server::{McpServer, ServerConfig};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "mcp_server=debug,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    // Load configurations
    let server_config = ServerConfig::from_env();
    let qdrant_config = vector_store::QdrantConfig::default();
    let memgraph_config = graph_store::MemgraphConfig::default();
    
    // Create and run server
    let server = McpServer::new(server_config, qdrant_config, memgraph_config).await?;
    
    // Run with graceful shutdown
    tokio::select! {
        result = server.run() => {
            if let Err(e) = result {
                tracing::error!("Server error: {}", e);
                return Err(e);
            }
        }
        _ = mcp_server::shutdown_signal() => {
            tracing::info!("Shutting down gracefully");
        }
    }
    
    Ok(())
}