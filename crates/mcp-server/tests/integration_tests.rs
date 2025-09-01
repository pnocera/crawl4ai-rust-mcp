use axum::http::StatusCode;
use mcp_protocol::{McpRequest, McpResponse, Tool};
use mcp_server::{McpServer, ServerConfig, QdrantConfig, MemgraphConfig};
use reqwest::Client;
use serde_json::json;
use tokio::time::{sleep, Duration};
use uuid::Uuid;

async fn start_test_server() -> (String, tokio::task::JoinHandle<()>) {
    let mut config = ServerConfig::default();
    config.port = 0; // Use random port
    config.storage_path = tempfile::tempdir().unwrap().path().to_path_buf();
    
    let qdrant_config = QdrantConfig::default();
    let memgraph_config = MemgraphConfig::default();
    
    let server = McpServer::new(config.clone(), qdrant_config, memgraph_config)
        .await
        .unwrap();
    
    let addr = server.addr;
    let handle = tokio::spawn(async move {
        server.run().await.unwrap();
    });
    
    // Wait for server to start
    sleep(Duration::from_millis(100)).await;
    
    (format!("http://{}", addr), handle)
}

#[tokio::test]
async fn test_health_endpoint() {
    let (base_url, _handle) = start_test_server().await;
    
    let client = Client::new();
    let response = client
        .get(format!("{}/health", base_url))
        .send()
        .await
        .unwrap();
    
    assert_eq!(response.status(), StatusCode::OK);
    
    let body: serde_json::Value = response.json().await.unwrap();
    assert_eq!(body["status"], "healthy");
    assert_eq!(body["service"], "mcp-crawl4ai-rag");
}

#[tokio::test]
async fn test_mcp_endpoint() {
    let (base_url, _handle) = start_test_server().await;
    
    let client = Client::new();
    let request = McpRequest {
        id: Uuid::new_v4(),
        tool: Tool::GetAvailableSources,
        params: json!({}),
    };
    
    let response = client
        .post(format!("{}/mcp", base_url))
        .json(&request)
        .send()
        .await
        .unwrap();
    
    assert_eq!(response.status(), StatusCode::OK);
    
    let body: McpResponse = response.json().await.unwrap();
    assert_eq!(body.id, request.id);
    assert_eq!(body.tool, Tool::GetAvailableSources);
}

#[tokio::test]
async fn test_sse_streaming() {
    let (base_url, _handle) = start_test_server().await;
    
    let client = Client::new();
    let request = McpRequest {
        id: Uuid::new_v4(),
        tool: Tool::CrawlSinglePage,
        params: json!({
            "url": "https://example.com",
            "wait_for": "domcontentloaded"
        }),
    };
    
    let response = client
        .post(format!("{}/mcp/stream", base_url))
        .json(&request)
        .send()
        .await
        .unwrap();
    
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get("content-type").unwrap(),
        "text/event-stream"
    );
    
    // In a real test, we would consume the SSE stream
    // For now, just verify the response headers
}

#[tokio::test]
async fn test_websocket_connection() {
    use tokio_tungstenite::{connect_async, tungstenite::Message};
    
    let (base_url, _handle) = start_test_server().await;
    let ws_url = base_url.replace("http://", "ws://") + "/ws";
    
    let (ws_stream, _) = connect_async(&ws_url).await.unwrap();
    let (mut write, mut read) = ws_stream.split();
    
    // Send MCP request
    let request = McpRequest {
        id: Uuid::new_v4(),
        tool: Tool::GetAvailableSources,
        params: json!({}),
    };
    
    use futures::{SinkExt, StreamExt};
    write
        .send(Message::Text(serde_json::to_string(&request).unwrap()))
        .await
        .unwrap();
    
    // Read response
    if let Some(Ok(Message::Text(text))) = read.next().await {
        let response: McpResponse = serde_json::from_str(&text).unwrap();
        assert_eq!(response.id, request.id);
        assert_eq!(response.tool, Tool::GetAvailableSources);
    } else {
        panic!("Expected text message");
    }
}

#[tokio::test]
async fn test_request_id_header() {
    let (base_url, _handle) = start_test_server().await;
    
    let client = Client::new();
    let response = client
        .get(format!("{}/health", base_url))
        .send()
        .await
        .unwrap();
    
    assert!(response.headers().contains_key("x-request-id"));
    
    let request_id = response.headers().get("x-request-id").unwrap();
    assert!(!request_id.is_empty());
}

#[tokio::test]
async fn test_cors_headers() {
    let (base_url, _handle) = start_test_server().await;
    
    let client = Client::new();
    let response = client
        .get(format!("{}/health", base_url))
        .header("Origin", "https://example.com")
        .send()
        .await
        .unwrap();
    
    assert!(response.headers().contains_key("access-control-allow-origin"));
}

#[tokio::test]
async fn test_invalid_tool_error() {
    let (base_url, _handle) = start_test_server().await;
    
    let client = Client::new();
    let response = client
        .post(format!("{}/mcp", base_url))
        .json(&json!({
            "id": Uuid::new_v4(),
            "tool": "invalid_tool",
            "params": {}
        }))
        .send()
        .await
        .unwrap();
    
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    
    let body: serde_json::Value = response.json().await.unwrap();
    assert!(body["error"].as_str().unwrap().contains("invalid"));
}