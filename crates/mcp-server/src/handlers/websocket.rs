use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    response::Response,
};
use futures::{sink::SinkExt, stream::StreamExt};
use mcp_protocol::{McpRequest, McpResponse};
use tracing::{error, info};

use crate::AppState;

pub async fn websocket_handler(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> Response {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

async fn handle_socket(socket: WebSocket, state: AppState) {
    let (mut sender, mut receiver) = socket.split();
    
    info!("WebSocket connection established");
    
    while let Some(msg) = receiver.next().await {
        match msg {
            Ok(Message::Text(text)) => {
                // Parse MCP request
                match serde_json::from_str::<McpRequest>(&text) {
                    Ok(request) => {
                        // Process request
                        match process_websocket_request(&state, request).await {
                            Ok(response) => {
                                let response_text = serde_json::to_string(&response).unwrap();
                                if let Err(e) = sender.send(Message::Text(response_text.into())).await {
                                    error!("WebSocket send error: {}", e);
                                    break;
                                }
                            }
                            Err(e) => {
                                error!("Request processing error: {}", e);
                                let error_response = serde_json::json!({
                                    "error": e.to_string()
                                });
                                if let Err(e) = sender.send(Message::Text(error_response.to_string().into())).await {
                                    error!("WebSocket send error: {}", e);
                                    break;
                                }
                            }
                        }
                    }
                    Err(e) => {
                        error!("Invalid MCP request: {}", e);
                        let error_response = serde_json::json!({
                            "error": format!("Invalid request: {}", e)
                        });
                        if let Err(e) = sender.send(Message::Text(error_response.to_string().into())).await {
                            error!("WebSocket send error: {}", e);
                            break;
                        }
                    }
                }
            }
            Ok(Message::Binary(_)) => {
                // Binary messages not supported
                if let Err(e) = sender.send(Message::Text(
                    r#"{"error":"Binary messages not supported"}"#.to_string().into()
                )).await {
                    error!("WebSocket send error: {}", e);
                    break;
                }
            }
            Ok(Message::Close(_)) => {
                info!("WebSocket connection closed by client");
                break;
            }
            Ok(Message::Ping(data)) => {
                if let Err(e) = sender.send(Message::Pong(data)).await {
                    error!("WebSocket pong error: {}", e);
                    break;
                }
            }
            Ok(Message::Pong(_)) => {}
            Err(e) => {
                error!("WebSocket error: {}", e);
                break;
            }
        }
    }
    
    info!("WebSocket connection closed");
}

async fn process_websocket_request(
    state: &AppState,
    request: McpRequest,
) -> anyhow::Result<McpResponse> {
    // Use the same handler logic as HTTP
    use crate::handlers::mcp::mcp_handler;
    
    match mcp_handler(State(state.clone()), axum::Json(request)).await {
        Ok(axum::Json(response)) => Ok(response),
        Err(e) => Err(anyhow::anyhow!(e.to_string())),
    }
}