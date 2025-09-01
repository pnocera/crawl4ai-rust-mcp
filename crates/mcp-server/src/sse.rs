use axum::response::sse::{Event, KeepAlive, Sse};
use futures::stream::Stream;
use mcp_protocol::SseEvent;
use std::convert::Infallible;
use std::pin::Pin;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;

// SSE stream type - using dynamic dispatch for simplicity
pub type SseStream = Sse<Pin<Box<dyn Stream<Item = Result<Event, Infallible>> + Send>>>;

// SSE event sender for streaming responses
#[derive(Clone)]
pub struct SseEventSender {
    tx: mpsc::Sender<SseEvent>,
}

impl SseEventSender {
    pub fn new() -> (Self, SseStream) {
        let (tx, rx) = mpsc::channel::<SseEvent>(100);
        
        let stream = ReceiverStream::new(rx).map(|event| {
            let data = serde_json::to_string(&event).unwrap_or_default();
            
            Ok(Event::default()
                .event(match &event {
                    SseEvent::Progress { .. } => "progress",
                    SseEvent::PartialResult { .. } => "partial",
                    SseEvent::Complete { .. } => "complete",
                    SseEvent::Error { .. } => "error",
                })
                .data(data))
        });
        
        let boxed_stream: Pin<Box<dyn Stream<Item = Result<Event, Infallible>> + Send>> = Box::pin(stream);
        let sse = Sse::new(boxed_stream)
            .keep_alive(
                KeepAlive::new()
                    .interval(Duration::from_secs(30))
                    .text("keep-alive"),
            );
        
        (Self { tx }, sse)
    }
    
    pub async fn send_progress(&self, message: String, percentage: Option<f32>) -> Result<(), mpsc::error::SendError<SseEvent>> {
        self.tx.send(SseEvent::Progress { message, percentage }).await
    }
    
    pub async fn send_partial(&self, data: serde_json::Value) -> Result<(), mpsc::error::SendError<SseEvent>> {
        self.tx.send(SseEvent::PartialResult { data }).await
    }
    
    pub async fn send_complete(&self, data: serde_json::Value) -> Result<(), mpsc::error::SendError<SseEvent>> {
        self.tx.send(SseEvent::Complete { data }).await
    }
    
    pub async fn send_error(&self, message: String, details: Option<serde_json::Value>) -> Result<(), mpsc::error::SendError<SseEvent>> {
        self.tx.send(SseEvent::Error { message, details }).await
    }
}

// Helper to create SSE response for long-running operations
pub async fn stream_operation<F, Fut>(operation: F) -> SseStream
where
    F: FnOnce(SseEventSender) -> Fut + Send + 'static,
    Fut: std::future::Future<Output = ()> + Send,
{
    let (sender, sse) = SseEventSender::new();
    
    // Run operation in background
    tokio::spawn(async move {
        operation(sender).await;
    });
    
    sse
}