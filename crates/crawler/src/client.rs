use bytes::{Bytes, BytesMut};
use futures::{Stream, StreamExt, TryStreamExt};
use reqwest::{Client as ReqwestClient, Response, StatusCode};
use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, info, warn};
use url::Url;

use crate::{CrawlerConfig, CrawlerError, Result};

// Zero-copy HTTP client wrapper
#[derive(Debug)]
pub struct HttpClient {
    client: ReqwestClient,
    config: Arc<CrawlerConfig>,
}

impl HttpClient {
    pub fn new(config: Arc<CrawlerConfig>) -> Result<Self> {
        let mut headers = reqwest::header::HeaderMap::new();
        
        // Add default headers
        for (key, value) in &config.headers {
            headers.insert(
                reqwest::header::HeaderName::from_bytes(key.as_bytes())
                    .map_err(|e| CrawlerError::ParseError(e.to_string()))?,
                reqwest::header::HeaderValue::from_str(value)
                    .map_err(|e| CrawlerError::ParseError(e.to_string()))?,
            );
        }
        
        let client = ReqwestClient::builder()
            .user_agent(&config.user_agent)
            .timeout(config.timeout)
            .default_headers(headers)
            .redirect(reqwest::redirect::Policy::limited(config.max_redirects as usize))
            .build()?;
        
        Ok(Self { client, config })
    }
    
    pub async fn fetch_bytes(&self, url: &Url) -> Result<Bytes> {
        debug!("Fetching URL: {}", url);
        
        let response = self.client.get(url.as_str()).send().await?;
        
        // Check status
        if !response.status().is_success() {
            return Err(CrawlerError::Http(
                reqwest::Error::from(response.error_for_status().unwrap_err())
            ));
        }
        
        // Check content type
        if let Some(content_type) = response.headers().get(reqwest::header::CONTENT_TYPE) {
            let content_type_str = content_type.to_str().unwrap_or("");
            if !self.is_allowed_content_type(content_type_str) {
                return Err(CrawlerError::UnsupportedContentType(
                    content_type_str.to_string()
                ));
            }
        }
        
        // Check content length
        if let Some(content_length) = response.content_length() {
            if content_length > self.config.max_content_size as u64 {
                return Err(CrawlerError::ContentTooLarge {
                    size: content_length as usize,
                    max: self.config.max_content_size,
                });
            }
        }
        
        // Stream response body with size limit
        let bytes = self.stream_limited(response).await?;
        
        info!("Fetched {} bytes from {}", bytes.len(), url);
        Ok(bytes)
    }
    
    pub async fn fetch_stream(
        &self,
        url: &Url,
    ) -> Result<impl Stream<Item = Result<Bytes>>> {
        debug!("Starting streaming fetch: {}", url);
        
        let response = self.client.get(url.as_str()).send().await?;
        
        if !response.status().is_success() {
            return Err(CrawlerError::Http(
                reqwest::Error::from(response.error_for_status().unwrap_err())
            ));
        }
        
        Ok(response
            .bytes_stream()
            .map_err(|e| CrawlerError::Http(e)))
    }
    
    async fn stream_limited(&self, response: Response) -> Result<Bytes> {
        let mut bytes = BytesMut::new();
        let mut stream = response.bytes_stream();
        let max_size = self.config.max_content_size;
        
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            
            if bytes.len() + chunk.len() > max_size {
                return Err(CrawlerError::ContentTooLarge {
                    size: bytes.len() + chunk.len(),
                    max: max_size,
                });
            }
            
            bytes.extend_from_slice(&chunk);
        }
        
        Ok(bytes.freeze())
    }
    
    fn is_allowed_content_type(&self, content_type: &str) -> bool {
        self.config.allowed_content_types.iter().any(|allowed| {
            content_type.starts_with(allowed)
        })
    }
    
    pub async fn head(&self, url: &Url) -> Result<HeadResponse> {
        let response = self.client.head(url.as_str()).send().await?;
        
        Ok(HeadResponse {
            status: response.status(),
            content_type: response
                .headers()
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok())
                .map(|s| s.to_string()),
            content_length: response.content_length(),
            last_modified: response
                .headers()
                .get(reqwest::header::LAST_MODIFIED)
                .and_then(|v| v.to_str().ok())
                .map(|s| s.to_string()),
        })
    }
}

#[derive(Debug)]
pub struct HeadResponse {
    pub status: StatusCode,
    pub content_type: Option<String>,
    pub content_length: Option<u64>,
    pub last_modified: Option<String>,
}

// Zero-copy response wrapper
pub struct CrawlResponse {
    pub url: Url,
    pub status: StatusCode,
    pub headers: reqwest::header::HeaderMap,
    pub body: Bytes,
    pub elapsed: Duration,
}

impl CrawlResponse {
    pub fn content_type(&self) -> Option<&str> {
        self.headers
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
    }
    
    pub fn is_html(&self) -> bool {
        self.content_type()
            .map(|ct| ct.contains("html"))
            .unwrap_or(false)
    }
    
    pub fn charset(&self) -> Option<&str> {
        self.content_type()
            .and_then(|ct| {
                ct.split(';')
                    .find(|part| part.trim().starts_with("charset="))
                    .and_then(|charset| charset.split('=').nth(1))
                    .map(|s| s.trim())
            })
    }
}

// Connection pool for efficient crawling
pub struct ConnectionPool {
    clients: Vec<HttpClient>,
    next_client: std::sync::atomic::AtomicUsize,
}

impl ConnectionPool {
    pub fn new(config: Arc<CrawlerConfig>, pool_size: usize) -> Result<Self> {
        let clients = (0..pool_size)
            .map(|_| HttpClient::new(config.clone()))
            .collect::<Result<Vec<_>>>()?;
        
        Ok(Self {
            clients,
            next_client: std::sync::atomic::AtomicUsize::new(0),
        })
    }
    
    pub fn get_client(&self) -> &HttpClient {
        let index = self.next_client.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        &self.clients[index % self.clients.len()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_allowed_content_types() {
        let config = Arc::new(CrawlerConfig::default());
        let client = HttpClient::new(config).unwrap();
        
        assert!(client.is_allowed_content_type("text/html"));
        assert!(client.is_allowed_content_type("text/html; charset=utf-8"));
        assert!(client.is_allowed_content_type("text/plain"));
        assert!(!client.is_allowed_content_type("image/jpeg"));
        assert!(!client.is_allowed_content_type("application/pdf"));
    }
}