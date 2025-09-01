pub mod client;
pub mod config;
pub mod error;
pub mod extractor;
pub mod orchestrator;
pub mod parser;
pub mod rate_limiter;
pub mod robots;

pub use client::*;
pub use config::*;
pub use error::*;
pub use extractor::*;
pub use orchestrator::*;
pub use parser::*;
pub use rate_limiter::*;

use std::sync::Arc;

pub struct Crawler {
    config: Arc<CrawlerConfig>,
}

impl Crawler {
    pub fn new() -> Result<Self> {
        Ok(Self {
            config: Arc::new(CrawlerConfig::default()),
        })
    }
    
    pub fn with_config(config: CrawlerConfig) -> Self {
        Self {
            config: Arc::new(config),
        }
    }
    
    pub async fn crawl(&self, url: url::Url, options: CrawlOptions) -> Result<ExtractedContent> {
        let orchestrator = CrawlOrchestrator::new((*self.config).clone(), options)?;
        orchestrator.crawl_single(url).await
    }
    
    pub async fn crawl_site(&self, start_url: url::Url, options: CrawlOptions) -> Result<tokio::sync::mpsc::Receiver<CrawlResult>> {
        let orchestrator = CrawlOrchestrator::new((*self.config).clone(), options)?;
        orchestrator.crawl_site(start_url).await
    }
}