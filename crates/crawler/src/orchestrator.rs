use futures::{stream, StreamExt};
use std::collections::{HashSet, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, RwLock, Semaphore};
use tracing::{debug, info, warn};
use url::Url;

use crate::{
    ConnectionPool, ContentExtractor, CrawlOptions, CrawlerConfig, CrawlerError,
    ExtractedContent, HttpClient, RateLimiter, Result,
    robots::RobotsChecker,
};

pub struct CrawlOrchestrator {
    config: Arc<CrawlerConfig>,
    options: CrawlOptions,
    client_pool: Arc<ConnectionPool>,
    rate_limiter: Arc<RateLimiter>,
    robots_checker: Arc<RobotsChecker>,
    extractor: Arc<ContentExtractor>,
    semaphore: Arc<Semaphore>,
}

impl CrawlOrchestrator {
    pub fn new(config: CrawlerConfig, options: CrawlOptions) -> Result<Self> {
        let config = Arc::new(config);
        let client_pool = Arc::new(ConnectionPool::new(config.clone(), 10)?);
        let rate_limiter = Arc::new(RateLimiter::new(config.rate_limit.clone()));
        
        let http_client = Arc::new(HttpClient::new(config.clone())?);
        let robots_checker = Arc::new(RobotsChecker::new(
            http_client,
            config.user_agent.clone(),
        ));
        
        let extractor = Arc::new(ContentExtractor::default());
        let semaphore = Arc::new(Semaphore::new(config.concurrent_requests));
        
        Ok(Self {
            config,
            options,
            client_pool,
            rate_limiter,
            robots_checker,
            extractor,
            semaphore,
        })
    }
    
    pub async fn crawl_single(&self, url: Url) -> Result<ExtractedContent> {
        // Check robots.txt
        if self.config.respect_robots_txt && !self.robots_checker.is_allowed(&url).await? {
            return Err(CrawlerError::RobotsDisallowed(url.to_string()));
        }
        
        // Rate limiting
        self.rate_limiter.check_and_wait(&url).await?;
        
        // Acquire semaphore permit
        let _permit = self.semaphore.acquire().await.unwrap();
        
        // Fetch content
        let client = self.client_pool.get_client();
        let start = Instant::now();
        let bytes = client.fetch_bytes(&url).await?;
        let elapsed = start.elapsed();
        
        debug!("Fetched {} in {:?}", url, elapsed);
        
        // Extract content
        let content = self.extractor.extract(&bytes, url)?;
        
        Ok(content)
    }
    
    pub async fn crawl_site(
        &self,
        start_url: Url,
    ) -> Result<mpsc::Receiver<CrawlResult>> {
        let (tx, rx) = mpsc::channel(100);
        
        let orchestrator = self.clone();
        tokio::spawn(async move {
            if let Err(e) = orchestrator.crawl_site_internal(start_url, tx).await {
                warn!("Crawl error: {}", e);
            }
        });
        
        Ok(rx)
    }
    
    async fn crawl_site_internal(
        &self,
        start_url: Url,
        tx: mpsc::Sender<CrawlResult>,
    ) -> Result<()> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let mut pages_crawled = 0;
        
        // Add start URL to queue
        queue.push_back(CrawlTask {
            url: start_url.clone(),
            depth: 0,
            referrer: None,
        });
        
        let start_domain = start_url.domain().map(|d| d.to_string());
        
        while let Some(task) = queue.pop_front() {
            // Check limits
            if pages_crawled >= self.options.max_pages {
                info!("Reached max pages limit: {}", self.options.max_pages);
                break;
            }
            
            if task.depth > self.options.max_depth {
                debug!("Skipping {} - max depth exceeded", task.url);
                continue;
            }
            
            // Check if already visited
            if visited.contains(&task.url) {
                continue;
            }
            
            // Check domain restriction
            if self.options.same_domain_only {
                if let (Some(start), Some(current)) = (&start_domain, task.url.domain()) {
                    if start != current {
                        debug!("Skipping {} - different domain", task.url);
                        continue;
                    }
                }
            }
            
            visited.insert(task.url.clone());
            
            // Crawl the page
            let result = match self.crawl_single(task.url.clone()).await {
                Ok(content) => {
                    pages_crawled += 1;
                    
                    // Extract links for further crawling
                    if self.options.extract_links && task.depth < self.options.max_depth {
                        for link in &content.links {
                            if !visited.contains(link) {
                                queue.push_back(CrawlTask {
                                    url: link.clone(),
                                    depth: task.depth + 1,
                                    referrer: Some(task.url.clone()),
                                });
                            }
                        }
                    }
                    
                    CrawlResult::Success {
                        url: task.url,
                        content,
                        depth: task.depth,
                    }
                }
                Err(e) => CrawlResult::Error {
                    url: task.url,
                    error: e,
                    depth: task.depth,
                },
            };
            
            // Send result
            if tx.send(result).await.is_err() {
                break; // Receiver dropped
            }
        }
        
        info!("Crawl completed. Pages crawled: {}", pages_crawled);
        Ok(())
    }
}

impl Clone for CrawlOrchestrator {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            options: self.options.clone(),
            client_pool: self.client_pool.clone(),
            rate_limiter: self.rate_limiter.clone(),
            robots_checker: self.robots_checker.clone(),
            extractor: self.extractor.clone(),
            semaphore: self.semaphore.clone(),
        }
    }
}

#[derive(Debug)]
struct CrawlTask {
    url: Url,
    depth: u32,
    referrer: Option<Url>,
}

#[derive(Debug)]
pub enum CrawlResult {
    Success {
        url: Url,
        content: ExtractedContent,
        depth: u32,
    },
    Error {
        url: Url,
        error: CrawlerError,
        depth: u32,
    },
}

// Parallel crawler for multiple independent URLs
pub struct ParallelCrawler {
    orchestrator: Arc<CrawlOrchestrator>,
    max_concurrent: usize,
}

impl ParallelCrawler {
    pub fn new(config: CrawlerConfig, max_concurrent: usize) -> Result<Self> {
        let options = CrawlOptions::single_page();
        let orchestrator = Arc::new(CrawlOrchestrator::new(config, options)?);
        
        Ok(Self {
            orchestrator,
            max_concurrent,
        })
    }
    
    pub async fn crawl_urls(&self, urls: Vec<Url>) -> Vec<CrawlResult> {
        let semaphore = Arc::new(Semaphore::new(self.max_concurrent));
        
        stream::iter(urls)
            .map(|url| {
                let orchestrator = self.orchestrator.clone();
                let semaphore = semaphore.clone();
                
                async move {
                    let _permit = semaphore.acquire().await.unwrap();
                    
                    match orchestrator.crawl_single(url.clone()).await {
                        Ok(content) => CrawlResult::Success {
                            url,
                            content,
                            depth: 0,
                        },
                        Err(e) => CrawlResult::Error {
                            url,
                            error: e,
                            depth: 0,
                        },
                    }
                }
            })
            .buffer_unordered(self.max_concurrent)
            .collect()
            .await
    }
}

// Sitemap crawler
pub struct SitemapCrawler {
    orchestrator: Arc<CrawlOrchestrator>,
    client: Arc<HttpClient>,
}

impl SitemapCrawler {
    pub fn new(config: CrawlerConfig) -> Result<Self> {
        let options = CrawlOptions::single_page();
        let orchestrator = Arc::new(CrawlOrchestrator::new(config.clone(), options)?);
        let client = Arc::new(HttpClient::new(Arc::new(config))?);
        
        Ok(Self {
            orchestrator,
            client,
        })
    }
    
    pub async fn crawl_from_sitemap(&self, sitemap_url: Url) -> Result<Vec<CrawlResult>> {
        // Fetch sitemap
        let sitemap_bytes = self.client.fetch_bytes(&sitemap_url).await?;
        let sitemap_content = String::from_utf8_lossy(&sitemap_bytes);
        
        // Parse URLs from sitemap (simplified - real implementation would use XML parser)
        let urls = self.parse_sitemap_urls(&sitemap_content);
        
        info!("Found {} URLs in sitemap", urls.len());
        
        // Crawl all URLs
        let parallel_crawler = ParallelCrawler {
            orchestrator: self.orchestrator.clone(),
            max_concurrent: 10,
        };
        
        Ok(parallel_crawler.crawl_urls(urls).await)
    }
    
    fn parse_sitemap_urls(&self, content: &str) -> Vec<Url> {
        let mut urls = Vec::new();
        
        // Simple regex-based parsing (in production, use proper XML parser)
        let re = regex::Regex::new(r"<loc>(.*?)</loc>").unwrap();
        
        for cap in re.captures_iter(content) {
            if let Some(url_str) = cap.get(1) {
                if let Ok(url) = Url::parse(url_str.as_str()) {
                    urls.push(url);
                }
            }
        }
        
        urls
    }
}

// Add regex dependency
use regex;