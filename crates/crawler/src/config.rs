use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrawlerConfig {
    pub user_agent: String,
    pub timeout: Duration,
    pub max_redirects: u32,
    pub max_content_size: usize,
    pub concurrent_requests: usize,
    pub rate_limit: RateLimitConfig,
    pub respect_robots_txt: bool,
    pub allowed_content_types: Vec<String>,
    pub headers: Vec<(String, String)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    pub requests_per_second: f64,
    pub burst_size: u32,
    pub per_domain: bool,
}

impl Default for CrawlerConfig {
    fn default() -> Self {
        Self {
            user_agent: "Mozilla/5.0 (compatible; MCP-Crawler/1.0; +https://github.com/mcp-crawler)".to_string(),
            timeout: Duration::from_secs(30),
            max_redirects: 5,
            max_content_size: 10 * 1024 * 1024, // 10MB
            concurrent_requests: 10,
            rate_limit: RateLimitConfig::default(),
            respect_robots_txt: true,
            allowed_content_types: vec![
                "text/html".to_string(),
                "text/plain".to_string(),
                "application/xhtml+xml".to_string(),
            ],
            headers: vec![
                ("Accept".to_string(), "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8".to_string()),
                ("Accept-Language".to_string(), "en-US,en;q=0.5".to_string()),
                ("Accept-Encoding".to_string(), "gzip, deflate, br".to_string()),
            ],
        }
    }
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            requests_per_second: 2.0,
            burst_size: 5,
            per_domain: true,
        }
    }
}

impl CrawlerConfig {
    pub fn aggressive() -> Self {
        Self {
            concurrent_requests: 50,
            rate_limit: RateLimitConfig {
                requests_per_second: 10.0,
                burst_size: 20,
                per_domain: true,
            },
            ..Default::default()
        }
    }
    
    pub fn polite() -> Self {
        Self {
            concurrent_requests: 2,
            rate_limit: RateLimitConfig {
                requests_per_second: 0.5,
                burst_size: 1,
                per_domain: true,
            },
            timeout: Duration::from_secs(60),
            ..Default::default()
        }
    }
    
    pub fn with_user_agent(mut self, user_agent: String) -> Self {
        self.user_agent = user_agent;
        self
    }
    
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
    
    pub fn with_header(mut self, key: String, value: String) -> Self {
        self.headers.push((key, value));
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrawlOptions {
    pub max_depth: u32,
    pub max_pages: usize,
    pub same_domain_only: bool,
    pub follow_redirects: bool,
    pub extract_links: bool,
    pub extract_metadata: bool,
    pub wait_for: WaitFor,
    pub screenshot: bool,
    pub javascript_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WaitFor {
    Load,
    DomContentLoaded,
    NetworkIdle,
    Selector(String),
    Time(u64),
}

impl Default for CrawlOptions {
    fn default() -> Self {
        Self {
            max_depth: 3,
            max_pages: 100,
            same_domain_only: true,
            follow_redirects: true,
            extract_links: true,
            extract_metadata: true,
            wait_for: WaitFor::DomContentLoaded,
            screenshot: false,
            javascript_enabled: false,
        }
    }
}

impl CrawlOptions {
    pub fn single_page() -> Self {
        Self {
            max_depth: 0,
            max_pages: 1,
            extract_links: false,
            ..Default::default()
        }
    }
    
    pub fn site_wide() -> Self {
        Self {
            max_depth: 10,
            max_pages: 10000,
            same_domain_only: true,
            ..Default::default()
        }
    }
    
    pub fn with_javascript(mut self) -> Self {
        self.javascript_enabled = true;
        self.wait_for = WaitFor::NetworkIdle;
        self
    }
}