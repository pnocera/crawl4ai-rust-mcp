use thiserror::Error;

#[derive(Error, Debug)]
pub enum CrawlerError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
    
    #[error("URL parse error: {0}")]
    UrlParse(#[from] url::ParseError),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Invalid URL: {0}")]
    InvalidUrl(String),
    
    #[error("Rate limit exceeded for domain: {0}")]
    RateLimited(String),
    
    #[error("Robots.txt disallows crawling: {0}")]
    RobotsDisallowed(String),
    
    #[error("Content too large: {size} bytes (max: {max})")]
    ContentTooLarge { size: usize, max: usize },
    
    #[error("Unsupported content type: {0}")]
    UnsupportedContentType(String),
    
    #[error("Parse error: {0}")]
    ParseError(String),
    
    #[error("Timeout after {0} seconds")]
    Timeout(u64),
    
    #[error("Max redirects ({0}) exceeded")]
    TooManyRedirects(u32),
    
    #[error("Crawl depth exceeded: {current} (max: {max})")]
    DepthExceeded { current: u32, max: u32 },
}

pub type Result<T> = std::result::Result<T, CrawlerError>;