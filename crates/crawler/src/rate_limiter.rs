use governor::{
    clock::DefaultClock,
    middleware::NoOpMiddleware,
    state::{InMemoryState, NotKeyed},
    Jitter, Quota, RateLimiter as GovernorRateLimiter,
};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::debug;
use url::Url;

use crate::{CrawlerError, RateLimitConfig, Result};

pub type RateLimiterImpl = GovernorRateLimiter<NotKeyed, InMemoryState, DefaultClock, NoOpMiddleware>;

pub struct RateLimiter {
    config: RateLimitConfig,
    global_limiter: Arc<RateLimiterImpl>,
    domain_limiters: Arc<RwLock<HashMap<String, Arc<RateLimiterImpl>>>>,
}

impl RateLimiter {
    pub fn new(config: RateLimitConfig) -> Self {
        let quota = Self::create_quota(&config);
        let global_limiter = Arc::new(GovernorRateLimiter::direct(quota));
        
        Self {
            config,
            global_limiter,
            domain_limiters: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    pub async fn check_and_wait(&self, url: &Url) -> Result<()> {
        if self.config.per_domain {
            self.check_domain_limit(url).await
        } else {
            self.check_global_limit().await
        }
    }
    
    async fn check_global_limit(&self) -> Result<()> {
        debug!("Checking global rate limit");
        
        self.global_limiter
            .until_ready_with_jitter(Jitter::up_to(Duration::from_millis(100)))
            .await;
        
        Ok(())
    }
    
    async fn check_domain_limit(&self, url: &Url) -> Result<()> {
        let domain = url.domain()
            .ok_or_else(|| CrawlerError::InvalidUrl("No domain found".to_string()))?;
        
        debug!("Checking rate limit for domain: {}", domain);
        
        // Get or create domain limiter
        let limiter = {
            let mut limiters = self.domain_limiters.write().await;
            
            limiters
                .entry(domain.to_string())
                .or_insert_with(|| {
                    let quota = Self::create_quota(&self.config);
                    Arc::new(GovernorRateLimiter::direct(quota))
                })
                .clone()
        };
        
        limiter
            .until_ready_with_jitter(Jitter::up_to(Duration::from_millis(100)))
            .await;
        
        Ok(())
    }
    
    fn create_quota(config: &RateLimitConfig) -> Quota {
        let cells = nonzero_ext::nonzero!(1u32);
        let period = Duration::from_secs_f64(1.0 / config.requests_per_second);
        
        Quota::with_period(period)
            .unwrap()
            .allow_burst(config.burst_size.try_into().unwrap())
    }
    
    pub async fn report_success(&self, url: &Url) {
        debug!("Request successful for: {}", url);
        // Could track success metrics here
    }
    
    pub async fn report_failure(&self, url: &Url, error: &CrawlerError) {
        debug!("Request failed for {}: {}", url, error);
        // Could implement backoff logic here
    }
    
    pub async fn get_stats(&self) -> RateLimiterStats {
        let domain_count = self.domain_limiters.read().await.len();
        
        RateLimiterStats {
            config: self.config.clone(),
            domains_tracked: domain_count,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RateLimiterStats {
    pub config: RateLimitConfig,
    pub domains_tracked: usize,
}

// Adaptive rate limiter that adjusts based on response times
pub struct AdaptiveRateLimiter {
    base_limiter: RateLimiter,
    response_times: Arc<RwLock<HashMap<String, ResponseTimeStats>>>,
    adaptation_config: AdaptationConfig,
}

#[derive(Debug, Clone)]
pub struct AdaptationConfig {
    pub target_response_time: Duration,
    pub increase_factor: f64,
    pub decrease_factor: f64,
    pub min_rate: f64,
    pub max_rate: f64,
}

impl Default for AdaptationConfig {
    fn default() -> Self {
        Self {
            target_response_time: Duration::from_millis(500),
            increase_factor: 1.1,
            decrease_factor: 0.9,
            min_rate: 0.1,
            max_rate: 100.0,
        }
    }
}

#[derive(Debug, Clone)]
struct ResponseTimeStats {
    total_time: Duration,
    request_count: u64,
    last_adjustment: std::time::Instant,
}

impl AdaptiveRateLimiter {
    pub fn new(base_config: RateLimitConfig, adaptation_config: AdaptationConfig) -> Self {
        Self {
            base_limiter: RateLimiter::new(base_config),
            response_times: Arc::new(RwLock::new(HashMap::new())),
            adaptation_config,
        }
    }
    
    pub async fn check_and_wait(&self, url: &Url) -> Result<()> {
        self.base_limiter.check_and_wait(url).await
    }
    
    pub async fn record_response_time(&self, url: &Url, response_time: Duration) {
        let domain = match url.domain() {
            Some(d) => d,
            None => return,
        };
        
        let mut stats = self.response_times.write().await;
        let entry = stats.entry(domain.to_string()).or_insert(ResponseTimeStats {
            total_time: Duration::ZERO,
            request_count: 0,
            last_adjustment: std::time::Instant::now(),
        });
        
        entry.total_time += response_time;
        entry.request_count += 1;
        
        // Check if we should adjust the rate
        if entry.last_adjustment.elapsed() > Duration::from_secs(10) && entry.request_count > 10 {
            let avg_response_time = entry.total_time / entry.request_count as u32;
            
            if avg_response_time > self.adaptation_config.target_response_time {
                // Slow down
                self.adjust_rate(domain, self.adaptation_config.decrease_factor).await;
            } else if avg_response_time < self.adaptation_config.target_response_time / 2 {
                // Speed up
                self.adjust_rate(domain, self.adaptation_config.increase_factor).await;
            }
            
            // Reset stats
            entry.total_time = Duration::ZERO;
            entry.request_count = 0;
            entry.last_adjustment = std::time::Instant::now();
        }
    }
    
    async fn adjust_rate(&self, domain: &str, factor: f64) {
        debug!("Adjusting rate for {} by factor {}", domain, factor);
        // In a real implementation, this would update the rate limiter's quota
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_rate_limiter() {
        let config = RateLimitConfig {
            requests_per_second: 2.0,
            burst_size: 1,
            per_domain: false,
        };
        
        let limiter = RateLimiter::new(config);
        let url = Url::parse("https://example.com").unwrap();
        
        // Should allow first request immediately
        let start = std::time::Instant::now();
        limiter.check_and_wait(&url).await.unwrap();
        assert!(start.elapsed() < Duration::from_millis(50));
        
        // Second request should be delayed
        let start = std::time::Instant::now();
        limiter.check_and_wait(&url).await.unwrap();
        assert!(start.elapsed() >= Duration::from_millis(400)); // ~0.5 seconds
    }
    
    #[tokio::test]
    async fn test_domain_rate_limiting() {
        let config = RateLimitConfig {
            requests_per_second: 1.0,
            burst_size: 1,
            per_domain: true,
        };
        
        let limiter = RateLimiter::new(config);
        let url1 = Url::parse("https://example.com").unwrap();
        let url2 = Url::parse("https://other.com").unwrap();
        
        // Both domains should allow immediate first request
        limiter.check_and_wait(&url1).await.unwrap();
        limiter.check_and_wait(&url2).await.unwrap();
        
        let stats = limiter.get_stats().await;
        assert_eq!(stats.domains_tracked, 2);
    }
}