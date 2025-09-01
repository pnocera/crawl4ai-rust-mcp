use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, warn};
use url::Url;

use crate::{HttpClient, Result, CrawlerError};

#[derive(Debug, Clone)]
pub struct RobotsChecker {
    client: Arc<HttpClient>,
    cache: Arc<RwLock<HashMap<String, CachedRobots>>>,
    user_agent: String,
    cache_duration: Duration,
}

#[derive(Debug, Clone)]
struct CachedRobots {
    rules: RobotsRules,
    fetched_at: Instant,
}

#[derive(Debug, Clone, Default)]
struct RobotsRules {
    disallow_paths: Vec<String>,
    allow_paths: Vec<String>,
    crawl_delay: Option<Duration>,
    sitemap: Option<String>,
}

impl RobotsChecker {
    pub fn new(client: Arc<HttpClient>, user_agent: String) -> Self {
        Self {
            client,
            cache: Arc::new(RwLock::new(HashMap::new())),
            user_agent,
            cache_duration: Duration::from_secs(3600), // 1 hour cache
        }
    }
    
    pub async fn is_allowed(&self, url: &Url) -> Result<bool> {
        // Get robots.txt URL
        let robots_url = self.get_robots_url(url)?;
        
        // Check cache
        let rules = self.get_or_fetch_rules(&robots_url).await?;
        
        // Check if path is allowed
        let path = url.path();
        Ok(self.check_path_allowed(path, &rules))
    }
    
    pub async fn get_crawl_delay(&self, url: &Url) -> Result<Option<Duration>> {
        let robots_url = self.get_robots_url(url)?;
        let rules = self.get_or_fetch_rules(&robots_url).await?;
        Ok(rules.crawl_delay)
    }
    
    pub async fn get_sitemap(&self, url: &Url) -> Result<Option<String>> {
        let robots_url = self.get_robots_url(url)?;
        let rules = self.get_or_fetch_rules(&robots_url).await?;
        Ok(rules.sitemap)
    }
    
    fn get_robots_url(&self, url: &Url) -> Result<Url> {
        let mut robots_url = url.clone();
        robots_url.set_path("/robots.txt");
        robots_url.set_query(None);
        robots_url.set_fragment(None);
        Ok(robots_url)
    }
    
    async fn get_or_fetch_rules(&self, robots_url: &Url) -> Result<RobotsRules> {
        let domain = robots_url.domain()
            .ok_or_else(|| CrawlerError::InvalidUrl("No domain".to_string()))?;
        
        // Check cache
        {
            let cache = self.cache.read().await;
            if let Some(cached) = cache.get(domain) {
                if cached.fetched_at.elapsed() < self.cache_duration {
                    debug!("Using cached robots.txt for {}", domain);
                    return Ok(cached.rules.clone());
                }
            }
        }
        
        // Fetch new rules
        debug!("Fetching robots.txt for {}", domain);
        let rules = self.fetch_and_parse_robots(robots_url).await?;
        
        // Update cache
        {
            let mut cache = self.cache.write().await;
            cache.insert(domain.to_string(), CachedRobots {
                rules: rules.clone(),
                fetched_at: Instant::now(),
            });
        }
        
        Ok(rules)
    }
    
    async fn fetch_and_parse_robots(&self, robots_url: &Url) -> Result<RobotsRules> {
        match self.client.fetch_bytes(robots_url).await {
            Ok(bytes) => {
                let content = String::from_utf8_lossy(&bytes);
                Ok(self.parse_robots_txt(&content))
            }
            Err(e) => {
                warn!("Failed to fetch robots.txt from {}: {}", robots_url, e);
                // If we can't fetch robots.txt, assume everything is allowed
                Ok(RobotsRules::default())
            }
        }
    }
    
    fn parse_robots_txt(&self, content: &str) -> RobotsRules {
        let mut rules = RobotsRules::default();
        let mut in_user_agent_section = false;
        let mut found_wildcard = false;
        
        for line in content.lines() {
            let line = line.trim();
            
            // Skip comments and empty lines
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            
            // Parse directive
            let parts: Vec<&str> = line.splitn(2, ':').collect();
            if parts.len() != 2 {
                continue;
            }
            
            let directive = parts[0].trim().to_lowercase();
            let value = parts[1].trim();
            
            match directive.as_str() {
                "user-agent" => {
                    // Check if this section applies to us
                    in_user_agent_section = value == "*" || 
                        value.to_lowercase() == self.user_agent.to_lowercase();
                    
                    if value == "*" {
                        found_wildcard = true;
                    }
                }
                "disallow" if in_user_agent_section => {
                    if !value.is_empty() {
                        rules.disallow_paths.push(value.to_string());
                    }
                }
                "allow" if in_user_agent_section => {
                    if !value.is_empty() {
                        rules.allow_paths.push(value.to_string());
                    }
                }
                "crawl-delay" if in_user_agent_section => {
                    if let Ok(seconds) = value.parse::<u64>() {
                        rules.crawl_delay = Some(Duration::from_secs(seconds));
                    }
                }
                "sitemap" => {
                    // Sitemap directive applies globally
                    rules.sitemap = Some(value.to_string());
                }
                _ => {}
            }
        }
        
        // If no specific rules for our user agent, but wildcard exists, use those
        if rules.disallow_paths.is_empty() && rules.allow_paths.is_empty() && found_wildcard {
            // Re-parse for wildcard rules
            in_user_agent_section = false;
            
            for line in content.lines() {
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') {
                    continue;
                }
                
                let parts: Vec<&str> = line.splitn(2, ':').collect();
                if parts.len() != 2 {
                    continue;
                }
                
                let directive = parts[0].trim().to_lowercase();
                let value = parts[1].trim();
                
                match directive.as_str() {
                    "user-agent" => {
                        in_user_agent_section = value == "*";
                    }
                    "disallow" if in_user_agent_section => {
                        if !value.is_empty() {
                            rules.disallow_paths.push(value.to_string());
                        }
                    }
                    "allow" if in_user_agent_section => {
                        if !value.is_empty() {
                            rules.allow_paths.push(value.to_string());
                        }
                    }
                    _ => {}
                }
            }
        }
        
        rules
    }
    
    fn check_path_allowed(&self, path: &str, rules: &RobotsRules) -> bool {
        // Check allow rules first (they take precedence)
        for allow_path in &rules.allow_paths {
            if self.path_matches(path, allow_path) {
                return true;
            }
        }
        
        // Check disallow rules
        for disallow_path in &rules.disallow_paths {
            if self.path_matches(path, disallow_path) {
                return false;
            }
        }
        
        // Default to allowed
        true
    }
    
    fn path_matches(&self, path: &str, pattern: &str) -> bool {
        // Support robots.txt wildcards: * (matches any sequence) and $ (end of string)
        if pattern.is_empty() {
            return true;
        }
        
        // If pattern ends with $, path must match exactly to the end
        if let Some(stripped_pattern) = pattern.strip_suffix('$') {
            return self.pattern_matches_exact(path, stripped_pattern);
        }
        
        // Pattern may contain wildcards
        self.pattern_matches_wildcard(path, pattern)
    }
    
    fn pattern_matches_exact(&self, path: &str, pattern: &str) -> bool {
        if !pattern.contains('*') {
            return path == pattern;
        }
        
        let parts: Vec<&str> = pattern.split('*').collect();
        let mut path_pos = 0;
        
        for (i, part) in parts.iter().enumerate() {
            if i == 0 {
                // First part must match from the beginning
                if !path[path_pos..].starts_with(part) {
                    return false;
                }
                path_pos += part.len();
            } else if i == parts.len() - 1 && !part.is_empty() {
                // Last part must match at the end (for $ patterns)
                return path.ends_with(part);
            } else if !part.is_empty() {
                // Find this part in the remaining path
                if let Some(found_pos) = path[path_pos..].find(part) {
                    path_pos += found_pos + part.len();
                } else {
                    return false;
                }
            }
        }
        
        true
    }
    
    fn pattern_matches_wildcard(&self, path: &str, pattern: &str) -> bool {
        if !pattern.contains('*') {
            return path.starts_with(pattern);
        }
        
        let parts: Vec<&str> = pattern.split('*').collect();
        let mut path_pos = 0;
        
        for (i, part) in parts.iter().enumerate() {
            if i == 0 {
                // First part must match from the beginning
                if !path[path_pos..].starts_with(part) {
                    return false;
                }
                path_pos += part.len();
            } else if !part.is_empty() {
                // Find this part in the remaining path
                if let Some(found_pos) = path[path_pos..].find(part) {
                    path_pos += found_pos + part.len();
                } else {
                    return false;
                }
            }
        }
        
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_robots_txt_parsing() {
        let checker = RobotsChecker {
            client: Arc::new(HttpClient::new(Arc::new(Default::default())).unwrap()),
            cache: Arc::new(RwLock::new(HashMap::new())),
            user_agent: "TestBot".to_string(),
            cache_duration: Duration::from_secs(3600),
        };
        
        let robots_txt = r#"
User-agent: *
Disallow: /admin/
Disallow: /private/
Allow: /private/public/

User-agent: TestBot
Disallow: /test/
Crawl-delay: 2

Sitemap: https://example.com/sitemap.xml
"#;
        
        let rules = checker.parse_robots_txt(robots_txt);
        
        assert_eq!(rules.disallow_paths, vec!["/test/"]);
        assert_eq!(rules.crawl_delay, Some(Duration::from_secs(2)));
        assert_eq!(rules.sitemap, Some("https://example.com/sitemap.xml".to_string()));
    }
    
    #[test]
    fn test_path_matching() {
        let checker = RobotsChecker {
            client: Arc::new(HttpClient::new(Arc::new(Default::default())).unwrap()),
            cache: Arc::new(RwLock::new(HashMap::new())),
            user_agent: "TestBot".to_string(),
            cache_duration: Duration::from_secs(3600),
        };
        
        let rules = RobotsRules {
            disallow_paths: vec!["/admin/".to_string(), "/private/".to_string()],
            allow_paths: vec!["/private/public/".to_string()],
            crawl_delay: None,
            sitemap: None,
        };
        
        assert!(!checker.check_path_allowed("/admin/users", &rules));
        assert!(!checker.check_path_allowed("/private/data", &rules));
        assert!(checker.check_path_allowed("/private/public/info", &rules));
        assert!(checker.check_path_allowed("/public/page", &rules));
    }
}