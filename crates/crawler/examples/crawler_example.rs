use crawler::{
    Crawler, CrawlerConfig, CrawlOptions, CrawlResult, RateLimitConfig,
    ArticleExtractor, ProductExtractor, WaitFor,
};
use std::time::Duration;
use url::Url;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    println!("=== Zero-Copy Web Crawler Example ===\n");
    
    // Example 1: Basic single page crawl
    basic_crawl_example().await?;
    
    // Example 2: Site-wide crawl with options
    site_crawl_example().await?;
    
    // Example 3: Polite crawling with rate limiting
    polite_crawl_example().await?;
    
    // Example 4: Extract structured data
    structured_data_example().await?;
    
    // Example 5: Parallel crawling
    parallel_crawl_example().await?;
    
    Ok(())
}

async fn basic_crawl_example() -> anyhow::Result<()> {
    println!("1. Basic Single Page Crawl");
    println!("--------------------------");
    
    let crawler = Crawler::new()?;
    let url = Url::parse("https://example.com")?;
    let options = CrawlOptions::single_page();
    
    match crawler.crawl(url.clone(), options).await {
        Ok(content) => {
            println!("URL: {}", content.url);
            println!("Title: {}", content.title);
            println!("Text length: {} chars", content.text.len());
            println!("Links found: {}", content.links.len());
            println!("Images found: {}", content.images.len());
            
            if let Some(desc) = &content.metadata.description {
                println!("Description: {}", desc);
            }
        }
        Err(e) => {
            println!("Crawl failed: {}", e);
        }
    }
    
    println!();
    Ok(())
}

async fn site_crawl_example() -> anyhow::Result<()> {
    println!("2. Site-Wide Crawl");
    println!("------------------");
    
    let crawler = Crawler::new()?;
    let start_url = Url::parse("https://example.com")?;
    
    let options = CrawlOptions {
        max_depth: 2,
        max_pages: 10,
        same_domain_only: true,
        extract_links: true,
        extract_metadata: true,
        ..Default::default()
    };
    
    let mut receiver = crawler.crawl_site(start_url, options).await?;
    let mut success_count = 0;
    let mut error_count = 0;
    
    println!("Starting site crawl...");
    
    while let Some(result) = receiver.recv().await {
        match result {
            CrawlResult::Success { url, content, depth } => {
                success_count += 1;
                println!("✓ [Depth {}] {}: {}", depth, url, content.title);
            }
            CrawlResult::Error { url, error, depth } => {
                error_count += 1;
                println!("✗ [Depth {}] {}: {}", depth, url, error);
            }
        }
    }
    
    println!("\nCrawl complete: {} success, {} errors", success_count, error_count);
    println!();
    Ok(())
}

async fn polite_crawl_example() -> anyhow::Result<()> {
    println!("3. Polite Crawling");
    println!("------------------");
    
    // Configure polite crawling
    let config = CrawlerConfig::polite()
        .with_user_agent("PoliteBot/1.0 (Educational)")
        .with_header("From".to_string(), "example@example.com".to_string());
    
    println!("Crawler configuration:");
    println!("- Rate limit: {} req/s", config.rate_limit.requests_per_second);
    println!("- Concurrent requests: {}", config.concurrent_requests);
    println!("- Timeout: {:?}", config.timeout);
    println!("- Respects robots.txt: {}", config.respect_robots_txt);
    
    let crawler = Crawler::with_config(config);
    
    // Crawl with JavaScript waiting
    let options = CrawlOptions::single_page()
        .with_javascript();
    
    println!("\nCrawling with JavaScript support...");
    
    // In a real scenario, would crawl actual page
    println!("(Crawl would happen here)");
    
    println!();
    Ok(())
}

async fn structured_data_example() -> anyhow::Result<()> {
    println!("4. Extract Structured Data");
    println!("--------------------------");
    
    // Simulate crawled content with structured data
    let mock_content = crawler::ExtractedContent {
        url: Url::parse("https://example.com/article")?,
        title: "Example Article".to_string(),
        text: "This is a long article about web crawling. ".repeat(50),
        markdown: "# Example Article\n\nContent here...".to_string(),
        links: vec![],
        images: vec![],
        metadata: crawler::ExtractedMetadata {
            description: Some("Article about crawling".to_string()),
            keywords: vec!["crawler".to_string(), "rust".to_string()],
            author: Some("John Doe".to_string()),
            published_date: Some("2024-01-01".to_string()),
            modified_date: None,
            language: Some("en".to_string()),
            open_graph: std::collections::HashMap::new(),
            twitter_card: std::collections::HashMap::new(),
        },
        code_blocks: vec![],
        structured_data: std::collections::HashMap::new(),
    };
    
    // Try to extract as article
    if let Some(article) = ArticleExtractor::extract(&mock_content) {
        println!("Extracted Article:");
        println!("- Headline: {}", article.headline);
        println!("- Author: {:?}", article.author);
        println!("- Published: {:?}", article.published_date);
        println!("- Word count: {}", article.word_count);
        println!("- Reading time: {} min", article.reading_time);
        println!("- Tags: {:?}", article.tags);
    }
    
    println!();
    Ok(())
}

async fn parallel_crawl_example() -> anyhow::Result<()> {
    println!("5. Parallel Crawling");
    println!("--------------------");
    
    let urls = vec![
        "https://example.com/page1",
        "https://example.com/page2",
        "https://example.com/page3",
        "https://example.com/page4",
        "https://example.com/page5",
    ].into_iter()
        .map(|u| Url::parse(u).unwrap())
        .collect();
    
    // Configure for parallel crawling
    let config = CrawlerConfig {
        concurrent_requests: 5,
        rate_limit: RateLimitConfig {
            requests_per_second: 10.0,
            burst_size: 5,
            per_domain: true,
        },
        ..Default::default()
    };
    
    println!("Crawling {} URLs in parallel...", 5);
    println!("Max concurrent: {}", config.concurrent_requests);
    
    let parallel_crawler = crawler::ParallelCrawler::new(config, 5)?;
    let start = std::time::Instant::now();
    
    let results = parallel_crawler.crawl_urls(urls).await;
    
    let elapsed = start.elapsed();
    let success_count = results.iter().filter(|r| matches!(r, CrawlResult::Success { .. })).count();
    
    println!("Completed in {:?}", elapsed);
    println!("Success: {}/{}", success_count, results.len());
    
    // Show performance metrics
    let urls_per_second = results.len() as f64 / elapsed.as_secs_f64();
    println!("Performance: {:.2} URLs/second", urls_per_second);
    
    println!();
    Ok(())
}

// Example: Custom content extractor
struct RecipeExtractor;

impl RecipeExtractor {
    fn extract(content: &crawler::ExtractedContent) -> Option<Recipe> {
        // Look for recipe schema.org data
        for (_, data) in &content.structured_data {
            if data.get("@type") == Some(&serde_json::json!("Recipe")) {
                return Some(Recipe {
                    name: data.get("name")?.as_str()?.to_string(),
                    ingredients: vec![], // Would extract from data
                    instructions: vec![], // Would extract from data
                    prep_time: data.get("prepTime").and_then(|v| v.as_str()).map(|s| s.to_string()),
                    cook_time: data.get("cookTime").and_then(|v| v.as_str()).map(|s| s.to_string()),
                });
            }
        }
        None
    }
}

struct Recipe {
    name: String,
    ingredients: Vec<String>,
    instructions: Vec<String>,
    prep_time: Option<String>,
    cook_time: Option<String>,
}