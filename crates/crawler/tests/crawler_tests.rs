use crawler::{
    CrawlerConfig, CrawlOptions, Crawler, ContentExtractor, MarkdownConverter,
    RateLimitConfig, WaitFor,
};
use url::Url;
use wiremock::{MockServer, Mock, ResponseTemplate};
use wiremock::matchers::{method, path};

#[tokio::test]
async fn test_single_page_crawl() {
    // Start a mock server
    let mock_server = MockServer::start().await;
    
    // Configure the mock
    Mock::given(method("GET"))
        .and(path("/test"))
        .respond_with(ResponseTemplate::new(200)
            .set_body_string(r#"
                <html>
                <head>
                    <title>Test Page</title>
                    <meta name="description" content="Test description">
                </head>
                <body>
                    <h1>Hello World</h1>
                    <p>This is a test page.</p>
                    <a href="/page2">Link to page 2</a>
                </body>
                </html>
            "#))
        .mount(&mock_server)
        .await;
    
    // Create crawler
    let crawler = Crawler::new().unwrap();
    let url = Url::parse(&format!("{}/test", mock_server.uri())).unwrap();
    let options = CrawlOptions::single_page();
    
    // Crawl the page
    let result = crawler.crawl(url, options).await.unwrap();
    
    // Verify results
    assert_eq!(result.title, "Test Page");
    assert!(result.text.contains("Hello World"));
    assert!(result.text.contains("This is a test page"));
    assert_eq!(result.metadata.description, Some("Test description".to_string()));
    assert_eq!(result.links.len(), 1);
}

#[tokio::test]
async fn test_markdown_conversion() {
    let html = r#"
        <h1>Main Title</h1>
        <p>This is a <strong>bold</strong> paragraph.</p>
        <h2>Subtitle</h2>
        <ul>
            <li>Item 1</li>
            <li>Item 2</li>
        </ul>
        <p>Link to <a href="https://example.com">example</a>.</p>
    "#;
    
    let markdown = MarkdownConverter::convert(html);
    
    assert!(markdown.contains("# Main Title"));
    assert!(markdown.contains("**bold**"));
    assert!(markdown.contains("## Subtitle"));
    assert!(markdown.contains("- Item 1"));
    assert!(markdown.contains("[example](https://example.com)"));
}

#[tokio::test]
async fn test_rate_limiting() {
    use std::time::Instant;
    
    let mock_server = MockServer::start().await;
    
    Mock::given(method("GET"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(200).set_body_string("<html></html>"))
        .mount(&mock_server)
        .await;
    
    // Configure aggressive rate limiting
    let config = CrawlerConfig {
        rate_limit: RateLimitConfig {
            requests_per_second: 1.0,
            burst_size: 1,
            per_domain: false,
        },
        ..Default::default()
    };
    
    let crawler = Crawler::with_config(config);
    let url = Url::parse(&mock_server.uri()).unwrap();
    let options = CrawlOptions::single_page();
    
    // First request should be immediate
    let start = Instant::now();
    crawler.crawl(url.clone(), options.clone()).await.unwrap();
    let first_duration = start.elapsed();
    
    // Second request should be delayed
    let start = Instant::now();
    crawler.crawl(url, options).await.unwrap();
    let second_duration = start.elapsed();
    
    // Second request should take at least 1 second due to rate limiting
    assert!(second_duration.as_millis() >= 900); // Allow some tolerance
}

#[tokio::test]
async fn test_robots_txt_handling() {
    let mock_server = MockServer::start().await;
    
    // Mock robots.txt
    Mock::given(method("GET"))
        .and(path("/robots.txt"))
        .respond_with(ResponseTemplate::new(200)
            .set_body_string("User-agent: *\nDisallow: /private/"))
        .mount(&mock_server)
        .await;
    
    // Mock allowed page
    Mock::given(method("GET"))
        .and(path("/public"))
        .respond_with(ResponseTemplate::new(200)
            .set_body_string("<html><body>Public content</body></html>"))
        .mount(&mock_server)
        .await;
    
    let crawler = Crawler::new().unwrap();
    let url = Url::parse(&format!("{}/public", mock_server.uri())).unwrap();
    let options = CrawlOptions::single_page();
    
    // This should succeed
    let result = crawler.crawl(url, options).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_content_extraction() {
    let html = r#"
        <html>
        <head>
            <title>Article Title</title>
            <meta property="og:title" content="Open Graph Title">
            <meta property="og:description" content="OG Description">
            <meta name="author" content="John Doe">
        </head>
        <body>
            <article>
                <h1>Article Headline</h1>
                <time datetime="2024-01-01">January 1, 2024</time>
                <p>Article content goes here.</p>
            </article>
            <img src="/image.jpg" alt="Test image">
        </body>
        </html>
    "#;
    
    let extractor = ContentExtractor::new();
    let url = Url::parse("https://example.com/article").unwrap();
    let content = extractor.extract(&html.as_bytes().into(), url).unwrap();
    
    assert_eq!(content.title, "Open Graph Title");
    assert_eq!(content.metadata.open_graph.get("og:title"), Some(&"Open Graph Title".to_string()));
    assert_eq!(content.metadata.author, Some("John Doe".to_string()));
    assert_eq!(content.images.len(), 1);
    assert!(content.text.contains("Article content goes here"));
}

#[tokio::test]
async fn test_parallel_crawling() {
    let mock_server = MockServer::start().await;
    
    // Set up multiple endpoints
    for i in 1..=3 {
        Mock::given(method("GET"))
            .and(path(format!("/page{}", i)))
            .respond_with(ResponseTemplate::new(200)
                .set_body_string(format!("<html><body>Page {}</body></html>", i)))
            .mount(&mock_server)
            .await;
    }
    
    let urls: Vec<Url> = (1..=3)
        .map(|i| Url::parse(&format!("{}/page{}", mock_server.uri(), i)).unwrap())
        .collect();
    
    let config = CrawlerConfig {
        concurrent_requests: 3,
        ..Default::default()
    };
    
    let parallel_crawler = crawler::ParallelCrawler::new(config, 3).unwrap();
    let results = parallel_crawler.crawl_urls(urls).await;
    
    assert_eq!(results.len(), 3);
    
    for (i, result) in results.iter().enumerate() {
        match result {
            crawler::CrawlResult::Success { content, .. } => {
                assert!(content.text.contains(&format!("Page {}", i + 1)));
            }
            _ => panic!("Expected success"),
        }
    }
}