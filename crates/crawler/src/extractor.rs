use bytes::Bytes;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::debug;
use url::Url;

use crate::{HtmlParser, MarkdownConverter, Metadata, Result, SelectorParser};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedContent {
    pub url: Url,
    pub title: String,
    pub text: String,
    pub markdown: String,
    pub links: Vec<Url>,
    pub images: Vec<String>,
    pub metadata: ExtractedMetadata,
    pub code_blocks: Vec<CodeBlock>,
    pub structured_data: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedMetadata {
    pub description: Option<String>,
    pub keywords: Vec<String>,
    pub author: Option<String>,
    pub published_date: Option<String>,
    pub modified_date: Option<String>,
    pub language: Option<String>,
    pub open_graph: HashMap<String, String>,
    pub twitter_card: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeBlock {
    pub language: Option<String>,
    pub content: String,
    pub line_count: usize,
}

pub struct ContentExtractor {
    min_text_length: usize,
    extract_code: bool,
    extract_structured_data: bool,
}

impl Default for ContentExtractor {
    fn default() -> Self {
        Self {
            min_text_length: 100,
            extract_code: true,
            extract_structured_data: true,
        }
    }
}

impl ContentExtractor {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn with_min_text_length(mut self, length: usize) -> Self {
        self.min_text_length = length;
        self
    }
    
    pub fn extract(&self, html: &Bytes, url: Url) -> Result<ExtractedContent> {
        let parser = HtmlParser::parse(html, url.clone())?;
        let html_str = String::from_utf8_lossy(html);
        let selector_parser = SelectorParser::parse(&html_str, url.clone());
        
        // Extract basic content
        let text = parser.extract_text();
        let markdown = MarkdownConverter::convert(&html_str);
        let links = parser.extract_links();
        let images = selector_parser.extract_images();
        
        // Extract metadata
        let raw_metadata = parser.extract_metadata();
        let metadata = self.enhance_metadata(raw_metadata.clone(), &selector_parser);
        
        // Extract title
        let title = metadata.open_graph.get("og:title")
            .or(raw_metadata.title.as_ref())
            .cloned()
            .unwrap_or_else(|| self.extract_title_from_content(&text));
        
        // Extract code blocks if enabled
        let code_blocks = if self.extract_code {
            self.extract_code_blocks(&selector_parser)
        } else {
            vec![]
        };
        
        // Extract structured data if enabled
        let structured_data = if self.extract_structured_data {
            self.extract_structured_data(&selector_parser)
        } else {
            HashMap::new()
        };
        
        Ok(ExtractedContent {
            url,
            title,
            text,
            markdown,
            links,
            images,
            metadata,
            code_blocks,
            structured_data,
        })
    }
    
    fn enhance_metadata(
        &self,
        raw: Metadata,
        parser: &SelectorParser,
    ) -> ExtractedMetadata {
        // Extract keywords
        let keywords = raw.keywords
            .map(|k| k.split(',').map(|s| s.trim().to_string()).collect())
            .unwrap_or_default();
        
        // Extract dates
        let published_date = parser
            .select_attr("meta[property='article:published_time']", "content")
            .ok()
            .and_then(|v| v.first().cloned())
            .or_else(|| {
                parser
                    .select_attr("time[pubdate]", "datetime")
                    .ok()
                    .and_then(|v| v.first().cloned())
            });
        
        let modified_date = parser
            .select_attr("meta[property='article:modified_time']", "content")
            .ok()
            .and_then(|v| v.first().cloned());
        
        // Extract language
        let language = parser
            .select_attr("html", "lang")
            .ok()
            .and_then(|v| v.first().cloned())
            .or_else(|| {
                parser
                    .select_attr("meta[name='language']", "content")
                    .ok()
                    .and_then(|v| v.first().cloned())
            });
        
        // Extract Twitter Card metadata
        let mut twitter_card = HashMap::new();
        if let Ok(twitter_metas) = parser.select_attr("meta[name^='twitter:']", "name") {
            for name in twitter_metas {
                if let Ok(contents) = parser.select_attr(&format!("meta[name='{}']", name), "content") {
                    if let Some(content) = contents.first() {
                        twitter_card.insert(name, content.clone());
                    }
                }
            }
        }
        
        ExtractedMetadata {
            description: raw.description,
            keywords,
            author: raw.author,
            published_date,
            modified_date,
            language,
            open_graph: raw.open_graph,
            twitter_card,
        }
    }
    
    fn extract_title_from_content(&self, text: &str) -> String {
        // Take first line or first N characters as title
        text.lines()
            .next()
            .map(|line| {
                if line.len() > 100 {
                    format!("{}...", &line[..97])
                } else {
                    line.to_string()
                }
            })
            .unwrap_or_else(|| "Untitled".to_string())
    }
    
    fn extract_code_blocks(&self, parser: &SelectorParser) -> Vec<CodeBlock> {
        let mut blocks = Vec::new();
        
        // Extract from <pre><code> tags
        if let Ok(code_elements) = parser.select("pre code") {
            for code in code_elements {
                let content = code.trim();
                if !content.is_empty() {
                    blocks.push(CodeBlock {
                        language: None, // Could be enhanced to detect language
                        content: content.to_string(),
                        line_count: content.lines().count(),
                    });
                }
            }
        }
        
        // Extract from ```language blocks in text
        debug!("Extracted {} code blocks", blocks.len());
        
        blocks
    }
    
    fn extract_structured_data(
        &self,
        parser: &SelectorParser,
    ) -> HashMap<String, serde_json::Value> {
        let mut data = HashMap::new();
        
        // Extract JSON-LD structured data
        if let Ok(scripts) = parser.select("script[type='application/ld+json']") {
            for (i, script) in scripts.iter().enumerate() {
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(script) {
                    data.insert(format!("json_ld_{}", i), json);
                }
            }
        }
        
        // Extract microdata
        // This is simplified - full microdata extraction would be more complex
        
        debug!("Extracted {} structured data items", data.len());
        
        data
    }
}

// Specialized extractors for different content types
pub struct ArticleExtractor;

impl ArticleExtractor {
    pub fn extract(content: &ExtractedContent) -> Option<Article> {
        // Check if this looks like an article
        if content.text.len() < 500 {
            return None;
        }
        
        // Extract article-specific metadata
        let headline = content.metadata.open_graph
            .get("og:title")
            .or(Some(&content.title))
            .cloned()?;
        
        let author = content.metadata.author.clone()
            .or_else(|| content.metadata.open_graph.get("article:author").cloned());
        
        let published_date = content.metadata.published_date.clone();
        
        let word_count = content.text.split_whitespace().count();
        let reading_time = (word_count / 200).max(1); // Assuming 200 words per minute
        
        Some(Article {
            headline,
            author,
            published_date,
            content: content.markdown.clone(),
            word_count,
            reading_time,
            tags: content.metadata.keywords.clone(),
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Article {
    pub headline: String,
    pub author: Option<String>,
    pub published_date: Option<String>,
    pub content: String,
    pub word_count: usize,
    pub reading_time: usize,
    pub tags: Vec<String>,
}

pub struct ProductExtractor;

impl ProductExtractor {
    pub fn extract(content: &ExtractedContent) -> Option<Product> {
        // Look for product schema in structured data
        for (_, value) in &content.structured_data {
            if let Some(type_field) = value.get("@type") {
                if type_field.as_str() == Some("Product") {
                    return Self::parse_product_schema(value);
                }
            }
        }
        
        // Fallback to Open Graph product data
        if content.metadata.open_graph.get("og:type") == Some(&"product".to_string()) {
            return Self::parse_og_product(&content.metadata.open_graph);
        }
        
        None
    }
    
    fn parse_product_schema(schema: &serde_json::Value) -> Option<Product> {
        Some(Product {
            name: schema.get("name")?.as_str()?.to_string(),
            description: schema.get("description").and_then(|v| v.as_str()).map(|s| s.to_string()),
            price: schema.get("offers").and_then(|o| o.get("price")).and_then(|p| p.as_f64()),
            currency: schema.get("offers").and_then(|o| o.get("priceCurrency")).and_then(|c| c.as_str()).map(|s| s.to_string()),
            availability: schema.get("offers").and_then(|o| o.get("availability")).and_then(|a| a.as_str()).map(|s| s.to_string()),
            brand: schema.get("brand").and_then(|b| b.get("name")).and_then(|n| n.as_str()).map(|s| s.to_string()),
            rating: schema.get("aggregateRating").and_then(|r| r.get("ratingValue")).and_then(|v| v.as_f64()),
        })
    }
    
    fn parse_og_product(og: &HashMap<String, String>) -> Option<Product> {
        Some(Product {
            name: og.get("og:title")?.clone(),
            description: og.get("og:description").cloned(),
            price: og.get("product:price:amount").and_then(|p| p.parse().ok()),
            currency: og.get("product:price:currency").cloned(),
            availability: og.get("product:availability").cloned(),
            brand: og.get("product:brand").cloned(),
            rating: None,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Product {
    pub name: String,
    pub description: Option<String>,
    pub price: Option<f64>,
    pub currency: Option<String>,
    pub availability: Option<String>,
    pub brand: Option<String>,
    pub rating: Option<f64>,
}