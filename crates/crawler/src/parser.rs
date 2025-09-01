use bytes::Bytes;
use ego_tree::{NodeRef, Tree};
use html5ever::{
    parse_document, tendril::TendrilSink, tree_builder::TreeBuilderOpts, ParseOpts,
};
use markup5ever_rcdom::{Handle, NodeData, RcDom};
use scraper::{Html, Selector};
use std::collections::HashSet;
use tracing::debug;
use url::Url;

use crate::{CrawlerError, Result};

// Zero-copy HTML parser
pub struct HtmlParser {
    dom: RcDom,
    base_url: Url,
}

impl HtmlParser {
    pub fn parse(html: &Bytes, base_url: Url) -> Result<Self> {
        let opts = ParseOpts {
            tree_builder: TreeBuilderOpts {
                drop_doctype: false,
                ..Default::default()
            },
            ..Default::default()
        };
        
        let dom = parse_document(RcDom::default(), opts)
            .from_utf8()
            .read_from(&mut &html[..])?;
        
        Ok(Self { dom, base_url })
    }
    
    pub fn extract_text(&self) -> String {
        let mut text = String::new();
        self.extract_text_recursive(&self.dom.document, &mut text);
        
        // Clean up whitespace
        text.split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }
    
    fn extract_text_recursive(&self, handle: &Handle, text: &mut String) {
        let node = handle;
        
        match &node.data {
            NodeData::Text { contents } => {
                text.push_str(&contents.borrow());
                text.push(' ');
            }
            NodeData::Element { name, .. } => {
                // Skip script and style tags
                let tag_name = name.local.as_ref();
                if tag_name != "script" && tag_name != "style" && tag_name != "noscript" {
                    for child in node.children.borrow().iter() {
                        self.extract_text_recursive(child, text);
                    }
                }
            }
            _ => {
                for child in node.children.borrow().iter() {
                    self.extract_text_recursive(child, text);
                }
            }
        }
    }
    
    pub fn extract_links(&self) -> Vec<Url> {
        let mut links = Vec::new();
        self.extract_links_recursive(&self.dom.document, &mut links);
        links
    }
    
    fn extract_links_recursive(&self, handle: &Handle, links: &mut Vec<Url>) {
        let node = handle;
        
        if let NodeData::Element { name, attrs, .. } = &node.data {
            if name.local.as_ref() == "a" {
                for attr in attrs.borrow().iter() {
                    if attr.name.local.as_ref() == "href" {
                        if let Ok(url) = self.base_url.join(&attr.value) {
                            links.push(url);
                        }
                    }
                }
            }
        }
        
        for child in node.children.borrow().iter() {
            self.extract_links_recursive(child, links);
        }
    }
    
    pub fn extract_metadata(&self) -> Metadata {
        let mut metadata = Metadata::default();
        self.extract_metadata_recursive(&self.dom.document, &mut metadata);
        metadata
    }
    
    fn extract_metadata_recursive(&self, handle: &Handle, metadata: &mut Metadata) {
        let node = handle;
        
        match &node.data {
            NodeData::Element { name, attrs, .. } => {
                let tag_name = name.local.as_ref();
                
                match tag_name {
                    "title" => {
                        let mut title = String::new();
                        for child in node.children.borrow().iter() {
                            if let NodeData::Text { contents } = &child.data {
                                title.push_str(&contents.borrow());
                            }
                        }
                        if !title.is_empty() {
                            metadata.title = Some(title.trim().to_string());
                        }
                    }
                    "meta" => {
                        let attrs = attrs.borrow();
                        let mut name_attr = None;
                        let mut property_attr = None;
                        let mut content_attr = None;
                        
                        for attr in attrs.iter() {
                            match attr.name.local.as_ref() {
                                "name" => name_attr = Some(attr.value.to_string()),
                                "property" => property_attr = Some(attr.value.to_string()),
                                "content" => content_attr = Some(attr.value.to_string()),
                                _ => {}
                            }
                        }
                        
                        if let Some(content) = content_attr {
                            if let Some(name) = name_attr {
                                match name.as_str() {
                                    "description" => metadata.description = Some(content),
                                    "keywords" => metadata.keywords = Some(content),
                                    "author" => metadata.author = Some(content),
                                    _ => {}
                                }
                            } else if let Some(property) = property_attr {
                                if property.starts_with("og:") {
                                    metadata.open_graph.insert(property, content);
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
            _ => {}
        }
        
        for child in node.children.borrow().iter() {
            self.extract_metadata_recursive(child, metadata);
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct Metadata {
    pub title: Option<String>,
    pub description: Option<String>,
    pub keywords: Option<String>,
    pub author: Option<String>,
    pub open_graph: std::collections::HashMap<String, String>,
}

// Fast CSS selector-based parser using scraper
pub struct SelectorParser {
    html: Html,
    base_url: Url,
}

impl SelectorParser {
    pub fn parse(html: &str, base_url: Url) -> Self {
        Self {
            html: Html::parse_document(html),
            base_url,
        }
    }
    
    pub fn select(&self, selector: &str) -> Result<Vec<String>> {
        let selector = Selector::parse(selector)
            .map_err(|e| CrawlerError::ParseError(format!("Invalid selector: {:?}", e)))?;
        
        Ok(self.html
            .select(&selector)
            .map(|el| el.text().collect::<String>())
            .collect())
    }
    
    pub fn select_attr(&self, selector: &str, attr: &str) -> Result<Vec<String>> {
        let selector = Selector::parse(selector)
            .map_err(|e| CrawlerError::ParseError(format!("Invalid selector: {:?}", e)))?;
        
        Ok(self.html
            .select(&selector)
            .filter_map(|el| el.value().attr(attr).map(|s| s.to_string()))
            .collect())
    }
    
    pub fn extract_all_links(&self) -> Vec<Url> {
        self.select_attr("a", "href")
            .unwrap_or_default()
            .into_iter()
            .filter_map(|href| self.base_url.join(&href).ok())
            .collect()
    }
    
    pub fn extract_images(&self) -> Vec<String> {
        self.select_attr("img", "src").unwrap_or_default()
    }
}

// Markdown converter for clean text extraction
pub struct MarkdownConverter;

impl MarkdownConverter {
    pub fn convert(html: &str) -> String {
        let fragment = Html::parse_fragment(html);
        let mut markdown = String::new();
        
        Self::convert_node(&fragment.root_element(), &mut markdown, 0);
        
        markdown.trim().to_string()
    }
    
    fn convert_node(node: &scraper::ElementRef, output: &mut String, depth: usize) {
        for child in node.children() {
            match child.value() {
                scraper::Node::Text(text) => {
                    output.push_str(text.trim());
                }
                scraper::Node::Element(element) => {
                    let elem = scraper::ElementRef::wrap(child).unwrap();
                    
                    match element.name() {
                        "p" => {
                            output.push_str("\n\n");
                            Self::convert_node(&elem, output, depth);
                            output.push_str("\n\n");
                        }
                        "h1" => {
                            output.push_str("\n\n# ");
                            Self::convert_node(&elem, output, depth);
                            output.push_str("\n\n");
                        }
                        "h2" => {
                            output.push_str("\n\n## ");
                            Self::convert_node(&elem, output, depth);
                            output.push_str("\n\n");
                        }
                        "h3" => {
                            output.push_str("\n\n### ");
                            Self::convert_node(&elem, output, depth);
                            output.push_str("\n\n");
                        }
                        "h4" => {
                            output.push_str("\n\n#### ");
                            Self::convert_node(&elem, output, depth);
                            output.push_str("\n\n");
                        }
                        "h5" => {
                            output.push_str("\n\n##### ");
                            Self::convert_node(&elem, output, depth);
                            output.push_str("\n\n");
                        }
                        "h6" => {
                            output.push_str("\n\n###### ");
                            Self::convert_node(&elem, output, depth);
                            output.push_str("\n\n");
                        }
                        "strong" | "b" => {
                            output.push_str("**");
                            Self::convert_node(&elem, output, depth);
                            output.push_str("**");
                        }
                        "em" | "i" => {
                            output.push_str("*");
                            Self::convert_node(&elem, output, depth);
                            output.push_str("*");
                        }
                        "code" => {
                            output.push('`');
                            Self::convert_node(&elem, output, depth);
                            output.push('`');
                        }
                        "pre" => {
                            output.push_str("\n\n```\n");
                            Self::convert_node(&elem, output, depth);
                            output.push_str("\n```\n\n");
                        }
                        "a" => {
                            if let Some(href) = elem.value().attr("href") {
                                output.push('[');
                                Self::convert_node(&elem, output, depth);
                                output.push_str("](");
                                output.push_str(href);
                                output.push(')');
                            } else {
                                Self::convert_node(&elem, output, depth);
                            }
                        }
                        "ul" | "ol" => {
                            output.push_str("\n");
                            Self::convert_node(&elem, output, depth + 1);
                        }
                        "li" => {
                            output.push_str("\n");
                            for _ in 0..depth {
                                output.push_str("  ");
                            }
                            output.push_str("- ");
                            Self::convert_node(&elem, output, depth);
                        }
                        "br" => {
                            output.push_str("\n");
                        }
                        "script" | "style" | "noscript" => {
                            // Skip these tags
                        }
                        _ => {
                            Self::convert_node(&elem, output, depth);
                        }
                    }
                }
                _ => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_markdown_converter() {
        let html = r#"
            <h1>Title</h1>
            <p>This is a <strong>bold</strong> paragraph with <em>italic</em> text.</p>
            <ul>
                <li>Item 1</li>
                <li>Item 2</li>
            </ul>
            <p>Link to <a href="https://example.com">example</a>.</p>
        "#;
        
        let markdown = MarkdownConverter::convert(html);
        
        assert!(markdown.contains("# Title"));
        assert!(markdown.contains("**bold**"));
        assert!(markdown.contains("*italic*"));
        assert!(markdown.contains("- Item 1"));
        assert!(markdown.contains("[example](https://example.com)"));
    }
}