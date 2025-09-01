use axum::{
    extract::State,
    Json,
};
use mcp_protocol::{McpRequest, McpResponse, McpResult, Tool};
use serde_json::json;

use crate::{
    sse::{stream_operation, SseEventSender, SseStream},
    AppState, Result,
};

// Main MCP request handler
pub async fn mcp_handler(
    State(state): State<AppState>,
    Json(request): Json<McpRequest>,
) -> Result<Json<McpResponse>> {
    let response = match request.tool {
        Tool::CrawlSinglePage => handle_crawl_single_page(&state, request).await?,
        Tool::SmartCrawlUrl => handle_smart_crawl(&state, request).await?,
        Tool::GetAvailableSources => handle_get_sources(&state, request).await?,
        Tool::PerformRagQuery => handle_rag_query(&state, request).await?,
        Tool::SearchCodeExamples => handle_code_search(&state, request).await?,
        Tool::ParseGithubRepository => handle_parse_github(&state, request).await?,
        Tool::CheckAiScriptHallucinations => handle_check_hallucinations(&state, request).await?,
        Tool::QueryKnowledgeGraph => handle_query_graph(&state, request).await?,
    };
    
    Ok(Json(response))
}

// Streaming MCP request handler
pub async fn mcp_stream_handler(
    State(state): State<AppState>,
    Json(request): Json<McpRequest>,
) -> Result<SseStream> {
    let stream = stream_operation(move |sender| async move {
        match request.tool {
            Tool::CrawlSinglePage => stream_crawl_single_page(state, request, sender).await,
            Tool::SmartCrawlUrl => stream_smart_crawl(state, request, sender).await,
            _ => {
                // For non-streaming tools, just execute and send complete
                match mcp_handler(State(state), Json(request)).await {
                    Ok(Json(response)) => {
                        if let McpResult::Success { data } = response.result {
                            let _ = sender.send_complete(data).await;
                        } else {
                            let _ = sender.send_error("Operation failed".to_string(), None).await;
                        }
                    }
                    Err(e) => {
                        let _ = sender.send_error(e.to_string(), None).await;
                    }
                }
            }
        }
    }).await;
    
    Ok(stream)
}

// Individual tool handlers
async fn handle_crawl_single_page(
    state: &AppState,
    request: McpRequest,
) -> Result<McpResponse> {
    let crawl_request: mcp_protocol::CrawlSinglePageRequest = 
        serde_json::from_value(request.params)?;
    
    let url = url::Url::parse(&crawl_request.url)
        .map_err(|e| crate::ServerError::InvalidRequest(format!("Invalid URL: {}", e)))?;
    
    // Create crawl options
    let options = crawler::CrawlOptions {
        max_depth: 0,
        max_pages: 1,
        extract_links: false, // Single page doesn't need links
        same_domain_only: true,
        follow_redirects: true,
        extract_metadata: true,
        wait_for: crawler::WaitFor::DomContentLoaded,
        screenshot: crawl_request.screenshot,
        javascript_enabled: false,
    };
    
    // Crawl the page
    let content = state.crawler.crawl(url.clone(), options).await?;
    
    // Generate embeddings for the content
    let embedding = state.embeddings.embed_text(content.text.clone()).await?;
    
    // Store in vector database
    let page_id = uuid::Uuid::new_v4();
    let page_point = vector_store::PagePoint {
        id: page_id,
        url: content.url.to_string(),
        title: content.title.clone(),
        domain: url.domain().unwrap_or("unknown").to_string(),
        content_preview: content.text.chars().take(500).collect(),
        crawled_at: chrono::Utc::now(),
        metadata: std::collections::HashMap::new(),
    };
    
    let page_vectors = vector_store::PageVectors {
        title_vector: embedding.clone(),
        content_vector: embedding,
        code_vector: None,
    };
    
    state.vector_store.upsert_page(page_point.clone(), page_vectors).await?;
    
    Ok(McpResponse {
        id: request.id,
        tool: request.tool,
        result: McpResult::Success {
            data: serde_json::to_value(&mcp_protocol::CrawledPage {
                id: page_id,
                url: content.url.to_string(),
                title: content.title,
                content: content.text,
                domain: url.domain().unwrap_or("unknown").to_string(),
                crawled_at: chrono::Utc::now(),
                screenshot_url: None,
                metadata: std::collections::HashMap::new(),
            })?,
        },
    })
}

async fn handle_smart_crawl(
    state: &AppState,
    request: McpRequest,
) -> Result<McpResponse> {
    let crawl_request: mcp_protocol::SmartCrawlUrlRequest = 
        serde_json::from_value(request.params)?;
    
    let start_url = url::Url::parse(&crawl_request.url)
        .map_err(|e| crate::ServerError::InvalidRequest(format!("Invalid URL: {}", e)))?;
    
    // Create crawl options
    let options = crawler::CrawlOptions {
        max_depth: crawl_request.max_depth,
        max_pages: crawl_request.max_pages as usize,
        same_domain_only: crawl_request.same_domain_only,
        follow_redirects: true,
        extract_links: true,
        extract_metadata: true,
        wait_for: crawler::WaitFor::DomContentLoaded,
        screenshot: false,
        javascript_enabled: false,
    };
    
    // Start multi-page crawl
    let mut receiver = state.crawler.crawl_site(start_url, options).await?;
    
    let mut crawled_pages = Vec::new();
    let mut success_count = 0;
    let mut error_count = 0;
    
    // Process crawl results
    while let Some(result) = receiver.recv().await {
        match result {
            crawler::CrawlResult::Success { url, content, depth: _ } => {
                // Generate embeddings
                let embedding = match state.embeddings.embed_text(content.text.clone()).await {
                    Ok(emb) => emb,
                    Err(e) => {
                        tracing::warn!("Failed to generate embeddings for {}: {}", url, e);
                        error_count += 1;
                        continue;
                    }
                };
                
                // Store in vector database
                let page_id = uuid::Uuid::new_v4();
                let page_point = vector_store::PagePoint {
                    id: page_id,
                    url: content.url.to_string(),
                    title: content.title.clone(),
                    domain: url.domain().unwrap_or("unknown").to_string(),
                    content_preview: content.text.chars().take(500).collect(),
                    crawled_at: chrono::Utc::now(),
                    metadata: std::collections::HashMap::new(),
                };
                
                let page_vectors = vector_store::PageVectors {
                    title_vector: embedding.clone(),
                    content_vector: embedding,
                    code_vector: None,
                };
                
                if let Err(e) = state.vector_store.upsert_page(page_point.clone(), page_vectors).await {
                    tracing::warn!("Failed to store page {}: {}", url, e);
                    error_count += 1;
                } else {
                    crawled_pages.push(mcp_protocol::CrawledPage {
                        id: page_id,
                        url: content.url.to_string(),
                        title: content.title,
                        content: content.text,
                        domain: url.domain().unwrap_or("unknown").to_string(),
                        crawled_at: chrono::Utc::now(),
                        screenshot_url: None,
                        metadata: std::collections::HashMap::new(),
                    });
                    success_count += 1;
                }
            }
            crawler::CrawlResult::Error { url, error, depth: _ } => {
                tracing::warn!("Failed to crawl {}: {}", url, error);
                error_count += 1;
            }
        }
    }
    
    Ok(McpResponse {
        id: request.id,
        tool: request.tool,
        result: McpResult::Success {
            data: json!({
                "pages": crawled_pages,
                "summary": {
                    "total_pages": success_count + error_count,
                    "successful_pages": success_count,
                    "failed_pages": error_count,
                    "start_url": crawl_request.url,
                }
            }),
        },
    })
}

async fn handle_get_sources(
    state: &AppState,
    request: McpRequest,
) -> Result<McpResponse> {
    // Get distinct domains from the vector store
    let _collection_info = state.vector_store.get_collection_info().await?;
    
    // In a real implementation, we'd query the vector store for distinct domains
    // For now, we'll use a placeholder approach
    let sources = vec![
        json!({
            "domain": "example.com",
            "page_count": 15,
            "last_crawled": chrono::Utc::now(),
        }),
        // In reality, this would be populated from the vector store
    ];
    
    Ok(McpResponse {
        id: request.id,
        tool: request.tool,
        result: McpResult::Success {
            data: json!({
                "sources": sources,
                "total_sources": sources.len(),
            }),
        },
    })
}

async fn handle_rag_query(
    state: &AppState,
    request: McpRequest,
) -> Result<McpResponse> {
    let rag_request: mcp_protocol::PerformRagQueryRequest = 
        serde_json::from_value(request.params)?;
    
    // Generate query embedding
    let query_embedding = state.embeddings.embed_text(rag_request.query.clone()).await?;
    
    // Build search query
    let search_query = vector_store::SearchQuery {
        query_text: rag_request.query.clone(),
        query_vectors: vector_store::QueryVectors {
            title_vector: Some(query_embedding.clone()),
            content_vector: Some(query_embedding),
            code_vector: None,
            weights: vector_store::VectorWeights {
                title: 0.3,
                content: 0.7,
                code: 0.0,
            },
        },
        limit: rag_request.limit as usize,
        offset: None,
        with_payload: true,
        with_vectors: false,
        score_threshold: Some(rag_request.similarity_threshold),
        filter: rag_request.source_filter.as_ref().map(|domain| {
            vector_store::SearchFilter {
                domain: Some(domain.clone()),
                date_from: None,
                date_to: None,
                has_code: None,
                metadata_filters: std::collections::HashMap::new(),
            }
        }),
    };
    
    // Perform search
    let search_results = state.vector_store.search(search_query).await?;
    
    // Convert to RAG results
    let rag_results: Vec<mcp_protocol::RagResult> = search_results
        .into_iter()
        .filter_map(|result| {
            result.payload.map(|payload| {
                mcp_protocol::RagResult {
                    id: result.id,
                    content: payload.content_preview,
                    url: payload.url,
                    title: payload.title,
                    similarity_score: result.score,
                    metadata: payload.metadata,
                }
            })
        })
        .collect();
    
    Ok(McpResponse {
        id: request.id,
        tool: request.tool,
        result: McpResult::Success {
            data: json!({
                "results": rag_results,
                "query": rag_request.query,
                "total_results": rag_results.len(),
            }),
        },
    })
}

async fn handle_code_search(
    state: &AppState,
    request: McpRequest,
) -> Result<McpResponse> {
    let code_request: mcp_protocol::SearchCodeExamplesRequest = 
        serde_json::from_value(request.params)?;
    
    // Generate query embedding for code search
    let query_embedding = state.embeddings.embed_text(code_request.query.clone()).await?;
    
    // Build search query focused on code vectors
    let search_query = vector_store::SearchQuery {
        query_text: code_request.query.clone(),
        query_vectors: vector_store::QueryVectors {
            title_vector: None,
            content_vector: Some(query_embedding.clone()),
            code_vector: Some(query_embedding),
            weights: vector_store::VectorWeights {
                title: 0.0,
                content: 0.3,
                code: 0.7,
            },
        },
        limit: code_request.limit as usize,
        offset: None,
        with_payload: true,
        with_vectors: false,
        score_threshold: Some(0.6), // Higher threshold for code
        filter: code_request.language.as_ref().map(|lang| {
            let mut metadata_filters = std::collections::HashMap::new();
            metadata_filters.insert("language".to_string(), serde_json::Value::String(lang.clone()));
            vector_store::SearchFilter {
                domain: None,
                date_from: None,
                date_to: None,
                has_code: Some(true),
                metadata_filters,
            }
        }),
    };
    
    // Perform search
    let search_results = state.vector_store.search(search_query).await?;
    
    // Extract code examples from results
    let code_examples: Vec<mcp_protocol::CodeExample> = search_results
        .into_iter()
        .filter_map(|result| {
            result.payload.and_then(|payload| {
                // Extract code from content (this would be more sophisticated in practice)
                if payload.content_preview.contains("```") {
                    Some(mcp_protocol::CodeExample {
                        id: result.id,
                        code: extract_code_blocks(&payload.content_preview),
                        language: code_request.language.clone().unwrap_or("unknown".to_string()),
                        url: payload.url,
                        description: payload.title,
                        similarity_score: result.score,
                    })
                } else {
                    None
                }
            })
        })
        .collect();
    
    Ok(McpResponse {
        id: request.id,
        tool: request.tool,
        result: McpResult::Success {
            data: json!({
                "examples": code_examples,
                "query": code_request.query,
                "language_filter": code_request.language,
                "total_examples": code_examples.len(),
            }),
        },
    })
}

async fn handle_parse_github(
    state: &AppState,
    request: McpRequest,
) -> Result<McpResponse> {
    let github_request: mcp_protocol::ParseGithubRepositoryRequest = 
        serde_json::from_value(request.params)?;
    
    // Parse repository structure and create knowledge graph
    let result = state.graph_store
        .parse_repository(&github_request.repo_url, github_request.branch.as_deref())
        .await?;
    
    Ok(McpResponse {
        id: request.id,
        tool: request.tool,
        result: McpResult::Success {
            data: json!({
                "repository_url": github_request.repo_url,
                "branch": github_request.branch.unwrap_or("main".to_string()),
                "nodes_created": result.nodes_created,
                "relationships_created": result.relationships_created,
                "files_processed": result.files_processed,
                "status": "completed"
            }),
        },
    })
}

async fn handle_check_hallucinations(
    state: &AppState,
    request: McpRequest,
) -> Result<McpResponse> {
    let hallucination_request: mcp_protocol::CheckAiScriptHallucinationsRequest = 
        serde_json::from_value(request.params)?;
    
    // Validate AI-generated script against knowledge graph
    let validation_result = state.graph_store
        .validate_script(&hallucination_request.script_path, hallucination_request.context.as_deref())
        .await?;
    
    let hallucination_result = mcp_protocol::HallucinationCheckResult {
        is_valid: validation_result.is_valid,
        confidence: validation_result.confidence,
        issues: validation_result.issues.into_iter().map(|issue| {
            mcp_protocol::ValidationIssue {
                line: issue.line,
                column: issue.column,
                severity: match issue.severity.as_str() {
                    "error" => mcp_protocol::IssueSeverity::Error,
                    "warning" => mcp_protocol::IssueSeverity::Warning,
                    _ => mcp_protocol::IssueSeverity::Info,
                },
                message: issue.message,
                suggestion: issue.suggestion,
            }
        }).collect(),
        suggestions: validation_result.suggestions,
    };
    
    Ok(McpResponse {
        id: request.id,
        tool: request.tool,
        result: McpResult::Success {
            data: serde_json::to_value(&hallucination_result)?,
        },
    })
}

async fn handle_query_graph(
    state: &AppState,
    request: McpRequest,
) -> Result<McpResponse> {
    let graph_request: mcp_protocol::QueryKnowledgeGraphRequest = 
        serde_json::from_value(request.params)?;
    
    // Execute Cypher query on the knowledge graph
    let query_results = state.graph_store
        .execute_query(&graph_request.query, graph_request.limit as usize)
        .await?;
    
    Ok(McpResponse {
        id: request.id,
        tool: request.tool,
        result: McpResult::Success {
            data: json!({
                "results": query_results,
                "query": graph_request.query,
                "total_results": query_results.len(),
            }),
        },
    })
}

// Streaming implementations
async fn stream_crawl_single_page(
    state: AppState,
    request: McpRequest,
    sender: SseEventSender,
) {
    let crawl_request: mcp_protocol::CrawlSinglePageRequest = match serde_json::from_value(request.params) {
        Ok(req) => req,
        Err(e) => {
            let _ = sender.send_error(format!("Invalid request: {}", e), None).await;
            return;
        }
    };
    
    let url = match url::Url::parse(&crawl_request.url) {
        Ok(u) => u,
        Err(e) => {
            let _ = sender.send_error(format!("Invalid URL: {}", e), None).await;
            return;
        }
    };
    
    let _ = sender.send_progress("Starting crawl...".to_string(), Some(10.0)).await;
    
    // Create crawl options
    let options = crawler::CrawlOptions {
        max_depth: 0,
        max_pages: 1,
        extract_links: false,
        same_domain_only: true,
        follow_redirects: true,
        extract_metadata: true,
        wait_for: crawler::WaitFor::DomContentLoaded,
        screenshot: crawl_request.screenshot,
        javascript_enabled: false,
    };
    
    let _ = sender.send_progress("Fetching page content...".to_string(), Some(30.0)).await;
    
    // Crawl the page
    let content = match state.crawler.crawl(url.clone(), options).await {
        Ok(c) => c,
        Err(e) => {
            let _ = sender.send_error(format!("Crawl failed: {}", e), None).await;
            return;
        }
    };
    
    let _ = sender.send_progress("Generating embeddings...".to_string(), Some(60.0)).await;
    
    // Generate embeddings for the content
    let embedding = match state.embeddings.embed_text(content.text.clone()).await {
        Ok(e) => e,
        Err(e) => {
            let _ = sender.send_error(format!("Embedding failed: {}", e), None).await;
            return;
        }
    };
    
    let _ = sender.send_progress("Storing in vector database...".to_string(), Some(80.0)).await;
    
    // Store in vector database
    let page_id = uuid::Uuid::new_v4();
    let page_point = vector_store::PagePoint {
        id: page_id,
        url: content.url.to_string(),
        title: content.title.clone(),
        domain: url.domain().unwrap_or("unknown").to_string(),
        content_preview: content.text.chars().take(500).collect(),
        crawled_at: chrono::Utc::now(),
        metadata: std::collections::HashMap::new(),
    };
    
    let page_vectors = vector_store::PageVectors {
        title_vector: embedding.clone(),
        content_vector: embedding,
        code_vector: None,
    };
    
    if let Err(e) = state.vector_store.upsert_page(page_point.clone(), page_vectors).await {
        let _ = sender.send_error(format!("Storage failed: {}", e), None).await;
        return;
    }
    
    let _ = sender.send_progress("Crawl completed successfully".to_string(), Some(100.0)).await;
    
    let _ = sender.send_complete(serde_json::to_value(&mcp_protocol::CrawledPage {
        id: page_id,
        url: content.url.to_string(),
        title: content.title,
        content: content.text,
        domain: url.domain().unwrap_or("unknown").to_string(),
        crawled_at: chrono::Utc::now(),
        screenshot_url: None,
        metadata: std::collections::HashMap::new(),
    }).unwrap_or(json!({"error": "Failed to serialize result"}))).await;
}

async fn stream_smart_crawl(
    state: AppState,
    request: McpRequest,
    sender: SseEventSender,
) {
    let crawl_request: mcp_protocol::SmartCrawlUrlRequest = match serde_json::from_value(request.params) {
        Ok(req) => req,
        Err(e) => {
            let _ = sender.send_error(format!("Invalid request: {}", e), None).await;
            return;
        }
    };
    
    let start_url = match url::Url::parse(&crawl_request.url) {
        Ok(u) => u,
        Err(e) => {
            let _ = sender.send_error(format!("Invalid URL: {}", e), None).await;
            return;
        }
    };
    
    let _ = sender.send_progress("Starting smart crawl...".to_string(), Some(0.0)).await;
    
    // Create crawl options
    let options = crawler::CrawlOptions {
        max_depth: crawl_request.max_depth,
        max_pages: crawl_request.max_pages as usize,
        same_domain_only: crawl_request.same_domain_only,
        follow_redirects: true,
        extract_links: true,
        extract_metadata: true,
        wait_for: crawler::WaitFor::DomContentLoaded,
        screenshot: false,
        javascript_enabled: false,
    };
    
    // Start multi-page crawl
    let mut receiver = match state.crawler.crawl_site(start_url, options).await {
        Ok(r) => r,
        Err(e) => {
            let _ = sender.send_error(format!("Failed to start crawl: {}", e), None).await;
            return;
        }
    };
    
    let mut crawled_pages = Vec::new();
    let mut success_count = 0;
    let mut error_count = 0;
    let max_pages = crawl_request.max_pages as f32;
    
    // Process crawl results with streaming updates
    while let Some(result) = receiver.recv().await {
        let progress = ((success_count + error_count) as f32 / max_pages * 100.0).min(100.0);
        
        match result {
            crawler::CrawlResult::Success { url, content, depth } => {
                let _ = sender.send_progress(
                    format!("Processing page: {} (depth {})", url, depth),
                    Some(progress),
                ).await;
                
                // Generate embeddings
                let embedding = match state.embeddings.embed_text(content.text.clone()).await {
                    Ok(emb) => emb,
                    Err(e) => {
                        tracing::warn!("Failed to generate embeddings for {}: {}", url, e);
                        error_count += 1;
                        let _ = sender.send_partial(json!({
                            "url": url.to_string(),
                            "status": "embedding_failed",
                            "error": e.to_string()
                        })).await;
                        continue;
                    }
                };
                
                // Store in vector database
                let page_id = uuid::Uuid::new_v4();
                let page_point = vector_store::PagePoint {
                    id: page_id,
                    url: content.url.to_string(),
                    title: content.title.clone(),
                    domain: url.domain().unwrap_or("unknown").to_string(),
                    content_preview: content.text.chars().take(500).collect(),
                    crawled_at: chrono::Utc::now(),
                    metadata: std::collections::HashMap::new(),
                };
                
                let page_vectors = vector_store::PageVectors {
                    title_vector: embedding.clone(),
                    content_vector: embedding,
                    code_vector: None,
                };
                
                if let Err(e) = state.vector_store.upsert_page(page_point.clone(), page_vectors).await {
                    tracing::warn!("Failed to store page {}: {}", url, e);
                    error_count += 1;
                    let _ = sender.send_partial(json!({
                        "url": url.to_string(),
                        "status": "storage_failed",
                        "error": e.to_string()
                    })).await;
                } else {
                    crawled_pages.push(mcp_protocol::CrawledPage {
                        id: page_id,
                        url: content.url.to_string(),
                        title: content.title,
                        content: content.text,
                        domain: url.domain().unwrap_or("unknown").to_string(),
                        crawled_at: chrono::Utc::now(),
                        screenshot_url: None,
                        metadata: std::collections::HashMap::new(),
                    });
                    success_count += 1;
                    
                    let _ = sender.send_partial(json!({
                        "url": url.to_string(),
                        "title": page_point.title,
                        "status": "completed",
                        "depth": depth
                    })).await;
                }
            }
            crawler::CrawlResult::Error { url, error, depth } => {
                tracing::warn!("Failed to crawl {} (depth {}): {}", url, depth, error);
                error_count += 1;
                let _ = sender.send_partial(json!({
                    "url": url.to_string(),
                    "status": "failed",
                    "error": error.to_string(),
                    "depth": depth
                })).await;
            }
        }
    }
    
    let _ = sender.send_progress("Smart crawl completed".to_string(), Some(100.0)).await;
    
    let _ = sender.send_complete(json!({
        "pages": crawled_pages,
        "summary": {
            "total_pages": success_count + error_count,
            "successful_pages": success_count,
            "failed_pages": error_count,
            "start_url": crawl_request.url,
        }
    })).await;
}

// Helper function to extract code blocks from markdown content
fn extract_code_blocks(content: &str) -> String {
    let mut code_blocks = Vec::new();
    let mut in_code_block = false;
    let mut current_block = String::new();
    
    for line in content.lines() {
        if line.starts_with("```") {
            if in_code_block {
                // End of code block
                code_blocks.push(current_block.trim().to_string());
                current_block.clear();
                in_code_block = false;
            } else {
                // Start of code block
                in_code_block = true;
            }
        } else if in_code_block {
            current_block.push_str(line);
            current_block.push('\n');
        }
    }
    
    code_blocks.join("\n\n")
}