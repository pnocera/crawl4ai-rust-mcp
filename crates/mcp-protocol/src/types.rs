use bytes::Bytes;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

// MCP Tool definitions matching the Python implementation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tool {
    CrawlSinglePage,
    SmartCrawlUrl,
    GetAvailableSources,
    PerformRagQuery,
    SearchCodeExamples,
    ParseGithubRepository,
    CheckAiScriptHallucinations,
    QueryKnowledgeGraph,
}

// Request types for each tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrawlSinglePageRequest {
    pub url: String,
    #[serde(default = "default_wait_for")]
    pub wait_for: String,
    #[serde(default)]
    pub screenshot: bool,
    #[serde(default = "default_timeout")]
    pub timeout: u32,
    #[serde(default)]
    pub extra_headers: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmartCrawlUrlRequest {
    pub url: String,
    #[serde(default = "default_max_depth")]
    pub max_depth: u32,
    #[serde(default = "default_max_pages")]
    pub max_pages: u32,
    #[serde(default)]
    pub same_domain_only: bool,
    #[serde(default = "default_wait_for")]
    pub wait_for: String,
    #[serde(default = "default_timeout")]
    pub timeout: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformRagQueryRequest {
    pub query: String,
    #[serde(default = "default_limit")]
    pub limit: u32,
    #[serde(default)]
    pub source_filter: Option<String>,
    #[serde(default = "default_similarity_threshold")]
    pub similarity_threshold: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchCodeExamplesRequest {
    pub query: String,
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default = "default_limit")]
    pub limit: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParseGithubRepositoryRequest {
    pub repo_url: String,
    #[serde(default)]
    pub branch: Option<String>,
    #[serde(default)]
    pub include_tests: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckAiScriptHallucinationsRequest {
    pub script_path: String,
    #[serde(default)]
    pub context: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryKnowledgeGraphRequest {
    pub query: String,
    #[serde(default = "default_limit")]
    pub limit: u32,
}

// Response types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrawledPage {
    pub id: Uuid,
    pub url: String,
    pub title: String,
    pub content: String,
    pub domain: String,
    pub crawled_at: DateTime<Utc>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub screenshot_url: Option<String>,
    pub metadata: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagResult {
    pub id: Uuid,
    pub content: String,
    pub url: String,
    pub title: String,
    pub similarity_score: f32,
    pub metadata: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeExample {
    pub id: Uuid,
    pub code: String,
    pub language: String,
    pub url: String,
    pub description: String,
    pub similarity_score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HallucinationCheckResult {
    pub is_valid: bool,
    pub confidence: f32,
    pub issues: Vec<ValidationIssue>,
    pub suggestions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationIssue {
    pub line: u32,
    pub column: u32,
    pub severity: IssueSeverity,
    pub message: String,
    pub suggestion: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum IssueSeverity {
    Error,
    Warning,
    Info,
}

// Generic MCP message structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpRequest {
    pub id: Uuid,
    pub tool: Tool,
    pub params: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpResponse {
    pub id: Uuid,
    pub tool: Tool,
    #[serde(flatten)]
    pub result: McpResult,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "lowercase")]
pub enum McpResult {
    Success { data: serde_json::Value },
    Error { message: String, details: Option<serde_json::Value> },
}

// SSE event types for streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SseEvent {
    Progress {
        message: String,
        percentage: Option<f32>,
    },
    PartialResult {
        data: serde_json::Value,
    },
    Complete {
        data: serde_json::Value,
    },
    Error {
        message: String,
        details: Option<serde_json::Value>,
    },
}

// Default value functions
fn default_wait_for() -> String {
    "domcontentloaded".to_string()
}

fn default_timeout() -> u32 {
    30000
}

fn default_max_depth() -> u32 {
    3
}

fn default_max_pages() -> u32 {
    50
}

fn default_limit() -> u32 {
    10
}

fn default_similarity_threshold() -> f32 {
    0.7
}