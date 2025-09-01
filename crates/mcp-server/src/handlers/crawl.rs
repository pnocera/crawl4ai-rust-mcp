use axum::{extract::State, Json};
use serde::{Deserialize, Serialize};

use crate::{AppState, Result};

#[derive(Debug, Deserialize)]
pub struct CrawlRequest {
    pub url: String,
    pub wait_for: Option<String>,
    pub screenshot: Option<bool>,
    pub timeout: Option<u32>,
}

#[derive(Debug, Serialize)]
pub struct CrawlResponse {
    pub url: String,
    pub title: String,
    pub content: String,
    pub screenshot_url: Option<String>,
}

pub async fn crawl_handler(
    State(state): State<AppState>,
    Json(request): Json<CrawlRequest>,
) -> Result<Json<CrawlResponse>> {
    // TODO: Implement actual crawling
    Ok(Json(CrawlResponse {
        url: request.url,
        title: "Example Title".to_string(),
        content: "Example content".to_string(),
        screenshot_url: None,
    }))
}