use axum::{extract::State, Json};
use serde::{Deserialize, Serialize};

use crate::{AppState, Result};

#[derive(Debug, Deserialize)]
pub struct SearchRequest {
    pub query: String,
    pub limit: Option<u32>,
    pub source_filter: Option<String>,
    pub similarity_threshold: Option<f32>,
}

#[derive(Debug, Serialize)]
pub struct SearchResult {
    pub id: String,
    pub url: String,
    pub title: String,
    pub content: String,
    pub similarity_score: f32,
}

#[derive(Debug, Serialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResult>,
    pub total: usize,
}

pub async fn search_handler(
    State(state): State<AppState>,
    Json(request): Json<SearchRequest>,
) -> Result<Json<SearchResponse>> {
    // TODO: Implement actual search
    Ok(Json(SearchResponse {
        results: vec![],
        total: 0,
    }))
}

pub async fn sources_handler(
    State(state): State<AppState>,
) -> Result<Json<Vec<String>>> {
    // TODO: Get actual sources from storage
    Ok(Json(vec![]))
}