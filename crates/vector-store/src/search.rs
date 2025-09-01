use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info};

use crate::{
    PagePoint, Result, SearchFilter, SearchQuery, SearchResult,
    VectorStore, VectorStoreError,
};

pub struct HybridSearchEngine {
    vector_store: Arc<VectorStore>,
    bm25_weight: f32,
    vector_weight: f32,
    reranking_enabled: bool,
}

impl HybridSearchEngine {
    pub fn new(vector_store: Arc<VectorStore>) -> Self {
        Self {
            vector_store,
            bm25_weight: 0.3,
            vector_weight: 0.7,
            reranking_enabled: true,
        }
    }
    
    pub fn with_weights(mut self, bm25: f32, vector: f32) -> Self {
        self.bm25_weight = bm25;
        self.vector_weight = vector;
        self
    }
    
    pub fn with_reranking(mut self, enabled: bool) -> Self {
        self.reranking_enabled = enabled;
        self
    }
    
    pub async fn search(&self, query: SearchQuery) -> Result<Vec<SearchResult>> {
        info!("Performing hybrid search for: {}", query.query_text);
        
        // Perform vector search
        let vector_results = self.vector_store.search(query.clone()).await?;
        
        // Perform BM25 search (placeholder - would integrate with tantivy)
        let bm25_results = self.bm25_search(&query).await?;
        
        // Merge results
        let merged_results = self.merge_results(vector_results, bm25_results);
        
        // Apply reranking if enabled
        let final_results = if self.reranking_enabled {
            self.rerank_results(merged_results, &query).await?
        } else {
            merged_results
        };
        
        debug!("Found {} results after hybrid search", final_results.len());
        
        Ok(final_results)
    }
    
    async fn bm25_search(&self, query: &SearchQuery) -> Result<Vec<SearchResult>> {
        // Basic BM25 implementation using in-memory text index
        // For production, consider using tantivy or elasticsearch
        
        let query_terms = self.tokenize(&query.query_text);
        if query_terms.is_empty() {
            return Ok(vec![]);
        }
        
        // Get all pages for BM25 scoring
        let _collection_info = self.vector_store.get_collection_info().await?;
        let all_pages = self.vector_store.list_all_pages().await?;
        
        if all_pages.is_empty() {
            return Ok(vec![]);
        }
        
        // Build document term frequencies
        let mut doc_term_freqs = Vec::new();
        let mut doc_lengths = Vec::new();
        let mut total_docs = 0;
        let mut term_doc_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        
        for page in &all_pages {
            let doc_text = format!("{} {}", page.title, page.content_preview);
            let doc_terms = self.tokenize(&doc_text);
            let doc_length = doc_terms.len();
            
            // Calculate term frequencies
            let mut term_freq: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
            for term in &doc_terms {
                *term_freq.entry(term.clone()).or_insert(0) += 1;
                
                // Count documents containing each term (for IDF)
                if term_freq.get(term) == Some(&1) { // First occurrence in this doc
                    *term_doc_counts.entry(term.clone()).or_insert(0) += 1;
                }
            }
            
            doc_term_freqs.push(term_freq);
            doc_lengths.push(doc_length);
            total_docs += 1;
        }
        
        // Calculate average document length
        let avg_doc_length = if total_docs > 0 {
            doc_lengths.iter().sum::<usize>() as f32 / total_docs as f32
        } else {
            0.0
        };
        
        // BM25 parameters
        let k1 = 1.5f32;
        let b = 0.75f32;
        
        // Score documents
        let mut scored_results = Vec::new();
        
        for (doc_idx, page) in all_pages.iter().enumerate() {
            let mut score = 0.0f32;
            let doc_length = doc_lengths[doc_idx] as f32;
            let term_freq: &std::collections::HashMap<String, usize> = &doc_term_freqs[doc_idx];
            
            for term in &query_terms {
                let tf = *term_freq.get(term).unwrap_or(&0) as f32;
                let df = *term_doc_counts.get(term).unwrap_or(&0) as f32;
                
                if tf > 0.0 && df > 0.0 {
                    // IDF calculation
                    let idf = ((total_docs as f32 - df + 0.5) / (df + 0.5)).ln();
                    
                    // BM25 formula
                    let numerator = tf * (k1 + 1.0);
                    let denominator = tf + k1 * (1.0 - b + b * (doc_length / avg_doc_length));
                    
                    score += idf * (numerator / denominator);
                }
            }
            
            if score > 0.0 {
                scored_results.push(SearchResult {
                    id: page.id,
                    score,
                    payload: Some(page.clone()),
                    vectors: None,
                });
            }
        }
        
        // Sort by score descending
        scored_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        
        // Apply limit if specified
        scored_results.truncate(query.limit);
        
        debug!("BM25 search found {} results for query: {}", scored_results.len(), query.query_text);
        
        Ok(scored_results)
    }
    
    fn tokenize(&self, text: &str) -> Vec<String> {
        // Simple tokenization - split on whitespace and punctuation, lowercase, remove short words
        text.to_lowercase()
            .split(|c: char| c.is_whitespace() || c.is_ascii_punctuation())
            .filter(|word| word.len() > 2) // Filter out very short words
            .map(|word| word.to_string())
            .collect()
    }
    
    fn merge_results(
        &self,
        vector_results: Vec<SearchResult>,
        bm25_results: Vec<SearchResult>,
    ) -> Vec<SearchResult> {
        let mut merged: HashMap<uuid::Uuid, MergedResult> = HashMap::new();
        
        // Add vector results
        for result in vector_results {
            merged.insert(
                result.id,
                MergedResult {
                    result,
                    vector_score: Some(0.0),
                    bm25_score: None,
                },
            );
        }
        
        // Add BM25 results
        for result in bm25_results {
            merged
                .entry(result.id)
                .and_modify(|e| e.bm25_score = Some(result.score))
                .or_insert(MergedResult {
                    result,
                    vector_score: None,
                    bm25_score: Some(0.0),
                });
        }
        
        // Calculate combined scores
        let mut final_results: Vec<SearchResult> = merged
            .into_values()
            .map(|mut m| {
                let vector_score = m.vector_score.unwrap_or(0.0) * self.vector_weight;
                let bm25_score = m.bm25_score.unwrap_or(0.0) * self.bm25_weight;
                m.result.score = vector_score + bm25_score;
                m.result
            })
            .collect();
        
        // Sort by score
        final_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        
        final_results
    }
    
    async fn rerank_results(
        &self,
        results: Vec<SearchResult>,
        _query: &SearchQuery,
    ) -> Result<Vec<SearchResult>> {
        // Cross-encoder reranking would go here
        // For now, just return the results as-is
        Ok(results)
    }
}

struct MergedResult {
    result: SearchResult,
    vector_score: Option<f32>,
    bm25_score: Option<f32>,
}

// Reciprocal Rank Fusion for combining multiple result lists
pub struct RecipRankFusion {
    k: f32,
}

impl RecipRankFusion {
    pub fn new() -> Self {
        Self { k: 60.0 }
    }
    
    pub fn fuse(&self, result_lists: Vec<Vec<SearchResult>>) -> Vec<SearchResult> {
        let mut scores: HashMap<uuid::Uuid, f32> = HashMap::new();
        let mut results: HashMap<uuid::Uuid, SearchResult> = HashMap::new();
        
        for list in result_lists {
            for (rank, result) in list.into_iter().enumerate() {
                let rrf_score = 1.0 / (self.k + rank as f32 + 1.0);
                
                scores
                    .entry(result.id)
                    .and_modify(|s| *s += rrf_score)
                    .or_insert(rrf_score);
                
                results.insert(result.id, result);
            }
        }
        
        // Convert to vector and sort by RRF score
        let mut final_results: Vec<SearchResult> = results
            .into_iter()
            .map(|(id, mut result)| {
                result.score = scores[&id];
                result
            })
            .collect();
        
        final_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        
        final_results
    }
}

// Query expansion for better recall
pub struct QueryExpander {
    synonyms: HashMap<String, Vec<String>>,
}

impl QueryExpander {
    pub fn new() -> Self {
        Self {
            synonyms: HashMap::new(),
        }
    }
    
    pub fn expand_query(&self, query: &str) -> Vec<String> {
        let mut expanded = vec![query.to_string()];
        
        // Simple word-based expansion
        let words: Vec<&str> = query.split_whitespace().collect();
        
        for word in words {
            if let Some(syns) = self.synonyms.get(word) {
                for syn in syns {
                    let expanded_query = query.replace(word, syn);
                    expanded.push(expanded_query);
                }
            }
        }
        
        expanded
    }
}

// Result diversification to avoid redundant results
pub struct ResultDiversifier {
    similarity_threshold: f32,
}

impl ResultDiversifier {
    pub fn new() -> Self {
        Self {
            similarity_threshold: 0.9,
        }
    }
    
    pub fn diversify(&self, results: Vec<SearchResult>) -> Vec<SearchResult> {
        if results.is_empty() {
            return results;
        }
        
        let mut diversified = vec![results[0].clone()];
        
        for candidate in results.into_iter().skip(1) {
            let is_diverse = diversified.iter().all(|selected| {
                // In a real implementation, calculate similarity between results
                // For now, just check if they're from the same domain
                if let (Some(candidate_payload), Some(selected_payload)) = 
                    (&candidate.payload, &selected.payload) {
                    candidate_payload.domain != selected_payload.domain
                } else {
                    true
                }
            });
            
            if is_diverse {
                diversified.push(candidate);
            }
        }
        
        diversified
    }
}

// Search analytics for improving results over time
pub struct SearchAnalytics {
    query_log: Vec<QueryLogEntry>,
}

#[derive(Debug, Clone)]
struct QueryLogEntry {
    query: String,
    results_count: usize,
    timestamp: chrono::DateTime<chrono::Utc>,
    clicked_results: Vec<uuid::Uuid>,
}

impl SearchAnalytics {
    pub fn new() -> Self {
        Self {
            query_log: Vec::new(),
        }
    }
    
    pub fn log_query(&mut self, query: &str, results_count: usize) {
        self.query_log.push(QueryLogEntry {
            query: query.to_string(),
            results_count,
            timestamp: chrono::Utc::now(),
            clicked_results: Vec::new(),
        });
    }
    
    pub fn log_click(&mut self, query_id: usize, result_id: uuid::Uuid) {
        if let Some(entry) = self.query_log.get_mut(query_id) {
            entry.clicked_results.push(result_id);
        }
    }
    
    pub fn get_popular_queries(&self, limit: usize) -> Vec<String> {
        use std::collections::HashMap;
        
        let mut query_counts: HashMap<String, usize> = HashMap::new();
        
        for entry in &self.query_log {
            *query_counts.entry(entry.query.clone()).or_insert(0) += 1;
        }
        
        let mut popular: Vec<(String, usize)> = query_counts.into_iter().collect();
        popular.sort_by(|a, b| b.1.cmp(&a.1));
        popular.truncate(limit);
        
        popular.into_iter().map(|(query, _)| query).collect()
    }
}