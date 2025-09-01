use std::collections::{HashMap, BTreeMap};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use serde::{Serialize, Deserialize};
use uuid::Uuid;

use crate::{Result, StorageError};

/// Query types supported by the optimizer
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum QueryType {
    VectorSearch {
        embedding: Vec<u8>, // Quantized for caching
        filters: Option<String>,
        limit: usize,
    },
    TextSearch {
        query: String,
        filters: Option<String>,
        limit: usize,
    },
    DomainQuery {
        domain: String,
        limit: Option<usize>,
    },
    CodeSearch {
        language: Option<String>,
        keyword: Option<String>,
        limit: Option<usize>,
    },
    AnalyticsQuery {
        sql: String,
    },
}

/// Query execution plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryPlan {
    pub query_id: Uuid,
    pub query_type: QueryType,
    pub estimated_cost: f64,
    pub cache_key: Option<String>,
    pub execution_steps: Vec<ExecutionStep>,
    pub expected_result_size: Option<usize>,
}

/// Individual execution steps
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionStep {
    CheckCache { cache_key: String },
    VectorLookup { collection: String, limit: usize },
    FilterResults { filter_type: String, criteria: String },
    RerankResults { method: String },
    JoinWithMetadata,
    CacheResult { cache_key: String, ttl_seconds: u64 },
}

/// Query cache entry
#[derive(Debug, Clone)]
pub struct CacheEntry {
    pub result: Vec<u8>, // Serialized result
    pub created_at: Instant,
    pub last_accessed: Instant,
    pub access_count: u64,
    pub ttl: Duration,
    pub result_size: usize,
}

/// Query statistics for optimization
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QueryStats {
    pub execution_count: u64,
    pub total_execution_time: Duration,
    pub average_execution_time: Duration,
    pub cache_hit_rate: f64,
    pub result_size_distribution: BTreeMap<usize, u64>, // size ranges -> counts
}

/// Query optimizer with intelligent caching and execution planning
pub struct QueryOptimizer {
    cache: Arc<RwLock<HashMap<String, CacheEntry>>>,
    stats: Arc<RwLock<HashMap<String, QueryStats>>>,
    config: OptimizerConfig,
}

/// Optimizer configuration
#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    pub max_cache_size: usize,
    pub default_ttl: Duration,
    pub max_query_execution_time: Duration,
    pub cache_hit_threshold: f64, // Cache queries with hit rate above this
    pub enable_cost_based_optimization: bool,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            max_cache_size: 10_000,
            default_ttl: Duration::from_secs(3600), // 1 hour
            max_query_execution_time: Duration::from_secs(30),
            cache_hit_threshold: 0.3,
            enable_cost_based_optimization: true,
        }
    }
}

impl QueryOptimizer {
    pub fn new(config: OptimizerConfig) -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(HashMap::new())),
            config,
        }
    }

    /// Create an optimized execution plan for a query
    pub fn create_plan(&self, query_type: QueryType) -> Result<QueryPlan> {
        let query_id = Uuid::new_v4();
        let cache_key = self.generate_cache_key(&query_type);
        let estimated_cost = self.estimate_query_cost(&query_type);
        
        let mut execution_steps = Vec::new();
        
        // Check if we should use cache for this query
        if self.should_cache_query(&cache_key) {
            execution_steps.push(ExecutionStep::CheckCache {
                cache_key: cache_key.clone(),
            });
        }

        // Add query-specific execution steps
        match &query_type {
            QueryType::VectorSearch { limit, filters, .. } => {
                execution_steps.push(ExecutionStep::VectorLookup {
                    collection: "main".to_string(),
                    limit: *limit,
                });

                if filters.is_some() {
                    execution_steps.push(ExecutionStep::FilterResults {
                        filter_type: "metadata".to_string(),
                        criteria: filters.as_ref().unwrap().clone(),
                    });
                }

                // Add reranking for large result sets
                if *limit > 50 {
                    execution_steps.push(ExecutionStep::RerankResults {
                        method: "cross_encoder".to_string(),
                    });
                }

                execution_steps.push(ExecutionStep::JoinWithMetadata);
            },
            QueryType::TextSearch { limit, filters, .. } => {
                // First try vector search with query embedding
                execution_steps.push(ExecutionStep::VectorLookup {
                    collection: "main".to_string(),
                    limit: (*limit).min(200), // Cap initial retrieval
                });

                if filters.is_some() {
                    execution_steps.push(ExecutionStep::FilterResults {
                        filter_type: "metadata".to_string(),
                        criteria: filters.as_ref().unwrap().clone(),
                    });
                }

                execution_steps.push(ExecutionStep::RerankResults {
                    method: "hybrid_search".to_string(),
                });

                execution_steps.push(ExecutionStep::JoinWithMetadata);
            },
            QueryType::DomainQuery { .. } => {
                // Simple metadata lookup, no vector search needed
                execution_steps.push(ExecutionStep::JoinWithMetadata);
            },
            QueryType::CodeSearch { .. } => {
                execution_steps.push(ExecutionStep::VectorLookup {
                    collection: "code".to_string(),
                    limit: 100,
                });
                execution_steps.push(ExecutionStep::JoinWithMetadata);
            },
            QueryType::AnalyticsQuery { .. } => {
                // Direct SQL execution, no optimization needed
            },
        }

        // Add caching step if beneficial
        if self.should_cache_query(&cache_key) {
            execution_steps.push(ExecutionStep::CacheResult {
                cache_key: cache_key.clone(),
                ttl_seconds: self.config.default_ttl.as_secs(),
            });
        }

        let expected_result_size = self.estimate_result_size(&query_type);

        Ok(QueryPlan {
            query_id,
            query_type,
            estimated_cost,
            cache_key: Some(cache_key),
            execution_steps,
            expected_result_size,
        })
    }

    /// Check if result is cached
    pub fn get_cached_result(&self, cache_key: &str) -> Option<Vec<u8>> {
        let mut cache = self.cache.write().ok()?;
        
        if let Some(entry) = cache.get_mut(cache_key) {
            // Check TTL
            if entry.created_at.elapsed() > entry.ttl {
                cache.remove(cache_key);
                return None;
            }

            // Update access info
            entry.last_accessed = Instant::now();
            entry.access_count += 1;
            
            Some(entry.result.clone())
        } else {
            None
        }
    }

    /// Cache a query result
    pub fn cache_result(
        &self,
        cache_key: String,
        result: Vec<u8>,
        custom_ttl: Option<Duration>,
    ) -> Result<()> {
        let mut cache = self.cache.write().unwrap();
        
        // Evict entries if cache is full
        if cache.len() >= self.config.max_cache_size {
            self.evict_cache_entries(&mut cache);
        }

        let entry = CacheEntry {
            result: result.clone(),
            created_at: Instant::now(),
            last_accessed: Instant::now(),
            access_count: 1,
            ttl: custom_ttl.unwrap_or(self.config.default_ttl),
            result_size: result.len(),
        };

        cache.insert(cache_key, entry);
        Ok(())
    }

    /// Record query execution statistics
    pub fn record_query_stats(
        &self,
        cache_key: &str,
        execution_time: Duration,
        was_cache_hit: bool,
        result_size: usize,
    ) {
        let mut stats = self.stats.write().unwrap();
        let query_stats = stats.entry(cache_key.to_string()).or_default();
        
        query_stats.execution_count += 1;
        query_stats.total_execution_time += execution_time;
        query_stats.average_execution_time = 
            query_stats.total_execution_time / query_stats.execution_count as u32;
        
        // Update cache hit rate
        let hit_count = if was_cache_hit {
            query_stats.cache_hit_rate * (query_stats.execution_count - 1) as f64 + 1.0
        } else {
            query_stats.cache_hit_rate * (query_stats.execution_count - 1) as f64
        };
        query_stats.cache_hit_rate = hit_count / query_stats.execution_count as f64;
        
        // Update result size distribution
        let size_bucket = Self::get_size_bucket(result_size);
        *query_stats.result_size_distribution.entry(size_bucket).or_insert(0) += 1;
    }

    /// Generate cache key for a query
    fn generate_cache_key(&self, query_type: &QueryType) -> String {
        use std::collections::hash_map::DefaultHasher;
        
        let mut hasher = DefaultHasher::new();
        query_type.hash(&mut hasher);
        format!("query_{:x}", hasher.finish())
    }

    /// Estimate query execution cost
    fn estimate_query_cost(&self, query_type: &QueryType) -> f64 {
        match query_type {
            QueryType::VectorSearch { limit, filters, .. } => {
                let mut cost = *limit as f64 * 0.1; // Base vector search cost
                if filters.is_some() {
                    cost += *limit as f64 * 0.05; // Filter overhead
                }
                cost
            },
            QueryType::TextSearch { limit, .. } => {
                *limit as f64 * 0.15 // Text search typically more expensive
            },
            QueryType::DomainQuery { limit, .. } => {
                limit.unwrap_or(100) as f64 * 0.02 // Simple metadata lookup
            },
            QueryType::CodeSearch { limit, .. } => {
                limit.unwrap_or(100) as f64 * 0.08 // Specialized search
            },
            QueryType::AnalyticsQuery { sql } => {
                sql.len() as f64 * 0.001 // Simple heuristic based on query length
            },
        }
    }

    /// Estimate result size
    fn estimate_result_size(&self, query_type: &QueryType) -> Option<usize> {
        match query_type {
            QueryType::VectorSearch { limit, .. } => Some(*limit),
            QueryType::TextSearch { limit, .. } => Some(*limit),
            QueryType::DomainQuery { limit, .. } => Some(limit.unwrap_or(100)),
            QueryType::CodeSearch { limit, .. } => Some(limit.unwrap_or(100)),
            QueryType::AnalyticsQuery { .. } => None, // Variable size
        }
    }

    /// Determine if a query should be cached
    fn should_cache_query(&self, cache_key: &str) -> bool {
        let stats = self.stats.read().unwrap();
        
        if let Some(query_stats) = stats.get(cache_key) {
            // Cache if hit rate is above threshold or query is slow
            query_stats.cache_hit_rate >= self.config.cache_hit_threshold ||
            query_stats.average_execution_time > Duration::from_millis(500)
        } else {
            // Cache new queries by default
            true
        }
    }

    /// Evict cache entries using LRU + access frequency
    fn evict_cache_entries(&self, cache: &mut HashMap<String, CacheEntry>) {
        let target_size = self.config.max_cache_size * 8 / 10; // Remove 20%
        
        let mut entries: Vec<_> = cache.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        
        // Sort by a combination of last access time and access count
        entries.sort_by(|a, b| {
            let score_a = a.1.access_count as f64 / a.1.last_accessed.elapsed().as_secs_f64().max(1.0);
            let score_b = b.1.access_count as f64 / b.1.last_accessed.elapsed().as_secs_f64().max(1.0);
            score_a.partial_cmp(&score_b).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        // Remove lowest scoring entries
        let to_remove = cache.len() - target_size;
        for (key, _) in entries.iter().take(to_remove) {
            cache.remove(key);
        }
    }

    /// Get size bucket for statistics
    fn get_size_bucket(size: usize) -> usize {
        match size {
            0..=1000 => 1000,
            1001..=10000 => 10000,
            10001..=100000 => 100000,
            _ => 1000000,
        }
    }

    /// Get optimizer statistics
    pub fn get_optimizer_stats(&self) -> OptimizerStats {
        let cache = self.cache.read().unwrap();
        let stats = self.stats.read().unwrap();
        
        let cache_size = cache.len();
        let total_cache_size_bytes: usize = cache.values().map(|e| e.result_size).sum();
        
        let total_queries: u64 = stats.values().map(|s| s.execution_count).sum();
        let average_hit_rate = if !stats.is_empty() {
            stats.values().map(|s| s.cache_hit_rate).sum::<f64>() / stats.len() as f64
        } else {
            0.0
        };
        
        OptimizerStats {
            cache_entries: cache_size,
            cache_size_bytes: total_cache_size_bytes,
            total_queries,
            average_cache_hit_rate: average_hit_rate,
            query_types_tracked: stats.len(),
        }
    }

    /// Clear all cached results
    pub fn clear_cache(&self) {
        let mut cache = self.cache.write().unwrap();
        cache.clear();
    }
}

/// Optimizer statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerStats {
    pub cache_entries: usize,
    pub cache_size_bytes: usize,
    pub total_queries: u64,
    pub average_cache_hit_rate: f64,
    pub query_types_tracked: usize,
}