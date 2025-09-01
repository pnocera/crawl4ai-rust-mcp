use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, Semaphore};
use tokio::time::sleep;
use uuid::Uuid;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};
use tracing::{info, warn, error};

use crate::{Result, StorageError, HybridStorage, DuckDbStore, RocksDbStore};

/// Synchronization events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SyncEvent {
    PageAdded {
        page_id: Uuid,
        url: String,
        domain: String,
        timestamp: DateTime<Utc>,
    },
    PageUpdated {
        page_id: Uuid,
        url: String,
        changes: Vec<String>,
        timestamp: DateTime<Utc>,
    },
    PageDeleted {
        page_id: Uuid,
        timestamp: DateTime<Utc>,
    },
    ChunkAdded {
        chunk_id: String,
        page_id: Uuid,
        timestamp: DateTime<Utc>,
    },
    EmbeddingUpdated {
        page_id: Uuid,
        embedding_offset: usize,
        timestamp: DateTime<Utc>,
    },
}

/// Synchronization strategy
#[derive(Debug, Clone, Copy)]
pub enum SyncStrategy {
    /// Immediate synchronization - sync on every write
    Immediate,
    /// Batch synchronization - sync in batches periodically
    Batch { batch_size: usize, interval_ms: u64 },
    /// Manual synchronization - only sync when requested
    Manual,
}

/// Synchronization manager for keeping stores consistent
pub struct SyncManager {
    storage: Arc<HybridStorage>,
    strategy: SyncStrategy,
    pending_events: Arc<RwLock<Vec<SyncEvent>>>,
    sync_semaphore: Arc<Semaphore>,
    last_sync: Arc<RwLock<Instant>>,
}

impl SyncManager {
    pub fn new(storage: Arc<HybridStorage>, strategy: SyncStrategy) -> Self {
        Self {
            storage,
            strategy,
            pending_events: Arc::new(RwLock::new(Vec::new())),
            sync_semaphore: Arc::new(Semaphore::new(1)), // Only one sync at a time
            last_sync: Arc::new(RwLock::new(Instant::now())),
        }
    }

    /// Add a synchronization event
    pub async fn add_sync_event(&self, event: SyncEvent) -> Result<()> {
        {
            let mut events = self.pending_events.write().await;
            events.push(event);
        }

        match self.strategy {
            SyncStrategy::Immediate => {
                self.sync_now().await?;
            },
            SyncStrategy::Batch { batch_size, .. } => {
                let events_count = {
                    let events = self.pending_events.read().await;
                    events.len()
                };
                
                if events_count >= batch_size {
                    self.sync_now().await?;
                }
            },
            SyncStrategy::Manual => {
                // Do nothing, wait for manual sync
            }
        }

        Ok(())
    }

    /// Force synchronization now
    pub async fn sync_now(&self) -> Result<()> {
        let _permit = self.sync_semaphore.acquire().await.unwrap();

        let events = {
            let mut events = self.pending_events.write().await;
            let current_events = events.clone();
            events.clear();
            current_events
        };

        if events.is_empty() {
            return Ok(());
        }

        info!("Starting synchronization of {} events", events.len());
        let start_time = Instant::now();

        // Process events by type for efficiency
        let mut stats = SyncStats::default();
        
        for event in events {
            match self.process_sync_event(event).await {
                Ok(_) => stats.success_count += 1,
                Err(e) => {
                    stats.error_count += 1;
                    warn!("Sync event failed: {}", e);
                }
            }
        }

        let sync_duration = start_time.elapsed();
        stats.duration = sync_duration;
        
        {
            let mut last_sync = self.last_sync.write().await;
            *last_sync = Instant::now();
        }

        info!(
            "Synchronization completed: {} success, {} errors, took {:?}",
            stats.success_count, stats.error_count, sync_duration
        );

        Ok(())
    }

    /// Process a single sync event
    async fn process_sync_event(&self, event: SyncEvent) -> Result<()> {
        match event {
            SyncEvent::PageAdded { page_id, url, domain, .. } => {
                self.sync_page_added(page_id, &url, &domain).await?;
            },
            SyncEvent::PageUpdated { page_id, url, changes, .. } => {
                self.sync_page_updated(page_id, &url, &changes).await?;
            },
            SyncEvent::PageDeleted { page_id, .. } => {
                self.sync_page_deleted(page_id).await?;
            },
            SyncEvent::ChunkAdded { chunk_id, page_id, .. } => {
                self.sync_chunk_added(&chunk_id, page_id).await?;
            },
            SyncEvent::EmbeddingUpdated { page_id, embedding_offset, .. } => {
                self.sync_embedding_updated(page_id, embedding_offset).await?;
            },
        }
        Ok(())
    }

    /// Synchronize a newly added page across all stores
    async fn sync_page_added(&self, page_id: Uuid, url: &str, domain: &str) -> Result<()> {
        // Get page data from hybrid storage
        let page = self.storage.get_page(page_id).await?
            .ok_or_else(|| StorageError::NotFound(format!("Page {}", page_id)))?;

        // Ensure it's in DuckDB analytics store
        self.storage.analytics_store().insert_page(
            &page_id.to_string(),
            url,
            &page.title,
            &page.content,
            domain,
            Some("unknown"), // Language detection can be added later
            &serde_json::json!({}), // Metadata
        )?;

        // Update counters in RocksDB
        self.storage.metadata_store().increment_counter("total_pages", 1)?;
        self.storage.metadata_store().increment_counter(&format!("domain_{}", domain), 1)?;

        Ok(())
    }

    /// Synchronize page updates
    async fn sync_page_updated(&self, page_id: Uuid, url: &str, _changes: &[String]) -> Result<()> {
        // Get updated page data
        let page = self.storage.get_page(page_id).await?
            .ok_or_else(|| StorageError::NotFound(format!("Page {}", page_id)))?;

        // Update in DuckDB
        let domain = Self::extract_domain(url);
        self.storage.analytics_store().insert_page(
            &page_id.to_string(),
            url,
            &page.title,
            &page.content,
            &domain,
            Some("unknown"),
            &serde_json::json!({}),
        )?;

        Ok(())
    }

    /// Synchronize page deletion
    async fn sync_page_deleted(&self, page_id: Uuid) -> Result<()> {
        // Remove from metadata store
        self.storage.metadata_store().delete(page_id.as_bytes())?;

        // Note: DuckDB doesn't support deletion easily in this simplified version
        // In a full implementation, you'd add a deleted_at timestamp column

        // Update counters
        self.storage.metadata_store().increment_counter("total_pages", u64::MAX)?; // Wrapping subtract

        Ok(())
    }

    /// Synchronize chunk addition
    async fn sync_chunk_added(&self, chunk_id: &str, page_id: Uuid) -> Result<()> {
        // This could involve updating chunk indexes in RocksDB
        // and ensuring DuckDB has the chunk information
        
        // For now, just increment chunk counter
        self.storage.metadata_store().increment_counter(&format!("page_{}_chunks", page_id), 1)?;

        Ok(())
    }

    /// Synchronize embedding updates
    async fn sync_embedding_updated(&self, page_id: Uuid, _embedding_offset: usize) -> Result<()> {
        // Update embedding metadata in RocksDB
        let metadata_key = format!("embedding_meta_{}", page_id);
        let metadata = serde_json::json!({
            "last_updated": Utc::now(),
            "page_id": page_id
        });
        
        self.storage.metadata_store().put_cf(
            "metadata",
            metadata_key.as_bytes(),
            &metadata,
        )?;

        Ok(())
    }

    /// Start background sync process for batch strategy
    pub async fn start_background_sync(&self) {
        if let SyncStrategy::Batch { interval_ms, .. } = self.strategy {
            let sync_manager = Arc::new(self.clone());
            let interval = Duration::from_millis(interval_ms);
            
            tokio::spawn(async move {
                loop {
                    sleep(interval).await;
                    
                    if let Err(e) = sync_manager.sync_now().await {
                        error!("Background sync failed: {}", e);
                    }
                }
            });
        }
    }

    /// Get synchronization statistics
    pub async fn get_sync_stats(&self) -> SyncStats {
        let pending_count = {
            let events = self.pending_events.read().await;
            events.len()
        };
        
        let last_sync_time = {
            let last_sync = self.last_sync.read().await;
            last_sync.elapsed()
        };

        SyncStats {
            pending_events: pending_count,
            last_sync_ago: last_sync_time,
            ..Default::default()
        }
    }

    /// Perform full consistency check between stores
    pub async fn consistency_check(&self) -> Result<ConsistencyReport> {
        info!("Starting consistency check across all stores");
        
        let mut report = ConsistencyReport::default();
        
        // Check RocksDB vs DuckDB consistency
        // This is a simplified version - a full implementation would be more comprehensive
        
        let total_pages_counter = self.storage.metadata_store()
            .get_counter("total_pages")?;
        
        let duckdb_pages = self.storage.analytics_store()
            .execute_custom_query("SELECT COUNT(*) as count FROM crawled_pages")?;
        
        let duckdb_count = duckdb_pages.first()
            .and_then(|v| v.get("count"))
            .and_then(|v| v.as_i64())
            .unwrap_or(0) as u64;
        
        if total_pages_counter != duckdb_count {
            report.inconsistencies.push(format!(
                "Page count mismatch: RocksDB counter={}, DuckDB count={}",
                total_pages_counter, duckdb_count
            ));
        }
        
        report.checks_performed = 1;
        report.consistent = report.inconsistencies.is_empty();
        
        info!("Consistency check completed: {} inconsistencies found", report.inconsistencies.len());
        
        Ok(report)
    }

    /// Extract domain from URL
    fn extract_domain(url: &str) -> String {
        url.parse::<url::Url>()
            .ok()
            .and_then(|u| u.host_str().map(|s| s.to_string()))
            .unwrap_or_else(|| "unknown".to_string())
    }
}

// Enable cloning for background tasks
impl Clone for SyncManager {
    fn clone(&self) -> Self {
        Self {
            storage: self.storage.clone(),
            strategy: self.strategy,
            pending_events: self.pending_events.clone(),
            sync_semaphore: self.sync_semaphore.clone(),
            last_sync: self.last_sync.clone(),
        }
    }
}

/// Synchronization statistics
#[derive(Debug, Default)]
pub struct SyncStats {
    pub pending_events: usize,
    pub success_count: u32,
    pub error_count: u32,
    pub duration: Duration,
    pub last_sync_ago: Duration,
}

/// Consistency check report
#[derive(Debug, Default)]
pub struct ConsistencyReport {
    pub consistent: bool,
    pub checks_performed: u32,
    pub inconsistencies: Vec<String>,
}