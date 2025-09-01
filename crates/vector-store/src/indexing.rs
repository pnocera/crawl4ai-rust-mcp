use std::sync::Arc;
use tokio::sync::{mpsc, Semaphore};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::{PagePoint, PageVectors, Result, VectorStore, VectorStoreError};

pub struct IndexingPipeline {
    vector_store: Arc<VectorStore>,
    batch_size: usize,
    max_concurrent: usize,
}

impl IndexingPipeline {
    pub fn new(vector_store: Arc<VectorStore>) -> Self {
        Self {
            vector_store,
            batch_size: 100,
            max_concurrent: 4,
        }
    }
    
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }
    
    pub fn with_max_concurrent(mut self, max: usize) -> Self {
        self.max_concurrent = max;
        self
    }
    
    pub async fn index_pages(
        &self,
        pages: Vec<(PagePoint, PageVectors)>,
    ) -> Result<IndexingStats> {
        let total = pages.len();
        let semaphore = Arc::new(Semaphore::new(self.max_concurrent));
        let (tx, mut rx) = mpsc::channel::<IndexResult>(100);
        
        info!("Starting indexing of {} pages", total);
        
        // Process in batches
        let chunks: Vec<_> = pages
            .into_iter()
            .collect::<Vec<_>>()
            .chunks(self.batch_size)
            .map(|chunk| chunk.to_vec())
            .collect();
        
        // Spawn tasks for each batch
        let mut handles = Vec::new();
        
        for (batch_idx, batch) in chunks.into_iter().enumerate() {
            let vector_store = self.vector_store.clone();
            let semaphore = semaphore.clone();
            let tx = tx.clone();
            
            let handle = tokio::spawn(async move {
                let _permit = semaphore.acquire().await.unwrap();
                
                for (point, vectors) in batch {
                    let id = point.id;
                    let url = point.url.clone();
                    
                    match vector_store.upsert_page(point, vectors).await {
                        Ok(_) => {
                            let _ = tx.send(IndexResult::Success { id, url }).await;
                        }
                        Err(e) => {
                            let _ = tx.send(IndexResult::Failed {
                                id,
                                url,
                                error: e.to_string(),
                            }).await;
                        }
                    }
                }
                
                debug!("Completed batch {}", batch_idx);
            });
            
            handles.push(handle);
        }
        
        // Drop the original sender so the receiver knows when we're done
        drop(tx);
        
        // Collect results
        let mut stats = IndexingStats::default();
        
        while let Some(result) = rx.recv().await {
            match result {
                IndexResult::Success { .. } => stats.successful += 1,
                IndexResult::Failed { error, .. } => {
                    stats.failed += 1;
                    error!("Indexing failed: {}", error);
                }
            }
        }
        
        // Wait for all tasks to complete
        for handle in handles {
            handle.await.map_err(|e| {
                VectorStoreError::Indexing(format!("Task join error: {}", e))
            })?;
        }
        
        stats.total = total;
        
        info!(
            "Indexing completed: {}/{} successful",
            stats.successful, stats.total
        );
        
        Ok(stats)
    }
    
    pub async fn reindex_with_binary_quantization(
        &self,
        collection_name: &str,
    ) -> Result<()> {
        info!("Starting reindexing with binary quantization for collection: {}", collection_name);
        
        // This would typically:
        // 1. Create a new collection with binary quantization enabled
        // 2. Copy all points from the old collection
        // 3. Delete the old collection
        // 4. Rename the new collection
        
        // For now, this is a placeholder
        warn!("Reindexing with binary quantization not yet implemented");
        
        Ok(())
    }
}

#[derive(Debug, Default)]
pub struct IndexingStats {
    pub total: usize,
    pub successful: usize,
    pub failed: usize,
}

enum IndexResult {
    Success { id: Uuid, url: String },
    Failed { id: Uuid, url: String, error: String },
}

// Batch builder for efficient indexing
pub struct BatchBuilder {
    points: Vec<PagePoint>,
    vectors: Vec<PageVectors>,
    max_size: usize,
}

impl BatchBuilder {
    pub fn new(max_size: usize) -> Self {
        Self {
            points: Vec::with_capacity(max_size),
            vectors: Vec::with_capacity(max_size),
            max_size,
        }
    }
    
    pub fn add(&mut self, point: PagePoint, vectors: PageVectors) -> Option<Vec<(PagePoint, PageVectors)>> {
        self.points.push(point);
        self.vectors.push(vectors);
        
        if self.points.len() >= self.max_size {
            self.flush()
        } else {
            None
        }
    }
    
    pub fn flush(&mut self) -> Option<Vec<(PagePoint, PageVectors)>> {
        if self.points.is_empty() {
            return None;
        }
        
        let points = std::mem::take(&mut self.points);
        let vectors = std::mem::take(&mut self.vectors);
        
        Some(points.into_iter().zip(vectors).collect())
    }
}

// Stream-based indexing for large datasets
pub struct StreamIndexer {
    pipeline: Arc<IndexingPipeline>,
    buffer_size: usize,
}

impl StreamIndexer {
    pub fn new(pipeline: Arc<IndexingPipeline>) -> Self {
        Self {
            pipeline,
            buffer_size: 1000,
        }
    }
    
    pub async fn index_stream<S>(
        &self,
        mut stream: S,
    ) -> Result<IndexingStats>
    where
        S: futures::Stream<Item = (PagePoint, PageVectors)> + Unpin,
    {
        use futures::StreamExt;
        
        let mut batch_builder = BatchBuilder::new(self.buffer_size);
        let mut total_stats = IndexingStats::default();
        
        while let Some((point, vectors)) = stream.next().await {
            if let Some(batch) = batch_builder.add(point, vectors) {
                let stats = self.pipeline.index_pages(batch).await?;
                total_stats.total += stats.total;
                total_stats.successful += stats.successful;
                total_stats.failed += stats.failed;
            }
        }
        
        // Index remaining items
        if let Some(batch) = batch_builder.flush() {
            let stats = self.pipeline.index_pages(batch).await?;
            total_stats.total += stats.total;
            total_stats.successful += stats.successful;
            total_stats.failed += stats.failed;
        }
        
        Ok(total_stats)
    }
}