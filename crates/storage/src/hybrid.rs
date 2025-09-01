use std::path::Path;
use std::sync::Arc;
use uuid::Uuid;

use crate::{
    AppendLog, DuckDbStore, MmapStorage, RocksDbStore, Result, StorageError,
};

// Hybrid storage combining RocksDB, DuckDB, and memory-mapped files
pub struct HybridStorage {
    // RocksDB for metadata and indexes
    metadata_store: Arc<RocksDbStore>,
    
    // DuckDB for analytics
    analytics_store: Arc<DuckDbStore>,
    
    // Memory-mapped append log for content
    content_log: Arc<AppendLog>,
    
    // Memory-mapped storage for embeddings
    embedding_store: Arc<MmapStorage>,
}

impl HybridStorage {
    pub fn new(base_path: impl AsRef<Path>) -> Result<Self> {
        let base_path = base_path.as_ref();
        
        // Create subdirectories
        std::fs::create_dir_all(base_path.join("metadata"))?;
        std::fs::create_dir_all(base_path.join("analytics"))?;
        std::fs::create_dir_all(base_path.join("content"))?;
        std::fs::create_dir_all(base_path.join("embeddings"))?;
        
        Ok(Self {
            metadata_store: Arc::new(RocksDbStore::new(base_path.join("metadata"))?),
            analytics_store: Arc::new(DuckDbStore::new(base_path.join("analytics/pages.db"))?),
            content_log: Arc::new(AppendLog::create(base_path.join("content/pages.log"))?),
            embedding_store: Arc::new(MmapStorage::create(
                base_path.join("embeddings/vectors.mmap"),
                1024 * 1024 * 100, // 100MB initial
            )?),
        })
    }

    pub async fn store_page(
        &self,
        url: String,
        title: String,
        content: String,
        domain: String,
        embedding: Vec<f32>,
        metadata: serde_json::Value,
    ) -> Result<Uuid> {
        let id = Uuid::new_v4();
        
        // Store content in append log
        let content_offset = self.content_log.append(content.as_bytes())?;
        
        // Store embedding in mmap
        let embedding_bytes = embedding_to_bytes(&embedding);
        let embedding_offset = self.embedding_store.append(&embedding_bytes)?;
        
        // Store metadata in RocksDB
        let page_meta = PageMetadata {
            id,
            url: url.clone(),
            title: title.clone(),
            domain: domain.clone(),
            content_offset,
            content_len: content.len(),
            embedding_offset,
            embedding_dims: embedding.len(),
        };
        
        self.metadata_store.put(id.as_bytes(), &page_meta)?;
        
        // Store in DuckDB for analytics
        self.analytics_store.insert_page(
            &id.to_string(),
            &url,
            &title,
            &content,
            &domain,
            None, // language
            &metadata,
        )?;
        
        Ok(id)
    }

    pub async fn get_page(&self, id: Uuid) -> Result<Option<Page>> {
        // Get metadata
        let meta: Option<PageMetadata> = self.metadata_store.get(id.as_bytes())?;
        
        match meta {
            Some(meta) => {
                // Get content from log
                let content_bytes = self.content_log.read(meta.content_offset)?;
                let content = String::from_utf8_lossy(&content_bytes).to_string();
                
                // Get embedding from mmap
                let embedding_bytes = self.embedding_store.read_bytes(
                    meta.embedding_offset,
                    meta.embedding_dims * 4, // f32 = 4 bytes
                )?;
                let embedding = bytes_to_embedding(&embedding_bytes);
                
                Ok(Some(Page {
                    id: meta.id,
                    url: meta.url,
                    title: meta.title,
                    content,
                    domain: meta.domain,
                    embedding,
                }))
            }
            None => Ok(None),
        }
    }

    pub fn metadata_store(&self) -> &Arc<RocksDbStore> {
        &self.metadata_store
    }

    pub fn analytics_store(&self) -> &Arc<DuckDbStore> {
        &self.analytics_store
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct PageMetadata {
    id: Uuid,
    url: String,
    title: String,
    domain: String,
    content_offset: usize,
    content_len: usize,
    embedding_offset: usize,
    embedding_dims: usize,
}

pub struct Page {
    pub id: Uuid,
    pub url: String,
    pub title: String,
    pub content: String,
    pub domain: String,
    pub embedding: Vec<f32>,
}

fn embedding_to_bytes(embedding: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(embedding.len() * 4);
    for &val in embedding {
        bytes.extend_from_slice(&val.to_le_bytes());
    }
    bytes
}

fn bytes_to_embedding(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}