use std::sync::Arc;

use crawler::Crawler;
use embeddings::EmbeddingService;
use graph_store::GraphStore;
use search::SearchService;
use storage::HybridStorage;
use vector_store::VectorStore;

use crate::ServerConfig;

// Application state shared across handlers
#[derive(Clone)]
pub struct AppState {
    pub config: Arc<ServerConfig>,
    pub storage: Arc<HybridStorage>,
    pub vector_store: Arc<VectorStore>,
    pub graph_store: Arc<GraphStore>,
    pub crawler: Arc<Crawler>,
    pub embeddings: Arc<EmbeddingService>,
    pub search: Arc<SearchService>,
}

impl AppState {
    pub async fn new(
        config: ServerConfig,
        qdrant_config: vector_store::QdrantConfig,
        memgraph_config: graph_store::MemgraphConfig,
    ) -> anyhow::Result<Self> {
        // Initialize storage
        let storage = Arc::new(HybridStorage::new(&config.storage_path)?);
        
        // Initialize vector store
        let collection_config = vector_store::CollectionConfig::default();
        let vector_store = Arc::new(
            VectorStore::new(qdrant_config, collection_config).await?
        );
        
        // Initialize graph store
        let graph_store = Arc::new(
            GraphStore::new(memgraph_config).await?
        );
        
        // Initialize embeddings service
        let embeddings = Arc::new(
            EmbeddingService::new().await?
        );
        
        // Initialize crawler
        let crawler = Arc::new(
            Crawler::new()?
        );
        
        // Initialize search service
        let search = Arc::new(
            SearchService::new(
                vector_store.clone(),
                embeddings.clone(),
                storage.clone(),
            ).await?
        );
        
        Ok(Self {
            config: Arc::new(config),
            storage,
            vector_store,
            graph_store,
            crawler,
            embeddings,
            search,
        })
    }
}