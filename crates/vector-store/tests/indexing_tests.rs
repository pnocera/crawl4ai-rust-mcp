use std::sync::Arc;
use vector_store::{
    BatchBuilder, CollectionConfig, IndexingPipeline, PagePoint, 
    PageVectors, QdrantConfig, StreamIndexer, VectorStore,
};
use futures::stream;

#[test]
fn test_batch_builder() {
    let mut builder = BatchBuilder::new(3);
    
    // Add items below batch size
    let batch = builder.add(
        create_test_point("1"),
        create_test_vectors(),
    );
    assert!(batch.is_none());
    
    let batch = builder.add(
        create_test_point("2"),
        create_test_vectors(),
    );
    assert!(batch.is_none());
    
    // This should trigger a batch
    let batch = builder.add(
        create_test_point("3"),
        create_test_vectors(),
    );
    assert!(batch.is_some());
    assert_eq!(batch.unwrap().len(), 3);
    
    // Builder should be empty after flush
    let batch = builder.flush();
    assert!(batch.is_none());
}

#[test]
fn test_batch_builder_flush() {
    let mut builder = BatchBuilder::new(10);
    
    // Add some items
    builder.add(create_test_point("1"), create_test_vectors());
    builder.add(create_test_point("2"), create_test_vectors());
    
    // Manual flush
    let batch = builder.flush();
    assert!(batch.is_some());
    assert_eq!(batch.unwrap().len(), 2);
    
    // Second flush should return None
    let batch = builder.flush();
    assert!(batch.is_none());
}

#[tokio::test]
#[ignore] // Requires running Qdrant instance
async fn test_indexing_pipeline() {
    let config = QdrantConfig::default();
    let collection_config = CollectionConfig {
        name: "test_indexing".to_string(),
        ..Default::default()
    };
    
    let store = Arc::new(
        VectorStore::new(config, collection_config)
            .await
            .unwrap()
    );
    
    let pipeline = IndexingPipeline::new(store)
        .with_batch_size(2)
        .with_max_concurrent(2);
    
    // Create test data
    let pages: Vec<(PagePoint, PageVectors)> = (0..5)
        .map(|i| (
            create_test_point(&i.to_string()),
            create_test_vectors(),
        ))
        .collect();
    
    let stats = pipeline.index_pages(pages).await.unwrap();
    
    assert_eq!(stats.total, 5);
    assert_eq!(stats.successful, 5);
    assert_eq!(stats.failed, 0);
}

#[tokio::test]
#[ignore] // Requires running Qdrant instance
async fn test_stream_indexer() {
    let config = QdrantConfig::default();
    let collection_config = CollectionConfig {
        name: "test_stream_indexing".to_string(),
        ..Default::default()
    };
    
    let store = Arc::new(
        VectorStore::new(config, collection_config)
            .await
            .unwrap()
    );
    
    let pipeline = Arc::new(IndexingPipeline::new(store));
    let stream_indexer = StreamIndexer::new(pipeline);
    
    // Create a stream of test data
    let data: Vec<(PagePoint, PageVectors)> = (0..10)
        .map(|i| (
            create_test_point(&i.to_string()),
            create_test_vectors(),
        ))
        .collect();
    
    let stream = stream::iter(data);
    
    let stats = stream_indexer.index_stream(stream).await.unwrap();
    
    assert_eq!(stats.total, 10);
    assert_eq!(stats.successful, 10);
    assert_eq!(stats.failed, 0);
}

#[tokio::test]
async fn test_concurrent_indexing() {
    use tokio::sync::mpsc;
    
    let (tx, rx) = mpsc::channel::<(PagePoint, PageVectors)>(100);
    
    // Spawn producer tasks
    let mut handles = vec![];
    
    for i in 0..3 {
        let tx = tx.clone();
        let handle = tokio::spawn(async move {
            for j in 0..5 {
                let point = create_test_point(&format!("{}-{}", i, j));
                let vectors = create_test_vectors();
                tx.send((point, vectors)).await.unwrap();
            }
        });
        handles.push(handle);
    }
    
    // Drop original sender so receiver knows when done
    drop(tx);
    
    // Collect all items
    let mut items = vec![];
    let mut rx = rx;
    while let Some(item) = rx.recv().await {
        items.push(item);
    }
    
    // Wait for all producers
    for handle in handles {
        handle.await.unwrap();
    }
    
    assert_eq!(items.len(), 15);
}

fn create_test_point(id: &str) -> PagePoint {
    PagePoint::new(
        format!("https://test.com/page{}", id),
        format!("Test Page {}", id),
        "test.com".to_string(),
        format!("Content for page {}", id),
    )
}

fn create_test_vectors() -> PageVectors {
    PageVectors {
        title_vector: vec![0.1; 768],
        content_vector: vec![0.2; 768],
        code_vector: Some(vec![0.3; 768]),
    }
}