use std::sync::Arc;
use tokio::sync::Semaphore;
use tracing::{debug, info};

use crate::{
    EmbeddingModel, EmbeddingsError, Result, TokenizerWrapper,
    normalize_embeddings, TextPreprocessor,
};

pub struct BatchProcessor {
    model: Arc<dyn EmbeddingModel>,
    tokenizer: Arc<TokenizerWrapper>,
    preprocessor: TextPreprocessor,
    max_batch_size: usize,
    normalize: bool,
    semaphore: Arc<Semaphore>,
}

impl BatchProcessor {
    pub fn new(
        model: Arc<dyn EmbeddingModel>,
        tokenizer: Arc<TokenizerWrapper>,
        max_batch_size: usize,
        max_concurrent: usize,
    ) -> Self {
        Self {
            model,
            tokenizer,
            preprocessor: TextPreprocessor::default(),
            max_batch_size,
            normalize: true,
            semaphore: Arc::new(Semaphore::new(max_concurrent)),
        }
    }
    
    pub fn with_preprocessor(mut self, preprocessor: TextPreprocessor) -> Self {
        self.preprocessor = preprocessor;
        self
    }
    
    pub fn with_normalization(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }
    
    pub async fn process_batch(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }
        
        if texts.len() > self.max_batch_size {
            return Err(EmbeddingsError::BatchSizeExceeded {
                max: self.max_batch_size,
                actual: texts.len(),
            });
        }
        
        let _permit = self.semaphore.acquire().await.unwrap();
        
        // Preprocess texts
        let processed = self.preprocessor.process_batch(&texts);
        
        // Tokenize
        let (input_ids, attention_mask) = self.tokenizer
            .encode_batch(&processed, self.model.device())?;
        
        // Generate embeddings
        let embeddings = self.model.embed(&input_ids, &attention_mask)?;
        
        // Normalize if requested
        let final_embeddings = if self.normalize {
            normalize_embeddings(&embeddings, 2.0)?
        } else {
            embeddings
        };
        
        // Convert to Vec<Vec<f32>>
        let shape = final_embeddings.shape();
        let batch_size = shape.dims()[0];
        let hidden_size = shape.dims()[1];
        
        let flat = final_embeddings.flatten_all()?.to_vec1::<f32>()?;
        let mut result = Vec::with_capacity(batch_size);
        
        for i in 0..batch_size {
            let start = i * hidden_size;
            let end = start + hidden_size;
            result.push(flat[start..end].to_vec());
        }
        
        debug!("Processed batch of {} texts", texts.len());
        
        Ok(result)
    }
    
    pub async fn process_single(&self, text: String) -> Result<Vec<f32>> {
        let results = self.process_batch(vec![text]).await?;
        Ok(results.into_iter().next().unwrap())
    }
    
    pub async fn process_stream<S>(
        &self,
        mut stream: S,
    ) -> Result<Vec<Vec<f32>>>
    where
        S: futures::Stream<Item = String> + Unpin,
    {
        use futures::StreamExt;
        
        let mut all_embeddings = Vec::new();
        let mut batch = Vec::new();
        
        while let Some(text) = stream.next().await {
            batch.push(text);
            
            if batch.len() >= self.max_batch_size {
                let embeddings = self.process_batch(std::mem::take(&mut batch)).await?;
                all_embeddings.extend(embeddings);
            }
        }
        
        // Process remaining
        if !batch.is_empty() {
            let embeddings = self.process_batch(batch).await?;
            all_embeddings.extend(embeddings);
        }
        
        Ok(all_embeddings)
    }
}

// Chunking strategies for long texts
pub struct TextChunker {
    chunk_size: usize,
    overlap: usize,
    separator: String,
}

impl Default for TextChunker {
    fn default() -> Self {
        Self {
            chunk_size: 256,
            overlap: 50,
            separator: " ".to_string(),
        }
    }
}

impl TextChunker {
    pub fn new(chunk_size: usize, overlap: usize) -> Self {
        Self {
            chunk_size,
            overlap,
            separator: " ".to_string(),
        }
    }
    
    pub fn with_separator(mut self, separator: String) -> Self {
        self.separator = separator;
        self
    }
    
    pub fn chunk_text(&self, text: &str) -> Vec<String> {
        if text.len() <= self.chunk_size {
            return vec![text.to_string()];
        }
        
        let words: Vec<String> = text.split(&self.separator).map(|s| s.to_string()).collect();
        let mut chunks = Vec::new();
        let mut current_chunk = Vec::new();
        let mut current_size = 0;
        
        for word in words {
            let word_size = word.len() + self.separator.len();
            
            if current_size + word_size > self.chunk_size && !current_chunk.is_empty() {
                // Save current chunk
                chunks.push(current_chunk.join(self.separator.as_str()));
                
                // Start new chunk with overlap
                let overlap_words = self.calculate_overlap(&current_chunk);
                current_chunk = overlap_words;
                current_size = current_chunk.iter()
                    .map(|w| w.len() + self.separator.len())
                    .sum();
            }
            
            current_chunk.push(word);
            current_size += word_size;
        }
        
        // Add final chunk
        if !current_chunk.is_empty() {
            chunks.push(current_chunk.join(self.separator.as_str()));
        }
        
        chunks
    }
    
    fn calculate_overlap(&self, chunk: &[String]) -> Vec<String> {
        let mut overlap_size = 0;
        let mut overlap_words = Vec::new();
        
        for word in chunk.iter().rev() {
            let word_size = word.len() + self.separator.len();
            if overlap_size + word_size <= self.overlap {
                overlap_words.insert(0, word.clone());
                overlap_size += word_size;
            } else {
                break;
            }
        }
        
        overlap_words
    }
}

// Parallel processing for large document collections
pub struct ParallelProcessor {
    processors: Vec<Arc<BatchProcessor>>,
    chunk_size: usize,
}

impl ParallelProcessor {
    pub fn new(
        model: Arc<dyn EmbeddingModel>,
        tokenizer: Arc<TokenizerWrapper>,
        num_workers: usize,
        max_batch_size: usize,
    ) -> Self {
        let processors = (0..num_workers)
            .map(|_| {
                Arc::new(BatchProcessor::new(
                    model.clone(),
                    tokenizer.clone(),
                    max_batch_size,
                    1, // Each processor has its own semaphore
                ))
            })
            .collect();
        
        Self {
            processors,
            chunk_size: max_batch_size * num_workers,
        }
    }
    
    pub async fn process_documents(&self, documents: Vec<String>) -> Result<Vec<Vec<f32>>> {
        use futures::future::join_all;
        
        info!("Processing {} documents in parallel", documents.len());
        
        let chunks: Vec<Vec<String>> = documents
            .chunks(self.chunk_size)
            .map(|chunk| chunk.to_vec())
            .collect();
        
        let mut handles = Vec::new();
        
        for (i, chunk) in chunks.into_iter().enumerate() {
            let processor = self.processors[i % self.processors.len()].clone();
            
            let handle = tokio::spawn(async move {
                let mut results = Vec::new();
                
                for batch in chunk.chunks(processor.max_batch_size) {
                    let embeddings = processor.process_batch(batch.to_vec()).await?;
                    results.extend(embeddings);
                }
                
                Ok::<Vec<Vec<f32>>, EmbeddingsError>(results)
            });
            
            handles.push(handle);
        }
        
        let results = join_all(handles).await;
        
        let mut all_embeddings = Vec::new();
        for result in results {
            let embeddings = result.map_err(|e| 
                EmbeddingsError::Model(format!("Task join error: {}", e))
            )??;
            all_embeddings.extend(embeddings);
        }
        
        Ok(all_embeddings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_text_chunker() {
        let chunker = TextChunker::new(50, 10);
        let text = "This is a long text that needs to be chunked into smaller pieces for processing";
        
        let chunks = chunker.chunk_text(text);
        assert!(chunks.len() > 1);
        
        // Check overlap
        for i in 1..chunks.len() {
            let prev_words: Vec<&str> = chunks[i-1].split(' ').collect();
            let curr_words: Vec<&str> = chunks[i].split(' ').collect();
            
            // Should have some overlapping words
            let overlap_exists = prev_words.iter()
                .any(|w| curr_words.contains(w));
            assert!(overlap_exists);
        }
    }
    
    #[test]
    fn test_chunk_size() {
        let chunker = TextChunker::new(20, 5);
        let text = "one two three four five six seven eight nine ten";
        
        let chunks = chunker.chunk_text(text);
        for chunk in &chunks {
            assert!(chunk.len() <= 30); // Some buffer for word boundaries
        }
    }
}