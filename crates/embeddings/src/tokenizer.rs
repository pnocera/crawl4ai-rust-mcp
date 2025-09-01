use candle_core::{Device, Tensor};
use tokenizers::{
    Encoding, PaddingParams, PaddingStrategy, Tokenizer, TruncationParams,
};
use tracing::debug;
use unicode_categories::UnicodeCategories;

use crate::{EmbeddingsError, Result};

pub struct TokenizerWrapper {
    tokenizer: Tokenizer,
    max_length: usize,
}

impl TokenizerWrapper {
    pub fn new(tokenizer_path: &std::path::Path, max_length: usize) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(tokenizer_path)?;
        
        Ok(Self {
            tokenizer,
            max_length,
        })
    }
    
    pub fn encode_batch(
        &self,
        texts: &[String],
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        // Set padding and truncation
        let mut tokenizer = self.tokenizer.clone();
        
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            ..Default::default()
        }));
        
        tokenizer.with_truncation(Some(TruncationParams {
            max_length: self.max_length,
            ..Default::default()
        }));
        
        // Encode texts
        let encodings = tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| EmbeddingsError::Tokenizer(e))?;
        
        // Convert to tensors
        let (input_ids, attention_masks) = self.encodings_to_tensors(&encodings, device)?;
        
        debug!(
            "Tokenized {} texts, shape: {:?}",
            texts.len(),
            input_ids.shape()
        );
        
        Ok((input_ids, attention_masks))
    }
    
    pub fn encode_single(
        &self,
        text: &str,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        self.encode_batch(&[text.to_string()], device)
    }
    
    fn encodings_to_tensors(
        &self,
        encodings: &[Encoding],
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        let batch_size = encodings.len();
        let seq_len = encodings[0].get_ids().len();
        
        // Flatten input IDs and attention masks
        let mut input_ids_flat = Vec::with_capacity(batch_size * seq_len);
        let mut attention_mask_flat = Vec::with_capacity(batch_size * seq_len);
        
        for encoding in encodings {
            input_ids_flat.extend(encoding.get_ids().iter().map(|&id| id as i64));
            attention_mask_flat.extend(
                encoding
                    .get_attention_mask()
                    .iter()
                    .map(|&mask| mask as f32),
            );
        }
        
        // Create tensors
        let input_ids = Tensor::from_vec(
            input_ids_flat,
            (batch_size, seq_len),
            device,
        )?;
        
        let attention_mask = Tensor::from_vec(
            attention_mask_flat,
            (batch_size, seq_len),
            device,
        )?;
        
        Ok((input_ids, attention_mask))
    }
}

// Text preprocessing utilities
pub struct TextPreprocessor {
    lowercase: bool,
    strip_accents: bool,
    max_length: usize,
}

impl Default for TextPreprocessor {
    fn default() -> Self {
        Self {
            lowercase: true,
            strip_accents: false,
            max_length: 512,
        }
    }
}

impl TextPreprocessor {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn with_lowercase(mut self, lowercase: bool) -> Self {
        self.lowercase = lowercase;
        self
    }
    
    pub fn with_strip_accents(mut self, strip: bool) -> Self {
        self.strip_accents = strip;
        self
    }
    
    pub fn process(&self, text: &str) -> String {
        let mut processed = text.trim().to_string();
        
        if self.lowercase {
            processed = processed.to_lowercase();
        }
        
        if self.strip_accents {
            processed = self.remove_accents(&processed);
        }
        
        // Truncate if needed
        if processed.len() > self.max_length {
            processed.truncate(self.max_length);
        }
        
        processed
    }
    
    pub fn process_batch(&self, texts: &[String]) -> Vec<String> {
        texts.iter().map(|text| self.process(text)).collect()
    }
    
    fn remove_accents(&self, text: &str) -> String {
        use unicode_normalization::UnicodeNormalization;
        
        text.nfd()
            .filter(|c| !c.is_mark())
            .collect()
    }
}

// Special tokens handling
pub struct SpecialTokens {
    pub cls_token: String,
    pub sep_token: String,
    pub pad_token: String,
    pub unk_token: String,
    pub mask_token: String,
}

impl Default for SpecialTokens {
    fn default() -> Self {
        Self {
            cls_token: "[CLS]".to_string(),
            sep_token: "[SEP]".to_string(),
            pad_token: "[PAD]".to_string(),
            unk_token: "[UNK]".to_string(),
            mask_token: "[MASK]".to_string(),
        }
    }
}

// Instruction formatting for models that support it
pub fn format_for_e5(text: &str, is_query: bool) -> String {
    if is_query {
        format!("query: {}", text)
    } else {
        format!("passage: {}", text)
    }
}

pub fn format_for_bge(text: &str, is_query: bool) -> String {
    if is_query {
        format!("Represent this sentence for searching relevant passages: {}", text)
    } else {
        text.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_text_preprocessor() {
        let processor = TextPreprocessor::new();
        
        let text = "  HELLO World!  ";
        let processed = processor.process(text);
        assert_eq!(processed, "hello world!");
        
        let processor = TextPreprocessor::new().with_lowercase(false);
        let processed = processor.process(text);
        assert_eq!(processed, "HELLO World!");
    }
    
    #[test]
    fn test_special_formatting() {
        let text = "search for rust documentation";
        
        let e5_query = format_for_e5(text, true);
        assert_eq!(e5_query, "query: search for rust documentation");
        
        let e5_passage = format_for_e5(text, false);
        assert_eq!(e5_passage, "passage: search for rust documentation");
        
        let bge_query = format_for_bge(text, true);
        assert!(bge_query.contains("Represent this sentence"));
    }
}