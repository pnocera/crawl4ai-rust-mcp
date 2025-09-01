use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QdrantConfig {
    pub url: String,
    pub api_key: Option<String>,
    pub use_binary_quantization: bool,
    pub binary_quantization_threshold: Option<f32>,
    pub oversampling: f32,
    pub rescore: bool,
}

impl Default for QdrantConfig {
    fn default() -> Self {
        Self {
            url: "http://localhost:6333".to_string(),
            api_key: None,
            use_binary_quantization: true,
            binary_quantization_threshold: Some(0.95),
            oversampling: 3.0,
            rescore: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionConfig {
    pub name: String,
    pub vectors: VectorConfig,
    pub shard_number: Option<u32>,
    pub replication_factor: Option<u32>,
    pub write_consistency_factor: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorConfig {
    pub title: VectorFieldConfig,
    pub content: VectorFieldConfig,
    pub code: Option<VectorFieldConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorFieldConfig {
    pub size: u64,
    pub distance: Distance,
    pub enable_binary_quantization: bool,
    pub on_disk: Option<bool>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Distance {
    Cosine,
    Euclid,
    Dot,
}

impl Default for CollectionConfig {
    fn default() -> Self {
        Self {
            name: "crawled_pages".to_string(),
            vectors: VectorConfig {
                title: VectorFieldConfig {
                    size: 768,
                    distance: Distance::Cosine,
                    enable_binary_quantization: true,
                    on_disk: Some(false),
                },
                content: VectorFieldConfig {
                    size: 768,
                    distance: Distance::Cosine,
                    enable_binary_quantization: true,
                    on_disk: Some(true),
                },
                code: Some(VectorFieldConfig {
                    size: 768,
                    distance: Distance::Cosine,
                    enable_binary_quantization: true,
                    on_disk: Some(true),
                }),
            },
            shard_number: Some(2),
            replication_factor: Some(1),
            write_consistency_factor: Some(1),
        }
    }
}

impl VectorFieldConfig {
    pub fn with_size(mut self, size: u64) -> Self {
        self.size = size;
        self
    }
    
    pub fn with_binary_quantization(mut self, enabled: bool) -> Self {
        self.enable_binary_quantization = enabled;
        self
    }
    
    pub fn on_disk(mut self, on_disk: bool) -> Self {
        self.on_disk = Some(on_disk);
        self
    }
}