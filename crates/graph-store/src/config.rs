use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemgraphConfig {
    pub url: String,
    pub username: Option<String>,
    pub password: Option<String>,
    pub database: Option<String>,
}

impl Default for MemgraphConfig {
    fn default() -> Self {
        Self {
            url: "bolt://localhost:7687".to_string(),
            username: None,
            password: None,
            database: None,
        }
    }
}