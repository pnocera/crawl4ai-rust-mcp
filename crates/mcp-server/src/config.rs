use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub workers: usize,
    pub max_connections: usize,
    pub storage_path: PathBuf,
    pub cors_origins: Vec<String>,
    pub sse_keepalive_interval: u64,
    pub request_timeout: u64,
    pub max_request_size: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8080,
            workers: num_cpus::get(),
            max_connections: 10000,
            storage_path: PathBuf::from("./data"),
            cors_origins: vec!["*".to_string()],
            sse_keepalive_interval: 30,
            request_timeout: 300,
            max_request_size: 10 * 1024 * 1024, // 10MB
        }
    }
}

impl ServerConfig {
    pub fn from_env() -> Self {
        let mut config = Self::default();
        
        if let Ok(host) = std::env::var("MCP_HOST") {
            config.host = host;
        }
        
        if let Ok(port) = std::env::var("MCP_PORT") {
            if let Ok(port) = port.parse() {
                config.port = port;
            }
        }
        
        if let Ok(workers) = std::env::var("MCP_WORKERS") {
            if let Ok(workers) = workers.parse() {
                config.workers = workers;
            }
        }
        
        if let Ok(path) = std::env::var("MCP_STORAGE_PATH") {
            config.storage_path = PathBuf::from(path);
        }
        
        config
    }
    
    pub fn socket_addr(&self) -> SocketAddr {
        format!("{}:{}", self.host, self.port)
            .parse()
            .expect("Invalid socket address")
    }
}

