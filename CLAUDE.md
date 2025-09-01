# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Build & Test
```bash
# Build all crates
cargo build --release

# Run all tests
cargo test --workspace

# Run specific crate tests  
cargo test --package mcp-server
cargo test --package crawler

# Unit tests only
cargo test --lib

# Integration tests only
cargo test --test '*'

# Run benchmarks
cargo bench

# Run with property-based testing
cargo test --features proptest
```

### Running the Server
```bash
# Run MCP server
cargo run --bin mcp-server

# Run with debug logging
RUST_LOG=debug cargo run --bin mcp-server

# Run examples
cargo run --example crawler_example --package crawler
cargo run --example embeddings_example --package embeddings
cargo run --example vector_store_example --package vector-store
```

## Architecture Overview

This is a high-performance Rust MCP (Model Context Protocol) server for web crawling and RAG capabilities, built as a workspace with specialized crates:

### Core Architecture
- **Zero-Copy Protocol**: Uses rkyv serialization for zero-copy data transfer between components
- **Hybrid Storage**: RocksDB for metadata, DuckDB for analytics, memory-mapped files for content
- **Async-First**: Built on Tokio with Axum web framework for concurrent operations
- **Type-Safe MCP**: Compile-time guarantees using type-state pattern

### Crate Responsibilities
- `mcp-server`: Axum-based HTTP server with SSE/WebSocket endpoints for MCP protocol
- `mcp-protocol`: Zero-copy protocol types and serialization using rkyv
- `crawler`: Memory-efficient web crawler with rate limiting and robots.txt support
- `embeddings`: Local embeddings using Candle framework (GPU-accelerated)  
- `vector-store`: Qdrant integration with binary quantization (32x memory reduction)
- `graph-store`: Memgraph knowledge graph integration
- `storage`: Hybrid storage layer combining RocksDB, DuckDB, and memory-mapped files
- `search`: SIMD-accelerated vector similarity search

### MCP Tools Implemented
1. `crawl_single_page` - Single page crawling
2. `smart_crawl_url` - Multi-page intelligent crawling  
3. `get_available_sources` - List crawled domains
4. `perform_rag_query` - Semantic search with filtering
5. `search_code_examples` - Code-specific search
6. `parse_github_repository` - Parse repos to knowledge graph
7. `check_ai_script_hallucinations` - Validate AI-generated code
8. `query_knowledge_graph` - Explore knowledge graph

### Performance Features
- Binary quantization for 32x memory reduction
- SIMD operations for hardware-accelerated search
- Memory-mapped files for efficient large file handling
- Zero-copy deserialization with rkyv
- Parallel processing with rayon

## Configuration

Server uses environment variables:
```env
MCP_HOST=0.0.0.0
MCP_PORT=8080
MCP_STORAGE_PATH=./data
QDRANT_URL=http://localhost:6333
MEMGRAPH_URL=bolt://localhost:7687
RUST_LOG=debug
```

## Development Status

- Phase 1 (Complete): Zero-copy foundation, memory-mapped storage, Axum server
- Phase 2 (In Progress): Vector store with binary quantization
- Future: Local embeddings, full crawler, hybrid storage, SIMD search, production hardening