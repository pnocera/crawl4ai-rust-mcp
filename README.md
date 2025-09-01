# crawl4ai-rust-mcp

A high-performance Rust MCP (Model Context Protocol) server for web crawling and RAG (Retrieval-Augmented Generation) capabilities. Built with zero-copy architecture, SIMD acceleration, and memory-efficient storage.

## Features

- 🚀 **Zero-Copy Protocol**: Efficient data transfer using rkyv serialization
- 🔍 **Intelligent Web Crawling**: Polite crawling with rate limiting and robots.txt support
- 🧠 **Local Embeddings**: GPU-accelerated text embeddings with Candle framework
- 💾 **32x Memory Reduction**: Binary quantization for vector storage
- ⚡ **SIMD-Accelerated Search**: Hardware-optimized vector similarity search
- 🗄️ **Hybrid Storage**: RocksDB + DuckDB + memory-mapped files
- 🔗 **Knowledge Graphs**: Memgraph integration for relationship tracking
- 🌐 **MCP Protocol**: Full compliance with Model Context Protocol

## Architecture

```
crawl4ai-rust-mcp/
├── mcp-server/         # Axum HTTP server with SSE/WebSocket support
├── mcp-protocol/       # MCP protocol types and serialization
├── crawler/            # Web crawler with JavaScript support
├── embeddings/         # Local embeddings generation
├── vector-store/       # Qdrant integration with binary quantization
├── graph-store/        # Knowledge graph storage
├── storage/            # Hybrid storage layer
└── search/             # SIMD-accelerated similarity search
```

## Quick Start

### Prerequisites

- Rust 1.70+
- Docker (for Qdrant and Memgraph)
- CUDA/Metal (optional, for GPU acceleration)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/crawl4ai-rust-mcp.git
cd crawl4ai-rust-mcp
```

2. Start required services:
```bash
# Qdrant vector database
docker run -p 6333:6333 qdrant/qdrant

# Memgraph (optional, for knowledge graphs)
docker run -p 7687:7687 memgraph/memgraph
```

3. Build the project:
```bash
cargo build --release
```

4. Run the server:
```bash
cargo run --bin mcp-server
```

## Configuration

Set environment variables to configure the server:

```env
MCP_HOST=0.0.0.0
MCP_PORT=8080
MCP_STORAGE_PATH=./data
QDRANT_URL=http://localhost:6333
MEMGRAPH_URL=bolt://localhost:7687
RUST_LOG=debug
```

## MCP Tools

The server implements 8 MCP tools:

### 1. `crawl_single_page`
Crawl a single web page and extract content.

### 2. `smart_crawl_url`
Intelligently crawl multiple pages from a starting URL.

### 3. `get_available_sources`
List all crawled domains and their metadata.

### 4. `perform_rag_query`
Perform semantic search across indexed content.

### 5. `search_code_examples`
Search specifically for code snippets and examples.

### 6. `parse_github_repository`
Parse GitHub repositories into a knowledge graph.

### 7. `check_ai_script_hallucinations`
Validate AI-generated code against known patterns.

### 8. `query_knowledge_graph`
Query the knowledge graph for relationships and insights.

## API Endpoints

- `POST /mcp` - Execute MCP tools
- `GET /mcp/stream` - Server-sent events for streaming responses
- `POST /api/crawl` - Direct crawling endpoint
- `POST /api/search` - Direct search endpoint
- `WS /ws` - WebSocket for bidirectional communication

## Development

### Building

```bash
# Build all crates
cargo build --release

# Build specific crate
cargo build --package mcp-server
```

### Testing

```bash
# Run all tests
cargo test --workspace

# Run specific crate tests
cargo test --package crawler

# Run with property-based testing
cargo test --features proptest
```

### Running Examples

```bash
# Crawler example
cargo run --example crawler_example --package crawler

# Embeddings example
cargo run --example embeddings_example --package embeddings

# Vector store example
cargo run --example vector_store_example --package vector-store
```

## Performance

- **Binary Quantization**: 32x memory reduction for vector storage
- **SIMD Operations**: Hardware-accelerated similarity search
- **Zero-Copy Design**: Minimal memory allocations during data transfer
- **Async Architecture**: Handles thousands of concurrent operations
- **Memory-Mapped Files**: Efficient handling of large documents

## Use Cases

- 🤖 **AI Assistants**: Build context-aware AI applications
- 📚 **Documentation Search**: Index and search technical documentation
- 🔬 **Research Tools**: Crawl and analyze academic papers
- 💻 **Code Intelligence**: Parse and understand code repositories
- 🔍 **Enterprise Search**: Build internal knowledge bases

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests to our repository.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [Axum](https://github.com/tokio-rs/axum) web framework
- Vector storage powered by [Qdrant](https://qdrant.tech/)
- Embeddings via [Candle](https://github.com/huggingface/candle)
- Knowledge graphs with [Memgraph](https://memgraph.com/)