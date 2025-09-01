# Build stage
FROM rustlang/rust:nightly-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    ca-certificates \
    build-essential \
    g++ \
    cmake \
    libclang-dev \
    clang \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy workspace manifests
COPY Cargo.toml Cargo.lock ./

# Copy all crate manifests
COPY crates/mcp-server/Cargo.toml ./crates/mcp-server/
COPY crates/mcp-protocol/Cargo.toml ./crates/mcp-protocol/
COPY crates/crawler/Cargo.toml ./crates/crawler/
COPY crates/embeddings/Cargo.toml ./crates/embeddings/
COPY crates/vector-store/Cargo.toml ./crates/vector-store/
COPY crates/graph-store/Cargo.toml ./crates/graph-store/
COPY crates/storage/Cargo.toml ./crates/storage/
COPY crates/search/Cargo.toml ./crates/search/

# Create dummy source files and required directories for dependency caching
RUN mkdir -p crates/mcp-server/src crates/mcp-protocol/src crates/crawler/src crates/embeddings/src \
    crates/vector-store/src crates/graph-store/src crates/storage/src crates/search/src \
    crates/vector-store/benches crates/crawler/examples crates/embeddings/examples crates/vector-store/examples && \
    echo "fn main() {}" > crates/mcp-server/src/main.rs && \
    touch crates/mcp-protocol/src/lib.rs && \
    touch crates/crawler/src/lib.rs && \
    touch crates/embeddings/src/lib.rs && \
    touch crates/vector-store/src/lib.rs && \
    touch crates/graph-store/src/lib.rs && \
    touch crates/storage/src/lib.rs && \
    touch crates/search/src/lib.rs && \
    echo "fn main() {}" > crates/vector-store/benches/quantization_bench.rs && \
    echo "fn main() {}" > crates/crawler/examples/crawler_example.rs && \
    echo "fn main() {}" > crates/embeddings/examples/embeddings_example.rs && \
    echo "fn main() {}" > crates/vector-store/examples/vector_store_example.rs

# Build dependencies only
RUN cargo build --release --bin mcp-server

# Remove dummy files but keep the target directory
RUN rm -rf crates/*/src crates/*/benches crates/*/examples

# Copy all source code and other required files
COPY crates/ ./crates/

# Build the actual application
RUN touch crates/mcp-server/src/main.rs && \
    cargo build --release --bin mcp-server

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1001 -s /bin/bash appuser

# Create necessary directories
RUN mkdir -p /data /models && \
    chown -R appuser:appuser /data /models

# Copy binary from builder
COPY --from=builder /app/target/release/mcp-server /usr/local/bin/mcp-server

# Switch to non-root user
USER appuser

# Set working directory
WORKDIR /app

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Set default environment variables
ENV RUST_LOG=info \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=8080 \
    MCP_STORAGE_PATH=/data

# Run the server
CMD ["mcp-server"]