# Build stage
FROM rust:1.75-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy manifests
COPY Cargo.toml Cargo.lock ./

# Copy workspace members
COPY mcp-server/Cargo.toml ./mcp-server/
COPY mcp-protocol/Cargo.toml ./mcp-protocol/
COPY crawler/Cargo.toml ./crawler/
COPY embeddings/Cargo.toml ./embeddings/
COPY vector-store/Cargo.toml ./vector-store/
COPY graph-store/Cargo.toml ./graph-store/
COPY storage/Cargo.toml ./storage/
COPY search/Cargo.toml ./search/

# Create dummy main.rs files to cache dependencies
RUN mkdir -p mcp-server/src mcp-protocol/src crawler/src embeddings/src \
    vector-store/src graph-store/src storage/src search/src && \
    echo "fn main() {}" > mcp-server/src/main.rs && \
    touch mcp-protocol/src/lib.rs && \
    touch crawler/src/lib.rs && \
    touch embeddings/src/lib.rs && \
    touch vector-store/src/lib.rs && \
    touch graph-store/src/lib.rs && \
    touch storage/src/lib.rs && \
    touch search/src/lib.rs

# Build dependencies
RUN cargo build --release --bin mcp-server

# Remove dummy files
RUN rm -rf mcp-server/src mcp-protocol/src crawler/src embeddings/src \
    vector-store/src graph-store/src storage/src search/src

# Copy actual source code
COPY mcp-server/src ./mcp-server/src
COPY mcp-protocol/src ./mcp-protocol/src
COPY crawler/src ./crawler/src
COPY embeddings/src ./embeddings/src
COPY vector-store/src ./vector-store/src
COPY graph-store/src ./graph-store/src
COPY storage/src ./storage/src
COPY search/src ./search/src

# Build the actual application
RUN touch mcp-server/src/main.rs && \
    cargo build --release --bin mcp-server

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
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