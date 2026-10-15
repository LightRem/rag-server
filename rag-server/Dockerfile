FROM rust:1-bookworm AS builder
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
RUN mkdir -p src static && \
    echo "pub fn dummy() {}" > src/lib.rs && \
    echo "fn main() {}" > src/main.rs && \
    touch static/index.html && \
    cargo build --release && \
    rm -rf src static
COPY src/ ./src/
COPY static/ ./static/
RUN touch src/lib.rs src/main.rs static/index.html && cargo build --release
FROM debian:bookworm-slim AS runner
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 -U appuser && \
    mkdir -p /app/data /app/models /app/uploads && \
    chown -R appuser:appuser /app
USER appuser
COPY --from=builder /app/target/release/rag-server /app/rag-server
ENV RAG_SERVER__HOST="0.0.0.0"
ENV RAG_SERVER__PORT="8080"
ENV RAG_LLM__ENDPOINT="http://host.docker.internal:11434/v1"
EXPOSE 8080
CMD ["/app/rag-server"]
