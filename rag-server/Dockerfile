FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    pkg-config \
    libssl-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"

COPY Cargo.toml Cargo.lock* ./
RUN mkdir -p src static && \
    echo "pub fn dummy() {}" > src/lib.rs && \
    echo "fn main() {}" > src/main.rs && \
    touch static/index.html && \
    cargo build --release && \
    rm -rf src static

COPY src/ ./src/
COPY static/ ./static/

RUN touch src/lib.rs src/main.rs static/index.html && cargo build --release

FROM ubuntu:24.04 AS runner

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgomp1 \
    libssl3 \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN (userdel -r ubuntu 2>/dev/null || true) && \
    useradd -m -u 1000 -U appuser && \
    mkdir -p /app/data /app/models /app/uploads && \
    chown -R appuser:appuser /app

COPY --from=builder /app/target/release/rag-server /app/rag-server
COPY --from=builder --chown=appuser:appuser /app/static /app/static

USER appuser

ENV RAG_SERVER__HOST="0.0.0.0"
ENV RAG_SERVER__PORT="8080"
ENV RAG_LLM__ENDPOINT="http://host.docker.internal:11434/v1"

EXPOSE 8080

CMD ["/app/rag-server"]
