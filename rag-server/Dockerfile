# ==========================================
# 1. Builder Stage (Ubuntu 24.04 + Rust)
# ==========================================
FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Установка компилятора C/C++, pkg-config и заголовков OpenSSL для линковки
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    pkg-config \
    libssl-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Установка актуального стабильного компилятора Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"

# Кэширование сборки зависимостей (dummy-билд)
COPY Cargo.toml Cargo.lock ./
RUN mkdir -p src static && \
    echo "pub fn dummy() {}" > src/lib.rs && \
    echo "fn main() {}" > src/main.rs && \
    touch static/index.html && \
    cargo build --release && \
    rm -rf src static

# Копирование исходного кода и статики
COPY src/ ./src/
COPY static/ ./static/

# Финальная компиляция релизного бинарника
RUN touch src/lib.rs src/main.rs static/index.html && cargo build --release

# ==========================================
# 2. Runner Stage (Ubuntu 24.04 Runtime)
# ==========================================
FROM ubuntu:24.04 AS runner

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Рантайм-зависимости (OpenSSL 3, OpenMP для ONNX, curl и сертификаты)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgomp1 \
    libssl3 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Удаляем встроенного пользователя ubuntu (UID 1000) и создаем appuser с UID 1000
RUN (userdel -r ubuntu 2>/dev/null || true) && \
    useradd -m -u 1000 -U appuser && \
    mkdir -p /app/data /app/models /app/uploads && \
    chown -R appuser:appuser /app

# Копирование скомпилированного сервиса и статики веб-панели
COPY --from=builder /app/target/release/rag-server /app/rag-server
COPY --from=builder --chown=appuser:appuser /app/static /app/static

USER appuser

# Переменные конфигурации сервиса
ENV RAG_SERVER__HOST="0.0.0.0"
ENV RAG_SERVER__PORT="8080"
ENV RAG_LLM__ENDPOINT="http://host.docker.internal:11434/v1"

EXPOSE 8080

CMD ["/app/rag-server"]
