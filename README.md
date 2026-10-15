# Multimodal RAG Server in Rust

[![Rust](https://img.shields.io/badge/rust-1.80%2B-orange.svg)](https://www.rust-lang.org)
[![Axum](https://img.shields.io/badge/framework-axum-blue.svg)](https://github.com/tokio-rs/axum)
[![Inference](https://img.shields.io/badge/inference-ONNX%20Runtime-green.svg)](https://onnxruntime.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-ready, asynchronous Multimodal Vector Search and Retrieval-Augmented Generation (RAG) server engineered in Rust. The engine features cross-modal embedding generation (CLIP ViT-B/32), sub-millisecond approximate nearest neighbor (ANN) retrieval via an in-memory partitioned HNSW index, mathematical score calibration, hybrid lexical re-ranking, and strict citation verification for grounded LLM synthesis.

---

<img width="1132" height="553" alt="image" src="https://github.com/user-attachments/assets/5bc77c6e-f7f3-438b-80c6-e3be4a30dff6" />

<p align="center">
  <img src="assets/dashboard.png" alt="Control Panel Dashboard Overview" width="100%" />
</p>

---

## Architecture Overview

```text
  [ User Query: Text / Image ]
               │
               ▼
┌─────────────────────────────────┐
│        Axum Web Layer           │ ◄── Multi-Stage SSRF Filter / Safe Socket Pinning
└──────────────┬──────────────────┘
               ▼
┌─────────────────────────────────┐
│   ONNX Runtime CLIP Embedder    │ ◄── SessionPool (Rayon Parallel Preprocessing)
└──────────────┬──────────────────┘
               ▼
┌─────────────────────────────────┐
│ Partitioned Memory Vector Store │
│  ├── HNSW Text Graph Index      │
│  └── HNSW Image Graph Index     │
└──────────────┬──────────────────┘
               ▼
┌─────────────────────────────────┐
│ Score Calibration & Lexical BM25│ ◄── Mitigates CLIP Anisotropy Noise (~0.67)
└──────────────┬──────────────────┘
               ▼
┌─────────────────────────────────┐
│   Context Assembler & Budget    │
└──────────────┬──────────────────┘
               ▼
┌─────────────────────────────────┐
│     LLM Generation Client       │ ◄── OpenAI-compatible (Ollama / vLLM)
└──────────────┬──────────────────┘
               ▼
┌─────────────────────────────────┐
│   Strict Citation Validator     │ ──► [Source ID: ...] Grounded Response
└─────────────────────────────────┘
```

---

## Core Engineering & Design Highlights

### 1. Modality-Partitioned HNSW Vector Store
Standard vector databases suffer from topological degradation when cross-modal embeddings (such as text vs. visual representations) share the same graph layer due to varying norm densities. This engine isolates modalities into dedicated HNSW graphs (`hnsw_text` and `hnsw_image`), unified by an atomic O(1) bidirectional index map:
- **Lock-Free Read Operations:** Vector distance calculations utilize contiguous flat buffers without clone overhead.
- **Topological Edge Pruning:** Heuristic neighbor diversity selection maintains strict degree bounds (M0 = 32, M = 16, ef = 64).
- **Contiguous Memory Swapping:** Document removals use a tombstone-free `swap_remove` strategy to eliminate vector memory fragmentation.

### 2. Overcoming CLIP Anisotropy (Score Calibration)
Raw cosine similarities for CLIP text embeddings inherently cluster in a narrow cone, producing a high background noise floor around ~0.67, while cross-modal image-to-text similarities typically range between 0.14 and 0.28.
- **Piecewise Non-Linear Calibration:** Normalizes raw similarity scores into a uniform [0.0, 1.0] confidence space.
- **Hybrid Lexical Boosting:** Dense vector retrieval is enriched with BM25-style token overlap matching, preventing semantic false positives on entity-specific queries.

### 3. Production Security & Multi-Stage SSRF Defense
Accepting remote image URLs exposes systems to Server-Side Request Forgery (SSRF) and DNS Rebinding. This engine implements:
- **DNS Socket Pinning:** Explicit DNS resolution followed by socket connection pinning to the verified IP, preventing Time-of-Check to Time-of-Use (TOCTOU) attacks.
- **IP Range Filtering:** Automatically blocks loopback (`127.0.0.0/8`), Link-Local (`169.254.0.0/16`), private RFC 1918 subnets, IPv6-mapped IPv4, and Carrier-Grade NAT (`100.64.0.0/10`).
- **Port Allowlisting & Redirect Limits:** Restricts inbound requests strictly to ports 80, 443, 8080, and 8443, enforcing a 5-hop redirect maximum.

### 4. OOM-Safe Streaming Persistence
Database snapshots are written to disk using a chunked streaming serializer via atomic temporary files (`.tmp.<pid>.<counter>`) and filesystem `rename`. This eliminates memory blowups during large dumps and prevents snapshot corruption on sudden termination.

---

## Interactive Web Console

The server includes an embedded, zero-dependency control panel accessible via browser at:
```text
[http://127.0.0.1:8080/](http://127.0.0.1:8080/)
```

<!-- 📸 PLACEHOLDER 2: СЮДА ВСТАВЬ ВТОРОЙ СКРИНШОТ (ПОИСК С РАНЖИРОВАНИЕМ ИЛИ RAG-ОТВЕТ) -->
<!-- Перетащи второй скриншот сюда или вставь через Ctrl+V -->
<p align="center">
  <img src="assets/rag_demo.png" alt="RAG Generation and Real-time Search" width="100%" />
</p>

### Built-in Capabilities:
- **Document Indexing:** Upload raw image files, provide external image URLs, or input structured text payloads.
- **Similarity Explorer:** Live inspection of calibrated vector distance scores across text, image, and hybrid modalities.
- **RAG Synthesis Inspector:** Real-time generation debugger displaying context budget allocations and verified source citations.

---

## Quick Start

### 1. Prerequisites
- [Rust toolchain](https://rustup.rs/) (1.80 or newer)
- `curl` (for model downloading)
- An active LLM backend (e.g., [Ollama](https://ollama.ai) running `qwen2.5:7b` or `llama3.2`)

### 2. Setup Configuration
```bash
git clone [https://github.com/your-username/rag-server.git](https://github.com/your-username/rag-server.git)
cd rag-server

# Copy environment template
cp .env.example .env
```

### 3. Download ONNX Model Weights
Run the download script corresponding to your operating system to fetch **CLIP ViT-B/32** weights (`tokenizer.json`, `text_model.onnx`, `vision_model.onnx`):

- **Linux & macOS:**
  ```bash
  chmod +x scripts/download_models.sh
  ./scripts/download_models.sh
  ```
- **Windows (PowerShell):**
  ```powershell
  .\scripts\download_models.ps1
  ```

### 4. Build and Run
```bash
cargo run --release
```
The server will bind to `http://127.0.0.1:8080`. Open your browser and navigate to `http://127.0.0.1:8080/` to access the interactive web interface.

---

## Docker Deployment

To launch the isolated server container along with pre-configured volume mounts:

```bash
docker compose up --build -d
```
Snapshots persist in `./data`, and uploaded images are cached in `./uploads`.

---

## REST API Specification

### Index Document
```http
POST /api/v1/index
Content-Type: application/json

{
  "id": "doc_001",
  "title": "Rust Concurrency",
  "text": "Rust delivers fearless concurrency through ownership and type-enforced thread safety.",
  "image": {
    "type": "Url",
    "data": "[https://example.com/diagram.png](https://example.com/diagram.png)"
  }
}
```

### Cross-Modal Vector Search
```http
POST /api/v1/search
Content-Type: application/json

{
  "query": "thread safety and race conditions",
  "top_k": 5,
  "modality": "Text"
}
```

### Grounded RAG Generation
```http
POST /api/v1/rag
Content-Type: application/json

{
  "query": "How does Rust avoid data races at compile time?",
  "top_k": 3,
  "temperature": 0.2
}
```

### Document Management
- `GET /api/v1/documents` — List indexed documents and metadata summaries.
- `DELETE /api/v1/documents/:id` — Delete document by ID and restore HNSW graph connectivity.

---

## Code Quality & Verification

The codebase compiles with zero warnings under strict compiler and clippy restrictions:

```bash
# Verify formatting
cargo fmt --check

# Enforce zero-warning policy
cargo clippy --all-targets --all-features -- -D warnings

# Execute test suite
cargo test
```

---

## License

This project is licensed under the [MIT License](LICENSE).
