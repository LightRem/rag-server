# Multimodal RAG Server in Rust

[![Rust](https://img.shields.io/badge/rust-1.80%2B-orange.svg)](https://www.rust-lang.org)
[![Axum](https://img.shields.io/badge/framework-axum-blue.svg)](https://github.com/tokio-rs/axum)
[![Inference](https://img.shields.io/badge/inference-ONNX%20Runtime-green.svg)](https://onnxruntime.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-ready, asynchronous Multimodal Vector Search and Retrieval-Augmented Generation (RAG) server engineered in Rust. The engine features cross-modal embedding generation (CLIP ViT-B/32), sub-millisecond approximate nearest neighbor (ANN) retrieval via an in-memory partitioned HNSW index, mathematical score calibration, hybrid lexical re-ranking, and strict citation verification for grounded LLM synthesis.

---

<p align="center">
  <img width="745" height="452" alt="image" src="https://github.com/user-attachments/assets/9e296f48-0810-424b-8da3-808ebb81cd4a" />
</p>

---

## Interactive Web Console & Usage Workflow

The server comes with an embedded, zero-dependency web control panel accessible directly in your browser at:
```text
[http://127.0.0.1:8080/](http://127.0.0.1:8080/)
```

The interface is structured into four sequential modules covering the entire multimodal lifecycle:

### 1. Document Indexing (`POST /api/v1/index`)
Populate the multimodal knowledge base with heterogeneous content:
- **Identifier (`ID`)**: Unique string key for the document (e.g., `doc_cat_habits`, `doc_test_1`).
- **Title & Text Payload**: Human-readable title and descriptive body text.
- **Image Input (Optional, up to 20MB)**: Attach local images in JPEG, PNG, WebP, or GIF formats.
- **Automatic Modality Detection**: 
  - Providing only text tags the record as `TEXT`.
  - Providing only an image tags it as `IMAGE`.
  - Providing both registers a unified `MULTIMODAL` vector point across both HNSW sub-graphs.

### 2. Vector Similarity Search (`POST /api/v1/search`)
Query the vector space across modalities with calibrated scoring:
- **Query Inputs**: Search using plain text query strings, reference image uploads, or both simultaneously.
- **Calibrated Scoring**: Outputs ranked results with calibrated similarity metrics ([0.0, 1.0] spectrum), mitigating CLIP's raw anisotropic noise floor.
- **Modality Badges**: Identifies which modality triggered the match (`Text`, `Image`, or fused cross-modal match).

<p align="center">
  <img src="https://github.com/user-attachments/assets/5705f212-11a0-4a97-adc0-5d126712dd08" alt="Vector Similarity Search" width="100%" />
</p>

### 3. RAG Query Pipeline (`POST /api/v1/rag`)
Ask natural-language questions grounded strictly against your indexed knowledge:
- **Question Formulation**: Submit an explicit user question.
- **Visual Context (Optional)**: Provide a query image to guide cross-modal context retrieval.
- **Strict Verification & Citations**: The pipeline fetches top-$k$ nearest context passages, injects budget-capped evidence into the prompt, and queries the configured LLM. Responses enforce inline source citations (`[Source ID: <id>]`), preventing hallucinations.

<p align="center">
  <img src="https://github.com/user-attachments/assets/df28dc9d-0e62-4aea-a4df-2c2bcfc86dd4" alt="RAG Query Pipeline" width="100%" />
</p>

### 4. Knowledge Base Explorer (`GET /api/v1/documents`)
Inspect and manage the active state of the in-memory vector store:
- **Live Inventory**: View total document counts and inspect stored records with explicit modality tags (`TEXT`, `IMAGE`, `MULTIMODAL`).
- **Real-Time Deletion**: The `Delete` button removes points using an atomic `swap_remove` strategy and reconstructs HNSW graph edge topology without memory leaks or index downtime.
- **Refresh List**: Poll the current database state at any time.

<p align="center">
  <img src="https://github.com/user-attachments/assets/5e7d72b2-0faf-4857-b2d0-508bf3825189" alt="Knowledge Base Explorer" width="100%" />
</p>

---

## Technical Highlights & Architecture

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

### 1. Modality-Partitioned HNSW Vector Store
Standard vector databases suffer from topological degradation when cross-modal embeddings share the same graph layer due to varying norm densities. This engine isolates modalities into dedicated HNSW graphs (`hnsw_text` and `hnsw_image`), unified by an atomic O(1) bidirectional index map:
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

## Quick Start

### 1. Prerequisites
- [Rust toolchain](https://rustup.rs/) (1.80 or newer)
- [Just](https://github.com/casey/just) command runner (`cargo install just`)
- `curl` (for model downloading)
- An active LLM backend (e.g., [Ollama](https://ollama.ai) running `qwen2.5:7b` or `llama3.2`)

### 2. Setup Configuration
```bash
git clone [https://github.com/LightRem/rag-server.git](https://github.com/LightRem/rag-server.git)
cd rag-server

# Copy environment template
cp .env.example .env
```

### 3. Download ONNX Model Weights
Download the pre-exported **CLIP ViT-B/32** weights (`tokenizer.json`, `text_model.onnx`, `vision_model.onnx`) using the cross-platform task runner:

```bash
just download-models
```

> **⚠️ Windows Troubleshooting (Script Execution Policy):**  
> If `just download-models` fails on Windows with an error stating that *running scripts is disabled on this system*, Windows has restricted `.ps1` execution policies. You can enable user-level script execution in PowerShell **at your own discretion and risk** by running:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```
> Once executed, re-run `just download-models`.

### 4. Build and Run
```bash
cargo run --release
```
The server will start at `http://127.0.0.1:8080`. Open your browser and navigate to `http://127.0.0.1:8080/` to access the interactive web interface.

---

## Docker Deployment

To launch the isolated server container along with pre-configured volume mounts:

```bash
docker compose up --build -d
```
Snapshots persist in `./data`, and uploaded images are cached in `./uploads`.

---

## REST API Specification

### 1. Index Document (`POST /api/v1/index`)
```bash
curl -X POST [http://127.0.0.1:8080/api/v1/index](http://127.0.0.1:8080/api/v1/index) \
  -H "Content-Type: application/json" \
  -d '{
    "id": "doc_001",
    "title": "Rust Concurrency",
    "text": "Rust delivers fearless concurrency through ownership and type-enforced thread safety."
  }'
```

### 2. Cross-Modal Vector Search (`POST /api/v1/search`)
```bash
curl -X POST [http://127.0.0.1:8080/api/v1/search](http://127.0.0.1:8080/api/v1/search) \
  -H "Content-Type: application/json" \
  -d '{
    "query": "thread safety and race conditions",
    "top_k": 5
  }'
```

### 3. Grounded RAG Generation (`POST /api/v1/rag`)
```bash
curl -X POST [http://127.0.0.1:8080/api/v1/rag](http://127.0.0.1:8080/api/v1/rag) \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How does Rust avoid data races at compile time?",
    "top_k": 3,
    "temperature": 0.2
  }'
```

### 4. Document Management
- **List indexed documents:**
  ```bash
  curl -X GET [http://127.0.0.1:8080/api/v1/documents](http://127.0.0.1:8080/api/v1/documents)
  ```
- **Delete document by ID:**
  ```bash
  curl -X DELETE [http://127.0.0.1:8080/api/v1/documents/doc_001](http://127.0.0.1:8080/api/v1/documents/doc_001)
  ```

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
