# CareMind · Technical Report

Last updated: 2025-12-28  
Scope: This report describes the current state of this workspace (Streamlit MVP + RAG pipeline + ingestion tooling). It is intended for engineers and maintainers.

---

## 1) Executive Summary

CareMind is a Streamlit-based clinical decision support (CDSS) prototype that answers clinician-style questions using retrieval-augmented generation (RAG).

- **Primary capability**: retrieve relevant guideline excerpts (ChromaDB vector store) and optionally enrich with **structured drug information** (SQLite), then produce a **structured, citation-backed** draft recommendation.
- **Bilingual UX**: Chinese/English prompts and UI paths are supported.
- **Deployment targets**: local (optionally GPU for embedding during ingestion) and Streamlit Community Cloud (CPU-only; includes SQLite compatibility shims).

Non-goals: It is not a regulated medical device, not an EHR integration, and does not ingest PHI/PII by design.

---

## 2) Repository Layout (Current Workspace)

Top-level structure (high-signal modules):

- [app.py](../../app.py) — Streamlit UI and orchestration.
- [rag/retriever.py](../../rag/retriever.py) — data access layer: Chroma search + optional SQLite drug lookup + cloud compatibility.
- [rag/pipeline.py](../../rag/pipeline.py) — answer orchestration: retrieve → (optional) OpenAI call → draft fallback.
- [rag/prompt.py](../../rag/prompt.py), [rag/prompt_cn.py](../../rag/prompt_cn.py) — strict output templates (Advice + Evidence List).
- [ingest/parse_docs.py](../../ingest/parse_docs.py) — PDF parsing + metadata extraction → JSONL chunks.
- [ingest/build_vectors.py](../../ingest/build_vectors.py) — JSONL → embeddings → persistent Chroma collection (OOM-safe).
- [ingest/load_drugs.py](../../ingest/load_drugs.py) — Excel → SQLite ingest/upsert (optional FTS5).

Data and persistence:

- [data/](../../data/) — guideline PDFs and intermediate parsed corpus.
- [chroma_store/](../../chroma_store/) — Chroma persistent store directory (demo / local).
- [db/](../../db/) — SQLite drug database (if present).

Note: Some older documents refer to `ingest/create_db.py`; in this workspace the current indexing entrypoint is `ingest/build_vectors.py`.

---

## 3) High-Level Architecture

### 3.1 Conceptual Components

1) **Ingestion layer** (offline, repeatable)
- Parse guideline PDFs → chunked JSONL with metadata
- Embed chunk text → write to ChromaDB collection
- Optionally ingest drug Excel → SQLite database

2) **Runtime layer** (interactive)
- Streamlit UI accepts query + optional drug
- Retriever performs vector search and returns evidence snippets
- Pipeline composes prompts and generates advice (OpenAI if configured; otherwise draft fallback)
- UI renders results and exports Markdown

### 3.2 Data Flow (Runtime)

1. User submits question (and optionally a drug name).
2. `rag.retriever.search_guidelines()` performs dense vector search against Chroma.
3. `rag.retriever.search_drug_structured()` optionally queries SQLite for drug info.
4. `rag.pipeline.answer()`:
   - builds a strict prompt (English: `rag.prompt`; Chinese: `rag.prompt_cn`)
   - if `OPENAI_API_KEY` is available and there are guideline hits, calls OpenAI
   - otherwise generates a deterministic “draft” answer and includes an Evidence List
5. [app.py](../../app.py) splits the model output into “Advice” vs “Evidence List” tabs and renders evidence anchors.

---

## 4) Storage and Data Model

### 4.1 ChromaDB (Vector Store)

- Persistence: directory configured by `CHROMA_PERSIST_DIR` (default is resolved to an absolute path rooted at the repo).
- Collection name: `CHROMA_COLLECTION` (defaults differ across docs; see §6).
- Stored objects:
  - `documents`: chunk text content
  - `embeddings`: SentenceTransformer embeddings
  - `metadatas`: JSON-safe scalar metadata (source filename, title/section, year, doc type, etc.)

Compatibility note: `rag/retriever.py` includes a “sysdb pre-migration” path for older stores missing configuration fields, and includes logic to use `pysqlite3` when the host’s stdlib SQLite is too old.

### 4.2 SQLite (Structured Drug DB)

- Path: `DRUG_DB_PATH` (defaults to `./db/drugs.sqlite` resolved to repo root).
- Ingestion: [ingest/load_drugs.py](../../ingest/load_drugs.py) reads an Excel file and upserts into a `drugs` table.
- Optional full-text search: creates an FTS5 virtual table and sync triggers when run with `--with-fts`.

---

## 5) Ingestion Pipeline (Offline)

### 5.1 Parse PDFs → JSONL

Primary script: [ingest/parse_docs.py](../../ingest/parse_docs.py)

- Reads PDFs using `pdfplumber`
- Extracts rich bibliographic metadata (authors, DOI, journal name, publish date, etc.) when present
- Classifies document type heuristically (guideline / interpretation / consensus / evidence summary)
- Chunks text using section-title heuristics
- Outputs JSONL with schema resembling:

```json
{"content": "…", "meta": {"source_filename": "…", "doc_title": "…", "section_title": "…", "year": "…", "doc_type": "…", "chunk_id": "…"}}
```

### 5.2 JSONL → Chroma Embeddings

Primary script: [ingest/build_vectors.py](../../ingest/build_vectors.py)

Key properties:

- **OOM-safe**: dynamic batch-size backoff on CUDA out-of-memory.
- **Idempotent-ish**: stable IDs derived from source + chunk id + content hash; uses upsert semantics where available.
- **Metadata-robust**: sanitizes metadata into Chroma-supported scalar types.
- **Device-aware**: uses CUDA if available; optional fp16 autocast.

Environment variables used by this script include:
- `CHROMA_PERSIST_DIR`, `CHROMA_COLLECTION`
- `CAREMIND_DATA` (defaults to `data/guidelines.parsed.jsonl`)
- `EMBEDDING_MODEL` (defaults to `BAAI/bge-large-zh-v1.5` in this workspace)
- `EMBED_BATCH_SIZE`, `EMBED_FP16`, `EMBED_MAX_LEN`, `OOM_CPU_FALLBACK`

### 5.3 Excel → SQLite Drug DB (Optional)

Primary script: [ingest/load_drugs.py](../../ingest/load_drugs.py)

- Normalizes column names across English/Chinese headers.
- Upserts by `drug_name` (unique).
- (Optional) creates FTS5 index for better search.

---

## 6) Configuration & Environment Variables

### 6.1 Runtime Configuration (App + Retriever)

Common environment variables:

- `CHROMA_PERSIST_DIR`
  - Points to the Chroma persistence directory, e.g. `./chroma_store`.
- `CHROMA_COLLECTION`
  - Collection name to open.
  - In this workspace, `rag/retriever.py` defaults to `guideline_chunks_v2`.
- `EMBEDDING_MODEL`
  - Embedding model identifier for SentenceTransformers; used by the retriever embedding function and ingestion.
- `DRUG_DB_PATH`
  - SQLite file path for structured drug lookup.

### 6.2 Pipeline / LLM Configuration

- `OPENAI_API_KEY`
  - If present, `rag/pipeline.py` prefers OpenAI for the “final” advice.
- `CAREMIND_OPENAI_MODEL`
  - Overrides model name (default: `gpt-4o-mini`).
- `CAREMIND_DEMO`
  - Demo-mode switch (default in the pipeline is demo ON in Cloud contexts).
- `CAREMIND_MAX_K`
  - Caps user top-k (default: 8).

Practical note: `rag/pipeline.answer()` will still retrieve evidence even if OpenAI is not configured; the output becomes a “draft/demo” answer.

---

## 7) Streamlit Application (UI + UX Notes)

Primary file: [app.py](../../app.py)

Notable behaviors:

- **SQLite bootstrap**: attempts to alias `pysqlite3` as `sqlite3` early for Cloud compatibility.
- **Evidence linking**: converts citations like `[3]` into anchors that jump to the corresponding evidence snippet.
- **Advice/Evidence separation**: splits model output into Advice vs Evidence List tabs using the “Evidence List / 证据清单” header.
- **Diagnostics panel**: surfaces effective config and Chroma health checks without creating multiple Chroma clients.
- **Session history**: supports reuse of prior prompts.

---

## 8) Deployment

### 8.1 Local

Typical flow:

1) Create environment and install deps:

```bash
conda create -n caremind python=3.10 -y
conda activate caremind
pip install -r requirements.txt
```

2) Build ingestion artifacts (if not already present):

```bash
python ingest/parse_docs.py
python ingest/build_vectors.py
```

3) Run UI:

```bash
streamlit run app.py
```

### 8.2 Streamlit Community Cloud

See [docs/guides/deployment_cloud.md](../guides/deployment_cloud.md).

Key points:

- Cloud is CPU-only; prefer a small demo `chroma_store/`.
- Ensure `CHROMA_PERSIST_DIR` and `CHROMA_COLLECTION` match what is shipped in the deployed branch.
- `pysqlite3-binary` is included to satisfy Chroma/SQLite version requirements.

---

## 9) Observability & Operations

- Retriever logs are timestamped and designed to be shown in Streamlit logs.
- Health-check helpers expose:
  - “what collection names exist?”
  - “how many records are in the primary collection?”
- Ingestion scripts print warnings for invalid JSONL lines and handle OOM via batch backoff.

---

## 10) Security, Privacy, and Compliance

- Intended for public guideline corpora and curated drug metadata; not designed for PHI/PII ingestion.
- Secrets management:
  - Local: `.env` is supported (project-root only).
  - Cloud: Streamlit Secrets recommended.
- Output includes an explicit clinical disclaimer line (enforced by prompt templates).

---

## 11) Performance Considerations

- Embedding model choice strongly affects ingestion time and runtime latency.
  - `BAAI/bge-large-zh-v1.5` improves recall/quality but is heavier.
  - smaller BGE variants may be preferable on CPU-only deployments.
- Retrieval top-k (`k`) trades recall for token/latency cost.
- OOM risk during ingestion is mitigated with dynamic batch-size backoff and optional CPU fallback.

---

## 12) Known Constraints / Caveats

- Collection naming and paths can drift across branches/environments; deployment relies on consistent `CHROMA_PERSIST_DIR` + `CHROMA_COLLECTION`.
- Some documentation in the repo reflects earlier ingestion script names; use the scripts under [ingest/](../../ingest/) in this workspace as the source of truth.
- Drug DB coverage is limited to what is ingested into SQLite; it is optional and the app should degrade gracefully if absent.

---

## 13) Extension Points (Engineering Roadmap)

- Add reranking (cross-encoder) for improved precision at small k.
- Add hybrid retrieval (BM25 + dense) for better recall.
- Add more robust chunking (semantic chunkers) and citation page anchoring.
- Expand drug DB schema and standardize bilingual naming.
- Add automated validation scripts for ingestion QA (schema checks + spot queries).
