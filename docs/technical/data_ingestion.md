# CareMind-Streamlit · Data Ingestion Guide

> This guide explains how to turn raw guideline PDFs into a searchable ChromaDB collection that the app can retrieve from. It covers local (Windows + WSL/Ubuntu) and Streamlit Cloud workflows, health checks, and troubleshooting.

---

## 0) Glossary
- **Parsed corpus**: `data/guidelines.parsed.jsonl` — one JSON object per chunk (fields like `id`, `title`, `section`, `text`, `source`, etc.).
- **Vector store**: Chroma persistent directory (e.g., `chroma_store/`), with a **collection** (e.g., `guideline_chunks_1024_v2`).
- **Embedding model**: typically `BAAI/bge-large-zh-v1.5` (or a smaller variant for low‑VRAM).

---

## 1) Prerequisites
- Python 3.10+ in a clean environment (Conda/venv).
- CUDA-enabled PyTorch (if using GPU locally) and matching CUDA/cuDNN drivers.
- The repo structure contains:
  - `ingest/parse_pdfs.py` (or equivalent) — *parsing stage* (PDF → JSONL).
  - `ingest/create_db.py` — *indexing stage* (JSONL → ChromaDB).
  - `rag/retriever.py`, `rag/pipeline.py` — retrieval & orchestration.
- **Environment variables** set (local run):

```bash
# adjust paths as needed
export CHROMA_PERSIST_DIR=./chroma_store
export CHROMA_COLLECTION=guideline_chunks_1024_v2
export EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5
```

> **Tip**: Put these in a shell profile (e.g., `~/.bashrc`) or a `.env` file if your tooling loads it. On Windows PowerShell, use `$Env:CHROMA_PERSIST_DIR = "./chroma_store"` style.

---

## 2) Pipeline Overview

### Stage A — Parse PDFs → JSONL
Use the provided parsing script (or your own) to normalize and chunk guidelines.

```bash
python ingest/parse_pdfs.py \
  --in-dir data/pdfs \
  --out data/guidelines.parsed.jsonl \
  --max-chunk-tokens 1024 \
  --overlap 128
```

**Outputs**
- `data/guidelines.parsed.jsonl`
- Logging summary (how many pages, chunks, skipped pages/OCR fallbacks, etc.)

**Quality checks**
- `wc -l data/guidelines.parsed.jsonl` (number of chunks lines)
- `head -n 3 data/guidelines.parsed.jsonl` (inspect schema)

### Stage B — Build / Update Chroma Collection
Index the JSONL into Chroma with your chosen embedding model.

```bash
python ingest/create_db.py \
  --in data/guidelines.parsed.jsonl \
  --persist-dir "$CHROMA_PERSIST_DIR" \
  --collection "$CHROMA_COLLECTION" \
  --embed-model "$EMBEDDING_MODEL"
```

**Optional flags (if your `create_db.py` supports them):**
- `--batch-size 64` — tune for VRAM/CPU RAM.
- `--max-retries 3` — retry on transient failures.
- `--quarantine` — auto-move conflicting/old stores before clean rebuild.

> If your version doesn’t expose a flag (e.g., `--retry`, `--resume`, `--upsert`), omit it. Some behaviors (resume/upsert) may be automatic in newer scripts.

---

## 3) Running on Windows + WSL/Ubuntu

### Quick start (fresh env)
```bash
# 1) Create & activate env
conda create -n caremind python=3.10 -y && conda activate caremind
# or: python -m venv .venv && source .venv/bin/activate

# 2) Install deps
pip install -r requirements.txt

# 3) Export env vars
export CHROMA_PERSIST_DIR=./chroma_store
export CHROMA_COLLECTION=guideline_chunks_1024_v2
export EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5

# 4) Parse → JSONL
python ingest/parse_pdfs.py --in-dir data/pdfs --out data/guidelines.parsed.jsonl

# 5) Index → Chroma
python ingest/create_db.py --in data/guidelines.parsed.jsonl \
  --persist-dir "$CHROMA_PERSIST_DIR" --collection "$CHROMA_COLLECTION" \
  --embed-model "$EMBEDDING_MODEL"
```

### Long‑running sessions (tmux)
```bash
tmux new -s ingest || true
# run your commands…
# detach: Ctrl-b then d
# reattach:
tmux attach -t ingest
```

> If WSL becomes unresponsive during heavy ingestion, check system monitors (Task Manager on Windows, `htop`/`nvidia-smi` in WSL), then gracefully stop, and **resume** ingestion later (if supported) or rerun safely.

---

## 4) Running on Streamlit Cloud
Streamlit Cloud builds from `requirements.txt`. Keep dependencies compatible (avoid pinned wheels unavailable on the platform). If your logs show **unsatisfiable** packages (e.g., certain `pysqlite3-binary` versions), loosen pins or add platform-appropriate alternatives.

**Recommended health logs in app startup**
Make sure your app prints something like:
```
Chroma version: x.y.z
CHROMA_PERSIST_DIR: ...
CHROMA_COLLECTION: ...
```
When the app detects an **old/incompatible** store, you might see messages like:
```
Quarantined old Chroma store at ...
Collection opened after quarantine+rebuild.
```
This is expected self-healing.

> On Streamlit Cloud, building the index at startup can incur cold‑start time. Consider prebuilding locally and committing the `chroma_store/` (if size/policy allows), or implement a small admin page/CLI to (re)build on demand.

---

## 5) Performance Tuning
- **Batch size**: Start with `--batch-size 64` (GPU 8–12GB) and reduce on OOM.
- **Model choice**: `bge-large-zh` → highest quality but heavy; consider `bge-small-zh` when VRAM is tight.
- **Chunking**: 800–1200 tokens with ~10–15% overlap is a good default; adjust by document structure.
- **Mixed precision**: Enable fp16 for embedding models when supported to save VRAM.
- **I/O**: Place `chroma_store/` and `data/` on fast SSD.

**Monitoring**
```bash
watch -n 1 nvidia-smi
htop
```

---

## 6) Verifying the Index
After `create_db.py` completes, run a quick script or use the app’s **Health Check** area to:
- Count collection documents > 0
- Sample a query and ensure non‑empty hits with plausible metadata

**CLI smoke test (example)**
```bash
python - <<'PY'
from rag.retriever import get_chroma_collection
col = get_chroma_collection()
print('count =', col.count())
print('sample =', col.peek())
PY
```

---

## 7) Common Errors & Fixes

### A) Meta tensor / device transfer error (PyTorch on Streamlit Cloud)
- Cause: moving a module from meta device incorrectly.
- Fix: Ensure correct model init path; avoid `.to()` from meta; initialize weights on the target device properly; prefer CPU embeddings on Cloud if GPU is unavailable.

### B) Dependency resolution fails (e.g., `pysqlite3-binary`)
- Fix: Unpin or choose a compatible version. On Cloud, wheels may differ from local. Keep `requirements.txt` minimal and broadly compatible.

### C) WSL appears stuck during ingestion
- Check `htop`/`nvidia-smi` to confirm progress.
- Use `tmux` to detach/reattach; avoid closing the Windows terminal abruptly.
- If interrupted, re‑run ingestion. If your script supports **idempotent upsert/resume**, it will skip already‑done chunks.

### D) Empty results at query time
- Confirm env vars match ingestion: `CHROMA_PERSIST_DIR`, `CHROMA_COLLECTION`, `EMBEDDING_MODEL`.
- Ensure `guidelines.parsed.jsonl` wasn’t empty or malformed.
- Rebuild the index; watch for warnings about quarantined stores.

---

## 8) Reproducibility Checklist
- ✅ Pin major versions of Chroma & embedding libs in `requirements.txt` (avoid overspecific pins).
- ✅ Store a **data manifest** (doc list + checksums) under `data/`.
- ✅ Log ingestion parameters (chunk size, overlap, model, batch size) to a run file, e.g., `ingest/.run-YYYYMMDD_HHMM.json`.
- ✅ Consider a `Makefile` for repeatable pipelines:

```Makefile
parse:
	python ingest/parse_pdfs.py --in-dir data/pdfs --out data/guidelines.parsed.jsonl

index:
	python ingest/create_db.py --in data/guidelines.parsed.jsonl \
	  --persist-dir $(CHROMA_PERSIST_DIR) --collection $(CHROMA_COLLECTION) \
	  --embed-model $(EMBEDDING_MODEL)

all: parse index
```

---

## 9) Minimal API Contract (expected fields in JSONL)
Each line in `guidelines.parsed.jsonl` should be a JSON object like:
```json
{
  "id": "<uuid>",
  "title": "Guideline Title",
  "section": "1.2 Contraindications",
  "text": "Chunk text…",
  "source": "<relative path or URL>",
  "page_start": 12,
  "page_end": 13,
  "lang": "zh"  
}
```
> Your `create_db.py` should: (1) validate required fields, (2) compute embeddings from `text`, (3) upsert into collection with meaningful metadata for later display.

---

## 10) Health Check Snippets (copy into app or standalone)
```python
# During app startup (recommended)
import chromadb, os
print("Chroma version:", getattr(chromadb, "__version__", "unknown"))
print("CHROMA_PERSIST_DIR:", os.getenv("CHROMA_PERSIST_DIR"))
print("CHROMA_COLLECTION:", os.getenv("CHROMA_COLLECTION"))
```

---

## 11) FAQ
- **Q: Do I have to rebuild after adding new PDFs?**
  - A: Yes. Re‑run parsing for the new files and re‑index. With upsert semantics, only new/changed chunks are added.
- **Q: Can I use an English embedding model?**
  - A: Yes. Choose a multilingual or English model consistent with your corpus; mixing models across the same collection is discouraged.
- **Q: Can I commit `chroma_store/` to Git?**
  - A: Possible for demos if size is manageable; otherwise generate on deployment.

---

## 12) Support Matrix (suggested defaults)
| Environment | Model | Batch | Notes |
|---|---|---|---|
| Local (RTX 4070) | `bge-large-zh-v1.5` | 64 | Use fp16 if available; monitor VRAM |
| Local (CPU only) | `bge-small-zh-v1.5` | 16 | Slower; keep overlap modest |
| Streamlit Cloud | `bge-small-zh-v1.5` | 16–32 | Avoid heavy wheels; prefer CPU embeddings |

---

## 13) Change Log
- v1.0 (2025‑09‑27): Initial draft for CareMind‑Streamlit ingestion pipeline with local & cloud guidance, health checks, and troubleshooting.

