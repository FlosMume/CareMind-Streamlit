# Chroma Vector Database Tutorial for CareMind

## Table of Contents
1. [Introduction](#introduction)
2. [What is Chroma?](#what-is-chroma)
3. [Installation & Setup](#installation--setup)
4. [Core Concepts](#core-concepts)
5. [CareMind Architecture](#caremind-architecture)
6. [Step-by-Step Guide](#step-by-step-guide)
7. [Common Operations](#common-operations)
8. [Troubleshooting](#troubleshooting)
9. [Advanced Topics](#advanced-topics)
10. [Best Practices](#best-practices)

---

## Introduction

This tutorial covers **Chroma**, the vector database system used in the CareMind clinical decision support application. Chroma enables fast semantic search over medical guidelines through dense vector embeddings.

**What you'll learn:**
- How to install and configure Chroma
- How to create and manage vector collections
- How to embed documents and perform semantic search
- How to integrate Chroma with Streamlit applications
- Troubleshooting and optimization strategies

---

## What is Chroma?

### Overview

**Chroma** is a lightweight, open-source vector database designed for embedding and semantic search. It provides:

- **Persistent storage** of document embeddings and metadata
- **Fast similarity search** using vector distances
- **Flexible metadata filtering** for structured queries
- **Easy integration** with Python applications
- **Multi-collection support** for organizing different datasets

### Why Chroma for CareMind?

In the CareMind project, Chroma serves as the **Retrieval-Augmented Generation (RAG)** backend:

1. **Medical guidelines** are parsed into chunks
2. Each chunk is **embedded** using a Chinese language model (e.g., `BAAI/bge-large-zh-v1.5`)
3. Embeddings are stored in a **persistent Chroma collection**
4. When a doctor asks a clinical question, the query is embedded and **semantically matched** against stored guidelines
5. Retrieved evidence snippets feed into an LLM for final clinical advice generation

**Chroma provides:**
- Efficient vector storage and indexing
- Similarity-based retrieval in milliseconds
- Metadata preservation (source, section, year, etc.)
- Seamless Streamlit integration via caching

---

## Installation & Setup

### Prerequisites

- **Python 3.8+** (3.10 recommended for CareMind)
- **pip** or **conda** package manager
- **SQLite** (included in Python, but may need shimming on some cloud platforms)

### Step 1: Install Chroma

```bash
# Using pip
pip install chromadb

# Or with conda
conda install -c conda-forge chromadb
```

**Check installation:**
```bash
python -c "import chromadb; print(chromadb.__version__)"
```

### Step 2: Install CareMind Dependencies

```bash
cd caremind-streamlit-dec
pip install -r requirements.txt
```

This installs:
- `chromadb` — vector database
- `sentence-transformers` — embedding models
- `torch` — deep learning backend (CPU or CUDA)
- `streamlit` — web UI framework
- Other dependencies (pandas, numpy, pydantic, etc.)

### Step 3: Configure Environment Variables

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```env
# Chroma persistence directory (will be created if it doesn't exist)
CHROMA_PERSIST_DIR=./chroma_store

# Collection name inside Chroma
CHROMA_COLLECTION=guideline_chunks_1024_v2

# Embedding model (Chinese-capable SentenceTransformer)
# Options: BAAI/bge-large-zh-v1.5, BAAI/bge-small-zh, sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5

# Optional: disable telemetry
CHROMA_TELEMETRY_OFF=1

# Optional: drug database path
DRUG_DB_PATH=./db/drugs.sqlite
```

**Load environment variables in your shell:**

```bash
export $(grep -v '^#' .env | xargs)
```

Or use `python-dotenv` in your scripts (CareMind does this automatically).

### Step 4: Verify Installation

```bash
# Test Chroma
python -c "
from chromadb import PersistentClient
client = PersistentClient(path='./chroma_store')
print('✓ Chroma client created successfully')
"

# Test embedding model
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-small-zh')
embeddings = model.encode(['hello', 'world'])
print(f'✓ Model loaded, embedding dim: {embeddings.shape[1]}')
"
```

---

## Core Concepts

### 1. Collections

A **collection** is a named container within Chroma that holds documents, embeddings, and metadata.

**Example:**
```python
from chromadb import PersistentClient

client = PersistentClient(path='./chroma_store')

# Create or get a collection
collection = client.get_or_create_collection(
    name="guideline_chunks",
    metadata={"description": "Medical guidelines split into 1024-token chunks"}
)
```

### 2. Documents and Embeddings

When you add a document to a collection:
1. **Text is embedded** into a vector (list of floats)
2. **Vector and metadata are stored** in the collection
3. **Document ID is indexed** for fast retrieval

**Structure:**
```python
# Adding documents
collection.add(
    ids=["doc_1", "doc_2"],
    documents=["Clinical guideline text 1...", "Clinical guideline text 2..."],
    metadatas=[
        {"source": "hypertension_guide.pdf", "section": "2.3", "year": 2024},
        {"source": "diabetes_guide.pdf", "section": "1.1", "year": 2023}
    ]
)
```

### 3. Embedding Functions

An **embedding function** converts text into dense vectors. CareMind uses **SentenceTransformers**, which are pre-trained on massive text corpora.

**Chinese Models (recommended for CareMind):**
- `BAAI/bge-large-zh-v1.5` — 1024-dim, high quality (requires ~2GB VRAM)
- `BAAI/bge-small-zh` — 384-dim, lightweight (fits on 8GB VRAM)
- `intfloat/e5-large-v2` — 1024-dim, multilingual

**How embeddings work:**
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('BAAI/bge-large-zh-v1.5')

# Embed a single query
query = "高血压患者的降压目标是多少?"
query_embedding = model.encode(query)
print(f"Query embedding shape: {query_embedding.shape}")  # (1024,)

# Embed a batch of documents
docs = ["文本1...", "文本2...", "文本3..."]
doc_embeddings = model.encode(docs)
print(f"Doc embeddings shape: {doc_embeddings.shape}")  # (3, 1024)
```

### 4. Similarity Search

Chroma computes **cosine similarity** between query and document vectors to find the most relevant results.

**Algorithm:**
1. Embed the query: `query_vec = embed(query)`
2. For each document: `similarity = cosine(query_vec, doc_vec)`
3. Return top-K documents sorted by similarity

**Example:**
```python
results = collection.query(
    query_texts=["高血压治疗"],
    n_results=5
)

# results = {
#     "ids": [["id_1", "id_2", "id_3", "id_4", "id_5"]],
#     "documents": [["doc_text_1", "doc_text_2", ...]],
#     "distances": [[0.1, 0.2, 0.3, 0.4, 0.5]],
#     "metadatas": [[{...}, {...}, ...]]
# }
```

### 5. Metadata and Filtering

Chroma stores metadata alongside embeddings, enabling filtered search.

**Supported metadata types:**
- Strings: `"source": "guidelines.pdf"`
- Numbers: `"year": 2024`, `"section_id": 3`
- Booleans: `"is_critical": true`
- Lists of strings/numbers: `"keywords": ["hypertension", "cardiovascular"]`

**Filtering example:**
```python
# Query with metadata filter
results = collection.query(
    query_texts=["降压策略"],
    n_results=5,
    where={"year": {"$gte": 2020}}  # Only recent guidelines
)
```

### 6. Persistence

Chroma saves collections to disk in a directory (e.g., `./chroma_store`), so data persists across application restarts.

**Directory structure:**
```
chroma_store/
├── chroma-*.db              # SQLite database files
├── <collection_uuid>/       # Collection-specific directory
│   └── ...                  # Embedding indexes and metadata
```

---

## CareMind Architecture

### System Overview

```
┌────────────────────┐
│  User Query (UI)   │
└──────────┬─────────┘
           │
           ▼
┌────────────────────────────────────────┐
│  1. Embed Query (SentenceTransformer)  │
│     (EMBEDDING_MODEL environment var) │
└──────────┬─────────────────────────────┘
           │
           ▼
┌────────────────────────────────────────┐
│  2. Chroma Vector Search               │
│     (CHROMA_PERSIST_DIR)               │
│     (CHROMA_COLLECTION)                │
└──────────┬─────────────────────────────┘
           │
           ▼
┌────────────────────────────────────────┐
│  3. Retrieved Evidence Chunks          │
│     + metadata (source, section, etc) │
└──────────┬─────────────────────────────┘
           │
           ▼
┌────────────────────────────────────────┐
│  4. LLM Orchestrator (pipeline.py)     │
│     Integrates evidence into prompt    │
└──────────┬─────────────────────────────┘
           │
           ▼
┌────────────────────────────────────────┐
│  5. Clinical Advice (with citations)   │
└────────────────────────────────────────┘
```

### Key Files in CareMind

| File | Purpose |
|------|---------|
| `rag/retriever.py` | Chroma client/collection management, search API |
| `ingest/build_vectors.py` | Parse JSONL → embed chunks → write to Chroma |
| `rag/pipeline.py` | Orchestrate retriever + LLM for final advice |
| `tools/check_chroma.py` | Diagnostic tool to inspect Chroma health |
| `app.py` | Streamlit UI that calls `retriever.search_guidelines()` |

---

## Step-by-Step Guide

### Phase 1: Data Preparation

#### 1.1 Parse Medical Guidelines

Convert raw PDF/text guidelines into structured JSONL:

```bash
python ingest/parse_docs.py \
  --in data/guidelines/ \
  --out data/guidelines.parsed.jsonl \
  --chunk-size 1024
```

**Input:** Directory of PDF or text files  
**Output:** JSONL file with structure:
```json
{"content": "chunk text here", "meta": {"source": "...", "year": 2024, ...}}
{"content": "chunk text here", "meta": {"source": "...", "year": 2024, ...}}
```

#### 1.2 Verify Input Data

```bash
# Check first few lines
head data/guidelines.parsed.jsonl

# Count total chunks
wc -l data/guidelines.parsed.jsonl
```

### Phase 2: Build Vector Database

#### 2.1 Configure Environment

```bash
export CHROMA_PERSIST_DIR=./chroma_store
export CHROMA_COLLECTION=guideline_chunks_1024_v2
export EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5
export EMBED_BATCH_SIZE=16
```

#### 2.2 Run Embedding Script

```bash
# For GPU (fast)
python ingest/build_vectors.py

# For CPU (slow but works)
export DEVICE=cpu
python ingest/build_vectors.py
```

**What this script does:**
1. Reads JSONL file line-by-line
2. Embeds document chunks in batches
3. Sanitizes metadata (handles complex types)
4. Writes embeddings to Chroma with metadata
5. Handles OOM gracefully (batch backoff)

**Progress output:**
```
📦 Persist dir: /home/user/caremind-streamlit-dec/chroma_store
🗃️  Collection: guideline_chunks_1024_v2
🔧 Embedding model: BAAI/bge-large-zh-v1.5
⚙️  Device: cuda
📚 Reading JSONL: 100%|████████| 5234/5234 [12:34<00:00, 6.94 docs/s]
✅ Upserted 5234 documents
```

#### 2.3 Verify Database Build

```bash
python tools/check_chroma.py --q "高血压 降压目标"
```

**Expected output:**
```
──────── Health Check ────────
Chroma dir:     ./chroma_store
Collection:     guideline_chunks_1024_v2
Embedding mdl:  BAAI/bge-large-zh-v1.5
Chroma version: 0.4.24

✔️ Collections found:
  - guideline_chunks_1024_v2 (5234 documents)

✔️ 能命中，Top1 预览：
高血压患者的降压目标应该根据不同的临床情况确定……
耗时：0.45s
```

### Phase 3: Query and Retrieve

#### 3.1 Direct Python Query

```python
from rag import retriever as R

# Search for relevant guidelines
query = "高血压患者合并糖尿病应该如何管理血压?"
hits = R.search_guidelines(query, k=5)

for i, hit in enumerate(hits, 1):
    print(f"{i}. Score: {hit['score']:.3f}")
    print(f"   Source: {hit.get('source', 'unknown')}")
    print(f"   Text: {hit['text'][:200]}...\n")
```

#### 3.2 Integration with Streamlit

The Streamlit UI automatically uses Chroma via `rag.retriever`:

```python
import streamlit as st
from rag import retriever as R

# User input
query = st.text_area("Clinical Question")

if st.button("Search Guidelines"):
    # Retrieve evidence
    hits = R.search_guidelines(query, k=5)
    
    # Display results
    for hit in hits:
        with st.expander(f"📄 {hit.get('source', 'Unknown')}"):
            st.write(hit['text'])
            st.write(f"**Similarity Score:** {hit['score']:.3f}")
```

### Phase 4: Production Deployment

#### 4.1 Streamlit Cloud

1. Push repository to GitHub
2. Connect to Streamlit Cloud
3. Set environment variables in Streamlit Cloud dashboard:
   ```
   CHROMA_PERSIST_DIR=./chroma_store
   CHROMA_COLLECTION=guideline_chunks_1024_v2
   EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5
   ```
4. Ensure `chroma_store/` is committed to Git (or use a separate branch)

#### 4.2 Self-Hosted Server

```bash
# Start Streamlit app
streamlit run app.py \
  --server.port 8501 \
  --logger.level=info

# Or with systemd/supervisor for auto-restart
```

---

## Common Operations

### 1. List All Collections

```python
from chromadb import PersistentClient

client = PersistentClient(path='./chroma_store')
collections = client.list_collections()

for col in collections:
    print(f"Collection: {col.name}")
    print(f"  Count: {col.count()}")
    print(f"  Metadata: {col.metadata}")
```

### 2. Query a Collection

```python
collection = client.get_collection("guideline_chunks_1024_v2")

# Simple query
results = collection.query(
    query_texts=["Hypertension treatment"],
    n_results=10
)

# Query with metadata filter
results = collection.query(
    query_texts=["Hypertension treatment"],
    n_results=10,
    where={"year": {"$gte": 2020}}
)

# Query with multiple conditions
results = collection.query(
    query_texts=["Hypertension treatment"],
    n_results=10,
    where={
        "$and": [
            {"year": {"$gte": 2020}},
            {"source": {"$in": ["hypertension_guide.pdf", "cardio_guide.pdf"]}}
        ]
    }
)
```

### 3. Add Documents

```python
collection.add(
    ids=["doc_1", "doc_2", "doc_3"],
    documents=[
        "Clinical guideline text 1...",
        "Clinical guideline text 2...",
        "Clinical guideline text 3..."
    ],
    metadatas=[
        {"source": "guide.pdf", "section": "2.1", "year": 2024},
        {"source": "guide.pdf", "section": "2.2", "year": 2024},
        {"source": "guide.pdf", "section": "2.3", "year": 2024}
    ]
)
```

### 4. Update/Upsert Documents

```python
# Upsert (update if exists, insert if new)
collection.upsert(
    ids=["doc_1"],
    documents=["Updated text..."],
    metadatas=[{"source": "guide.pdf", "updated": True}]
)
```

### 5. Delete Documents

```python
# Delete by ID
collection.delete(ids=["doc_1", "doc_2"])

# Delete by metadata filter
collection.delete(where={"source": "old_guide.pdf"})
```

### 6. Get Document by ID

```python
result = collection.get(ids=["doc_1"])
print(result['documents'][0])
print(result['metadatas'][0])
```

### 7. Delete a Collection

```python
client.delete_collection("old_collection")
```

---

## Troubleshooting

### Issue 1: "No such module chromadb"

**Cause:** Chroma not installed  
**Solution:**
```bash
pip install chromadb
```

### Issue 2: "Chroma already exists with different settings"

**Cause:** Multiple Chroma clients created with different Settings  
**Solution:** In CareMind, use the singleton pattern via `retriever.get_chroma_client()`:
```python
from rag import retriever as R

# Uses cached, single client
client = R.get_chroma_client()
collection = R.get_chroma_collection()
```

### Issue 3: "Collection not found"

**Cause:** Collection name mismatch or incorrect persist directory  
**Solution:**
```bash
# Check what collections exist
python tools/check_chroma.py

# Verify environment variables
echo $CHROMA_PERSIST_DIR
echo $CHROMA_COLLECTION
```

### Issue 4: Slow Queries (>1 second)

**Causes & Solutions:**
1. **Large embedding dimension:** Use a smaller model
   ```bash
   export EMBEDDING_MODEL=BAAI/bge-small-zh  # 384-dim instead of 1024
   ```
2. **Too many documents:** Rebuild with smaller chunk size or filter metadata
3. **CPU embedding:** Use GPU
   ```bash
   export DEVICE=cuda
   ```

### Issue 5: Out of Memory (OOM) During Embedding

**Cause:** Batch size too large for GPU  
**Solution:** Lower batch size
```bash
export EMBED_BATCH_SIZE=8  # default 16
python ingest/build_vectors.py
```

Or use CPU fallback:
```bash
export OOM_CPU_FALLBACK=1
python ingest/build_vectors.py
```

### Issue 6: Embedding Dimension Mismatch

**Cause:** Changed embedding model after building collection  
**Example:** Built with `BAAI/bge-large-zh-v1.5` (1024-dim) but querying with `BAAI/bge-small-zh` (384-dim)  
**Solution:** Rebuild entire collection with consistent model
```bash
export EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5
python ingest/create_db.py --in data/guidelines.parsed.jsonl
```

### Issue 7: SQLite Lock (Streamlit Cloud)

**Cause:** Concurrent access to SQLite database  
**Solution:** Use `pysqlite3-binary` in requirements.txt for Streamlit Cloud
```bash
pip install pysqlite3-binary
```

---

## Advanced Topics

### 1. Custom Embedding Functions

Create a custom embedding function if needed:

```python
from chromadb.api.types import EmbeddingFunction

class CustomEmbedder(EmbeddingFunction):
    def __init__(self, model_name):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
    
    def __call__(self, input):
        # input is list of strings
        return self.model.encode(input).tolist()

# Use in collection
embedder = CustomEmbedder("BAAI/bge-large-zh-v1.5")
collection = client.get_or_create_collection(
    name="my_collection",
    embedding_function=embedder
)
```

### 2. Metadata Filtering with Complex Conditions

```python
# OR condition
results = collection.query(
    query_texts=["hypertension"],
    where={
        "$or": [
            {"source": "guide1.pdf"},
            {"source": "guide2.pdf"}
        ]
    }
)

# NOT condition
results = collection.query(
    query_texts=["hypertension"],
    where={
        "$not": {"source": "outdated_guide.pdf"}
    }
)

# String contains
results = collection.query(
    query_texts=["hypertension"],
    where={"source": {"$regex": ".*guide.*"}}
)
```

### 3. Batch Operations for Large Datasets

```python
from tqdm import tqdm

docs = load_all_documents()  # list of {text, meta}
batch_size = 100

for i in tqdm(range(0, len(docs), batch_size)):
    batch = docs[i:i+batch_size]
    
    collection.upsert(
        ids=[d["id"] for d in batch],
        documents=[d["text"] for d in batch],
        metadatas=[d["meta"] for d in batch]
    )
```

### 4. Reranking Retrieved Results

After Chroma retrieves top-K candidates, rerank with a more powerful model:

```python
from sentence_transformers import CrossEncoder

# Initial retrieval (fast)
results = collection.query(query_texts=[query], n_results=50)

# Rerank (slower but more accurate)
reranker = CrossEncoder('BAAI/bge-reranker-large')
documents = results['documents'][0]
scores = reranker.predict([[query, doc] for doc in documents])

# Sort by reranker scores
ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
top_5 = ranked[:5]
```

### 5. Monitoring and Metrics

```python
import time

start = time.time()
results = collection.query(query_texts=[query], n_results=5)
latency_ms = (time.time() - start) * 1000

print(f"Query latency: {latency_ms:.1f}ms")
print(f"Results returned: {len(results['documents'][0])}")
print(f"Average score: {sum(results['distances'][0]) / len(results['distances'][0]):.3f}")
```

---

## Best Practices

### 1. **Use Environment Variables**
Always externalise configuration:
```python
import os
persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")
collection_name = os.getenv("CHROMA_COLLECTION", "guideline_chunks")
embedding_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
```

### 2. **Cache Client & Collection**
In Streamlit, use caching to avoid recreating Chroma clients:
```python
import streamlit as st
from chromadb import PersistentClient

@st.cache_resource
def get_chroma_client():
    return PersistentClient(path="./chroma_store")

@st.cache_resource
def get_collection():
    client = get_chroma_client()
    return client.get_collection("guideline_chunks")
```

### 3. **Validate Metadata**
Ensure metadata is scalar or JSON-serializable:
```python
def sanitize_metadata(meta):
    """Convert complex types to strings for Chroma."""
    sanitized = {}
    for key, value in meta.items():
        if isinstance(value, (str, int, float, bool)):
            sanitized[key] = value
        elif isinstance(value, list):
            sanitized[key] = [str(v) for v in value]  # convert to strings
        else:
            sanitized[key] = str(value)  # fallback
    return sanitized
```

### 4. **Log Queries for Debugging**
```python
import logging

logger = logging.getLogger(__name__)

def search_with_logging(query, k=5):
    logger.info(f"Query: {query} | k={k}")
    
    results = collection.query(query_texts=[query], n_results=k)
    
    logger.info(f"Results: {len(results['documents'][0])} docs retrieved")
    logger.debug(f"Scores: {results['distances'][0]}")
    
    return results
```

### 5. **Handle Empty Results Gracefully**
```python
results = collection.query(query_texts=[query], n_results=5)

if not results['documents'] or not results['documents'][0]:
    st.warning("No relevant guidelines found. Please refine your search.")
else:
    for i, (doc, score, meta) in enumerate(zip(
        results['documents'][0],
        results['distances'][0],
        results['metadatas'][0]
    )):
        st.write(f"**Result {i+1}** (Score: {score:.3f})")
        st.write(doc)
        st.caption(f"Source: {meta.get('source', 'unknown')}")
```

### 6. **Version Control Collections**
Track collection configurations in your code:
```python
COLLECTION_VERSION = "guideline_chunks_1024_v2"  # Include embedding dim and version
EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"      # Lock to specific model

# Document why this version exists
"""
Changelog:
- v1: Initial collection (2024-01-15) | BAAI/bge-small-zh | 384-dim | 3000 docs
- v2: Added 2024 guidelines (2024-03-20) | BAAI/bge-large-zh-v1.5 | 1024-dim | 5234 docs
"""
```

### 7. **Monitor Collection Health**
```python
def health_check(collection):
    """Verify collection is healthy."""
    try:
        # Can we query?
        test_query = "test"
        results = collection.query(query_texts=[test_query], n_results=1)
        
        # Count documents
        count = collection.count()
        
        return {
            "status": "healthy",
            "document_count": count,
            "can_query": True
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

health = health_check(collection)
print(health)
```

### 8. **Backup Your Data**
```bash
# Backup Chroma directory
cp -r chroma_store chroma_store.backup.$(date +%Y%m%d)

# Or use cloud storage
aws s3 sync chroma_store s3://my-bucket/chroma-backups/
```

---

## Summary

Chroma is a powerful, lightweight vector database that enables semantic search in CareMind. Key takeaways:

✅ **Install & configure** via environment variables  
✅ **Embed documents** using Chinese-capable language models  
✅ **Query collections** with semantic similarity search  
✅ **Integrate with Streamlit** using caching and RAG patterns  
✅ **Monitor health** with diagnostic tools  
✅ **Scale responsibly** by optimizing batch sizes and models  

For more information, see:
- [Chroma Official Docs](https://docs.trychroma.com/)
- [CareMind Architecture](../technical/architecture.md)
- [Retriever Module Reference](../../rag/retriever.md)
- [Data Ingestion Guide](../technical/data_ingestion.md)

---

**Last Updated:** December 2024  
**Authors:** CareMind Development Team
