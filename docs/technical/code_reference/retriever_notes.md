# retriever_notes.md

## Overview
The **Retriever** module is a core component of the CareMind-Streamlit project.  
It is responsible for **dense vector search** over clinical guideline chunks stored in **ChromaDB**, returning candidate passages that serve as evidence for clinical decision support.

The retriever acts as the **first stage in the RAG pipeline**:
1. Accepts a clinical query (and optionally, a drug name).  
2. Encodes the query using the configured embedding model.  
3. Searches the ChromaDB collection for the top-N most relevant passages.  
4. Returns structured hits (text + metadata), which may later be reranked or passed directly to the LLM orchestrator.

---

## File Location
```
caremind-streamlit/rag/retriever.py
```

---

## Key Responsibilities
- **ChromaDB connection management**  
  - Open/initialize a persistent collection (`CHROMA_PERSIST_DIR`, `CHROMA_COLLECTION`).
  - Handle versioning, quarantining, or rebuilding if schema mismatch is detected.

- **Vector embedding of queries**  
  - Use model defined in `EMBEDDING_MODEL` environment variable (e.g., `BAAI/bge-large-zh-v1.5`).
  - Encode both queries and stored guideline chunks into dense vectors.

- **Search functions**  
  - `search_guidelines(query, k=5)` → main retrieval function.
  - Optionally integrates reranking before returning top-K.

- **Structured output**  
  - Returns results as list of dicts with fields:  
    ```json
    {
      "text": "...", 
      "source": "...", 
      "page": ..., 
      "score": ...
    }
    ```

---

## Environment Variables
The retriever depends on the following `.env` variables:

| Variable              | Required | Description |
|------------------------|----------|-------------|
| `CHROMA_PERSIST_DIR`  | Yes      | Directory path for Chroma persistence (e.g., `./chroma_store`) |
| `CHROMA_COLLECTION`   | Yes      | Name of the collection to open (e.g., `guideline_chunks_1024_v2`) |
| `EMBEDDING_MODEL`     | Yes      | Embedding model to use (e.g., `BAAI/bge-large-zh-v1.5`) |

---

## Typical Usage
```python
from rag import retriever as R

hits = R.search_guidelines(
    "合并支气管哮喘的高血压患者是否可用β受体阻滞剂？",
    k=5
)

for h in hits:
    print(h["text"], h["score"])
```

---

## Debugging Notes
- **Empty results**:  
  Ensure the ingestion step has been completed successfully (`create_db.py` → Chroma collection populated).  
  Run health check: look for `Chroma version: ...` in app startup logs.

- **Device errors (`Cannot copy out of meta tensor`)**:  
  Usually caused by improper GPU initialization in Torch. Ensure your environment uses the correct CUDA/cuDNN versions.

- **Performance tuning**:  
  - Increase/decrease `k` (retrieved chunk count).  
  - Adjust embedding model (`bge-small` for faster, `bge-large` for more accurate).  
  - Consider reranker integration for better relevance at cost of speed.

---

## Extension Ideas
- **Hybrid retrieval**: add keyword search (BM25 or SQLite FTS5) alongside dense retrieval.  
- **Domain-aware reranking**: integrate a cross-encoder reranker (e.g., `bge-reranker-base`).  
- **Structured metadata filters**: constrain search by drug, guideline section, or publication year.  
- **Logging**: store retrieved chunks and query metadata in a `retrieval_log/` folder for audit.

---

## Related Files
- [`ingest/create_db.py`](../ingest/create_db.py) → builds the Chroma collection.  
- [`rag/pipeline.py`](./pipeline.py) → orchestrates retrieval + LLM prompt construction.  
- [`app.py`](../app.py) → integrates retriever into Streamlit UI.  
