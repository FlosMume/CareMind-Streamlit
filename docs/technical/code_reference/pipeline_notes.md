# CareMind-Streamlit: Pipeline Notes

This document provides notes on the **pipeline** used in the CareMind-Streamlit project.

---

## Overview

The pipeline orchestrates the end-to-end workflow for clinical question answering. It connects the retriever, reranker, large language model (LLM) orchestrator, and output formatter into a structured sequence.

---

## Key Components

### 1. Retriever (ChromaDB)
- Performs dense vector search over guideline embeddings.
- Controlled by environment variables:
  - `CHROMA_PERSIST_DIR`
  - `CHROMA_COLLECTION`
  - `EMBEDDING_MODEL`
- Returns candidate evidence chunks with metadata.

### 2. Reranker (Optional)
- Reorders the top-N hits from the retriever.
- Uses relevance scoring models or heuristics.
- Improves evidence precision for downstream LLM.

### 3. LLM Orchestrator
- Integrates evidence into prompts.
- Handles user query, optional drug input, and system constraints.
- Generates structured drafts (advice + references).

### 4. Formatter
- Produces standardized output:
  - **Clinical suggestion (draft)**
  - **Evidence snippets**
  - **Drug structured data**
- Includes disclaimers for compliance.

---

## Pipeline Flow (Simplified)

1. **User Query** → Preprocess (optional drug input).
2. **Retriever** → Fetch evidence chunks.
3. **Reranker** → Refine top-K evidence (if enabled).
4. **LLM Orchestrator** → Draft suggestion with references.
5. **Formatter** → Output advice, evidence, and structured drug info.
6. **Streamlit UI** → Render results + export options.

---

## Notes on Implementation

- Implemented in `rag/pipeline.py`.
- Functions to check:
  - `answer(question, drug_name=None, k=5, lang="zh")`
  - `format_output(advice, evidence, drugs)`
- Error handling ensures fallback behavior when no evidence is found.
- Export to Markdown supported.

---

## Future Enhancements

- Add **reranker integration** (BM25, cross-encoder, etc.).
- More robust **prompt templates** for structured outputs.
- Multi-lingual support expansion (Chinese + English drafts).
- Fine-tuning for clinical domain specificity.

---

*Last updated: September 27, 2025*
