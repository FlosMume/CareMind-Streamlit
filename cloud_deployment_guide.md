# CareMind · Streamlit Community Cloud Deployment Guide

Author: Samuel Huang  
Assisted by: GitHub Copilot (GPT-5.2)  
Last updated: 2025-12-25  

This file documents the **current, working** Streamlit Community Cloud deployment setup for CareMind, and the exact fixes we applied after the Cloud error:

> `InvalidCollectionException: Collection guideline_chunks_v2 does not exist`

---

## 1) Current Working State (as of 2025-12-25)

- Deployment branch: `demo-data` (last known working commit: `913c096`)
- Active vector store: `chroma_store/` (a smaller ~26MB demo store)
- Expected collection: `guideline_chunks_v2`
- Code hardening: `rag/retriever.py` now defaults to `guideline_chunks_v2` and has safer “missing collection” handling.

Important: in this workspace, the `main` branch currently points to an **initial commit** (`c3b8f30`). Do **not** deploy `main` unless you intentionally want that minimal state.

---

## 2) Streamlit Cloud App Settings

In Streamlit Community Cloud:

- **Repository**: `FlosMume/CareMind-Streamlit`
- **Branch**: `demo-data`
- **Main file path**: `app.py`
- **Python**: matches `runtime.txt` (currently Python 3.10.x)

---

## 3) Required Secrets / Environment Variables

Set these in Streamlit Cloud “Secrets” (recommended):

```toml
CHROMA_PERSIST_DIR = "./chroma_store"
CHROMA_COLLECTION = "guideline_chunks_v2"
```

Optional (defaults are usually fine):

```toml
DRUG_DB_PATH = "./db/drugs.sqlite"
EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"
```

---

## 4) Why the Cloud Error Happened (Root Cause)

Streamlit Cloud deploys **only what’s in the selected Git branch**.

The error `Collection guideline_chunks_v2 does not exist` happens when:

- `chroma_store/` was not shipped to Cloud (ignored/untracked), or
- Cloud is pointing at the wrong `CHROMA_PERSIST_DIR`, or
- The deployed store contains a different collection name.

Fix applied: we ensured `chroma_store/` is included in `demo-data`, and standardized the default collection to `guideline_chunks_v2`.

---

## 5) Verify Locally (Quick Checks)

From repo root:

```bash
# Confirm the store exists
ls -lh chroma_store/chroma.sqlite3

# Confirm the expected collection exists
sqlite3 chroma_store/chroma.sqlite3 "SELECT name FROM collections;"
```

---

## 6) Git LFS Note (Very Important)

This repo is configured to store `chroma_store/**` and `*.sqlite3` using Git LFS via `.gitattributes`.

If Streamlit Cloud ever fails to fetch LFS files (symptoms: `chroma.sqlite3` is a tiny text pointer file; SQLite open errors), you have two options:

1) Keep LFS, and ensure the deployment environment fetches LFS objects (preferred if supported).
2) Remove LFS tracking for the demo store (since the demo store is small enough), then commit the real files as normal Git blobs.

---

## 7) Troubleshooting Cheatsheet

### Error: `InvalidCollectionException: ... does not exist`
- Confirm secrets: `CHROMA_PERSIST_DIR=./chroma_store`, `CHROMA_COLLECTION=guideline_chunks_v2`.
- Confirm the store shipped: `chroma_store/chroma.sqlite3` exists in the deployed branch.
- Confirm the collection exists by checking the sysdb query in section 5.

### Error: SQLite / Chroma import failures
- Confirm `requirements.txt` includes `chromadb` and `pysqlite3-binary`.
- Confirm `runtime.txt` is set (Python 3.10.x).

---

## 8) Optional: Sync `main` into `demo-data` (Only If You Actually Need It)

You only need this if you want recent application changes from `main` while keeping demo data.

Given the current state of this workspace (`main` is an initial commit), validate your remote branches first:

```bash
git fetch origin
git log --oneline --decorate --max-count=5 origin/main
git log --oneline --decorate --max-count=5 origin/demo-data
```

If `origin/main` is truly your latest work, merge or cherry-pick into `demo-data`. If not, stop and recover the correct `main` history first.

---

## 9) Certificate (Deployment Record)

Certificate ID: `CareMind-SCC-2025-12-25`  
Certified by: Samuel Huang  

I certify (as an internal deployment record) that on 2025-12-25 the following configuration is expected to run on Streamlit Community Cloud:

- Branch: `demo-data`
- Vector store path: `./chroma_store`
- Chroma collection: `guideline_chunks_v2`
- Store size target: ~26MB demo store

Signature: Samuel Huang  
Date: 2025-12-25

---

## 🤔 Decision Matrix

| Scenario | Option 1 (Merge) | Option 2 (Rebuild) |
|----------|------------------|-------------------|
| Want to preserve demo-data history | ✅ Yes | ❌ No |
| Confidence in handling conflicts | ✅ Medium-High | ⚠️ Any level |
| Risk tolerance | 🟢 Low risk | 🟡 Medium risk |
| Time to complete | ⏱️ 30-60 min | ⏱️ 60-120 min |
| Data preservation | ✅ Automatic | ⚠️ Manual backup |

---

## 📞 Need Help?

If you encounter issues:
1. Check this file's rollback section
2. Don't panic - your backup branch exists!
3. Use `git reflog` to find any lost commits
4. Post in Streamlit Community forums if Streamlit-specific issues

---
