# -*- coding: utf-8 -*-
"""
CareMind · RAG Retriever (richly commented, future-proof)

This consolidates the best parts of your previous retriever while fixing the
partial-merge/indentation issues that broke `streamlit run`.

Design goals
------------
1) Be **import-safe** in Streamlit Cloud and local dev:
   - Provide a SQLite compatibility shim (uses stdlib if >=3.35.0, otherwise
     aliases `pysqlite3` to `sqlite3`).
   - Lazy-import Chroma so that missing/old system libs won’t crash the module
     import (errors appear only when the functions are executed).
2) Keep **one** Chroma client and **one** collection cache to avoid duplicate
   stacks and race conditions across hot-reloads. (Uses st.cache_resource when
   available to prevent “An instance of Chroma already exists …with different
   settings”.)
3) Pre-migrate Chroma sysdb (`collections.configuration`) to insert a minimal
   JSON with `_type` when missing — this unblocks older Chroma stores.
4) Provide **stable public API** that `rag/pipeline.py` already calls:
      - search_guidelines(question: str, k: int=4) -> List[dict]
      - search_drug_structured(drug_name: str) -> Optional[dict]
   …and keep extra helpers you will likely need soon:
      - hybrid_search(query, k=8, use_sqlite=True) with RRF fusion
      - _sqlite_search_drugfacts(name_substr, topn=5) -> List[dict]
5) Rich logs to diagnose Cloud issues quickly and a `primary_collection_count()`
   helper so Streamlit can show “chunks available”.

Environment variables
---------------------
CHROMA_PERSIST_DIR  : default "./chroma_store"
CHROMA_COLLECTION   : default "guideline_chunks"
EMBEDDING_MODEL     : default "BAAI/bge-large-zh-v1.5"
DRUG_DB_PATH        : default "./db/drugs.sqlite"
CHROMA_TELEMETRY_OFF: default "1" (turn off anonymized telemetry)

CLI
---
$ python -m rag.retriever --q "哮喘 β受体阻滞剂" --topn 6
$ python rag/retriever.py --q "aspirin" --method sqlite
"""
from __future__ import annotations

import os
import sys
import json
import contextlib
from typing import Any, Dict, List, Optional, Tuple

# =============================================================================
# 0) Environment & logging
# =============================================================================

# Disable Chroma telemetry by default (good for Cloud)
os.environ.setdefault("CHROMA_TELEMETRY_ENABLED", "false")
os.environ.setdefault("CHROMA_ANONYMIZED_TELEMETRY", "false")

CHROMA_TELEMETRY_OFF = os.getenv("CHROMA_TELEMETRY_OFF", "1") not in ("0", "false", "False")

# Use absolute path to avoid ambiguity across working dirs / reloads
def _abs(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))

CHROMA_PERSIST_DIR = _abs(os.getenv("CHROMA_PERSIST_DIR", "./chroma_store"))
CHROMA_COLLECTION  = os.getenv("CHROMA_COLLECTION",  "guideline_chunks")
EMBEDDING_MODEL    = os.getenv("EMBEDDING_MODEL",    "BAAI/bge-large-zh-v1.5")
DRUG_DB_PATH       = _abs(os.getenv("DRUG_DB_PATH",  "./db/drugs.sqlite"))

VERSION = "retriever-2025-09-30"

def _log(*msg: Any) -> None:
    """Single place for retriever logs (helps when debugging Cloud deploys)."""
    from datetime import datetime
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] retriever:", *msg, flush=True)


# =============================================================================
# 1) SQLite compatibility shim (Cloud-friendly)
# =============================================================================
# Many managed environments ship an old libsqlite3 (<3.35.0) which breaks Chroma.
# Strategy: try stdlib sqlite3; if version too old or missing, alias pysqlite3.

def _ensure_sqlite() -> None:
    MIN = (3, 35, 0)
    try:
        import sqlite3 as _stdlib  # type: ignore
        try:
            ver_tuple = tuple(map(int, _stdlib.sqlite_version.split(".")))
        except Exception:
            ver_tuple = (0, 0, 0)
        if ver_tuple >= MIN:
            return
        # Too old → prefer pysqlite3 (Wheel works on Cloud)
        import pysqlite3 as _py  # type: ignore
        sys.modules["sqlite3"] = _py
    except Exception:
        # stdlib missing or unusable → last resort
        try:
            import pysqlite3 as _py  # type: ignore
            sys.modules["sqlite3"] = _py
        except Exception as e:
            raise RuntimeError(
                "SQLite not available. Install `pysqlite3-binary>=0.5.3` or "
                "use Python built with SQLite >= 3.35.0."
            ) from e

_ensure_sqlite()
import sqlite3  # type: ignore  # noqa: E402


# =============================================================================
# 2) Lazy embedder (SentenceTransformers) and Chroma loader
# =============================================================================
# We avoid importing heavy libs at module import time.

class _LazyEmbedder:
    """Load the embedding model on the first call."""
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def __call__(self, texts: List[str]) -> List[List[float]]:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            _log("Loading embedding model:", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        texts = [t if isinstance(t, str) else str(t) for t in texts]
        vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return vecs.tolist()

class _ChromaEmbedFn:
    """Adapter matching Chroma's embedding_function signature."""
    def __init__(self, embedder: _LazyEmbedder):
        self._embedder = embedder
    def __call__(self, inputs: List[str]) -> List[List[float]]:
        return self._embedder(inputs)

_EMBED = _LazyEmbedder(EMBEDDING_MODEL)

# =============================================================================
# 3) Streamlit-friendly caching primitives (one client per process)
# =============================================================================
try:
    import streamlit as st
    cache_resource = st.cache_resource
except Exception:
    def cache_resource(func):  # no-op fallback
        return func

# =============================================================================
# 4) Lazy import Chroma
# =============================================================================

def _chroma_import():
    """Import Chroma on demand."""
    try:
        from chromadb import PersistentClient, Settings  # type: ignore
        return PersistentClient, Settings
    except Exception as e:
        raise RuntimeError(
            "Failed to import `chromadb`. Install it and ensure SQLite is usable.\n"
            "Tip: add `pysqlite3-binary` to requirements on Streamlit Cloud."
        ) from e


# =============================================================================
# 5) Chroma sysdb pre-migration (fix missing configuration._type)
# =============================================================================

def _sysdb_paths(persist_dir: str) -> List[str]:
    """Return likely sysdb sqlite file paths inside `persist_dir`."""
    import glob
    pats = [
        os.path.join(persist_dir, "chroma-*.db"),   # modern
        os.path.join(persist_dir, "chroma.sqlite"), # older
        os.path.join(persist_dir, "chroma.db"),     # very old
    ]
    files: List[str] = []
    for p in pats:
        files.extend(glob.glob(p))
    return [f for f in files if os.path.isfile(f)]

def _has_type(conf: Optional[str]) -> bool:
    if not conf:
        return False
    try:
        obj = json.loads(conf)
        return isinstance(obj, dict) and "_type" in obj
    except Exception:
        return False

def _premigrate_sysdb(persist_dir: str) -> int:
    """
    Patch rows in `collections` where `configuration` is NULL/empty or lacks `_type`.
    Returns how many rows were updated across all discovered sysdb files.
    """
    paths = _sysdb_paths(persist_dir)
    if not paths:
        return 0
    total = 0
    for dbfile in paths:
        try:
            con = sqlite3.connect(dbfile)
            cur = con.cursor()
            cur.execute("SELECT id, configuration FROM collections")
            rows = cur.fetchall()
            fixed = 0
            for cid, conf in rows:
                if not _has_type(conf):
                    payload = json.dumps({"_type": "CollectionConfigurationInternal"}, ensure_ascii=False)
                    cur.execute("UPDATE collections SET configuration=? WHERE id=?", (payload, cid))
                    fixed += 1
            if fixed:
                con.commit()
                total += fixed
                _log(f"[premigrate] {dbfile}: patched {fixed} row(s).")
        except Exception as e:
            _log(f"[premigrate] {dbfile}: error:", repr(e))
        finally:
            with contextlib.suppress(Exception):
                con.close()
    return total  # <-- fixed (was a typo `tota`)


# =============================================================================
# 6) Client / collection getters (single, cached, consistent Settings)
# =============================================================================

@cache_resource
def get_chroma_client():
    """
    Create (once) and return a PersistentClient, after a best-effort premigrate.
    Using st.cache_resource guarantees a single instance per Streamlit process,
    preventing 'An instance of Chroma already exists …with different settings'.
    """
    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
    # best-effort migration before touching Chroma
    try:
        changed = _premigrate_sysdb(CHROMA_PERSIST_DIR)
        if changed:
            _log(f"Premigrated collections: {changed}")
    except Exception as e:
        _log("Premigrate skipped due to error:", repr(e))
    PersistentClient, Settings = _chroma_import()
    # IMPORTANT: keep Settings EXACTLY the same everywhere
    return PersistentClient(
        path=CHROMA_PERSIST_DIR,
        settings=Settings(
            anonymized_telemetry=not CHROMA_TELEMETRY_OFF,
            allow_reset=False,
        ),
    )

@cache_resource
def get_chroma_collection():
    """
    Open and cache the primary collection.
    We try to *get* first (don’t silently create a fresh, empty one),
    so that diagnostics reflect a genuinely missing collection.
    """
    client = get_chroma_client()
    embed_fn = _ChromaEmbedFn(_EMBED)
    # Try to get existing; if missing, log available names to help debugging
    try:
        col = client.get_collection(CHROMA_COLLECTION, embedding_function=embed_fn)
        return col
    except Exception:
        # Not found — list available for better logs
        with contextlib.suppress(Exception):
            names = [c.name for c in client.list_collections()]
            _log(f"⚠️ Collection '{CHROMA_COLLECTION}' not found in {CHROMA_PERSIST_DIR}. "
                 f"Available: {names}")
        # Fallback: create to avoid hard-crash, but user will see zero count
        return client.get_or_create_collection(name=CHROMA_COLLECTION, embedding_function=embed_fn)


# =============================================================================
# 7) Public API — Guideline vector search
# =============================================================================

def search_guidelines(query: str, k: int = 4) -> List[Dict[str, Any]]:
    """
    Retrieve Top-k guideline chunks via Chroma.
    Returns a list of dicts:
        { 'id': str, 'content': str, 'meta': dict, 'score': float }
    where `score` is (1 - distance) so that **higher is better**.
    Safe: returns [] if query is empty or Chroma is unavailable.
    """
    q = (query or "").strip()
    if not q:
        return []
    try:
        col = get_chroma_collection()
        res = col.query(
            query_texts=[q],
            n_results=max(1, int(k)),
            include=["documents", "metadatas", "distances", "ids"],
        )
        ids   = (res.get("ids") or [[]])[0]
        docs  = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        out: List[Dict[str, Any]] = []
        for i, _id in enumerate(ids):
            content = docs[i] if i < len(docs) else ""
            meta    = metas[i] if i < len(metas) else {}
            dist    = float(dists[i] if i < len(dists) else 0.0)
            out.append({
                "id": _id,
                "content": content,
                "meta": meta,
                "score": 1.0 - dist,
            })
        return out
    except Exception as e:
        _log("search_guidelines error:", repr(e))
        return []


# =============================================================================
# 8) Public API — Structured drug lookup (SQLite)
# =============================================================================

def _connect_sqlite(path: str) -> Optional[sqlite3.Connection]:
    if not path or not os.path.exists(path):
        return None
    try:
        con = sqlite3.connect(path)
        con.row_factory = sqlite3.Row
        return con
    except Exception as e:
        _log("sqlite connect failed:", repr(e))
        return None

def _sqlite_search_drugfacts(name_substr: str, topn: int = 5) -> List[Dict[str, Any]]:
    """
    Return lightweight text snippets from a simple `drugs` table. You can adapt
    the SQL/columns to your actual schema later.
    """
    key = (name_substr or "").strip()
    if not key:
        return []
    con = _connect_sqlite(DRUG_DB_PATH)
    if con is None:
        return []
    try:
        cur = con.cursor()
        # Adjust the table/columns below to match your schema
        cur.execute(
            "SELECT name, indications, contraindications, interactions, pregnancy, source "
            "FROM drugs WHERE name LIKE ? ORDER BY name LIMIT ?",
            (f\"%{key}%\", int(topn)),
        )
        rows = cur.fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            name, indications, contraindications, interactions, pregnancy, source = r
            text = \"\\n\".join([
                str(name or \"\"),
                f\"适应症: {indications or ''}\",
                f\"禁忌: {contraindications or ''}\",
                f\"相互作用: {interactions or ''}\",
                f\"妊娠分级: {pregnancy or ''}\",
            ]).strip()
            out.append({
                \"id\": f\"sqlite:{name}\",
                \"content\": text,
                \"meta\": {\"title\": name, \"source\": source or \"sqlite\", \"type\": \"drug\"},
                \"score\": 0.50,  # neutral-ish; will be fused by RRF if needed
            })
        return out
    finally:
        with contextlib.suppress(Exception):
            con.close()

def search_drug_structured(drug_name: str) -> Optional[Dict[str, Any]]:
    """
    Very small helper used by pipeline.py.
    Returns a single best-effort dict:
        { 'name': str, 'row': dict }  OR  None
    If your schema is different, update `_sqlite_search_drugfacts` accordingly.
    """
    key = (drug_name or "").strip()
    if not key:
        return None
    con = _connect_sqlite(DRUG_DB_PATH)
    if con is None:
        return None
    try:
        cur = con.cursor()
        cur.execute(\"SELECT * FROM drugs WHERE name LIKE ? ORDER BY name LIMIT 1\", (f\"%{key}%\",))
        row = cur.fetchone()
        if not row:
            return None
        return {\"name\": row[\"name\"] if \"name\" in row.keys() else key, \"row\": dict(row)}
    finally:
        with contextlib.suppress(Exception):
            con.close()


# =============================================================================
# 9) Hybrid search (RRF fusion) — optional, for future use
# =============================================================================

def _rrf(lists: List[List[Dict[str, Any]]], k: int, k_rrf: float = 60.0) -> List[Dict[str, Any]]:
    """
    Reciprocal Rank Fusion across multiple result lists.
    Each item must have a stable id (we synthesize one if missing).
    """
    from collections import defaultdict
    bucket: Dict[str, Dict[str, Any]] = {}
    score = defaultdict(float)
    for lst in lists:
        for rank, item in enumerate(lst):
            _id = str(item.get(\"id\") or f\"@{id(item)}\")
            bucket[_id] = item
            score[_id] += 1.0 / (k_rrf + rank + 1)
    ranked = sorted(bucket.items(), key=lambda kv: score[kv[0]], reverse=True)
    out: List[Dict[str, Any]] = []
    for _id, it in ranked[:max(1, int(k))]:
        it = dict(it)
        it[\"rrf\"] = score[_id]
        out.append(it)
    return out

def hybrid_search(query: str, k: int = 8, use_sqlite: bool = True) -> List[Dict[str, Any]]:
    """
    Combine vector guidelines and (optional) SQLite drugfacts via RRF.
    Handy for quick demos; not used by pipeline.py today.
    """
    base = search_guidelines(query, k=k)
    lists = [base]
    if use_sqlite:
        lists.append(_sqlite_search_drugfacts(query, topn=k))
    return _rrf(lists, k=k)


# =============================================================================
# 10) Diagnostics helpers (called from app.py)
# =============================================================================

def list_collections_safe(max_items: int = 100) -> List[Dict[str, Any]]:
    """
    List Chroma collections with basic stats, but never raise on error.
    Each item: {"name": str, "id": str, "count": int, "metadata": dict}
    """
    try:
        client = get_chroma_client()
        cols = client.list_collections()
        out: List[Dict[str, Any]] = []
        for c in cols[:max_items]:
            try:
                cnt = int(c.count())
            except Exception:
                cnt = -1
            out.append({
                "name": getattr(c, "name", ""),
                "id": getattr(c, "id", ""),
                "count": cnt,
                "metadata": getattr(c, "metadata", {}) or {},
            })
        return out
    except Exception as e:
        _log("list_collections_safe error:", repr(e))
        return []

def primary_collection_count() -> int:
    """
    Return number of items (chunks) in the active collection.
    - If the collection does not exist yet, returns 0 (and app can show a
      helpful message telling user to run ingestion).
    - If counting fails for any reason, returns -1.
    """
    try:
        col = get_chroma_collection()
        return int(col.count())
    except Exception as e:
        _log("primary_collection_count error:", repr(e))
        return -1


# =============================================================================
# 11) CLI smoke test
# =============================================================================

def _pretty(hits: List[Dict[str, Any]]) -> None:
    for i, h in enumerate(hits, 1):
        m = h.get("meta") or {}
        title = m.get("title") or m.get("section") or ""
        src   = m.get("source") or ""
        typ   = m.get("type") or ""
        line1 = (h.get("content") or "").strip().replace("\n", " ")
        print(f"{i:>2}. score={h.get('score',0):.3f} rrf={h.get('rrf',0):.4f} | {title} [{typ}] | {src}")
        if line1:
            print("    ", (line1[:160] + "…") if len(line1) > 160 else line1)

def main():
    import argparse
    ap = argparse.ArgumentParser(description="CareMind RAG Retriever")
    ap.add_argument("--q", "--query", dest="query", type=str, required=True)
    ap.add_argument("--topn", type=int, default=8)
    ap.add_argument("--method", type=str, default="guideline", choices=["guideline", "sqlite", "rrf"])
    ap.add_argument("--no-sqlite", action="store_true")
    args = ap.parse_args()

    _log("Version:", VERSION)
    _log("Embedding model:", EMBEDDING_MODEL)
    _log("Chroma dir:", CHROMA_PERSIST_DIR, "| collection:", CHROMA_COLLECTION)
    _log("SQLite path:", DRUG_DB_PATH)

    if args.method == "guideline":
        hits = search_guidelines(args.query, k=args.topn)
    elif args.method == "sqlite":
        hits = _sqlite_search_drugfacts(args.query, topn=args.topn)
    else:
        hits = hybrid_search(args.query, k=args.topn, use_sqlite=not args.no_sqlite)
    _pretty(hits or [])

if __name__ == "__main__":
    main()