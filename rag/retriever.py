# -*- coding: utf-8 -*-
"""
CareMind · RAG Retriever
========================

This module centralizes *all* data-access concerns for CareMind's MVP:
- Vector search over guideline chunks using ChromaDB + SentenceTransformers.
- Optional structured lookup from a small SQLite drug DB.
- Robust environment / platform compatibility (Streamlit Cloud friendly).
- Single, cached Chroma client and collection to avoid settings conflicts.
- Diagnostics helpers that your Streamlit UI can surface directly.

Why this file is long and richly commented
------------------------------------------
The intent is to leave you with a *self-documenting* reference that a future
collaborator can read without needing our chat context. Each section begins
with a short rationale and gives practical tips for debugging.
>>>>>>> b3ed42d (Make the retriever truly “single-client” & add a chunk counter API.)

Design principles
-----------------
1) **Single source of truth** for Chroma access
   Hot-reloads on Streamlit can easily create multiple Chroma clients with
   slightly different Settings (e.g., telemetry flags). We use a cached
   factory to create *one* client per process and keep Settings identical.

2) **Absolute paths & consistent Settings everywhere**
   Relative paths change when the working directory changes; absolute paths
   prevent “same directory, different identity” surprises. All Chroma Settings
   are set in exactly one place to avoid “different settings” conflicts.

3) **Backwards-compatibility**
   Managed hosts sometimes ship an older SQLite. We shim to `pysqlite3` when
   needed. We also apply a tiny pre-migration for Chroma sysdb
   (`collections.configuration`) if `_type` is missing.

4) **Return safe values**
   User-facing API (`search_guidelines`, etc.) never throws; it returns `[]`
   or `None` and logs a concise hint.

5) **Diagnose fast**
   The module prints short, timestamped messages; the UI can call
   `list_collections_safe()` and `primary_collection_count()` to show live
   health signals (“how many chunks do we actually have?”).

Environment variables (with sensible defaults)
----------------------------------------------
- CHROMA_PERSIST_DIR : "./chroma_store"
- CHROMA_COLLECTION  : "guideline_chunks"  (you set to "guideline_chunks_v2")
- EMBEDDING_MODEL    : "BAAI/bge-large-zh-v1.5"
- DRUG_DB_PATH       : "./db/drugs.sqlite"
- CHROMA_TELEMETRY_OFF : "1" disables anonymized telemetry (default)

Quick smoke tests (CLI)
-----------------------
$ python -m rag.retriever --q "哮喘 β受体阻滞剂" --topn 6
$ python rag/retriever.py --q "阿司匹林" --method sqlite
"""

from __future__ import annotations

import os
import sys
import json
import glob
import contextlib
from typing import Any, Dict, List, Optional, Tuple

# =============================================================================
# 0) Lightweight logging + path helpers
# =============================================================================

def _log(*msg: Any) -> None:
    """Single place for retriever logs with timestamps (works in Cloud/stdout)."""
    from datetime import datetime
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] retriever:", *msg, flush=True)

def _abs(p: str) -> str:
    """Expand ~ and make path absolute to avoid CWD/hot-reload ambiguity."""
    return os.path.abspath(os.path.expanduser(p))


# =============================================================================
# 1) Effective configuration (read env once; use absolute paths)
# =============================================================================

# Disable Chroma telemetry by default to keep Settings consistent and quiet.
os.environ.setdefault("CHROMA_TELEMETRY_ENABLED", "false")
os.environ.setdefault("CHROMA_ANONYMIZED_TELEMETRY", "false")
CHROMA_TELEMETRY_OFF = os.getenv("CHROMA_TELEMETRY_OFF", "1") not in ("0", "false", "False")

CHROMA_PERSIST_DIR = _abs(os.getenv("CHROMA_PERSIST_DIR", "./chroma_store"))
CHROMA_COLLECTION  = os.getenv("CHROMA_COLLECTION", "guideline_chunks")
EMBEDDING_MODEL    = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
DRUG_DB_PATH       = _abs(os.getenv("DRUG_DB_PATH", "./db/drugs.sqlite"))

VERSION = "retriever-2025-09-30"


# =============================================================================
# 2) SQLite compatibility shim (Streamlit Cloud friendly)
# =============================================================================
# Chroma (and sometimes your own SQLite helpers) require SQLite ≥ 3.35.0 for
# certain features (e.g., RETURNING). Many hosted envs ship older SQLite in
# the Python stdlib. We alias `pysqlite3` as `sqlite3` when needed.

def _ensure_sqlite() -> None:
    MIN = (3, 35, 0)
    try:
        import sqlite3 as _stdlib  # type: ignore
        try:
            ver_tuple = tuple(map(int, _stdlib.sqlite_version.split(".")))
        except Exception:
            ver_tuple = (0, 0, 0)
        if ver_tuple >= MIN:
            return  # stdlib is new enough
        # Too old → alias pysqlite3
        import pysqlite3 as _py  # type: ignore
        sys.modules["sqlite3"] = _py
    except Exception:
        try:
            import pysqlite3 as _py  # type: ignore
            sys.modules["sqlite3"] = _py
        except Exception as e:
            raise RuntimeError(
                "SQLite not available. Install `pysqlite3-binary>=0.5.3` or "
                "use Python built with SQLite >= 3.35.0."
            ) from e

_ensure_sqlite()
import sqlite3  # noqa: E402


# =============================================================================
# 3) Lazy embedding + Chroma import (we avoid heavy imports on module import)
# =============================================================================

class _LazyEmbedder:
    """
    Thin wrapper that loads a SentenceTransformer model on first call.
    We normalize embeddings so cosine ~ dot similarity; this matches common
    retrieval practice for BGE-family models.
    """
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
    """Adapter so our embedder can be passed as `embedding_function` into Chroma."""
    def __init__(self, embedder: _LazyEmbedder):
        self._embedder = embedder
    def __call__(self, inputs: List[str]) -> List[List[float]]:
        return self._embedder(inputs)

_EMBED = _LazyEmbedder(EMBEDDING_MODEL)

def _chroma_import():
    """Import Chroma on demand; gives a clear error if it isn’t available."""
    try:
        from chromadb import PersistentClient, Settings  # type: ignore
        return PersistentClient, Settings
    except Exception as e:
        raise RuntimeError(
            "Failed to import `chromadb`. Install it and ensure SQLite works.\n"
            "Tip (Cloud): include `pysqlite3-binary` in requirements."
        ) from e


# =============================================================================
# 4) Streamlit-aware caching (ensure a single client/collection per process)
# =============================================================================

try:
    import streamlit as st
    cache_resource = st.cache_resource  # persists across reruns in one process
except Exception:
    def cache_resource(func):  # no-op fallback for non-Streamlit contexts
        return func


# =============================================================================
# 5) Chroma sysdb pre-migration (fix older stores missing configuration._type)
# =============================================================================

def _sysdb_paths(persist_dir: str) -> List[str]:
    pats = [
        os.path.join(persist_dir, "chroma-*.db"),   # modern layout
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
                    payload = json.dumps({"_type":"CollectionConfigurationInternal"}, ensure_ascii=False)
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

    return total  # (typo guard: never return a misspelled variable)


# =============================================================================
# 6) Client/collection factories — *single cached instances*
# =============================================================================

@cache_resource
def get_chroma_client(persist_dir: Optional[str] = None):
    """
    Create (once) and return a PersistentClient, after a best-effort pre-migration.
    Using st.cache_resource guarantees a single instance per Streamlit process,
    preventing 'An instance of Chroma already exists …with different settings'.
    """
    pdir = _abs(persist_dir or CHROMA_PERSIST_DIR)
    os.makedirs(pdir, exist_ok=True)

    try:
        changed = _premigrate_sysdb(pdir)
        if changed:
            _log(f"Premigrated collections: {changed}")
    except Exception as e:
        _log("Premigrate skipped due to error:", repr(e))
    PersistentClient, Settings = _chroma_import()

    # IMPORTANT: keep Settings EXACTLY the same everywhere across the project.
    return PersistentClient(
        path=pdir,
        settings=Settings(
            anonymized_telemetry=not CHROMA_TELEMETRY_OFF,
            allow_reset=False,
        ),

    )
    return _CLIENT


@cache_resource
def get_chroma_collection(name: Optional[str] = None, embed_model: Optional[str] = None):
    """
    Open and cache the primary collection. We try `get_collection()` first
    (don’t silently create a brand new empty one), and if missing, we print
    the available names to guide debugging, then optionally create an empty
    collection so the app doesn’t crash.
    """
    global _COLLECTION
    if _COLLECTION is not None:
        return _COLLECTION
    client = get_chroma_client()
    embed_fn = _ChromaEmbedFn(_EMBED if not embed_model else _LazyEmbedder(embed_model))
    target = (name or CHROMA_COLLECTION)
    try:
        return client.get_collection(target, embedding_function=embed_fn)
    except Exception:
        with contextlib.suppress(Exception):
            names = [c.name for c in client.list_collections()]
            _log(f"⚠️ Collection '{target}' not found in {CHROMA_PERSIST_DIR}. Available: {names}")
        # Fallback: create (empty) to avoid hard crash — UI can show count=0.
        return client.get_or_create_collection(target, embedding_function=embed_fn)


# =============================================================================
# 7) Public API — Vector search over guideline chunks
# =============================================================================

def search_guidelines(query: str, k: int = 4) -> List[Dict[str, Any]]:
    """
    Retrieve Top-k guideline chunks via Chroma.

    Returns a list of dicts:
        {
          'id': str,
          'content': str,
          'meta': dict,
          'score': float  # 1 - distance (higher is better)
        }

    Safe behavior:
    - Empty query → [].
    - Any error → [] with a concise log (does not raise to the UI).
    """
    q = (query or "").strip()
    if not q:
        return []
    try:
        col = get_chroma_collection()
        res = col.query(
            query_texts=[q],
            n_results=max(1, int(k)),
            include=["documents", "metadatas", "distances"],
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
# 8) Optional structured lookup — small SQLite drug database
# =============================================================================
# Keep this minimal; you can evolve the schema later. We purposefully keep the
# code defensive: missing DB → None/[] without exceptions.

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
    Return lightweight text snippets from a simple `drugs` table. Adapt the
    SQL to your actual schema when needed.
    """
    key = (name_substr or "").strip()
    if not key:
        return []
    con = _connect_sqlite(DRUG_DB_PATH)
    if con is None:
        return []
    try:
        cur = con.cursor()
        cur.execute(
            "SELECT name, indications, contraindications, interactions, pregnancy, source "
            "FROM drugs WHERE name LIKE ? ORDER BY name LIMIT ?",
            (f"%{key}%", int(topn)),
        )
        rows = cur.fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            name, indications, contraindications, interactions, pregnancy, source = r
            text = "\n".join([
                str(name or ""),
                f"适应症: {indications or ''}",
                f"禁忌: {contraindications or ''}",
                f"相互作用: {interactions or ''}",
                f"妊娠分级: {pregnancy or ''}",
            ]).strip()
            out.append({
                "id": f"sqlite:{name}",
                "content": text,
                "meta": {"title": name, "source": source or "sqlite", "type": "drug"},
                "score": 0.50,  # neutral-ish; if you fuse results later, this plays well
            })
        return out
    finally:
        with contextlib.suppress(Exception):
            con.close()

def search_drug_structured(drug_name: str) -> Optional[Dict[str, Any]]:
    """
    A small helper used by pipeline.py. Returns one best row (or None):
        { 'name': str, 'row': dict }
    """
    key = (drug_name or "").strip()
    if not key:
        return None
    con = _connect_sqlite(DRUG_DB_PATH)
    if con is None:
        return None
    try:
        cur = con.cursor()
        cur.execute("SELECT * FROM drugs WHERE name LIKE ? ORDER BY name LIMIT 1", (f"%{key}%",))
        row = cur.fetchone()
        if not row:
            return None
        return {"name": row["name"] if "name" in row.keys() else key, "row": dict(row)}
    finally:
        with contextlib.suppress(Exception):
            con.close()


# =============================================================================
# 9) Simple hybrid (RRF) — optional helper for demos / future fusion
# =============================================================================
def _rrf(lists: List[List[Dict[str, Any]]], k: int, k_rrf: float = 60.0) -> List[Dict[str, Any]]:
    """Reciprocal Rank Fusion across multiple ranked lists."""
    from collections import defaultdict
    bucket: Dict[str, Dict[str, Any]] = {}
    score = defaultdict(float)
    for lst in lists:
        for rank, item in enumerate(lst):
            _id = str(item.get("id") or f"@{id(item)}")
            bucket[_id] = item
            score[_id] += 1.0 / (k_rrf + rank + 1)
    ranked = sorted(bucket.items(), key=lambda kv: score[kv[0]], reverse=True)
    out: List[Dict[str, Any]] = []
    for _id, it in ranked[:max(1, int(k))]:
        it = dict(it)
        it["rrf"] = score[_id]
        out.append(it)
    return out

def hybrid_search(query: str, k: int = 8, use_sqlite: bool = True) -> List[Dict[str, Any]]:
    """Fuse guideline vector results with optional SQLite snippets."""
    base = search_guidelines(query, k=k)
    lists = [base]
    if use_sqlite:
        lists.append(_sqlite_search_drugfacts(query, topn=k))
    return _rrf(lists, k=k)


# =============================================================================
# 10) Diagnostics helpers — Streamlit UI can call these directly
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

# ---- Diagnostics helpers (called from app.py) ----
from typing import Any, Dict, List

def list_collections_safe(max_items: int = 100) -> List[Dict[str, Any]]:
    """
    List Chroma collections with basic stats, *never* raising an exception.
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
    Return the number of items (chunks) in the active collection.
    0  → collection exists but is empty or just created.
    -1 → a counting error occurred (the UI can render this as a warning).
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
        print(f"{i:>2}. score={h.get('score',0):.3f} | {title} [{typ}] | {src}")
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