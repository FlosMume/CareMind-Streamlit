# -*- coding: utf-8 -*-
"""
retriever.py | CareMind (updated 2025-09-21)
-------------------------------------------
• Purpose
  1) Retrieve guideline chunks from a Chroma vector DB
  2) Look up structured drug info from SQLite

• Key updates in this revision
  - Robust, CONDITIONAL SQLite shim: only alias pysqlite3 → sqlite3 if the
    stdlib sqlite3 is missing or too old (< 3.35.0). This avoids unnecessary
    overrides on modern local setups, while fixing Streamlit Cloud / older OS
    images that ship an older SQLite.
  - Still lazy-loads Chroma and disables anonymized telemetry by default.
  - Safer collection discovery fallback; clearer diagnostics.
  - Non‑breaking: all prior functions remain and signatures are unchanged.

Tip: Set these env vars (or Streamlit secrets) if needed:
  CHROMA_PERSIST_DIR, CHROMA_COLLECTION, EMBEDDING_MODEL, DRUG_DB_PATH,
  CAREMIND_DEMO, CHROMA_ANONYMIZED_TELEMETRY
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

# Version tag to verify deployment picked up the latest file
VERSION = "retriever-2025-09-21c"

# =============================================================================
# 0) Conditional SQLite compatibility shim (Cloud friendly)
# -----------------------------------------------------------------------------
# Many managed environments ship an old libsqlite3 (<3.35.0) which breaks Chroma
# (it relies on modern features). We try stdlib sqlite3 first; if it's missing
# or too old, we swap in pysqlite3-binary and alias it as sqlite3.

def _ensure_sqlite() -> None:
    MIN = (3, 35, 0)
    try:
        import sqlite3 as _stdlib
        try:
            ver_tuple = tuple(map(int, _stdlib.sqlite_version.split(".")))
        except Exception:
            ver_tuple = (0, 0, 0)
        if ver_tuple >= MIN:
            # Good enough; keep stdlib
            return
        # Too old → try pysqlite3
        import pysqlite3 as _py
        sys.modules["sqlite3"] = _py  # alias
    except Exception:
        # stdlib missing or unusable → try pysqlite3 as last resort
        try:
            import pysqlite3 as _py
            sys.modules["sqlite3"] = _py
        except Exception as e:
            raise RuntimeError(
                "Neither a new-enough stdlib sqlite3 nor pysqlite3-binary is available.\n"
                "Install `pysqlite3-binary>=0.5.3` (prefer 0.5.4) or use a Python build\n"
                "that bundles SQLite >= 3.35.0."
            ) from e

_ensure_sqlite()
import sqlite3  # after shim


# =============================================================================
# 1) Secrets-aware env helpers
# -----------------------------------------------------------------------------

def _env(key: str, default: str | None = None) -> str | None:
    # Prefer Streamlit secrets when available, fall back to os.environ
    try:
        import streamlit as st  # type: ignore
        return os.getenv(key, st.secrets.get(key, default))
    except Exception:
        return os.getenv(key, default)

def _as_bool(val: str | None, default: bool = False) -> bool:
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "on"}


# =============================================================================
# 2) Environment values & defaults
# -----------------------------------------------------------------------------
CHROMA_PERSIST_DIR: str = _env("CHROMA_PERSIST_DIR", "./chroma_store") or "./chroma_store"
CHROMA_COLLECTION: str  = _env("CHROMA_COLLECTION",  "guideline_chunks") or "guideline_chunks"
EMBED_MODEL: str        = _env("EMBEDDING_MODEL",    "sentence-transformers/all-MiniLM-L6-v2") \
                          or "sentence-transformers/all-MiniLM-L6-v2"
DRUG_DB_PATH: str       = _env("DRUG_DB_PATH",       "./db/drugs.sqlite") or "./db/drugs.sqlite"
DEMO: bool              = _as_bool(_env("CAREMIND_DEMO", "1"), default=True)
CHROMA_TELEMETRY_OFF: bool = not _as_bool(_env("CHROMA_ANONYMIZED_TELEMETRY", "False"), default=False)


# =============================================================================
# 3) Lazy Chroma import + simple client/collection caches
# -----------------------------------------------------------------------------
_CLIENT = None
_COLLECTION = None


def _chroma():
    # Import inside function to avoid crashing on module import when chroma
    # deps are not fully ready (esp. on cloud cold start).
    try:
        from chromadb import PersistentClient, Settings  # type: ignore
        from chromadb.utils import embedding_functions   # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Failed to import Chroma. Ensure `chromadb` is installed and that\n"
            "SQLite is available (we auto-shim pysqlite3-binary when needed)."
        ) from e
    return PersistentClient, embedding_functions, Settings


def clear_chroma_cache() -> None:
    """For debugging: clear in-process client/collection caches."""
    global _CLIENT, _COLLECTION
    _CLIENT = None
    _COLLECTION = None


def get_chroma_client(persist_dir: Optional[str] = None):
    """Get (and cache) a Chroma client."""
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    PersistentClient, _, Settings = _chroma()
    _CLIENT = PersistentClient(
        path=(persist_dir or CHROMA_PERSIST_DIR),
        settings=Settings(
            anonymized_telemetry=not CHROMA_TELEMETRY_OFF,
            allow_reset=True,
        ),
    )
    return _CLIENT


def _preferred_collection_name(client) -> Optional[str]:
    """Pick a collection name, preferring names containing 'guideline'."""
    try:
        names = [getattr(c, "name", None) for c in client.list_collections()]
        names = [n for n in names if n]
        for n in names:
            if "guideline" in n.lower():
                return n
        return names[0] if names else None
    except Exception:
        return None


def get_chroma_collection(name: Optional[str] = None, embed_model: Optional[str] = None):
    """Get (and cache) a Chroma collection; will try preferred fallback names."""
    global _COLLECTION
    if _COLLECTION is not None:
        return _COLLECTION

    _, embedding_functions, _ = _chroma()
    client = get_chroma_client()
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=(embed_model or EMBED_MODEL)
    )

    target = name or CHROMA_COLLECTION
    try:
        # Try existing collection first
        _COLLECTION = client.get_collection(name=target, embedding_function=embed_fn)
        return _COLLECTION
    except Exception:
        # Fallback: pick by heuristic or create if nothing exists
        pick = _preferred_collection_name(client)
        if pick:
            _COLLECTION = client.get_collection(name=pick, embedding_function=embed_fn)
            return _COLLECTION
        # As final resort, create the configured target
        _COLLECTION = client.get_or_create_collection(name=target, embedding_function=embed_fn)
        return _COLLECTION


# =============================================================================
# 4) Guideline search (Chroma)
# -----------------------------------------------------------------------------

def search_guidelines(query: str, k: int = 4) -> List[Dict[str, Any]]:
    """
    Retrieve top-k guideline chunks.
    Returns: list of { 'content': str, 'meta': dict, 'score': float }
    Note: `k` must be >=1; we include distances as scores (smaller is closer in Chroma).
    """
    if not query:
        return []
    col = get_chroma_collection()
    res = col.query(
        query_texts=[query],
        n_results=max(1, int(k)),
        include=["documents", "metadatas", "distances"],
    )
    docs = (res or {}).get("documents") or [[]]
    metas = (res or {}).get("metadatas") or [[]]
    dists = (res or {}).get("distances") or [[]]

    out: List[Dict[str, Any]] = []
    for i, doc in enumerate(docs[0]):
        meta = (metas[0][i] if i < len(metas[0]) else {}) or {}
        score = (dists[0][i] if i < len(dists[0]) else None)
        out.append({"content": doc, "meta": meta, "score": score})
    return out


# =============================================================================
# 5) Structured drug lookup (SQLite)
# -----------------------------------------------------------------------------

def _connect_sqlite(path: str) -> sqlite3.Connection:
    if not os.path.exists(path):
        if DEMO:
            return sqlite3.connect(":memory:")
        raise FileNotFoundError(path)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    return con


def search_drug_structured(name_substr: str, limit: int = 10) -> List[Dict[str, Any]]:
    if not name_substr:
        return []
    con = _connect_sqlite(DRUG_DB_PATH)
    cur = con.cursor()
    try:
        cur.execute("SELECT * FROM drugs WHERE name LIKE ? LIMIT ?", (f"%{name_substr}%", int(limit)))
        rows = [dict(r) for r in cur.fetchall()]
    finally:
        try:
            con.close()
        except Exception:
            pass
    return rows


# =============================================================================
# 6) Diagnostics: list collections (safe) & environment info
# -----------------------------------------------------------------------------

def _fallback_collections_from_sqlite_dir(dir_path: str) -> List[str]:
    """If Chroma API listing fails, scan the on-disk SQLite to read collection names."""
    try:
        import glob, os as _os
        candidates = []
        p1 = _os.path.join(dir_path, "chroma.sqlite3")
        if _os.path.exists(p1):
            candidates.append(p1)
        candidates.extend(glob.glob(_os.path.join(dir_path, "*.sqlite*")))

        names, seen = [], set()
        for fp in candidates:
            con = None
            try:
                con = sqlite3.connect(fp)
                cur = con.cursor()
                cur.execute("SELECT name FROM collections")
                for (nm,) in cur.fetchall():
                    if nm and nm not in seen:
                        seen.add(nm); names.append(nm)
            except Exception:
                pass
            finally:
                try:
                    con and con.close()
                except Exception:
                    pass
            if names:
                break
        return names
    except Exception:
        return []


def list_collections_safe() -> List[Dict[str, Any]]:
    """
    Prefer Chroma API; fall back to direct SQLite scan.
    Returns: [{"name": str, "count": int|"?"}] or [{"error": str}]
    """
    try:
        client = get_chroma_client()
        out: List[Dict[str, Any]] = []
        for c in client.list_collections():
            name = getattr(c, "name", None) or "?"
            try:
                count = int(c.count())
            except Exception:
                count = "?"
            out.append({"name": name, "count": count})
        if out:
            return out
    except Exception:
        pass

    # Fallback scan
    try:
        names = _fallback_collections_from_sqlite_dir(CHROMA_PERSIST_DIR)
        if names:
            return [{"name": n, "count": "?"} for n in names]
        return [{"error": "no collections found"}]
    except Exception as e:
        return [{"error": str(e)}]


def environment_summary() -> Dict[str, Any]:
    """Quick diagnostic block you can print/log in app.py panels."""
    try:
        import sqlite3 as _sq
        sql_ver = getattr(_sq, "sqlite_version", "?")
    except Exception:
        sql_ver = "?"
    return {
        "module_version": VERSION,
        "chroma_dir": CHROMA_PERSIST_DIR,
        "chroma_collection": CHROMA_COLLECTION,
        "embed_model": EMBED_MODEL,
        "drug_db": DRUG_DB_PATH,
        "demo_mode": DEMO,
        "sqlite_version": sql_ver,
        "telemetry_off": CHROMA_TELEMETRY_OFF,
    }


# =============================================================================
# __main__ (CLI smoke test)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import json, argparse

    p = argparse.ArgumentParser(description="CareMind retriever smoke test")
    p.add_argument("--q", dest="query", type=str, default="高血压 治疗 指南")
    p.add_argument("--k", dest="k", type=int, default=4)
    p.add_argument("--list", dest="do_list", action="store_true")
    args = p.parse_args()

    print("ENV:")
    print(json.dumps(environment_summary(), ensure_ascii=False, indent=2))

    if args.do_list:
        print("\nCollections:")
        print(json.dumps(list_collections_safe(), ensure_ascii=False, indent=2))

    if args.query:
        print("\nSearch results:")
        hits = search_guidelines(args.query, k=args.k)
        for i, h in enumerate(hits, 1):
            m = h.get("meta") or {}
            t = (m.get("title") or m.get("source") or "?")
            print(f"[{i}] {t} | score={h.get('score')}")
