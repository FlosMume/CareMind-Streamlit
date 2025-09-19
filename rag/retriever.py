# -*- coding: utf-8 -*-
"""
CareMind | Retrieval helpers (Chroma + SQLite)
- Streamlit Cloud friendly: modern SQLite shim, lazy chroma import.
- Public API:
    search_guidelines(query: str, k: int = 6) -> list[dict]
    search_drugs(query: str, k: int = 6) -> list[dict]
    search_drug_structured(drug_name: str) -> dict | None
"""

from __future__ import annotations
import os
import sys
from typing import Any, Dict, List, Optional

# -----------------------------------------------------------------------------
# Environment / defaults (kept compatible with your previous code)
# -----------------------------------------------------------------------------
CHROMA_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")
PERSIST_DIR: str = CHROMA_DIR  # alias
COLLECTION: str = os.getenv("CHROMA_COLLECTION", "guideline_chunks_1024_v2")
SQLITE_PATH: str = os.getenv("SQLITE_PATH", "./db/drugs.sqlite")
EMBED_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
DEMO: bool = os.getenv("CAREMIND_DEMO", "1") == "1"  # default ON in Cloud

# -----------------------------------------------------------------------------
# Ensure a modern SQLite on hosts with old stdlib sqlite3 (Cloud fix)
# -----------------------------------------------------------------------------
def _alias_sqlite_if_needed() -> None:
    try:
        import sqlite3  # noqa: F401
        # Quick feature test: FTS5 exists from SQLite 3.9+, Chroma wants >=3.35
        # We still alias proactively if pysqlite3 is available.
        pass
    except Exception:
        pass
    try:
        import pysqlite3  # provided by pysqlite3-binary
        sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
    except Exception:
        # If not available, we proceed; Chroma may later raise a clear error.
        pass

_alias_sqlite_if_needed()
import sqlite3  # now safe to import

# -----------------------------------------------------------------------------
# Lazy chroma import (so app boots even if Chroma/SQLite not ready)
# -----------------------------------------------------------------------------
def _chroma():
    """
    Import chromadb only when we actually need it.
    Returns (PersistentClient, embedding_functions)
    """
    from chromadb import PersistentClient
    from chromadb.utils import embedding_functions
    return PersistentClient, embedding_functions

def _get_embed_fn(model_name: str = EMBED_MODEL):
    """
    Returns a SentenceTransformerEmbeddingFunction (lazy load).
    """
    _, embedding_functions = _chroma()
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)

def _get_client() -> "PersistentClient":
    PersistentClient, _ = _chroma()
    if not os.path.isdir(PERSIST_DIR):
        raise FileNotFoundError(
            f"Chroma dir not found: {PERSIST_DIR}\n"
            "请先构建向量库（运行你的向量化脚本）或设置 CHROMA_PERSIST_DIR。"
        )
    return PersistentClient(path=PERSIST_DIR)

def _pick_collection(client: "PersistentClient", preferred: str, embed_fn) -> Any:
    """
    选择可用集合：
      1) preferred 存在且非空 → 用它
      2) 否则扫描非空集合 → 用第一个
      3) 否则抛错（或在 DEMO 下返回 None）
    """
    def bind(name: str):
        try:
            return client.get_collection(name=name, embedding_function=embed_fn)
        except Exception:
            return None

    # Try preferred
    col = bind(preferred)
    if col:
        try:
            if col.count() > 0:
                return col
        except Exception:
            # Old chroma may fail count(); try a minimal query
            try:
                res = col.query(query_texts=["."], n_results=1)
                if res and res.get("ids") and res["ids"][0]:
                    return col
            except Exception:
                pass

    # Try any non-empty collection
    try:
        for c in client.list_collections():
            col = bind(c.name)
            if not col:
                continue
            try:
                if col.count() > 0:
                    return col
            except Exception:
                try:
                    res = col.query(query_texts=["."], n_results=1)
                    if res and res.get("ids") and res["ids"][0]:
                        return col
                except Exception:
                    continue
    except Exception:
        pass

    if DEMO:
        # In demo mode, returning None allows the app to render without hard failing.
        return None

    raise RuntimeError(
        "未找到可用的 Chroma 集合（所有集合不存在或为空）。"
        f"\n目录: {PERSIST_DIR} | preferred={preferred}\n"
        "请先把 ./data/guidelines.parsed.jsonl 写入向量库，或检查 CHROMA_PERSIST_DIR。"
    )

# -----------------------------------------------------------------------------
# Guideline search (Chroma)
# -----------------------------------------------------------------------------
def search_guidelines(query: str, k: int = 6, where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    语义检索（Chroma）
    Returns: list of dicts with keys: id, content, meta, score, source
    """
    try:
        client = _get_client()
        embed_fn = _get_embed_fn(EMBED_MODEL)
        col = _pick_collection(client, COLLECTION, embed_fn)
        if col is None:
            # Demo fallback: no Chroma available
            return []
        kwargs = dict(query_texts=[query], n_results=int(k), include=["documents", "metadatas", "distances"])
        if where:
            kwargs["where"] = where  # don't pass empty {}
        res = col.query(**kwargs)
        ids   = res.get("ids", [[]])[0]
        docs  = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]

        out: List[Dict[str, Any]] = []
        for i in range(len(ids)):
            distance = float(dists[i]) if dists else 1.0
            sim = max(0.0, min(1.0, 1.0 - distance))
            out.append({
                "id": ids[i],
                "content": docs[i],
                "meta": metas[i],
                "score": sim,
                "source": "guideline",
            })
        return out
    except Exception:
        if DEMO:
            # Quietly fallback in demo mode
            return []
        raise

# -----------------------------------------------------------------------------
# SQLite helpers (structured drug info)
# -----------------------------------------------------------------------------
def _connect_sqlite() -> sqlite3.Connection:
    if not os.path.isfile(SQLITE_PATH):
        # In demo mode we tolerate missing DB and just return a placeholder
        if DEMO:
            # Return a connection to an in-memory DB with no tables to avoid crashes
            con = sqlite3.connect(":memory:")
            con.row_factory = sqlite3.Row
            return con
        raise FileNotFoundError(
            f"SQLite DB not found: {SQLITE_PATH}\n"
            "请先运行 ingest/load_drugs.py 以生成 db/drugs.sqlite，或设置 SQLITE_PATH。"
        )
    con = sqlite3.connect(SQLITE_PATH)
    con.row_factory = sqlite3.Row
    return con

def _has_fts(con: sqlite3.Connection) -> bool:
    cur = con.cursor()
    cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='drugs_fts'")
    return cur.fetchone() is not None

def _trim(text: Optional[str], n: int = 120) -> str:
    if not text:
        return "-"
    t = str(text).replace("\n", " ").strip()
    return (t[:n] + "…") if len(t) > n else t

# -----------------------------------------------------------------------------
# Drug search (full-text list, as you had)
# -----------------------------------------------------------------------------
def search_drugs(query: str, k: int = 6) -> List[Dict[str, Any]]:
    con = _connect_sqlite()
    try:
        if _has_fts(con):
            sql = """
                SELECT d.*, bm25(drugs_fts) AS rank
                FROM drugs_fts JOIN drugs d ON d.rowid = drugs_fts.rowid
                WHERE drugs_fts MATCH ?
                ORDER BY rank ASC
                LIMIT ?
            """
            cur = con.cursor()
            cur.execute(sql, (query, k))
            rows = cur.fetchall()
        else:
            cur = con.cursor()
            kw = f"%{query}%"
            sql = """
                SELECT *
                FROM drugs
                WHERE name LIKE ? OR generic_name LIKE ?
                   OR indications LIKE ? OR contraindications LIKE ?
                   OR interactions LIKE ?
                LIMIT ?
            """
            cur.execute(sql, (kw, kw, kw, kw, kw, k))
            rows = cur.fetchall()

        out: List[Dict[str, Any]] = []
        for r in rows:
            meta = {c: r[c] for c in r.keys()}
            fields = ["name","generic_name","indications","contraindications","interactions"]
            hay = " ".join((meta.get(f) or "") for f in fields)
            hits = sum(1 for f in fields if meta.get(f) and (query in (meta.get(f) or "")))
            sim = max(0.3, min(1.0, (hits + (1 if query in hay else 0)) / (len(fields) + 1)))
            content = f"{meta.get('name','?')}（{meta.get('generic_name','-')}）\n适应症: {_trim(meta.get('indications'))}"
            out.append({
                "id": f"drug:{meta.get('id','')}",
                "content": content,
                "meta": meta,
                "score": sim,
                "source": "drug",
            })
        return out
    finally:
        try:
            con.close()
        except Exception:
            pass

# -----------------------------------------------------------------------------
# Single-drug structured lookup (for pipeline.answer)
# -----------------------------------------------------------------------------
def search_drug_structured(drug_name: str) -> Optional[Dict[str, Any]]:
    """
    Returns a single structured record for the given drug name (best match),
    or None if not found / DB unavailable in DEMO.
    """
    if not (drug_name and drug_name.strip()):
        return None

    con = _connect_sqlite()
    try:
        cur = con.cursor()
        # Try exact-ish first
        cur.execute(
            "SELECT * FROM drugs WHERE name = ? OR generic_name = ? LIMIT 1",
            (drug_name, drug_name),
        )
        row = cur.fetchone()
        if not row:
            # Fallback: LIKE search, pick the first
            kw = f"%{drug_name}%"
            cur.execute(
                """SELECT * FROM drugs
                   WHERE name LIKE ? OR generic_name LIKE ?
                   ORDER BY LENGTH(name) ASC
                   LIMIT 1""",
                (kw, kw),
            )
            row = cur.fetchone()
        return {c: row[c]} if row and False else ({c: row[c] for c in row.keys()} if row else None)
    except Exception:
        if DEMO:
            return None
        raise
    finally:
        try:
            con.close()
        except Exception:
            pass
