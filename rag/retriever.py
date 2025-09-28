# -*- coding: utf-8 -*-
"""

retriever.py — CareMind RAG 检索层（更强自愈版：全库预迁移 + 失败重试）

变更要点（相对上一版）：
- 在创建 Chroma 客户端之前，执行一次“全库预迁移”：
  * 打开 sysdb sqlite，扫描 collections 表，对所有 configuration 为 NULL/空/无"_type" 的行
    写入最小 JSON: {"_type": "CollectionConfigurationInternal"}。
- get_or_create_collection() 失败时（不论异常类型），再次执行全库迁移并重试一次。
- 其余功能保持：惰性嵌入、指南检索、SQLite 检索、RRF 融合、CLI 自测。
"""
from __future__ import annotations
import os, sys, time, json, glob, shutil, datetime, contextlib
from typing import Any, Dict, List, Optional, Tuple, TypedDict

# --- 把 pysqlite3-binary 映射为 sqlite3（云端常见做法） ---
try:
    import pysqlite3 as sqlite3  # type: ignore
    sys.modules["sqlite3"] = sqlite3
except Exception:
    import sqlite3  # type: ignore


# --- 关闭 Chroma 遥测 ---
os.environ.setdefault("CHROMA_TELEMETRY_ENABLED", "false")
os.environ.setdefault("CHROMA_ANONYMIZED_TELEMETRY", "false")

# --- 环境配置 ---
PERSIST_DIR      = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")
COLLECTION_NAME  = os.getenv("CHROMA_COLLECTION", "guideline_chunks")
EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
SQLITE_PATH      = os.getenv("DRUG_DB_PATH", "./db/drugs.sqlite")

# --- 版本标签（诊断展示） ---
__RETRIEVER_VERSION__ = "retriever-2025-09-21q"

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

def retriever_version() -> str:
    """返回检索器版本号（供诊断面板显示）。"""
    return __RETRIEVER_VERSION__

def _log(*msg: Any) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] retriever:", *msg, flush=True)


# ---------------------------
# 嵌入：SentenceTransformers（惰性加载）
# ---------------------------
class _LazyEmbedder:
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
    """适配 chromadb 的 embedding_function 调用签名。"""
    def __init__(self, embedder: _LazyEmbedder):
        self._embedder = embedder
    def __call__(self, input: List[str]) -> List[List[float]]:
        return self._embedder(input)

_EMBED = _LazyEmbedder(EMBEDDING_MODEL)

# ---------------------------
# sysdb 定位 & 迁移
# ---------------------------
def _find_sysdb_sqlite_file(persist_dir: str) -> Optional[str]:
    cand = [
        os.path.join(persist_dir, "chroma.sqlite3"),
        os.path.join(persist_dir, "chroma.sqlite"),
        os.path.join(persist_dir, "chroma.db"),
    ]
    for p in cand:
        if os.path.isfile(p):
            return p
    globs = glob.glob(os.path.join(persist_dir, "*.sqlite*"))
    if globs:
        globs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return globs[0]
    return None

def _config_has_type(config_text: Optional[str]) -> bool:
    if not config_text:
        return False
    try:
        obj = json.loads(config_text)
        return isinstance(obj, dict) and "_type" in obj
    except Exception:
        return False

def _migrate_all_collections(persist_dir: str) -> int:
    """
    对 sysdb 的 collections 表做“全库预迁移”：
    - 若 configuration 为 NULL/空/无"_type"，写入 {"_type":"CollectionConfigurationInternal"}
    返回：修复的行数
    """
    dbfile = _find_sysdb_sqlite_file(persist_dir)
    if not dbfile:
        _log("No sysdb sqlite found in", persist_dir, "— skip migrate.")
        return 0

    _log("Pre-migrate sysdb:", dbfile)
    fixed = 0
    try:
        con = sqlite3.connect(dbfile)
        cur = con.cursor()
        cur.execute("SELECT id, name, configuration FROM collections")
        rows = cur.fetchall()
        for cid, name, conf in rows:
            if not _config_has_type(conf):
                conf_json = json.dumps({"_type":"CollectionConfigurationInternal"}, ensure_ascii=False)
                cur.execute("UPDATE collections SET configuration = ? WHERE id = ?", (conf_json, cid))
                fixed += 1
        if fixed:
            con.commit()
            _log(f"Collections patched (missing _type): {fixed}")
        else:
            _log("No collection needs patch.")
    except Exception as e:
        _log("Migrate failed:", repr(e))
    finally:
        with contextlib.suppress(Exception):
            con.close()
    return fixed

# ---------------------------
# Chroma 客户端 & 集合
# ---------------------------
_chroma_client = None
_collection = None

def get_chroma_client():
    """创建 Chroma PersistentClient（带全库预迁移）。"""
    global _chroma_client
    if _chroma_client is not None:
        return _chroma_client

    # 先保证目录存在
    os.makedirs(PERSIST_DIR, exist_ok=True)

    # **关键：在创建客户端之前做一次全库预迁移**
    _migrate_all_collections(PERSIST_DIR)

    import chromadb
    from chromadb import PersistentClient
    _log("Chroma version:", getattr(chromadb, "__version__", "unknown"))
    _log("CHROMA_PERSIST_DIR:", PERSIST_DIR)
    _log("CHROMA_COLLECTION:", COLLECTION_NAME)

    _chroma_client = PersistentClient(path=PERSIST_DIR)
    return _chroma_client

def get_chroma_collection():
    """
    获取（或创建）指定集合；
    - 失败时（不论异常类型），再次全库迁移并重试一次。
    """
    global _collection, _chroma_client
    if _collection is not None:
        return _collection

    client = get_chroma_client()
    embed_fn = _ChromaEmbedFn(_EMBED)
    target = COLLECTION_NAME

    try:
        _collection = client.get_or_create_collection(name=target, embedding_function=embed_fn)
        return _collection
    except Exception as e:

        _log("get_or_create_collection error (first try):", repr(e))
        _log("Retry after migrating all collections...")
        _migrate_all_collections(PERSIST_DIR)
        # 再试一次
        _collection = client.get_or_create_collection(name=target, embedding_function=embed_fn)
        _log("Collection opened after migrate+retry.")
        return _collection

# ---------------------------
# 指南向量检索
# ---------------------------
def search_guidelines(query: str, k: int = 5, include_metadata: bool = True) -> List[Dict[str, Any]]:

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
        n_results=k,
        include=["documents", "metadatas", "distances"] if include_metadata else ["documents"]
    )
    out: List[Dict[str, Any]] = []
    ids   = (res.get("ids") or [[]])[0]
    docs  = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    for i, _id in enumerate(ids):
        item = {
            "id": _id,
            "text": docs[i] if i < len(docs) else None,
            "meta": metas[i] if i < len(metas) else {},
            "score": 1.0 - (dists[i] if i < len(dists) else 0.0)
        }
        out.append(item)
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
# ---------------------------
# （可选）SQLite 药品库检索
# ---------------------------
def _sqlite_search_drugfacts(q: str, topn: int = 5) -> List[Dict[str, Any]]:
    if not os.path.isfile(SQLITE_PATH):
        return []
    rows: List[Tuple] = []
    try:
        cur.execute("SELECT * FROM drugs WHERE name LIKE ? LIMIT ?", (f"%{name_substr}%", int(limit)))
        rows = [dict(r) for r in cur.fetchall()]
    finally:
        with contextlib.suppress(Exception):
            con.close()

    hits: List[Dict[str, Any]] = []
    for r in rows:
        name, indications, contraindications, interactions, pregnancy, source = r
        hits.append({
            "id": f"sqlite:{name}",
            "text": f"{name}\n适应症: {indications}\n禁忌: {contraindications}\n相互作用: {interactions}\n妊娠分级: {pregnancy}",
            "meta": {"title": name, "source": source or "sqlite", "type": "drug"},
            "score": 0.5,
        })
    return hits


# ---------------------------
# 混合检索（RRF）
# ---------------------------
def _rrf_rank(lists: List[List[Dict[str, Any]]], k: int, k_rrf: float = 60.0) -> List[Dict[str, Any]]:
    from collections import defaultdict
    score = defaultdict(float)
    bag: Dict[str, Dict[str, Any]] = {}
    for lst in lists:
        for rank, item in enumerate(lst):
            _id = str(item.get("id") or f"@{id(item)}")
            score[_id] += 1.0 / (k_rrf + rank + 1)
            if _id not in bag:
                bag[_id] = item
    merged = []
    for _id, s in score.items():
        it = dict(bag[_id]); it["rrf"] = s
        merged.append(it)
    merged.sort(key=lambda x: x.get("rrf", 0.0), reverse=True)
    return merged[:k]

def hybrid_search(query: str,
                  k: int = 8,
                  k_guideline: Optional[int] = None,
                  k_sqlite: Optional[int] = None,
                  use_sqlite: bool = True) -> List[Dict[str, Any]]:
    k_guideline = k_guideline or max(3, k)
    k_sqlite = k_sqlite or max(3, k // 2)
    g_hits = search_guidelines(query, k=k_guideline)
    s_hits: List[Dict[str, Any]] = _sqlite_search_drugfacts(query, topn=k_sqlite) if use_sqlite else []
    if g_hits and s_hits:
        return _rrf_rank([g_hits, s_hits], k=k)
    return (g_hits or s_hits)[:k]

# ---------------------------
# CLI 自测
# ---------------------------
def _pretty(hits: List[Dict[str, Any]]) -> None:
    for i, h in enumerate(hits, 1):
        m = h.get("meta") or {}
        title = m.get("title") or m.get("section") or ""
        src   = m.get("source") or ""
        year  = m.get("year") or ""
        print(f"{i:>2}. score={h.get('score'):.3f} rrf={h.get('rrf', 0):.4f} | {title} | {src} | {year}")
        txt = (h.get("text") or "").strip().replace("\n", " ")
        print("    ", (txt[:160] + "…") if len(txt) > 160 else txt)

def main():
    import argparse
    ap = argparse.ArgumentParser(description="CareMind RAG Retriever (pre-migrate & retry)")
    ap.add_argument("--q", "--query", dest="query", type=str, required=True)
    ap.add_argument("--topn", type=int, default=8)
    ap.add_argument("--method", type=str, default="rrf", choices=["guideline", "sqlite", "rrf"])
    ap.add_argument("--no-sqlite", action="store_true")
    args = ap.parse_args()

    _log("Embedding model:", EMBEDDING_MODEL)
    _log("Chroma dir:", PERSIST_DIR, "| collection:", COLLECTION_NAME)
    _log("SQLite path:", SQLITE_PATH)

    if args.method == "guideline":
        hits = search_guidelines(args.query, k=args.topn)
    elif args.method == "sqlite":
        hits = _sqlite_search_drugfacts(args.query, topn=args.topn)
    else:
        hits = hybrid_search(args.query, k=args.topn, use_sqlite=not args.no_sqlite)
    _pretty(hits or [])

# =============================================================================
# __main__ (CLI smoke test)
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    main()