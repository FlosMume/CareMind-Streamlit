# -*- coding: utf-8 -*-
"""
retriever.py | CareMind
-----------------------
职责 / Responsibilities
1) 在 Chroma 向量库中检索指南片段
   Retrieve guideline text chunks from a Chroma vector DB.
2) 在 SQLite 中查询药品结构化信息
   Look up structured drug info in SQLite.

关键设计 / Key design choices
- ✅ Cloud 兼容：把 pysqlite3-binary 别名为 sqlite3，规避旧版 sqlite3 导致的 Chroma 报错。
  Cloud-compat: alias pysqlite3-binary → sqlite3 to satisfy Chroma's sqlite ≥3.35.
- ✅ 惰性导入 Chroma：只在函数调用时导入，防止模块导入阶段崩溃。
  Lazy import chroma so the module never crashes during import.
- ✅ 关闭 Chroma 遥测：通过 chromadb.config.Settings(anonymized_telemetry=False)。
  Turn off anonymized telemetry by default.
- ✅ Secrets 优先：通过 _env() 读取配置（先 Secrets，再环境变量，最后默认）。
  Secrets-first config via _env() (Secrets → env → default).
- ✅ 安全的集合枚举：list_collections_safe() 仅返回 {name, count}，避免 _type 等序列化问题。
  Safe collection listing that avoids serializing Chroma internals like `_type`.
- ✅ 连接/集合缓存：get_chroma_client / get_chroma_collection 做简单缓存，减少反复打开。
  Simple in-process cache for client/collection.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

# 模块版本号（用于诊断显示云端是否更新）
VERSION = "retriever-2025-09-21b"

# =============================================================================
# 0) SQLite 兼容补丁（Cloud）/ SQLite compatibility shim (Cloud)
# -----------------------------------------------------------------------------
# 优先使用 pysqlite3-binary，避免云端缺失 _sqlite3 动态库导致 Chroma 初始化失败
try:
    import pysqlite3  # type: ignore
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    # 本地通常自带 sqlite3，忽略即可
    pass

import sqlite3  # after aliasing


# =============================================================================
# 1) Secrets-aware env helpers / 读取配置优先 Secrets
# -----------------------------------------------------------------------------
def _env(key: str, default: str | None = None) -> str | None:
    try:
        import streamlit as st
        return os.getenv(key, st.secrets.get(key, default))
    except Exception:
        return os.getenv(key, default)

def _as_bool(val: str | None, default: bool = False) -> bool:
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "on"}


# =============================================================================
# 2) 环境变量与默认配置 / Env vars & defaults
# -----------------------------------------------------------------------------
CHROMA_PERSIST_DIR: str = _env("CHROMA_PERSIST_DIR", "./chroma_store") or "./chroma_store"
CHROMA_COLLECTION: str  = _env("CHROMA_COLLECTION",  "guideline_chunks") or "guideline_chunks"
EMBED_MODEL: str        = _env("EMBEDDING_MODEL",    "sentence-transformers/all-MiniLM-L6-v2") \
                          or "sentence-transformers/all-MiniLM-L6-v2"
DRUG_DB_PATH: str       = _env("DRUG_DB_PATH",       "./db/drugs.sqlite") or "./db/drugs.sqlite"
DEMO: bool              = _as_bool(_env("CAREMIND_DEMO", "1"), default=True)
CHROMA_TELEMETRY_OFF: bool = not _as_bool(_env("CHROMA_ANONYMIZED_TELEMETRY", "False"), default=False)


# =============================================================================
# 3) 惰性导入 Chroma + 客户端/集合缓存 / Lazy import + cache
# -----------------------------------------------------------------------------
_CLIENT = None
_COLLECTION = None

def _chroma():
    # 惰性导入，防止模块导入阶段崩溃
    from chromadb import PersistentClient, Settings  # type: ignore
    from chromadb.utils import embedding_functions   # type: ignore
    return PersistentClient, embedding_functions, Settings

def clear_chroma_cache() -> None:
    """用于调试：清除进程内客户端/集合缓存。"""
    global _CLIENT, _COLLECTION
    _CLIENT = None
    _COLLECTION = None

def get_chroma_client(persist_dir: Optional[str] = None):
    """获取（或创建并缓存）Chroma 客户端。"""
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

def get_chroma_collection(name: Optional[str] = None, embed_model: Optional[str] = None):
    """获取（或创建并缓存）Chroma 集合；兼容指定 embedding 模型。"""
    global _COLLECTION
    if _COLLECTION is not None:
        return _COLLECTION
    _, embedding_functions, _ = _chroma()
    client = get_chroma_client()
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=(embed_model or EMBED_MODEL)
    )
    try:
        _COLLECTION = client.get_collection(
            name=(name or CHROMA_COLLECTION),
            embedding_function=embed_fn,
        )
        return _COLLECTION
    except Exception:
        # 自动探测集合名（优先包含 "guideline"）
        try:
            cands = []
            try:
                for d in list_collections_safe():
                    nm = d.get("name")
                    if nm:
                        cands.append(nm)
            except Exception:
                pass
            pick = None
            for nm in cands:
                if "guideline" in nm.lower():
                    pick = nm; break
            if not pick and cands:
                pick = cands[0]
            if not pick:
                raise
            _COLLECTION = client.get_collection(name=pick, embedding_function=embed_fn)
            return _COLLECTION
        except Exception:
            # 继续抛出，由上层处理（UI 会给出友好提示）
            raise


# =============================================================================
# 4) 指南检索（Chroma）/ Guideline search (Chroma)
# -----------------------------------------------------------------------------
def search_guidelines(query: str, k: int = 4) -> List[Dict[str, Any]]:
    """
    返回结构：[{ 'content': str, 'meta': {...}, 'score': float }, ...]
    * 不改变既有签名；内部走缓存的 client/collection。
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
# 5) 药品结构化检索（SQLite）/ Structured drug lookup (SQLite)
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
        cur.execute("SELECT * FROM drugs WHERE name LIKE ? LIMIT ?",
                    (f\"%{name_substr}%\", int(limit)))
        rows = [dict(r) for r in cur.fetchall()]
    finally:
        try:
            con.close()
        except Exception:
            pass
    return rows


# =============================================================================
# 6) 列出集合（用于诊断面板）/ Safe collection listing for diagnostics
# -----------------------------------------------------------------------------
def _fallback_collections_from_sqlite_dir(dir_path: str) -> List[str]:
    """
    当 API 枚举失败时，直接读取 {dir}/chroma.sqlite3 或 {dir}/*.sqlite* 的 collections 表拿集合名。
    仅用于诊断显示。
    """
    try:
        import glob, sqlite3 as _sq, os as _os
        candidates = []
        p1 = _os.path.join(dir_path, "chroma.sqlite3")
        if _os.path.exists(p1):
            candidates.append(p1)
        candidates.extend(glob.glob(_os.path.join(dir_path, "*.sqlite*")))

        names, seen = [], set()
        for fp in candidates:
            con = None
            try:
                con = _sq.connect(fp)
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
    优先使用 Chroma API；失败则回退到 SQLite 扫描。
    返回 [{"name": 名称, "count": 计数或'?' } ...] 或 [{"error": "..."}]
    """
    # 优先使用已缓存的 client（若尚未创建则创建）
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

    # 回退：直接扫 SQLite
    try:
        names = _fallback_collections_from_sqlite_dir(CHROMA_PERSIST_DIR)
        if names:
            return [{"name": n, "count": "?"} for n in names]
        return [{"error": "no collections found"}]
    except Exception as e:
        return [{"error": str(e)}]
