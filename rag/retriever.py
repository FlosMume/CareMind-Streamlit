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
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

# 模块版本号（用于诊断显示云端是否更新）
VERSION = "retriever-2025-09-21a"

# =============================================================================
# 0) SQLite 兼容补丁（Cloud）/ SQLite compatibility shim (Cloud)
# -----------------------------------------------------------------------
# 如果安装了 pysqlite3-binary，则将其别名为标准库 sqlite3，以获得 SQLite ≥ 3.35
# If pysqlite3-binary is present, alias it to stdlib sqlite3 to get SQLite ≥ 3.35.
try:
    import pysqlite3  # type: ignore
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    # 如果不可用，则继续使用系统自带 sqlite3；若版本过低，Chroma 端可能在运行时报错
    # If unavailable, we keep system sqlite3; Chroma may later complain if too old.
    pass

import sqlite3  # after aliasing


# =============================================================================
# 1) Secrets-aware env helpers / 读取配置优先 Secrets
# -----------------------------------------------------------------------------
def _env(key: str, default: str | None = None) -> str | None:
    """
    优先从 st.secrets 读取（Cloud 上 App settings → Secrets），否则读取环境变量，最后默认值。
    Prefer st.secrets on Cloud, then os.environ, otherwise default.
    """
    try:
        import streamlit as st  # imported lazily; safe when Streamlit is absent
        return os.getenv(key, st.secrets.get(key, default))
    except Exception:
        return os.getenv(key, default)

def _as_bool(val: str | None, default: bool = False) -> bool:
    """将 '1'/'true'/'yes' 等解析为布尔值 / Parse common truthy strings to bool."""
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

# 允许通过 Secrets/env 覆盖是否关闭 Chroma 遥测（默认关闭）
# Allow overriding anonymized telemetry via Secrets/env (default: OFF)
CHROMA_TELEMETRY_OFF: bool = not _as_bool(_env("CHROMA_ANONYMIZED_TELEMETRY", "False"), default=False)


# =============================================================================
# 3) 惰性导入 Chroma / Lazy-import Chroma
# ----------------------------------------
def _chroma():
    from chromadb import PersistentClient, Settings  # type: ignore
    from chromadb.utils import embedding_functions   # type: ignore
    return PersistentClient, embedding_functions, Settings


# =============================================================================
# 4) 指南检索（Chroma）/ Guideline search (Chroma)
# -----------------------------------------------------------------------------
def search_guidelines(query: str, k: int = 4) -> List[Dict[str, Any]]:
    """
    使用 Chroma 进行语义检索；返回 [{"content": 文本, "meta": 元数据}, ...]
    Semantic search via Chroma; returns [{"content": str, "meta": dict}, ...].

    Parameters
    ----------
    query : str
        临床问题（中/英均可） / clinical question (cn/en)
    k : int
        返回片段条数 / top-k snippets
    """
    if not query:
        return []

    PersistentClient, embedding_functions, Settings = _chroma()
    client = PersistentClient(
        path=CHROMA_PERSIST_DIR,
        settings=Settings(
            anonymized_telemetry=not CHROMA_TELEMETRY_OFF,
            allow_reset=True,
        ),
    )
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )
    try:
        collection = client.get_collection(
            name=CHROMA_COLLECTION,
            embedding_function=embed_fn,
        )
    except Exception:
        # 自动探测集合名（优先包含 "guideline" 的集合）
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
                    pick = nm
                    break
            if not pick and cands:
                pick = cands[0]
            if not pick:
                raise
            collection = client.get_collection(name=pick, embedding_function=embed_fn)
        except Exception as e:
            # 直接抛出，交给上层处理
            raise e

    res = collection.query(
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
    """
    打开 SQLite 连接，Row 工厂方便以 dict-like 访问列。
    Open SQLite connection with Row factory for dict-like column access.
    """
    if not os.path.exists(path):
        if DEMO:
            # 演示模式：使用内存库避免崩溃（无表即无结果）
            # Demo mode: use in-memory db to avoid crash when missing file
            return sqlite3.connect(":memory:")
        raise FileNotFoundError(path)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    return con

def search_drug_structured(name_substr: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    模糊匹配药名（LIKE '%xxx%'），返回若干行（dict 列表）。
    """
    if not name_substr:
        return []
    con = _connect_sqlite(DRUG_DB_PATH)
    cur = con.cursor()
    try:
        cur.execute(
            "SELECT * FROM drugs WHERE name LIKE ? LIMIT ?",
            (f"%{name_substr}%", int(limit)),
        )
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
# --- 最小补丁（1）：SQLite 回退函数 ---
# 当 client.list_collections() 遇到 `_type` 等序列化问题时，直接读 SQLite 获取集合名
def _fallback_collections_from_sqlite_dir(dir_path: str) -> List[str]:
    """
    读取 {dir}/chroma.sqlite3 或 {dir}/*.sqlite* 的 collections 表以获取集合名。
    仅用于诊断显示。
    """
    try:
        import glob, sqlite3 as _sq
        candidates: List[str] = []
        p1 = os.path.join(dir_path, "chroma.sqlite3")
        if os.path.exists(p1):
            candidates.append(p1)
        candidates.extend(glob.glob(os.path.join(dir_path, "*.sqlite*")))

        names: List[str] = []
        seen = set()
        for fp in candidates:
            con = None
            try:
                con = _sq.connect(fp)
                cur = con.cursor()
                cur.execute("SELECT name FROM collections")
                for (nm,) in cur.fetchall():
                    if nm and nm not in seen:
                        seen.add(nm)
                        names.append(nm)
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
    安全地列出 Chroma 集合名称与条目数，避免把内部对象（含 `_type`）直接序列化。
    优先尝试 Chroma API；失败则回退到 SQLite 扫描。
    """
    try:
        PersistentClient, _, Settings = _chroma()
        client = PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(
                anonymized_telemetry=not CHROMA_TELEMETRY_OFF,
                allow_reset=True,
            ),
        )

        out: List[Dict[str, Any]] = []
        for c in client.list_collections():
            name = getattr(c, "name", None) or "?"
            try:
                # 一些后端支持 c.count()；如不支持则宽容处理
                count = int(c.count())
            except Exception:
                count = "?"
            out.append({"name": name, "count": count})
        if out:
            return out

    except Exception:
        # 忽略异常，进入回退
        pass

    # --- 最小补丁（2）：SQLite 回退 ---
    names = _fallback_collections_from_sqlite_dir(CHROMA_PERSIST_DIR)
    return [{"name": n, "count": "?"} for n in names] if names else [{"error": "no collections found"}]
