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
  Disable Chroma telemetry via Settings to silence ClientStartEvent noise.
- ✅ Secrets 优先：通过 _env() 读取配置（先 Secrets，再环境变量，最后默认）。
  Secrets-first config via _env() (Secrets → env → default).
- ✅ 安全的集合枚举：list_collections_safe() 仅返回 {name, count}，避免 _type 等序列化问题。
  Safe collection listing that avoids serializing Chroma internals like `_type`.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

# =============================================================================
# 0) SQLite 兼容补丁（Cloud）/ SQLite compatibility shim (Cloud)
# -----------------------------------------------------------------------------
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
# -----------------------------------------------------------------------------
def _chroma():
    """
    惰性导入 Chroma；返回 (PersistentClient, embedding_functions, Settings)
    Lazy-import chroma; return (PersistentClient, embedding_functions, Settings).

    说明 / Notes:
    - 仅在需要访问向量库时才导入，避免模块导入期失败。
      Importing only when needed avoids import-time crashes on Cloud.
    """
    from chromadb import PersistentClient
    from chromadb.utils import embedding_functions
    from chromadb.config import Settings  # 0.5.x: use Settings to configure the client
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
        临床问题（中/英均可） / clinical question (cn/en ok)
    k : int
        返回的片段数量 / number of results to return
    """
    try:
        # 1) 获取客户端、嵌入函数与设置 / Client + embedding fn + settings
        PersistentClient, embedding_functions, Settings = _chroma()

        # 关闭匿名遥测并允许 reset（在临时/容器环境更安全）
        # Disable anonymized telemetry & allow_reset for ephemeral environments
        client = PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(
                anonymized_telemetry=not CHROMA_TELEMETRY_OFF,  # False ⇒ disable telemetry
                allow_reset=True,
            ),
        )

        # 2) 绑定集合（若不存在则创建）/ Bind the collection (create if missing)
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL
        )
        collection = client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            embedding_function=embed_fn,
        )

        # 3) 查询 / Query
        res = collection.query(
            query_texts=[query],
            n_results=int(k),
            include=["documents", "metadatas"],
        )

        # 4) 结果整形 / Shape results
        docs  = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        return [{"content": d, "meta": m} for d, m in zip(docs, metas)]

    except Exception as e:
        # 失败时温和降级：打印日志并返回空列表（由上层决定如何回退）
        # Soft-degrade: log & return [], letting pipeline decide a fallback/demo.
        print("[retriever] search_guidelines error:", e)
        return []


# =============================================================================
# 5) 列出集合（用于诊断面板）/ List collections for diagnostics
# -----------------------------------------------------------------------------
def list_collections_safe() -> List[Dict[str, Any]]:
    """
    安全地列出 Chroma 集合名称与条目数，避免把内部对象（含 `_type`）直接序列化。
    Safely list Chroma collections with document counts, avoiding serialization of
    internal objects (which may include `_type` and cause UI dump errors).
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
            # c 可能是一个带内部元数据的对象；只提取可序列化字段
            # `c` may carry non-serializable fields; extract only safe fields.
            name = getattr(c, "name", None) or "?"
            try:
                # 一些后端支持 c.count()；如不支持则以 1 条查询作为探针
                # Some backends support c.count(); if not, probe with 1-result query
                count = int(c.count())
            except Exception:
                try:
                    col = client.get_collection(name=name)
                    q = col.query(query_texts=["."], n_results=1)
                    ids = q.get("ids", [[]])[0]
                    count = len(ids)
                except Exception as e:
                    count = f"error: {e}"
            out.append({"name": name, "count": count})
        return out

    except Exception as e:
        return [{"error": str(e)}]


# =============================================================================
# 6) 药品结构化检索（SQLite）/ Structured drug lookup (SQLite)
# -----------------------------------------------------------------------------
def _connect_sqlite(path: str) -> sqlite3.Connection:
    """
    打开 SQLite 连接，Row 工厂方便以 dict-like 访问列。
    Open SQLite connection with Row factory for dict-like column access.
    """
    if not os.path.exists(path):
        if DEMO:
            # 演示模式：使用内存库避免崩溃（无表即无结果）
            # Demo mode: in-memory DB to avoid crashes (no tables ⇒ no results).
            con = sqlite3.connect(":memory:")
            con.row_factory = sqlite3.Row
            return con
        raise FileNotFoundError(f"SQLite DB not found: {path}")

    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    return con


def search_drug_structured(drug_name: str) -> Optional[Dict[str, Any]]:
    """
    模糊检索药品信息（示例字段）；返回 dict 或 None。
    Fuzzy lookup of a drug (example fields); returns dict or None.

    假定表结构 / Assumed schema:
      CREATE TABLE drugs (
          id INTEGER PRIMARY KEY,
          name TEXT, generic_name TEXT,
          indications TEXT, contraindications TEXT,
          interactions TEXT, pregnancy TEXT, source TEXT
      )
    """
    name = (drug_name or "").strip()
    if not name:
        return None

    con = _connect_sqlite(DRUG_DB_PATH)
    try:
        cur = con.cursor()

        # 1) 近似精确匹配 / Near-exact match first
        cur.execute(
            """
            SELECT name, generic_name, indications, contraindications,
                   interactions, pregnancy, source
            FROM drugs
            WHERE name = ? OR generic_name = ?
            LIMIT 1
            """,
            (name, name),
        )
        row = cur.fetchone()

        # 2) LIKE 模糊匹配 / Fallback to LIKE fuzzy search
        if not row:
            kw = f"%{name}%"
            cur.execute(
                """
                SELECT name, generic_name, indications, contraindications,
                       interactions, pregnancy, source
                FROM drugs
                WHERE name LIKE ? OR generic_name LIKE ?
                ORDER BY LENGTH(name) ASC
                LIMIT 1
                """,
                (kw, kw),
            )
            row = cur.fetchone()

        if not row:
            return None

        keys = ["name", "generic_name", "indications", "contraindications",
                "interactions", "pregnancy", "source"]
        return {k: row[idx] for idx, k in enumerate(keys)}

    except Exception as e:
        print("[retriever] search_drug_structured error:", e)
        if DEMO:
            return None
        raise
    finally:
        try:
            con.close()
        except Exception:
            pass
