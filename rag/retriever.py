# -*- coding: utf-8 -*-
"""
retriever.py | CareMind
-----------------------
职责 / Responsibilities
1) 在 Chroma 向量库中检索指南片段
   Retrieve guideline text chunks from a Chroma vector DB.
2) 在 SQLite 中查询药品结构化信息
   Look up structured drug info in SQLite.

设计要点 / Design Notes
- 为了兼容 Streamlit Cloud 的较旧 sqlite3，优先使用 pysqlite3-binary 作为替代。
  Use pysqlite3-binary to provide a modern sqlite3 on Streamlit Cloud.
- 对 chroma 的导入采用“惰性导入”（在函数内导入），防止模块导入阶段直接失败。
  Lazy-import chroma inside functions to avoid import-time crashes.
- 返回尽量朴素的 Python dict / list，方便上层 pipeline 与 UI 处理。
  Return plain dict/list structures for easy handling by pipeline/UI.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

# =============================================================================
# 0) SQLite 兼容补丁（Streamlit Cloud 常见问题）
#    SQLite compatibility patch for Streamlit Cloud
# -----------------------------------------------------------------------------
# 如果存在 pysqlite3，则把它映射为标准库 sqlite3，以获得较新的 SQLite 版本（>=3.35）
# If pysqlite3 exists, alias it to stdlib sqlite3 to get a newer SQLite (>=3.35).
try:
    import pysqlite3  # type: ignore
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    # 不可用时，继续使用系统自带 sqlite3；如后续 Chroma 需要更高版本，会在运行时报错
    # If unavailable, keep system sqlite3; Chroma may later complain if too old.
    pass

import sqlite3  # after aliasing


# =============================================================================
# 1) 环境变量与默认配置 / Env vars & defaults
# -----------------------------------------------------------------------------
CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")
CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "guideline_chunks")
EMBED_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

DRUG_DB_PATH: str = os.getenv("DRUG_DB_PATH", "./db/drugs.sqlite")
DEMO: bool = os.getenv("CAREMIND_DEMO", "1") == "1"  # 云端默认演示模式 ON / demo ON by default on Cloud


# =============================================================================
# 2) 惰性导入 Chroma / Lazy-import Chroma
# -----------------------------------------------------------------------------
def _chroma():
    """
    惰性导入 Chroma 所需对象；在真正需要时才导入。
    Lazy-import chroma objects; import only when actually needed.

    Returns
    -------
    (PersistentClient, embedding_functions)
    """
    from chromadb import PersistentClient
    from chromadb.utils import embedding_functions
    return PersistentClient, embedding_functions


# =============================================================================
# 3) 指南检索（Chroma）/ Guideline search (Chroma)
# -----------------------------------------------------------------------------
def search_guidelines(query: str, k: int = 4) -> List[Dict[str, Any]]:
    """
    使用 Chroma 进行语义检索，返回文本片段与元数据。
    Semantic search against Chroma; return snippets with metadata.

    Parameters
    ----------
    query : str   # 可为中文或英文 | can be Chinese or English
    k     : int   # 返回的片段数量 | number of top results

    Returns
    -------
    List[dict]    # [{"content": str, "meta": dict}, ...]
    """
    try:
        # 1) 获取 Chroma 客户端与嵌入函数 / get client & embedding fn
        PersistentClient, embedding_functions = _chroma()
        client = PersistentClient(path=CHROMA_PERSIST_DIR)

        # 2) 绑定集合（不存在则创建）/ bind collection (create if missing)
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL
        )
        collection = client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            embedding_function=embed_fn,
        )

        # 3) 查询 / query
        res = collection.query(
            query_texts=[query],
            n_results=int(k),
            include=["documents", "metadatas"],
        )

        # 4) 结果整形 / shape results
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        out: List[Dict[str, Any]] = []
        for d, m in zip(docs, metas):
            out.append({"content": d, "meta": m})
        return out

    except Exception as e:
        # 失败时的温和降级：打印日志并返回空列表（让上层决定如何回退）
        # Soft-degrade: log & return empty so pipeline can handle demo fallback.
        print("[retriever] search_guidelines error:", e)
        return []


# =============================================================================
# 4) 药品结构化检索（SQLite）/ Structured drug lookup (SQLite)
# -----------------------------------------------------------------------------
def _connect_sqlite(path: str) -> sqlite3.Connection:
    """
    打开 SQLite 连接（行工厂设置为 dict-like Row）
    Open SQLite connection; row_factory -> sqlite3.Row for dict-like access.
    """
    if not os.path.exists(path):
        if DEMO:
            # 演示模式下容忍缺库：返回内存库，避免崩溃（无表即无结果）
            # In demo mode tolerate missing DB: return in-memory DB to avoid crashes.
            con = sqlite3.connect(":memory:")
            con.row_factory = sqlite3.Row
            return con
        raise FileNotFoundError(f"SQLite DB not found: {path}")

    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    return con


def search_drug_structured(drug_name: str) -> Optional[Dict[str, Any]]:
    """
    在 SQLite 中按名称模糊检索药品的结构化信息（示例字段）。
    Fuzzy lookup a drug's structured info in SQLite (example schema).

    Parameters
    ----------
    drug_name : str  # 关键词（如 '阿司匹林' 或 'Aspirin'）

    Returns
    -------
    dict | None      # e.g. {"name": ..., "indications": ..., ...} or None
    """
    name = (drug_name or "").strip()
    if not name:
        return None

    con = _connect_sqlite(DRUG_DB_PATH)
    try:
        cur = con.cursor()

        # 先尝试“近似精确匹配” / try near-exact match first
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

        # 若无命中则做 LIKE 模糊检索 / fallback to LIKE search
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
