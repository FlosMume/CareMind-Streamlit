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
- Cloud 常见问题：系统自带 sqlite3 版本可能过旧；使用 pysqlite3-binary 作为替代并别名到 sqlite3。
  Streamlit Cloud may ship an old sqlite3; alias pysqlite3-binary to sqlite3 for >=3.35 features.
- Chroma 采用“惰性导入”，避免导入阶段崩溃；只有在真正查询时才加载。
  Chroma is lazy-imported so module import never fails—only load it when needed.
- 通过 chromadb.config.Settings 关闭匿名遥测，解决 “Failed to send telemetry ClientStartEvent … get_settings” 噪声。
  Use chromadb.config.Settings to disable telemetry (stops the ClientStartEvent/get_settings noise).
- 返回朴素 dict/list，便于 pipeline 与 UI 处理。
  Return plain dicts/lists so pipeline/UI can compose responses easily.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

# =============================================================================
# 0) SQLite 兼容补丁（Cloud）/ SQLite compatibility shim (Cloud)
# -----------------------------------------------------------------------------
# 如果安装了 pysqlite3-binary，则把它映射为标准库 sqlite3，以获得新版本 SQLite(>=3.35)
# If pysqlite3-binary is present, alias it to stdlib sqlite3 to get >=3.35 features.
try:
    import pysqlite3  # type: ignore
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
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
        import streamlit as st  # imported lazily; safe outside Streamlit too
        return os.getenv(key, st.secrets.get(key, default))
    except Exception:
        return os.getenv(key, default)

def _as_bool(val: str | None, default: bool = False) -> bool:
    """将 '1'/'true'/'True'/'yes' 等解析为布尔值 / Parse common truthy strings to bool."""
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
# Allow overriding anonymized telemetry via Secrets/env (default: off)
CHROMA_TELEMETRY_OFF: bool = not _as_bool(_env("CHROMA_ANONYMIZED_TELEMETRY", "False"), default=False)


# =============================================================================
# 3) 惰性导入 Chroma / Lazy-import Chroma
# -----------------------------------------------------------------------------
def _chroma():
    """
    惰性导入 Chroma；返回 (PersistentClient, embedding_functions, Settings)
    Lazy-import chroma; return (PersistentClient, embedding_functions, Settings).
    """
    from chromadb import PersistentClient
    from chromadb.utils import embedding_functions
    from chromadb.config import Settings  # 0.5.x: use Settings to configure client
    return PersistentClient, embedding_functions, Settings


# =============================================================================
# 4) 指南检索（Chroma）/ Guideline search (Chroma)
# -----------------------------------------------------------------------------
def search_guidelines(query: str, k: int = 4) -> List[Dict[str, Any]]:
    """
    使用 Chroma 进行语义检索；返回 [{"content": 文本, "meta": 元数据}, ...]
    Semantic search via Chroma; returns [{"content": str, "meta": dict}, ...]
    """
    try:
        # 1) 获取客户端、嵌入函数与设置 / Client + embedding fn + settings
        PersistentClient, embedding_functions, Settings = _chroma()

        # 关闭匿名遥测 & 在临时环境中允许 reset（可选）
        # Disable anonymized telemetry & allow_reset in ephemeral envs
        client = PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(
                anonymized_telemetry=not CHROMA_TELEMETRY_OFF,  # False → disable telemetry
                allow_reset=True,                                # optional safety
            ),
        )

        # 2) 绑定集合（若不存在则创建）/ Bind (create if missing)
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
        # Soft-degrade: log & return [], letting pipeline decide a fallback.
        print("[retriever] search_guidelines error:", e)
        return []


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
            # 演示模式：内存库避免崩溃（无表即无结果）
            # Demo mode: use in-memory DB to avoid crashes (no tables → no results).
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
    """
    name = (drug_name or "").strip()
    if not name:
        return None

    con = _connect_sqlite(DRUG_DB_PATH)
    try:
        cur = con.cursor()

        # 先尝试近似精确匹配 / Try near-exact first
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

        # 若无命中则 LIKE 模糊 / Fallback to LIKE fuzzy
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
