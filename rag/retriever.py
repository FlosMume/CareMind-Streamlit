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
import os, sys, time, json, glob, shutil, datetime, contextlib
from typing import Any, Dict, List, Optional, Tuple, TypedDict

# --- 把 pysqlite3-binary 映射为 sqlite3（云端常见做法） ---
try:
    import pysqlite3 as sqlite3  # type: ignore
    sys.modules["sqlite3"] = sqlite3
except Exception:
    import sqlite3  # type: ignore

# --- 关闭 Chroma 遥测（两种变量都兜底一遍） ---
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
>>>>>>> bc8a0fc (avoid non exsisting pysqlite3-binary==0.5.3.post3)

def retriever_version() -> str:
    """返回检索器版本号（供诊断面板显示）。"""
    return __RETRIEVER_VERSION__

# ---------------------------
# 日志
# ---------------------------
def _log(*msg: Any) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] retriever:", *msg, flush=True)


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
#（可选）sysdb 定位（仅用于诊断/兼容；不再在线改库）
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

# ---------------------------
# 旧索引目录隔离
# ---------------------------
def _quarantine_persist_dir(path: str) -> str:
    """
    将现有 Chroma 持久化目录整体隔离为备份目录：<path>.broken-YYYYmmdd-HHMMSS
    在原路径上新建一个空目录供重新初始化。
    """
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
        return path
    ts = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    bak = f"{path}.broken-{ts}"
    try:
        os.rename(path, bak)  # 同盘原子重命名
    except OSError:
        # 跨设备或占用时退化为 copytree + rmtree
        shutil.copytree(path, bak, dirs_exist_ok=True)
        shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)
    _log(f"Quarantined old Chroma store to: {bak}")
    return path

# ---------------------------
# Chroma 客户端 & 集合（统一入口）
# ---------------------------
_chroma_client = None
_collection = None

def get_chroma_client():
    """
    新式客户端：PersistentClient(path=...)。
    三层兜底：
      A. 创建失败（含 tenant 不存在） -> 隔离目录 -> 显式 tenant/database 再试
      B. 创建成功但预检 list_collections() 失败 -> 隔离目录 -> 再试
      C. 若仍异常 -> 再隔离一次 -> 最后重试
    """
    global _chroma_client
    if _chroma_client is not None:
        return _chroma_client

    # 采用绝对路径，避免相对路径在不同 cwd 下造成多套实例
    persist_path = os.path.abspath(PERSIST_DIR)
    os.makedirs(persist_path, exist_ok=True)

    # 清理可能引发“旧配置”判定的环境键
    for k in [
        "CHROMA_DB_IMPL",
        "CHROMA_PERSIST_DIRECTORY",
        "CHROMA_IS_PERSISTENT",
        "CHROMA_ALLOW_RESET",
        "CHROMA_TELEMETRY_IMPLEMENTATION",
    ]:
        os.environ.pop(k, None)

    import chromadb
    from chromadb import PersistentClient
    from chromadb.config import Settings

    settings = Settings(anonymized_telemetry=False)

    def _make_client(explicit_td: bool = False):
        if explicit_td:
            # 显式 tenant/database，有些环境下能避免“tenant 不存在”的校验失败
            return PersistentClient(
                path=persist_path,
                settings=settings,
                tenant="default_tenant",
                database="default_database",
            )
        else:
            return PersistentClient(path=persist_path, settings=settings)

    # ------- 第一枪：正常创建 -------
    try:
        _chroma_client = _make_client(explicit_td=False)
    except Exception as e:
        _log("PersistentClient init failed (first try):", repr(e))
        _quarantine_persist_dir(persist_path)
        try:
            _chroma_client = _make_client(explicit_td=True)
        except Exception as e2:
            _log("PersistentClient init failed (second try, with tenant/database):", repr(e2))
            # 最后再隔离一次后做最后重试
            _quarantine_persist_dir(persist_path)
            _chroma_client = _make_client(explicit_td=True)

    # ------- 预检：列集合，若失败则隔离并重建 -------
    try:
        _ = _chroma_client.list_collections()
    except Exception as e:
        _log("Preflight list_collections() failed:", repr(e))
        _quarantine_persist_dir(persist_path)
        try:
            _chroma_client = _make_client(explicit_td=True)
            _ = _chroma_client.list_collections()
        except Exception as e2:
            _log("Preflight failed again after quarantine:", repr(e2))
            _quarantine_persist_dir(persist_path)
            _chroma_client = _make_client(explicit_td=True)
            _ = _chroma_client.list_collections()  # 若再失败让其冒泡

    _log("Chroma version:", getattr(chromadb, "__version__", "unknown"))
    _log("CHROMA_PERSIST_DIR:", persist_path)
    _log("CHROMA_COLLECTION:", COLLECTION_NAME)
    return _chroma_client


def get_chroma_collection():
    """
    获取（或创建）指定集合。
    - 若创建失败（任意异常），隔离旧目录 -> 重建客户端 -> 再试一次。
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
        _log("get_or_create_collection failed; quarantining and retrying once...", repr(e))
        _quarantine_persist_dir(PERSIST_DIR)
        _chroma_client = None
        client = get_chroma_client()
        _collection = client.get_or_create_collection(name=target, embedding_function=embed_fn)
        _log("Collection opened after quarantine+rebuild.")
        return _collection


# ---------------------------
# 指南向量检索
# ---------------------------
def search_guidelines(query: str, k: int = 5, include_metadata: bool = True) -> List[Dict[str, Any]]:
    """
    语义检索（Chroma 0.5.x API）
    返回：[{id, score, text, meta}, ...]
=======
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
>>>>>>> bc8a0fc (avoid non exsisting pysqlite3-binary==0.5.3.post3)
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
>>>>>>> bc8a0fc (avoid non exsisting pysqlite3-binary==0.5.3.post3)
        return []
    rows: List[Tuple] = []
    try:
        cur.execute("SELECT * FROM drugs WHERE name LIKE ? LIMIT ?", (f"%{name_substr}%", int(limit)))
        rows = [dict(r) for r in cur.fetchall()]
>>>>>>> bc8a0fc (avoid non exsisting pysqlite3-binary==0.5.3.post3)
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
        cols = client.list_collections()  # 0.5.x
        for c in cols:
            info: _ColInfo = {
                "id": getattr(c, "id", None),
                "name": getattr(c, "name", None),
                "metadata": getattr(c, "metadata", {}) or {},
            }
            # 统计条数：优先用 count()；退化为 get() 长度估计
            try:
                if hasattr(c, "count"):
                    info["count"] = int(c.count())
                else:
                    q = c.get()
                    ids = (q.get("ids") or [])
                    info["count"] = sum(len(x) for x in ids) if ids and isinstance(ids[0], list) else len(ids)
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
