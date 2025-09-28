# -*- coding: utf-8 -*-
"""
retriever.py — CareMind RAG 检索层（稳定版）
- 新式 Chroma 客户端：PersistentClient(path=...) + 关闭匿名遥测
- 自动隔离旧/损坏索引目录（*_type/deprecated config/setting 冲突等），重建后重试
- 保留：惰性嵌入、指南检索、SQLite（可选）检索、RRF 融合、CLI 自测
- 诊断接口：list_collections_safe(), retriever_version()

环境变量（Secrets 优先）：
  CHROMA_PERSIST_DIR   默认: ./chroma_store
  CHROMA_COLLECTION    默认: guideline_chunks
  EMBEDDING_MODEL      默认: BAAI/bge-large-zh-v1.5（或 small-zh，也可）
  DRUG_DB_PATH         默认: ./db/drugs.sqlite
  CHROMA_ANONYMIZED_TELEMETRY 可设为 "false"
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

def retriever_version() -> str:
    """返回检索器版本号（供诊断面板显示）。"""
    return __RETRIEVER_VERSION__

# ---------------------------
# 日志
# ---------------------------
def _log(*msg: Any) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] retriever:", *msg, flush=True)

# ---------------------------
# 嵌入（惰性加载）
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

# ---------------------------
# （可选）SQLite 药品库检索（示例）
# ---------------------------
def _sqlite_search_drugfacts(q: str, topn: int = 5) -> List[Dict[str, Any]]:
    """
    朴素 LIKE 检索（按你的实际表结构修改）
    假设表：drugs(name TEXT, indications TEXT, contraindications TEXT, interactions TEXT, pregnancy TEXT, source TEXT)
    """
    if not os.path.isfile(SQLITE_PATH):
        return []
    rows: List[Tuple] = []
    try:
        con = sqlite3.connect(SQLITE_PATH)
        cur = con.cursor()
        like = f"%{q}%"
        cur.execute("""
            SELECT name, indications, contraindications, interactions, pregnancy, source
            FROM drugs
            WHERE name LIKE ? OR indications LIKE ? OR interactions LIKE ?
            LIMIT ?
        """, (like, like, like, topn))
        rows = cur.fetchall()
    except Exception as e:
        _log("SQLite search failed:", repr(e))
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
    """
    向量检索 +（可选）SQLite 关键词检索，经 RRF 融合。
    """
    k_guideline = k_guideline or max(3, k)
    k_sqlite = k_sqlite or max(3, k // 2)

    g_hits = search_guidelines(query, k=k_guideline)
    s_hits: List[Dict[str, Any]] = _sqlite_search_drugfacts(query, topn=k_sqlite) if use_sqlite else []

    if g_hits and s_hits:
        return _rrf_rank([g_hits, s_hits], k=k)
    return (g_hits or s_hits)[:k]

# ---------------------------
# 诊断：列出集合（稳定兜底）
# ---------------------------
class _ColInfo(TypedDict, total=False):
    id: str
    name: str
    count: int
    metadata: dict

def list_collections_safe() -> List[_ColInfo]:
    """
    安全地列出当前 PERSIST_DIR 下的集合。
    任何异常都不会向上抛出，以免影响前端诊断卡片。
    """
    infos: List[_ColInfo] = []
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
                pass
            infos.append(info)
    except Exception as e:
        _log("list_collections_safe error:", repr(e))
        return []
    return infos

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
    ap = argparse.ArgumentParser(description="CareMind Retriever (quarantine legacy store & retry)")
    ap.add_argument("--q", "--query", dest="query", type=str, required=True)
    ap.add_argument("--topn", type=int, default=8)
    ap.add_argument("--method", type=str, default="rrf", choices=["guideline", "sqlite", "rrf"])
    ap.add_argument("--no-sqlite", action="store_true")
    args = ap.parse_args()

    _log("Version:", __RETRIEVER_VERSION__)
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

if __name__ == "__main__":
    main()