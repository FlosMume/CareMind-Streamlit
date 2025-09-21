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
import os, json, sys, glob, time, contextlib
from typing import Any, Dict, List, Optional, Tuple

# --- 优先把 pysqlite3-binary 映射为 sqlite3（云端常见做法） ---
try:
    import pysqlite3 as sqlite3  # type: ignore
    sys.modules["sqlite3"] = sqlite3
except Exception:
    import sqlite3  # type: ignore

# --- 关闭 Chroma 遥测 ---
os.environ.setdefault("CHROMA_TELEMETRY_ENABLED", "false")

# --- 环境 & 默认配置 ---
PERSIST_DIR      = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")
COLLECTION_NAME  = os.getenv("CHROMA_COLLECTION", "guideline_chunks")
EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh")
SQLITE_PATH      = os.getenv("SQLITE_PATH", "./db/drugs.sqlite")

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

def get_chroma_client():
    """创建 Chroma Client（带全库预迁移 & 显式 Settings，一次且仅一次）。"""
    global _chroma_client
    if _chroma_client is not None:
        return _chroma_client

    # 目录与预迁移
    os.makedirs(PERSIST_DIR, exist_ok=True)
    _migrate_all_collections(PERSIST_DIR)

    # 显式、稳定的 Settings —— 确保与进程内其他地方完全一致
    import chromadb
    from chromadb.config import Settings

    # 有些托管环境把 CHROMA_ANONYMIZED_TELEMETRY 默认为 true，
    # 我们强制为 false，以避免与之前已创建的实例设置不一致
    settings = Settings(
        chroma_db_impl="duckdb+parquet",   # 0.5.x 的持久化默认实现
        is_persistent=True,
        persist_directory=PERSIST_DIR,
        anonymized_telemetry=False,
        allow_reset=False,                 # 保守：不允许 reset（可按需改 True）
    )

    # 统一用 Client(settings)（避免不同入口用 PersistentClient(path=...) 时引入隐式差异）
    _chroma_client = chromadb.Client(settings)

    _log("Chroma version:", getattr(chromadb, "__version__", "unknown"))
    _log("CHROMA_PERSIST_DIR:", PERSIST_DIR)
    _log("CHROMA_COLLECTION:", COLLECTION_NAME)
    _log("Chroma settings:", {
        "chroma_db_impl": "duckdb+parquet",
        "is_persistent": True,
        "anonymized_telemetry": False,
    })
    return _chroma_client


# ---------------------------
# 指南向量检索
# ---------------------------
def search_guidelines(query: str, k: int = 5, include_metadata: bool = True) -> List[Dict[str, Any]]:
    if not query:
        return []
    col = get_chroma_collection()
    res = col.query(
        query_texts=[query],
        n_results=k,
        include=["documents", "metadatas", "distances", "embeddings"] if include_metadata else ["documents"]
    )
    out: List[Dict[str, Any]] = []
    ids = (res.get("ids") or [[]])[0]
    docs = (res.get("documents") or [[]])[0]
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
# （可选）SQLite 药品库检索
# ---------------------------
def _sqlite_search_drugfacts(q: str, topn: int = 5) -> List[Dict[str, Any]]:
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

if __name__ == "__main__":
    main()
