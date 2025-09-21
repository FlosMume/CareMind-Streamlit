# -*- coding: utf-8 -*-
"""
retriever.py  — CareMind RAG 检索层（增强自愈版）

核心特性
1) 惰性导入 chromadb，避免没有 GPU/依赖时硬失败；
2) Chroma PersistentClient + 指定集合（从环境变量读取）；
3) 自动修复老版本 Chroma 索引在 0.5.x 下出现的 KeyError: '_type'；
4) 可选对接 SQLite 药品数据库（用于 keyword/结构化补充检索）；
5) 提供纯向量检索和混合检索（RRF 简化版）；
6) 可作为模块导入，也可命令行运行做快速测试。

环境变量（均有默认）
- CHROMA_PERSIST_DIR: ./chroma_store
- CHROMA_COLLECTION : guideline_chunks
- EMBEDDING_MODEL   : BAAI/bge-small-zh
- SQLITE_PATH       : ./db/drugs.sqlite
- CHROMA_TELEMETRY_ENABLED: false（默认在此模块里设为 false）
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

# --- 最早关闭 Chroma 遥测，避免云端告警/出网 ---
os.environ.setdefault("CHROMA_TELEMETRY_ENABLED", "false")

# --- 环境 & 默认配置 ---
PERSIST_DIR      = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")
COLLECTION_NAME  = os.getenv("CHROMA_COLLECTION", "guideline_chunks")
EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh")
SQLITE_PATH      = os.getenv("SQLITE_PATH", "./db/drugs.sqlite")

# ---------------------------
# 日志工具
# ---------------------------
def _log(*msg: Any) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] retriever:", *msg, flush=True)

# ---------------------------
# 嵌入函数（SentenceTransformers）
# ---------------------------
class _LazyEmbedder:
    """惰性加载 SentenceTransformer，减少无谓的启动时间与失败面。"""
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def __call__(self, texts: List[str]) -> List[List[float]]:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            _log("Loading embedding model:", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        # 保证输入为字符串列表
        texts = [t if isinstance(t, str) else str(t) for t in texts]
        vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return vecs.tolist()

# 供 Chroma 的 embedding_function 使用的适配器
class _ChromaEmbedFn:
    def __init__(self, embedder: _LazyEmbedder):
        self._embedder = embedder
    def __call__(self, input: List[str]) -> List[List[float]]:  # chromadb>=0.5
        return self._embedder(input)

_EMBED = _LazyEmbedder(EMBEDDING_MODEL)

# ---------------------------
# Chroma 客户端 & 集合
# ---------------------------
_chroma_client = None
_collection = None

def get_chroma_client():
    """惰性创建 Chroma PersistentClient，并打印版本与目录。"""
    global _chroma_client
    if _chroma_client is not None:
        return _chroma_client
    # 惰性导入 chromadb
    import chromadb
    from chromadb import PersistentClient
    _log("Chroma version:", getattr(chromadb, "__version__", "unknown"))
    _log("CHROMA_PERSIST_DIR:", PERSIST_DIR)
    _log("CHROMA_COLLECTION:", COLLECTION_NAME)
    # 保证目录存在（不存在时由 Chroma 创建）
    os.makedirs(PERSIST_DIR, exist_ok=True)
    _chroma_client = PersistentClient(path=PERSIST_DIR)
    return _chroma_client

def _find_sysdb_sqlite_file(persist_dir: str) -> Optional[str]:
    """
    兼容不同版本命名，尽力找到 chroma 的 sysdb sqlite 文件。
    常见文件名：chroma.sqlite3 / chroma.sqlite / chroma.db
    """
    cand = [
        os.path.join(persist_dir, "chroma.sqlite3"),
        os.path.join(persist_dir, "chroma.sqlite"),
        os.path.join(persist_dir, "chroma.db"),
    ]
    for p in cand:
        if os.path.isfile(p):
            return p
    # 兜底：找任意 *.sqlite*，取最新修改时间的一个
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

def _patch_chroma_configuration_sqlite(persist_dir: str, collection_name: str) -> bool:
    """
    针对 Chroma 0.5.x 在读取老库时出现的 KeyError: '_type' 的一次性补丁。
    做法：将 collections.configuration 写入最小合法 JSON:
      {"_type": "CollectionConfigurationInternal"}
    只对指定集合名生效；若已包含 _type 则不做任何修改。

    返回：是否执行了补丁（True=写入过；False=未写入或未找到库）
    """
    dbfile = _find_sysdb_sqlite_file(persist_dir)
    if not dbfile:
        _log("No sysdb sqlite found in", persist_dir, "— skip patch.")
        return False

    _log("Attempt patch on sysdb:", dbfile)
    patched = False
    try:
        con = sqlite3.connect(dbfile)
        cur = con.cursor()
        # 读取目标集合的 configuration
        cur.execute(
            "SELECT id, name, configuration FROM collections WHERE name = ?",
            (collection_name,)
        )
        row = cur.fetchone()
        if not row:
            _log(f"Collection '{collection_name}' not found in sysdb — skip patch.")
            return False

        cid, name, conf = row[0], row[1], row[2]
        if _config_has_type(conf):
            _log("configuration already has _type — skip patch.")
        else:
            # 最小可用 JSON（让 0.5.x 能反序列化；其余参数走默认）
            minimal = {"_type": "CollectionConfigurationInternal"}
            conf_json = json.dumps(minimal, ensure_ascii=False)
            cur.execute(
                "UPDATE collections SET configuration = ? WHERE id = ?",
                (conf_json, cid)
            )
            con.commit()
            patched = True
            _log(f"Patched configuration for collection '{name}' (_type injected).")
    except Exception as e:
        _log("Patch failed:", repr(e))
        patched = False
    finally:
        with contextlib.suppress(Exception):
            con.close()
    return patched

def get_chroma_collection():
    """
    获取（或创建）指定集合；如果触发 KeyError: '_type'，自动补丁一次再重试。
    """
    global _collection
    if _collection is not None:
        return _collection

    client = get_chroma_client()
    embed_fn = _ChromaEmbedFn(_EMBED)
    target = COLLECTION_NAME

    # 第一次尝试
    try:
        _collection = client.get_or_create_collection(name=target, embedding_function=embed_fn)
        return _collection
    except KeyError as e:
        # 专门捕获老库配置缺少 _type 的情况
        if str(e).strip("'") == "_type":
            _log("Detected legacy configuration without _type; applying one-shot patch...")
            ok = _patch_chroma_configuration_sqlite(PERSIST_DIR, target)
            if not ok:
                _log("Patch did not apply or failed; re-raising the original error.")
                raise
            # 补丁后重试一次
            _collection = client.get_or_create_collection(name=target, embedding_function=embed_fn)
            _log("Collection opened after patch.")
            return _collection
        # 其它 KeyError 原样抛出
        raise
    except Exception as e:
        _log("get_or_create_collection error:", repr(e))
        raise

# ---------------------------
# 指南向量检索
# ---------------------------
def search_guidelines(query: str, k: int = 5, include_metadata: bool = True) -> List[Dict[str, Any]]:
    """
    对指南集合进行语义检索。
    返回命中列表：[{id, score, text, meta}, ...]
    """
    if not query:
        return []
    col = get_chroma_collection()

    # chromadb >=0.5: 使用 query()
    # 注意：embedding_function 已在集合级别提供，这里直接传文本
    res = col.query(
        query_texts=[query],
        n_results=k,
        include=["documents", "metadatas", "distances", "embeddings"] if include_metadata else ["documents"]
    )
    # 解析结果
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
            "score": 1.0 - (dists[i] if i < len(dists) else 0.0)  # 简单把 distance 转为相似度感知
        }
        out.append(item)
    return out

# ---------------------------
# （可选）SQLite 药品库检索（关键词/LIKE 示例）
# ---------------------------
def _sqlite_search_drugfacts(q: str, topn: int = 5) -> List[Dict[str, Any]]:
    """
    对 ./db/drugs.sqlite 进行一个非常朴素的 LIKE 检索，示例用途：
      表结构假设：drugs(name TEXT, indications TEXT, contraindications TEXT, interactions TEXT, pregnancy TEXT, source TEXT)
    你可根据自己真实库结构改写此函数（或在 pipeline 调用端关掉）。
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
            "score": 0.5,  # 关键词命中给一个中性分数，混合时再归一
        })
    return hits

# ---------------------------
# 混合检索（RRF 简化实现）
# ---------------------------
def _rrf_rank(lists: List[List[Dict[str, Any]]], k: int, k_rrf: float = 60.0) -> List[Dict[str, Any]]:
    """
    Reciprocal Rank Fusion (简化)：对多个候选列表做融合打分。
    lists: 各通道的命中列表（顺序表示初排）
    k: 最终返回数
    """
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
        it = dict(bag[_id])
        it["rrf"] = s
        merged.append(it)
    merged.sort(key=lambda x: x.get("rrf", 0.0), reverse=True)
    return merged[:k]

def hybrid_search(query: str,
                  k: int = 8,
                  k_guideline: Optional[int] = None,
                  k_sqlite: Optional[int] = None,
                  use_sqlite: bool = True) -> List[Dict[str, Any]]:
    """
    混合检索：指南向量检索 + （可选）SQLite 关键词检索，RRF 融合。
    """
    k_guideline = k_guideline or max(3, k)
    k_sqlite = k_sqlite or max(3, k // 2)

    g_hits = search_guidelines(query, k=k_guideline)
    s_hits: List[Dict[str, Any]] = _sqlite_search_drugfacts(query, topn=k_sqlite) if use_sqlite else []

    # 若某一路为空，就退化为另一路
    if g_hits and s_hits:
        return _rrf_rank([g_hits, s_hits], k=k)
    return (g_hits or s_hits)[:k]

# ---------------------------
# 命令行入口（便于快速自测）
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
    ap = argparse.ArgumentParser(description="CareMind RAG Retriever (with legacy Chroma auto-patch)")
    ap.add_argument("--q", "--query", dest="query", type=str, required=True, help="query text")
    ap.add_argument("--topn", type=int, default=8, help="number of results to show")
    ap.add_argument("--method", type=str, default="rrf", choices=["guideline", "sqlite", "rrf"], help="search method")
    ap.add_argument("--no-sqlite", action="store_true", help="disable sqlite side search")
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