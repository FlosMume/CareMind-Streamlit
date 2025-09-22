# -*- coding: utf-8 -*-
"""
CareMind · RAG Retriever (Full, Rich-Commented)
================================================
版本: 2025-09-22-merged

本模块提供 CareMind 项目“检索侧”的所有能力与统一入口，包括：
  1) ChromaDB 0.5.x 兼容的集合访问/查询（不依赖内部私有属性）
  2) 查询端自编码 query_embeddings（与 ingest 模型一致，避免 0 命中）
  3) 关键词直搜（where_document 只能配合 collection.get(...) 使用）
  4) 双路检索策略：向量优先 → 关键词兜底，确保“证据不为空”
  5) 可选 Cross-Encoder 重排（不可用时降级）
  6) 本地 SQLite 药品库查询（结构化字段）
  7) 健康检查/环境诊断（供前端日志面板使用）
  8) 命令行 CLI（便于在服务器/WSL 直接验证）

重要提示
--------
- 本文件**最顶部**含有 sqlite3 版本热修（HOTFIX），务必保留在最上方，确保在任何 chromadb 导入之前执行。
- where_document 直搜**只能**用于 collection.get(...)，切勿塞入 collection.query(...)。
- 如果你的 ingest 阶段使用的嵌入模型是 "BAAI/bge-large-zh-v1.5"，查询端也要一致；否则相似度会错位。
"""

from __future__ import annotations

# ---------------------------------------------------------------------
# HOTFIX: 强制使用“足够新的” sqlite3（Chroma 要求 sqlite3 >= 3.35.0）
# 必须放在任何 chromadb 导入之前；否则 Chroma 先导入旧 sqlite3 就会报错。
# ---------------------------------------------------------------------
import sys as _sys
try:
    import sqlite3 as _sqlite3_check  # 初步尝试系统自带
    from sqlite3 import sqlite_version as _sv
    _maj, _min, _pat = (int(x) for x in (_sv.split(".") + ["0", "0"])[:3])
    if (_maj, _min, _pat) < (3, 35, 0):
        raise ImportError(f"sqlite3 too old: { _sv }")
    # 若版本足够新，继续；否则进入 except 分支
except Exception:
    # 使用 manylinux 的 pysqlite3-binary 并别名为 sqlite3
    import pysqlite3 as sqlite3  # type: ignore
    _sys.modules["sqlite3"] = sqlite3
else:
    # 让后续代码都统一 from sqlite3 import ... 即可
    import sqlite3  # noqa: E402

# ---------------------------------------------------------------------
# 常规标准库
# ---------------------------------------------------------------------
import os
import re
import json
import math
import time
import logging
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------
# 全局配置（从环境变量读取，可被上层覆盖）
# ---------------------------------------------------------------------
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")
COLLECTION  = os.getenv("CHROMA_COLLECTION", "guideline_chunks")
DRUG_DB     = os.getenv("DRUG_DB_PATH", "./db/drugs.sqlite")
RETRIEVER_VERSION = os.getenv("RETRIEVER_VERSION", "2025-09-22-merged")

# ---------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------
logger = logging.getLogger("caremind.retriever")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

# ---------------------------------------------------------------------
# 懒加载单例（避免启动卡顿；Streamlit 每次热刷新也会重复 import）
# ---------------------------------------------------------------------
_CHROMA_CLIENT = None
_CHROMA_COL    = None
_EMBEDDER      = None
_CROSS_ENCODER = None

# ---------------------------------------------------------------------
# Chroma 客户端与集合（0.5.x 安全用法）
# ---------------------------------------------------------------------
def _get_client():
    """
    懒加载 Chroma 持久化客户端；不访问私有属性。
    注意：确保在本文件顶部的 sqlite3 HOTFIX 之后再导入 chromadb
    """
    global _CHROMA_CLIENT
    if _CHROMA_CLIENT is None:
        try:
            from chromadb import PersistentClient
        except Exception as e:
            # 把错误完整抛上去，便于前端展示
            raise RuntimeError(f"Chroma import failed: {e}")
        _CHROMA_CLIENT = PersistentClient(path=PERSIST_DIR)
    return _CHROMA_CLIENT


def get_chroma_collection():
    """
    获取/创建集合：0.5.x 推荐 get_or_create_collection(name=...)
    """
    global _CHROMA_COL
    if _CHROMA_COL is not None:
        return _CHROMA_COL
    client = _get_client()
    _CHROMA_COL = client.get_or_create_collection(name=COLLECTION)
    return _CHROMA_COL


def list_collections() -> List[Dict[str, Any]]:
    """
    列出集合（仅返回名称/错误），避免重 IO。
    """
    out: List[Dict[str, Any]] = []
    try:
        client = _get_client()
        cols = client.list_collections()
        for cinfo in cols:
            name = getattr(cinfo, "name", None) or str(cinfo)
            try:
                _ = client.get_collection(name=name).get(limit=1)  # 触发 lazy init
                out.append({"name": name})
            except Exception as e:
                out.append({"name": name, "error": str(e)})
    except Exception as e:
        out.append({"error": f"list_collections failed: {e}"})
    return out


def peek_collection(n: int = 3) -> List[Dict[str, Any]]:
    """
    安全“抽样”：通过 get(limit=..., include=...) 读取几条并返回简要信息。
    彻底替代旧版 peek 对内部属性（如 max_seq_id）的访问。
    """
    try:
        col = get_chroma_collection()
        rec = col.get(limit=n, include=["ids", "documents", "metadatas"])
        ids   = rec.get("ids") or []
        docs  = rec.get("documents") or []
        metas = rec.get("metadatas") or []
        out: List[Dict[str, Any]] = []
        for i in range(min(n, len(ids))):
            m = metas[i] or {}
            d = (docs[i] or "").replace("\n", " ")
            out.append({
                "id": ids[i],
                "title": m.get("title"),
                "year": m.get("year"),
                "source": m.get("source"),
                "preview": d[:240] + ("…" if len(d) > 240 else "")
            })
        return out
    except Exception as e:
        return [{"error": f"peek failed: {e}"}]

# ---------------------------------------------------------------------
# 嵌入与编码（与 ingest 模型保持一致）
# ---------------------------------------------------------------------
def _get_embedder():
    """
    统一查询端编码器；与 ingest 模型一致，避免相似度错位。
    """
    global _EMBEDDER
    if _EMBEDDER is None:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            raise RuntimeError(f"sentence-transformers unavailable: {e}")
        _EMBEDDER = SentenceTransformer(EMBED_MODEL)
    return _EMBEDDER


def encode_query(texts: List[str]) -> List[List[float]]:
    """
    将查询文本编码为向量（normalize_embeddings=True 利于 cosine）。
    """
    model = _get_embedder()
    vecs = model.encode(texts, normalize_embeddings=True)
    return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vecs]

# ---------------------------------------------------------------------
# 关键词直搜（不走向量）：collection.get(where_document=...)
# ---------------------------------------------------------------------
def keyword_get(where_text: str, limit: int = 8) -> List[Dict[str, Any]]:
    """
    关键词直搜（不走向量）：
      - 0.5.x 的 where_document 只能搭配 collection.get(...)
      - 模糊匹配：{"$contains": "..."}
    """
    col = get_chroma_collection()
    try:
        rec = col.get(
            where_document={"$contains": where_text},
            include=["ids", "documents", "metadatas"],
            limit=limit
        )
        hits: List[Dict[str, Any]] = []
        ids   = rec.get("ids") or []
        docs  = rec.get("documents") or []
        metas = rec.get("metadatas") or []
        for i, _id in enumerate(ids):
            md = metas[i] or {}
            dc = docs[i] or ""
            hits.append({
                "id": _id,
                "doc": dc,
                "meta": md,
                "score": None,       # 直搜无距离分数
                "channel": "keyword"
            })
        return hits
    except Exception as e:
        return [{"error": f"where_document get failed: {e}"}]

# ---------------------------------------------------------------------
# 主检索：向量优先 + 关键词兜底（确保结果不空）
# ---------------------------------------------------------------------
def search_guidelines(question: str, k: int = 8) -> List[Dict[str, Any]]:
    """
    先做向量检索；若失败或 0 命中，则用关键词直搜兜底。
    返回结构：[ {score, doc, meta, channel} ... ]
      - 向量检索：channel="vector"，score=距离（越小越相似）
      - 关键词直搜：channel="keyword"，score=None
    """
    col = get_chroma_collection()
    q = (question or "").strip()
    if not q:
        return []

    # 1) 向量检索
    try:
        embeds = encode_query([q])
        res = col.query(
            query_embeddings=embeds,
            n_results=max(1, k),
            include=["documents", "metadatas", "distances"],
        )
        docs  = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        hits: List[Dict[str, Any]] = []
        for i, d in enumerate(docs):
            hits.append({
                "score": float(dists[i]) if i < len(dists) else None,
                "doc": d,
                "meta": metas[i] if i < len(metas) else {},
                "channel": "vector"
            })
        if hits:
            return hits
    except Exception as e:
        logger.warning(f"vector query failed: {e}")

    # 2) 关键词兜底：简单切词（中英混合）
    terms = [t for t in re.split(r"[^\w\u4e00-\u9fff]+", q) if len(t) >= 2]
    seen = set()
    merged: List[Dict[str, Any]] = []
    for t in terms[:6]:
        ks = keyword_get(t, limit=max(2, k // 2))
        for h in ks:
            _id = h.get("id")
            if not _id or _id in seen:
                continue
            seen.add(_id)
            merged.append(h)
            if len(merged) >= k:
                break
        if len(merged) >= k:
            break
    return merged


def ensure_non_empty_evidence(question: str, k: int = 6) -> List[Dict[str, Any]]:
    """
    UI 兜底：即便检索失败，也返回“提示占位”，避免前端显示空白。
    """
    hits = search_guidelines(question, k=k) or []
    if hits:
        return hits
    return [{
        "score": None,
        "doc": "（未命中向量与直搜。建议：缩短问题、加入关键实体词；或检查集合名与嵌入模型是否一致。）",
        "meta": {"title": "系统提示", "source": "CareMind Retriever"},
        "channel": "hint"
    }]

# ---------------------------------------------------------------------
# 轻量重排（可选 Cross-Encoder；不可用时降级）
# ---------------------------------------------------------------------
def _get_cross_encoder():
    """
    返回 Cross-Encoder 实例；不可用时返回 None（上层降级）。
    优先尝试独立包 cross-encoder，然后尝试 sentence-transformers 内置 CrossEncoder。
    """
    global _CROSS_ENCODER
    if _CROSS_ENCODER is not None:
        return _CROSS_ENCODER
    model_name = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    try:
        from cross_encoder import CrossEncoder  # 独立包
        _CROSS_ENCODER = CrossEncoder(model_name)
        return _CROSS_ENCODER
    except Exception:
        try:
            from sentence_transformers import CrossEncoder  # ST 内置
            _CROSS_ENCODER = CrossEncoder(model_name)
            return _CROSS_ENCODER
        except Exception as e:
            logger.info(f"Cross-Encoder unavailable, fallback to base ranking: {e}")
            return None


def rrf_merge(rank_lists: List[List[Dict[str, Any]]], k: int = 8, k_rrf: float = 60.0) -> List[Dict[str, Any]]:
    """
    Reciprocal Rank Fusion（RRR）：对若干个排序列表做 1/(k_rrf + rank) 融合。
    用于多通道融合（如向量 + 关键词）。
    """
    score_map: Dict[str, float] = {}
    store_map: Dict[str, Dict[str, Any]] = {}

    for results in rank_lists:
        for rnk, item in enumerate(results, start=1):
            _id = item.get("id") or f"NOID::{(item.get('doc') or '')[:40]}"
            score_map[_id] = score_map.get(_id, 0.0) + 1.0 / (k_rrf + rnk)
            if _id not in store_map:
                store_map[_id] = item

    fused = [{**store_map[_id], "rrf": sc} for _id, sc in score_map.items()]
    fused.sort(key=lambda x: x.get("rrf", 0.0), reverse=True)
    return fused[:k]


def rerank(question: str, hits: List[Dict[str, Any]], k: int = 8) -> List[Dict[str, Any]]:
    """
    若可用 CE，则用 CE 对 topN 重排；否则返回截断。
    """
    if not hits:
        return hits
    ce = _get_cross_encoder()
    if ce is None:
        return hits[:k]
    try:
        pairs = [(question, h.get("doc", "")) for h in hits[: max(24, k * 2)]]
        scores = ce.predict(pairs)  # 越高越好
        aug = []
        for h, s in zip(hits, scores):
            hh = dict(h)
            hh["ce_score"] = float(s)
            aug.append(hh)
        aug.sort(key=lambda x: x.get("ce_score", 0.0), reverse=True)
        return aug[:k]
    except Exception as e:
        logger.info(f"Cross-Encoder predict failed, fallback: {e}")
        return hits[:k]

# ---------------------------------------------------------------------
# 药品结构化查询（SQLite）
# ---------------------------------------------------------------------
def _open_drug_db(path: str = DRUG_DB) -> Optional[sqlite3.Connection]:
    """
    打开本地药品数据库；若不存在则返回 None。
    """
    if not path or not os.path.isfile(path):
        logger.warning(f"SQLite not found: {path}")
        return None
    try:
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        logger.error(f"open sqlite failed: {e}")
        return None


def lookup_drug(name: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    在表 `drugs` 中按名称/别名模糊检索，返回结构化字段。
    你可根据实际库结构调整 SELECT 列。
    """
    conn = _open_drug_db()
    if conn is None:
        return []
    q = (name or "").strip()
    if not q:
        return []
    try:
        sql = """
        SELECT name, aliases, indication, contraindication, interactions, pregnancy, source
        FROM drugs
        WHERE name LIKE ? OR (aliases IS NOT NULL AND aliases LIKE ?)
        LIMIT ?
        """
        like = f"%{q}%"
        cur = conn.cursor()
        cur.execute(sql, (like, like, limit))
        rows = cur.fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            out.append({
                "name": r["name"],
                "aliases": r["aliases"],
                "indication": r["indication"],
                "contraindication": r["contraindication"],
                "interactions": r["interactions"],
                "pregnancy": r["pregnancy"],
                "source": r["source"],
            })
        return out
    except Exception as e:
        logger.error(f"lookup_drug failed: {e}")
        return []
    finally:
        try:
            conn.close()
        except Exception:
            pass

# ---------------------------------------------------------------------
# 健康检查 / 环境诊断
# ---------------------------------------------------------------------
def print_health() -> Dict[str, Any]:
    """
    返回可 JSON 化的运行时信息，供前端日志面板展示。
    """
    info: Dict[str, Any] = {
        "env": {
            "CAREMIND_DEMO": os.getenv("CAREMIND_DEMO", "0"),
            "CHROMA_PERSIST_DIR": PERSIST_DIR,
            "CHROMA_COLLECTION": COLLECTION,
            "EMBEDDING_MODEL": EMBED_MODEL,
            "DRUG_DB_PATH": DRUG_DB,
            "RETRIEVER_VERSION": RETRIEVER_VERSION,
        },
        "sqlite": {
            "runtime_sqlite_version": getattr(sqlite3, "sqlite_version", "unknown"),
            "path": os.path.abspath(DRUG_DB),
            "exists": os.path.isfile(DRUG_DB),
        },
        "chroma": {
            "dir": os.path.abspath(PERSIST_DIR),
            "dir_exists": os.path.isdir(PERSIST_DIR),
            "collections": [],
            "peek": [],
        }
    }

    # 列集合
    try:
        info["chroma"]["collections"] = list_collections()
    except Exception as e:
        info["chroma"]["collections"] = [{"error": str(e)}]

    # 快速抽样
    try:
        info["chroma"]["peek"] = peek_collection(3)
    except Exception as e:
        info["chroma"]["peek"] = [{"error": str(e)}]

    # SQLite 表列表
    if info["sqlite"]["exists"]:
        try:
            conn = sqlite3.connect(DRUG_DB)
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
            info["sqlite"]["tables"] = [r[0] for r in cur.fetchall()]
            conn.close()
        except Exception as e:
            info["sqlite"]["error"] = str(e)

    return info

# ---------------------------------------------------------------------
# 统一检索入口（供 pipeline/app 调用）
# ---------------------------------------------------------------------
@dataclass
class RetrieveOptions:
    topn: int = 8              # 返回条数
    method: str = "rrf"        # "rrf" / "none" / "ce"


def retrieve(question: str, opts: RetrieveOptions = RetrieveOptions()) -> Dict[str, Any]:
    """
    统一入口：
      - 先基础命中（向量 + 关键词兜底）
      - 再按 method 做重排（ce / rrf / none）
      - 永不返回空 evidence（最差一条“提示占位”）
    """
    t0 = time.time()
    base_hits = ensure_non_empty_evidence(question, k=max(2, opts.topn * 2))

    if opts.method == "ce":
        final_hits = rerank(question, base_hits, k=opts.topn)
    elif opts.method == "rrf":
        # 若你保留两路原始列表，可在此处做真正的 RRF；这里简化为截断（base_hits 已混合）
        final_hits = base_hits[:opts.topn]
    else:
        final_hits = base_hits[:opts.topn]

    return {
        "question": question,
        "hits": final_hits,
        "t_sec": round(time.time() - t0, 3),
        "meta": {
            "method": opts.method,
            "topn": opts.topn,
            "embedding_model": EMBED_MODEL,
            "collection": COLLECTION,
            "retriever_version": RETRIEVER_VERSION
        }
    }

# ---------------------------------------------------------------------
# CLI 调试入口
#   例：
#     python rag/retriever.py --health
#     python rag/retriever.py --peek
#     python rag/retriever.py --kw 哮喘 --topn 5
#     python rag/retriever.py --q "老年 高血压 糖尿病 目标" --topn 8 --method rrf
# ---------------------------------------------------------------------
def _fmt_hit(h: Dict[str, Any], idx: int) -> str:
    m = h.get("meta") or {}
    head = f"[{idx}] {m.get('title') or 'Untitled'} | {m.get('source') or '-'} | {m.get('year') or '-'}"
    ch = h.get("channel") or "-"
    # 分数字段优先级：ce_score > rrf > score
    score_fields = [k for k in ("ce_score", "rrf", "score") if h.get(k) is not None]
    score_str = ", ".join([f"{k}={h.get(k):.4f}" if isinstance(h.get(k),(int,float)) else f"{k}={h.get(k)}" for k in score_fields])
    body = (h.get("doc") or "").strip().replace("\n", " ")
    body = re.sub(r"\s+", " ", body)[:300]
    return f"{head}  [{ch}]  ({score_str})\n    {body}"


def main():
    ap = argparse.ArgumentParser(description="CareMind Retriever · CLI")
    ap.add_argument("--q", "--query", dest="query", type=str, default="老年 高血压 糖尿病 目标", help="查询")
    ap.add_argument("--topn", type=int, default=8, help="返回条数")
    ap.add_argument("--method", type=str, default="rrf", choices=["rrf", "none", "ce"], help="重排方法")
    ap.add_argument("--peek", action="store_true", help="仅抽样集合内容")
    ap.add_argument("--kw", type=str, default="", help="仅关键词直搜（不走向量）")
    ap.add_argument("--health", action="store_true", help="打印健康检查 JSON")
    args = ap.parse_args()

    if args.health:
        print(json.dumps(print_health(), ensure_ascii=False, indent=2))
        return

    if args.peek:
        sample = peek_collection(3)
        print("PEEK:")
        for i, it in enumerate(sample, 1):
            print(f"  {i}. {json.dumps(it, ensure_ascii=False)[:400]}")
        return

    if args.kw:
        ks = keyword_get(args.kw, limit=args.topn)
        print(f"KW GET [{args.kw}] -> {len(ks)} hit(s)")
        for i, h in enumerate(ks, 1):
            print(_fmt_hit(h, i))
        return

    # 常规：统一检索入口
    out = retrieve(args.query, RetrieveOptions(topn=args.topn, method=args.method))
    print(f"Question:  {out['question']}")
    print(f"Method:    {out['meta']['method']} | TopN={out['meta']['topn']}")
    print(f"Embed:     {out['meta']['embedding_model']}")
    print(f"Collection:{out['meta']['collection']}")
    print(f"Version:   {out['meta']['retriever_version']}")
    print(f"Time:      {out['t_sec']}s")
    print("-"*80)
    for i, h in enumerate(out["hits"], 1):
        print(_fmt_hit(h, i))


if __name__ == "__main__":
    main()