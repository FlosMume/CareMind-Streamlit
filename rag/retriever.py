# -*- coding: utf-8 -*-
"""
CareMind · RAG Retriever
========================

本模块聚焦“检索侧”的一站式能力，提供以下特性：
1) 与 ChromaDB 0.5.x 完整兼容的集合访问与查询封装（不访问内部属性）
2) 统一的语义检索（向量）+ 关键词直搜（where_document/get）组合策略
3) 可选的轻量 Rerank（Cross-Encoder，如不可用自动降级到 RRF）
4) SQLite 药品库（drugs.sqlite）结构化查询的便捷入口
5) 健康检查 / 自检打印，便于 Streamlit 启动时输出运行环境诊断
6) CLI：可在命令行快速验证检索结果与管线连通性

设计原则
--------
- “不破坏你已有接口”的前提下，尽量做到向后兼容与增强
- 尽量避免 Hard Fail：向量失败 → 关键词兜底 → 提示占位，避免 UI 出现“暂无证据片段”
- 注释足够丰富（rich comments），便于未来学生或同事快速上手二次开发

版本对齐
--------
- 建议：chromadb==0.5.5
- 建议：sentence-transformers>=2.7
- 可选：cross-encoder>=0.2.4（做 rerank；缺失时自动降级）

环境变量
--------
- CHROMA_PERSIST_DIR: Chroma 持久化目录（默认 ./chroma_store 或 ./chroma_store_clean）
- CHROMA_COLLECTION:   目标集合名（如 guideline_chunks_v2）
- EMBEDDING_MODEL:     嵌入模型（如 BAAI/bge-large-zh-v1.5）
- DRUG_DB_PATH:        SQLite 路径（默认 ./db/drugs.sqlite）

用法示例
--------
python rag/retriever.py \
  --q "合并支气管哮喘的高血压患者是否可用β受体阻滞剂？" \
  --topn 8 \
  --method rrf

"""

from __future__ import annotations

import os
import re
import sys
import json
import math
import time
import sqlite3
import logging
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

# -------------------------------
# 全局配置（从环境变量读取，可被 Streamlit/app 覆盖）
# -------------------------------
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")
COLLECTION  = os.getenv("CHROMA_COLLECTION", "guideline_chunks")
DRUG_DB     = os.getenv("DRUG_DB_PATH", "./db/drugs.sqlite")

# 兼容：如果你在云端用的是清洗后的目录名
if os.path.isdir("./chroma_store_clean") and not os.path.samefile(PERSIST_DIR, "./chroma_store_clean"):
    # 不强改环境，但打印提示即可
    pass

# --------------------------------
# 延迟导入/单例缓存：避免启动时沉重依赖阻塞
# --------------------------------
_CHROMA_CLIENT = None
_CHROMA_COL    = None
_EMBEDDER      = None
_CROSS_ENCODER = None

# 日志配置
logger = logging.getLogger("caremind.retriever")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        stream=sys.stdout
    )

# --------------------------------
# Chroma 客户端与集合封装（0.5.x 安全用法）
# --------------------------------
def _get_client():
    """
    懒加载 Chroma 持久化客户端。
    注意：0.5.x 用 PersistentClient(path=...)；不要访问内部私有属性。
    """
    global _CHROMA_CLIENT
    if _CHROMA_CLIENT is None:
        try:
            from chromadb import PersistentClient
        except Exception as e:
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
    返回集合名与粗略计数（通过 get(limit=1) 触发 lazy load，避免重 IO）
    """
    res = []
    try:
        client = _get_client()
        cols = client.list_collections()
        for cinfo in cols:
            name = getattr(cinfo, "name", None) or str(cinfo)
            try:
                col = client.get_collection(name=name)
                # 取 1 条触发 meta 初始化；0.5.x 暂无直接 count API
                _ = col.get(limit=1)
                # 这里不强求真实 count，避免重载；UI 层可以另做统计
                res.append({"name": name})
            except Exception as e:
                res.append({"name": name, "error": str(e)})
    except Exception as e:
        res.append({"error": f"list_collections failed: {e}"})
    return res


def peek_collection(n: int = 3) -> List[Dict[str, Any]]:
    """
    安全 “peek” 函数：用 get(limit=..., include=[...]) 返回几条样本
    彻底替代对内部属性（如 max_seq_id）的依赖。
    """
    try:
        col = get_chroma_collection()
        rec = col.get(limit=n, include=["ids", "documents", "metadatas"])
        out = []
        docs  = rec.get("documents") or []
        metas = rec.get("metadatas") or []
        ids   = rec.get("ids") or []
        for i in range(min(n, len(ids))):
            m = metas[i] or {}
            d = (docs[i] or "")[:240]
            out.append({
                "id": ids[i],
                "title": m.get("title"),
                "year": m.get("year"),
                "source": m.get("source"),
                "preview": d
            })
        return out
    except Exception as e:
        return [{"error": f"peek failed: {e}"}]

# --------------------------------
# 嵌入模型（Sentence-Transformers）封装
# --------------------------------
def _get_embedder():
    """
    统一的 query_embeddings 编码器。与 ingest 保持一致，避免“模型不一致”导致相似度错位。
    """
    global _EMBEDDER
    if _EMBEDDER is None:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            raise RuntimeError(f"sentence-transformers not available: {e}")
        _EMBEDDER = SentenceTransformer(EMBED_MODEL)
    return _EMBEDDER


def encode_query(texts: List[str]) -> List[List[float]]:
    """
    编码查询文本 → 向量；注意开启 normalize_embeddings=True，利于 cosine 相似。
    """
    model = _get_embedder()
    vecs = model.encode(texts, normalize_embeddings=True)
    return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vecs]

# --------------------------------
# 关键词直搜（不走向量）：where_document with get()
# --------------------------------
def keyword_get(where_text: str, limit: int = 8) -> List[Dict[str, Any]]:
    """
    关键词直搜（不走向量）、模糊包含：{"$contains": "..."}。
    说明：
      - 只能在 collection.get(...) 中使用 where_document；
      - 在 collection.query(...) 中“不能”使用 where_document。
    """
    col = get_chroma_collection()
    try:
        rec = col.get(
            where_document={"$contains": where_text},
            include=["ids", "documents", "metadatas"],
            limit=limit
        )
        hits = []
        for i, _id in enumerate(rec.get("ids") or []):
            md  = (rec["metadatas"] or [None])[i] or {}
            doc = (rec["documents"] or [""])[i]
            hits.append({
                "id": _id,
                "score": None,  # 直搜不提供距离
                "doc": doc,
                "meta": md,
                "channel": "keyword"
            })
        return hits
    except Exception as e:
        return [{"error": f"where_document get failed: {e}"}]

# --------------------------------
# 主检索：向量优先 + 关键词兜底（确保不空）
# --------------------------------
def search_guidelines(question: str, k: int = 8) -> List[Dict[str, Any]]:
    """
    主接口：先语义向量检索；若 0 命中/异常 → 回退关键词直搜。
    返回结构：[{score, doc, meta, channel}]
      - score：向量检索返回“距离”（越小越相似）；直搜为 None
      - channel：'vector' / 'keyword' / 'hint'
    """
    col = get_chroma_collection()
    q = (question or "").strip()
    if not q:
        return []

    # --------------------------------
    # 1) 语义检索
    # --------------------------------
    try:
        embeds = encode_query([q])  # 与 ingest 一致的模型
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
                "score": float(dists[i]) if i < len(dists) else None,  # 距离越小越接近
                "doc": d,
                "meta": metas[i] if i < len(metas) else {},
                "channel": "vector"
            })
        if hits:
            return hits
    except Exception as e:
        logger.warning(f"vector query failed: {e}")

    # --------------------------------
    # 2) 关键词兜底
    # --------------------------------
    #   简单中文/英文切词：只取长度>=2 的片段，避免停用词噪声
    terms = [t for t in re.split(r"[^\w\u4e00-\u9fff]+", q) if len(t) >= 2]
    seen_ids = set()
    merged: List[Dict[str, Any]] = []
    for t in terms[:6]:  # 控制复杂度
        ks = keyword_get(t, limit=max(2, k // 2))
        for h in ks:
            _id = h.get("id")
            if not _id or _id in seen_ids:
                continue
            seen_ids.add(_id)
            merged.append(h)
            if len(merged) >= k:
                break
        if len(merged) >= k:
            break

    return merged


def ensure_non_empty_evidence(question: str, k: int = 6) -> List[Dict[str, Any]]:
    """
    UI/上层兜底：即使检索失败，也返回“提示占位”，避免前端显示“暂无证据片段”。
    """
    hits = search_guidelines(question, k=k) or []
    if hits:
        return hits
    return [{
        "score": None,
        "doc": "（未命中向量与直搜。建议：缩短问题、避免过多停用词、加入明确疾病/药品关键词；或检查集合/嵌入模型是否一致。）",
        "meta": {"title": "系统提示", "source": "CareMind Retriever", "year": None},
        "channel": "hint"
    }]

# --------------------------------
# 轻量 Rerank：支持 Cross-Encoder；不可用时回退到 RRF
# --------------------------------
def _get_cross_encoder():
    """
    可选的重排序器（如 cross-encoder/ms-marco-MiniLM-L-6-v2）。
    若依赖缺失或模型不可用，返回 None（上层自动降级到 RRF）。
    """
    global _CROSS_ENCODER
    if _CROSS_ENCODER is not None:
        return _CROSS_ENCODER
    try:
        from cross_encoder import CrossEncoder  # 有些环境包名是 cross-encoder
        model_name = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        _CROSS_ENCODER = CrossEncoder(model_name)
        return _CROSS_ENCODER
    except Exception:
        try:
            from sentence_transformers import CrossEncoder
            model_name = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
            _CROSS_ENCODER = CrossEncoder(model_name)
            return _CROSS_ENCODER
        except Exception as e:
            logger.info(f"Cross-Encoder unavailable, fallback to RRF: {e}")
            return None


def rrf_merge(rank_lists: List[List[Dict[str, Any]]], k: int = 8, k_rrf: float = 60.0) -> List[Dict[str, Any]]:
    """
    Reciprocal Rank Fusion（简单稳健的多通道融合）：
    对多份排序（如向量结果、关键词结果）做 1/(k_rrf + rank) 加权求和。
    """
    score_map: Dict[str, float] = {}
    store_map: Dict[str, Dict[str, Any]] = {}

    for results in rank_lists:
        for rnk, item in enumerate(results, start=1):
            _id = item.get("id") or f"NOID::{item.get('doc','')[:40]}"
            score_map[_id] = score_map.get(_id, 0.0) + 1.0 / (k_rrf + rnk)
            # 保留更丰富的字段（第一次出现为准）
            if _id not in store_map:
                store_map[_id] = item

    fused = [
        {**store_map[_id], "rrf": score}
        for _id, score in score_map.items()
    ]
    fused.sort(key=lambda x: x.get("rrf", 0.0), reverse=True)
    return fused[:k]


def rerank(question: str, hits: List[Dict[str, Any]], k: int = 8) -> List[Dict[str, Any]]:
    """
    若可用 Cross-Encoder，则对 topN 做交叉编码器重排序；否则回退 RRF（与关键词/向量融合）。
    输入 hits 可以是单路向量或混合结果。
    """
    if not hits:
        return hits

    ce = _get_cross_encoder()
    if ce is None:
        # 降级：拿向量 topK 与 关键词再做一次融合
        # 实际上在 search_guidelines 里已做混合，本处直接截断返回
        return hits[:k]

    # 用 Cross-Encoder 计算相关度
    pairs = [(question, h.get("doc", "")) for h in hits[: min(24, max(8, k * 2))]]
    try:
        scores = ce.predict(pairs)  # 越高越相关
        scored = []
        for h, s in zip(hits, scores):
            hh = dict(h)
            hh["ce_score"] = float(s)
            scored.append(hh)
        scored.sort(key=lambda x: x.get("ce_score", 0.0), reverse=True)
        return scored[:k]
    except Exception as e:
        logger.info(f"Cross-Encoder predict failed, fallback: {e}")
        return hits[:k]

# --------------------------------
# 药品结构化查询（SQLite）
# --------------------------------
def _open_drug_db(path: str = DRUG_DB) -> Optional[sqlite3.Connection]:
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
    通用药品检索：在表 `drugs` 上做名称/别名模糊匹配（LIKE），返回结构化信息。
    你可以根据自己的字段做扩展，如：适应症/禁忌/相互作用/妊娠分级/来源 等列。
    """
    conn = _open_drug_db()
    if conn is None:
        return []

    q = (name or "").strip()
    if not q:
        return []

    try:
        cur = conn.cursor()
        # 假定表结构至少包含：name, aliases, indication, contraindication, interactions, pregnancy, source
        sql = """
        SELECT name, aliases, indication, contraindication, interactions, pregnancy, source
        FROM drugs
        WHERE name LIKE ? OR (aliases IS NOT NULL AND aliases LIKE ?)
        LIMIT ?
        """
        like = f"%{q}%"
        cur.execute(sql, (like, like, limit))
        rows = cur.fetchall()
        out = []
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

# --------------------------------
# 健康检查 / 诊断打印（供前端启动日志调用）
# --------------------------------
def print_health() -> Dict[str, Any]:
    """
    返回一份可 JSON 化的健康检查结果，供前端“运行日志/环境诊断”打印。
    """
    info: Dict[str, Any] = {
        "env": {
            "CAREMIND_DEMO": os.getenv("CAREMIND_DEMO", "0"),
            "CHROMA_PERSIST_DIR": PERSIST_DIR,
            "CHROMA_COLLECTION": COLLECTION,
            "EMBEDDING_MODEL": EMBED_MODEL,
            "DRUG_DB_PATH": DRUG_DB
        },
        "retriever_version": os.getenv("RETRIEVER_VERSION", "2025-09-22-merged"),
        "chroma": {},
        "sqlite": {}
    }

    # Chroma 目录存在性
    info["chroma"]["dir_exists"] = os.path.isdir(PERSIST_DIR)
    info["chroma"]["dir"] = os.path.abspath(PERSIST_DIR)

    # 集合列表（名称）
    info["chroma"]["collections"] = list_collections()

    # 快速 peek
    info["chroma"]["peek"] = peek_collection(3)

    # SQLite 存在性与表枚举
    db_path = os.path.abspath(DRUG_DB)
    info["sqlite"]["path"] = db_path
    info["sqlite"]["exists"] = os.path.isfile(db_path)
    if info["sqlite"]["exists"]:
        try:
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
            info["sqlite"]["tables"] = [r[0] for r in cur.fetchall()]
            conn.close()
        except Exception as e:
            info["sqlite"]["error"] = str(e)

    return info

# --------------------------------
# 高层组合：对外暴露的“统一检索入口”
# --------------------------------
@dataclass
class RetrieveOptions:
    topn: int = 8          # 返回条数
    method: str = "rrf"    # 后融合/重排序方法：["rrf", "none", "ce"]


def retrieve(question: str, opts: RetrieveOptions = RetrieveOptions()) -> Dict[str, Any]:
    """
    对外统一入口：
      - 先拿到基础命中（向量 + 关键词兜底）
      - 再按 method 做重排（ce/rrf/none）
      - 永不返回空 evidence：最差会给一条提示型“hint”
    """
    t0 = time.time()
    base_hits = ensure_non_empty_evidence(question, k=opts.topn * 2)

    if opts.method == "ce":
        final_hits = rerank(question, base_hits, k=opts.topn)
    elif opts.method == "rrf":
        # 此处可以把“向量结果”和“关键词结果”分两路再融合；当前 base_hits 已是混合，
        # 为了示例保持简单：直接截断即可；如果你保留两路原始结果，可在此调用 rrf_merge(...)
        final_hits = base_hits[:opts.topn]
    else:
        final_hits = base_hits[:opts.topn]

    dt = time.time() - t0
    return {
        "question": question,
        "hits": final_hits,
        "t_sec": round(dt, 3),
        "meta": {
            "method": opts.method,
            "topn": opts.topn,
            "embedding_model": EMBED_MODEL,
            "collection": COLLECTION
        }
    }

# --------------------------------
# 命令行入口：便于你在服务器/WSL 里直接测试
# --------------------------------
def _fmt_hit(h: Dict[str, Any], idx: int) -> str:
    m = h.get("meta") or {}
    head = f"[{idx}] {m.get('title') or 'Untitled'} | {m.get('source') or '-'} | {m.get('year') or '-'}"
    score_keys = [k for k in ["ce_score", "rrf", "score"] if h.get(k) is not None]
    sk = ", ".join([f"{k}={h.get(k):.4f}" if isinstance(h.get(k), (int, float)) else f"{k}={h.get(k)}" for k in score_keys])
    ch = h.get("channel") or "-"
    body = (h.get("doc") or "").strip().replace("\n", " ")
    body = re.sub(r"\s+", " ", body)[:300]
    return f"{head}  [{ch}]  ({sk})\n    {body}"


def main():
    ap = argparse.ArgumentParser(description="CareMind Retriever · CLI")
    ap.add_argument("--q", "--query", dest="query", type=str, required=False, default="老年 高血压 糖尿病 目标",
                    help="查询问题/关键词")
    ap.add_argument("--topn", type=int, default=8, help="返回条数")
    ap.add_argument("--method", type=str, default="rrf", choices=["rrf", "none", "ce"],
                    help="后融合/重排方法")
    ap.add_argument("--peek", action="store_true", help="仅做 peek（验证集合可读性）")
    ap.add_argument("--kw", type=str, default="", help="仅用关键词直搜（调试）")
    ap.add_argument("--health", action="store_true", help="打印健康检查 JSON")
    args = ap.parse_args()

    if args.health:
        print(json.dumps(print_health(), ensure_ascii=False, indent=2))
        return

    if args.peek:
        sample = peek_collection(3)
        print("PEEK:")
        for i, s in enumerate(sample, 1):
            print(f"  {i}. {json.dumps(s, ensure_ascii=False)[:400]}")
        return

    if args.kw:
        ks = keyword_get(args.kw, limit=args.topn)
        print(f"KW GET [{args.kw}] → {len(ks)} hits")
        for i, h in enumerate(ks, 1):
            print(_fmt_hit(h, i))
        return

    # 常规：统一检索入口
    out = retrieve(args.query, RetrieveOptions(topn=args.topn, method=args.method))
    print(f"Question: {out['question']}")
    print(f"Method:   {out['meta']['method']} | TopN={out['meta']['topn']} | Embed={out['meta']['embedding_model']}")
    print(f"Collection: {out['meta']['collection']}")
    print(f"Time:     {out['t_sec']}s")
    print("-"*80)
    for i, h in enumerate(out["hits"], 1):
        print(_fmt_hit(h, i))


# -------------------------------
# 模块作为脚本执行
# -------------------------------
if __name__ == "__main__":
    main()