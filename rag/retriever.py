# -*- coding: utf-8 -*-
"""
retriever.py  — CareMind RAG 检索层（Chroma 0.5.x 版）

功能要点：
1) 显式与集合绑定相同的 SentenceTransformer 嵌入函数，避免 “384 vs 1024 维度不匹配”。
2) 仅 get_collection，不 create，防止误建空集合导致“无命中假象”。
3) 健壮日志：集合缺失、集合为空、维度不匹配时直观报错并给修复建议。
4) 可选：SQLite FTS5 药品库检索，与向量检索做 RRF 融合（--method rrf）。
5) 兼容脚本运行：python rag/retriever.py --q "..." --method rrf --topn 10
"""

from __future__ import annotations
import os
import sys
import argparse
import importlib
from typing import List, Dict, Any, Optional

# ---------------------------
# sqlite3 兜底别名（云环境常用）
# ---------------------------
try:
    import sqlite3  # type: ignore
except Exception:
    try:
        import pysqlite3 as sqlite3  # type: ignore
        sys.modules["sqlite3"] = sqlite3
        print("ℹ️ retriever: sqlite3 → pysqlite3 别名启用")
    except Exception as e:
        print(f"⚠️ retriever: 无法提供 sqlite3: {e}")

# ---------------------------
# 环境变量与默认值
# ---------------------------
PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "guideline_chunks_1024_v2")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")  # 1024维
DRUG_DB_PATH = os.getenv("DRUG_DB_PATH", "./db/drugs.sqlite")  # 可选



# Add a version string for easier debugging
VERSION = "2025-09-21"

# ---------------------------
# A safe collections lister
# ---------------------------

def list_collections_safe():
    """
    Return [{'name': ..., 'count': int}] without touching private attrs,
    so it won't break on Chroma 0.5.x internals.
    """
    client = _lazy_chroma_client()
    out = []
    try:
        for c in client.list_collections():
            try:
                # Avoid accessing c._collection/_type; just open and count.
                col = client.get_collection(name=c.name, embedding_function=_embedding_fn())
                out.append({"name": c.name, "count": col.count()})
            except Exception:
                out.append({"name": c.name, "count": None})
    except Exception as e:
        raise RuntimeError(f"list_collections failed: {e}")
    return out


# ---------------------------
# Chroma 客户端 & 嵌入函数
# ---------------------------
def _lazy_chroma_client():
    chroma = importlib.import_module("chromadb")
    from chromadb import PersistentClient
    ver = getattr(chroma, "__version__", "unknown")
    print(f"[Retriever] Chroma v{ver} | dir={PERSIST_DIR}")
    return PersistentClient(path=PERSIST_DIR)

def _embedding_fn():
    from chromadb.utils import embedding_functions
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )

def get_chroma_collection():
    client = _lazy_chroma_client()
    try:
        names = [c.name for c in client.list_collections()]
    except Exception as e:
        raise RuntimeError(f"❌ list_collections 失败：{e}")
    if COLLECTION_NAME not in names:
        raise RuntimeError(f"❌ 找不到集合《{COLLECTION_NAME}》。可用集合：{names}")

    try:
        col = client.get_collection(name=COLLECTION_NAME, embedding_function=_embedding_fn())
    except Exception as e:
        raise RuntimeError(f"❌ 打开集合失败：{e}")

    n = col.count()
    if n == 0:
        raise RuntimeError(f"❌ 集合《{COLLECTION_NAME}》为空。请检查 ingest 是否写入成功。")
    print(f"[Retriever] 打开集合《{COLLECTION_NAME}》 | docs={n} | embed={EMBED_MODEL}")
    return col

# ---------------------------
# 元数据归一化
# ---------------------------
def _normalize_meta(m: dict) -> dict:
    """把常见别名归一化到 UI 需要的字段：source/title/year/page/chunk_id。"""
    if not isinstance(m, dict):
        return {}
    n = dict(m)  # shallow copy

    aliases = {
        # source
        "src": "source",
        "file": "source",
        "filename": "source",
        "filepath": "source",
        "source_filename": "source",
        # title
        "name": "title",
        "doc_title": "title",
        # year / page
        "pub_year": "year",
        "year_": "year",
        "pages": "page",
        "pg": "page",
        "page_no": "page",
        # chunk id
        "chunk": "chunk_id",
        "chunkId": "chunk_id",
        "id": "chunk_id",
    }
    for k, v in list(n.items()):
        if k in aliases and aliases[k] not in n:
            n[aliases[k]] = v

    # 页码清洗
    try:
        if isinstance(n.get("page"), str):
            import re
            m0 = re.search(r"\d+", n["page"])
            if m0:
                n["page"] = int(m0.group(0))
    except Exception:
        pass

    # 保证这些键存在
    for k in ("source", "title", "year", "page", "chunk_id"):
        n.setdefault(k, None)

    # 保留常见科研字段
    for k in ("doi", "journal_name", "issue", "volume", "authors", "section_title"):
        if k not in n and k in m:
            n[k] = m[k]

    return n

# ---------------------------
# 指南向量检索
# ---------------------------
def search_guidelines(query: str, k: int = 5) -> List[Dict[str, Any]]:
    col = get_chroma_collection()
    try:
        res = col.query(
            query_texts=[query],
            n_results=max(1, k),
            include=["documents", "metadatas"],
        )
    except Exception as e:
        raise RuntimeError(
            f"❌ 向量检索异常：{e}\n"
            f"→ 高概率为嵌入维度/模型不一致，请确保 EMBEDDING_MODEL 与 ingest 一致（当前：{EMBED_MODEL}）。"
        )

    ids = (res or {}).get("ids") or [[]]
    docs = (res or {}).get("documents") or [[]]
    metas = (res or {}).get("metadatas") or [[]]

    hits: List[Dict[str, Any]] = []
    n = len(ids[0]) if ids else 0
    for i in range(n):
        txt = docs[0][i] if i < len(docs[0]) else ""
        meta_raw = metas[0][i] if i < len(metas[0]) else {}
        title = str(meta_raw.get("title") or "")
        # 关键词简单打分：标题*2 + 正文*1
        kw = ["老年","糖尿病","冠心病","β受体阻滞剂","目标","首选","禁忌","监测","ACEI","ARB","钙拮抗剂","利尿剂"]
        s = 0
        tl = title.lower(); xl = txt.lower()
        for k in kw:
            k2 = k.lower()
            if k2 in tl: s += 2
            if k2 in xl: s += 1
        hits.append({
            "id": ids[0][i] if i < len(ids[0]) else None,
            "doc": txt,
            "content": txt,
            "meta": _normalize_meta(meta_raw),
            "_kwscore": s,  # 仅用于本地重排
        })
    # 用关键词分数做一次稳定重排：先按 _kwscore，再按原顺序
    hits = sorted(enumerate(hits), key=lambda p: (p[1]["_kwscore"], -p[0]), reverse=True)
    hits = [{k:v for k,v in h.items() if k != "_kwscore"} for _, h in hits]
    return hits

# ---------------------------
# 药品库 SQLite FTS 检索（可选）
# ---------------------------
def _has_drug_db(path: str) -> bool:
    return os.path.exists(path) and os.path.isfile(path)

def search_drug_fts(query: str, k: int = 5) -> List[Dict[str, Any]]:
    if not _has_drug_db(DRUG_DB_PATH):
        return []
    try:
        conn = sqlite3.connect(DRUG_DB_PATH)
        conn.row_factory = sqlite3.Row
    except Exception as e:
        print(f"⚠️ 无法打开药品数据库：{e}")
        return []

    sql = """
    SELECT
        name,
        COALESCE(indication,'') AS indication,
        COALESCE(contraindication,'') AS contraindication,
        COALESCE(interaction,'') AS interaction,
        COALESCE(pregnancy,'') AS pregnancy,
        COALESCE(source,'') AS source,
        bm25(drugs_fts) AS rank
    FROM drugs_fts
    WHERE drugs_fts MATCH ?
    ORDER BY rank LIMIT ?;
    """
    q = query.replace(" ", " OR ")
    try:
        cur = conn.execute(sql, (q, int(max(1, k))))
        rows = cur.fetchall()
    except Exception as e:
        print(f"⚠️ 药品FTS检索失败：{e}")
        rows = []
    finally:
        try:
            conn.close()
        except Exception:
            pass

    hits: List[Dict[str, Any]] = []
    for r in rows:
        hits.append({
            "name": r["name"],
            "rank": r["rank"],
            "indication": r["indication"],
            "contraindication": r["contraindication"],
            "interaction": r["interaction"],
            "pregnancy": r["pregnancy"],
            "source": r["source"],
        })
    return hits

# ---------------------------
# RRF 融合
# ---------------------------
def _rrf_merge(
    vec_hits: List[Dict[str, Any]],
    drug_hits: List[Dict[str, Any]],
    topn: int = 10,
    k_rrf: float = 60.0,
) -> List[Dict[str, Any]]:
    scores: Dict[str, float] = {}
    payload: Dict[str, Dict[str, Any]] = {}

    def add_score(key: str, add: float, obj: Dict[str, Any]):
        scores[key] = scores.get(key, 0.0) + add
        if key not in payload:
            payload[key] = obj

    for i, h in enumerate(vec_hits):
        key = h.get("id") or f"vec_{i}"
        add_score(key, 1.0 / (k_rrf + i + 1), {"type": "guideline", **h})

    for j, h in enumerate(drug_hits):
        key = f"drug_{h.get('name','?')}_{j}"
        add_score(key, 1.0 / (k_rrf + j + 1), {"type": "drug", **h})

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:topn]
    return [payload[k] for k, _ in ranked]

# ---------------------------
# 混合检索入口
# ---------------------------
def hybrid_search(
    query: str,
    method: str = "rrf",
    topn: int = 10,
    k_guideline: Optional[int] = None,
    k_drug: Optional[int] = None,
) -> Dict[str, Any]:
    k_guideline = k_guideline or max(3, min(20, topn))
    k_drug = k_drug or max(3, min(20, topn))

    out: Dict[str, Any] = {"query": query}
    vec_hits: List[Dict[str, Any]] = []
    drug_hits: List[Dict[str, Any]] = []

    if method in ("vec", "rrf"):
        try:
            vec_hits = search_guidelines(query, k=k_guideline)
        except Exception as e:
            print(str(e))
            vec_hits = []
    if method in ("sql", "rrf"):
        drug_hits = search_drug_fts(query, k=k_drug)

    out["guideline_hits"] = vec_hits
    out["drug_hits"] = drug_hits

    if method == "rrf":
        out["fused"] = _rrf_merge(vec_hits, drug_hits, topn=topn)
    elif method == "vec":
        out["fused"] = vec_hits[:topn]
    elif method == "sql":
        out["fused"] = drug_hits[:topn]
    else:
        out["fused"] = []
    return out

# ---------------------------
# CLI
# ---------------------------
def _preview_doc(text: Optional[str], limit: int = 160) -> str:
    if not text:
        return ""
    return (text[:limit] + ("…" if len(text) > limit else "")).replace("\n", " ")

def main():
    parser = argparse.ArgumentParser(description="CareMind Retriever (Chroma 0.5.x)")
    parser.add_argument("--q", "--query", dest="query", required=True, help="查询语句")
    parser.add_argument("--method", choices=["vec", "sql", "rrf"], default="rrf", help="检索方式")
    parser.add_argument("--topn", type=int, default=10, help="返回条数")
    args = parser.parse_args()

    print(f"Embedding model: {EMBED_MODEL}")
    print(f"Chroma dir:     {PERSIST_DIR} | collection={COLLECTION_NAME}")
    if os.path.exists(DRUG_DB_PATH):
        print(f"SQLite path:    {DRUG_DB_PATH}")
    else:
        print("SQLite path:    (未配置或文件不存在，跳过药品 FTS)")

    if args.method == "vec":
        vec = search_guidelines(args.query, k=args.topn)
        print(f"\n[Guideline(Vector)] Hits={len(vec)}")
        for i, h in enumerate(vec, 1):
            m = h.get("meta") or {}
            print(f"{i:02d}. {_preview_doc(h.get('doc'))}")
            print(f"    来源: {m.get('source')} | 标题: {m.get('title')} | 年份: {m.get('year')}")
    elif args.method == "sql":
        drug = search_drug_fts(args.query, k=args.topn)
        print(f"\n[Drug(FTS)] Hits={len(drug)}")
        for i, h in enumerate(drug, 1):
            print(f"{i:02d}. {h.get('name')}  (rank={h.get('rank'):.4f})")
            print(f"    适应症: {_preview_doc(h.get('indication'))}")
            print(f"    禁忌症: {_preview_doc(h.get('contraindication'))}")
            print(f"    交互:   {_preview_doc(h.get('interaction'))}")
            print(f"    妊娠:   {_preview_doc(h.get('pregnancy'))}")
            print(f"    来源:   {_preview_doc(h.get('source'))}")
    else:
        res = hybrid_search(args.query, method="rrf", topn=args.topn)
        fused = res.get("fused") or []
        print(f"\n[RRF Fused] TopN={len(fused)}")
        for i, h in enumerate(fused, 1):
            if h.get("type") == "guideline":
                m = h.get("meta") or {}
                print(f"{i:02d}. [指南] {_preview_doc(h.get('doc'))}")
                print(f"    来源: {m.get('source')} | 标题: {m.get('title')} | 年份: {m.get('year')}")
            else:
                print(f"{i:02d}. [药品] {h.get('name')} (rank≈{h.get('rank'):.4f})")
                print(f"    适应症: {_preview_doc(h.get('indication'))}")
                print(f"    禁忌症: {_preview_doc(h.get('contraindication'))}")
                print(f"    交互:   {_preview_doc(h.get('interaction'))}")
                print(f"    妊娠:   {_preview_doc(h.get('pregnancy'))}")
                print(f"    来源:   {_preview_doc(h.get('source'))}")

if __name__ == "__main__":
    main()
