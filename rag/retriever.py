# -*- coding: utf-8 -*-
"""
retriever.py — CareMind RAG 检索层（Chroma 0.5.x）

本版在保留你“完整版”检索能力的同时，适配了 Streamlit Cloud：
- 打开集合时不加载模型（不传 embedding_function），避免 Torch/Transformers 在 import/open 阶段触发。
- 向量查询仅在需要时懒加载 ST 模型；若失败或禁用，则自动回退到 lexical contains。
- 保留并完善：SQLite FTS 药品检索 + RRF 融合、元数据归一化、CLI、自诊断报错。

环境变量
--------
CHROMA_PERSIST_DIR   (默认 "./chroma_store")
CHROMA_COLLECTION    (默认 "guideline_chunks_v2")
EMBEDDING_MODEL      (默认 "BAAI/bge-small-zh-v1.5")  # 云端更稳
DISABLE_VECTOR       ("1" 则禁用向量检索，仅走 lexical/SQL)
DRUG_DB_PATH         (默认 "./db/drugs.sqlite")

使用
----
向量优先、失败自动回退：
    from rag import retriever as R
    hits = R.search_guidelines("哮喘 β受体阻滞剂 禁忌", k=8)

混合检索 + 融合：
    res = R.hybrid_search("问题…", method="rrf", topn=10)

命令行：
    python -m rag.retriever --q "哮喘 β受体阻滞剂 禁忌" --method rrf --topn 10
"""

from __future__ import annotations
import os
import sys
import re
import argparse
import importlib
from typing import List, Dict, Any, Optional

# 版本号便于日志排错
VERSION = "2025-09-22-merged"

# 读取环境变量（在 import 时读取一次）
PERSIST_DIR     = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "guideline_chunks_v2")
EMBED_MODEL     = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
DISABLE_VECTOR  = os.getenv("DISABLE_VECTOR", os.getenv("CARE_MIND_DISABLE_VECTOR", "0"))
DRUG_DB_PATH    = os.getenv("DRUG_DB_PATH", "./db/drugs.sqlite")

# 单例：Chroma 客户端与 ST 模型
_CLIENT = None          # type: ignore
_ST_MODEL = None        # type: ignore


# ---------------------------
# Chroma 客户端 / 集合打开（无 embedding_function）
# ---------------------------
def _lazy_chroma_client():
    """延迟创建/缓存 Chroma PersistentClient。"""
    global _CLIENT
    if _CLIENT is None:
        chroma = importlib.import_module("chromadb")
        from chromadb import PersistentClient
        ver = getattr(chroma, "__version__", "unknown")
        print(f"[Retriever] v{VERSION} | Chroma v{ver} | dir={PERSIST_DIR}")
        _CLIENT = PersistentClient(path=PERSIST_DIR)
    return _CLIENT


def list_collections_safe() -> List[Dict[str, Any]]:
    """不访问私有属性，返回 {'name','count'} 列表。"""
    client = _lazy_chroma_client()
    out: List[Dict[str, Any]] = []
    try:
        for c in client.list_collections():
            try:
                col = client.get_collection(name=c.name)  # 不传 embedding_function
                out.append({"name": c.name, "count": col.count()})
            except Exception:
                out.append({"name": c.name, "count": None})
    except Exception as e:
        raise RuntimeError(f"list_collections failed: {e}")
    return out


def get_chroma_collection():
    """
    仅打开已存在集合（不 create），并进行直观校验：
    - 集合必须存在
    - 集合必须非空
    注意：不传 embedding_function，避免打开时加载模型。
    """
    client = _lazy_chroma_client()

    try:
        names = [c.name for c in client.list_collections()]
    except Exception as e:
        raise RuntimeError(f"❌ list_collections 失败：{e}")

    if COLLECTION_NAME not in names:
        raise RuntimeError(f"❌ 找不到集合《{COLLECTION_NAME}》。可用集合：{names}")

    try:
        col = client.get_collection(name=COLLECTION_NAME)  # 不传 embedding_function
    except Exception as e:
        raise RuntimeError(f"❌ 打开集合失败：{e}")

    try:
        n = col.count()
    except Exception as e:
        raise RuntimeError(f"❌ 集合计数失败：{e}")

    if n == 0:
        raise RuntimeError(f"❌ 集合《{COLLECTION_NAME}》为空。请检查 ingest 是否写入成功。")

    print(f"[Retriever] 打开集合《{COLLECTION_NAME}》 | docs={n}")
    return col


# ---------------------------
# 向量编码（仅在需要时加载 ST；失败返回 None）
# ---------------------------
def _maybe_load_st_model():
    """按需加载 SentenceTransformer；若禁用或失败则返回 None。"""
    global _ST_MODEL

    if str(DISABLE_VECTOR) == "1":
        return None

    if _ST_MODEL is not None:
        return _ST_MODEL

    try:
        from sentence_transformers import SentenceTransformer
        _ST_MODEL = SentenceTransformer(EMBED_MODEL, device="cpu")
        try:
            import torch  # noqa: F401
            print(f"[Retriever] ST 模型已加载：{EMBED_MODEL}")
        except Exception:
            # 即使 torch 不可用，加载失败也会在 encode 时被捕获
            pass
        return _ST_MODEL
    except Exception as e:
        print(f"[Retriever] ⚠️ 加载 ST 模型失败（{EMBED_MODEL}）：{e}")
        _ST_MODEL = None
        return None


def _encode_query(text: str):
    """将查询文本编码为 embedding（list），失败时返回 None。"""
    model = _maybe_load_st_model()
    if model is None:
        return None
    try:
        emb = model.encode([text], normalize_embeddings=True)
        return emb.tolist()  # chroma 需要 python list
    except Exception as e:
        print(f"[Retriever] ⚠️ encode 失败，将回退：{e}")
        return None


# ---------------------------
# 元数据归一化（保留你原来的完善处理）
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

    try:
        if isinstance(n.get("page"), str):
            m0 = re.search(r"\d+", n["page"])
            if m0:
                n["page"] = int(m0.group(0))
    except Exception:
        pass

    for k in ("source", "title", "year", "page", "chunk_id"):
        n.setdefault(k, None)

    for k in ("doi", "journal_name", "issue", "volume", "authors", "section_title"):
        if k not in n and k in m:
            n[k] = m[k]

    return n


# ---------------------------
# 纯向量检索（有则用；失败自动回退）
# ---------------------------
def _vector_search(col, query: str, k: int) -> List[Dict[str, Any]]:
    qemb = _encode_query(query)
    if not qemb:
        return []
    try:
        res = col.query(
            query_embeddings=qemb,
            n_results=max(k, 10),
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        print(f"[Retriever] 向量检索失败：{e}")
        return []

    docs = (res.get("documents") or [[]])[0] or []
    metas = (res.get("metadatas") or [[]])[0] or []
    dists = (res.get("distances") or [[]])[0] or []
    hits: List[Dict[str, Any]] = []
    for i, txt in enumerate(docs[:k]):
        md = metas[i] if i < len(metas) else {}
        dist = float(dists[i]) if i < len(dists) and dists[i] is not None else 0.0
        hits.append({
            "id": (res.get("ids") or [[]])[0][i] if (res.get("ids") or [[]])[0:] else None,
            "doc": txt,
            "content": txt,
            "meta": _normalize_meta(md),
            "score": 1.0 - dist,
        })
    return hits


# ---------------------------
# 词法回退（where_document $contains）
# ---------------------------
def _lex_contains(col, query: str, k: int) -> List[Dict[str, Any]]:
    syns = ["β受体阻滞剂", "β阻滞剂", "β-blocker", "哮喘", "支气管哮喘"]
    terms = [t for t in syns if t in query] or [query]
    for kw in terms:
        try:
            res = col.query(
                where_document={"$contains": kw},
                n_results=max(k, 10),
                include=["documents", "metadatas"],
            )
        except Exception as e:
            print(f"[Retriever] where_document 失败 '{kw}': {e}")
            continue

        docs = (res.get("documents") or [[]])[0] or []
        metas = (res.get("metadatas") or [[]])[0] or []
        if not docs:
            continue

        hits: List[Dict[str, Any]] = []
        for i, txt in enumerate(docs[:k]):
            meta = metas[i] if i < len(metas) else {}
            hits.append({
                "id": None,
                "doc": txt,
                "content": txt,
                "meta": _normalize_meta(meta),
                "score": 0.0,
            })
        return hits
    return []


# ---------------------------
# 对外：指南检索（向量→词法）
# ---------------------------
def search_guidelines(query: str, k: int = 5) -> List[Dict[str, Any]]:
    col = get_chroma_collection()

    # 1) 向量路径（若可用）
    vec_hits = _vector_search(col, query, k)
    if vec_hits:
        # 你原来有一个简单的关键词重排；保留其思想（可选）
        kw = ["老年","糖尿病","冠心病","β受体阻滞剂","目标","首选","禁忌","监测","ACEI","ARB","钙拮抗剂","利尿剂"]
        def kwscore(h):
            t = (h.get("meta", {}).get("title") or "")
            x = (h.get("content") or "")
            s = 0
            tl = t.lower(); xl = x.lower()
            for kword in kw:
                k2 = kword.lower()
                if k2 in tl: s += 2
                if k2 in xl: s += 1
            return (s, h.get("score", 0))
        vec_hits = sorted(vec_hits, key=kwscore, reverse=True)
        return vec_hits[:k]

    # 2) 词法回退
    return _lex_contains(col, query, k)


# ---------------------------
# 药品库 SQLite FTS 检索（可选）
# ---------------------------
def _has_drug_db(path: str) -> bool:
    return os.path.isfile(path)

def search_drug_fts(query: str, k: int = 5) -> List[Dict[str, Any]]:
    if not _has_drug_db(DRUG_DB_PATH):
        return []
    # sqlite3 由 app.py 中的 shim 保障；此处按需导入
    try:
        import sqlite3  # type: ignore
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
# RRF 融合（向量/词法 vs 药品FTS）
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
    parser = argparse.ArgumentParser(description="CareMind Retriever (Chroma 0.5.x, vector→lexical fallback, FTS+RRF)")
    parser.add_argument("--q", "--query", dest="query", required=True, help="查询语句")
    parser.add_argument("--method", choices=["vec", "sql", "rrf"], default="rrf", help="检索方式")
    parser.add_argument("--topn", type=int, default=10, help="返回条数")
    args = parser.parse_args()

    print(f"Retriever VERSION:  {VERSION}")
    print(f"Embedding model:    {EMBED_MODEL}")
    print(f"Chroma dir:         {PERSIST_DIR} | collection={COLLECTION_NAME}")
    print(f"DISABLE_VECTOR:     {DISABLE_VECTOR}")
    if os.path.exists(DRUG_DB_PATH):
        print(f"SQLite path:        {DRUG_DB_PATH}")
    else:
        print("SQLite path:        (未配置或文件不存在，跳过药品 FTS)")

    if args.method == "vec":
        vec = search_guidelines(args.query, k=args.topn)
        print(f"\n[Guideline(Vector/lex-fallback)] Hits={len(vec)}")
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
