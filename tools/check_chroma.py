#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_chroma.py
用途：
1) 打印 Chroma 版本、目录、集合名；
2) 列出所有集合与文档数；
3) 针对指定集合做一次试探查询，确认是否能命中；
4) 自动给出“最可能原因与修复建议”。

运行：
  export CHROMA_PERSIST_DIR=./chroma_store
  export CHROMA_COLLECTION=guideline_chunks_1024_v2
  export EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5
  python tools/check_chroma.py --q "老年 高血压 糖尿病 目标"
"""
import os, sys, argparse, importlib, time

def alias_sqlite_if_needed():
    try:
        import sqlite3  # noqa
    except Exception:
        try:
            import pysqlite3 as sqlite3  # noqa
            sys.modules["sqlite3"] = sqlite3
            print("ℹ️  sqlite3 → pysqlite3 别名启用")
        except Exception as e:
            print(f"⚠️ 无法提供 sqlite3: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", default="高血压 指南 目标", help="试探查询")
    args = parser.parse_args()

    alias_sqlite_if_needed()

    persist = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")
    coll    = os.getenv("CHROMA_COLLECTION", "guideline_chunks")
    model   = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")

    print("──────── Health Check ────────")
    print(f"Chroma dir:     {persist}")
    print(f"Collection:     {coll}")
    print(f"Embedding mdl:  {model}")

    try:
        chroma = importlib.import_module("chromadb")
        print(f"Chroma version: {getattr(chroma, '__version__', 'unknown')}")
        from chromadb import PersistentClient
    except Exception as e:
        print(f"❌ 无法导入 chromadb：{e}")
        sys.exit(1)

    t0 = time.time()
    try:
        client = PersistentClient(path=persist)
    except Exception as e:
        print(f"❌ 无法打开持久化目录：{e}")
        print("→ 检查 CHROMA_PERSIST_DIR 路径是否存在 / 权限是否正确")
        sys.exit(1)

    # 列集合
    try:
        colls = client.list_collections()
        if not colls:
            print("❌ 该目录下没有任何集合。")
            print("→ 可能原因：目录不对；未把本地 chroma_store 推到此环境；或曾用不同路径/分支。")
            sys.exit(2)
        print("✔️ 发现集合：")
        for c in colls:
            try:
                n = c.count()
            except Exception:
                n = "?"
            mark = "  "
            if c.name == coll:
                mark = "👉"
            print(f"{mark} {c.name} | docs={n}")
    except Exception as e:
        print(f"❌ list_collections 失败：{e}")
        sys.exit(3)

    # 打开目标集合
    try:
        col = client.get_collection(name=coll)
    except Exception as e:
        print(f"❌ 找不到集合 {coll}：{e}")
        print("→ 请把 CHROMA_COLLECTION 设为上方列出的实际集合名之一。")
        sys.exit(4)

    try:
        n = col.count()
        print(f"集合《{coll}》文档数：{n}")
        if n == 0:
            print("❌ 集合为空。→ 不是查询问题，而是索引未写入/被清空")
            print("修复：确认 ingest 是否指向同一目录；必要时将本地 chroma_store 整个目录同步过来。")
            sys.exit(5)
    except Exception as e:
        print(f"⚠️ 无法统计文档数：{e}")

    # 试探查询
    try:
        res = col.query(query_texts=[args.q], n_results=5)
        ids = (res or {}).get("ids") or [[]]
        docs = (res or {}).get("documents") or [[]]
        topn = len(ids[0]) if ids else 0
        print(f"试探查询：{args.q} → 命中条数 {topn}")
        if topn == 0:
            print("❌ 未命中。常见原因：")
            print("1) 嵌入模型与 ingest 不一致（维度不匹配或相似度弱）；")
            print("2) 文本规范化差异（你问的关键词未出现于索引片段，或切分太碎）；")
            print("3) 版本迁移导致 store 不兼容（请统一 chromadb 版本并复用同一 store）；")
            print("修复手段：改回 ingest 时的 EMBEDDING_MODEL；尝试更宽松的查询词；必要时重建索引。")
        else:
            print("✔️ 能命中，Top1 预览：")
            print((docs[0][0] or "")[:160].replace("\n", " "))
    except Exception as e:
        print(f"❌ 查询异常：{e}")
        print("→ 高概率是版本或嵌入维度不一致。确保两端 chromadb 与 EMBEDDING_MODEL 完全一致。")
        sys.exit(6)

    print(f"耗时：{time.time() - t0:.2f}s")
    print("──────── Done ────────")

if __name__ == "__main__":
    main()
