# -*- coding: utf-8 -*-
"""
CareMind · 临床决策支持（MVP）
长版 app.py（已合并补丁 · 2025-09-22）

本文件的目标：
1) 维持你原有的 UI 区块布局（问题输入 / 建议 / 证据 / 药品结构化 / 运行日志）
2) 与新版 rag/retriever.py 的 API 对齐（0.5.x 兼容）：
   - list_collections_safe → list_collections()
   - 旧 peek（访问 max_seq_id） → peek_collection(n)
   - where_document 直搜 → 使用 retriever.keyword_get()（内部用 collection.get(..., where_document=...)）
3) 让“导出”按钮更紧凑 + 防空内容导出
4) 提供中英双语切换（简易），便于现场演示

如需对接你更早的长版 UI，只要将核心调用替换为本文对应函数即可。
"""

from __future__ import annotations

import os
import io
import re
import json
import time
import textwrap
from datetime import datetime
from typing import List, Dict, Any, Optional

import streamlit as st

# ---------------------------------------------------------------------
# 环境兜底（云端常见问题）：若安装了 pysqlite3-binary，则别名为 sqlite3
# （有些托管环境缺系统级 sqlite3，改用 manylinux 轮子）
# ---------------------------------------------------------------------
try:
    import sqlite3  # noqa: F401
except Exception:
    try:
        import pysqlite3 as sqlite3  # type: ignore
        import sys as _sys
        _sys.modules["sqlite3"] = sqlite3
    except Exception as e:
        # 不在此处硬失败；retriever 的健康检查会给出更明确的错误
        st.warning(f"⚠️ SQLite 初始化警告：{e}")

# ---------------------------------------------------------------------
# 项目内模块
# ---------------------------------------------------------------------
from rag import retriever as R  # 新版 retriever：含 list_collections/peek_collection/keyword_get/print_health/retrieve 等

# ---------------------------------------------------------------------
# 页面设置与全局样式
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="CareMind · 临床决策支持（MVP）",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 轻量 CSS：压缩按钮与卡片留白；避免导出按钮过于占位
st.markdown(
    """
    <style>
      .small-btn button {padding: 0.35rem 0.6rem; font-size: 0.85rem;}
      .tight-block {margin-top: 0.25rem; margin-bottom: 0.25rem;}
      .code-wrap pre {white-space: pre-wrap;}
      .muted {opacity: 0.7;}
      .mono {font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;}
      .warn-box {background:#fff7ed;border:1px solid #fed7aa;padding:0.75rem;border-radius:0.5rem;}
      .ok-box {background:#f0fdf4;border:1px solid #bbf7d0;padding:0.75rem;border-radius:0.5rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# 简单的中英双语开关（不改核心逻辑，只改标签）
# ---------------------------------------------------------------------
LANG = st.sidebar.selectbox("界面语言 / Language", ["中文", "English"], index=0)

def T(cn: str, en: str) -> str:
    return cn if LANG == "中文" else en

# ---------------------------------------------------------------------
# 环境变量读入（用于展示 + 便于在云端核对）
# ---------------------------------------------------------------------
DEMO_MODE = os.getenv("CAREMIND_DEMO", "0")
CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")
CHROMA_COL = os.getenv("CHROMA_COLLECTION", "guideline_chunks")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
DRUG_DB = os.getenv("DRUG_DB_PATH", "./db/drugs.sqlite")
RETRIEVER_VERSION = os.getenv("RETRIEVER_VERSION", "2025-09-22-merged")

# ---------------------------------------------------------------------
# 页面顶部：标题与说明
# ---------------------------------------------------------------------
st.title("CareMind · 临床决策支持（MVP）")

st.caption(
    T(
        "⚠️ 本工具仅供临床决策参考，不替代医师诊断与处方。",
        "⚠️ This tool is for clinical decision support only; it does not replace professional diagnosis or prescription."
    )
)

# ---------------------------------------------------------------------
# 输入区：临床问题 + 可选药品名 + 检索参数
# ---------------------------------------------------------------------
with st.container():
    col_q1, col_q2 = st.columns([3, 1.2], gap="large")
    with col_q1:
        q = st.text_area(
            T("输入临床问题", "Enter a clinical question"),
            height=96,
            placeholder=T("例如：合并支气管哮喘的高血压患者是否可用β受体阻滞剂？", "e.g., Can beta-blockers be used in hypertensive patients with bronchial asthma?"),
        )
    with col_q2:
        drug = st.text_input(T("（可选）指定药品名", "(Optional) Specify drug name"), value="")
        topn = st.slider(T("返回条数", "Top-N results"), min_value=4, max_value=20, value=8, step=1)
        method = st.selectbox(
            T("重排序/融合", "Rerank/Fusion"),
            ["rrf", "none", "ce"],
            index=0,
            help=T("rrf=稳健融合；ce=交叉编码器；none=不重排", "rrf=robust fusion; ce=cross-encoder; none=no rerank"),
        )

    btn_run = st.button(T("🧭 检索并生成建议", "🧭 Retrieve & Draft Suggestion"), type="primary")

# ---------------------------------------------------------------------
# 主动作：检索 + 组织草案
# ---------------------------------------------------------------------
suggest_md: str = ""           # 建议（Markdown）
evidence_items: List[Dict[str, Any]] = []
drug_structured: List[Dict[str, Any]] = []
elapsed_sec = 0.0
now_ts = datetime.now().strftime("%Y-%m-%d %H:%M")

def _mk_md_header(text: str) -> str:
    return f"**{text}**\n"

def _format_hit_md(hit: Dict[str, Any], idx: int) -> str:
    """将单条命中渲染为 MD（标题 + 片段 + 来源）"""
    m = hit.get("meta") or {}
    title = m.get("title") or T("未命名", "Untitled")
    src = m.get("source") or "-"
    yr = m.get("year") or "-"
    body = (hit.get("doc") or "").strip()
    body = re.sub(r"\s+", " ", body)
    body = body[:600] + ("…" if len(body) > 600 else "")
    return f"{idx}. **{title}**（{yr}，{src}）\n\n> {body}\n"

def _draft_suggestion_cn(question: str, hits: List[Dict[str, Any]], drug_name: Optional[str]) -> str:
    """中文草案（可按需替换为你的结构化提示输出）"""
    bullets = []
    if drug_name:
        bullets.append(f"- 优先围绕 **{drug_name}** 的适应证/禁忌证/相互作用进行核对。")
    bullets.append("- 优先考虑权威指南与共识推荐；结合患者伴随疾病进行个体化权衡。")
    bullets.append("- 注意监测用药后可能的不良反应与疗效指标，必要时及时复评。")
    hint = "\n".join(bullets)

    if "哮喘" in question and ("β" in question or "beta" in question.lower()):
        # 针对示例问题给出保守医学常识占位（非处方建议）
        domain_hint = (
            "对于合并支气管哮喘的高血压患者，通常**避免使用非选择性 β 受体阻滞剂**；"
            "若有强适应证（如缺血性心脏病等），可**慎用 β1 选择性**药物并从低剂量开始，密切监测气道反应。"
            "降压方案可优先考虑 ACEI/ARB 或长效二氢吡啶类钙拮抗剂。"
        )
    else:
        domain_hint = "请结合患者人群特征、并发症与药物相互作用做个体化优化。"

    md = []
    md.append(_mk_md_header("临床建议（草案）"))
    md.append(f"问题: {question.strip() or '（未输入）'}")
    md.append("")
    md.append(domain_hint)
    md.append("")
    md.append(hint)
    md.append("")
    md.append("合规提示：本工具仅供临床决策参考，不代替医生诊断与处方。")
    return "\n".join(md)

def _draft_suggestion_en(question: str, hits: List[Dict[str, Any]], drug_name: Optional[str]) -> str:
    bullets = []
    if drug_name:
        bullets.append(f"- Prioritize verifying **{drug_name}** indications/contraindications/interactions.")
    bullets.append("- Prefer recommendations from authoritative guidelines and consensus; individualize to comorbidities.")
    bullets.append("- Monitor efficacy and adverse events, and re-assess when needed.")
    hint = "\n".join(bullets)

    if "asthma" in question.lower() and ("β" in question or "beta" in question.lower()):
        domain_hint = (
            "In hypertensive patients with bronchial asthma, generally **avoid non-selective β-blockers**; "
            "if a strong indication exists (e.g., ischemic heart disease), consider **β1-selective** agents with caution and close airway monitoring. "
            "ACEI/ARB or long-acting dihydropyridine CCBs are reasonable options for hypertension."
        )
    else:
        domain_hint = "Individualize therapy by phenotype, comorbidities, and drug–drug interactions."

    md = []
    md.append(_mk_md_header("Clinical Suggestion (Draft)"))
    md.append(f"Question: {question.strip() or '(empty)'}")
    md.append("")
    md.append(domain_hint)
    md.append("")
    md.append(hint)
    md.append("")
    md.append("Compliance: This tool assists clinical decision-making and does not replace medical judgment.")
    return "\n".join(md)

def _make_suggestion_md(question: str, hits: List[Dict[str, Any]], drug_name: Optional[str]) -> str:
    return _draft_suggestion_cn(question, hits, drug_name) if LANG == "中文" else _draft_suggestion_en(question, hits, drug_name)

def _mk_filename(prefix: str, ext: str = ".md") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = re.sub(r"[^\w\-]+", "_", prefix.strip())[:32] or "CareMind"
    return f"{safe}_{ts}{ext}"

if btn_run:
    t0 = time.time()
    # 1) 文本检索（统一入口：内部做向量检索 + 关键词兜底；可选 ce 重排）
    try:
        res = R.retrieve(
            q or "",
            R.RetrieveOptions(topn=int(topn), method=method)
        )
        evidence_items = res.get("hits") or []
        elapsed_sec = float(res.get("t_sec") or 0.0)
    except Exception as e:
        st.error(T(f"检索失败：{e}", f"Retrieval failed: {e}"))
        evidence_items = []
        elapsed_sec = time.time() - t0

    # 2) 药品结构化查询（可选）
    if drug.strip():
        try:
            drug_structured = R.lookup_drug(drug.strip(), limit=5) or []
        except Exception as e:
            st.warning(T(f"药品结构化查询失败：{e}", f"Drug lookup failed: {e}"))

    # 3) 草案生成（简单模板；你也可以换成 Structured Prompt）
    suggest_md = _make_suggestion_md(q, evidence_items, drug.strip() or None)

# ---------------------------------------------------------------------
# 展示区：建议 / 证据片段 / 药品结构化 / 导出按钮
# ---------------------------------------------------------------------
col_a, col_b = st.columns([2.2, 1], gap="large")

with col_a:
    st.subheader(T("🧭 建议", "🧭 Suggestion"))
    if suggest_md:
        st.markdown(suggest_md)
        st.caption(T(f"⏱️ 用时：{elapsed_sec:.2f}s", f"⏱️ Time: {elapsed_sec:.2f}s"))
    else:
        st.info(T("尚未生成建议。", "No suggestion yet."))

    st.markdown("---")
    st.subheader(T("📚 证据片段", "📚 Evidence Snippets"))
    if evidence_items:
        ev_md_parts = []
        for i, h in enumerate(evidence_items, 1):
            block = _format_hit_md(h, i)
            st.markdown(block)
            ev_md_parts.append(block)
    else:
        st.warning(T("暂无证据片段（可能因向量库不命中或关键词过泛）。", "No evidence snippets yet (vector store miss or query too broad)."))

    # 导出按钮（更紧凑 + 判空）
    st.markdown("")
    with st.container():
        c1, c2 = st.columns([0.24, 0.24])
        with c1:
            with st.container():
                st.markdown('<div class="small-btn">', unsafe_allow_html=True)
                if st.button(T("⬇️ 导出建议(MD)", "⬇️ Export Suggestion (MD)")):
                    if suggest_md.strip():
                        fn = _mk_filename("CareMind_Suggestion")
                        st.download_button(
                            label=T("下载", "Download"),
                            data=suggest_md,
                            file_name=fn,
                            mime="text/markdown",
                            key="dl_suggest",
                        )
                    else:
                        st.warning(T("无可导出的建议内容。", "Nothing to export."))
                st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            with st.container():
                st.markdown('<div class="small-btn">', unsafe_allow_html=True)
                if st.button(T("⬇️ 导出证据(MD)", "⬇️ Export Evidence (MD)")):
                    if evidence_items:
                        fn = _mk_filename("CareMind_Evidence")
                        md = []
                        md.append(_mk_md_header(T("证据片段", "Evidence")))
                        for i, h in enumerate(evidence_items, 1):
                            md.append(_format_hit_md(h, i))
                        data = "\n".join(md)
                        st.download_button(
                            label=T("下载", "Download"),
                            data=data,
                            file_name=fn,
                            mime="text/markdown",
                            key="dl_evidence",
                        )
                    else:
                        st.warning(T("当前无可导出的证据。", "No evidence to export."))
                st.markdown("</div>", unsafe_allow_html=True)

with col_b:
    st.subheader(T("💊 药品结构化", "💊 Drug (Structured)"))
    if drug.strip():
        if drug_structured:
            for row in drug_structured:
                with st.container():
                    st.markdown("**" + (row.get("name") or "-") + "**")
                    st.caption((row.get("aliases") or "")[:200])
                    st.markdown(T("**适应症**", "**Indication**") + f": {row.get('indication') or '-'}")
                    st.markdown(T("**禁忌**", "**Contraindication**") + f": {row.get('contraindication') or '-'}")
                    st.markdown(T("**相互作用**", "**Interactions**") + f": {row.get('interactions') or '-'}")
                    st.markdown(T("**妊娠分级**", "**Pregnancy**") + f": {row.get('pregnancy') or '-'}")
                    st.markdown(T("**来源**", "**Source**") + f": {row.get('source') or '-'}")
                    st.markdown("---")
        else:
            st.info(T("未在本地药品库中找到匹配项。", "No match found in the local drug DB."))
    else:
        st.caption(T("输入药品名以查看结构化信息。", "Enter a drug name to view structured info."))

# ---------------------------------------------------------------------
# 运行日志 / 环境诊断（**关键修复点：使用 retriever 的新 API**）
# ---------------------------------------------------------------------
st.markdown("---")
st.subheader(T("🪵 运行日志 / 环境诊断", "🪵 Runtime Log / Health Check"))

try:
    info = R.print_health()  # ✅ 新版封装（内部已用 list_collections()/peek_collection()）
    # 顶部摘要
    st.markdown(
        f"""
        <div class="ok-box mono">
          Retriever version: {RETRIEVER_VERSION}<br/>
          Chroma 目录存在： {info['chroma'].get('dir')} → {str(info['chroma'].get('dir_exists'))}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 环境
    with st.expander(T("环境变量", "Environment"), expanded=False):
        st.code(json.dumps(info.get("env", {}), ensure_ascii=False, indent=2), language="json")

    # Chroma 集合列表
    with st.expander(T("Chroma 集合", "Chroma Collections"), expanded=False):
        st.code(json.dumps(info.get("chroma", {}).get("collections", []), ensure_ascii=False, indent=2), language="json")

    # 快速抽样（peek） —— **不再访问 max_seq_id**
    with st.expander(T("Chroma 快速抽样（peek）", "Chroma Peek"), expanded=False):
        st.code(json.dumps(info.get("chroma", {}).get("peek", []), ensure_ascii=False, indent=2), language="json")

    # SQLite
    st.caption(T(
        f"SQLite 文件存在： {info.get('sqlite', {}).get('path')} → {str(info.get('sqlite', {}).get('exists'))}",
        f"SQLite exists: {info.get('sqlite', {}).get('path')} → {str(info.get('sqlite', {}).get('exists'))}",
    ))
    if info.get("sqlite", {}).get("tables"):
        st.code(json.dumps(info.get("sqlite", {}).get("tables"), ensure_ascii=False, indent=2), language="json")

    # 关键词直搜（**修复点**：使用 retriever.keyword_get，而非 query(where_document=...)）
    with st.expander(T("where_document 直搜（不走向量）", "where_document direct search (non-vector)"), expanded=False):
        kw = st.text_input(T("关键词（示例：哮喘）", "Keyword (e.g., asthma)"), value="哮喘", key="kw_demo")
        if st.button(T("执行直搜", "Run keyword search"), key="btn_kw_demo"):
            hits = R.keyword_get(kw, limit=5)  # ✅ 正确用法
            st.code(json.dumps(hits, ensure_ascii=False, indent=2), language="json")

except Exception as e:
    st.markdown(
        f'<div class="warn-box mono">Chroma 访问错误：{e}</div>',
        unsafe_allow_html=True
    )

# ---------------------------------------------------------------------
# 页脚与合规声明
# ---------------------------------------------------------------------
st.markdown("---")
st.caption(
    T(
        "⚠️ 本工具用于教学与演示，不直接用于临床决策；请以最新指南与专科医师判断为准。",
        "⚠️ For education and demonstration only; consult up-to-date guidelines and specialist judgment."
    )
)
