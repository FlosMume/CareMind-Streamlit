# -*- coding: utf-8 -*-
"""
CareMind · MVP CDSS (Streamlit, bilingual zh/en)
------------------------------------------------
特性 / Features
- 双语 UI（中文 / English）
- 通过 rag.pipeline.answer 提供建议文本（反射式调用，兼容是否含 lang 参数）
- 证据片段/药品结构化/运行日志 Tab
- ✅ 诊断面板：展示有效配置（Secrets 优先）、chroma_store 是否存在、
  Chroma 集合与条目数（调用 retriever.list_collections_safe / primary_collection_count，
  避免额外创建第二个 Chroma 客户端）
- ✅ 本会话历史记录与“一键复用”
- 保留 Python/sqlite/Torch/Chroma 版本信息显示（不构造额外 Chroma 客户端）
"""

from __future__ import annotations
# --- SQLite bootstrap for Chroma on Streamlit Cloud ---
try:
    import pysqlite3  # must be installed via requirements.txt
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass
# ------------------------------------------------------
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import re
import sys
import json
import time
import inspect
from typing import Any, Dict, List, Optional
import importlib

import streamlit as st
# 先导入 retriever：其内部会根据需要安装 SQLite shim（老版本 SQLite 的云端有用）
from rag import retriever as R  # noqa: F401   # 用于 shim 与缓存化的 Chroma 客户端
import rag.pipeline as cm_pipeline          # 用模块导入，避免热重载下的符号遮蔽
import sqlite3
from dotenv import load_dotenv
load_dotenv()


# =============================================================================
# 0) 侧边栏：环境诊断（版本/路径/集合计数）
#    小心：不要在这里直接实例化另一个 chromadb.PersistentClient，
#    统一走 retriever 的单例客户端，以避免“different settings”报错。
# -----------------------------------------------------------------------------
with st.sidebar.expander("🔎 Environment Diagnostics", expanded=False):
    st.write("**Python version**:", sys.version)
    st.write("**sqlite3 module version**:", sqlite3.version)          # Python wrapper version
    st.write("**sqlite3 library version**:", sqlite3.sqlite_version)  # Underlying lib version

    # 版本信息不依赖客户端；纯 import 安全
    try:
        import torch
        st.write("**Torch version**:", torch.__version__)
    except Exception as e:
        st.error(f"Torch not available: {e}")

    try:
        chromadb = importlib.import_module("chromadb")
        st.write("**Chroma version**:", chromadb.__version__)
    except Exception as e:
        st.error(f"Chroma import failed: {e}")

    persist = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")
    coll = os.getenv("CHROMA_COLLECTION", "guideline_chunks_v2")
    st.write("**CHROMA_PERSIST_DIR**:", persist)
    st.write("**CHROMA_COLLECTION (from env)**:", coll)

    # ✅ 使用 retriever 的单例客户端做统计，避免创建第二个客户端
    try:
        count = R.primary_collection_count()
        st.write("**Collection count (active)**:", count)
        # 如需列出集合，用 list_collections_safe（不会抛异常）
        cols = R.list_collections_safe()
        if cols:
            st.write("**Collections (name → count)**:",
                     {c.get("name", ""): c.get("count", -1) for c in cols})
    except Exception as e:
        st.error(f"❌ Failed to query collections via retriever: {e}")


# =============================================================================
# 1) Helper：Secrets 优先的 env 读取 + 友好工具
# -----------------------------------------------------------------------------
def _env(key: str, default: str | None = None) -> str | None:
    """
    Secrets-aware env reader:
    优先 st.secrets[key]，其后 os.environ[key]，最后 default。
    """
    try:
        return os.getenv(key, st.secrets.get(key, default))
    except Exception:
        return os.getenv(key, default)

def link_citations(md: str) -> str:
    """把 "[#3]" / "[3]" 转为 "#hit-3" 锚点链接，便于从建议跳回证据片段。"""
    # Keep the visible text as "[n]" (not just "n") so users can see bracketed citations.
    # Use escaped brackets inside the link text to avoid Markdown parsing quirks.
    return re.sub(r"\[(?:#)?(\d+)\]", r"[\\[\1\\]](#hit-\1)", md or "")

def split_advice_and_evidence_list(md: str) -> tuple[str, str]:
    """Split model output into (advice_md, evidence_list_md).

    The OpenAI prompt templates ask for two sections:
    - "Clinical Recommendation Points" / "临床建议要点"
    - "Evidence List" / "证据清单"
    plus a final compliance line.

    We render the Evidence List only in the Evidence tab (not inside Advice).
    """
    txt = (md or "").strip()
    if not txt:
        return "", ""

    # 1) Prefer splitting on the explicit Evidence List header (English/Chinese).
    m = re.search(
        r"(?mi)^\s*(?:\*\*|__)?\s*(evidence\s+list|证据清单)\s*(?:\*\*|__)?\s*[:：]\s*$",
        txt,
    )

    # 2) If the model forgot the header, fall back to splitting on the first numbered
    #    evidence list item (e.g. "[1] ..."), which is how duplicates typically appear.
    if not m:
        m = re.search(r"(?m)^\s*\[\d+\]\s+", txt)
    if not m:
        return txt, ""

    advice = txt[: m.start()].rstrip()
    evidence_block = txt[m.start():].strip()

    # Remove a trailing single-line compliance statement from the evidence block.
    lines = [ln.rstrip() for ln in evidence_block.splitlines()]
    while lines and not lines[-1].strip():
        lines.pop()
    if lines:
        last = lines[-1].strip()
        if re.match(r"(?i)^this tool is for clinical decision reference only\b", last) or re.match(
            r"(?i)^compliance\s+note:\b", last
        ) or re.match(r"^本工具仅供临床决策参考\b", last) or re.match(r"^合规提示：本工具\b", last):
            lines.pop()
    evidence_list = "\n".join(lines).strip()

    return advice, evidence_list


def strip_redundant_advice_heading(md: str, lang: str) -> str:
    """Remove a leading '建议：'/'Advice:' line when the UI already provides the section header."""
    if not md:
        return md
    heading_only = r"(?mi)^\s*(?:\*\*|__)?\s*(建议|advice)\s*(?:\*\*|__)?\s*[:：]\s*$"
    lines = md.splitlines()
    # Case A: first line is just a heading like "建议：" / "Advice:"
    if lines and re.match(heading_only, lines[0] or ""):
        lines = lines[1:]
        if lines and not lines[0].strip():
            lines = lines[1:]
        return "\n".join(lines).lstrip()

    # Case B: first line starts with a redundant prefix like "建议： 结论…"
    if lines:
        m = re.match(r"(?i)^\s*(建议|advice)\s*[:：]\s*(.+)$", lines[0].strip())
        if m:
            lines[0] = m.group(2).strip()
    return "\n".join(lines).lstrip()


def normalize_evidence_list_md(md: str, lang: str) -> str:
    """Normalize Evidence List: keep a single header and render each [n] entry as one line."""
    txt = (md or "").strip()
    if not txt:
        return txt

    hdr = "证据清单：" if lang == "zh" else "Evidence List:"
    # Ensure header exists
    if re.match(r"(?m)^\s*\[1\]\s+", txt) and not re.search(
        r"(?mi)^\s*(evidence\s+list|证据清单)\s*[:：]", txt
    ):
        txt = hdr + "\n" + txt

    lines = [ln.rstrip() for ln in txt.splitlines()]
    # Remove duplicate headers, keep the first one.
    out_lines: List[str] = []
    header_seen = False
    for ln in lines:
        if re.match(r"(?mi)^\s*(evidence\s+list|证据清单)\s*[:：]\s*$", ln.strip()):
            if header_seen:
                continue
            header_seen = True
            out_lines.append(hdr)
        else:
            out_lines.append(ln)
    lines = out_lines

    # Collapse multi-line items into one line per [n]
    header_line = hdr
    items: List[str] = []
    current: Optional[str] = None
    extras: List[str] = []
    for ln in lines:
        if not ln.strip():
            continue
        if re.match(r"(?mi)^\s*(evidence\s+list|证据清单)\s*[:：]", ln.strip()):
            header_line = hdr
            continue
        if re.match(r"^\s*(?:[-*]\s*)?\[\d+\]\s+", ln):
            if current:
                items.append(current.strip())
            current = re.sub(r"^\s*[-*]\s*", "", ln).strip()
        else:
            if current:
                current = (current + " " + ln.strip()).strip()
            else:
                extras.append(ln.strip())
    if current:
        items.append(current.strip())

    if not items:
        # Preserve non-item informational lines (e.g. "暂无证据片段")
        if extras:
            return (header_line + "\n" + "\n".join(extras)).strip() + "\n"
        return header_line
    # Render as a Markdown bullet list so each entry is one line in Streamlit.
    items = [f"- {it}" for it in items]
    return (header_line + "\n" + "\n".join(items)).strip() + "\n"

def evidence_list_md_from_hits(lang: str, hits: List[Dict[str, Any]]) -> str:
    """Render a compact Evidence List (title/source/year only) from retrieved hits."""
    hdr = "证据清单：" if lang == "zh" else "Evidence List:"
    if not hits:
        return hdr + "\n" + ("（暂无证据片段）" if lang == "zh" else "(No evidence snippets available.)")

    lines = [hdr]
    for i, h in enumerate(hits or [], 1):
        m = h.get("meta") or {}
        title = (m.get("title") or m.get("doc_title") or m.get("section_title") or ("无标题" if lang == "zh" else "Untitled"))
        source = (m.get("source") or m.get("source_filename") or ("未知来源" if lang == "zh" else "Unknown"))
        year = (m.get("year") or "")
        yr = f"{year}".strip()
        tail = (f"（{yr}）" if lang == "zh" else f"({yr})") if yr else ""
        lines.append(f"- [{i}] {title} — {source} {tail}".rstrip())
    return "\n".join(lines).strip() + "\n"

def evidence_md(lang: str, hits: List[Dict[str, Any]]) -> str:
    """将证据片段渲染为 Markdown（用于下载）。"""
    lines = []
    for i, h in enumerate(hits or [], 1):
        m = h.get("meta") or {}
        # title  = str(m.get("title")  or ("无标题" if lang == "zh" else "Untitled"))
        # source = str(m.get("source") or ("未知"   if lang == "zh" else "Unknown"))
        # year   = str(m.get("year")   or "—")
        title  = (m.get("title") or m.get("doc_title") or m.get("section_title") or "Untitled")
        source = (m.get("source") or m.get("source_filename") or "Unknown")
        year   = (m.get("year") or "")
        
        head = (
            f"### {i} {title}\n\n"
            + (f"- 来源：{source} · 年份：{year}\n\n" if lang == "zh"
               else f"- Source: {source} · Year: {year}\n\n")
        )
        lines.append(head + (h.get("content") or "") + "\n")
    return "\n".join(lines)

def friendly_hints(lang: str, exc: Exception) -> List[str]:
    """把常见后端异常翻译成友好的排障提示。"""
    msg = str(exc).lower()
    zh = (lang == "zh")
    tips = []
    if "chromadb" in msg:
        tips.append("· 检查 CHROMA_PERSIST_DIR / CHROMA_COLLECTION" if zh else
                    "· Check CHROMA_PERSIST_DIR / CHROMA_COLLECTION")
    if "sqlite" in msg:
        tips.append("· 检查 SQLite 路径与表结构" if zh else
                    "· Verify SQLite path & schema")
    if "cuda" in msg or "cudnn" in msg:
        tips.append("· 检查 CUDA/cuDNN 或切到 CPU" if zh else
                    "· Check CUDA/cuDNN or switch to CPU")
    if "module" in msg and "not found" in msg:
        tips.append("· 确认 rag/__init__.py 与导入路径" if zh else
                    "· Ensure rag/__init__.py and import path")
    return tips


# =============================================================================
# 2) 极简 i18n（页面文案；pipeline 内部生成的文本已在后端本地化）
# -----------------------------------------------------------------------------
I18N: Dict[str, Dict[str, str]] = {
    "zh": {
        "title": "CareMind · 临床决策支持（MVP）",
        "question_label": "输入临床问题",
        "question_ph": "例如：慢性肾病（CKD）患者使用 ACEI/ARB 时如何监测？多久复查？",
        "drug_label": "（可选）指定药品名（如：阿司匹林）",
        "submit": "生成建议",
        "tab_advice": "🧭 建议",
        "tab_evidence_list": "📑 证据清单",
        "tab_hits_raw": "🎯 命中",
        "tab_hits": "📚 证据片段",
        "tab_drug": "💊 药品结构化",
        "tab_log": "🪵 运行日志",
        "settings": "⚙️ 设置",
        "k_slider": "检索片段数（Top-K）",
        "show_meta": "显示片段元数据",
        "expand_hits": "展开所有片段",
        "filters": "🧩 证据筛选（前端）",
        "filter_src": "按来源包含过滤（可留空）",
        "filter_year": "年份范围",
        "presets": "🧪 问题模板",
        "preset_select": "快速选择",
        "preset_none": "——",
        "preset1": "CKD 合并高血压 ACEI/ARB 监测",
        "preset2": "老年合并 T2DM+CAD：降压目标与方案",
        "preset3": "GDM 胰岛素起始（指征与剂量）",
        "advice_hdr": "建议（含引用与合规声明）",
        "advice_hdr_llm": "建议（含引用与合规声明）",
        "advice_hdr_draft": "临床建议（草案）",
        "advice_hdr_demo": "演示输出",
        "time_used": "⏱️ 用时：{:.2f}s",
        "export_advice": "导出建议（Markdown）",
        "export_evidence": "导出证据（Markdown）",
        "disclaimer": "⚠️ 本工具仅供临床决策参考，不替代医师诊断与处方。",
        "hits_hdr": "检索片段（Top-{k}，过滤后 {n} 条）",
        "no_hits": "未检索到符合筛选条件的片段。",
        "drug_hdr": "药品结构化信息（SQLite）",
        "no_drug": "未提供或未检索到对应药品的结构化信息。",
        "log_export": "导出本会话全部日志（JSON）",
        "history_hdr": "🗂️ 本会话历史（点击复用）",
        "no_history": "暂无历史记录。",
        "reuse": "复用",
        "reused_tip": "已复用：{q}（药品：{drug}，K={k}）。可编辑后再次生成。",
        "page_footer": "© CareMind · MVP CDSS | 本工具仅供临床决策参考，不替代医师诊断与处方。",
        "chips_src": "来源：",
        "chips_year": "年份：",
        "chips_id": "ID：",
        "stats_hits": "片段数：{n} · 总字数：{c}",
        "warn_need_q": "请输入临床问题后再生成建议。",
        "err_backend": "后端错误（详见下方日志/诊断）。",
        "diag_title": "🔎 环境诊断",
        "diag_cfg": "有效配置（优先 Secrets）",
        "diag_chroma": "Chroma 集合：",
        "diag_chroma_err": "Chroma 访问错误：",
        "diag_sqlite": "SQLite 表：",
        "diag_sqlite_err": "SQLite 错误：",
        "draft_reason_missing_key": "ℹ️ 进入草案模式：未检测到 OPENAI_API_KEY（请在 Streamlit Cloud → Manage app → Secrets 中配置）。",
        "draft_reason_openai_error": "ℹ️ 进入草案模式：OpenAI 调用失败（{err}）。请查看 Cloud 日志。",
        "draft_reason_no_hits": "ℹ️ 进入草案模式：未检索到证据片段，未调用 OpenAI。",
        "draft_reason_demo": "ℹ️ 已进入演示模式：检索后端在当前环境不可用。",
    },
    "en": {
        "title": "CareMind · Clinical Decision Support (MVP)",
        "question_label": "Enter your clinical question",
        "question_ph": "e.g., For CKD patients on ACEI/ARB, how to monitor and how often?",
        "drug_label": "(Optional) Drug name (e.g., Aspirin)",
        "submit": "Generate Advice",
        "tab_advice": "🧭 Advice",
        "tab_evidence_list": "📑 Evidence List",
        "tab_hits_raw": "🎯 Hits (Raw)",
        "tab_hits": "📚 Evidence",
        "tab_drug": "💊 Drug (Structured)",
        "tab_log": "🪵 Run Logs",
        "settings": "⚙️ Settings",
        "k_slider": "Top-K retrieved segments",
        "show_meta": "Show snippet metadata",
        "expand_hits": "Expand all snippets",
        "filters": "🧩 Evidence Filters (client-side)",
        "filter_src": "Filter by source (optional, substring)",
        "filter_year": "Year range",
        "presets": "🧪 Question Presets",
        "preset_select": "Quick pick",
        "preset_none": "——",
        "preset1": "Monitoring ACEI/ARB in CKD + Hypertension",
        "preset2": "Elderly with T2DM+CAD: target BP and first-line therapy",
        "preset3": "GDM: when to start insulin",
        "advice_hdr": "Advice (with citations & compliance note)",
        "advice_hdr_llm": "Advice (with citations & compliance statement)",
        "advice_hdr_draft": "Clinical Advice (Draft)",
        "advice_hdr_demo": "Demo Output",
        "time_used": "⏱️ Elapsed: {:.2f}s",
        "export_advice": "Export Advice (Markdown)",
        "export_evidence": "Export Evidence (Markdown)",
        "disclaimer": "⚠️ For clinical reference only. Not a substitute for diagnosis/prescription.",
        "hits_hdr": "Retrieved segments (Top-{k}, {n} after filtering)",
        "no_hits": "No snippets match the current filters.",
        "drug_hdr": "Drug Structured Info (SQLite)",
        "no_drug": "No structured drug info provided or found.",
        "log_export": "Export session logs (JSON)",
        "history_hdr": "🗂️ Session History (click to reuse)",
        "no_history": "No history yet.",
        "reuse": "Reuse",
        "reused_tip": "Reused: {q} (Drug: {drug}, K={k}). Edit then generate again.",
        "page_footer": "© CareMind · MVP CDSS | For clinical reference only.",
        "chips_src": "Source:",
        "chips_year": "Year:",
        "chips_id": "ID:",
        "stats_hits": "Snippets: {n} · Total chars: {c}",
        "warn_need_q": "Please enter a clinical question first.",
        "err_backend": "Backend error (see logs/diagnostics below).",
        "diag_title": "🔎 Environment Diagnostics",
        "diag_cfg": "Effective config (Secrets-first):",
        "diag_chroma": "Chroma collections:",
        "diag_chroma_err": "Chroma access error: ",
        "diag_sqlite": "SQLite tables:",
        "diag_sqlite_err": "SQLite error: ",
        "draft_reason_missing_key": "ℹ️ Draft mode: OPENAI_API_KEY is not set (Streamlit Cloud → Manage app → Secrets).",
        "draft_reason_openai_error": "ℹ️ Draft mode: OpenAI call failed ({err}). Check Cloud logs.",
        "draft_reason_no_hits": "ℹ️ Draft mode: no evidence snippets were retrieved; OpenAI was not called.",
        "draft_reason_demo": "ℹ️ Demo mode: retrieval backend is unavailable in this environment.",
    },
}
def t(lang: str, key: str) -> str:
    return I18N.get(lang, I18N["zh"]).get(key, key)


# =============================================================================
# 3) 轻量样式
# -----------------------------------------------------------------------------
st.set_page_config(page_title="CareMind · MVP CDSS", layout="wide", page_icon="💊")
st.markdown("""
<style>
.cm-badge{display:inline-block;padding:2px 8px;border-radius:12px;font-size:12px;background:#eef2ff;border:1px solid #c7d2fe;margin-right:6px;white-space:nowrap;}
.cm-chip{display:inline-block;padding:2px 8px;border-radius:8px;font-size:12px;background:#f1f5f9;border:1px solid #e2e8f0;margin:0 6px 6px 0;}
.cm-muted{color:#64748b;font-size:13px;}
.cm-output{line-height:1.75;font-size:17px;}
.cm-card{border:1px solid #e5e7eb;background:#fff;border-radius:12px;padding:12px 14px;margin-bottom:10px;}
footer{visibility:hidden;}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# 4) 侧边栏（设置/过滤/预设/历史）
# -----------------------------------------------------------------------------
with st.sidebar:
    lang = st.selectbox("Language / 语言", options=["zh", "en"], index=0,
                        format_func=lambda x: "中文" if x == "zh" else "English")
    st.header(t(lang, "settings"))

    k = st.slider(t(lang, "k_slider"), min_value=2, max_value=8, value=4, step=1)
    show_meta = st.toggle(t(lang, "show_meta"), value=True)
    expand_hits = st.toggle(t(lang, "expand_hits"), value=False)

    st.divider()
    st.markdown(f"#### {t(lang, 'filters')}")
    src_filter = st.text_input(t(lang, "filter_src"))
    year_min, year_max = st.slider(t(lang, "filter_year"), 2000, 2035, (2005, 2035))
    st.divider()

    st.markdown(f"#### {t(lang, 'presets')}")
    presets = {
        "zh": {
            t("zh","preset1"): "慢性肾病（CKD）患者使用 ACEI/ARB 时如何监测？多久复查？",
            t("zh","preset2"): "老年合并糖尿病与冠心病的降压目标与首选方案？",
            t("zh","preset3"): "妊娠期糖尿病控制不佳时胰岛素起始指征与剂量？",
        },
        "en": {
            t("en","preset1"): "For CKD on ACEI/ARB, what to monitor and how often?",
            t("en","preset2"): "Elderly with T2DM+CAD: target BP and first-line therapy?",
            t("en","preset3"): "GDM: when to start insulin",
        }
    }
    preset_none = t(lang, "preset_none")
    preset_choice = st.selectbox(t(lang, "preset_select"),
                                 options=[preset_none] + list(presets[lang].keys()),
                                 index=0)

    # 会话历史（侧边栏显示概览）
    st.markdown(f"#### {t(lang, 'history_hdr')}")
    hist = st.session_state.setdefault("cm_history", [])
    if not hist:
        st.caption(t(lang, "no_history"))
    else:
        for idx, h in enumerate(reversed(hist[-8:]), 1):
            st.write(f"{idx}. {h.get('q')[:36]}{'...' if len(h.get('q'))>36 else ''}")
            if st.button(t(lang, "reuse"), key=f"reuse_side_{idx}"):
                st.session_state["prefill"] = h


# =============================================================================
# 5) 输入表单
# -----------------------------------------------------------------------------
st.title(t(lang, "title"))
with st.form("cm_query"):
    prefill = st.session_state.pop("prefill", None)
    q_init = (prefill or {}).get("q") or (presets[lang].get(preset_choice, "") if preset_choice != preset_none else "")
    k_pref = (prefill or {}).get("k")
    if k_pref is not None:
        k = int(k_pref)

    q = st.text_input(t(lang, "question_label"),
                      placeholder=t(lang, "question_ph"),
                      value=q_init)
    drug = st.text_input(t(lang, "drug_label"), value=(prefill or {}).get("drug", ""))
    submitted = st.form_submit_button(t(lang, "submit"), use_container_width=True)


# =============================================================================
# 6) 页签：建议 / 证据片段 / 药品结构化 / 运行日志
# -----------------------------------------------------------------------------
# tab_adv, tab_hits, tab_drug, tab_log = st.tabs([
#     t(lang, "tab_advice"),
#    t(lang, "tab_hits"),
#    t(lang, "tab_drug"),
#    t(lang, "tab_log"),
#])

tab_adv, tab_evidence, tab_hits, tab_drug, tab_log = st.tabs([
    t(lang, "tab_advice"),
    t(lang, "tab_evidence_list"),
    t(lang, "tab_hits_raw"),
    t(lang, "tab_drug"),
    t(lang, "tab_log"),
])



res: Optional[Dict[str, Any]] = None
elapsed: Optional[float] = None


# =============================================================================
# 7) 调用后端（反射式，兼容是否含 lang 参数）
# -----------------------------------------------------------------------------
if submitted:
    if not (q and q.strip()):
        st.warning(t(lang, "warn_need_q"))
    else:
        with st.spinner("..."):
            try:
                t0 = time.time()
                # Users sometimes paste "药品名称：xxx" into the question box; strip such UI-label lines.
                q_clean = "\n".join(
                    [
                        ln
                        for ln in (q or "").splitlines()
                        if not re.match(r"(?i)^\s*(药品名称|药品|drug\s*name|drug)\s*[:：]", ln.strip())
                    ]
                ).strip()
                sig_params = inspect.signature(cm_pipeline.answer).parameters
                if "lang" in sig_params:
                    res = cm_pipeline.answer(
                        q_clean, drug_name=(drug.strip() or None), k=int(k), lang=lang
                    )
                else:
                    res = cm_pipeline.answer(
                        q_clean, drug_name=(drug.strip() or None), k=int(k)
                    )
                elapsed = time.time() - t0

                # Store last retrieval stats for diagnostics.
                try:
                    st.session_state["cm_last_question"] = q_clean
                    st.session_state["cm_last_hit_count"] = len(res.get("guideline_hits") or [])
                except Exception:
                    pass

                # 记录到会话历史
                st.session_state.setdefault("cm_history", []).append(
                    {"q": q.strip(), "drug": (drug.strip() or None), "k": int(k), "time": time.time()}
                )
            except Exception as e:
                st.error(t(lang, "err_backend"))
                hints = friendly_hints(lang, e)
                if hints:
                    st.info("· " + "\n· ".join(hints))
                st.exception(e)
                res = None


# =============================================================================
# 8) 渲染结果
# -----------------------------------------------------------------------------
if res:
    # --- 建议 ---
    with tab_adv:
        mode = res.get("mode")
        hdr_key = "advice_hdr"
        if mode == "draft":
            hdr_key = "advice_hdr_draft"
        elif mode == "demo":
            hdr_key = "advice_hdr_demo"
        elif mode == "llm":
            hdr_key = "advice_hdr_llm"

        st.subheader(t(lang, hdr_key))
        if mode == "draft":
            if not res.get("openai_key_present"):
                st.info(t(lang, "draft_reason_missing_key"))
            elif res.get("openai_error_type"):
                st.info(t(lang, "draft_reason_openai_error").format(err=res.get("openai_error_type")))
            else:
                st.info(t(lang, "draft_reason_no_hits"))
        elif mode == "demo":
            st.info(t(lang, "draft_reason_demo"))

        raw_out = res.get("output") or ""
        advice_md, evidence_list_md = split_advice_and_evidence_list(raw_out)
        advice_md = link_citations(advice_md)
        evidence_list_md = link_citations(evidence_list_md)
        advice_md = strip_redundant_advice_heading(advice_md, lang)

        # Render only the advice section inside the Advice tab.
        st.markdown(advice_md, unsafe_allow_html=False)
        if elapsed is not None:
            st.caption(t(lang, "time_used").format(elapsed))

        # 2) compact downloads below, side-by-side
        ev_md = evidence_md(lang, res.get("guideline_hits") or [])
        b1, b2, _spacer = st.columns([1, 1, 4])
        with b1:
            st.download_button(
                t(lang, "export_advice"),
                data=(advice_md or "").encode("utf-8"),
                file_name="caremind_advice.md",
                mime="text/markdown",
                use_container_width=True,
                disabled=not bool((advice_md or "").strip()),
            )
        with b2:
            st.download_button(
                t(lang, "export_evidence"),
                data=(ev_md or "").encode("utf-8"),
                file_name="caremind_evidence.md",
                mime="text/markdown",
                use_container_width=True,
                disabled=not bool((ev_md or "").strip()),
            )
        st.caption(t(lang, "disclaimer"))
      
    # --- 证据片段 ---

    # --- 证据页签：只显示一次（后端整理的证据清单） ---
    with tab_evidence:
        hits_for_list: List[Dict[str, Any]] = res.get("guideline_hits") or []
        ev_list = evidence_list_md.strip() if evidence_list_md.strip() else evidence_list_md_from_hits(lang, hits_for_list)
        ev_list = link_citations(ev_list)
        ev_list = normalize_evidence_list_md(ev_list, lang)
        st.markdown(ev_list, unsafe_allow_html=False)

    with tab_hits:
        hits: List[Dict[str, Any]] = res.get("guideline_hits") or []

        def pass_filter(h: Dict[str, Any]) -> bool:
            m = h.get("meta") or {}
            src_ok = (src_filter.strip().lower() in (m.get("source", "").lower())) if src_filter.strip() else True
            try:
                y = int(m.get("year"))
            except Exception:
                y = None
            year_ok = (year_min <= y <= year_max) if y else True
            return src_ok and year_ok

        hits = [h for h in hits if pass_filter(h)]
        st.subheader(t(lang, "hits_hdr").format(k=k, n=len(hits)))
        if not hits:
            st.info(t(lang, "no_hits"))
        else:
            counts: Dict[str, int] = {}
            for h in hits:
                m = h.get("meta") or {}
                s = str(m.get("source") or ("未知来源" if lang == "zh" else "Unknown")).strip()
                counts[s] = counts.get(s, 0) + 1
            st.markdown(" ".join(
                [f"<span class='cm-chip'>{s} × {n}</span>" for s, n in counts.items()]
            ), unsafe_allow_html=True)

            for i, h in enumerate(hits, 1):
                m = h.get("meta") or {}
                # title  = str(m.get("title")  or ("无标题" if lang == "zh" else "Untitled"))
                # source = str(m.get("source") or ("未知"   if lang == "zh" else "Unknown"))
                # year   = str(m.get("year")   or "—")
                title  = (m.get("title") or m.get("doc_title") or m.get("section_title") or "Untitled")
                source = (m.get("source") or m.get("source_filename") or "Unknown")
                year   = (m.get("year") or "")
                doc_id = str(m.get("id")     or "—")
                label = f"{i} · {title[:60]}"
                st.markdown(f"<a id='hit-{i}'></a>", unsafe_allow_html=True)
                with st.expander(label, expanded=False):
                    if show_meta:
                        st.markdown(
                            f"<div class='cm-muted'>"
                            f"<span class='cm-badge'>{t(lang, 'chips_src')} {source}</span>"
                            f"<span class='cm-badge'>{t(lang, 'chips_year')} {year}</span>"
                            f"<span class='cm-badge'>{t(lang, 'chips_id')} {doc_id}</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                    st.markdown(h.get("content") or ("（空片段）" if lang == "zh" else "(empty)"))

    # --- 药品结构化 ---
    with tab_drug:
        st.subheader(t(lang, "drug_hdr"))
        if res.get("drug"):
            st.json(res["drug"], expanded=False)
        else:
            st.caption(t(lang, "no_drug"))

    # --- 运行日志 ---
    with tab_log:
        log = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "lang": lang,
            "question": q.strip(),
            "drug": drug.strip() or None,
            "k": int(k),
            "elapsed_sec": round(elapsed or 0, 3),
            "sources": [ (h.get("meta") or {}).get("source") for h in (res.get("guideline_hits") or []) ],
        }
        st.json(log)
        st.download_button(
            t(lang, "log_export"),
            data=json.dumps([log], ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="caremind_logs.json",
            mime="application/json",
            use_container_width=True,
        )

    with st.expander("⚙️ 开发者工具 / Dev tools"):
        if st.button("🔄 清理后端缓存 (Chroma Client/Collection)"):
            try:
                st.cache_resource.clear()
                st.success("已清理，请重新提交查询。")
            except Exception as e:
                st.error(f"清理失败：{e}")

# =============================================================================
# 9) 诊断面板（始终可见；统一使用 retriever 的安全接口）
# -----------------------------------------------------------------------------
def render_diagnostics(lang: str = "zh") -> None:
    title = t(lang, "diag_title")
    with st.expander(title, expanded=False):
        # 有效配置（Secrets 优先）
        keys = ["CAREMIND_DEMO", "CHROMA_PERSIST_DIR", "CHROMA_COLLECTION",
                "EMBEDDING_MODEL", "DRUG_DB_PATH"]
        eff = {k: _env(k, None) for k in keys}
        st.write(t(lang, "diag_cfg"))
        st.code(json.dumps(eff, ensure_ascii=False, indent=2))
        # retriever 版本号（确认云端是否更新到位）
        st.write("Retriever version:", getattr(R, "VERSION", "unknown"))

        # Chroma 目录存在性
        chroma_dir = eff.get("CHROMA_PERSIST_DIR") or "./chroma_store"
        abs_chroma = os.path.abspath(chroma_dir)
        st.write(f"{'Chroma 目录存在：' if lang=='zh' else 'Chroma dir exists:'} "
                 f"{abs_chroma} → {os.path.exists(abs_chroma)}")

        # 集合列表（安全方式）与活动集合块数
        try:
            cols = R.list_collections_safe()
            st.write(t(lang, "diag_chroma"))
            st.json(cols)
        except Exception as e:
            st.warning(t(lang, "diag_chroma_err") + str(e))

        try:
            count = R.primary_collection_count()
            st.markdown(f"**Chunks in active collection (`{os.getenv('CHROMA_COLLECTION')}`)**: `{count}`")
        except Exception:
            st.markdown("**Chunks in active collection**: `-`")

        # If the last query got 0 hits, show an actionable hint.
        last_hits = st.session_state.get("cm_last_hit_count", None)
        if last_hits == 0:
            q_last = (st.session_state.get("cm_last_question") or "").strip()
            if lang == "zh":
                msg = (
                    "本次检索命中为 0，因此‘证据清单’与‘药品结构化’可能为空。\n\n"
                    "建议排查：\n"
                    "- CHROMA_PERSIST_DIR 指向的目录在 Cloud 中是否存在且包含索引文件\n"
                    "- CHROMA_COLLECTION 名称是否正确，且集合内是否有向量（上方 chunks > 0）\n"
                    "- 若尚未构建索引：先在本地运行 ingest/build_vectors.py 生成向量库，并确保 Cloud 可访问该目录"
                )
                if q_last:
                    msg += f"\n\n最近一次问题：{q_last}"
                st.warning(msg)
            else:
                msg = (
                    "The last retrieval returned 0 hits, so the Evidence List / Drug sections may be empty.\n\n"
                    "Checks:\n"
                    "- Does CHROMA_PERSIST_DIR exist on Cloud and contain the index files?\n"
                    "- Is CHROMA_COLLECTION correct and does it contain vectors (chunks > 0 above)?\n"
                    "- If you haven't built the index yet, run ingest/build_vectors.py locally and make the index available to Cloud"
                )
                if q_last:
                    msg += f"\n\nLast question: {q_last}"
                st.warning(msg)

        # SQLite 存在性与表
        db_path = eff.get("DRUG_DB_PATH") or "./db/drugs.sqlite"
        abs_db = os.path.abspath(db_path)
        st.write(f"{'SQLite 文件存在：' if lang=='zh' else 'SQLite file exists:'} "
                 f"{abs_db} → {os.path.exists(abs_db)}")
        try:
            con = sqlite3.connect(abs_db)
            cur = con.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [r[0] for r in cur.fetchall()]
            con.close()
            st.write(t(lang, "diag_sqlite"))
            st.json(tables)
        except Exception as e:
            st.warning(t(lang, "diag_sqlite_err") + str(e))

APPEND_DISCLAIMER = False  # 统一由 prompt 生成

if APPEND_DISCLAIMER:
    st.info("本工具仅供临床决策参考，不替代医师诊断与处方。")

# 页面底部渲染诊断
render_diagnostics(lang)


# =============================================================================
# 10) 页脚
# -----------------------------------------------------------------------------
st.caption(t(lang, "page_footer"))