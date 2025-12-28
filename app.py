# -*- coding: utf-8 -*-
"""
CareMind · MVP CDSS (Streamlit; bilingual zh/en)
------------------------------------------------
Features
- Bilingual UI (Chinese / English)
- Generates advice via rag.pipeline.answer (reflective call; compatible with or without a lang parameter)
- Tabs for Evidence List / Evidence Snippets / Drug (structured) / Run Logs
- Diagnostics panel: shows effective config (Secrets-first), chroma_store existence, and Chroma collection counts
    (via retriever.list_collections_safe / primary_collection_count to avoid constructing a second Chroma client)
- Session history with one-click reuse
- Shows Python/sqlite/Torch/Chroma versions (without constructing an extra Chroma client)
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
# Import retriever first: it installs the SQLite shim when needed (helpful on Streamlit Cloud with older SQLite).
from rag import retriever as R  # noqa: F401   # Provides shim + cached singleton Chroma client
import rag.pipeline as cm_pipeline          # Import as a module to avoid symbol shadowing under hot reload
import sqlite3
from dotenv import load_dotenv
load_dotenv()


# =============================================================================
# 1) Helpers: Secrets-first env reading + utility helpers
# -----------------------------------------------------------------------------
def _env(key: str, default: str | None = None) -> str | None:
    """
    Secrets-aware env reader:
    prefer st.secrets[key], then os.environ[key], then default.
    """
    try:
        return os.getenv(key, st.secrets.get(key, default))
    except Exception:
        return os.getenv(key, default)

def link_citations(md: str) -> str:
    """Convert "[#3]" / "[3]" into a "#hit-3" anchor link so advice can jump back to evidence snippets."""
    # Keep the visible text as "[n]" (not just "n") so users can see bracketed citations.
    # Use escaped brackets inside the link text to avoid Markdown parsing quirks.
    return re.sub(r"\[(?:#)?(\d+)\]", r"[\\[\1\\]](#hit-\1)", md or "")

def split_advice_and_evidence_list(md: str) -> tuple[str, str]:
    """Split model output into (advice_md, evidence_list_md).

    The OpenAI prompt templates ask for two sections:
    - "Clinical Recommendation Points"
    - "Evidence List"
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
    """Remove a leading 'Advice:' line (and its localized equivalent) when the UI already provides the section header."""
    if not md:
        return md
    heading_only = r"(?mi)^\s*(?:\*\*|__)?\s*(建议|advice)\s*(?:\*\*|__)?\s*[:：]\s*$"
    lines = md.splitlines()
    # Case A: first line is just a standalone "Advice:" heading (or equivalent).
    if lines and re.match(heading_only, lines[0] or ""):
        lines = lines[1:]
        if lines and not lines[0].strip():
            lines = lines[1:]
        return "\n".join(lines).lstrip()

    # Case B: first line starts with a redundant "Advice: ..." prefix (or equivalent).
    if lines:
        m = re.match(r"(?i)^\s*(建议|advice)\s*[:：]\s*(.+)$", lines[0].strip())
        if m:
            lines[0] = m.group(2).strip()
    return "\n".join(lines).lstrip()


def strip_evidence_list_heading_in_advice(md: str) -> str:
    """Remove stray Evidence List headings that occasionally leak into the advice body."""
    if not md:
        return md
    # Remove standalone "Evidence List:" header lines (including bold wrappers).
    md = re.sub(
        r"(?mi)^\s*(?:\*\*|__)?\s*(evidence\s+list|证据清单)\s*(?:\*\*|__)?\s*[:：]?\s*$\n?",
        "",
        md,
    )
    return md.strip()


def normalize_evidence_list_md(md: str, lang: str) -> str:
    """Normalize Evidence List: remove the visible header and render each [n] entry as one bullet line."""
    txt = (md or "").strip()
    if not txt:
        return txt

    # Strip any explicit header lines; the tab title already provides the label.
    txt = re.sub(
        r"(?mi)^\s*(?:\*\*|__)?\s*(evidence\s+list|证据清单)\s*(?:\*\*|__)?\s*[:：]?\s*$",
        "",
        txt,
    ).strip()

    # Extract items even if the model puts multiple entries on one line.
    matches = list(re.finditer(r"\[(\d+)\]\s+", txt))
    items: List[str] = []
    if matches:
        for idx, m in enumerate(matches):
            start = m.start()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(txt)
            chunk = txt[start:end].strip()
            chunk = re.sub(r"^\s*[-*]\s*", "", chunk).strip()
            chunk = re.sub(r"\s+", " ", chunk).strip()
            if chunk:
                items.append(chunk)
    else:
        # Fallback: treat each non-empty line as an item.
        for ln in txt.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            ln = re.sub(r"^\s*[-*]\s*", "", ln).strip()
            items.append(re.sub(r"\s+", " ", ln).strip())

    if not items:
        return "（暂无证据片段）" if lang == "zh" else "(No evidence snippets available.)"

    # Render as Markdown bullets so Streamlit guarantees one item per line.
    return "\n".join([f"- {it}" for it in items]).strip() + "\n"


def extract_drug_from_question(q: str) -> tuple[str, Optional[str]]:
    """Extract a trailing drug-name line from a pasted template (best-effort).

    Conservative heuristic: only extracts the last non-empty line if it looks like
    a labeled drug line (e.g., "Medicine (optional): ...") and returns (question, drug).
    """
    txt = (q or "").rstrip()
    lines = [ln.rstrip() for ln in txt.splitlines()]
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return (q or ""), None

    last = lines[-1].strip()

    m = re.match(r"(?i)^\s*(medicine\s*\(optional\)|drug\s*name|drug)\s*[:：]\s*(.+)$", last)
    if not m:
        return (q or ""), None
    drug = (m.group(2) or "").strip()
    if not drug:
        return (q or ""), None

    return "\n".join(lines[:-1]).strip(), drug


_SOURCE_FILENAME_EN: Dict[str, str] = {
    "2型糖尿病患者运动方案的最佳证据总结(2019).pdf": "Best Evidence Summary: Exercise Programs for Type 2 Diabetes (2019).pdf",
    "《中国高血压防治指南(2024年修订版)》新增内容解读 ——以改善血压变异和降压目标范围内时间为核心的高质量降压策略浅析_张新军.pdf": "Interpretation: Updates in Chinese Hypertension Guidelines (2024 revision) (Zhang Xinjun).pdf",
    "《妊娠期糖尿病临床护理实践指南》推荐意见专家共识_周英凤(2020).pdf": "Expert Consensus: Recommendations for the Nursing Practice Guideline on Gestational Diabetes (Zhou Yingfeng, 2020).pdf",
    "中国2型糖尿病防治指南（2020年版）.pdf": "Chinese Guideline for Prevention and Treatment of Type 2 Diabetes (2020).pdf",
    "中国慢性肾脏病早期评价与管理指南(2023).pdf": "Chinese Guideline for Early Evaluation and Management of Chronic Kidney Disease (2023).pdf",
    "中国高血压患者心率管理多学科专家共识（2021年版）_高血压心率管理多学科共识组.pdf": "Multidisciplinary Expert Consensus: Heart Rate Management in Hypertensive Patients (2021).pdf",
    "中国高血压防治指南(2024年修订版).pdf": "Chinese Guideline for Prevention and Treatment of Hypertension (2024 revision).pdf",
    "冠心病合并2 型糖尿病患者的血糖管理专家共识(2024)_中国医疗保健国际交流促进会心血管病学分会.pdf": "Expert Consensus: Glycemic Management in CHD with Type 2 Diabetes (2024).pdf",
    "国家基层糖尿病防治管理指南（2022）.pdf": "National Primary Care Guideline: Diabetes Prevention, Treatment, and Management (2022).pdf",
    "国家基层高血压防治管理指南 2020版.pdf": "National Primary Care Guideline: Hypertension Prevention, Treatment, and Management (2020).pdf",
    "妊娠期糖尿病患者产前血糖管理的证据总结_秦煜(2023).pdf": "Evidence Summary: Antenatal Glycemic Management in Gestational Diabetes (Qin Yu, 2023).pdf",
    "成人2型糖尿病的高血压管理中国专家共识(2025).pdf": "Chinese Expert Consensus: Hypertension Management in Adults with Type 2 Diabetes (2025).pdf",
    "成人糖尿病患者血压管理专家共识(2021).pdf": "Expert Consensus: Blood Pressure Management in Adults with Diabetes (2021).pdf",
    "糖尿病患者体重管理专家共识(2024版).pdf": "Expert Consensus: Weight Management in Patients with Diabetes (2024).pdf",
    "糖尿病患者甲病管理的最佳证据总结_陈欢(2022).pdf": "Best Evidence Summary: Thyroid Disorders Management in Patients with Diabetes (Chen Huan, 2022).pdf",
    "糖尿病患者血脂管理中国专家共识（2024版）.pdf": "Chinese Expert Consensus: Lipid Management in Patients with Diabetes (2024).pdf",
}


def _norm_doc_label(s: str) -> str:
    """Normalize doc labels/titles for fuzzy matching (UI-only)."""
    s = (s or "").strip()
    if not s:
        return ""
    s = os.path.basename(s)
    s = os.path.splitext(s)[0]
    s = s.split("_")[0]
    s = s.replace("《", "").replace("》", "")
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[—–\-·•:：]", "", s)
    s = re.sub(r"[\(（]\s*20\d\d[^\)）]*[\)）]", "", s)
    s = re.sub(r"[\(（][^\)）]*年版[\)）]", "", s)
    return s


_DOC_LABEL_EN_BY_NORM: Dict[str, str] = {}
for _zh_filename, _en_filename in _SOURCE_FILENAME_EN.items():
    _en_noext = os.path.splitext(_en_filename)[0]
    _zh_main = os.path.splitext(_zh_filename)[0].split("_")[0]
    _DOC_LABEL_EN_BY_NORM[_norm_doc_label(_zh_filename)] = _en_noext
    _DOC_LABEL_EN_BY_NORM[_norm_doc_label(_zh_main)] = _en_noext


def localize_doc_label(text: str, lang: str) -> str:
    """Localize guideline doc titles/labels for UI display only (English mode)."""
    s = str(text or "").strip()
    if not s or lang != "en":
        return s

    mapped = localize_source_name(s, lang)
    if mapped != s:
        return os.path.splitext(mapped)[0]

    norm = _norm_doc_label(s)
    if norm and norm in _DOC_LABEL_EN_BY_NORM:
        return _DOC_LABEL_EN_BY_NORM[norm]
    return s


def localize_source_name(source: str, lang: str) -> str:
    """Localize guideline source filenames for UI display only."""
    s = str(source or "").strip()
    if not s or lang != "en":
        return s
    base = os.path.basename(s)
    if base in _SOURCE_FILENAME_EN:
        return _SOURCE_FILENAME_EN[base]
    if s in _SOURCE_FILENAME_EN:
        return _SOURCE_FILENAME_EN[s]

    for zh_name, en_name in _SOURCE_FILENAME_EN.items():
        if zh_name in s:
            return s.replace(zh_name, en_name)

    base_noext = os.path.splitext(base)[0]
    for zh_name, en_name in _SOURCE_FILENAME_EN.items():
        zh_noext = os.path.splitext(zh_name)[0]
        if base_noext == zh_noext:
            return os.path.splitext(en_name)[0]
    return s


def localize_source_names_in_text(md: str, lang: str) -> str:
    """Replace known guideline filenames inside Markdown text (UI-only)."""
    if lang != "en":
        return md
    out = md or ""
    for zh_name, en_name in _SOURCE_FILENAME_EN.items():
        out = out.replace(zh_name, en_name)
        out = out.replace(os.path.splitext(zh_name)[0], os.path.splitext(en_name)[0])
        out = out.replace(os.path.splitext(zh_name)[0].split("_")[0], os.path.splitext(en_name)[0])
    return out


def evidence_list_md_from_hits(lang: str, hits: List[Dict[str, Any]]) -> str:
    """Render a compact Evidence List (title/source/year only) from retrieved hits."""
    if not hits:
        return "（暂无证据片段）" if lang == "zh" else "(No evidence snippets available.)"

    lines: List[str] = []
    for i, h in enumerate(hits or [], 1):
        m = h.get("meta") or {}
        title = (m.get("title") or m.get("doc_title") or m.get("section_title") or ("无标题" if lang == "zh" else "Untitled"))
        source = (m.get("source") or m.get("source_filename") or ("未知来源" if lang == "zh" else "Unknown"))
        title = localize_doc_label(title, lang)
        source = localize_doc_label(source, lang)
        year = (m.get("year") or "")
        yr = f"{year}".strip()
        tail = (f"（{yr}）" if lang == "zh" else f"({yr})") if yr else ""
        lines.append(f"- [{i}] {title} — {source} {tail}".rstrip())
    return "\n".join(lines).strip() + "\n"


def evidence_md(lang: str, hits: List[Dict[str, Any]]) -> str:
    """Render evidence snippets as Markdown (for download)."""
    lines: List[str] = []
    for i, h in enumerate(hits or [], 1):
        m = h.get("meta") or {}
        title = (m.get("title") or m.get("doc_title") or m.get("section_title") or "Untitled")
        source = (m.get("source") or m.get("source_filename") or "Unknown")
        title = localize_doc_label(title, lang)
        source = localize_doc_label(source, lang)
        year = (m.get("year") or "")

        head = (
            f"### {i} {title}\n\n"
            + (f"- 来源：{source} · 年份：{year}\n\n" if lang == "zh" else f"- Source: {source} · Year: {year}\n\n")
        )
        lines.append(head + (h.get("content") or "") + "\n")
    return "\n".join(lines)

def friendly_hints(lang: str, exc: Exception) -> List[str]:
    """Map common backend exceptions to user-friendly troubleshooting hints."""
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
# 2) Minimal i18n (UI copy; pipeline-generated text is localized server-side)
# -----------------------------------------------------------------------------
I18N: Dict[str, Dict[str, str]] = {
    "zh": {
        "title": "CareMind · 临床决策支持（MVP）",
        "question_label": "输入临床问题",
        "question_ph": "例如：慢性肾病（CKD）患者使用 ACEI/ARB 时如何监测？多久复查？",
        "drug_label": "指定药品名（可选）",
        "drug_ph": "例如：阿司匹林",
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
        "hits_hdr": "检索片段（Top-{k}即过滤后留 {n} 条）",
        "no_hits": "未检索到符合筛选条件的片段。",
        "drug_hdr": "药品结构化信息（SQLite）",
        "no_drug": "未提供或未检索到对应药品的结构化信息。",
        "log_export": "导出本会话全部日志（JSON）",
        "history_hdr": "🗂️ 本会话历史（点击复用）",
        "no_history": "暂无历史记录。",
        "reuse": "复用",
        "reused_tip": "已复用：{q}（药品：{drug}，检索片段数（Top-K）={k}）。可编辑后再次生成。",
        "page_footer": "© CareMind · MVP CDSS | 本工具仅供临床决策参考，不替代医师诊断与处方。",
        "chips_src": "来源：",
        "chips_year": "年份：",
        "chips_id": "ID：",
        "stats_hits": "片段数：{n} · 总字数：{c}",
        "warn_need_q": "请输入临床问题后再生成建议。",
        "err_backend": "后端错误（详见下方日志/诊断）。",
        "diag_title": "🔎 环境诊断",
        "diag_note": "仅供开发/调试使用；普通用户可忽略。",
        "diag_cfg": "有效配置（优先 Secrets）",
        "diag_chroma": "Chroma 集合：",
        "diag_chroma_err": "Chroma 访问错误：",
        "diag_sqlite": "SQLite 表：",
        "diag_sqlite_err": "SQLite 错误：",
        "draft_reason_missing_key": "ℹ️ 进入草案模式：未检测到 OPENAI_API_KEY（请在 Streamlit Cloud → Manage app → Secrets 中配置）。",
        "draft_reason_openai_error": "ℹ️ 进入草案模式：OpenAI 调用失败（{err}）。请查看 Cloud 日志。",
        "draft_reason_no_hits": "ℹ️ 进入草案模式：未检索到证据片段，未调用 OpenAI。",
        "draft_reason_demo": "ℹ️ 已进入演示模式：检索后端在当前环境不可用。",
        "dev_tools_hdr": "⚙️ 开发者工具",
        "clear_backend_cache": "清理后端缓存",
        "clear_backend_cache_ok": "已清理，请重新提交查询。",
        "clear_backend_cache_fail": "清理失败：{err}",
    },
    "en": {
        "title": "CareMind · Clinical Decision Support (MVP)",
        "question_label": "Enter your clinical question",
        "question_ph": "e.g., For CKD patients on ACEI/ARB, how to monitor and how often?",
        "drug_label": "(Optional) Drug name",
        "drug_ph": "e.g., Aspirin",
        "submit": "Generate Advice",
        "tab_advice": "🧭 Advice",
        "tab_evidence_list": "📑 Evidence List",
        "tab_hits_raw": "🎯 Hits (Raw)",
        "tab_hits": "📚 Evidence",
        "tab_drug": "💊 Drug (Structured)",
        "tab_log": "🪵 Run Logs",
        "settings": "⚙️ Settings",
        "k_slider": "Top-K (retrieval snippets)",
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
        "reused_tip": "Reused: {q} (Drug: {drug}, Top-K={k}). Edit then generate again.",
        "page_footer": "© CareMind · MVP CDSS | For clinical reference only.",
        "chips_src": "Source:",
        "chips_year": "Year:",
        "chips_id": "ID:",
        "stats_hits": "Snippets: {n} · Total chars: {c}",
        "warn_need_q": "Please enter a clinical question first.",
        "err_backend": "Backend error (see logs/diagnostics below).",
        "diag_title": "🔎 Environment Diagnostic",
        "diag_note": "For developers/debugging only; most users can ignore.",
        "diag_cfg": "Effective config (Secrets-first):",
        "diag_chroma": "Chroma collections:",
        "diag_chroma_err": "Chroma access error: ",
        "diag_sqlite": "SQLite tables:",
        "diag_sqlite_err": "SQLite error: ",
        "draft_reason_missing_key": "ℹ️ Draft mode: OPENAI_API_KEY is not set (Streamlit Cloud → Manage app → Secrets).",
        "draft_reason_openai_error": "ℹ️ Draft mode: OpenAI call failed ({err}). Check Cloud logs.",
        "draft_reason_no_hits": "ℹ️ Draft mode: no evidence snippets were retrieved; OpenAI was not called.",
        "draft_reason_demo": "ℹ️ Demo mode: retrieval backend is unavailable in this environment.",
        "dev_tools_hdr": "⚙️ Dev tools",
        "clear_backend_cache": "Clear backend cache",
        "clear_backend_cache_ok": "Cleared. Please submit your query again.",
        "clear_backend_cache_fail": "Clear failed: {err}",
    },
}
def t(lang: str, key: str) -> str:
    return I18N.get(lang, I18N["zh"]).get(key, key)


def localize_run_log_for_ui(log: Dict[str, Any], lang: str) -> Dict[str, Any]:
    """Localize run-log keys/values for UI display only."""
    if lang == "zh":
        return {
            "运行日期": log.get("time"),
            "语言": "中文" if log.get("lang") == "zh" else "英文",
            "问题": log.get("question"),
            "药品": log.get("drug"),
            "检索片段数（Top-K）": log.get("k"),
            "耗时（秒）": log.get("elapsed_sec"),
            "来源": log.get("sources"),
        }
    return {
        "Run time": log.get("time"),
        "Language": "English" if log.get("lang") == "en" else "Chinese",
        "Question": log.get("question"),
        "Drug": log.get("drug"),
        "Top-K (retrieval snippets)": log.get("k"),
        "Elapsed (sec)": log.get("elapsed_sec"),
        "Sources": log.get("sources"),
    }


_DRUG_FIELD_LABELS_ZH: Dict[str, str] = {
    "row": "行",
    "ID": "识别码",
    "id": "识别码",
    "indications": "适应症",
    "contraindications": "禁忌症",
}


def localize_drug_record_for_ui(obj: Any, lang: str) -> Any:
    """Localize keys in the structured drug payload for display purposes only."""
    if lang != "zh":
        return obj
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            kk = str(k)
            label = _DRUG_FIELD_LABELS_ZH.get(kk, kk)
            out[label] = localize_drug_record_for_ui(v, lang)
        return out
    if isinstance(obj, list):
        return [localize_drug_record_for_ui(v, lang) for v in obj]
    return obj


# =============================================================================
# 3) Lightweight styles
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
# 4) Sidebar (settings / filters / presets / history)
# -----------------------------------------------------------------------------
with st.sidebar:
    lang = st.selectbox(
        "Language / 语言",
        options=["zh", "en"],
        index=0,
        format_func=lambda x: "中文" if x == "zh" else "English",
        key="cm_lang",
    )

    # Sidebar: environment diagnostics (versions/paths/collection counts)
    # Note: don't instantiate another chromadb.PersistentClient here;
    # always use retriever's singleton client to avoid "different settings" errors.
    with st.expander(t(lang, "diag_title"), expanded=False):
        st.caption(t(lang, "diag_note"))
        st.write("**Python version**:", sys.version)
        st.write("**sqlite3 module version**:", sqlite3.version)          # Python wrapper version
        st.write("**sqlite3 library version**:", sqlite3.sqlite_version)  # Underlying lib version

        # Version info doesn't require a client; pure imports are safe.
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

        # ✅ Use retriever's singleton client for counts to avoid creating a second client.
        try:
            count = R.primary_collection_count()
            st.write("**Collection count (active)**:", count)
            # To list collections, use list_collections_safe (won't raise).
            cols = R.list_collections_safe()
            if cols:
                st.write(
                    "**Collections (name → count)**:",
                    {c.get("name", ""): c.get("count", -1) for c in cols},
                )
        except Exception as e:
            st.error(f"❌ Failed to query collections via retriever: {e}")
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

    # Session history (sidebar summary)
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
# 5) Input form
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
    drug = st.text_input(
        t(lang, "drug_label"),
        placeholder=t(lang, "drug_ph"),
        value=(prefill or {}).get("drug", ""),
    )
    submitted = st.form_submit_button(t(lang, "submit"), use_container_width=True)


# =============================================================================
# 6) Tabs: Advice / Evidence / Drug (Structured) / Run Logs
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

# Persist results across reruns (e.g., clicking download triggers a rerun).
try:
    if st.session_state.get("cm_last_lang") == lang and st.session_state.get("cm_last_res"):
        res = st.session_state.get("cm_last_res")
        elapsed = st.session_state.get("cm_last_elapsed")
except Exception:
    pass


# =============================================================================
# 7) Call backend (reflective; compatible with or without a lang parameter)
# -----------------------------------------------------------------------------
if submitted:
    if not (q and q.strip()):
        st.warning(t(lang, "warn_need_q"))
    else:
        with st.spinner("..."):
            try:
                t0 = time.time()
                # Users sometimes paste a labeled line like "Drug name: ..." into the question box; strip such label lines.
                q_clean = "\n".join(
                    [
                        ln
                        for ln in (q or "").splitlines()
                        if not re.match(r"(?i)^\s*(药品名称|药品|drug\s*name|drug)\s*[:：]", ln.strip())
                    ]
                ).strip()
                # If the dedicated drug input is empty, try extracting a trailing template line.
                extracted_drug: Optional[str] = None
                if not (drug or "").strip():
                    q_clean, extracted_drug = extract_drug_from_question(q_clean)

                drug_effective = (drug or "").strip() or (extracted_drug or "").strip() or None
                sig_params = inspect.signature(cm_pipeline.answer).parameters
                if "lang" in sig_params:
                    res = cm_pipeline.answer(
                        q_clean, drug_name=drug_effective, k=int(k), lang=lang
                    )
                else:
                    res = cm_pipeline.answer(
                        q_clean, drug_name=drug_effective, k=int(k)
                    )
                elapsed = time.time() - t0

                # Persist the full result so UI doesn't clear on reruns (downloads, toggles, etc.).
                try:
                    st.session_state["cm_last_res"] = res
                    st.session_state["cm_last_elapsed"] = elapsed
                    st.session_state["cm_last_lang"] = lang
                except Exception:
                    pass

                # Store last retrieval stats for diagnostics.
                try:
                    st.session_state["cm_last_question"] = q_clean
                    st.session_state["cm_last_hit_count"] = len(res.get("guideline_hits") or [])
                except Exception:
                    pass

                # Record into session history
                st.session_state.setdefault("cm_history", []).append(
                    {"q": q.strip(), "drug": drug_effective, "k": int(k), "time": time.time()}
                )
            except Exception as e:
                st.error(t(lang, "err_backend"))
                hints = friendly_hints(lang, e)
                if hints:
                    st.info("· " + "\n· ".join(hints))
                st.exception(e)
                res = None


# =============================================================================
# 8) Render results
# -----------------------------------------------------------------------------
if res:
    # --- Advice ---
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
        advice_md = strip_evidence_list_heading_in_advice(advice_md)

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

    # --- Evidence List tab: show it once (backend-prepared evidence list) ---
    with tab_evidence:
        hits_for_list: List[Dict[str, Any]] = res.get("guideline_hits") or []
        ev_list = evidence_list_md.strip() if evidence_list_md.strip() else evidence_list_md_from_hits(lang, hits_for_list)
        ev_list = localize_source_names_in_text(ev_list, lang)
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
                s = localize_doc_label(s, lang)
                counts[s] = counts.get(s, 0) + 1
            st.markdown(" ".join(
                [f"<span class='cm-chip'>{s} × {n}</span>" for s, n in counts.items()]
            ), unsafe_allow_html=True)

            for i, h in enumerate(hits, 1):
                m = h.get("meta") or {}
                title  = (m.get("title") or m.get("doc_title") or m.get("section_title") or "Untitled")
                source = (m.get("source") or m.get("source_filename") or "Unknown")
                title = localize_doc_label(title, lang)
                source = localize_doc_label(source, lang)
                year   = (m.get("year") or "")
                doc_id = str(m.get("id")     or "—")
                label = f"[{i}] · {title[:60]}"
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

    # --- Drug (structured) ---
    with tab_drug:
        st.subheader(t(lang, "drug_hdr"))
        if res.get("drug"):
              st.json(localize_drug_record_for_ui(res["drug"], lang), expanded=False)
        else:
            st.caption(t(lang, "no_drug"))

    # --- Run logs ---
    with tab_log:
        log = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "lang": lang,
            "question": q.strip(),
            "drug": drug.strip() or None,
            "k": int(k),
            "elapsed_sec": round(elapsed or 0, 3),
            "sources": [
                localize_doc_label(((h.get("meta") or {}).get("source") or ""), lang)
                for h in (res.get("guideline_hits") or [])
            ],
        }
        st.json(localize_run_log_for_ui(log, lang))
        st.download_button(
            t(lang, "log_export"),
            data=json.dumps([log], ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="caremind_logs.json",
            mime="application/json",
            use_container_width=True,
        )

    with st.expander(t(lang, "dev_tools_hdr")):
        if st.button(f"🔄 {t(lang, 'clear_backend_cache')} (Chroma Client/Collection)"):
            try:
                st.cache_resource.clear()
                st.success(t(lang, "clear_backend_cache_ok"))
            except Exception as e:
                st.error(t(lang, "clear_backend_cache_fail").format(err=e))

# =============================================================================
# 9) Diagnostics panel (always visible; uses retriever safe APIs)
# -----------------------------------------------------------------------------
def render_diagnostics(lang: str = "zh") -> None:
    title = t(lang, "diag_title")
    with st.expander(title, expanded=False):
        st.caption(t(lang, "diag_note"))
        # Effective config (Secrets-first)
        keys = ["CAREMIND_DEMO", "CHROMA_PERSIST_DIR", "CHROMA_COLLECTION",
                "EMBEDDING_MODEL", "DRUG_DB_PATH"]
        eff = {k: _env(k, None) for k in keys}
        st.write(t(lang, "diag_cfg"))
        st.code(json.dumps(eff, ensure_ascii=False, indent=2))
        # Retriever version (helps confirm Cloud updates)
        st.write("Retriever version:", getattr(R, "VERSION", "unknown"))

        # Chroma dir existence
        chroma_dir = eff.get("CHROMA_PERSIST_DIR") or "./chroma_store"
        abs_chroma = os.path.abspath(chroma_dir)
        st.write(f"{'Chroma 目录存在：' if lang=='zh' else 'Chroma dir exists:'} "
                 f"{abs_chroma} → {os.path.exists(abs_chroma)}")

        # Collection list (safe) and active collection chunk count
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

        # SQLite existence and tables
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

APPEND_DISCLAIMER = False  # Generated by the prompt (kept for backward compatibility)

if APPEND_DISCLAIMER:
    st.info("本工具仅供临床决策参考，不替代医师诊断与处方。")

# Render diagnostics at the bottom
render_diagnostics(lang)


# =============================================================================
# 10) Footer
# -----------------------------------------------------------------------------
st.caption(t(lang, "page_footer"))