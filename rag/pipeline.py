# -*- coding: utf-8 -*-
"""
rag/pipeline.py
----------------
Orchestrates retrieval + (optional) reasoning between the Streamlit UI (app.py)
and the backend retriever (rag/retriever.py).

Design goals:
- Keep imports light so the app can boot on Streamlit Cloud even if
  Chroma/SQLite aren't available.
- Provide a DEMO mode fallback (no hard crash on Cloud); in DEMO we still return
  a well-formed response so the UI renders end-to-end.
- **NEW**: Accept a `lang` parameter ("zh" | "en") so the *answers* match the UI
  language, not just the UI chrome.

Public API (called from app.py):
    answer(question: str, drug_name: Optional[str], k: int = 4, lang: str = "zh")
        -> Dict[str, Any] with keys: output, guideline_hits, drug
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import os
import re
import traceback

# We defer heavy libs to retriever.py (which lazy-imports chromadb/sqlite)
from . import retriever as R
from .prompt import SYSTEM, USER_TEMPLATE  # keep your existing prompt templates

# -----------------------------------------------------------------------------
# Config flags
# -----------------------------------------------------------------------------
DEMO: bool = os.getenv("CAREMIND_DEMO", "1") == "1"   # default ON for Cloud
MAX_K: int = int(os.getenv("CAREMIND_MAX_K", "8"))

# -----------------------------------------------------------------------------
# Lightweight data model
# -----------------------------------------------------------------------------
@dataclass
class AnswerBundle:
    output: str
    guideline_hits: List[Dict[str, Any]]
    drug: Optional[Dict[str, Any]]

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _clamp_k(k: int) -> int:
    try:
        k = int(k)
    except Exception:
        k = 4
    return max(1, min(MAX_K, k))

def _render_with_citations(raw_text: str) -> str:
    """
    Hook for post-processing citations (e.g., mapping [#1] to anchors).
    Currently a no-op; kept for clarity/extension.
    """
    return raw_text or ""

def _compose_user_prompt(question: str, drug_name: Optional[str], hits: List[Dict[str, Any]]) -> str:
    """
    Fill USER_TEMPLATE with question, optional drug name, and selected evidence
    (as small Markdown sections). Keeps this module model-agnostic.
    """
    lines = []
    for i, h in enumerate(hits or [], 1):
        m = h.get("meta") or {}
        title  = str(m.get("title")  or "Untitled")
        source = str(m.get("source") or "Unknown")
        year   = str(m.get("year")   or "—")
        content = str(h.get("content") or "")
        lines.append(f"### #{i} {title}\n- Source: {source} · Year: {year}\n\n{content}\n")

    evidence_md = "\n".join(lines)
    # USER_TEMPLATE should define placeholders: {question}, {drug}, {evidence_md}
    return USER_TEMPLATE.format(question=question, drug=(drug_name or ""), evidence_md=evidence_md)

def _i18n(lang: str, key: str) -> str:
    """
    Minimal inline i18n for texts generated *by the pipeline* (demo messages, headings).
    UI strings remain in app.py's I18N; we only localize pipeline-generated content here.
    """
    ZH = {
        "hdr_demo":        "临床建议（演示）",
        "hdr_draft":       "临床建议（草案）",
        "q":               "问题",
        "drug":            "药品",
        "evidence":        "证据（选摘）",
        "none":            "无",
        "none_hits":       "暂无证据片段。",
        "note":            "合规提示：本工具仅供临床决策参考，不代替医生诊断与处方。",
        "demo_explain_1":  "这是演示回退（检索后端在当前环境不可用）。",
        "demo_explain_2":  "要在 Streamlit Cloud 启用完整检索：",
        "demo_step_1":     "1) 安装 `pysqlite3-binary` 并在 retriever.py 中别名为 `sqlite3`；",
        "demo_step_2":     "2) 准备/挂载 Chroma 索引与集合（或构建一个小型演示集）；",
        "demo_step_3":     "3) 在 retriever 函数内部惰性导入 chromadb，避免导入期失败。",
    }
    EN = {
        "hdr_demo":        "Clinical Advice (Demo)",
        "hdr_draft":       "Clinical Advice (Draft)",
        "q":               "Question",
        "drug":            "Drug",
        "evidence":        "Rationale / Evidence (selected)",
        "none":            "None",
        "none_hits":       "No evidence snippets available.",
        "note":            "Compliance note: for clinical reference only; not a substitute for diagnosis/prescription.",
        "demo_explain_1":  "This is a demo fallback because the retrieval backend isn't available in this environment.",
        "demo_explain_2":  "To enable full retrieval on Streamlit Cloud:",
        "demo_step_1":     "1) Install `pysqlite3-binary` and alias it to `sqlite3` in retriever.py;",
        "demo_step_2":     "2) Ensure your Chroma index path & collection exist (or build a tiny demo set);",
        "demo_step_3":     "3) Lazy-import chromadb inside retriever functions to avoid import-time failures.",
    }
    return (ZH if lang == "zh" else EN).get(key, key)

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def answer(question: str,
           drug_name: Optional[str] = None,
           k: int = 4,
           lang: str = "zh") -> Dict[str, Any]:
    """
    Main entry called by app.py.

    Parameters
    ----------
    question : str
        The clinical question from user input (in UI language).
    drug_name : Optional[str]
        Optional drug name (e.g., 'Aspirin'); may drive structured lookup.
    k : int
        Top-K for snippet retrieval (clamped to a safe range).
    lang : str
        'zh' or 'en'. Controls the **language of the generated text** in this pipeline.

    Returns
    -------
    dict:
        {
          "output": "<markdown text in selected language>",
          "guideline_hits": [...],
          "drug": {...} | None
        }
    """
    kk = _clamp_k(k)

    try:
        # 1) Retrieve guideline snippets (Chroma; lazy-imported in retriever.py)
        hits: List[Dict[str, Any]] = R.search_guidelines(question, k=kk) or []

        # 2) Structured drug info (SQLite; optional)
        drug_struct = None
        if drug_name and drug_name.strip():
            try:
                drug_struct = R.search_drug_structured(drug_name.strip())
            except Exception:
                # Do not fail the whole pipeline on structured lookup
                drug_struct = None

        # 3) Compose prompt for your LLM (if you wire one later).
        # For now we produce a concise, localized "draft" with citations list.
        # If you connect an LLM, replace the block below with the actual call.
        lines: List[str] = []
        lines.append(f"**{_i18n(lang, 'hdr_draft')}**")
        lines.append("")
        q_label = _i18n(lang, "q")
        d_label = _i18n(lang, "drug")
        lines.append(f"- **{q_label}:** {question}")
        if drug_name:
            lines.append(f"- **{d_label}:** {drug_name}")
        lines.append("")
        lines.append(f"**{_i18n(lang, 'evidence')}:**")
        if not hits:
            lines.append(f"- {_i18n(lang, 'none_hits')}")
        else:
            for i, h in enumerate(hits, 1):
                m = h.get("meta") or {}
                title = str(m.get("title") or ("无标题" if lang == "zh" else "Untitled"))
                src   = str(m.get("source") or ("未知" if lang == "zh" else "Unknown"))
                year  = str(m.get("year") or "—")
                lines.append(f"- [#{i}] {title} ({src}, {year})")
        lines.append("")
        lines.append(f"_{_i18n(lang, 'note')}_")

        output_text = _render_with_citations("\n".join(lines))

        return AnswerBundle(
            output=output_text,
            guideline_hits=hits,
            drug=drug_struct,
        ).__dict__

    except Exception:
        # DEMO fallback: localized message so UI + content match the user's language.
        if DEMO:
            traceback.print_exc()
            hits: List[Dict[str, Any]] = [{
                "content": _i18n(lang, "demo_explain_1"),
                "meta": {"title": "Demo", "source": "Demo", "year": "—", "id": "demo-0001"},
            }]
            lines = [
                f"**{_i18n(lang, 'hdr_demo')}**",
                "",
                f"- **{_i18n(lang, 'q')}:** {question}",
            ]
            if drug_name:
                lines.append(f"- **{_i18n(lang, 'drug')}:** {drug_name}")
            lines += [
                "",
                _i18n(lang, "demo_explain_2"),
                _i18n(lang, "demo_step_1"),
                _i18n(lang, "demo_step_2"),
                _i18n(lang, "demo_step_3"),
                "",
                f"_{_i18n(lang, 'note')}_",
            ]
            return AnswerBundle(
                output="\n".join(lines),
                guideline_hits=hits,
                drug=None,
            ).__dict__

        # Non-demo: re-raise so Streamlit shows the real error in logs
        raise
