# -*- coding: utf-8 -*-
"""
rag/pipeline.py
----------------
Orchestrates retrieval + (optional) reasoning between the Streamlit UI (app.py)
and the backend retriever (rag/retriever.py).

设计目标 / Design goals
- 轻量导入，避免在 Cloud 因 sqlite/Chroma 问题导致模块导入即失败。
  Keep imports light so Cloud can boot even if Chroma/SQLite are absent.
- 提供“演示模式”回退（DEMO），即使后端不可用也能渲染 UI。
  Provide a DEMO fallback to render UI even when retrieval backend is unavailable.
- **新增**：支持 `lang`（"zh" 或 "en"），使答案语言与 UI 一致。
  Accepts `lang` to keep generated text consistent with UI language.

Public API (called from app.py):
    answer(question: str, drug_name: Optional[str], k: int = 4, lang: str = "zh")
        -> Dict[str, Any] with keys: output, guideline_hits, drug
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import os
import traceback

# 读取 Secrets 的小助手（Secrets > env > default）/ Secrets-aware env reader
def _env(key: str, default: str | None = None) -> str | None:
    import os
    try:
        import streamlit as st
        return os.getenv(key, st.secrets.get(key, default))  # Secrets 覆盖默认
    except Exception:
        return os.getenv(key, default)

# 延迟把重活交给 retriever（其中做了 lazy import & sqlite shim）
# Defer heavy work to retriever (which lazy-imports chroma & patches sqlite)
from . import retriever as R
from .prompt import SYSTEM, USER_TEMPLATE  # 你的提示模板 / your prompt templates

# -----------------------------------------------------------------------------
# Config flags（Secrets 可覆盖）/ Config flags (overridable via Secrets)
# -----------------------------------------------------------------------------
DEMO: bool = (_env("CAREMIND_DEMO", "1") == "1")   # Cloud 缺省演示模式 ON
MAX_K: int = int(_env("CAREMIND_MAX_K", "8"))

# -----------------------------------------------------------------------------
# Data model 返回给 app.py 的结构 / Bundle returned to app.py
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
    """限制 Top-K 的范围 / Clamp Top-K to a safe range."""
    try:
        k = int(k)
    except Exception:
        k = 4
    return max(1, min(MAX_K, k))

def _render_with_citations(raw_text: str) -> str:
    """
    citation 后处理钩子（留作扩展）/ Citation post-processor hook (no-op for now).
    """
    return raw_text or ""

def _compose_user_prompt(question: str, drug_name: Optional[str], hits: List[Dict[str, Any]]) -> str:
    """
    组装用户提示词，把检索证据拼接到 USER_TEMPLATE 中。
    Compose the end-user prompt by inserting selected evidence into USER_TEMPLATE.
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
    # USER_TEMPLATE 需包含 {question}/{drug}/{evidence_md} 占位符
    return USER_TEMPLATE.format(question=question, drug=(drug_name or ""), evidence_md=evidence_md)

def _i18n(lang: str, key: str) -> str:
    """极简内置文案 i18n，仅覆盖 pipeline 生成的文本 / Minimal inline i18n for pipeline text."""
    ZH = {
        "hdr_demo":  "临床建议（演示）",
        "hdr_draft": "临床建议（草案）",
        "q": "问题",
        "drug": "药品",
        "evidence": "证据（选摘）",
        "none_hits": "暂无证据片段。",
        "note": "合规提示：本工具仅供临床决策参考，不代替医生诊断与处方。",
        "demo_explain_1": "这是演示回退（检索后端在当前环境不可用）。",
        "demo_explain_2": "要在 Streamlit Cloud 启用完整检索：",
        "demo_step_1": "1) 安装 `pysqlite3-binary` 并在 retriever.py 中别名为 `sqlite3`；",
        "demo_step_2": "2) 准备/挂载 Chroma 索引与集合（或构建一个小型演示集）；",
        "demo_step_3": "3) 在 retriever 函数内部惰性导入 chromadb，避免导入期失败。",
    }
    EN = {
        "hdr_demo":  "Clinical Advice (Demo)",
        "hdr_draft": "Clinical Advice (Draft)",
        "q": "Question",
        "drug": "Drug",
        "evidence": "Rationale / Evidence (selected)",
        "none_hits": "No evidence snippets available.",
        "note": "Compliance note: for clinical reference only; not a substitute for diagnosis/prescription.",
        "demo_explain_1": "This is a demo fallback because the retrieval backend isn't available in this environment.",
        "demo_explain_2": "To enable full retrieval on Streamlit Cloud:",
        "demo_step_1": "1) Install `pysqlite3-binary` and alias it to `sqlite3` in retriever.py;",
        "demo_step_2": "2) Ensure your Chroma index path & collection exist (or build a tiny demo set);",
        "demo_step_3": "3) Lazy-import chromadb inside retriever functions to avoid import-time failures.",
    }
    return (ZH if lang == "zh" else EN).get(key, key)

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def answer(
    question: str,
    drug_name: Optional[str] = None,
    k: int = 4,
    lang: str = "zh"
) -> Dict[str, Any]:
    """
    主入口：负责调用检索、拼装草案与合规提示，并返回结构化结果。
    Main entry: orchestrates retrieval, assembles a draft + compliance note, returns a dict.
    """
    kk = _clamp_k(k)

    try:
        # 1) 指南检索（由 retriever 处理 Chroma & sqlite 兼容）/ Guideline search
        hits: List[Dict[str, Any]] = R.search_guidelines(question, k=kk) or []

        # 2) 可选：结构化药品信息（SQLite）/ Optional structured drug info
        drug_struct = None
        if drug_name and drug_name.strip():
            try:
                drug_struct = R.search_drug_structured(drug_name.strip())
            except Exception:
                drug_struct = None  # 不因药品库出错而失败 / don't fail the whole pipeline

        # 3)（预留）可在此调用 LLM；现用“草案”模板输出 / Hook for LLM – we render a localized draft
        lines: List[str] = []
        lines.append(f"**{_i18n(lang, 'hdr_draft')}**")
        lines.append("")
        lines.append(f"- **{_i18n(lang, 'q')}:** {question}")
        if drug_name:
            lines.append(f"- **{_i18n(lang, 'drug')}:** {drug_name}")
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

        return AnswerBundle(
            output=_render_with_citations("\n".join(lines)),
            guideline_hits=hits,
            drug=drug_struct,
        ).__dict__

    except Exception:
        # DEMO 回退：在 Cloud 无后端依赖时，保持 UI 可用 / DEMO fallback to keep UI usable
        if DEMO:
            traceback.print_exc()
            hits = [{
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
            return AnswerBundle(output="\n".join(lines), guideline_hits=hits, drug=None).__dict__

        # 非演示模式则抛出，让日志显示真实错误 / Re-raise in non-demo mode
        raise
