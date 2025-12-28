# -*- coding: utf-8 -*-
"""
rag/pipeline.py
----------------
Orchestrates retrieval + (optional) reasoning between the Streamlit UI (app.py)
and the backend retriever (rag/retriever.py).

Design goals
- Keep imports light so Cloud can boot even if Chroma/SQLite are absent.
- Provide a DEMO fallback to render UI even when the retrieval backend is unavailable.
- Accept `lang` ("zh" or "en") so generated text matches the UI language.
- If OPENAI_API_KEY is present, prefer OpenAI for a “final” answer; fall back to a “draft” answer on failure.

Public API (called from app.py):
    answer(question: str, drug_name: Optional[str], k: int = 4, lang: str = "zh")
        -> Dict[str, Any] with keys: output, guideline_hits, drug
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import os
import traceback

# Secrets-aware env reader (Secrets > env > default)
def _env(key: str, default: str | None = None) -> str | None:
    import os
    try:
        import streamlit as st
        return os.getenv(key, st.secrets.get(key, default))  # Secrets override default
    except Exception:
        return os.getenv(key, default)

# Defer heavy work to retriever (which lazy-imports chroma & patches sqlite)
from . import retriever as R
from . import prompt as prompt_en
from . import prompt_cn as prompt_zh

# -----------------------------------------------------------------------------
# Config flags (overridable via Secrets)
# -----------------------------------------------------------------------------
DEMO: bool = (_env("CAREMIND_DEMO", "1") == "1")   # Cloud default: demo mode ON
MAX_K: int = int(_env("CAREMIND_MAX_K", "8"))

# OpenAI config (model overridable via CAREMIND_OPENAI_MODEL)
OPENAI_MODEL_DEFAULT = _env("CAREMIND_OPENAI_MODEL", "gpt-4o-mini")

# -----------------------------------------------------------------------------
# Bundle returned to app.py
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
    """Clamp Top-K to a safe range."""
    try:
        k = int(k)
    except Exception:
        k = 4
    return max(1, min(MAX_K, k))

def _render_with_citations(raw_text: str) -> str:
    """
    Citation post-processor hook (no-op for now).
    """
    return raw_text or ""

def _compose_user_prompt(question: str, drug_name: Optional[str], hits: List[Dict[str, Any]], lang: str) -> str:
    """
    Compose the end-user prompt by inserting selected evidence into USER_TEMPLATE.
    """
    lines = []
    for i, h in enumerate(hits or [], 1):
        m = h.get("meta") or {}
        title  = str(m.get("title")  or m.get("doc_title") or m.get("section_title") or "Untitled")
        source = str(m.get("source") or m.get("source_filename") or "Unknown")
        year   = str(m.get("year")   or "—")
        content = str(h.get("content") or "")
        # Represent each evidence item as a Markdown section for citation.
        if lang == "zh":
            lines.append(f"### [{i}] {title}\n- 来源: {source} · 年份: {year}\n\n{content}\n")
        else:
            lines.append(f"### [{i}] {title}\n- Source: {source} · Year: {year}\n\n{content}\n")

    evidence_md = "\n".join(lines)
    templates = prompt_zh if lang == "zh" else prompt_en
    # USER_TEMPLATE should include {question}/{drug}/{evidence_md} placeholders.
    return templates.USER_TEMPLATE.format(question=question, drug=(drug_name or ""), evidence_md=evidence_md)

def _i18n(lang: str, key: str) -> str:
    """Minimal inline i18n for pipeline-generated text."""
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

# --- OpenAI helpers ---
def _load_dotenv_if_present() -> None:
    """If a .env file exists, attempt to load it."""
    try:
        from dotenv import load_dotenv
        load_dotenv()  # best-effort; optional dependency
    except Exception:
        pass

def _openai_available() -> bool:
    """Only checks whether OPENAI_API_KEY is present; API errors are handled at call time."""
    _load_dotenv_if_present()
    return bool(os.getenv("OPENAI_API_KEY"))

def _openai_chat(system_prompt: str, user_prompt: str, model: Optional[str] = None) -> str:
    """
    Thin wrapper around the OpenAI Chat call.
    - Reads OPENAI_API_KEY from environment (and related OpenAI env vars)
    - Default model: gpt-4o-mini (overridable via CAREMIND_OPENAI_MODEL)
    """
    _load_dotenv_if_present()
    from openai import OpenAI  # official 1.x client
    client = OpenAI()  # reads OPENAI_API_KEY / OPENAI_BASE_URL (if set)
    mdl = model or OPENAI_MODEL_DEFAULT or "gpt-4o-mini"

    resp = client.chat.completions.create(
        model=mdl,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
    )
    # Take the first candidate
    return (resp.choices[0].message.content or "").strip()

def _first_sentences(txt: str, n_sent: int = 1, limit: int = 180) -> str:
    """Simple sentence-based summarizer used in draft mode."""
    import re
    sents = re.split(r"(?:。|！|？|\.)", (txt or "").strip())
    pick = "。".join([s for s in sents if s][:n_sent]).strip("。")
    pick = pick[:limit] + ("…" if len(pick) > limit else "")
    return pick

def _compact_evidence_list(lang: str, hits: List[Dict[str, Any]]) -> str:
    """Render a compact evidence list mapping [i] -> title/source/year."""
    hdr = "证据清单：" if lang == "zh" else "Evidence List:"
    if not hits:
        return hdr + "\n" + ("（暂无证据片段）" if lang == "zh" else "(No evidence snippets available.)")

    lines: List[str] = [hdr]
    for i, h in enumerate(hits or [], 1):
        m = h.get("meta") or {}
        title  = str(m.get("title") or m.get("doc_title") or m.get("section_title") or ("无标题" if lang == "zh" else "Untitled"))
        source = str(m.get("source") or m.get("source_filename") or ("未知来源" if lang == "zh" else "Unknown"))
        year   = str(m.get("year") or "").strip()
        tail = (f"（{year}）" if lang == "zh" else f"({year})") if year else ""
        lines.append(f"[{i}] {title} — {source} {tail}".rstrip())
    return "\n".join(lines).strip()

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
    Main entry: retrieval → (prefer) OpenAI reasoning → fallback to draft; returns a dict.
    """
    kk = _clamp_k(k)

    # Expose why we might fall back to draft mode (used by Streamlit UI).
    _load_dotenv_if_present()
    openai_key_present: bool = bool(os.getenv("OPENAI_API_KEY"))
    openai_attempted: bool = False
    openai_error_type: Optional[str] = None
    openai_reason: str = "missing_key" if not openai_key_present else ""

    try:
        # 1) Guideline retrieval (retriever handles Chroma & sqlite compatibility)
        hits: List[Dict[str, Any]] = R.search_guidelines(question, k=kk) or []

        if openai_key_present and not hits:
            openai_reason = "no_hits"

        # 2) Optional structured drug info (SQLite)
        drug_struct = None
        if drug_name and drug_name.strip():
            try:
                drug_struct = R.search_drug_structured(drug_name.strip())
            except Exception:
                drug_struct = None  # don't fail the whole pipeline due to drug DB errors

        # 3) Prefer OpenAI for a "final" answer
        #    - System prompt matches UI language
        #    - User prompt includes evidence as Markdown sections; require in-text citations like [1][2][3]
        if openai_key_present and hits:
            try:
                openai_attempted = True
                templates = prompt_zh if lang == "zh" else prompt_en
                user_prompt = _compose_user_prompt(question, drug_name, hits, lang=lang)

                # Add a clear formatting/citation instruction to ensure numbered citations like [1][2][3].
                if lang == "zh":
                    citation_hint = (
                        "\n\n请在结论/建议段落中使用形如 [1][2][3] 的文内引用，"
                        "编号与上方证据小节的 [1]/[2]/[3] 对应；不要产生不存在的编号。"
                        "请输出 Markdown。"
                    )
                else:
                    citation_hint = (
                        "\n\nIn your conclusion/recommendation paragraphs, include in-text citations "
                        "like [1][2][3], where the numbers refer to the evidence sections ([1], [2], [3]) above. "
                        "Do not invent citations. Output Markdown."
                    )

                final_user = user_prompt + citation_hint
                sys_prompt = templates.SYSTEM

                llm_out = _openai_chat(system_prompt=sys_prompt, user_prompt=final_user)

                if llm_out and llm_out.strip():
                    # OpenAI path succeeded: return the model output directly.
                    out = AnswerBundle(
                        output=_render_with_citations(llm_out),
                        guideline_hits=hits,
                        drug=drug_struct,
                    ).__dict__
                    out.update(
                        {
                            "mode": "llm",
                            "openai_key_present": openai_key_present,
                            "openai_attempted": openai_attempted,
                            "openai_error_type": openai_error_type,
                            "openai_reason": "",
                            "openai_model": (OPENAI_MODEL_DEFAULT or ""),
                        }
                    )
                    return out
            except Exception as e:
                # On OpenAI error, fall back to draft mode; Streamlit UI can surface details.
                openai_attempted = True
                openai_error_type = type(e).__name__
                openai_reason = "openai_error"
                traceback.print_exc()

        # 4) Fallback: generate a minimal draft answer (keeps UI non-empty)
        ev_list = _compact_evidence_list(lang, hits)
        # Natural-language draft (still conservative): keeps UI readable even without OpenAI.
        if lang == "zh":
            p1 = (
                "建议：\n"
                "结论：对合并支气管哮喘的高血压患者，β 受体阻滞剂通常需要谨慎评估；"
                "若仅为降压目的，一般优先选择对气道影响更小的替代降压方案。\n\n"
                "若存在必须使用 β 受体阻滞剂的明确心血管指征（例如心衰/心梗后/部分心律失常），"
                "临床上通常倾向选择 β1 选择性药物、从小剂量开始，并在哮喘稳定期严密监测呼吸症状与峰流速；"
                "若出现喘息加重或急救支气管舒张剂反应变差，应及时复评并调整方案。\n"
            )
            p2 = (
                "说明：以上为在未调用大模型生成‘正式建议’时的保守草案，"
                "可结合下方证据清单 [1][2]… 与患者具体情况由临床医生综合判断。\n"
            )
            lines = [
                f"问题：{question}",
            ]
            if drug_name:
                lines.append(f"药品：{drug_name}")
            lines += ["", p1, p2, "", ev_list, "", f"{_i18n(lang, 'note')}" ]
        else:
            p1 = (
                "Advice:\n"
                "Bottom line: In patients with asthma and hypertension, beta-blockers often require careful risk–benefit assessment; "
                "if the goal is blood-pressure control alone, alternatives with less airway risk are typically preferred.\n\n"
                "If there is a compelling cardiovascular indication (e.g., heart failure, post-MI, certain arrhythmias), clinicians often favor "
                "a cardioselective (beta-1 selective) agent at the lowest effective dose with close monitoring for bronchospasm and rescue-inhaler response; "
                "worsening wheeze or reduced bronchodilator effect should prompt reassessment.\n"
            )
            p2 = (
                "Note: This is a conservative draft fallback when a full LLM-generated response is not available; "
                "please interpret together with the Evidence List [1][2]… and clinical context.\n"
            )
            lines = [
                f"Question: {question}",
            ]
            if drug_name:
                lines.append(f"Drug: {drug_name}")
            lines += ["", p1, p2, "", ev_list, "", f"{_i18n(lang, 'note')}" ]

        out = AnswerBundle(
            output=_render_with_citations("\n".join(lines)),
            guideline_hits=hits,
            drug=drug_struct,
        ).__dict__
        out.update(
            {
                "mode": "draft",
                "openai_key_present": openai_key_present,
                "openai_attempted": openai_attempted,
                "openai_error_type": openai_error_type,
                "openai_reason": openai_reason,
                "openai_model": (OPENAI_MODEL_DEFAULT or ""),
            }
        )
        return out

    except Exception:
        # DEMO fallback: keep UI usable when backend dependencies are unavailable on Cloud
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
            out = AnswerBundle(output="\n".join(lines), guideline_hits=hits, drug=None).__dict__
            out.update(
                {
                    "mode": "demo",
                    "openai_key_present": openai_key_present,
                    "openai_attempted": False,
                    "openai_error_type": None,
                    "openai_reason": "demo_backend_error",
                    "openai_model": (OPENAI_MODEL_DEFAULT or ""),
                }
            )
            return out

        # Re-raise in non-demo mode so the real error is visible.
        raise