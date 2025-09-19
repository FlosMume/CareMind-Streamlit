# -*- coding: utf-8 -*-
"""
rag/pipeline.py
----------------
Thin orchestration layer between the Streamlit UI (app.py) and backend retrieval.
- Keeps imports light so the app can boot on Streamlit Cloud even if Chroma/SQLite aren't available
- Provides a DEMO mode fallback (no hard crash; returns stubbed response)
- Centralizes prompt composition / output formatting
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import os
import traceback

# Import our local modules only (these should NOT import heavy libs at module top-level)
# retriever.py should lazy-import chromadb and alias pysqlite3->sqlite3 as needed.
from . import retriever as R
from .prompt import SYSTEM, USER_TEMPLATE  # keep your existing prompt strings

# -----------------------------------------------------------------------------
# Config & Flags
# -----------------------------------------------------------------------------
DEMO: bool = os.getenv("CAREMIND_DEMO", "1") == "1"  # default demo ON for Cloud
MAX_K: int = int(os.getenv("CAREMIND_MAX_K", "8"))

# Optional envs used by retriever; not required here but useful for logs
CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")
CHROMA_COLL = os.getenv("CHROMA_COLLECTION", "guideline_chunks")

# -----------------------------------------------------------------------------
# Data model for the final answer returned to app.py
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


def _render_with_citations(raw_text: str, hits: List[Dict[str, Any]]) -> str:
    """
    Optionally post-process the LLM text to ensure [#1], [#2] style citations map to hits.
    Here we just return raw_text; keep hook for future formatting.
    """
    return raw_text or ""


def _compose_user_prompt(question: str, drug_name: Optional[str], hits: List[Dict[str, Any]]) -> str:
    """
    Fills USER_TEMPLATE with question, optional drug, and selected evidence.
    Expect USER_TEMPLATE to have placeholders like {question}, {drug}, {evidence_md}.
    """
    lines = []
    for i, h in enumerate(hits, 1):
        m = h.get("meta") or {}
        title = str(m.get("title") or "Untitled")
        source = str(m.get("source") or "Unknown")
        year = str(m.get("year") or "—")
        content = str(h.get("content") or "")
        lines.append(f"### #{i} {title}\n- Source: {source} · Year: {year}\n\n{content}\n")

    evidence_md = "\n".join(lines)
    return USER_TEMPLATE.format(question=question, drug=(drug_name or ""), evidence_md=evidence_md)


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def answer(question: str, drug_name: Optional[str] = None, k: int = 4) -> Dict[str, Any]:
    """
    Orchestrates retrieval + (optional) LLM reasoning.
    Returns a dict with keys: output (str), guideline_hits (list[dict]), drug (dict|None)

    This function is intentionally resilient:
    - If Chroma/SQLite are unavailable (e.g., Streamlit Cloud old sqlite), we will:
        * In DEMO mode: return a helpful stub result (no crash)
        * Otherwise: re-raise the exception for visibility
    """
    kk = _clamp_k(k)

    try:
        # 1) Retrieve guideline snippets (Chroma/whatever retriever uses internally)
        hits: List[Dict[str, Any]] = R.search_guidelines(question, k=kk) or []

        # 2) Optional: retrieve structured drug info (SQLite, etc.)
        drug_struct = None
        if drug_name:
            try:
                drug_struct = R.search_drug_structured(drug_name.strip())
            except Exception:
                # Structured drug DB might be optional; don't fail the whole pipeline
                drug_struct = None

        # 3) Compose prompt for your reasoning model (LLM call could be here)
        user_prompt = _compose_user_prompt(question, drug_name, hits)

        # --- If you have an LLM, call it here. For now, we create a concise, cited draft. ---
        # Replace the below with your actual LLM call if desired.
        draft = []
        draft.append("**Clinical Advice (Draft)**")
        draft.append("")
        draft.append(f"- **Question:** {question}")
        if drug_name:
            draft.append(f"- **Drug:** {drug_name}")
        draft.append("")
        draft.append("**Rationale / Evidence (selected):**")
        if not hits:
            draft.append("- No evidence snippets available.")
        else:
            for i, h in enumerate(hits, 1):
                m = h.get("meta") or {}
                title = str(m.get("title") or "Untitled")
                src = str(m.get("source") or "Unknown")
                year = str(m.get("year") or "—")
                draft.append(f"- [#{i}] {title} ({src}, {year})")
        draft.append("")
        draft.append("_Compliance note: for clinical reference only; not a substitute for diagnosis/prescription._")

        output_text = _render_with_citations("\n".join(draft), hits)

        return AnswerBundle(
            output=output_text,
            guideline_hits=hits,
            drug=drug_struct,
        ).__dict__

    except Exception as e:
        # If running on Cloud and Chroma/SQLite is unavailable, offer a DEMO fallback.
        if DEMO:
            # Log a concise traceback to help debugging in Cloud logs
            traceback.print_exc()

            stub_hits: List[Dict[str, Any]] = []
            if question.strip():
                # Provide a tiny, fake snippet so the UI still demonstrates the flow
                stub_hits = [{
                    "content": "Demo mode: retrieval disabled. Provide small bundled data or connect a remote DB.",
                    "meta": {"title": "Demo Stub", "source": "Demo", "year": "—", "id": "demo-0001"},
                }]

            draft = [
                "**Clinical Advice (Demo)**",
                "",
                f"- **Question:** {question}",
                (f"- **Drug:** {drug_name}" if drug_name else ""),
                "",
                "This is a demo fallback because the retrieval backend isn't available in this environment.",
                "To enable full retrieval on Streamlit Cloud:",
                "1) Install `pysqlite3-binary` and alias it to `sqlite3` in retriever.py",
                "2) Ensure your Chroma index path & collection exist (or build a tiny demo set)",
                "3) Consider lazy-importing chromadb inside retriever functions",
                "",
                "_Compliance note: for clinical reference only._",
            ]
            return AnswerBundle(
                output="\n".join([line for line in draft if line != ""]),
                guideline_hits=stub_hits,
                drug=None,
            ).__dict__

        # In non-demo mode, propagate the real error to surface it in logs/UI
        raise
