# -*- coding: utf-8 -*-
"""
CareMind · MVP CDSS (Streamlit, bilingual zh/en)
"""

from __future__ import annotations

import json, re, time
from typing import Any, Dict, List, Optional

import streamlit as st
from rag.pipeline import answer  # backend entry

# ---------------------------
# i18n
# ---------------------------
I18N: Dict[str, Dict[str, str]] = {
    "zh": {
        "title": "CareMind · 临床决策支持（MVP）",
        "question_label": "输入临床问题",
        "question_ph": "例如：慢性肾病（CKD）患者使用 ACEI/ARB 时如何监测？多久复查？",
        "drug_label": "（可选）指定药品名（如：阿司匹林）",
        "submit": "生成建议",
        "tab_advice": "🧭 建议",
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
    },
    "en": {
        "title": "CareMind · Clinical Decision Support (MVP)",
        "question_label": "Enter your clinical question",
        "question_ph": "e.g., For CKD patients on ACEI/ARB, how to monitor and how often?",
        "drug_label": "(Optional) Drug name (e.g., Aspirin)",
        "submit": "Generate Advice",
        "tab_advice": "🧭 Advice",
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
        "preset2": "Elderly with T2DM+CAD: BP target & therapy",
        "preset3": "GDM: Initiating insulin",
        "advice_hdr": "Advice (with citations & compliance note)",
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
    },
}

def t(lang: str, key: str) -> str:
    return I18N.get(lang, I18N["zh"]).get(key, key)

# ---------------------------
# Page config + CSS
# ---------------------------
st.set_page_config(page_title="CareMind · MVP CDSS", layout="wide", page_icon="💊")

CSS = """
<style>
.cm-badge{display:inline-block;padding:2px 8px;border-radius:12px;font-size:12px;background:#eef2ff;border:1px solid #c7d2fe;margin-right:6px;white-space:nowrap;}
.cm-chip{display:inline-block;padding:2px 8px;border-radius:8px;font-size:12px;background:#f1f5f9;border:1px solid #e2e8f0;margin:0 6px 6px 0;}
.cm-muted{color:#64748b;font-size:13px;}
.cm-output{line-height:1.75;font-size:17px;}
.cm-card{border:1px solid #e5e7eb;background:#fff;border-radius:12px;padding:12px 14px;margin-bottom:10px;}
footer{visibility:hidden;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    lang = st.selectbox("Language / 语言", options=["zh", "en"], index=0, format_func=lambda x: "中文" if x=="zh" else "English")
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
            t("en","preset3"): "GDM: when to start insulin and starting dose?",
        }
    }
    preset_none = t(lang, "preset_none")
    preset_choice = st.selectbox(t(lang, "preset_select"), options=[preset_none] + list(presets[lang].keys()), index=0)
    st.caption(t(lang, "page_footer"))

# ---------------------------
# Input area
# ---------------------------
st.title(t(lang, "title"))

with st.form("cm_query"):
    q_init = presets[lang].get(preset_choice, "") if preset_choice != preset_none else ""
    q = st.text_input(t(lang, "question_label"), placeholder=t(lang, "question_ph"), value=q_init)
    drug = st.text_input(t(lang, "drug_label"), value="")
    submitted = st.form_submit_button(t(lang, "submit"), use_container_width=True)

# ---------------------------
# Helpers
# ---------------------------
def link_citations(md: str) -> str:
    return re.sub(r"\[(?:#)?(\d+)\]", r"[\1](#hit-\1)", md or "")

def evidence_md(lang: str, hits: List[Dict[str, Any]]) -> str:
    lines = []
    for i, h in enumerate(hits or [], 1):
        m = h.get("meta") or {}
        title = str(m.get("title") or ( "无标题" if lang=="zh" else "Untitled"))
        source = str(m.get("source") or ("未知" if lang=="zh" else "Unknown"))
        year = str(m.get("year") or "—")
        if lang == "zh":
            lines.append(f"### #{i} {title}\n\n- 来源：{source} · 年份：{year}\n\n{h.get('content','')}\n")
        else:
            lines.append(f"### #{i} {title}\n\n- Source: {source} · Year: {year}\n\n{h.get('content','')}\n")
    return "\n".join(lines)

def friendly_hints(lang: str, exc: Exception) -> List[str]:
    msg = str(exc).lower()
    zh = (lang == "zh")
    tips = []
    if "chromadb" in msg:
        tips.append("· 检查 CHROMA_PERSIST_DIR / CHROMA_COLLECTION" if zh else "· Check CHROMA_PERSIST_DIR / CHROMA_COLLECTION")
    if "sqlite" in msg:
        tips.append("· 检查 SQLite 路径与表结构" if zh else "· Verify SQLite path & schema")
    if "cuda" in msg or "cudnn" in msg:
        tips.append("· 检查 CUDA/cuDNN 或切到 CPU" if zh else "· Check CUDA/cuDNN or switch to CPU")
    if "module" in msg and "not found" in msg:
        tips.append("· 确认 rag/__init__.py 与导入路径" if zh else "· Ensure rag/__init__.py and import path")
    return tips

# ---------------------------
# Tabs
# ---------------------------
tab_adv, tab_hits, tab_drug, tab_log = st.tabs([
    t(lang, "tab_advice"),
    t(lang, "tab_hits"),
    t(lang, "tab_drug"),
    t(lang, "tab_log"),
])

res: Optional[Dict[str, Any]] = None
elapsed = None

# ---------------------------
# Invoke backend
# ---------------------------
if submitted:
    if not (q and q.strip()):
        st.warning(t(lang, "warn_need_q"))
    else:
        with st.spinner("..."):
            try:
                t0 = time.time()
                res = answer(q.strip(), drug_name=(drug.strip() or None), k=int(k))
                elapsed = time.time() - t0
            except Exception as e:
                st.error(t(lang, "err_backend"))
                tips = friendly_hints(lang, e)
                if tips:
                    st.info("· " + "\n· ".join(tips))
                st.exception(e)
                res = None

# ---------------------------
# Render
# ---------------------------
if res:
    # Advice
    with tab_adv:
        st.subheader(t(lang, "advice_hdr"))
        output_text = link_citations(res.get("output") or "")
        st.markdown(f"<div class='cm-output'>{output_text}</div>", unsafe_allow_html=True)
        if elapsed is not None:
            st.caption(t(lang, "time_used").format(elapsed))
        c1, c2 = st.columns(2)
        with c1:
            st.code(output_text, language="markdown")
        with c2:
            st.download_button(
                t(lang, "export_advice"),
                data=(output_text or "").encode("utf-8"),
                file_name="caremind_advice.md",
                mime="text/markdown",
                use_container_width=True,
            )
            st.download_button(
                t(lang, "export_evidence"),
                data=evidence_md(lang, res.get("guideline_hits") or []).encode("utf-8"),
                file_name="caremind_evidence.md",
                mime="text/markdown",
                use_container_width=True,
            )
        st.caption(t(lang, "disclaimer"))

    # Evidence
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
            # top source chips
            sources: Dict[str, int] = {}
            for h in hits:
                m = h.get("meta") or {}
                s = str(m.get("source") or ("未知来源" if lang=="zh" else "Unknown")).strip()
                sources[s] = sources.get(s, 0) + 1
            st.markdown(" ".join([f"<span class='cm-chip'>{s} × {n}</span>" for s, n in sources.items()]), unsafe_allow_html=True)

            for i, h in enumerate(hits, 1):
                m = h.get("meta") or {}
                title = str(m.get("title") or ("无标题" if lang=="zh" else "Untitled"))
                source = str(m.get("source") or ("未知" if lang=="zh" else "Unknown"))
                year = str(m.get("year") or "—")
                doc_id = str(m.get("id") or "—")
                label = f"#{i} · {title[:60]}"
                st.markdown(f"<a id='hit-{i}'></a>", unsafe_allow_html=True)
                with st.expander(label, expanded=expand_hits):
                    if show_meta:
                        st.markdown(
                            f"<div class='cm-muted'>"
                            f"<span class='cm-badge'>{t(lang, 'chips_src')} {source}</span>"
                            f"<span class='cm-badge'>{t(lang, 'chips_year')} {year}</span>"
                            f"<span class='cm-badge'>{t(lang, 'chips_id')} {doc_id}</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                    st.markdown(h.get("content") or ("（空片段）" if lang=="zh" else "(empty)"))

    # Drug
    with tab_drug:
        st.subheader(t(lang, "drug_hdr"))
        if res.get("drug"):
            st.json(res["drug"], expanded=False)
        else:
            st.caption(t(lang, "no_drug"))

    # Logs
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

# Footer
st.caption(t(lang, "page_footer"))
