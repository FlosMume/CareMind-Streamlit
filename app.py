# -*- coding: utf-8 -*-
"""
CareMind · MVP CDSS · 前端 (Streamlit)
依赖：
  - streamlit>=1.32
项目约定：
  - 后端推理入口：rag.pipeline.answer(question: str, drug_name: Optional[str], k: int) -> dict
返回字典示例：
  {
    "output": "...模型生成建议（带引用与合规说明）...",
    "guideline_hits": [
        {"content": "...片段...", "meta": {"source": "...", "year": 2024, "title": "...", "id": "..."}},
        ...
    ],
    "drug": {"药品名称": "...", "适应症": "...", ...}  # 若有，则为结构化信息
  }
CareMind · MVP CDSS · Bilingual Frontend (Streamlit)： 
-  Clinical Decision Support System (CDSS) - Minimal Viable Product (MVP)
Language switch: 中文 / English (sidebar)
Backend API: from rag.pipeline import answer(question, drug_name=None, k:int) -> dict

"""

from __future__ import annotations

import json

from typing import Any, Dict, List, Optional

import streamlit as st
from pipeline import answer  # 你已实现的后端入口

# ---------------------------
# 基础页面配置 & 轻量样式
# ---------------------------
st.set_page_config(
    page_title="CareMind · MVP CDSS",
    layout="wide",
    page_icon="💊",
)

# 轻量 CSS：更紧凑的卡片风格 & 引用徽章
st.markdown(
    """
    <style>
    .cm-badge {
        display:inline-block;
        padding:2px 8px;
        border-radius:12px;
        font-size:12px;
        background:#eef2ff;
        border:1px solid #c7d2fe;
        margin-right:6px;
        white-space:nowrap;
    }
    .cm-chip {
        display:inline-block;
        padding:2px 8px;
        border-radius:8px;
        font-size:12px;
        background:#f1f5f9;
        border:1px solid #e2e8f0;
        margin:0 6px 6px 0;
    }
    .cm-card {
        border:1px solid #e5e7eb;
        background:#ffffff;
        border-radius:12px;
        padding:12px 14px;
        margin-bottom:10px;
    }
    .cm-muted {
        color:#64748b;
        font-size:13px;
    }
    .cm-output {
        line-height:1.6;
        font-size:16px;
    }
    footer {visibility: hidden;}

import re
import time
from typing import Any, Dict, List, Optional

import streamlit as st
from rag.pipeline import answer  # 后端推理入口（已实现）

# ---------------------------
# i18n: 文案字典
# ---------------------------
I18N: Dict[str, Dict[str, str]] = {
    "zh": {
        "title": "CareMind · 临床决策支持（MVP）",
        "question_label": "输入临床问题",
        "question_ph": "例如：慢性肾病患者使用ACEI注意什么？",
        "drug_label": "（可选）指定药品名，用于结构化比对（如：阿司匹林）",
        "submit": "生成建议",
        "tab_advice": "🧭 建议",
        "tab_hits": "📚 证据片段",
        "tab_drug": "💊 药品结构化",
        "tab_log": "🪵 运行日志",
        "settings": "⚙️ 设置",
        "k_slider": "检索片段数（Top-K）",
        "show_meta": "显示片段元数据行",
        "expand_hits": "展开所有片段",
        "filters": "🧩 证据筛选（前端）",
        "filter_src": "按来源包含过滤（可留空）",
        "filter_year": "年份范围",
        "presets": "🧪 问题模板",
        "preset_select": "快速选择",
        "preset_none": "——",
        "preset1": "CKD 合并高血压用药监测",
        "preset2": "老年多病共存降压策略",
        "preset3": "妊娠期糖尿病胰岛素起始",
        "advice_hdr": "建议（含引用与合规声明）",
        "time_used": "⏱️ 用时：{:.2f}s",
        "export_advice": "导出建议（Markdown）",
        "export_evidence": "导出证据（Markdown）",
        "disclaimer": "⚠️ 本工具仅供临床决策参考，不替代医师诊断与处方。",
        "hits_hdr": "检索片段（Top-{k}，已过滤后 {n} 条）",
        "no_hits": "未检索到符合筛选条件的片段。",
        "drug_hdr": "药品结构化信息（SQLite）",
        "no_drug": "未提供或未检索到对应药品的结构化信息。",
        "log_export": "导出本会话全部日志（JSON）",
        "history_hdr": "🗂️ 本会话历史（点击复用）",
        "no_history": "暂无历史记录。",
        "reuse": "复用",
        "reused_tip": "已复用历史查询：{q}（药品：{drug}，K={k}）。请根据需要编辑后再次点击【生成建议】。",
        "warn_need_q": "请输入临床问题。",
        "err_backend": "后端推理出现异常。",
        "hint_title": "排查建议：",
        "page_footer": "© CareMind · MVP CDSS | 本工具仅供临床决策参考，不替代医师诊断与处方。",
        "version_tip": "版本：MVP • 本工具仅供临床决策参考，不替代医师诊断与处方。",
        "chips_src": "来源：",
        "chips_year": "年份：",
        "chips_id": "ID：",
        "stats_hits": "片段数：{n} · 总字数：{c}",
    },
    "en": {
        "title": "CareMind · Clinical Decision Support (MVP)",
        "question_label": "Enter your clinical question",
        "question_ph": "e.g., What should be monitored when CKD patients are on ACEIs?",
        "drug_label": "(Optional) Specify a drug for structured comparison (e.g., Aspirin)",
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
        "preset2": "Elderly w/ T2DM + CAD: BP target & therapy",
        "preset3": "GDM: Initiating insulin (indication & dose)",
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
        "reused_tip": "Reused: {q} (Drug: {drug}, K={k}). Edit if needed then click [Generate Advice] again.",
        "warn_need_q": "Please enter a clinical question.",
        "err_backend": "Backend inference error.",
        "hint_title": "Troubleshooting hints:",
        "page_footer": "© CareMind · MVP CDSS | For clinical reference only.",
        "version_tip": "Version: MVP • For clinical reference only.",
        "chips_src": "Source:",
        "chips_year": "Year:",
        "chips_id": "ID:",
        "stats_hits": "Snippets: {n} · Total chars: {c}",
    },
}

PRESETS = {
    "zh": {
        "CKD 合并高血压用药监测": "慢性肾病（CKD）患者使用 ACEI/ARB 时如何监测血肌酐与血钾？需要多久复查？",
        "老年多病共存降压策略": "老年人合并糖尿病与冠心病的降压目标与首选药物方案是什么？",
        "妊娠期糖尿病胰岛素起始": "妊娠期糖尿病控制不佳时胰岛素起始指征、起始剂量与监测策略是什么？",
    },
    "en": {
        "Monitoring ACEI/ARB in CKD + Hypertension": "For CKD patients on ACEI/ARB, how to monitor serum creatinine and potassium, and how often to recheck?",
        "Elderly w/ T2DM + CAD: BP target & therapy": "For elderly with T2DM and CAD, what BP target and first-line antihypertensives are recommended?",
        "GDM: Initiating insulin (indication & dose)": "For gestational diabetes with poor control, when to start insulin and how to set starting dose and monitoring?",
    },
}

# ---------------------------
# UI 配置 & 轻量样式
# ---------------------------
st.set_page_config(page_title="CareMind · MVP CDSS", layout="wide", page_icon="💊")
# type: ignore
CSS = """
<style>
.cm-badge{display:inline-block;padding:2px 8px;border-radius:12px;font-size:12px;background:#eef2ff;border:1px solid #c7d2fe;margin-right:6px;white-space:nowrap;}
.cm-chip{display:inline-block;padding:2px 8px;border-radius:8px;font-size:12px;background:#f1f5f9;border:1px solid #e2e8f0;margin:0 6px 6px 0;}
.cm-muted{color:#64748b;font-size:13px;}
.cm-output{line-height:1.75;font-size:17px;}
footer{visibility:hidden;}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------
# Sidebar：检索与显示设置
# ---------------------------
with st.sidebar:
    st.header("⚙️ 设置")
    k = st.slider("检索片段数（Top-K）", min_value=2, max_value=8, value=4, step=1)
    show_meta = st.toggle("显示片段元数据行", value=True)
    expand_hits = st.toggle("展开所有片段", value=False)
    st.divider()
    st.markdown(
        "#### 🧪 使用建议\n"
        "- 临床问题尽量**具体**，可包含人群限定、并发症与药名\n"
        "- 可选输入药品名，用于**结构化比对**（如：阿司匹林）\n"
    )
    st.divider()
    st.caption("版本：MVP • 本工具仅供临床决策参考，不替代医师诊断与处方。")

# ---------------------------
# Session State：简单历史记录
# ---------------------------
if "history" not in st.session_state:
    st.session_state.history: List[Dict[str, Any]] = []

# ---------------------------
# 页面主区
# ---------------------------
st.title("CareMind · 临床决策支持（MVP）")

# 两列布局：左侧输入/结果，右侧为命中与药品信息
col_left, col_right = st.columns([1.3, 1.0])

with col_left:
    # —— 输入表单
    with st.form("cm_query"):
        q = st.text_input(
            "输入临床问题",
            placeholder="例如：慢性肾病（CKD）合并高血压患者使用 ACEI/ARB 有何监测要点？",
        )
        drug = st.text_input(
            "（可选）指定药品名，用于结构化比对",
            placeholder="例如：阿司匹林 / 氨氯地平 / 依那普利...",
        )
        submitted = st.form_submit_button("生成建议", use_container_width=True)

    # —— 调用后端
    res: Optional[Dict[str, Any]] = None
    if submitted:
        if not q or not q.strip():
            st.warning("请输入临床问题。")
        else:
            with st.spinner("检索与生成中…"):
                try:
                    res = answer(q.strip(), drug_name=(drug.strip() or None), k=int(k))
                except Exception as e:
                    st.error("后端推理出现异常，请查看后端日志或稍后重试。")
                    st.exception(e)
                    res = None

    # —— 渲染输出
    if res:
        # 存历史
        st.session_state.history.insert(
            0,
            {
                "q": q.strip(),
                "drug": (drug.strip() or None),
                "k": k,
                "res": res,
            },
        )

        st.subheader("🧭 建议（含引用与合规声明）")
        output_text = res.get("output") or "（无生成内容）"
        st.markdown(f"<div class='cm-output'>{output_text}</div>", unsafe_allow_html=True)

        # 复制 & 下载
        col_btn1, col_btn2, _ = st.columns([0.25, 0.25, 0.5])
        with col_btn1:
            st.code(output_text, language="markdown")
        with col_btn2:
            download_payload = json.dumps(res, ensure_ascii=False, indent=2)
            st.download_button(
                "下载本次结果（JSON）",
                data=download_payload.encode("utf-8"),
                file_name="caremind_response.json",
                mime="application/json",
                use_container_width=True,
            )

        st.caption("⚠️ 本工具仅供临床决策参考，不替代医师诊断与处方。")

with col_right:
    # —— Top-K 命中展示
    if res:
        hits: List[Dict[str, Any]] = res.get("guideline_hits") or []
        st.subheader(f"📚 检索片段（Top-{k}）")

        if not hits:
            st.info("未检索到相关指南/共识片段。请尝试调整问题或增大 Top-K。")
        else:
            # 顶部来源统计 chips
            sources = {}
            for h in hits:
                meta = (h.get("meta") or {})
                src = str(meta.get("source") or "未知来源").strip()
                sources[src] = sources.get(src, 0) + 1
            st.markdown(
                " ".join([f"<span class='cm-chip'>{s} × {n}</span>" for s, n in sources.items()]),
                unsafe_allow_html=True,
            )

            for i, h in enumerate(hits, 1):
                meta = h.get("meta") or {}
                source = str(meta.get("source") or "")
                year = str(meta.get("year") or "")
                title = str(meta.get("title") or "")
                doc_id = str(meta.get("id") or "")

                label = f"#{i} · {title}" if title else f"#{i} · 无标题片段"

# 辅助函数
# ---------------------------
def t(lang: str, key: str) -> str:
    return I18N.get(lang, I18N["zh"]).get(key, key)

def link_citations(md: str) -> str:
    """将 [#1]/[1] 引用号变为指向 #hit-1 的锚点"""
    return re.sub(r"\[(?:#)?(\d+)\]", r"[\1](#hit-\1)", md or "")

def highlight(text: str, q: str) -> str:
    """对问题中的分词进行简单高亮（大小写不敏感）"""
    import html
    esc = html.escape(text or "")
    kws = [x for x in re.split(r"\s+", (q or "").strip()) if x]
    for kw in kws:
        esc = re.sub(re.escape(kw), f"<mark>{kw}</mark>", esc, flags=re.I)
    return esc

def dl_md_button(label: str, text: str, filename: str):
    st.download_button(label, data=(text or "").encode("utf-8"), file_name=filename, mime="text/markdown", use_container_width=True)

def evidence_md(lang: str, hits: List[Dict[str, Any]]) -> str:
    lines = []
    for i, h in enumerate(hits or [], 1):
        m = h.get("meta") or {}
        title = str(m.get("title") or "")
        source = str(m.get("source") or "")
        year = str(m.get("year") or "")
        if lang == "zh":
            lines.append(f"### #{i} {title}\n\n- 来源：{source} · 年份：{year}\n\n{h.get('content','')}\n")
        else:
            lines.append(f"### #{i} {title}\n\n- Source: {source} · Year: {year}\n\n{h.get('content','')}\n")
    return "\n".join(lines)

def friendly_hints(lang: str, exc: Exception) -> List[str]:
    msg = str(exc)
    zh = (lang == "zh")
    tips = []
    if "chromadb" in msg.lower():
        tips.append("· 检查 CHROMA_PERSIST_DIR / CHROMA_COLLECTION") if zh else tips.append("· Check CHROMA_PERSIST_DIR / CHROMA_COLLECTION")
    if "sqlite" in msg.lower():
        tips.append("· 检查 SQLite 路径与表结构") if zh else tips.append("· Verify SQLite path & schema")
    if "cuda" in msg.lower() or "cudnn" in msg.lower():
        tips.append("· 检查 CUDA/cuDNN 或切到 CPU") if zh else tips.append("· Check CUDA/cuDNN or switch to CPU")
    if "module" in msg.lower() and "not found" in msg.lower():
        tips.append("· 确认 rag/__init__.py 与导入路径") if zh else tips.append("· Ensure rag/__init__.py and import path")
    return tips

# ---------------------------
# Session State
# ---------------------------
if "history" not in st.session_state:
    st.session_state.history: List[Dict[str, Any]] = []
if "logs" not in st.session_state:
    st.session_state.logs: List[Dict[str, Any]] = []
if "lang" not in st.session_state:
    st.session_state.lang = "zh"

# ---------------------------
# Sidebar：语言/设置/筛选/模板
# ---------------------------
with st.sidebar:
    lang = st.selectbox("Language / 语言", options=["zh", "en"], index=0, format_func=lambda x: "中文" if x=="zh" else "English")
    st.session_state.lang = lang

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
    preset_label_map = {
        "zh": [t("zh", "preset_none"), t("zh", "preset1"), t("zh", "preset2"), t("zh", "preset3")],
        "en": [t("en", "preset_none"), t("en", "preset1"), t("en", "preset2"), t("en", "preset3")],
    }
    preset_choice = st.selectbox(t(lang, "preset_select"), options=preset_label_map[lang], index=0)

    st.divider()
    st.caption(t(lang, "version_tip"))

# ---------------------------
# 主区：标题 & 输入区
# ---------------------------
st.title(t(lang, "title"))

with st.form("cm_query"):
    q_init = ""
    if preset_choice != t(lang, "preset_none"):
        q_init = PRESETS[lang].get(preset_choice, "")
    q = st.text_input(t(lang, "question_label"), placeholder=t(lang, "question_ph"), value=q_init)
    drug = st.text_input(t(lang, "drug_label"), value="")
    submitted = st.form_submit_button(t(lang, "submit"), use_container_width=True)

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
# 调用后端
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
                    st.info(t(lang, "hint_title") + "\n" + "\n".join(tips))
                st.exception(e)
                res = None

# ---------------------------
# 渲染结果
# ---------------------------
if res:
    st.session_state.history.insert(0, {"q": q.strip(), "drug": (drug.strip() or None), "k": k, "res": res})

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
            dl_md_button(t(lang, "export_advice"), output_text, "caremind_advice.md")
            dl_md_button(t(lang, "export_evidence"), evidence_md(lang, res.get("guideline_hits") or []), "caremind_evidence.md")
        st.caption(t(lang, "disclaimer"))

    # Evidence
    with tab_hits:
        hits: List[Dict[str, Any]] = res.get("guideline_hits") or []

        def pass_filter(h: Dict[str, Any]) -> bool:
            m = h.get("meta") or {}
            src_ok = (src_filter.strip().lower() in (m.get("source", "").lower())) if src_filter.strip() else True
            y = m.get("year")
            try:
                y = int(y)
            except Exception:
                y = None
            year_ok = (year_min <= y <= year_max) if y else True
            return src_ok and year_ok

        hits = [h for h in hits if pass_filter(h)]
        st.subheader(t(lang, "hits_hdr").format(k=k, n=len(hits)))
        if not hits:
            st.info(t(lang, "no_hits"))
        else:
            # source chips
            sources = {}
            for h in hits:
                m = h.get("meta") or {}
                s = str(m.get("source") or ("未知来源" if lang=="zh" else "Unknown")).strip()
                sources[s] = sources.get(s, 0) + 1
            st.markdown(" ".join([f"<span class='cm-chip'>{s} × {n}</span>" for s, n in sources.items()]), unsafe_allow_html=True)

            for i, h in enumerate(hits, 1):
                m = h.get("meta") or {}
                title = str(m.get("title") or ( "无标题" if lang=="zh" else "Untitled"))
                source = str(m.get("source") or ("未知" if lang=="zh" else "Unknown"))
                year = str(m.get("year") or "—")
                doc_id = str(m.get("id") or "—")
                label = f"#{i} · {title[:60]}"
                st.markdown(f"<a id='hit-{i}'></a>", unsafe_allow_html=True)
                with st.expander(label, expanded=expand_hits):
                    if show_meta:
                        st.markdown(
                            f"<div class='cm-muted'>"
                            f"<span class='cm-badge'>来源：{source or '未知'}</span>"
                            f"<span class='cm-badge'>年份：{year or '—'}</span>"
                            f"<span class='cm-badge'>ID：{doc_id or '—'}</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                    st.markdown(h.get("content") or "（空片段）")

        # —— 结构化药品信息
        drug_obj = res.get("drug")
        st.divider()
        st.subheader("💊 药品结构化信息（SQLite）")
        if drug_obj:
            st.json(drug_obj, expanded=False)
        else:
            st.caption("未提供或未检索到对应药品的结构化信息。")

# ---------------------------
# 历史记录（本会话）
# ---------------------------
with st.expander("🗂️ 本会话历史（仅本地会话内可见）", expanded=False):
    if not st.session_state.history:
        st.caption("暂无历史记录。")
    else:
        for idx, item in enumerate(st.session_state.history, 1):
            st.markdown(
                f"**{idx}.** Q: `{item['q']}` | 药品: `{item['drug'] or '—'}` | Top-K: `{item['k']}`"
            )

# ---------------------------
# 页脚提示
# ---------------------------
st.caption("© CareMind · MVP CDSS | 本工具仅供临床决策参考，不替代医师诊断与处方。")
                            f"<span class='cm-badge'>{t(lang, 'chips_src')} {source}</span>"
                            f"<span class='cm-badge'>{t(lang, 'chips_year')} {year}</span>"
                            f"<span class='cm-badge'>{t(lang, 'chips_id')} {doc_id}</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                    st.markdown(highlight(h.get("content") or "", q), unsafe_allow_html=True)

            total_chars = sum(len((h.get("content") or "")) for h in hits)
            st.caption(t(lang, "stats_hits").format(n=len(hits), c=total_chars))

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
        st.session_state.logs.insert(0, log)
        st.json(log)
        st.download_button(
            t(lang, "log_export"),
            data=json.dumps(st.session_state.logs, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="caremind_logs.json",
            mime="application/json",
            use_container_width=True,
        )

# History reuse
with st.expander(t(lang, "history_hdr"), expanded=False):
    if not st.session_state.history:
        st.caption(t(lang, "no_history"))
    else:
        for idx, item in enumerate(st.session_state.history, 1):
            c1, c2 = st.columns([0.8, 0.2])
            with c1:
                st.caption(f"{idx}. Q: {item['q']} | Drug: {item['drug'] or '—'} | K={item['k']}")
            with c2:
                if st.button(t(lang, "reuse"), key=f"reuse_{idx}"):
                    st.session_state["reseed"] = (item["q"], item["drug"] or "", item["k"], lang)
                    st.rerun()

if "reseed" in st.session_state:
    q_old, drug_old, k_old, lang_old = st.session_state.pop("reseed")
    # 仅提示用户复用成功（Streamlit 无法直接回填 form 里的 value）
    st.info(t(lang, "reused_tip").format(q=q_old, drug=(drug_old or ("—" if lang=="zh" else "—")), k=k_old))

# Footer
st.caption(t(lang, "page_footer"))