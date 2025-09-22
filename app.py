# -*- coding: utf-8 -*-
"""
CareMind · MVP CDSS (Streamlit, bilingual zh/en)
------------------------------------------------
保留原有特性基础上做了两点增强：
1) “建议”页签中的导出按钮更紧凑（两列），并且**仅在内容非空时显示**，避免浏览器出现 .crdownload
2) 与 pipeline 的“无 LLM 草案”配合，建议区会显示 3–5 条要点，每条带 [#i] 可跳回证据片段

其余：诊断面板、证据筛选、历史复用等保持不变。
"""

from __future__ import annotations
import os, re, json, time, inspect
from typing import Any, Dict, List, Optional

import streamlit as st
import rag.pipeline as cm_pipeline          # 用模块导入，避免热重载遮蔽
from rag import retriever as R              # 供诊断面板使用（读取版本 + 安全列集合）

# ---------------------------
# 小工具
# ---------------------------
def _env(key: str, default: str | None = None) -> str | None:
    """Secrets-aware env reader：优先 st.secrets，再读 os.environ，最后默认值。"""
    try:
        return os.getenv(key, st.secrets.get(key, default))
    except Exception:
        return os.getenv(key, default)

def link_citations(md: str) -> str:
    """把 "[#3]" 或 "[3]" 替换为页面锚点 "#hit-3"，便于从建议快速跳回证据。"""
    return re.sub(r"\[(?:#)?(\d+)\]", r"[\1](#hit-\1)", md or "")

def evidence_md(lang: str, hits: List[Dict[str, Any]]) -> str:
    """将证据片段渲染为 Markdown（用于下载）。"""
    lines = []
    for i, h in enumerate(hits or [], 1):
        m = h.get("meta") or {}
        title  = str(m.get("title")  or ("无标题" if lang == "zh" else "Untitled"))
        source = str(m.get("source") or ("未知"   if lang == "zh" else "Unknown"))
        year   = str(m.get("year")   or "—")
        head = f"### #{i} {title}\n\n" + (f"- 来源：{source} · 年份：{year}\n\n" if lang=="zh"
                                          else f"- Source: {source} · Year: {year}\n\n")
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

# ---------------------------
# I18N（页面文案）
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
        "warn_need_q": "请输入临床问题后再生成建议。",
        "err_backend": "后端错误（详见下方日志/诊断）。",
        "diag_title": "运行日志 / 环境诊断",
        "diag_cfg": "有效配置（优先 Secrets）",
        "diag_chroma": "Chroma 集合：",
        "diag_chroma_err": "Chroma 访问错误：",
        "diag_sqlite": "SQLite 表：",
        "diag_sqlite_err": "SQLite 错误：",
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
        "preset2": "Elderly with T2DM+CAD: target BP and first-line therapy",
        "preset3": "GDM: when to start insulin",
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
        "page_footer": "© CareMind · MVP CDSS | For clinical reference only.",
        "chips_src": "Source:",
        "chips_year": "Year:",
        "chips_id": "ID:",
        "warn_need_q": "Please enter a clinical question first.",
        "err_backend": "Backend error (see logs/diagnostics below).",
        "diag_title": "Runtime Log / Diagnostics",
        "diag_cfg": "Effective config (Secrets-first):",
        "diag_chroma": "Chroma collections:",
        "diag_chroma_err": "Chroma access error: ",
        "diag_sqlite": "SQLite tables:",
        "diag_sqlite_err": "SQLite error: ",
    },
}
def t(lang: str, key: str) -> str:
    return I18N.get(lang, I18N["zh"]).get(key, key)

# ---------------------------
# 页面配置 & 轻量样式
# ---------------------------
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

# ---------------------------
# 侧边栏
# ---------------------------
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
            t("en","preset3"): "GDM: when to start insulin and starting dose?",
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

# ---------------------------
# 输入区
# ---------------------------
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

# ---------------------------
# 页签
# ---------------------------
tab_adv, tab_hits, tab_drug, tab_log = st.tabs([
    t(lang, "tab_advice"),
    t(lang, "tab_hits"),
    t(lang, "tab_drug"),
    t(lang, "tab_log"),
])

res: Optional[Dict[str, Any]] = None
elapsed: Optional[float] = None

# ---------------------------
# 调用后端（反射式，兼容是否含 lang 参数）
# ---------------------------
if submitted:
    if not (q and q.strip()):
        st.warning(t(lang, "warn_need_q"))
    else:
        with st.spinner("..."):
            try:
                t0 = time.time()
                sig_params = inspect.signature(cm_pipeline.answer).parameters
                if "lang" in sig_params:
                    res = cm_pipeline.answer(q.strip(), drug_name=(drug.strip() or None), k=int(k), lang=lang)
                else:
                    res = cm_pipeline.answer(q.strip(), drug_name=(drug.strip() or None), k=int(k))
                elapsed = time.time() - t0

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

# ---------------------------
# 渲染结果
# ---------------------------
if res:
    # --- 建议 ---
    with tab_adv:
        st.subheader(t(lang, "advice_hdr"))

        # 由 pipeline 生成的 Markdown（含 [#i] 引用），此处把引用转锚点
        output_text = link_citations(res.get("output") or "")
        st.markdown(f"<div class='cm-output'>{output_text}</div>", unsafe_allow_html=True)
        if elapsed is not None:
            st.caption(t(lang, "time_used").format(elapsed))

        # ✅ 导出区域：并排两列，且只有内容非空才显示按钮（避免 .crdownload）
        exp1, exp2 = st.columns(2)
        with exp1:
            if (res.get("output") or "").strip():
                st.download_button(
                    t(lang, "export_advice"),
                    data=(res["output"]).encode("utf-8"),
                    file_name="caremind_advice.md",
                    mime="text/markdown",
                    key="dl_advice",
                    use_container_width=True,
                )
            else:
                st.caption("（当前无可导出的建议文本）" if lang=="zh" else "(No advice to export)")
        with exp2:
            ev_md = evidence_md(lang, res.get("guideline_hits") or [])
            if ev_md.strip():
                st.download_button(
                    t(lang, "export_evidence"),
                    data=ev_md.encode("utf-8"),
                    file_name="caremind_evidence.md",
                    mime="text/markdown",
                    key="dl_evidence",
                    use_container_width=True,
                )
            else:
                st.caption("（当前无可导出的证据）" if lang=="zh" else "(No evidence to export)")

        st.caption(t(lang, "disclaimer"))

    # --- 证据片段 ---
    with tab_hits:
        hits: List[Dict[str, Any]] = res.get("guideline_hits") or []

        # 前端筛选：来源包含 / 年份范围
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
            # 顶部显示来源计数的“小筹码”
            counts: Dict[str, int] = {}
            for h in hits:
                m = h.get("meta") or {}
                s = str(m.get("source") or ("未知来源" if lang == "zh" else "Unknown")).strip()
                counts[s] = counts.get(s, 0) + 1
            st.markdown(" ".join([f"<span class='cm-chip'>{s} × {n}</span>" for s, n in counts.items()]),
                        unsafe_allow_html=True)

            # 逐条展开
            for i, h in enumerate(hits, 1):
                m = h.get("meta") or {}
                title  = str(m.get("title")  or ("无标题" if lang == "zh" else "Untitled"))
                source = str(m.get("source") or ("未知"   if lang == "zh" else "Unknown"))
                year   = str(m.get("year")   or "—")
                doc_id = str(m.get("id")     or "—")
                label = f"#{i} · {title[:60]}"
                st.markdown(f"<a id='hit-{i}'></a>", unsafe_allow_html=True)
                with st.expander(label, expanded=False):
                    st.markdown(
                        f"<div class='cm-muted'>"
                        f"<span class='cm-badge'>{('来源：' if lang=='zh' else 'Source: ')}{source}</span>"
                        f"<span class='cm-badge'>{('年份：' if lang=='zh' else 'Year: ')}{year}</span>"
                        f"<span class='cm-badge'>ID: {doc_id}</span>"
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
            "导出本会话日志（JSON）" if lang=="zh" else "Export session logs (JSON)",
            data=json.dumps([log], ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="caremind_logs.json",
            mime="application/json",
            use_container_width=True,
        )

# ---------------------------
# 诊断面板（始终可见；使用 retriever.list_collections_safe）
# ---------------------------
def render_diagnostics(lang: str = "zh") -> None:
    with st.expander("运行日志 / 环境诊断" if lang=="zh" else "Runtime Log / Diagnostics", expanded=False):
        # 有效配置（Secrets 优先）
        keys = ["CAREMIND_DEMO", "CHROMA_PERSIST_DIR", "CHROMA_COLLECTION", "EMBEDDING_MODEL", "DRUG_DB_PATH"]
        eff = {k: _env(k, None) for k in keys}
        st.code(json.dumps(eff, ensure_ascii=False, indent=2))
        st.write("Retriever version:", getattr(R, "VERSION", "unknown"))

        # Chroma 目录存在性
        chroma_dir = eff.get("CHROMA_PERSIST_DIR") or "./chroma_store"
        abs_chroma = os.path.abspath(chroma_dir)
        st.write(("Chroma 目录存在：" if lang=='zh' else "Chroma dir exists:"),
                 abs_chroma, "→", os.path.exists(abs_chroma))

        # 集合与条数（安全方式；避免 _type）
        try:
            cols = R.list_collections_safe()
            st.json(cols)
        except Exception as e:
            st.warning(("Chroma 访问错误：" if lang=='zh' else "Chroma access error: ") + str(e))

        # SQLite 存在性与表
        db_path = eff.get("DRUG_DB_PATH") or "./db/drugs.sqlite"
        abs_db = os.path.abspath(db_path)
        st.write(("SQLite 文件存在：" if lang=='zh' else "SQLite file exists:"), abs_db, "→", os.path.exists(abs_db))
        try:
            import sqlite3
            con = sqlite3.connect(abs_db)
            cur = con.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [r[0] for r in cur.fetchall()]
            con.close()
            st.json(tables)
        except Exception as e:
            st.warning(("SQLite 错误：" if lang=='zh' else "SQLite error: ") + str(e))

# 页面底部渲染诊断与页脚
render_diagnostics(lang)
st.caption("© CareMind · MVP CDSS | 本工具仅供临床决策参考，不替代医师诊断与处方。" if lang=="zh"
           else "© CareMind · MVP CDSS | For clinical reference only.")