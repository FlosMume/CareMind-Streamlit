# -*- coding: utf-8 -*-
"""
CareMind · MVP CDSS (Streamlit, bilingual zh/en)
------------------------------------------------
特性 / Features
- 双语 UI（中文 / English）
- 通过 rag.pipeline.answer 提供建议文本（反射式调用，兼容是否含 lang 参数）
- 证据片段/药品结构化/运行日志 Tab
- ✅ 诊断面板：展示有效配置（Secrets 优先）、chroma_store 是否存在、
  Chroma 集合与条目数（调用 retriever.list_collections_safe 防止 `_type` 报错）、
  SQLite 文件是否存在与表清单
- 不显示 Python 版本信息
"""

from __future__ import annotations

import os
import re
import json
import time
import inspect
from typing import Any, Dict, List, Optional

import streamlit as st
import rag.pipeline as cm_pipeline          # 用模块导入，避免热重载下的符号遮蔽
from rag import retriever as R              # 供诊断面板使用（读取常量 + 安全列集合）


# =============================================================================
# 0) 辅助函数 / Helpers
# -----------------------------------------------------------------------------
def _env(key: str, default: str | None = None) -> str | None:
    """优先 Secrets 再 env，最后默认。"""
    try:
        return os.getenv(key, st.secrets.get(key, default))
    except Exception:
        return os.getenv(key, default)

def _safe_int(x: Any, default: int = 4) -> int:
    try:
        v = int(x)
        return v if v > 0 else default
    except Exception:
        return default

def _count_chars(snippets: List[Dict[str, Any]]) -> int:
    c = 0
    for h in snippets or []:
        c += len((h.get("content") or "").strip())
    return c


# =============================================================================
# 1) 文案 / i18n
# -----------------------------------------------------------------------------
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
        "export_hits": "导出证据（JSON）",
        "chips_src": "来源：",
        "chips_year": "年份：",
        "chips_id": "ID：",
        "stats_hits": "片段数：{n} · 总字数：{c}",
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
        "preset1": "CKD + HTN on ACEI/ARB monitoring",
        "preset2": "Elderly with T2DM+CAD: BP target & plan",
        "preset3": "GDM insulin initiation (indications & dose)",
        "advice_hdr": "Advice (with citations & compliance)",
        "time_used": "⏱️ Time used: {:+.2f}s",
        "export_advice": "Export Advice (Markdown)",
        "export_hits": "Export Evidence (JSON)",
        "chips_src": "Source:",
        "chips_year": "Year:",
        "chips_id": "ID:",
        "stats_hits": "Snippets: {n} · Total chars: {c}",
        "warn_need_q": "Please enter a clinical question first.",
        "err_backend": "Backend error (see logs/diagnostics below).",
        "diag_title": "Run Logs / Environment Diagnostics",
        "diag_cfg": "Effective configuration (Secrets first)",
        "diag_chroma": "Chroma collections:",
        "diag_chroma_err": "Chroma error:",
        "diag_sqlite": "SQLite tables:",
        "diag_sqlite_err": "SQLite error: ",
    },
}
def t(lang: str, key: str) -> str:
    return I18N.get(lang, I18N["zh"]).get(key, key)


# =============================================================================
# 2) 页面配置 & 轻量样式
# -----------------------------------------------------------------------------
st.set_page_config(page_title="CareMind · MVP CDSS", layout="wide", page_icon="💊")
st.markdown("""
<style>
.cm-badge{display:inline-block;padding:2px 8px;border-radius:12px;background:#eef2ff;color:#3730a3;font-size:12px;border:1px solid #c7d2fe;margin-right:6px;white-space:nowrap;}
.cm-chip{display:inline-block;padding:2px 8px;border-radius:8px;background:#f1f5f9;color:#0f172a;font-size:12px;border:1px solid #e2e8f0;margin-right:6px;margin-bottom:6px;}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# 3) 侧边栏设置
# -----------------------------------------------------------------------------
st.sidebar.header(t("zh", "settings"))
lang = st.sidebar.selectbox("Language / 语言", ["zh", "en"], index=0)
k = st.sidebar.slider(t(lang, "k_slider"), min_value=1, max_value=10, value=4, step=1)
show_meta = st.sidebar.checkbox(t(lang, "show_meta"), value=False)
expand_hits = st.sidebar.checkbox(t(lang, "expand_hits"), value=False)

# 这里是你原始的 UI/逻辑区域（保持原样）……
# （为了篇幅，这里省略你原本的业务区块；请直接用你仓库里的那一段）

# =============================================================================
# 8) 诊断面板（使用 retriever.list_collections_safe）
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
        # ✅ 最小补丁：显示 retriever 版本号，确认云端是否更新到位
        st.write("Retriever version:", getattr(R, "VERSION", "unknown"))

        # Chroma 目录存在性
        chroma_dir = eff.get("CHROMA_PERSIST_DIR") or "./chroma_store"
        abs_chroma = os.path.abspath(chroma_dir)
        st.write(f"{'Chroma 目录存在：' if lang=='zh' else 'Chroma dir exists:'} "
                 f"{abs_chroma} → {os.path.exists(abs_chroma)}")

        # 集合与条数（安全方式；避免 _type）
        try:
            cols = R.list_collections_safe()
            st.write(t(lang, "diag_chroma"))
            st.json(cols)
        except Exception as e:
            st.warning(t(lang, "diag_chroma_err") + str(e))

        # SQLite 存在性与表
        db_path = eff.get("DRUG_DB_PATH") or "./db/drugs.sqlite"
        abs_db = os.path.abspath(db_path)
        st.write(f"{'SQLite 文件存在：' if lang=='zh' else 'SQLite file exists:'} "
                 f"{abs_db} → {os.path.exists(abs_db)}")
        try:
            import sqlite3
            con = sqlite3.connect(abs_db)
            cur = con.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [r[0] for r in cur.fetchall()]
            con.close()
            st.write(t(lang, "diag_sqlite"))
            st.json(tables)
        except Exception as e:
            st.warning(t(lang, "diag_sqlite_err") + str(e))

# 页面底部渲染诊断
render_diagnostics(lang)

# =============================================================================
# 9) 页脚
# -----------------------------------------------------------------------------
st.caption("© CareMind demo – for internal testing/education only.")
