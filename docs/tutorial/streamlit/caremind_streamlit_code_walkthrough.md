# Streamlit Tutorial for CareMind: Building a Clinical Decision Support System

## Overview

This tutorial teaches **Streamlit** through the lens of building CareMind, a bilingual clinical decision support system. You'll learn core Streamlit concepts by exploring real production code.

**What you'll learn:**
- Streamlit app structure and execution model
- State management and caching strategies
- Multi-tab layouts and forms
- Bilingual internationalization (i18n)
- Session state and history tracking
- Dynamic content rendering
- Error handling and diagnostics

**Prerequisites:**
- Python 3.9+
- Basic understanding of web applications
- Familiarity with dictionary and function concepts

---

## Table of Contents

1. [Streamlit Execution Model](#1-streamlit-execution-model)
2. [Basic App Structure](#2-basic-app-structure)
3. [Page Configuration and Styling](#3-page-configuration-and-styling)
4. [Sidebar Components](#4-sidebar-components)
5. [Forms and User Input](#5-forms-and-user-input)
6. [Tabs and Multi-View Layout](#6-tabs-and-multi-view-layout)
7. [Session State Management](#7-session-state-management)
8. [Caching with @st.cache_resource](#8-caching-with-stcache_resource)
9. [Internationalization (i18n)](#9-internationalization-i18n)
10. [Dynamic Content Rendering](#10-dynamic-content-rendering)
11. [Error Handling and Diagnostics](#11-error-handling-and-diagnostics)
12. [Advanced Patterns](#12-advanced-patterns)

---

## 1. Streamlit Execution Model

### How Streamlit Works

**Key concept:** Streamlit reruns your entire script from top to bottom on every user interaction.

```python
import streamlit as st

# This code runs EVERY TIME user interacts with ANY widget
st.title("CareMind")

# User types in text input → ENTIRE SCRIPT RERUNS
query = st.text_input("Enter question")

# User clicks button → ENTIRE SCRIPT RERUNS AGAIN
if st.button("Submit"):
    st.write("You clicked!")
```

**Rerun triggers:**
- Text input changes
- Button clicks
- Slider movements
- Selectbox changes
- Any widget interaction

### Why This Matters for CareMind

```python
# ❌ BAD: This loads the model on EVERY rerun (every keystroke!)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-large-zh-v1.5")  # 1.2GB, takes 10 seconds!

query = st.text_input("Query")  # User types "M" → rerun → load model
                                 # User types "Me" → rerun → load model again!

# ✓ GOOD: Cache expensive operations (covered later)
@st.cache_resource
def load_model():
    return SentenceTransformer("BAAI/bge-large-zh-v1.5")

model = load_model()  # Loads once, cached forever
query = st.text_input("Query")  # User types → rerun → uses cached model
```

**Mental model:**
```
User Action → Script Reruns → Widgets Redraw → Display Updates
     ↑                                              |
     └──────────────────────────────────────────────┘
```

---

## 2. Basic App Structure

### Minimal Streamlit App

```python
# minimal_app.py
import streamlit as st

# Every rerun starts here
st.title("Hello CareMind")
name = st.text_input("Your name")
if name:
    st.write(f"Welcome, {name}!")
```

Run with:
```bash
streamlit run minimal_app.py
```

### CareMind App Structure

CareMind's `app.py` follows this pattern:

```python
# 1. IMPORTS - Run once per rerun (but imports are cached by Python)
import streamlit as st
from rag import retriever as R
import rag.pipeline as cm_pipeline

# 2. CONSTANTS - Define once, available throughout script
VERSION = "app-2025-12-20"

# 3. HELPER FUNCTIONS - Define but don't execute until called
def _env(key: str, default: str = None) -> str:
    """Secrets-aware environment reader."""
    return os.getenv(key, st.secrets.get(key, default))

def link_citations(md: str) -> str:
    """Convert [3] to anchor links [3](#hit-3)."""
    return re.sub(r"\[(?:#)?(\d+)\]", r"[\1](#hit-\1)", md or "")

# 4. PAGE CONFIGURATION - Must be first st.* call
st.set_page_config(
    page_title="CareMind · MVP CDSS",
    layout="wide",  # Use full browser width
    page_icon="💊"
)

# 5. SIDEBAR - Fixed position, always visible
with st.sidebar:
    lang = st.selectbox("Language", ["zh", "en"])
    k = st.slider("Top-K", 2, 8, 4)

# 6. MAIN CONTENT - Center of page
st.title("CareMind")
query = st.text_input("Enter question")

# 7. CONDITIONAL RENDERING - Based on user actions
if query:
    results = cm_pipeline.answer(query, k=k)
    st.write(results)
```

**Key insight:** Everything after imports runs on EVERY rerun. Use caching and session state to preserve expensive operations.

---

## 3. Page Configuration and Styling

### Page Configuration

**Must be the FIRST Streamlit command:**

```python
import streamlit as st

# ✓ GOOD: First st.* call
st.set_page_config(
    page_title="CareMind · MVP CDSS",  # Browser tab title
    layout="wide",                      # "wide" or "centered"
    page_icon="💊",                     # Emoji or URL to icon
    initial_sidebar_state="expanded"    # "expanded" or "collapsed"
)

# ❌ BAD: st.title() before set_page_config
# st.title("App")  # This would cause error
# st.set_page_config(...)  # Too late!
```

### Custom CSS

Streamlit allows custom CSS via `st.markdown()`:

```python
st.markdown("""
<style>
    /* Custom badge style for metadata */
    .cm-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 12px;
        background: #eef2ff;
        border: 1px solid #c7d2fe;
        margin-right: 6px;
        white-space: nowrap;
    }
    
    /* Chip style for filters */
    .cm-chip {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 8px;
        font-size: 12px;
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
        margin: 0 6px 6px 0;
    }
    
    /* Muted text for secondary info */
    .cm-muted {
        color: #64748b;
        font-size: 13px;
    }
    
    /* Hide Streamlit's default footer */
    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)

# Use the custom classes:
st.markdown(
    f"<span class='cm-badge'>Source: KDIGO</span>"
    f"<span class='cm-badge'>Year: 2023</span>",
    unsafe_allow_html=True
)
```

**CareMind usage:**
- `.cm-badge` - Metadata badges (source, year, ID)
- `.cm-chip` - Filter chips showing distribution
- `.cm-muted` - Subtle secondary information

---

## 4. Sidebar Components

### Sidebar Basics

The sidebar is a **fixed** column on the left, perfect for settings and filters:

```python
# Everything in this block goes to sidebar
with st.sidebar:
    st.header("Settings")
    
    # Selectbox
    lang = st.selectbox(
        "Language",
        options=["zh", "en"],
        index=1,  # Default to second option (en)
        format_func=lambda x: "中文" if x == "zh" else "English"
    )
    
    # Slider
    k = st.slider(
        "Top-K Results",
        min_value=2,
        max_value=8,
        value=4,      # Default value
        step=1
    )
    
    # Toggle switches
    show_meta = st.toggle("Show metadata", value=True)
    expand_all = st.toggle("Expand all", value=False)
    
    # Divider
    st.divider()
    
    # Text input
    filter_text = st.text_input("Filter by source")
```

### CareMind Sidebar Structure

```python
with st.sidebar:
    # 1. LANGUAGE SELECTOR (affects entire UI)
    lang = st.selectbox("Language / 语言", ["zh", "en"], index=1)
    
    # 2. SETTINGS SECTION
    st.header(t(lang, "settings"))  # t() is i18n helper
    k = st.slider(t(lang, "k_slider"), 2, 8, 4)
    show_meta = st.toggle(t(lang, "show_meta"), True)
    
    # 3. FILTERS SECTION
    st.divider()
    st.markdown(f"#### {t(lang, 'filters')}")
    src_filter = st.text_input(t(lang, "filter_src"))
    year_min, year_max = st.slider(
        t(lang, "filter_year"),
        2000, 2035,
        (2005, 2035)  # Range slider: returns tuple
    )
    
    # 4. PRESETS SECTION
    st.divider()
    st.markdown(f"#### {t(lang, 'presets')}")
    presets = {
        "zh": {
            "CKD监测": "慢性肾病患者使用ACEI/ARB时如何监测？",
            "降压目标": "老年合并糖尿病与冠心病的降压目标？"
        },
        "en": {
            "CKD Monitoring": "For CKD on ACEI/ARB, how to monitor?",
            "BP Target": "Elderly with T2DM+CAD: target BP?"
        }
    }
    preset = st.selectbox("Quick pick", list(presets[lang].keys()))
    
    # 5. HISTORY SECTION
    st.markdown(f"#### {t(lang, 'history')}")
    hist = st.session_state.get("cm_history", [])
    for i, h in enumerate(hist[-5:], 1):  # Show last 5
        if st.button(f"{i}. {h['q'][:30]}...", key=f"hist_{i}"):
            st.session_state["prefill"] = h  # Store for next rerun
```

**Key patterns:**
- `st.divider()` - Visual separator between sections
- `st.markdown(f"#### {title}")` - Section headers with i18n
- `st.button(..., key=f"hist_{i}")` - Unique keys for button loops
- `st.session_state` - Preserve data across reruns

---

## 5. Forms and User Input

### Why Use Forms?

**Without form:** Every keystroke triggers a rerun
```python
# User types "M" → rerun → slow backend call
# User types "Me" → rerun → slow backend call again
# User types "Metformin" → rerun → 10+ backend calls!
query = st.text_input("Query")
if query:
    results = slow_backend_call(query)  # Called on EVERY keystroke!
```

**With form:** Only rerun when user clicks submit
```python
with st.form("my_form"):
    query = st.text_input("Query")
    # User types "Metformin" → NO rerun yet
    
    submitted = st.form_submit_button("Submit")
    # User clicks submit → NOW rerun happens

if submitted and query:
    results = slow_backend_call(query)  # Called ONCE per submit
```

### CareMind Form

```python
# Form prevents reruns until user clicks submit
with st.form("cm_query"):
    # 1. PREFILL LOGIC (from history or presets)
    prefill = st.session_state.pop("prefill", None)
    # pop() removes from session_state after reading
    # Ensures prefill only applies once
    
    # Determine initial value
    q_init = ""
    if prefill:
        # From history: user clicked "Reuse" button
        q_init = prefill.get("q", "")
    elif preset_choice != "——":
        # From preset: user selected quick question
        q_init = presets[lang].get(preset_choice, "")
    
    # 2. INPUT FIELDS
    q = st.text_input(
        t(lang, "question_label"),
        placeholder=t(lang, "question_ph"),
        value=q_init  # Pre-filled value
    )
    
    drug = st.text_input(
        t(lang, "drug_label"),
        value=(prefill or {}).get("drug", "")
    )
    
    # 3. SUBMIT BUTTON (must be inside form)
    submitted = st.form_submit_button(
        t(lang, "submit"),
        use_container_width=True  # Full-width button
    )

# 4. PROCESS SUBMISSION (outside form)
if submitted:
    if not q.strip():
        # Validation: show warning if empty
        st.warning(t(lang, "warn_need_q"))
    else:
        # Call backend
        with st.spinner("..."):  # Show loading spinner
            res = cm_pipeline.answer(q, drug_name=drug, k=k)
        
        # Save to history
        st.session_state.setdefault("cm_history", []).append({
            "q": q, "drug": drug, "k": k, "time": time.time()
        })
```

**Key patterns:**
- `st.session_state.pop("prefill", None)` - Read and remove in one step
- `value=q_init` - Pre-fill input with history/preset
- `st.spinner("...")` - Show loading indicator
- `st.warning()` - Friendly validation messages

### Form Submit Flow

```
User fills form fields → User types, no rerun happens
                         ↓
User clicks submit button → Form submits
                         ↓
Script reruns with submitted=True
                         ↓
if submitted: block executes
                         ↓
Backend called, results displayed
```

---

## 6. Tabs and Multi-View Layout

### Basic Tabs

```python
# Create tabs
tab1, tab2, tab3 = st.tabs(["Tab 1", "Tab 2", "Tab 3"])

# Content for tab 1
with tab1:
    st.write("This is tab 1")
    st.button("Button 1")

# Content for tab 2
with tab2:
    st.write("This is tab 2")
    st.slider("Slider 2", 0, 10)

# Content for tab 3
with tab3:
    st.write("This is tab 3")
```

### CareMind Tabs with i18n

```python
# 1. DEFINE TAB NAMES (translated)
tab_adv, tab_evidence, tab_drug, tab_hits, tab_log = st.tabs([
    t(lang, "tab_advice"),         # "🧭 建议" or "🧭 Advice"
    t(lang, "tab_evidence_list"),  # "📑 证据清单" or "📑 Evidence List"
    t(lang, "tab_hits_raw"),       # "🎯 命中" or "🎯 Hits (Raw)"
    t(lang, "tab_drug"),           # "💊 药品结构化" or "💊 Drug (Structured)"
    t(lang, "tab_log"),            # "🪵 运行日志" or "🪵 Run Logs"
])

# 2. POPULATE TABS (only if results exist)
if res:  # res is None until user submits query
    
    # --- ADVICE TAB ---
    with tab_adv:
        st.subheader(t(lang, "advice_hdr"))
        
        # Display advice with citation links
        output_text = link_citations(res.get("output", ""))
        st.markdown(output_text)
        
        # Show elapsed time
        if elapsed is not None:
            st.caption(f"⏱️ Elapsed: {elapsed:.2f}s")
        
        # Download buttons (side-by-side using columns)
        col1, col2, _ = st.columns([1, 1, 4])
        with col1:
            st.download_button(
                "Export Advice",
                data=output_text.encode("utf-8"),
                file_name="advice.md",
                mime="text/markdown"
            )
        with col2:
            st.download_button(
                "Export Evidence",
                data=evidence_md(lang, res["hits"]).encode("utf-8"),
                file_name="evidence.md"
            )
    
    # --- EVIDENCE TAB ---
    with tab_evidence:
        # Clean, formatted evidence (no metadata clutter)
        ev = res.get("guideline_hits_md", "(No evidence)")
        st.markdown(ev)
    
    # --- HITS TAB (detailed) ---
    with tab_hits:
        hits = res.get("guideline_hits", [])
        
        # Client-side filtering
        hits = [h for h in hits if pass_filter(h)]
        
        st.subheader(f"Retrieved ({len(hits)} hits)")
        
        # Show distribution chips
        counts = {}
        for h in hits:
            src = h["meta"].get("source", "Unknown")
            counts[src] = counts.get(src, 0) + 1
        
        st.markdown(" ".join([
            f"<span class='cm-chip'>{src} × {n}</span>"
            for src, n in counts.items()
        ]), unsafe_allow_html=True)
        
        # Expandable hit cards
        for i, h in enumerate(hits, 1):
            meta = h.get("meta", {})
            title = meta.get("title", "Untitled")
            
            # Anchor for citation links
            st.markdown(f"<a id='hit-{i}'></a>", unsafe_allow_html=True)
            
            with st.expander(f"#{i} · {title[:60]}", expanded=False):
                # Show metadata badges if enabled
                if show_meta:
                    st.markdown(
                        f"<span class='cm-badge'>Source: {meta['source']}</span>"
                        f"<span class='cm-badge'>Year: {meta['year']}</span>",
                        unsafe_allow_html=True
                    )
                
                # Content
                st.markdown(h.get("content", ""))
    
    # --- DRUG TAB ---
    with tab_drug:
        st.subheader("Drug Info")
        if res.get("drug"):
            st.json(res["drug"])  # Pretty-print JSON
        else:
            st.caption("No drug info")
    
    # --- LOG TAB ---
    with tab_log:
        log = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "question": q,
            "drug": drug,
            "k": k,
            "elapsed_sec": round(elapsed, 3)
        }
        st.json(log)
        
        st.download_button(
            "Export Logs",
            data=json.dumps([log], ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="logs.json",
            mime="application/json"
        )
```

**Key patterns:**
- Tabs only show content if `res` exists (after submission)
- Each tab is independent - user can switch freely
- `st.markdown(f"<a id='hit-{i}'></a>", ...)` - Anchor for navigation
- `st.columns([1, 1, 4])` - Layout buttons side-by-side

---

## 7. Session State Management

### What is Session State?

**Problem:** Variables reset on every rerun
```python
count = 0  # This resets to 0 on EVERY rerun!

if st.button("Increment"):
    count += 1  # Never persists

st.write(f"Count: {count}")  # Always shows 0
```

**Solution:** Use `st.session_state` to persist data across reruns
```python
# Initialize on first run
if "count" not in st.session_state:
    st.session_state.count = 0

if st.button("Increment"):
    st.session_state.count += 1  # Persists!

st.write(f"Count: {st.session_state.count}")  # Shows actual count
```

### Session State Patterns

**Pattern 1: Initialize with setdefault**
```python
# Shorter way to initialize
st.session_state.setdefault("count", 0)

# Equivalent to:
if "count" not in st.session_state:
    st.session_state.count = 0
```

**Pattern 2: Append to list**
```python
# Initialize list if not exists
history = st.session_state.setdefault("history", [])

# Append new item
history.append({"query": "diabetes", "time": time.time()})

# Access later
for item in st.session_state.history:
    st.write(item["query"])
```

**Pattern 3: Temporary data with pop()**
```python
# Store data for next rerun
if st.button("Reuse last query"):
    st.session_state["prefill"] = {"query": "diabetes", "k": 5}

# On next rerun, read and remove
prefill = st.session_state.pop("prefill", None)
# prefill is now removed from session_state

if prefill:
    query = st.text_input("Query", value=prefill["query"])
    # This only happens once, then prefill is gone
```

### CareMind Session State Usage

```python
# 1. HISTORY TRACKING
# Store all queries in session
if submitted and q:
    # Initialize history list if first query
    hist = st.session_state.setdefault("cm_history", [])
    
    # Append new query
    hist.append({
        "q": q.strip(),
        "drug": drug.strip() or None,
        "k": int(k),
        "time": time.time()
    })

# 2. DISPLAY HISTORY IN SIDEBAR
with st.sidebar:
    st.markdown("#### History")
    hist = st.session_state.get("cm_history", [])
    
    if not hist:
        st.caption("No history yet")
    else:
        # Show last 8 queries (reversed = newest first)
        for i, h in enumerate(reversed(hist[-8:]), 1):
            # Truncate long queries
            q_short = h["q"][:36]
            if len(h["q"]) > 36:
                q_short += "..."
            
            st.write(f"{i}. {q_short}")
            
            # Reuse button
            if st.button("Reuse", key=f"reuse_{i}"):
                # Store in session state for next rerun
                st.session_state["prefill"] = h
                # Streamlit will rerun automatically

# 3. PREFILL FROM HISTORY
with st.form("cm_query"):
    # Read and remove prefill data
    prefill = st.session_state.pop("prefill", None)
    
    # Use prefill if exists
    q_init = (prefill or {}).get("q", "")
    drug_init = (prefill or {}).get("drug", "")
    
    q = st.text_input("Question", value=q_init)
    drug = st.text_input("Drug", value=drug_init)
    
    # If prefill exists, also update K slider
    if prefill and "k" in prefill:
        k = prefill["k"]  # This updates the slider default
    
    submitted = st.form_submit_button("Submit")
```

**Flow:**
```
1. User submits query
   → Saved to st.session_state["cm_history"]

2. User clicks "Reuse" button in sidebar
   → Sets st.session_state["prefill"] = history_item
   → Triggers rerun

3. Form reads prefill
   → prefill = st.session_state.pop("prefill", None)
   → Removes prefill so it only applies once
   → Pre-fills form inputs

4. User can edit and submit again
```

---

## 8. Caching with @st.cache_resource

### Why Caching?

**Problem:** Expensive operations run on every rerun
```python
# This loads a 1.2GB model on EVERY rerun!
model = SentenceTransformer("BAAI/bge-large-zh-v1.5")  # 10 seconds

query = st.text_input("Query")  # User types → model reloads!
```

**Solution:** Cache expensive resources
```python
@st.cache_resource
def load_model():
    # This runs ONCE, then cached forever
    return SentenceTransformer("BAAI/bge-large-zh-v1.5")

model = load_model()  # First call: loads model (10s)
                       # Subsequent calls: returns cached (instant)

query = st.text_input("Query")  # User types → uses cached model
```

### When to Use @st.cache_resource

**Cache these (expensive, stateless, global resources):**
- Machine learning models
- Database connections
- Large datasets
- File handles

```python
@st.cache_resource
def get_chroma_client():
    """ChromaDB client (expensive to create, thread-safe)."""
    import chromadb
    return chromadb.PersistentClient(path="./chroma_store")

@st.cache_resource
def load_embedder():
    """Embedding model (1.2GB, slow to load)."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("BAAI/bge-large-zh-v1.5")
```

**Don't cache these (dynamic, user-specific, cheap operations):**
- User inputs
- Search queries
- SQLite connections (not thread-safe)
- Simple calculations

```python
# ❌ BAD: Caches first user's query for everyone!
@st.cache_resource
def get_query():
    return st.text_input("Query")  # Wrong!

# ✓ GOOD: No caching for user input
query = st.text_input("Query")

# ❌ BAD: SQLite connections have state
@st.cache_resource
def get_db_connection():
    return sqlite3.connect("db.sqlite")  # Not thread-safe!

# ✓ GOOD: Fresh connection each time
def get_db_connection():
    return sqlite3.connect("db.sqlite")
```

### CareMind Caching Pattern

**In retriever.py:**
```python
@st.cache_resource
def get_chroma_client(persist_dir=None, version=VERSION):
    """
    Get cached ChromaDB client.
    
    Why cached:
    - Expensive: Loading metadata and indices takes ~2-5 seconds
    - Thread-safe: Can be shared across all users/sessions
    - Stateless: Doesn't hold user-specific data
    
    Parameters:
    - persist_dir: Path to chroma_store folder
    - version: Used to invalidate cache when code updates
    """
    import chromadb
    path = persist_dir or CHROMA_PERSIST_DIR
    return chromadb.PersistentClient(path=path)

# Usage in app.py:
client = get_chroma_client()  # First call: creates client
collection = client.get_collection("chunks")  # Uses cached client
```

**Why NOT cache SQLite?**
```python
def _connect_sqlite():
    """
    Create fresh SQLite connection (NOT cached).
    
    Why NOT cached:
    - Fast: sqlite3.connect() takes < 1ms for local files
    - Stateful: Connections hold transaction state and locks
    - Not thread-safe: Sharing across users causes corruption
    - Needs cleanup: Must close() after use
    """
    return sqlite3.connect("./db/drugs.sqlite")

# Usage:
conn = _connect_sqlite()  # Fresh connection
try:
    results = conn.execute("SELECT ...").fetchall()
finally:
    conn.close()  # Always close
```

### Cache Parameters

```python
@st.cache_resource(
    show_spinner=True,      # Show "Loading..." spinner (default: True)
    ttl=3600,               # Time-to-live in seconds (default: None = forever)
    max_entries=1           # Max cached entries (default: None = unlimited)
)
def load_config(path: str):
    with open(path) as f:
        return json.load(f)

# With TTL:
# - First call at 10:00 → loads config, caches for 1 hour
# - Call at 10:30 → returns cached
# - Call at 11:01 → cache expired, reloads
```

---

## 9. Internationalization (i18n)

### i18n Pattern

**Goal:** Support multiple languages without duplicating code.

**CareMind approach: Dictionary-based i18n**

```python
# 1. DEFINE TRANSLATIONS
I18N = {
    "zh": {
        "title": "CareMind · 临床决策支持（MVP）",
        "question_label": "输入临床问题",
        "question_ph": "例如：慢性肾病（CKD）患者使用 ACEI/ARB 时如何监测？",
        "submit": "生成建议",
        "warn_need_q": "请输入临床问题后再生成建议。",
    },
    "en": {
        "title": "CareMind · Clinical Decision Support (MVP)",
        "question_label": "Enter your clinical question",
        "question_ph": "e.g., For CKD patients on ACEI/ARB, how to monitor and how often?",
        "submit": "Generate Advice",
        "warn_need_q": "Please enter a clinical question first.",
    },
}

# 2. HELPER FUNCTION
def t(lang: str, key: str) -> str:
    """
    Translate key to language.
    
    Args:
        lang: "zh" or "en"
        key: Translation key (e.g., "title", "submit")
    
    Returns:
        Translated string, or key itself if not found
    """
    return I18N.get(lang, I18N["zh"]).get(key, key)

# 3. USAGE IN UI
lang = st.selectbox("Language", ["zh", "en"], index=1)

st.title(t(lang, "title"))
# lang="zh" → "CareMind · 临床决策支持（MVP）"
# lang="en" → "CareMind · Clinical Decision Support (MVP)"

query = st.text_input(
    t(lang, "question_label"),
    placeholder=t(lang, "question_ph")
)

if st.form_submit_button(t(lang, "submit")):
    if not query:
        st.warning(t(lang, "warn_need_q"))
```

### Complete i18n Example

```python
# Sidebar settings
with st.sidebar:
    # Language selector (affects all UI elements)
    lang = st.selectbox(
        "Language / 语言",  # Show both languages in label
        options=["zh", "en"],
        index=1,  # Default: English
        format_func=lambda x: "中文" if x == "zh" else "English"
    )
    
    st.header(t(lang, "settings"))
    k = st.slider(t(lang, "k_slider"), 2, 8, 4)
    # lang="zh" → "检索片段数（Top-K）"
    # lang="en" → "Top-K retrieved segments"

# Main content
st.title(t(lang, "title"))

with st.form("query"):
    q = st.text_input(
        t(lang, "question_label"),
        placeholder=t(lang, "question_ph")
    )
    submitted = st.form_submit_button(t(lang, "submit"))

if submitted:
    if not q:
        st.warning(t(lang, "warn_need_q"))
    else:
        # Pass language to backend for LLM response
        results = cm_pipeline.answer(q, lang=lang)
        
        # Display results (UI already translated via t())
        st.subheader(t(lang, "advice_hdr"))
        st.markdown(results["output"])
```

### Adding New Language Support

**Step 1: Add translations to I18N dict**
```python
I18N = {
    "zh": { ... },
    "en": { ... },
    "es": {  # Add Spanish
        "title": "CareMind · Sistema de Apoyo a Decisiones Clínicas",
        "submit": "Generar Consejo",
        # ... add all keys
    }
}
```

**Step 2: Update language selector**
```python
lang = st.selectbox(
    "Language / 语言 / Idioma",
    options=["zh", "en", "es"],
    index=1,
    format_func=lambda x: {
        "zh": "中文",
        "en": "English",
        "es": "Español"
    }[x]
)
```

**Step 3: Update backend (if needed)**
```python
# Pass language to LLM for language-specific responses
results = cm_pipeline.answer(q, lang=lang)
```

**No other changes needed!** All UI elements automatically use new language via `t(lang, key)`.

---

## 10. Dynamic Content Rendering

### Conditional Rendering

Only show content after certain conditions are met:

```python
# Initialize result to None
res = None

# Form submission
with st.form("query"):
    q = st.text_input("Question")
    submitted = st.form_submit_button("Submit")

if submitted and q:
    # Call backend
    res = cm_pipeline.answer(q)

# Only render results if res exists
if res:
    st.subheader("Results")
    st.write(res["output"])
    
    # Nested conditions
    if res.get("drug"):
        st.subheader("Drug Info")
        st.json(res["drug"])
else:
    # Show placeholder when no results
    st.info("Enter a question above to get started.")
```

### Dynamic Loops

Render variable number of items:

```python
# Get hits from backend
hits = res.get("guideline_hits", [])

if not hits:
    st.caption("No evidence found.")
else:
    # Render each hit
    for i, hit in enumerate(hits, 1):
        title = hit["meta"].get("title", "Untitled")
        content = hit.get("content", "")
        
        # Unique expander for each hit
        with st.expander(f"#{i} · {title}", expanded=False):
            st.markdown(content)
```

**Key:** Use `enumerate(..., 1)` to get 1-based index for display.

### Dynamic Columns

Layout items side-by-side:

```python
# Fixed-width columns
col1, col2 = st.columns(2)
with col1:
    st.write("Left column")
with col2:
    st.write("Right column")

# Variable-width columns (ratios)
col1, col2, col3 = st.columns([1, 2, 1])  # Middle is 2x wider

# CareMind example: Download buttons
b1, b2, spacer = st.columns([1, 1, 4])
with b1:
    st.download_button("Export Advice", data=advice_md, ...)
with b2:
    st.download_button("Export Evidence", data=evidence_md, ...)
# spacer column takes up remaining space, pushing buttons left
```

### Dynamic Tabs

```python
# Create tabs with translated names
tabs = st.tabs([
    t(lang, "tab_advice"),
    t(lang, "tab_evidence"),
    t(lang, "tab_drug")
])

# Populate tabs dynamically
if res:
    with tabs[0]:  # Advice tab
        st.markdown(res["output"])
    
    with tabs[1]:  # Evidence tab
        hits = res.get("guideline_hits", [])
        for hit in hits:
            st.markdown(hit["content"])
    
    with tabs[2]:  # Drug tab
        if res.get("drug"):
            st.json(res["drug"])
        else:
            st.caption("No drug info")
```

### Anchor Links

Create clickable links within the same page:

```python
# In Advice tab: Citations like [1], [2], [3]
def link_citations(md: str) -> str:
    """Convert [3] to [3](#hit-3)."""
    return re.sub(r"\[(?:#)?(\d+)\]", r"[\1](#hit-\1)", md or "")

advice = link_citations(res["output"])
st.markdown(advice)  # [3] becomes clickable link

# In Evidence tab: Anchor targets
for i, hit in enumerate(hits, 1):
    # Create anchor
    st.markdown(f"<a id='hit-{i}'></a>", unsafe_allow_html=True)
    
    # Show content
    with st.expander(f"#{i} · {hit['title']}"):
        st.markdown(hit["content"])

# Click [3] in Advice → jumps to #hit-3 in Evidence tab
```

---

## 11. Error Handling and Diagnostics

### Try-Except with User-Friendly Messages

```python
if submitted and q:
    try:
        # Spinner during backend call
        with st.spinner("Generating advice..."):
            res = cm_pipeline.answer(q, drug_name=drug, k=k)
    
    except Exception as e:
        # Show error to user
        st.error("Backend error occurred. See details below.")
        
        # Provide friendly hints
        hints = friendly_hints(lang, e)
        if hints:
            st.info("Troubleshooting tips:\n" + "\n".join(hints))
        
        # Show full traceback in expander (for developers)
        with st.expander("Error details"):
            st.exception(e)
        
        res = None

def friendly_hints(lang: str, exc: Exception) -> list[str]:
    """Convert technical errors to user-friendly hints."""
    msg = str(exc).lower()
    hints = []
    
    if "chromadb" in msg:
        hints.append(
            "· 检查 CHROMA_PERSIST_DIR / CHROMA_COLLECTION" if lang == "zh"
            else "· Check CHROMA_PERSIST_DIR / CHROMA_COLLECTION"
        )
    
    if "sqlite" in msg:
        hints.append(
            "· 检查 SQLite 路径与表结构" if lang == "zh"
            else "· Verify SQLite path & schema"
        )
    
    if "cuda" in msg or "cudnn" in msg:
        hints.append(
            "· 检查 CUDA/cuDNN 或切到 CPU" if lang == "zh"
            else "· Check CUDA/cuDNN or switch to CPU"
        )
    
    return hints
```

### Diagnostics Panel

Always-visible panel showing system status:

```python
def render_diagnostics(lang: str):
    """Show environment diagnostics (always visible)."""
    with st.expander("🔎 Diagnostics", expanded=False):
        # 1. VERSIONS
        st.write("**Python version**:", sys.version)
        st.write("**SQLite version**:", sqlite3.sqlite_version)
        st.write("**App version**:", VERSION)
        st.write("**Retriever version**:", R.VERSION)
        
        # 2. CONFIGURATION
        st.write("**Configuration (Secrets-first)**:")
        config = {
            "CHROMA_PERSIST_DIR": _env("CHROMA_PERSIST_DIR"),
            "CHROMA_COLLECTION": _env("CHROMA_COLLECTION"),
            "EMBEDDING_MODEL": _env("EMBEDDING_MODEL"),
        }
        st.json(config)
        
        # 3. CHROMA STATUS
        chroma_dir = config["CHROMA_PERSIST_DIR"]
        st.write(f"**Chroma dir exists**: {os.path.exists(chroma_dir)}")
        
        try:
            # Use retriever's safe methods (no extra client instantiation)
            collections = R.list_collections_safe()
            st.write("**Collections**:")
            st.json(collections)
            
            count = R.primary_collection_count()
            st.write(f"**Chunks in active collection**: {count}")
        except Exception as e:
            st.warning(f"Chroma access error: {e}")
        
        # 4. SQLITE STATUS
        db_path = config.get("DRUG_DB_PATH", "./db/drugs.sqlite")
        st.write(f"**SQLite file exists**: {os.path.exists(db_path)}")
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            st.write("**SQLite tables**:")
            st.json(tables)
        except Exception as e:
            st.warning(f"SQLite error: {e}")

# Render at bottom of page
render_diagnostics(lang)
```

**Benefits:**
- Instant visibility into configuration
- Confirms file paths are correct
- Shows version mismatches (local vs deployed)
- No need to check logs or terminal

### Validation Warnings

```python
if submitted:
    # Validate before calling backend
    if not q.strip():
        st.warning(t(lang, "warn_need_q"))
        # Stop here, don't call backend
    elif len(q) > 500:
        st.warning("Question too long (max 500 chars)")
    else:
        # Valid input, proceed
        res = cm_pipeline.answer(q, ...)
```

### Success Messages

```python
if submitted and res:
    st.success("✓ Generated successfully!")
    
    # Show elapsed time
    st.caption(f"⏱️ Elapsed: {elapsed:.2f}s")
```

---

## 12. Advanced Patterns

### Pattern 1: Secrets-Aware Configuration

Support both local `.env` and cloud `secrets.toml`:

```python
def _env(key: str, default: str = None) -> str:
    """
    Read environment variable with Secrets priority.
    
    Priority:
    1. st.secrets[key] (Streamlit Cloud secrets.toml)
    2. os.environ[key] (Local .env file)
    3. default value
    
    Usage:
        CHROMA_DIR = _env("CHROMA_PERSIST_DIR", "./chroma_store")
    """
    try:
        # Try Streamlit secrets first (cloud deployment)
        return st.secrets.get(key, os.getenv(key, default))
    except Exception:
        # Fall back to environment (local development)
        return os.getenv(key, default)

# Usage:
CHROMA_DIR = _env("CHROMA_PERSIST_DIR", "./chroma_store")
COLLECTION = _env("CHROMA_COLLECTION", "guideline_chunks")
```

**Why this matters:**
- Local: Uses `.env` file (git-ignored, developer-specific)
- Cloud: Uses `secrets.toml` (Streamlit Cloud, encrypted)
- Same code works everywhere

### Pattern 2: Backend Compatibility Detection

Support multiple backend versions without breaking:

```python
import inspect

# Detect if backend function accepts 'lang' parameter
sig_params = inspect.signature(cm_pipeline.answer).parameters

if "lang" in sig_params:
    # New backend: supports language parameter
    res = cm_pipeline.answer(q, drug_name=drug, k=k, lang=lang)
else:
    # Old backend: no language support yet
    res = cm_pipeline.answer(q, drug_name=drug, k=k)
```

**Benefit:** Frontend works with both old and new backend versions. No breaking changes.

### Pattern 3: Client-Side Filtering

Filter backend results in frontend (no roundtrip):

```python
# Get all hits from backend
hits = res.get("guideline_hits", [])

# Sidebar filters
src_filter = st.text_input("Filter by source")
year_min, year_max = st.slider("Year range", 2000, 2035, (2005, 2035))

# Filter function
def pass_filter(hit):
    meta = hit.get("meta", {})
    
    # Source filter
    src = meta.get("source", "").lower()
    src_ok = (src_filter.lower() in src) if src_filter else True
    
    # Year filter
    try:
        year = int(meta.get("year", 0))
        year_ok = year_min <= year <= year_max
    except:
        year_ok = True  # Keep if year missing/invalid
    
    return src_ok and year_ok

# Apply filter
filtered_hits = [h for h in hits if pass_filter(h)]

# Display
st.write(f"Showing {len(filtered_hits)} of {len(hits)} hits")
for hit in filtered_hits:
    st.markdown(hit["content"])
```

**Benefits:**
- Instant feedback (no backend call)
- User controls precision/recall tradeoff
- Backend focuses on relevance, frontend on filtering

### Pattern 4: Download Buttons

```python
# Generate downloadable content
advice_md = res["output"]
evidence_md = evidence_md(lang, res["hits"])

# Side-by-side buttons
col1, col2, _ = st.columns([1, 1, 4])

with col1:
    st.download_button(
        label="📄 Export Advice",
        data=advice_md.encode("utf-8"),  # Must be bytes
        file_name="caremind_advice.md",
        mime="text/markdown",
        use_container_width=True,
        disabled=not bool(advice_md.strip())  # Disable if empty
    )

with col2:
    st.download_button(
        label="📚 Export Evidence",
        data=evidence_md.encode("utf-8"),
        file_name="caremind_evidence.md",
        mime="text/markdown",
        use_container_width=True
    )
```

**Key points:**
- `data` must be bytes: `.encode("utf-8")`
- `disabled=True` grays out button
- `use_container_width=True` makes button full-width
- No backend call needed - pure client-side download

### Pattern 5: Expander with Custom Styling

```python
# Render metadata with custom badges
for i, hit in enumerate(hits, 1):
    meta = hit["meta"]
    title = meta.get("title", "Untitled")
    source = meta.get("source", "Unknown")
    year = meta.get("year", "—")
    
    with st.expander(f"#{i} · {title[:60]}", expanded=False):
        # Custom-styled metadata
        st.markdown(
            f"<div class='cm-muted'>"
            f"<span class='cm-badge'>Source: {source}</span>"
            f"<span class='cm-badge'>Year: {year}</span>"
            f"</div>",
            unsafe_allow_html=True
        )
        
        # Content
        st.markdown(hit["content"])
```

**Result:** Professional-looking metadata badges instead of plain text.

---

## Summary: Key Takeaways

### Execution Model
- **Every widget interaction triggers a full script rerun**
- Use `@st.cache_resource` for expensive operations
- Use `st.session_state` to persist data across reruns

### Core Components
- `st.set_page_config()` - Must be first st.* call
- `with st.sidebar:` - Fixed left column for settings
- `with st.form():` - Batch input, only rerun on submit
- `st.tabs()` - Multi-view layout
- `st.expander()` - Collapsible sections

### State Management
- `st.session_state["key"]` - Persist data across reruns
- `.setdefault()` - Initialize if not exists
- `.pop()` - Read and remove (one-time use)

### Caching Strategy
- `@st.cache_resource` - Models, DB connections, global resources
- `@st.cache_data` - DataFrames, lists, dicts (serializable data)
- Don't cache: User inputs, dynamic queries, non-thread-safe objects

### i18n Pattern
- Dictionary with language keys: `I18N[lang][key]`
- Helper function: `t(lang, "key")`
- Pass `lang` throughout UI and to backend

### Best Practices
- Validate user input before calling backend
- Show friendly error messages with hints
- Provide diagnostics panel for troubleshooting
- Use anchor links for navigation
- Client-side filtering for instant feedback
- Download buttons for export functionality

---

## Next Steps

### Explore CareMind Code
- **app.py (lines 1-50)**: Imports and helpers
- **app.py (lines 50-80)**: Sidebar diagnostics
- **app.py (lines 247-314)**: Sidebar controls
- **app.py (lines 319-338)**: Form input
- **app.py (lines 345-519)**: Tab rendering
- **rag/retriever.py (lines 279-291)**: Caching pattern

### Experiment
1. Clone the CareMind repo
2. Run `streamlit run app.py`
3. Modify `I18N` dict to add a new language
4. Add a new tab with custom content
5. Create your own `@st.cache_resource` function

### Further Reading
- **Official Streamlit Docs**: https://docs.streamlit.io/
- **Caching Guide**: https://docs.streamlit.io/develop/concepts/architecture/caching
- **Session State**: https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state
- **CareMind Streamlit Patterns**: [../technical/streamlit/streamlit_patterns.md](../technical/streamlit/streamlit_patterns.md)

---

## Quick Reference

### Common Commands
```python
# Page setup (must be first)
st.set_page_config(title="App", layout="wide")

# Text elements
st.title("Title")
st.header("Header")
st.subheader("Subheader")
st.write("Text")
st.markdown("**Bold**")
st.caption("Small text")

# Input widgets
text = st.text_input("Label", value="default")
num = st.number_input("Label", value=0)
choice = st.selectbox("Label", ["A", "B"])
slider = st.slider("Label", 0, 10, 5)
toggle = st.toggle("Label", value=True)
button = st.button("Label")

# Layout
with st.sidebar:
    st.write("Sidebar content")

col1, col2 = st.columns(2)
with col1:
    st.write("Column 1")

tab1, tab2 = st.tabs(["Tab 1", "Tab 2"])
with tab1:
    st.write("Tab 1 content")

with st.expander("Click to expand"):
    st.write("Hidden content")

# Forms (batch input)
with st.form("my_form"):
    text = st.text_input("Input")
    submitted = st.form_submit_button("Submit")

if submitted:
    st.write(f"You entered: {text}")

# Status messages
st.success("Success!")
st.info("Info")
st.warning("Warning")
st.error("Error")

with st.spinner("Loading..."):
    # Long operation
    time.sleep(2)

# Session state
if "count" not in st.session_state:
    st.session_state.count = 0

st.session_state.count += 1

# Caching
@st.cache_resource
def expensive_function():
    return load_big_model()

# Downloads
st.download_button(
    "Download",
    data=content.encode("utf-8"),
    file_name="file.txt"
)
```

---

**Tutorial Version**: 2025-12-21  
**License**: MIT
