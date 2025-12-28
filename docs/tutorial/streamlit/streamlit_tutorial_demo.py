"""
Streamlit Tutorial Demo for CareMind
====================================

This script demonstrates key Streamlit concepts used in CareMind:
- Session state management
- Caching (cache_resource, cache_data)
- Forms and input validation
- Tabs and expandable sections
- Bilingual UI
- State persistence across reruns

Run with:
    streamlit run streamlit_tutorial_demo.py
"""

import streamlit as st
import time
from datetime import datetime

# =============================================================================
# 1. INITIALIZE SESSION STATE (run once per user session)
# =============================================================================

def init_session_state():
    """Initialize session state variables that persist across reruns."""
    DEFAULT_STATE = {
        "language": "English",
        "messages": [],
        "message_count": 0,
        "model_loaded": False,
        "theme_color": "blue",
    }
    
    for key, default_value in DEFAULT_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

init_session_state()

# =============================================================================
# 2. CACHING EXAMPLES
# =============================================================================

@st.cache_resource
def load_expensive_model(model_name: str = "bert-base"):
    """
    Simulates loading a large ML model.
    
    @cache_resource ensures this runs ONCE per process, not per rerun.
    Subsequent calls use cached result.
    """
    st.write(f"⏳ Loading {model_name}...")
    time.sleep(2)  # Simulate slow load
    st.success(f"✅ {model_name} loaded!")
    return {"model": model_name, "loaded_at": datetime.now()}

@st.cache_data
def fetch_database_sample(limit: int = 100):
    """
    Simulates fetching data from a database.
    
    @cache_data caches per user session.
    Different users may have separate cache entries.
    """
    st.write(f"📚 Fetching {limit} records from database...")
    time.sleep(1)  # Simulate slow query
    
    sample_data = [
        {"id": i, "drug": f"Drug_{i}", "indication": f"Condition_{i}"}
        for i in range(1, limit + 1)
    ]
    return sample_data

# =============================================================================
# 3. TRANSLATION STRINGS (Bilingual Support)
# =============================================================================

STRINGS = {
    "English": {
        "title": "Streamlit Tutorial Demo",
        "subtitle": "Learn Core Concepts with Interactive Examples",
        "sidebar_title": "⚙️ Settings",
        "language_label": "Language",
        "theme_label": "Theme Color",
        
        # Session State Section
        "session_title": "📝 Session State: Conversation History",
        "session_desc": "Messages persist across reruns (script restarts). Try clicking buttons multiple times!",
        "your_message": "Your message:",
        "add_button": "Add Message",
        "clear_button": "Clear History",
        "no_messages": "No messages yet. Add one!",
        
        # Caching Section
        "cache_title": "⚡ Caching: Load Models & Data",
        "cache_desc": "Click buttons below. First click is slow (loading), subsequent clicks are fast (cached).",
        "load_model_btn": "Load Model (cache_resource)",
        "fetch_data_btn": "Fetch Data (cache_data)",
        "model_info": "Model Info:",
        
        # Forms Section
        "form_title": "📋 Forms: Multi-Step Input",
        "form_desc": "Forms prevent duplicate submissions and validate input.",
        "clinical_question": "Clinical Question:",
        "use_evidence": "Include Evidence Snippets",
        "language_override": "Response Language:",
        "submit_question": "Submit Question",
        "question_answered": "Question received and processed!",
        
        # Tabs Section
        "tabs_title": "📑 Tabs: Organize Content",
        "tab_answer": "Answer",
        "tab_evidence": "Evidence",
        "tab_debug": "Debug Logs",
        
        # Advanced Features
        "advanced_title": "🔧 Advanced: Control Flow",
        "stop_example": "This button demonstrates st.stop()",
        "stop_button": "Click to Stop Execution",
        "rerun_button": "🔄 Rerun Script",
        "execution_stopped": "Execution stopped here! (st.stop() called)",
    },
    
    "中文": {
        "title": "Streamlit 教程演示",
        "subtitle": "通过交互示例学习核心概念",
        "sidebar_title": "⚙️ 设置",
        "language_label": "语言",
        "theme_label": "主题颜色",
        
        # Session State Section
        "session_title": "📝 会话状态：对话历史",
        "session_desc": "消息在脚本重新运行时保留。尝试多次点击按钮！",
        "your_message": "您的消息：",
        "add_button": "添加消息",
        "clear_button": "清除历史",
        "no_messages": "还没有消息。添加一条！",
        
        # Caching Section
        "cache_title": "⚡ 缓存：加载模型和数据",
        "cache_desc": "点击下面的按钮。第一次点击很慢（加载中），随后的点击很快（已缓存）。",
        "load_model_btn": "加载模型 (cache_resource)",
        "fetch_data_btn": "获取数据 (cache_data)",
        "model_info": "模型信息：",
        
        # Forms Section
        "form_title": "📋 表单：多步骤输入",
        "form_desc": "表单可以防止重复提交并验证输入。",
        "clinical_question": "临床问题：",
        "use_evidence": "包含证据片段",
        "language_override": "响应语言：",
        "submit_question": "提交问题",
        "question_answered": "问题已收到并处理！",
        
        # Tabs Section
        "tabs_title": "📑 选项卡：组织内容",
        "tab_answer": "答案",
        "tab_evidence": "证据",
        "tab_debug": "调试日志",
        
        # Advanced Features
        "advanced_title": "🔧 高级：控制流",
        "stop_example": "此按钮演示了 st.stop()",
        "stop_button": "点击以停止执行",
        "rerun_button": "🔄 重新运行脚本",
        "execution_stopped": "执行在此停止！（调用了 st.stop()）",
    }
}

# Get current language
lang = st.session_state.language
S = STRINGS[lang]  # Shorthand for current language strings

# =============================================================================
# 4. SIDEBAR: SETTINGS & CONFIGURATION
# =============================================================================

with st.sidebar:
    st.header(S["sidebar_title"])
    
    # Language selector
    st.session_state.language = st.selectbox(
        S["language_label"],
        ["English", "中文"],
        index=0 if st.session_state.language == "English" else 1
    )
    
    # Theme selector
    st.session_state.theme_color = st.selectbox(
        S["theme_label"],
        ["blue", "red", "green", "orange"],
        index=["blue", "red", "green", "orange"].index(st.session_state.theme_color)
    )
    
    st.divider()
    
    # Stats
    st.subheader("📊 Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Messages", st.session_state.message_count)
    with col2:
        st.metric("Sessions", "1")
    
    # Reset button
    if st.button("🗑️ Clear All Data"):
        st.session_state.clear()
        init_session_state()
        st.rerun()

# =============================================================================
# 5. MAIN CONTENT
# =============================================================================

st.title(S["title"])
st.markdown(f"__{S['subtitle']}__")

# Display current settings in a nice box
with st.expander("ℹ️ Current Configuration"):
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Language", st.session_state.language)
    with col2:
        st.metric("Theme", st.session_state.theme_color)

st.divider()

# =========================================================================
# SECTION 1: SESSION STATE
# =========================================================================

st.header(S["session_title"])
st.info(S["session_desc"])

col1, col2 = st.columns([3, 1])

with col1:
    user_message = st.text_input(S["your_message"], key="message_input")

with col2:
    add_btn = st.button(S["add_button"], use_container_width=True)

if add_btn and user_message:
    st.session_state.messages.append({
        "text": user_message,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "language": st.session_state.language,
    })
    st.session_state.message_count += 1
    st.rerun()

# Display messages
if st.session_state.messages:
    st.subheader("💬 Message History")
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message("user"):
            st.write(f"**{msg['text']}** _(_{msg['timestamp']}_)_")
else:
    st.warning(S["no_messages"])

# Clear history button
if st.button(S["clear_button"]):
    st.session_state.messages = []
    st.session_state.message_count = 0
    st.rerun()

st.divider()

# =========================================================================
# SECTION 2: CACHING
# =========================================================================

st.header(S["cache_title"])
st.info(S["cache_desc"])

col1, col2 = st.columns(2)

with col1:
    if st.button(S["load_model_btn"]):
        model_info = load_expensive_model("SentenceTransformer-MiniLM")
        st.subheader(S["model_info"])
        st.json(model_info)

with col2:
    if st.button(S["fetch_data_btn"]):
        data = fetch_database_sample(50)
        st.subheader("📄 Sample Data")
        st.dataframe(data, use_container_width=True)

# Show cache status
with st.expander("🔍 View Cache Metrics"):
    st.write("**@st.cache_resource**: One instance per process, shared across reruns")
    st.write("**@st.cache_data**: One cache per user session")
    st.code("""
@st.cache_resource
def load_expensive_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data
def fetch_database():
    return pandas.read_csv("data.csv")
    """, language="python")

st.divider()

# =========================================================================
# SECTION 3: FORMS
# =========================================================================

st.header(S["form_title"])
st.info(S["form_desc"])

with st.form("question_form", border=True):
    st.subheader("Clinical Question Submission")
    
    question = st.text_area(S["clinical_question"], max_chars=200)
    use_evidence = st.checkbox(S["use_evidence"], value=True)
    response_lang = st.selectbox(S["language_override"], ["English", "中文"])
    
    submitted = st.form_submit_button(S["submit_question"], use_container_width=True)

if submitted and question:
    st.success(S["question_answered"])
    st.write(f"**Question**: {question}")
    st.write(f"**Include Evidence**: {'Yes' if use_evidence else 'No'}")
    st.write(f"**Response Language**: {response_lang}")

st.divider()

# =========================================================================
# SECTION 4: TABS
# =========================================================================

st.header(S["tabs_title"])

tab1, tab2, tab3 = st.tabs([S["tab_answer"], S["tab_evidence"], S["tab_debug"]])

with tab1:
    st.write("""
    ### Sample Clinical Answer
    
    **Q**: Can β-blockers be used in hypertensive patients with bronchial asthma?
    
    **A**: Generally **contraindicated** due to risk of bronchospasm. 
    Consider alternatives like calcium channel blockers or ACE inhibitors.
    """)

with tab2:
    st.write("""
    ### Retrieved Evidence
    
    **Source**: Clinical Guidelines 2024
    
    - β-blockers can precipitate asthma attacks
    - Selective β1-blockers may be safer but still require caution
    - Alternative: Use calcium channel blockers (e.g., nifedipine)
    """)

with tab3:
    st.code("""
# Debug Logs
[2025-12-13 10:30:15] Query received: "Can β-blockers..."
[2025-12-13 10:30:16] Retrieved 3 guideline chunks
[2025-12-13 10:30:17] LLM response generated in 1.2s
[2025-12-13 10:30:18] Response sent to user
    """, language="log")

st.divider()

# =========================================================================
# SECTION 5: ADVANCED CONTROL FLOW
# =========================================================================

st.header(S["advanced_title"])

col1, col2 = st.columns(2)

with col1:
    st.write(S["stop_example"])
    if st.button(S["stop_button"], type="primary"):
        st.warning(S["execution_stopped"])
        st.stop()  # Stops execution here

with col2:
    if st.button(S["rerun_button"], type="secondary"):
        st.rerun()  # Force full rerun

st.divider()

# =========================================================================
# FOOTER
# =========================================================================

st.markdown("""
---
### 📚 Key Takeaways

1. **Script Reruns**: Streamlit reruns entire script on every user interaction
2. **Session State**: Use `st.session_state` to persist data across reruns
3. **Caching**: Use `@cache_resource` for singletons, `@cache_data` for per-user data
4. **Forms**: Use `st.form()` to group inputs and prevent duplicate submissions
5. **Organization**: Use tabs, expanders, and columns for clean UI
6. **Bilingual**: Dictionary-based translations for multi-language apps
7. **Control Flow**: Use `st.stop()` and `st.rerun()` to control execution

### 🔗 Useful Links

- [Streamlit Docs](https://docs.streamlit.io)
- [Session State API](https://docs.streamlit.io/develop/api-reference/session-state)
- [Caching](https://docs.streamlit.io/develop/concepts/execution-model/caching)
- [CareMind Docs](../README.md)
""")

st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
