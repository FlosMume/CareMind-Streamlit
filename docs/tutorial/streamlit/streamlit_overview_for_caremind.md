# Streamlit Tutorial for CareMind

A comprehensive guide to understanding **Streamlit** and how it powers the CareMind clinical decision support system.

---

## Table of Contents

1. [What is Streamlit?](#what-is-streamlit)
2. [Why Streamlit for CareMind?](#why-streamlit-for-caremind)
3. [Core Concepts](#core-concepts)
4. [Key Features Used in CareMind](#key-features-used-in-caremind)
5. [How to Run CareMind](#how-to-run-caremind)
6. [Advanced Patterns in CareMind](#advanced-patterns-in-caremind)
7. [Common Issues & Troubleshooting](#common-issues--troubleshooting)
8. [Best Practices for CareMind](#best-practices-for-caremind)

---

## What is Streamlit?

**Streamlit** is a Python framework for building interactive web applications with **zero HTML/CSS/JavaScript knowledge required**.

### Key Characteristics

| Aspect | Details |
|--------|---------|
| **Language** | Pure Python |
| **Rendering** | Server-side (Python controls UI) |
| **Execution Model** | Script reruns from top to bottom on every user interaction |
| **Use Cases** | Data apps, dashboards, ML demos, clinical tools |
| **Deployment** | Streamlit Cloud, Heroku, Docker, custom servers |

### Hello World Example

```python
import streamlit as st

st.title("My App")
name = st.text_input("Enter your name:")
if name:
    st.write(f"Hello, {name}!")
```

Run it:
```bash
streamlit run app.py
```

Browser opens at `http://localhost:8501`

---

## Why Streamlit for CareMind?

CareMind uses Streamlit because:

1. **Fast Prototyping**: Clinical UI built in days, not weeks
2. **Pure Python**: Medical logic (RAG, LLM, database queries) stays in Python
3. **Bilingual Support**: Easy to build Chinese/English UI with `st.selectbox()`, `st.write()`
4. **State Management**: `st.session_state` handles conversation history naturally
5. **Caching**: `@st.cache_resource` solves the "Chroma singleton" problem
6. **Deployment**: Streamlit Cloud handles production hosting

---

## Core Concepts

### 1. Script Reruns

**The Fundamental Principle:**

Every time a user interacts with the app (button click, text input, slider change), **Streamlit reruns the entire script from top to bottom**.

```python
import streamlit as st

x = 5  # ← Recomputed on every interaction
st.write(f"x is {x}")

if st.button("Increment"):
    x = x + 1  # ← This does NOT persist!
    st.write(f"x is now {x}")
```

**Problem:** Clicking the button shows `x is now 6`, but the next interaction shows `x is 5` again.

**Solution:** Use `st.session_state`

```python
import streamlit as st

if "x" not in st.session_state:
    st.session_state.x = 5

st.write(f"x is {st.session_state.x}")

if st.button("Increment"):
    st.session_state.x += 1
    st.write(f"x is now {st.session_state.x}")
```

### 2. Session State (`st.session_state`)

**What it is:** A dictionary that **persists across script reruns within one user session**.

**CareMind Usage** ([app.py](app.py#L150-L200)):
```python
# Initialize conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Add user question
if st.button("Ask"):
    user_input = st.text_input("Your question:")
    st.session_state.messages.append({"role": "user", "text": user_input})

# Display history
for msg in st.session_state.messages:
    st.write(msg["text"])
```

**Key Points:**
- `st.session_state` is **per user**, not global
- Persists until user closes browser or session times out
- Dict-like access: `st.session_state.key` or `st.session_state["key"]`

### 3. Caching (`@st.cache_resource` and `@st.cache_data`)

**Problem:** Streamlit reruns trigger expensive operations repeatedly.

```python
# ❌ Without caching: Loads model on EVERY rerun
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
embedding = model.encode("Hello")
```

**Solution:** Use `@st.cache_resource` for expensive **one-time** operations.

```python
import streamlit as st
from sentence_transformers import SentenceTransformer

@st.cache_resource
def load_embedding_model():
    """Load once per process, reuse across reruns."""
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_embedding_model()
embedding = model.encode("Hello")  # ← Fast on reruns
```

**CareMind Usage** ([retriever.py](retriever.py#L278)):
```python
@cache_resource
def get_chroma_client(pdir: str, settings: Optional[Settings] = None):
    """Get a cached Chroma client (singleton pattern)."""
    return PersistentClient(path=pdir, settings=settings)
```

**Difference: `@cache_resource` vs `@cache_data`**

| Decorator | Use Case | Scope |
|-----------|----------|-------|
| `@st.cache_resource` | DB connections, ML models, Chroma clients | **One per process** (shared across users) |
| `@st.cache_data` | Data transformations, file reads, API calls | **One per user session** |

### 4. Sessions vs Processes

**Session (User Level):**
- One browser connection = one session
- User closes browser → session ends
- Data stored in `st.session_state`
- **Isolated per user**

**Process (Server Level):**
- One Python interpreter running the app
- Multiple sessions can share one process (e.g., on your laptop)
- Data cached with `@st.cache_resource`
- **Shared across users**

**On Streamlit Cloud:**
- Each user gets a **separate process**
- So `@st.cache_resource` effectively acts like per-user caching

**On Streamlit Community Cloud (public):**
- Multiple users may share one process
- `@st.cache_resource` is truly shared

**CareMind Production Implication:**
```python
# This runs ONCE per process, shared across users
@cache_resource
def get_chroma_client(...):
    return PersistentClient(path=pdir, settings=Settings(...))

# This runs per user, isolated
@st.cache_data
def retrieve_guidelines(query: str):
    client = get_chroma_client(...)
    return client.query(query)
```

---

## Key Features Used in CareMind

### 1. UI Components

**Text Input:**
```python
question = st.text_input("Clinical Question:", placeholder="e.g., β-blockers in asthma?")
```

**Buttons:**
```python
if st.button("Ask Question"):
    answer = retrieve_and_answer(question)
    st.write(answer)
```

**Tabs:**
```python
tab1, tab2, tab3 = st.tabs(["Answer", "Evidence", "Debug"])
with tab1:
    st.write("Main answer")
with tab2:
    st.write("Retrieved evidence snippets")
with tab3:
    st.write("Execution logs")
```

**Expandable Sections:**
```python
with st.expander("🔎 Environment Diagnostics"):
    st.write(f"Python: {sys.version}")
    st.write(f"Chroma: {__version__}")
```

**Download Button:**
```python
st.download_button(
    label="Download Report",
    data=report_content,
    file_name="recommendation.md",
    mime="text/markdown"
)
```

### 2. Sidebar

Sidebar components don't trigger reruns immediately; they behave like form inputs:

```python
with st.sidebar:
    language = st.selectbox("Language / 语言", ["English", "中文"])
    model_name = st.text_input("Model Name", value="gpt-4")
    
    if st.button("Reset"):
        st.session_state.messages = []
        st.rerun()  # Force rerun
```

### 3. Columns & Layout

```python
col1, col2 = st.columns(2)

with col1:
    st.write("Left side")
    
with col2:
    st.write("Right side")
```

### 4. Forms (Prevent Multiple Submissions)

```python
with st.form("ask_form"):
    question = st.text_input("Your question:")
    submitted = st.form_submit_button("Submit")
    
    if submitted:
        # Process question
        # ← Only runs once per form submission
```

---

## How to Run CareMind

### Prerequisites

```bash
cd /home/myunix/dev/Python/caremind-streamlit-dec

# Python 3.9+
python --version

# Install dependencies
pip install -r requirements.txt
```

### Run Locally

```bash
streamlit run app.py
```

**Output:**
```
  You can now view your Streamlit app in your browser.

  URL: http://localhost:8501
```

Open browser → `http://localhost:8501`

### Configuration

Streamlit config is in `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[client]
showErrorDetails = true
```

### Environment Variables

Set in `.env` file or export:

```bash
# Disable Chroma telemetry (privacy)
export CHROMA_TELEMETRY_OFF=1

# LLM API key
export OPENAI_API_KEY=sk-...

# Custom Chroma path
export CHROMA_DB_PATH=./chroma_store
```

---

## Advanced Patterns in CareMind

### 1. The Chroma Singleton Pattern

**Problem:** Streamlit reruns create multiple Chroma clients → "different settings" error.

**Solution:** Cached resource decorator ensures one client per process.

```python
@cache_resource
def get_chroma_client(pdir: str, settings: Optional[Settings] = None) -> PersistentClient:
    """
    Get or create a Chroma client (singleton per process).
    
    - Uses cache_resource to persist across reruns
    - Settings must be identical to avoid RuntimeError
    """
    return PersistentClient(
        path=pdir,
        settings=settings or Settings(
            anonymized_telemetry=not CHROMA_TELEMETRY_OFF,
            allow_reset=False,
        ),
    )
```

**CareMind Usage** ([retriever.py](retriever.py#L298)):
```python
client = get_chroma_client(path=CHROMA_PATH, settings=None)
results = client.get_or_create_collection(name="guidelines")
```

### 2. SQLite Version Shimming

**Problem:** Streamlit Cloud has SQLite 3.8 (from 2013); Chroma needs 3.35+ (window functions, RETURNING clause).

**Solution:** Detect and alias modern SQLite.

```python
# From retriever.py (lines 100-120)
def _ensure_sqlite() -> None:
    MIN = (3, 35, 0)
    try:
        import sqlite3 as _stdlib
        ver_tuple = tuple(map(int, _stdlib.sqlite_version.split(".")))
        if ver_tuple >= MIN:
            return  # ✅ stdlib is new enough
        # ❌ Too old → alias pysqlite3
        import pysqlite3 as _py
        sys.modules["sqlite3"] = _py
    except Exception:
        import pysqlite3 as _py
        sys.modules["sqlite3"] = _py
```

**Why This Matters:**
- Without the shim, Chroma crashes with `sqlite3.OperationalError: near "RETURNING": syntax error`
- Must run **before** importing Chroma

### 3. Lazy Model Loading

**Problem:** Embedding models (SentenceTransformer) are slow to load; don't load on import.

**Solution:** Use lazy loading with `@cache_resource`.

```python
class _LazyEmbedder:
    """Load SentenceTransformer model on first call, not on import."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
    
    def get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model
    
    def encode(self, text: str) -> list:
        return self.get_model().encode(text)

@cache_resource
def get_embedder() -> _LazyEmbedder:
    return _LazyEmbedder()
```

### 4. Bilingual UI

**Pattern:** Use language state to conditionally render Chinese or English.

```python
if "language" not in st.session_state:
    st.session_state.language = "English"

with st.sidebar:
    st.session_state.language = st.selectbox(
        "Language / 语言",
        ["English", "中文"]
    )

# Conditional rendering
if st.session_state.language == "English":
    st.title("CareMind · Clinical Decision Support")
    prompt = st.text_input("Your clinical question:")
else:
    st.title("CareMind · 临床决策支持")
    prompt = st.text_input("您的临床问题：")
```

---

## Common Issues & Troubleshooting

### Issue 1: "An instance of Chroma already exists...with different settings"

**Cause:** Multiple Chroma clients created in same process.

**Solution:** Always use cached client:

```python
# ✅ Correct
@cache_resource
def get_chroma_client():
    return PersistentClient(path=...)

client = get_chroma_client()

# ❌ Wrong
from chromadb import PersistentClient
client = PersistentClient(path=...)  # New instance on every rerun!
```

### Issue 2: "near 'RETURNING': syntax error"

**Cause:** Outdated SQLite on Streamlit Cloud.

**Solution:** Ensure `pysqlite3` is in `requirements.txt` and `_ensure_sqlite()` is called early.

```python
# Add to requirements.txt
pysqlite3-binary==0.5.1

# In app.py (before importing chroma):
from rag.retriever import _ensure_sqlite
_ensure_sqlite()
```

### Issue 3: Model loads every rerun (slow)

**Cause:** Not using `@st.cache_resource`.

**Solution:** Cache the model:

```python
@st.cache_resource
def load_model(model_name: str):
    return SentenceTransformer(model_name)

model = load_model("all-MiniLM-L6-v2")
```

### Issue 4: Conversation history disappears on reload

**Cause:** Not storing in `st.session_state`.

**Solution:**

```python
# ✅ Correct
if "messages" not in st.session_state:
    st.session_state.messages = []

# Add message
st.session_state.messages.append({"role": "user", "text": user_input})

# ❌ Wrong
messages = []  # Lost on rerun!
messages.append(...)
```

### Issue 5: "RuntimeError: Missing permission to clear cache"

**Cause:** Trying to clear cache that's locked.

**Solution:** Use the API correctly:

```python
# ✅ Correct
if st.button("Clear All Caches"):
    st.cache_resource.clear()
    st.cache_data.clear()

# ❌ Wrong
st.cache_resource.clear()  # No await/try-except
```

---

## Best Practices for CareMind

### 1. Initialize Session State Early

```python
# ✅ At the top of app.py
DEFAULT_STATE = {
    "messages": [],
    "language": "English",
    "model_name": "gpt-4",
}

for key, default in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = default
```

### 2. Use `st.form()` for Multi-Step Inputs

```python
# ✅ Prevents duplicate submissions
with st.form("question_form"):
    question = st.text_input("Your question:")
    use_evidence = st.checkbox("Include evidence snippets")
    submitted = st.form_submit_button("Ask")
    
    if submitted:
        process_question(question, use_evidence)
```

### 3. Use `@st.cache_resource` Only for True Singletons

```python
# ✅ Use for: DB connections, ML models, Chroma clients
@st.cache_resource
def get_chroma_client():
    return PersistentClient(path=...)

# ❌ Don't use for: Data that changes, user-specific results
@st.cache_resource
def search_guidelines(query: str):  # WRONG!
    ...
```

### 4. Use `@st.cache_data` for Expensive Data Operations

```python
# ✅ Cache file reads, API calls, data transforms
@st.cache_data
def load_drug_database():
    """Load drug data once per user session."""
    import sqlite3
    conn = sqlite3.connect("drug_db.sqlite")
    return pd.read_sql("SELECT * FROM drugs", conn)

drugs_df = load_drug_database()
```

### 5. Always Check Secrets First

```python
# ✅ Correct (use Streamlit Secrets for production)
openai_api_key = st.secrets.get("openai_api_key")
if not openai_api_key:
    openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error("❌ No OpenAI API key configured!")
    st.stop()
```

### 6. Use Structured Logging

```python
# ✅ Log to debug tab
with st.expander("🔧 Debug Logs"):
    st.text(f"Query: {question}")
    st.text(f"Retrieved {len(results)} snippets")
    st.text(f"LLM response time: {elapsed:.2f}s")
```

### 7. Cache Resource State Management

```python
# ✅ Clear caches only when necessary
if st.sidebar.button("🔄 Reset All"):
    st.cache_resource.clear()
    st.cache_data.clear()
    st.session_state.clear()
    st.rerun()
```

### 8. Bilingual Design Pattern

```python
# ✅ Define translations once
STRINGS = {
    "English": {
        "title": "CareMind · Clinical Decision Support",
        "ask": "Your clinical question:",
        "submit": "Ask",
    },
    "中文": {
        "title": "CareMind · 临床决策支持",
        "ask": "您的临床问题：",
        "submit": "提问",
    },
}

lang = st.selectbox("Language", list(STRINGS.keys()))
st.title(STRINGS[lang]["title"])
```

---

## Deployment

### Local Development
```bash
streamlit run app.py
```

### Streamlit Cloud
1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app" → select repo
4. Configure secrets in Settings

### Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "app.py"]
```

Run:
```bash
docker build -t caremind .
docker run -p 8501:8501 caremind
```

---

## Summary

| Concept | Usage in CareMind |
|---------|-------------------|
| **Script Reruns** | Every user action reruns entire `app.py` |
| **Session State** | Stores conversation history, language preference |
| **Cache Resource** | Chroma client singleton, embedding models |
| **Cache Data** | Drug database, guideline queries |
| **Forms** | Multi-step question input with validation |
| **Tabs** | Evidence, recommendations, debug logs |
| **Sidebar** | Settings, language, reset button |

---

## Further Reading

- [Streamlit Docs](https://docs.streamlit.io)
- [Caching](https://docs.streamlit.io/develop/concepts/execution-model/caching)
- [Session State](https://docs.streamlit.io/develop/concepts/design/caching)
- [API Reference](https://docs.streamlit.io/develop/api-reference)

---

*Last updated: December 13, 2025*
