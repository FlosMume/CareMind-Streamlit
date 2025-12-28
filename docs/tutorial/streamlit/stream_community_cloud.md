# Deploying CareMind on Streamlit Community Cloud

This README walks through deploying this CareMind Streamlit app to **Streamlit Community Cloud**.

## What you will deploy

- **Streamlit app entrypoint:** `app.py`
- **Python version (Cloud):** from `runtime.txt` (this repo pins `python-3.10.18`)
- **Dependencies:** `requirements.txt`
- **Runtime config (recommended):** Streamlit Cloud **Secrets** (maps to `.streamlit/secrets.toml`)

## Before you start

- A GitHub account
- Access to this repository (or a fork)
- Streamlit Community Cloud account (https://share.streamlit.io)

## 1) Fork / prepare the repo

1. Fork this repo to your GitHub account (recommended if you’ll customize settings).
2. Confirm these files exist at the repository root:
   - `app.py`
   - `requirements.txt`
   - `runtime.txt`
3. Confirm the app’s data folders exist in the repo (or will be created at runtime):
   - `chroma_store/` (Chroma persistence directory)
   - `db/` (SQLite databases)

Note: this repo’s main `readme.md` describes two deployment modes:
- `main` branch: minimal demo
- `demo-data` branch: includes demo data (SQLite + Chroma index)

If your Cloud deployment needs the bundled demo data, deploy the `demo-data` branch.

## 2) Create the app on Streamlit Community Cloud

1. Go to Streamlit Community Cloud and click **New app**.
2. Select:
   - **Repository:** your fork (or the upstream repo)
   - **Branch:** `main` (or `demo-data` if you need bundled data)
   - **Main file path:** `app.py`
3. Click **Deploy**.

Streamlit Cloud will:
- Create a clean environment
- Install dependencies from `requirements.txt`
- Use the Python version from `runtime.txt`
- Start the app by running `streamlit run app.py`

## 3) Configure Secrets (recommended)

This project reads configuration via environment variables and/or Streamlit secrets.

### 3.1 Cloud UI: set Secrets

In your Streamlit Cloud app settings, open **Secrets** and paste a TOML block.

Start with the same keys used locally in `.streamlit/secrets.toml`:

```toml
CHROMA_PERSIST_DIR = "./chroma_store"
CHROMA_COLLECTION  = "guideline_chunks_v2"
EMBEDDING_MODEL    = "BAAI/bge-large-zh-v1.5"
DRUG_DB_PATH       = "./db/drugs.sqlite"
CAREMIND_DEMO      = "0"
```

### 3.2 Notes on Secrets

- **Do not commit real API keys** into the repo.
- Streamlit Cloud secrets are injected at runtime and are not visible to other users.
- In code, you can typically access secrets via `st.secrets[...]` (Streamlit-managed), while this app also uses environment variables (via `dotenv`).

## 4) Data & persistence notes (Chroma + SQLite)

### 4.1 SQLite

The app expects a SQLite file at:
- `DRUG_DB_PATH` (default: `./db/drugs.sqlite`)

If you deploy a branch that doesn’t include the DB, the app won’t be able to answer drug lookups.

### 4.2 Chroma persistence

The app expects Chroma data under:
- `CHROMA_PERSIST_DIR` (default: `./chroma_store`)

Streamlit Community Cloud deployments are typically **stateless** across rebuilds/restarts.
- If the Chroma directory is included in the repo (e.g., `demo-data` branch), it will be available at runtime.
- If it must be generated dynamically, you’ll need a build step (or runtime ingest) and should expect slower cold starts.

## 5) Common deployment issues & fixes

### 5.1 sqlite3 / Chroma errors on Cloud

This repo includes a Cloud-oriented SQLite compatibility shim:
- `pysqlite3-binary` is pinned in `requirements.txt`
- `app.py` attempts to swap `sqlite3` with `pysqlite3` at startup

If you see SQLite/Chroma errors on Cloud:
- Keep `pysqlite3-binary` in `requirements.txt`
- Redeploy to ensure the new environment is rebuilt

### 5.2 PyTorch / Transformers on Cloud

- Cloud environments usually do **CPU-only**.
- `requirements.txt` pins CPU-only PyTorch (`torch==2.6.0`) via the PyTorch CPU wheel index.

If deploy is slow or memory-limited:
- Prefer a smaller embedding model
- Reduce retrieval `k`
- Use the `demo-data` branch (avoids heavy rebuild steps)

### 5.3 “File not found” for data paths

If the app complains about missing `./chroma_store` or `./db/drugs.sqlite`:
- Verify the branch you deployed includes those folders/files
- Verify your Secrets values match the actual repo paths

## 6) Local parity (recommended before Cloud deploy)

Run locally the same way Cloud will run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

If local works but Cloud fails, check the Cloud logs first; most issues are:
- missing data files
- missing or incorrect Secrets
- resource constraints (RAM/CPU)

## 7) Quick checklist

- [ ] Deploy `app.py`
- [ ] `runtime.txt` present and valid
- [ ] `requirements.txt` installs cleanly on CPU-only
- [ ] Secrets configured in Cloud
- [ ] `db/drugs.sqlite` available
- [ ] `chroma_store/` available (or ingested at runtime)

---

If you want, tell me which branch you plan to deploy (`main` vs `demo-data`) and I can tailor the Secrets + expected data layout to that branch.