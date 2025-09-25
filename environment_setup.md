## 1. System Requirements
```
Operating System:

Linux (Ubuntu 20.04/22.04 recommended)

Windows 11 with WSL2 (tested)

Python: 3.10–3.13 (Anaconda/Miniconda recommended)

GPU (for local runs):

NVIDIA GPU with CUDA support (tested on RTX 4070 SUPER)

Minimum: 8GB VRAM

CUDA/cuDNN:

CUDA 12.8

cuDNN 9.10.2

⚠️ On Streamlit Cloud, GPU acceleration is not available. Use a lightweight embedding/demo dataset (demo-data branch).
```
## 2. Repository Setup
```
# Clone the project
git clone https://github.com/your-username/caremind-streamlit.git
cd caremind-streamlit


Switch to the desired branch:

main: lightweight demo (no large data store).

demo-data: includes chroma_store/ for full demo with real guideline data.
```
## 3. Conda Environment (Recommended)
```
# Create environment
conda create -n caremind python=3.10 -y
conda activate caremind

# Install dependencies
pip install -r requirements.txt


Verify installation:

python -V         # should show Python 3.10.x
pip list | grep chromadb
pip list | grep torch
```
## 4. Environment Variables
```
Define required environment variables in a .env file (local only).

Example .env.example:

# Directory where Chroma vector store is located
CHROMA_PERSIST_DIR=./chroma_store

# Collection name inside Chroma
CHROMA_COLLECTION=guideline_chunks_1024_v2

# Embedding model used
EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5

# Optional API keys (only needed if LLM calls are enabled)
# OPENAI_API_KEY=sk-...
# HF_TOKEN=hf_...


Developers should copy .env.example → .env and edit as needed.

cp .env.example .env
```
## 5. GPU Setup (Local Only)
```
Ensure CUDA toolkit is properly configured:

nvcc --version        # check CUDA compiler
nvidia-smi            # check GPU availability


Torch should recognize GPU:

python -c "import torch; print(torch.cuda.is_available())"
# Expected output: True
```
## 6. Database & Ingestion
```
To build the Chroma database locally:

# Parse guidelines
python ingest/parse_docs.py \
  --in data/guidelines/ \
  --out data/guidelines.parsed.jsonl

# Create vector store
python ingest/create_db.py \
  --in data/guidelines.parsed.jsonl \
  --collection $CHROMA_COLLECTION \
  --persist-dir $CHROMA_PERSIST_DIR \
  --embed-model $EMBEDDING_MODEL \
  --batch-size 64


For Streamlit Cloud, use prebuilt data (demo-data branch).
```
## 7. Run the App

### Local run:

* streamlit run app.py


### Streamlit Cloud:

1 Push branch to GitHub.

2 Connect repo on streamlit.io.

3 Set Secrets for environment variables if needed.

## 8. Troubleshooting
```
Issue: pysqlite3 errors on Streamlit Cloud
→ Ensure pysqlite3-binary is in requirements.txt and aliased in retriever.py.

GPU memory errors
→ Reduce batch size during ingestion (--batch-size 16).

No results returned
→ Check .env values (CHROMA_COLLECTION, EMBEDDING_MODEL).

Streamlit Cloud startup fails
→ Try deploying from demo-data branch with smaller chroma_store/.
```
### ✅ With this setup, you should be able to:

* Run the full MVP locally with GPU.

* Run a demo version on Streamlit Cloud.