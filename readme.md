# CareMind · Clinical Decision Support (MVP)

CareMind is a **clinical decision support system (CDSS)** prototype built with **Streamlit**.  
It integrates **retrieval-augmented generation (RAG)**, guideline databases, and drug knowledge bases to provide **evidence-backed, explainable, and bilingual (中/EN)** answers to clinical questions.

---

## 🚀 Features

- **Clinical Q&A**  
  Ask questions like:  
  > “Can β-blockers be used in hypertensive patients with bronchial asthma?”  
  The system retrieves guideline evidence and generates structured suggestions.

- **Evidence Retrieval (RAG)**  
  Uses [ChromaDB](https://www.trychroma.com/) vector store for guideline chunks and [SQLite](https://sqlite.org/) for structured drug data.

- **Structured Outputs**
  - 📚 Evidence snippets  
  - 💊 Drug interactions / adverse effects (from DB)  
  - 🧭 Draft clinical recommendations with compliance disclaimer  

- **UI**
  - Streamlit frontend  
  - Download buttons for **recommendation reports** and **evidence exports** (Markdown format)  
  - Compact bilingual interface (English / 中文)

---

## 📂 Project Structure

```bash
caremind-streamlit/
├── app.py                  # Streamlit frontend
├── rag/
│   ├── retriever.py        # Guideline & drug database retriever
│   ├── pipeline.py         # RAG pipeline & LLM integration
│   └── prompt.py           # Structured prompt templates
├── ingest/
│   └── create_db.py        # Ingest guideline files into ChromaDB
├── data/
│   ├── guidelines/         # Raw guideline PDFs
│   ├── guidelines.parsed.jsonl  # Parsed guideline text
│   └── drug_db.sqlite      # Drug structured database (optional)
├── chroma_store/           # Vector DB persistence (local / demo branch)
├── requirements.txt        # Python dependencies
└── .streamlit/config.toml  # Streamlit UI configuration

```
## ⚙️ Installation
```

1. Clone the repository
git clone https://github.com/<your-username>/caremind-streamlit.git
cd caremind-streamlit

2. Create environment
conda create -n caremind python=3.10 -y
conda activate caremind
pip install -r requirements.txt

3. Set environment variables
export CHROMA_PERSIST_DIR=./chroma_store
export CHROMA_COLLECTION=guideline_chunks_1024_v2
export EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5

4. Build the vector DB
python ingest/create_db.py \
  --in data/guidelines.parsed.jsonl \
  --collection $CHROMA_COLLECTION

5. Run Streamlit
streamlit run app.py
The app will be available at http://localhost:8501.

```
## 🌐 Deployment
```

Local: Works on Windows/WSL + GPU (RTX 4070 SUPER tested)

Streamlit Cloud:

main branch: minimal demo (no full dataset)

demo-data branch: full dataset (≈ 18MB SQLite + 17MB Chroma index)
```

## 🛡️ Compliance & Disclaimer
```
This tool is for research and demonstration purposes only.
It does not replace professional medical judgment.
All outputs include compliance disclaimers.
```
## 📌 Roadmap
```
 Improve evidence ranking with rerankers

 Enhance prompt templates for structured outputs

 Support larger LLMs (H100/A100 inference-ready)

 Multi-user demo via Docker or Cloud Run

 Paper publication on clinical RAG systems
```
## 🤝 Contribution
```
Pull requests are welcome.
For major changes, please open an issue first to discuss.
```
## 📜 License
```
MIT License © 2025 Samuel Huang
```
