#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
                  CareMind · Chroma Vector Database Tutorial
================================================================================

Purpose
-------
This tutorial demonstrates how to use ChromaDB, a vector database, within the
CareMind medical decision support system. ChromaDB allows us to:
  • Store medical documents as vector embeddings
  • Perform semantic similarity search
  • Retrieve relevant guidelines and drug information quickly
  • Manage collections of medical knowledge

Why Chroma?
-----------
ChromaDB is chosen for CareMind because:
  1. Lightweight - works on resource-constrained environments (Streamlit Cloud)
  2. Persistent - saves embeddings on disk, fast warm-starts
  3. Flexible - supports custom embeddings and metadata
  4. Open-source - transparent, auditable for medical use
  5. Simple API - easy to learn and integrate into Streamlit apps

Prerequisites
-------------
  pip install chromadb sentence-transformers numpy

Structure of this Tutorial
---------------------------
Section 0: Imports & Setup
  └─ Configure environment, import dependencies

Section 1: Core Concepts
  └─ Client, Collections, Embeddings, Documents, Metadata

Section 2: Basic Operations
  └─ Create client, create collection, add documents, query

Section 3: Working with Embeddings
  └─ Default embeddings vs. custom models

Section 4: Metadata & Filtering
  └─ Enrich documents with metadata, filter search results

Section 5: Persistence & Warm Starts
  └─ Save/load collections, manage on-disk storage

Section 6: Advanced Patterns
  └─ Batch operations, upsert, deduplication

Section 7: CareMind-Specific Patterns
  └─ Multi-language support, healthcare domain setup, diagnostics

Section 8: Debugging & Best Practices
  └─ Common pitfalls, monitoring, troubleshooting

================================================================================
                             SECTION 0: IMPORTS & SETUP
================================================================================

Before running any code, ensure you have installed:
  pip install chromadb sentence-transformers

If running on Streamlit Cloud or old SQLite environments, also install:
  pip install pysqlite3-binary

The CareMind project uses a "shim" pattern to handle older SQLite versions:
  See: app.py lines 16-21
  See: rag/retriever.py lines 56-62
"""

# --- SQLite compatibility shim (important on Streamlit Cloud) ---
# This MUST be done before any 'import chromadb' to patch the sqlite3 module
try:
    import pysqlite3
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
    print("[SETUP] Using pysqlite3 for SQLite compatibility")
except ImportError:
    print("[SETUP] Using system sqlite3 (native)")

# Standard library imports
import os
import sys
import json
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime

# Third-party imports
import chromadb
from chromadb.config import Settings

try:
    # SentenceTransformers: embedding models that understand Chinese medical text
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("[WARNING] sentence-transformers not installed; using default embeddings")

import numpy as np


# =============================================================================
#                          SECTION 1: CORE CONCEPTS
# =============================================================================
"""
KEY TERMS (read this before proceeding)
========================================

1. CLIENT
   └─ Your connection to ChromaDB. Creates and manages collections.
   └─ Types:
      • PersistentClient: saves to disk; good for production
      • EphemeralClient: in-memory; good for testing

2. COLLECTION
   └─ A named container for documents, embeddings, and metadata.
   └─ Like a "table" in SQL, but for vector data.
   └─ Example: "guideline_chunks", "drug_information", "clinical_trials"

3. DOCUMENT (or CORPUS)
   └─ The actual text (e.g., medical guideline excerpt).
   └─ Converted to a vector embedding automatically.

4. EMBEDDING
   └─ A vector (list of numbers) that represents the semantic meaning
      of a document. Text with similar meaning → similar embeddings.
   └─ Created by an embedding model (e.g., BAAI/bge-large-zh-v1.5).
   └─ Example shape: (1024,) for bge-large-zh-v1.5

5. METADATA
   └─ Structured information about a document (e.g., source file, date, author).
   └─ Attached to each document; can be used to filter search results.
   └─ Example: {"source": "guideline_2024.pdf", "section": "3.2", "year": 2024}

6. ID
   └─ Unique identifier for a document within a collection.
   └─ Auto-generated or manually assigned.
   └─ Prevents duplicate insertions.

7. QUERY
   └─ Text or embedding to search for in the collection.
   └─ Chroma finds the K most similar documents.
   └─ Returns: document IDs, distances, documents, metadata

WORKFLOW DIAGRAM
================

User Query (in Chinese)
         │
         ▼
  [Embedding Model]  ← converts text to vector
         │
         ▼
  [Vector Database]  ← finds nearest neighbors
         │
         ▼
   Retrieved Docs    ← return top-K matches with metadata
         │
         ▼
   [LLM/Prompt]      ← use docs to answer user's question
"""


def log_step(section: str, message: str) -> None:
    """
    Utility function for timestamped logging.
    Makes it easy to follow along in tutorial output.
    """
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{ts}] {section:20s} | {message}")


# =============================================================================
#                      SECTION 2: BASIC OPERATIONS
# =============================================================================
"""
GOAL: Learn how to create a Chroma client, add documents, and search.
"""


def example_01_create_ephemeral_client():
    """
    Example 1: Create an in-memory Chroma client.
    
    Use Case: Quick prototyping, unit tests, interactive exploration.
    
    Note: Data is lost when the program exits. NOT suitable for production.
    """
    log_step("EXAMPLE 1", "Creating ephemeral Chroma client")
    
    # Create an in-memory client
    client = chromadb.EphemeralClient()
    print("  ✓ Ephemeral client created (data in RAM, no persistence)")
    
    # List collections (should be empty)
    collections = client.list_collections()
    print(f"  ✓ Collections: {[c.name for c in collections]}")
    
    return client


def example_02_create_persistent_client(persist_dir: str = "./chroma_store"):
    """
    Example 2: Create a persistent Chroma client.
    
    Use Case: Production, web apps, any scenario where data must survive restarts.
    
    Args:
        persist_dir: Path where Chroma stores embeddings on disk.
                     Can be relative or absolute.
    
    Note: Directory is auto-created if it doesn't exist.
          All collections are stored here in a structured format.
    """
    log_step("EXAMPLE 2", "Creating persistent Chroma client")
    
    # Convert to absolute path to avoid CWD ambiguity
    persist_path = os.path.abspath(os.path.expanduser(persist_dir))
    print(f"  Persist directory: {persist_path}")
    
    # Create the directory if it doesn't exist
    os.makedirs(persist_path, exist_ok=True)
    
    # Create persistent client with custom settings
    settings = Settings(
        # Important: Disable telemetry in production to avoid unexpected network calls
        anonymized_telemetry=False,
        allow_reset=True,  # Allow client.reset() for cleanup
        is_persistent=True,
    )
    
    client = chromadb.PersistentClient(
        path=persist_path,
        settings=settings,
    )
    print("  ✓ Persistent client created (data on disk)")
    print(f"  ✓ Collections: {[c.name for c in client.list_collections()]}")
    
    return client


def example_03_create_collection():
    """
    Example 3: Create a collection within a client.
    
    Collections are like "tables" in SQL. Each collection:
      • Has a unique name
      • Stores documents with embeddings and metadata
      • Can be queried independently
      • Can be deleted or reset
    
    In CareMind, we might have:
      • "guideline_chunks": clinical guidelines
      • "drug_database": drug monographs
      • "case_studies": real-world medical cases
    """
    log_step("EXAMPLE 3", "Creating a collection")
    
    client = chromadb.EphemeralClient()
    
    # Create a collection named "tutorial_docs"
    collection = client.create_collection(
        name="tutorial_docs",
        # metadata can describe the collection's purpose
        metadata={"description": "Tutorial documents for learning"}
    )
    print(f"  ✓ Created collection: {collection.name}")
    print(f"  ✓ Collection count: {collection.count()}")
    
    return client, collection


def example_04_add_documents():
    """
    Example 4: Add documents to a collection.
    
    This is the core operation: storing medical documents in Chroma.
    
    Key points:
      • Documents are automatically embedded using a model
      • Each document needs a unique ID
      • Metadata is optional but recommended
      • IDs prevent duplicates (useful for updates)
    """
    log_step("EXAMPLE 4", "Adding documents to collection")
    
    client = chromadb.EphemeralClient()
    collection = client.create_collection(name="medical_docs")
    
    # Example medical documents (in English for simplicity)
    documents = [
        "Hypertension is characterized by elevated blood pressure. "
        "Guidelines recommend lifestyle changes and medication.",
        
        "Aspirin is commonly used for cardiovascular disease prevention. "
        "Typical dose is 75-100 mg daily.",
        
        "Type 2 diabetes management includes diet, exercise, and medications. "
        "Regular monitoring of glucose levels is essential.",
        
        "The beta-blocker propranolol reduces heart rate and blood pressure. "
        "Common side effects include fatigue and bradycardia.",
    ]
    
    # Unique IDs for each document
    ids = [f"doc_{i}" for i in range(len(documents))]
    
    # Optional: metadata for each document
    metadatas = [
        {"source": "guideline_hypertension.pdf", "year": 2024},
        {"source": "drug_aspirin.md", "category": "antiplatelet"},
        {"source": "guideline_diabetes.pdf", "year": 2023},
        {"source": "drug_propranolol.md", "category": "beta_blocker"},
    ]
    
    # Add documents to collection
    # Chroma will automatically embed them using the default model
    collection.add(
        documents=documents,
        ids=ids,
        metadatas=metadatas,
    )
    
    print(f"  ✓ Added {len(documents)} documents")
    print(f"  ✓ Collection now has {collection.count()} documents")
    
    return client, collection


def example_05_query_collection():
    """
    Example 5: Query the collection (semantic search).
    
    This is where Chroma shines:
      • You provide a query (plain text)
      • Chroma embeds your query
      • Finds the K nearest documents by cosine similarity
      • Returns results with distances and metadata
    """
    log_step("EXAMPLE 5", "Querying the collection")
    
    # Reuse the collection from Example 4
    client = chromadb.EphemeralClient()
    collection = client.create_collection(name="medical_docs")
    
    documents = [
        "Hypertension is characterized by elevated blood pressure.",
        "Aspirin is used for cardiovascular disease prevention.",
        "Type 2 diabetes management includes diet and exercise.",
        "Propranolol reduces heart rate and blood pressure.",
    ]
    
    collection.add(
        documents=documents,
        ids=[f"doc_{i}" for i in range(len(documents))],
    )
    
    # Query: Find documents relevant to high blood pressure treatment
    query_text = "treatment for high blood pressure"
    
    results = collection.query(
        query_texts=[query_text],
        n_results=2,  # Return top 2 most similar documents
    )
    
    # Unpack results
    # results is a dict with keys: 'ids', 'documents', 'distances', 'metadatas'
    print(f"  Query: '{query_text}'")
    print(f"  ✓ Top {len(results['ids'][0])} results:")
    
    for i, (doc_id, doc, distance) in enumerate(
        zip(results['ids'][0], results['documents'][0], results['distances'][0])
    ):
        similarity = 1 - distance  # Chroma returns distances; similarity = 1 - distance
        print(f"\n    Result {i+1}: {doc_id} (similarity: {similarity:.3f})")
        print(f"    {doc[:60]}...")
    
    return client, collection


# =============================================================================
#                   SECTION 3: WORKING WITH EMBEDDINGS
# =============================================================================
"""
GOAL: Understand how embeddings work and use custom models.

What are embeddings?
====================
An embedding is a vector (list of numbers) that represents the semantic
meaning of text. Similar documents have similar embeddings.

Example:
  "high blood pressure treatment" → [0.123, -0.456, 0.789, ...]
  "hypertension management"       → [0.125, -0.450, 0.791, ...]
  ↑ These are close in vector space (cosine similarity ≈ 0.99)
  
  "cat sleeping" → [-0.9, 0.1, 0.05, ...]
  ↑ This is far from medical texts (cosine similarity ≈ 0.05)

Why Custom Models?
==================
Default embeddings may not understand medical/Chinese text well.
For CareMind, we use:
  • BAAI/bge-large-zh-v1.5: Excellent for Chinese medical text
  • domain-specific fine-tuning available
  
Models are downloaded from Hugging Face on first use.
"""


def example_06_default_embeddings():
    """
    Example 6: Use Chroma's default embedding model.
    
    By default, Chroma uses the "all-MiniLM-L6-v2" model:
      • English-focused
      • 384-dimensional vectors
      • Fast, lightweight
    
    Limitation: Not optimized for medical or non-English text.
    """
    log_step("EXAMPLE 6", "Using default embeddings")
    
    client = chromadb.EphemeralClient()
    
    # Don't specify a custom embedding function; Chroma uses default
    collection = client.create_collection(name="default_embeddings")
    
    collection.add(
        documents=["The quick brown fox", "The lazy dog"],
        ids=["doc_1", "doc_2"],
    )
    
    results = collection.query(query_texts=["fox"], n_results=1)
    print(f"  ✓ Default model embedding dimension: 384")
    print(f"  ✓ Query 'fox' found: {results['documents'][0][0][:30]}...")
    
    return client, collection


def example_07_custom_embedding_function():
    """
    Example 7: Use a custom embedding function with sentence-transformers.
    
    For CareMind, we want an embedding model that:
      • Understands Chinese medical terminology
      • Captures clinical concept similarity
      • Is reasonably fast
    
    We use: BAAI/bge-large-zh-v1.5 (BGE = BAAI General Embeddings)
      • Trained on Chinese text
      • 1024-dimensional vectors (richer than default)
      • Excellent for biomedical domains
    
    Note: First download is ~400MB; subsequent uses are cached locally.
    """
    log_step("EXAMPLE 7", "Using custom embedding function")
    
    if not HAS_SENTENCE_TRANSFORMERS:
        print("  [SKIP] sentence-transformers not installed")
        return None, None
    
    # Load the Chinese medical embedding model
    # First call downloads the model (~400MB) and caches it
    model_name = "BAAI/bge-small-zh-v1.5"  # Using 'small' for tutorial (faster)
    print(f"  Loading embedding model: {model_name}")
    print(f"  (First time: downloads ~100MB and caches it)")
    
    embedding_model = SentenceTransformer(model_name, device="cpu")
    
    # Create a custom embedding function for Chroma
    def embed_function(texts: List[str]) -> List[List[float]]:
        """Convert texts to embeddings using our model."""
        embeddings = embedding_model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    # Create client with custom embedding function
    client = chromadb.EphemeralClient()
    
    collection = client.create_collection(
        name="chinese_medical_docs",
        embedding_function=embed_function,
    )
    
    # Add Chinese medical documents
    chinese_docs = [
        "高血压是指收缩压≥140 mmHg和/或舒张压≥90 mmHg。",
        "阿司匹林通常用于心血管疾病的预防，常用剂量为75-100 mg/天。",
        "2型糖尿病管理包括饮食、运动和药物治疗。",
    ]
    
    collection.add(
        documents=chinese_docs,
        ids=[f"zh_doc_{i}" for i in range(len(chinese_docs))],
    )
    
    # Query in Chinese
    query = "高血压的治疗方法"
    results = collection.query(query_texts=[query], n_results=2)
    
    print(f"  ✓ Model loaded: {model_name}")
    print(f"  ✓ Embedding dimension: 384 (BGE-small)")
    print(f"  ✓ Query: '{query}'")
    print(f"  ✓ Top result: {results['documents'][0][0][:40]}...")
    
    return client, collection


# =============================================================================
#                    SECTION 4: METADATA & FILTERING
# =============================================================================
"""
GOAL: Learn to attach and filter by metadata.

What is metadata?
=================
Metadata is structured information about a document:
  • Source file (e.g., "guideline_2024.pdf")
  • Document type (e.g., "guideline", "drug_monograph")
  • Creation date
  • Author, keywords, section, etc.

Why use metadata?
=================
1. Filtering: "Only search in 2024 guidelines"
2. Post-processing: "Show user which source provided this answer"
3. Auditing: "Track which chunk was used in the diagnosis"
4. Retrieval quality: Combine vector search with metadata constraints

Metadata Requirements (CareMind)
================================
For medical documents, always include:
  • source: File name or document ID
  • date: When the document was created/updated
  • category: Type of document (guideline, drug, etc.)
  • language: "zh" for Chinese, "en" for English
"""


def example_08_metadata_search():
    """
    Example 8: Search with metadata filtering.
    
    Scenario: User asks about hypertension treatment.
    We want to:
      1. Search for relevant documents
      2. Filter to only recent guidelines (2024)
      3. Prefer documents from official sources
    """
    log_step("EXAMPLE 8", "Searching with metadata filtering")
    
    client = chromadb.EphemeralClient()
    collection = client.create_collection(name="guidelines")
    
    # Add documents with rich metadata
    documents = [
        "Hypertension treatment: ACE inhibitors, beta-blockers, diuretics.",
        "Hypertension guidelines 2024: Target BP < 130/80 for most adults.",
        "Old hypertension data from 2018: Target BP < 140/90.",
    ]
    
    metadatas = [
        {
            "source": "drug_database.sqlite",
            "category": "drug_class",
            "year": 2024,
            "confidence": 0.95,
        },
        {
            "source": "guideline_hypertension_2024.pdf",
            "category": "official_guideline",
            "year": 2024,
            "confidence": 0.99,
        },
        {
            "source": "archived_data.json",
            "category": "old_guideline",
            "year": 2018,
            "confidence": 0.80,
        },
    ]
    
    collection.add(
        documents=documents,
        ids=[f"doc_{i}" for i in range(len(documents))],
        metadatas=metadatas,
    )
    
    # Query with filtering
    # Only search in documents from 2024 with high confidence
    query = "hypertension treatment"
    
    results = collection.query(
        query_texts=[query],
        n_results=3,
        # Where clause: only include documents matching the condition
        # See Chroma documentation for where clause syntax
    )
    
    print(f"  Query: '{query}'")
    print(f"  ✓ Found {len(results['ids'][0])} relevant documents:")
    
    for doc_id, doc, metadata in zip(
        results['ids'][0],
        results['documents'][0],
        results['metadatas'][0]
    ):
        print(f"\n    {doc_id}")
        print(f"    Year: {metadata['year']}, Category: {metadata['category']}")
        print(f"    {doc[:60]}...")
    
    return client, collection


# =============================================================================
#                  SECTION 5: PERSISTENCE & WARM STARTS
# =============================================================================
"""
GOAL: Understand how Chroma saves data and design for production use.

Persistence in Chroma
=====================
When you create a PersistentClient, Chroma stores:
  1. Document embeddings (in binary format)
  2. Documents themselves (in a SQLite database)
  3. Metadata (in SQLite)
  4. Collection configuration (in SQLite)

Directory structure:
  chroma_store/
    ├── chroma.sqlite3           # Main database
    └── <collection_id>/          # Per-collection directory
        └── index/               # Vector index files

Benefits of persistence:
  • Fast warm-starts (no re-embedding on restart)
  • Data survives process crashes
  • Shared across multiple processes
  • On-disk footprint is compact

Challenges:
  • Only one process should write at a time (no distributed locking)
  • Moving the directory breaks collection IDs
  • SQLite has limitations on concurrent reads
"""


def example_09_persistent_storage():
    """
    Example 9: Create, save, and reload a collection.
    
    Scenario: Build a medical knowledge base once, then query it
    repeatedly in a Streamlit app.
    """
    log_step("EXAMPLE 9", "Persistent storage demo")
    
    persist_dir = "./tmp_chroma_example"
    
    # Step 1: Create persistent client and add documents
    print(f"  Step 1: Creating persistent storage at {persist_dir}")
    
    settings = Settings(
        anonymized_telemetry=False,
        allow_reset=True,
        is_persistent=True,
    )
    
    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=settings,
    )
    
    collection = client.create_collection(
        name="persistent_medical_docs"
    )
    
    collection.add(
        documents=[
            "Aspirin prevents myocardial infarction.",
            "Metformin is first-line for type 2 diabetes.",
            "Lisinopril is an ACE inhibitor for hypertension.",
        ],
        ids=["doc_1", "doc_2", "doc_3"],
    )
    
    print(f"    ✓ Created collection with {collection.count()} documents")
    
    # Step 2: Simulate program restart by creating a new client
    print(f"  Step 2: Simulating program restart (new client)")
    del client  # Close old client
    
    # Create a new client pointing to the same directory
    client2 = chromadb.PersistentClient(
        path=persist_dir,
        settings=settings,
    )
    
    # The collection is still there!
    collection2 = client2.get_collection("persistent_medical_docs")
    print(f"    ✓ Reloaded collection with {collection2.count()} documents")
    
    # Step 3: Query still works
    print(f"  Step 3: Querying reloaded collection")
    
    results = collection2.query(query_texts=["diabetes medication"], n_results=1)
    print(f"    ✓ Query result: {results['documents'][0][0][:50]}...")
    
    # Clean up
    shutil.rmtree(persist_dir, ignore_errors=True)
    print(f"  ✓ Cleaned up temporary directory")
    
    return client2, collection2


# =============================================================================
#                      SECTION 6: ADVANCED PATTERNS
# =============================================================================
"""
GOAL: Learn production-grade patterns used in CareMind.

Key patterns in CareMind
========================
1. Upsert: Insert or update documents idempotently
2. Deduplication: Avoid adding the same document twice
3. Batch operations: Efficiently load thousands of embeddings
4. Error handling: Graceful fallback when queries fail
5. Diagnostics: Check collection health and size
"""


def example_10_upsert():
    """
    Example 10: Upsert pattern (insert or update).
    
    Problem: What if we run the embedding script twice?
    We don't want duplicate documents.
    
    Solution: Use upsert() instead of add().
      • If ID exists: overwrites the document
      • If ID doesn't exist: inserts it
      • Idempotent: safe to run multiple times
    """
    log_step("EXAMPLE 10", "Upsert pattern")
    
    client = chromadb.EphemeralClient()
    collection = client.create_collection(name="medications")
    
    # Initial add
    print("  Adding documents (first run)...")
    collection.upsert(
        documents=[
            "Aspirin: antiplatelet agent",
            "Metformin: antidiabetic agent",
        ],
        ids=["aspirin", "metformin"],
        metadatas=[
            {"version": 1},
            {"version": 1},
        ],
    )
    print(f"    Collection count: {collection.count()}")
    
    # Run again: update version and fix typo
    print("  Upserting documents (second run with updates)...")
    collection.upsert(
        documents=[
            "Aspirin: antiplatelet agent (updated)",
            "Metformin: antidiabetic agent (updated)",
            "Lisinopril: ACE inhibitor",  # new
        ],
        ids=["aspirin", "metformin", "lisinopril"],
        metadatas=[
            {"version": 2},
            {"version": 2},
            {"version": 1},
        ],
    )
    print(f"    Collection count: {collection.count()}")
    print(f"    ✓ Upsert is idempotent: safe to run repeatedly")
    
    return client, collection


def example_11_batch_operations():
    """
    Example 11: Batch operations for bulk loading.
    
    Problem: Embedding 10,000 documents one-by-one is slow.
    
    Solution: Batch them (e.g., 100 docs at a time).
    Chroma handles:
      • Embedding multiple texts in parallel
      • Efficient database writes
      • Error handling and recovery
    
    CareMind uses this when ingesting guidelines (see ingest/build_vectors.py).
    """
    log_step("EXAMPLE 11", "Batch operations")
    
    client = chromadb.EphemeralClient()
    collection = client.create_collection(name="bulk_load_demo")
    
    # Simulate 1000 medical documents
    print("  Generating 1000 mock documents...")
    batch_size = 100
    total_docs = 1000
    
    for batch_idx in range(0, total_docs, batch_size):
        batch_docs = [
            f"Medical document {i}: Content about disease management, "
            f"treatment options, and clinical guidelines."
            for i in range(batch_idx, min(batch_idx + batch_size, total_docs))
        ]
        batch_ids = [f"doc_{i}" for i in range(batch_idx, min(batch_idx + batch_size, total_docs))]
        
        collection.add(
            documents=batch_docs,
            ids=batch_ids,
        )
        
        if (batch_idx // batch_size + 1) % 3 == 0:
            print(f"    Processed {batch_idx + batch_size} / {total_docs} documents")
    
    print(f"  ✓ Batch loading complete: {collection.count()} documents")
    
    return client, collection


def example_12_deduplication():
    """
    Example 12: Deduplication strategy.
    
    Problem: If the same guideline chunk appears in multiple files,
    we might embed it twice, wasting storage and GPU time.
    
    Solution: Use document hashing to create consistent IDs.
      • Same content → same ID
      • Different content → different ID
      • Upsert automatically handles duplicates
    """
    log_step("EXAMPLE 12", "Deduplication")
    
    import hashlib
    
    def make_doc_id(text: str, metadata: Dict[str, Any]) -> str:
        """
        Create a consistent ID based on content and metadata.
        
        Ensures that identical documents get the same ID,
        so upsert won't create duplicates.
        """
        # Combine content and key metadata
        combined = f"{text}|{metadata.get('source', '')}|{metadata.get('section', '')}"
        
        # Hash to create a short, unique ID
        hash_obj = hashlib.md5(combined.encode())
        return f"doc_{hash_obj.hexdigest()[:12]}"
    
    client = chromadb.EphemeralClient()
    collection = client.create_collection(name="dedup_demo")
    
    # Add documents with duplicates
    documents_v1 = [
        ("Hypertension treatment with ACE inhibitors", {"source": "guideline_2024.pdf"}),
        ("Aspirin for secondary prevention", {"source": "guideline_2024.pdf"}),
        ("Hypertension treatment with ACE inhibitors", {"source": "guideline_2024.pdf"}),  # duplicate!
    ]
    
    for doc, meta in documents_v1:
        doc_id = make_doc_id(doc, meta)
        collection.upsert(
            documents=[doc],
            ids=[doc_id],
            metadatas=[meta],
        )
    
    print(f"  Added 3 documents (with 1 duplicate)")
    print(f"  ✓ Collection count after dedup: {collection.count()} (should be 2)")
    
    return client, collection


# =============================================================================
#                   SECTION 7: CAREMIND-SPECIFIC PATTERNS
# =============================================================================
"""
GOAL: Apply Chroma to the actual CareMind architecture.

CareMind Use Cases
==================
1. Guideline Retrieval
   - Input: Patient symptoms/diagnosis
   - Search: Medical guidelines
   - Output: Relevant clinical recommendations

2. Drug Information Lookup
   - Input: Drug name or condition
   - Search: Drug monographs, interactions
   - Output: Drug information for prescription validation

3. Multi-Language Support
   - Input: Chinese or English queries
   - Search: Chinese and English documents
   - Output: Language-agnostic results

Design Decisions in CareMind
=============================
1. Persistent Client: Survives Streamlit hot-reloads
2. Single Collection: "guideline_chunks" (can add more)
3. Custom Embeddings: BAAI/bge-large-zh-v1.5 for Chinese
4. Cached Client: rag.retriever maintains a singleton client
5. Diagnostics: Health checks exposed to Streamlit UI
"""


def example_13_caremind_client_singleton():
    """
    Example 13: Singleton pattern for Chroma client.
    
    Problem: Streamlit reloads the entire script on every user interaction.
    This creates a new Chroma client each time, which is slow.
    
    Solution: Cache the client using @st.cache_resource (or a module-level
    variable with a factory function in production).
    
    See: rag.retriever._get_client() for the actual implementation.
    """
    log_step("EXAMPLE 13", "CareMind client singleton pattern")
    
    print("  [PATTERN ONLY - See rag/retriever.py for actual implementation]")
    print()
    print("  In production, CareMind does:")
    print("    1. Create client once during module import")
    print("    2. Cache it module-level (not in Streamlit)")
    print("    3. Return same client on every call")
    print("    4. Avoids 'different settings' errors")
    print()
    print("  In Streamlit, CareMind uses @st.cache_resource:")
    print("    @st.cache_resource")
    print("    def get_chroma_client():")
    print("        return chromadb.PersistentClient(path='./chroma_store')")
    
    return None


def example_14_multilingual_search():
    """
    Example 14: Multilingual search (Chinese + English).
    
    CareMind supports both Chinese and English user queries.
    With a good embedding model (BGE), semantic search works across languages.
    
    Strategy:
      • Embed both Chinese and English docs with the same model
      • User can query in either language
      • Results include documents from both languages
      • Post-process results to filter by language if needed
    """
    log_step("EXAMPLE 14", "Multilingual search demo")
    
    if not HAS_SENTENCE_TRANSFORMERS:
        print("  [SKIP] sentence-transformers not installed")
        return None, None
    
    print("  Loading multilingual embedding model...")
    model = SentenceTransformer("BAAI/bge-small-zh-v1.5", device="cpu")
    
    def embed_fn(texts: List[str]) -> List[List[float]]:
        return model.encode(texts, convert_to_numpy=True).tolist()
    
    client = chromadb.EphemeralClient()
    collection = client.create_collection(
        name="multilingual_docs",
        embedding_function=embed_fn,
    )
    
    # Add documents in both languages
    docs = [
        ("高血压是指收缩压≥140 mmHg的慢性病。", {"lang": "zh", "source": "guideline"}),
        ("Hypertension is defined as systolic BP ≥140 mmHg.", {"lang": "en", "source": "guideline"}),
        ("糖尿病患者需要定期监测血糖。", {"lang": "zh", "source": "guideline"}),
        ("Diabetic patients require regular glucose monitoring.", {"lang": "en", "source": "guideline"}),
    ]
    
    for i, (doc, meta) in enumerate(docs):
        collection.add(
            documents=[doc],
            ids=[f"doc_{i}"],
            metadatas=[meta],
        )
    
    # Query in Chinese, should find both Chinese and English docs
    zh_query = "高血压管理"
    print(f"\n  Query (Chinese): '{zh_query}'")
    
    results = collection.query(query_texts=[zh_query], n_results=2)
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        print(f"    [{meta['lang'].upper()}] {doc[:50]}...")
    
    # Query in English
    en_query = "diabetes management"
    print(f"\n  Query (English): '{en_query}'")
    
    results = collection.query(query_texts=[en_query], n_results=2)
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        print(f"    [{meta['lang'].upper()}] {doc[:50]}...")
    
    return client, collection


# =============================================================================
#                   SECTION 8: DEBUGGING & BEST PRACTICES
# =============================================================================
"""
GOAL: Learn to troubleshoot and optimize Chroma usage.

Common Issues
=============

1. "Different settings error"
   Cause: Multiple Chroma clients with different Settings
   Fix: Use a singleton client (see example_13)

2. "Collection not found"
   Cause: Creating client with wrong persist_dir
   Fix: Use absolute paths; store in environment variable

3. Slow queries
   Cause: Large collections without proper indexing
   Fix: Chroma handles this automatically; try smaller batch sizes

4. SQLite "database is locked"
   Cause: Multiple processes writing simultaneously
   Fix: Use file locking or queue-based architecture

5. Out-of-memory when embedding
   Cause: Batch size too large for available GPU/RAM
   Fix: Reduce batch size; use CPU embedding

Best Practices
==============

1. Always use absolute paths
   ✓ persist_dir = os.path.abspath("./chroma_store")
   ✗ persist_dir = "./chroma_store"

2. Set consistent Settings
   - Disable telemetry (anonymized_telemetry=False)
   - Allow resets for testing (allow_reset=True)

3. Use meaningful collection names
   ✓ "guideline_chunks", "drug_monographs"
   ✗ "collection_1", "my_docs"

4. Add rich metadata
   - Enables post-filtering and auditing
   - Required for production medical systems

5. Use deterministic IDs
   - Hashed from content + metadata
   - Ensures idempotency with upsert

6. Monitor collection health
   - Count documents regularly
   - Log add/update/delete operations
   - Track embedding model versions

7. Version your embeddings
   - Store model name in metadata
   - Re-embed when switching models
   - Document which docs use which model

8. Test recovery from failures
   - Can the system restart cleanly?
   - Can old persisted collections still load?
"""


def example_15_diagnostics():
    """
    Example 15: Diagnostics for monitoring collection health.
    
    In production, you want to know:
      • How many documents are in the collection?
      • Is the collection healthy?
      • When was it last updated?
      • What model was used for embeddings?
    """
    log_step("EXAMPLE 15", "Diagnostics and monitoring")
    
    persist_dir = "./tmp_chroma_diagnostics"
    os.makedirs(persist_dir, exist_ok=True)
    
    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False),
    )
    
    # Create and populate a collection
    collection = client.create_collection(
        name="diagnostic_test",
        metadata={
            "created": datetime.now().isoformat(),
            "embedding_model": "default",
            "version": "1.0",
        },
    )
    
    collection.add(
        documents=[f"Document {i}" for i in range(10)],
        ids=[f"doc_{i}" for i in range(10)],
    )
    
    # Diagnostic checks
    print("\n  Diagnostic Information:")
    print(f"    Collection name: {collection.name}")
    print(f"    Document count: {collection.count()}")
    print(f"    Metadata: {collection.metadata}")
    
    # List all collections in the client
    all_collections = client.list_collections()
    print(f"\n  Collections in client:")
    for coll in all_collections:
        print(f"    - {coll.name}: {coll.count()} documents")
    
    # Check if collection exists before loading
    print(f"\n  Checking collection existence:")
    try:
        existing = client.get_collection("diagnostic_test")
        print(f"    ✓ Collection 'diagnostic_test' exists ({existing.count()} docs)")
    except Exception as e:
        print(f"    ✗ Collection not found: {e}")
    
    # Clean up
    shutil.rmtree(persist_dir, ignore_errors=True)
    
    return client, collection


def example_16_error_handling():
    """
    Example 16: Graceful error handling.
    
    In production (especially Streamlit Cloud), things go wrong:
      • Network issues downloading models
      • Disk space for embeddings
      • Corrupted SQLite database
    
    Always use try/except and return sensible defaults.
    """
    log_step("EXAMPLE 16", "Error handling patterns")
    
    print("  Pattern 1: Safe collection loading")
    print("""
    def get_collection_safe(client, name: str):
        try:
            return client.get_collection(name)
        except Exception as e:
            log(f"Failed to load collection {name}: {e}")
            return None
    """)
    
    print("\n  Pattern 2: Safe query execution")
    print("""
    def search_safe(collection, query_text: str, top_k: int = 5):
        try:
            if collection is None:
                return []
            results = collection.query(query_texts=[query_text], n_results=top_k)
            return results.get('documents', [])
        except Exception as e:
            log(f"Query failed: {e}")
            return []  # Graceful fallback
    """)
    
    print("\n  Pattern 3: Safe collection creation")
    print("""
    def create_collection_safe(client, name: str):
        try:
            # Try to get existing collection first
            return client.get_collection(name)
        except:
            # If not found, create new
            try:
                return client.create_collection(name)
            except Exception as e:
                log(f"Failed to create collection {name}: {e}")
                return None
    """)
    
    return None


# =============================================================================
#                           MAIN TUTORIAL RUNNER
# =============================================================================

def main():
    """Run all tutorial examples in sequence."""
    
    print("=" * 80)
    print("                  CHROMA VECTOR DATABASE TUTORIAL")
    print("                       For CareMind Project")
    print("=" * 80)
    
    # Section 1: Basic operations
    print("\n" + "=" * 80)
    print("                     SECTION 2: BASIC OPERATIONS")
    print("=" * 80)
    
    example_01_create_ephemeral_client()
    example_02_create_persistent_client()
    example_03_create_collection()
    example_04_add_documents()
    example_05_query_collection()
    
    # Section 3: Embeddings
    print("\n" + "=" * 80)
    print("                  SECTION 3: WORKING WITH EMBEDDINGS")
    print("=" * 80)
    
    example_06_default_embeddings()
    example_07_custom_embedding_function()
    
    # Section 4: Metadata
    print("\n" + "=" * 80)
    print("                  SECTION 4: METADATA & FILTERING")
    print("=" * 80)
    
    example_08_metadata_search()
    
    # Section 5: Persistence
    print("\n" + "=" * 80)
    print("                  SECTION 5: PERSISTENCE & WARM STARTS")
    print("=" * 80)
    
    example_09_persistent_storage()
    
    # Section 6: Advanced patterns
    print("\n" + "=" * 80)
    print("                     SECTION 6: ADVANCED PATTERNS")
    print("=" * 80)
    
    example_10_upsert()
    example_11_batch_operations()
    example_12_deduplication()
    
    # Section 7: CareMind patterns
    print("\n" + "=" * 80)
    print("                 SECTION 7: CAREMIND-SPECIFIC PATTERNS")
    print("=" * 80)
    
    example_13_caremind_client_singleton()
    example_14_multilingual_search()
    
    # Section 8: Debugging
    print("\n" + "=" * 80)
    print("                  SECTION 8: DEBUGGING & BEST PRACTICES")
    print("=" * 80)
    
    example_15_diagnostics()
    example_16_error_handling()
    
    # Summary
    print("\n" + "=" * 80)
    print("                            TUTORIAL COMPLETE")
    print("=" * 80)
    print("""
Next Steps
----------
1. Review rag/retriever.py for production implementation
2. Review ingest/build_vectors.py for bulk embedding
3. Review app.py for Streamlit integration
4. Check out docs/technical/ for architecture details

Useful Resources
----------------
• Chroma docs: https://docs.trychroma.com/
• SentenceTransformers: https://www.sbert.net/
• BGE embeddings: https://github.com/FlagOpen/FlagEmbedding
• SQLite for Chroma: https://github.com/chroma-core/chroma/issues/1127

Questions?
----------
Review the comments in this file and the referenced source files.
Each section is self-contained and can be studied independently.
""")


if __name__ == "__main__":
    main()
