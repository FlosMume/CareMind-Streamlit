#!/usr/bin/env python3
"""
caremind_tutorial.py

A richly-commented tutorial script for the CareMind project.

Purpose:
- Explain the high-level components in this repo (ingest, vector build, retriever, RAG pipeline).
- Provide runnable example functions demonstrating how to call those components.
- Offer guidance, prerequisites, and tips for debugging and extension.

This is intentionally verbose and educational: read the inline comments.

Usage (examples):
    # list available demo helpers
    python docs/tutorial/caremind_tutorial.py --list

    # run a simple 'show structure' demo
    python docs/tutorial/caremind_tutorial.py --show-structure

    # run the ingestion demo (non-destructive; many functions are safe to read-only)
    python docs/tutorial/caremind_tutorial.py --demo ingest

    # run a quick query demo (requires vectors/building step to have run)
    python docs/tutorial/caremind_tutorial.py --demo query --q "What is the recommended dosage of X?"

Notes:
- This file demonstrates patterns rather than being a production-ready CLI.
- Many functions catch ImportError and print guidance so the tutorial remains useful
  even if optional dependencies (like Chroma or SentenceTransformer) are not installed.
"""

from __future__ import annotations

import argparse
import inspect
import os
import sys
from typing import Optional


# --------------------------- Utility helpers ---------------------------

def project_root() -> str:
    """Return the repo root path (heuristic: two directories up from this file).

    This is useful so the tutorial can import project modules by adding
    the repo root to sys.path when needed.
    """
    return os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")) # two levels up: Give me the clean absolute path to the grandparent directory of this Python file.


def safe_add_repo_root() -> None:
    """Add repo root to `sys.path` if not already present.

    Why: many internal modules (e.g., `ingest`, `rag`) are top-level modules
    in the repository root. Adding root to `sys.path` makes imports simpler
    for demonstration purposes.
    """
    root = project_root()
    if root not in sys.path:
        sys.path.insert(0, root)


# --------------------------- Project overview ---------------------------

PROJECT_OVERVIEW = """
CareMind Project - high level

- `ingest/` : helpers to parse documents and prepare data for embedding.
- `ingest/build_vectors.py` and `ingest/parse_docs.py`: scripts used to
  create embeddings and chunk documents.
- `chroma_store/` : example local Chroma DB used as a vector store.
- `rag/` : retrieval-augmented generation logic, prompt helpers and pipeline.
- `docs/` : documentation and tutorials (this file lives under `docs/tutorial`).

This tutorial shows small, focused code snippets to help you explore the
above components interactively.
"""


def show_project_structure() -> None:
    """Print a tiny map of the repo sections most relevant to RAG workflows."""
    print(PROJECT_OVERVIEW)


# --------------------------- Demo functions ---------------------------

def demo_imports() -> None:
    """Demonstrate importing common modules from the repo with graceful messages.

    The goal is to show where to look if a module is missing.
    """
    safe_add_repo_root()

    print("\nAttempting to import core modules used in CareMind demos...\n")

    imports = [
        ("ingest.parse_docs", "parsing and chunking documents"),
        ("ingest.build_vectors", "vector building and embedding helpers"),
        ("rag.retriever", "retriever code used by the RAG pipeline"),
        ("rag.pipeline", "end-to-end RAG pipeline orchestration"),
    ]

    for modname, purpose in imports:
        try:
            __import__(modname)
            mod = sys.modules[modname]
            print(f"OK: imported {modname} — {purpose}")
            # show key functions in that module to help learners
            names = [n for n in dir(mod) if not n.startswith("_")][:8]
            if names:
                print(f"  example attributes: {', '.join(names)}")
        except Exception as exc:  # broad on purpose: tutorial should not crash
            print(f"WARN: could not import {modname} ({purpose}) — {exc}")
            print(f"  Tip: check that `{modname.split('.')[0]}` exists in the repo root.")


def demo_ingest_preview(max_lines: int = 20) -> None:
    """Show how to call the document parsing scripts for a preview.

    This demo does not write by default; it attempts to import the parsing
    helper and call a safe 'preview' routine if available.
    """
    safe_add_repo_root()
    try:
        from ingest import parse_docs
    except Exception as exc:
        print("Could not import `ingest.parse_docs` — skipping ingest preview.")
        print("Error:", exc)
        return

    print("\nFound `ingest.parse_docs` — attempting to run safe preview...\n")
    # Many user scripts expose CLI entry points. Here we try to use a preview
    # function if the module defines one; otherwise we print the docstring.
    if hasattr(parse_docs, "preview_documents"):
        try:
            # preview_documents should be a safe read-only helper in well-structured modules
            parse_docs.preview_documents(max_lines=max_lines)
        except Exception as exc:
            print("Preview function exists but raised an error:", exc)
    else:
        print("Module `ingest.parse_docs` does not define `preview_documents()`")
        print("Read its docstring for guidance:\n")
        print(inspect.getdoc(parse_docs) or "<no docstring available>")


def demo_build_vectors_hint() -> None:
    """Explain the vector-build step and show the central command/file used.

    This prints guidance rather than running a potentially heavy build.
    """
    safe_add_repo_root()
    print("\nVector build: what to run and where to look")
    candidate = os.path.join(project_root(), "ingest", "build_vectors.py")
    if os.path.exists(candidate):
        print(f"- Script: {candidate}")
        print("- Typical flow: parse documents -> chunk -> embed -> persist vector store")
    else:
        print("- No `ingest/build_vectors.py` script found. Check `ingest/` folder.")


def demo_query_example(query: Optional[str] = None) -> None:
    """Demonstrate how to load a retriever and run a single query.

    This function tries to import `rag.retriever` and `rag.pipeline` and will
    print helpful error messages if dependencies (vector DB, embedding service)
    are not configured. The aim is to give learners the exact function calls
    they'd use when running queries programmatically.
    """
    safe_add_repo_root()
    try:
        from rag import retriever as retriever_mod
    except Exception as exc:
        print("Cannot import `rag.retriever`. Skipping query demo.")
        print("Error:", exc)
        return

    print("\n`rag.retriever` imported — showing example usage...\n")

    # Try to find an obvious factory method (project modules differ); we inspect
    # the module for likely helpers so the tutorial adapts to code structure.
    if hasattr(retriever_mod, "get_retriever"):
        print("- Found `get_retriever()` factory — example call below:")
        print("    retr = rag.retriever.get_retriever()")
        print("    docs = retr.retrieve('your question', k=5)")
        print("    # then pass docs into your prompt/pipeline")
    else:
        print("- No `get_retriever()` factory found. Inspect the module for constructors.")
        print("  Available public names:", [n for n in dir(retriever_mod) if not n.startswith('_')][:20])

    if query:
        print(f"\nIf you have a running vector store and embeddings, you could run:\n  Query: {query}\n")


# --------------------------- CLI wiring ---------------------------

def list_demos() -> None:
    """List useful demo functions in this file with one-line descriptions."""
    demos = [
        ("show-structure", "Print a short project overview"),
        ("imports", "Try importing key modules with helpful tips"),
        ("ingest-preview", "Preview the ingest/parse pipeline (read-only)") ,
        ("vector-hint", "Show which script performs vector building") ,
        ("query", "Show example calls for running a query against the retriever"),
    ]
    print("Available demos:")
    for name, desc in demos:
        print(f" - {name}: {desc}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="CareMind tutorial helper CLI")
    parser.add_argument("--list", action="store_true", help="List available demos")
    parser.add_argument("--show-structure", action="store_true", help="Show project overview")
    parser.add_argument("--demo", choices=["imports", "ingest", "vector-hint", "query"], help="Run a specific demo")
    parser.add_argument("--q", type=str, default=None, help="Query text for the `query` demo")

    args = parser.parse_args(args=argv)

    if args.list:
        list_demos()
        return 0

    if args.show_structure:
        show_project_structure()
        return 0

    if args.demo == "imports":
        demo_imports()
        return 0

    if args.demo == "ingest":
        demo_ingest_preview()
        return 0

    if args.demo == "vector-hint":
        demo_build_vectors_hint()
        return 0

    if args.demo == "query":
        demo_query_example(query=args.q)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
