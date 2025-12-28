# Worklog — Dec 28, 2025 — Tutorial Docs Restore

This worklog records:
1) why the tutorial documentation folders/files appeared “missing”, and
2) how they were recovered and restored into the repository.

## What was “missing”

A set of tutorial documents and code examples that historically lived under:

- `docs/tutorial/`
  - `api/`
  - `caremind/`
  - `chromadb/`
  - `embeddings/`
  - `python/`
  - `streamlit/`

In the working tree on Dec 28, 2025, these paths were not present, leading to the question: “Were tutorials removed?”

## Why those folders/files were removed/changed in the past

Based on commit messages and file move/rename operations found in git metadata:

- 2025-12-21 — **Reorganized tutorials into topic-based folders**
  - The tutorial area was reshaped from a flatter layout into topic folders (e.g., `docs/tutorial/python/`, `docs/tutorial/streamlit/`, etc.).

- 2025-12-21 — **Renamed Streamlit tutorial files**
  - Some Streamlit tutorial filenames were renamed to clarify their scope.

- 2025-12-14 — **Removed CHROMA tutorial documentation**
  - `docs/tutorial/CHROMA_TUTORIAL.md` was deleted in that commit.

These changes were likely motivated by docs hygiene (reducing duplication, clarifying scope, and organizing by topic). Regardless of intent, the net effect was that older tutorial paths were moved/renamed, and at least one file was explicitly deleted.

## Why normal searches didn’t find them

In the branch state being inspected, `docs/tutorial/` did not exist, and a normal search for `tutorial` in the current working tree returned nothing.

Additionally, the specific tutorial-related commits were discovered via git’s internal logs (reflog entries), which can reference commits that are not easily discovered by simple `git log -- docs/...` queries in a given branch state.

## How the tutorials were restored

### 1) Identify the tutorial-related commits

We located tutorial-related commits by searching git metadata for “tutorial” references and then inspecting the commits to enumerate exact file paths.

Key commits identified:

- `bd44665` (2025-12-21) — “Reorganize tutorials into topic-based folders”
  - Moves and organizes tutorial files into subfolders.

- `72f31e9` (2025-12-21) — “Rename Streamlit tutorials to clarify scope”
  - Renames Streamlit tutorial markdown files.

- `630df82` (2025-12-25) — “Docs: remove per-file author metadata from tutorial”
  - Updates `docs/tutorial/streamlit/caremind_streamlit_code_walkthrough.md`.

- `0942004` (2025-12-14) — “Remove CHROMA tutorial documentation”
  - Deletes `docs/tutorial/CHROMA_TUTORIAL.md`.

### 2) Restore the tutorial tree

On `main`, the restore used:

- Restore the tutorial directory from the latest tutorial commit:
  - `git checkout 630df82 -- docs/tutorial`

- Restore the deleted file from the parent of the deletion commit:
  - `git checkout 0942004^ -- docs/tutorial/CHROMA_TUTORIAL.md`

### 3) Commit and push

The restored tutorial docs were committed and pushed.

- `main`
  - Restored tutorials commit: `0beaef3` — “docs: restore tutorial documentation”
  - Docs index link update: `0859fbb` — “docs: link worklogs and tutorials”

- `demo-data`
  - Restored tutorials commit: `88daf3f` — “docs: restore tutorial documentation”
  - Added Dec 26 worklogs: `1a0e9c9` — “docs: add Dec 26 worklogs”
  - Docs index link update: `6ae6cee` — “docs: link worklogs and tutorials”

## What was restored (high-level)

The following tutorial areas were restored into `docs/tutorial/`:

- API examples: `docs/tutorial/api/fastapi_demo.py`
- CareMind tutorial code: `docs/tutorial/caremind/caremind_tutorial.py`
- ChromaDB tutorials: `docs/tutorial/chromadb/...`
- Embedding model notes: `docs/tutorial/embeddings/...`
- Python notes: `docs/tutorial/python/python_stdlib_tutorial.md`
- Streamlit tutorials and demo assets (including a PDF): `docs/tutorial/streamlit/...`

## Follow-up (navigation)

To make these docs discoverable without changing folder structure, `docs/00_README.md` was updated to include links to:

- `docs/worklogs/`
- `docs/tutorial/`

This preserves stable paths while improving discoverability.
