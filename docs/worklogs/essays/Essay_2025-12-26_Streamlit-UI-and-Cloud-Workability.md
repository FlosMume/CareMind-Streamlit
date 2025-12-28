# Essay — Making the Streamlit UI and Streamlit Community Cloud behave as expected (Dec 26, 2025)

> Note on date: the repository’s git history shows the relevant changes landing on **Dec 26, 2025**. If you meant Dec 26, 2026, this write‑up should be re-generated from that date’s commit log.

## Why this work mattered

A Streamlit app that “works on my machine” can still fail—or behave inconsistently—once deployed to Streamlit Community Cloud. The platform has different constraints: limited control over system libraries, cold starts, caching nuances, and a tendency for subtle environment issues (SQLite/Chroma, missing dependencies, path assumptions) to surface only after deployment.

The work done on Dec 26 focused on one core outcome: **make the CareMind Streamlit UI reliably usable and predictable**, both locally and on Streamlit Community Cloud.

That meant tightening up:

- How the app renders model output into UI sections (Advice vs Evidence).
- How citations behave (anchors, visible bracketed IDs, stable formatting).
- How the app falls back when the backend can’t call the LLM.
- How the app behaves when retrieval returns 0 hits.
- How cloud deployment dependencies and environment quirks (especially SQLite/Chroma) are handled.

## Making the UI output “stable”: advice, evidence, and citations

### A predictable split between “Advice” and “Evidence List”

One of the easiest ways to confuse users is when model output arrives as a single blob of Markdown but is displayed across multiple UI tabs. If the UI doesn’t reliably separate sections, you get duplicate evidence, missing evidence, or “headers inside headers” that clutter the tab.

The Dec 26 changes improved how the app:

- **Detects the Evidence section** and splits it away from the Advice section.
- **Keeps Evidence List rendering in the Evidence tab only**, preventing duplicates inside Advice.
- **Normalizes Evidence List items** so each citation becomes a clean, single bullet line.

This turned the Evidence tab into something users can trust: consistent structure, consistent spacing, and consistent scanning behavior.

### Citations users can actually use

A clinical decision support UI lives or dies by traceability. Even a correct recommendation feels unsafe if the user can’t jump to supporting sources.

On Dec 26, citation rendering was tightened in two important ways:

- Preserve **visible bracketed citations** like `[1]` and avoid losing the brackets or changing them into confusing prefixes.
- Make citations **clickable anchors** so a user can jump from the Advice tab to the corresponding evidence snippet in the Hits view.

This makes the UI feel “workable” because the user experience becomes interactive rather than static: advice is no longer detached from evidence.

## Better behavior under real-world constraints: draft mode and graceful fallback

Streamlit Community Cloud often exposes issues that are easy to hide locally:

- Missing or misconfigured secrets.
- Rate limits or errors from the LLM provider.
- Cold start costs that make timeouts more likely.

The Dec 26 changes improved the app’s behavior when it cannot produce a full LLM answer:

- The pipeline gained a more **natural-language fallback**, rather than failing hard.
- The UI began exposing **draft-mode reason metadata**, so the user sees *why* the app responded in a certain mode (missing key vs no hits vs provider error).
- Advice headers in the UI were adjusted to **match the current mode**, so the user isn’t misled about the confidence/quality of the output.

This matters for “workability” because deployed apps must communicate state clearly. A silent fallback feels like a bug; an explained fallback feels like a feature.

## Diagnostics that match how Cloud failures happen

When something goes wrong on Streamlit Community Cloud, it’s frequently environmental:

- A vector store directory isn’t present.
- A collection name is wrong.
- The database path is misconfigured.
- Retrieval returns 0 hits because the index isn’t there.

The Dec 26 work added or refined diagnostics—especially around the critical condition of **0 retrieval hits**.

Instead of leaving users with an empty Evidence tab and no explanation, the UI now warns when retrieval returns 0 hits and suggests checks, which is exactly the kind of failure mode you see on Cloud after a redeploy.

## Drug-related UX: extracting intent and displaying structured data cleanly

Two improvements made the drug flow more robust:

1. **Infer drug name from pasted templates** in the question box. Users don’t always use the UI fields as intended; they paste a whole template. The app now tries to extract the drug name safely rather than forcing the user to reformat input.
2. **Localize structured drug keys in Chinese UI mode**, so the JSON doesn’t look like a developer payload. This is the difference between “data exists” and “data is usable.”

Additionally, the demo drug database was updated (via `db/drugs.sqlite` + ingestion updates), improving the end-to-end behavior of drug lookup in the deployed environment.

## Bilingual behavior as a first-class feature

The changes also reinforced bilingual expectations:

- UI tab titles were localized.
- A Chinese prompt file (`rag/prompt_cn.py`) was tracked in version control, making deployment consistent (Cloud pulls from git; untracked files don’t exist there).

This is a common Cloud failure pattern: local files exist but aren’t committed. Tracking the prompt file directly addresses “works locally, fails on Cloud.”

## Cloud readiness: dependencies and environment fixes

Finally, a small but critical Cloud-specific fix landed: **OpenAI dependency added in `requirements.txt`**.

On Streamlit Community Cloud, your environment is built from `requirements.txt`. Missing a dependency doesn’t fail gracefully—it prevents the app from building or starting. This kind of change often looks “minor” but has outsized impact on deployability.

## What “workable and as expected” means after Dec 26

After these changes, the UI has clearer, more predictable behavior:

- Advice and evidence no longer fight for the same space.
- Citations remain visible and usable.
- Evidence lists render in a stable one-item-per-line format.
- Fallback modes and missing configuration are explained.
- Diagnostics guide the user/admin toward actionable fixes.
- Drug extraction and localization match how real users input data.
- Cloud deployments are less brittle (tracked prompt assets + corrected dependencies).

In short: the app moved from “prototype output in a web page” to “a deployable Streamlit experience that behaves consistently under Cloud constraints.”

## Source of truth

This essay is grounded in the Dec 26 commit history (examples):

- `fix(cloud): add openai dep; dedupe draft header`
- `improve(draft): natural-language fallback + evidence list`
- `feat(diag): warn when retrieval hits are 0`
- `fix(ui): keep bracketed citations; bullet evidence list`
- `fix(ui): evidence list per-item lines; hide header; improve drug lookup`
- `fix(ui): infer drug from pasted question; update demo drug db`
- `fix(ui): localize drug structured keys in zh`
