# CareMind-Streamlit: Testing & Validation Notes

This document outlines the **testing and validation** procedures for the CareMind-Streamlit project.

---

## Overview

Testing and validation are critical to ensure that the CareMind-Streamlit system is **reliable, accurate, and safe** for clinical decision support (CDSS).  
This document covers functional testing, integration testing, and validation of outputs.

---

## Testing Strategy

### 1. Unit Testing
- Validate individual modules (retriever, pipeline, formatter).
- Example tests:
  - **Retriever**: returns correct number of results given `k`.
  - **Pipeline**: handles empty evidence gracefully.
  - **Formatter**: produces Markdown-compliant output.

### 2. Integration Testing
- Validate the flow across retriever → reranker → LLM → formatter → UI.
- Check that session state is correctly maintained.
- Ensure environment variables are properly read.

### 3. End-to-End (E2E) Testing
- Full query cycle from user input to UI output.
- Validate against known test cases:
  - Clinical guideline with expected references.
  - Drug queries with structured data outputs.

### 4. Regression Testing
- Re-run tests after updates to confirm no breaking changes.
- Maintain a minimal set of baseline queries for consistency.

---

## Validation Procedures

### 1. Clinical Relevance
- Cross-check generated suggestions with guideline references.
- Validate evidence snippets are correctly linked to sources.

### 2. Output Format
- Ensure Markdown export contains all expected sections:
  - Suggestion
  - Evidence
  - Drug info
  - Disclaimer

### 3. Multilingual Support
- Validate both Chinese and English queries produce correct output.
- Ensure formatting remains consistent across languages.

### 4. Performance Testing
- Measure response time under typical load (query length, evidence size).
- Confirm GPU acceleration is correctly utilized during embedding/retrieval.

---

## Known Issues

- Evidence relevance may vary due to embedding limitations.
- Export button layout may appear oversized in UI.
- Empty outputs possible if retriever returns no results.

---

## Future Improvements

- Automate testing via `pytest` + Streamlit test runner.
- Add mock guideline datasets for reproducible validation.
- Implement CI/CD pipeline for automated regression testing.
- Expand validation dataset with domain expert annotations.

---

*Last updated: September 27, 2025*
