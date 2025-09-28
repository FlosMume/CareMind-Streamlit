# CareMind-Streamlit: App Notes

This document provides notes on the **app.py** file in the CareMind-Streamlit project.

---

## Overview

The `app.py` file is the main entry point of the CareMind-Streamlit application.  
It provides the **user interface** and integrates with the pipeline to deliver clinical decision support.

---

## Key Responsibilities

1. **Streamlit UI Setup**
   - Defines page layout, sidebar, and main content areas.
   - Collects user input: clinical question and optional drug name.
   - Offers controls for parameters (e.g., top-K results).

2. **Pipeline Integration**
   - Calls `cm_pipeline.answer(...)` to process user queries.
   - Passes environment variables and session parameters as needed.
   - Retrieves structured output: draft advice, evidence, drug data.

3. **Results Display**
   - Sections for:
     - **🧭 Suggestion** (clinical advice draft)
     - **📚 Evidence snippets**
     - **💊 Drug structured data**
     - **🪵 Logs** (debug/trace info)
   - Output is interactive and expandable.

4. **Export Functions**
   - Buttons for exporting results to **Markdown** (advice + evidence).
   - Error-handled to avoid empty file generation.

---

## UI Flow

1. **User Input**
   - Query (text area)
   - Drug name (optional input box)
   - Number of results (slider or input field)

2. **Backend Processing**
   - Call to retriever → reranker (optional) → LLM orchestrator → formatter.

3. **Output**
   - Advice draft with disclaimer.
   - Evidence snippets (with references).
   - Structured drug info table.
   - Logs (execution trace).

4. **Export**
   - Save outputs as `.md` files for offline reference.

---

## Notes on Implementation

- Rich comments added for maintainability.
- Handles both Chinese and English queries.
- Uses session state to maintain continuity between queries.
- Error handling ensures resilience against empty or failed retrievals.

---

## Future Enhancements

- Compact layout for export buttons.
- Improve evidence snippet formatting for clarity.
- Integrate reranker in real-time for higher evidence quality.
- Add support for multiple export formats (PDF, DOCX).

---

*Last updated: September 27, 2025*
