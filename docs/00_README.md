# CareMind Documentation

This folder contains technical notes, operations/deployment guides, and project documentation.

## Start here

- Cloud deployment guide: [docs/guides/deployment_cloud.md](guides/deployment_cloud.md)
- Testing & validation: [docs/guides/testing_validation.md](guides/testing_validation.md)
- Architecture overview: [docs/technical/architecture.md](technical/architecture.md)
- Data ingestion notes: [docs/technical/data_ingestion.md](technical/data_ingestion.md)
- Developer notes: [docs/technical/code_reference/](technical/code_reference/)

## Notes

- Project doc / planning notes: [docs/notes/caremind_doc.txt](notes/caremind_doc.txt)

## Worklogs

- Worklogs index: [docs/worklogs/](worklogs/)
- Dec 26, 2025 worklog: [docs/worklogs/README_2025-12-26.md](worklogs/README_2025-12-26.md)
- Dec 26, 2025 essay: [docs/worklogs/essays/Essay_2025-12-26_Streamlit-UI-and-Cloud-Workability.md](worklogs/essays/Essay_2025-12-26_Streamlit-UI-and-Cloud-Workability.md)

## Tutorials

- Tutorials: [docs/tutorial/](tutorial/)

## Documentation Metadata Policy

Use Git history as the primary source of authorship. Only add explicit metadata blocks when the document is an operational record or requires accountability.

### MUST include `Author:` (and optionally `Assisted by:`)

- Operational runbooks and deployment guides under `docs/guides/` (example: `deployment_cloud.md`).

### MAY include a `Certificate` section

- Only for documents that record a specific executed state/event (e.g., a Cloud deployment configuration that was validated on a given date).

### SHOULD NOT include Author/Certificate blocks

- General technical docs (architecture, data ingestion, code reference, testing notes).
- Root `readme.md` (keep it product-focused; rely on Git for authorship).

### Style rules (keep it light)

- Prefer a short header block: `Author:` + `Last updated:`.
- If present, certificates should be explicit about scope/date and written as an internal record (not a legal claim).
