# Batch Processing

## GUI Batch Mode

- Import multiple files by group/condition in the Streamlit app.
- Run model fitting across all groups.
- Export compiled results tables and reports.

## CLI Mode

`frap2025.cli` remains available for non-UI execution.

Example:

```bash
python -m frap2025.cli
```

For reproducible workflows, use synthetic fixtures in `sample_data/` and keep real data in `data/` (git-ignored).
