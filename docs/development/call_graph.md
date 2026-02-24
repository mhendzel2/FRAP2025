# FRAP2025 Call Graph Snapshot

This snapshot summarizes the high-level module graph used to plan the refactor.

## Canonical Streamlit Entry

- **Canonical UI entrypoint**: `app.py` (root shim) -> `frap2025/ui/app.py` -> `frap2025/ui/legacy_app.py`
- Rationale:
  - `app.py` is now a stable root launcher for `streamlit run app.py`.
  - The original full Streamlit implementation is preserved as `frap2025/ui/legacy_app.py`.
  - Package script entrypoint (`frap2025`) resolves to the same UI path.

## Core Dependency Graph (selected)

- `app.py` -> `frap2025/ui/app.py` -> `frap2025/ui/legacy_app.py`
- `frap2025/ui/legacy_app.py` -> `frap_analysis_enhanced.py`, `frap_core.py`, `frap_global_fitting.py`, `frap_input_handler.py`, `frap_pdf_reports.py`, `frap_report_generator.py`, `frap_visualizer.py`
- `frap_analysis_enhanced.py` -> `frap_core.py`, `frap_input_handler.py`
- `frap_group_comparison.py` -> `frap_core.py`, `frap_advanced_fitting.py`
- `frap_bootstrap.py` -> `frap_core.py`
- `frap_comparison_v2.py` -> `frap_global_fitting.py`
- `frap2025/cli.py` -> `frap_core.py`, `frap_data_loader.py`, `frap_input_handler.py`

## What `frap_core.py` Exports vs `frap_fitting.py`

### `frap_core.py` (legacy root)
- Central FRAP utilities and model helpers used by legacy UI/analysis modules.
- Key exposed utilities include bleach indexing, post-bleach slicing, diffusion conversion, and model fitting helpers wrapped in `FRAPAnalysisCore` patterns.

### `frap_fitting.py` (legacy root)
- Focused fitting primitives and diagnostics:
  - single/double model equations
  - residual functions
  - single/double fitting routines
  - model-selection and diagnostic helpers

### New canonical core package (`frap2025/core/`)
- `frap2025/core/frap_core.py`: explicit normalization API (`simple`, `double`, `full_scale`) and validation errors.
- `frap2025/core/frap_fitting.py`: audited fit models and dataclass result schema.
- `frap2025/core/frap_advanced_fitting.py`: AIC/BIC model ranking.
- `frap2025/core/frap_bootstrap.py`: bootstrap uncertainty estimation.

## Does `frap_analysis_enhanced.py` supersede or wrap `frap_core.py`?

- It **wraps/orchestrates** `frap_core.py` rather than replacing it.
- Evidence: explicit imports and runtime calls into `FRAPAnalysisCore` fitting/model-selection methods.

## What the UI Implementation Imports (top-level local modules)

- `frap_input_handler` (`FRAPInputHandler`, `FRAPCurveData`)
- `frap_analysis_enhanced` (`FRAPGroupAnalyzer`, `FRAPStatisticalComparator`)
- `frap_visualizer` (`FRAPVisualizer`)
- `frap_report_generator` (`EnhancedFRAPReportGenerator`)
- `frap_global_fitting` (optional workflow import block)

## Legacy Streamlit Variants

- `streamlit_frap.py`, `streamlit_frap_final_clean.py`, and `streamlit_frap_final_restored.py` are still present for backward compatibility/reference.
- They are treated as legacy variants; `app.py` is the current canonical entry.
