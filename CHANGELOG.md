# Changelog

## [0.3.1] - 2026-02-23

### Added

- New package namespace `frap2025/` with structured subpackages:
  - `core`, `io`, `image`, `stats`, `reports`, `database`, `calibration`, `ui`
- Physics validation suite under `tests/`:
  - `tests/test_physics.py` (includes `verify_diffusion_formula()`)
  - `tests/test_fitting.py`, `tests/test_normalization.py`, `tests/test_statistics.py`
- Formula audit report:
  - `FORMULA_AUDIT_RESULT.md`
  - `docs/development/formula_audit.md`
- CI workflow:
  - `.github/workflows/ci.yml`
- Citation/release metadata:
  - `CITATION.cff`, `.zenodo.json`, `.github/workflows/release.yml`

### Changed

- Corrected packaging backend in `pyproject.toml` to `setuptools.build_meta`.
- Implemented audited fitting engine in `frap2025/core/frap_fitting.py`:
  - single, double, triple, Soumpasis, and reaction-diffusion fits
  - `FRAPFitResult` dataclass with uncertainty fields
  - explicit diffusion-conversion convention comments (`k_s = 1/tau_s`)
- Added explicit normalization entrypoint in `frap2025/core/frap_core.py`:
  - `normalize_frap_curve(..., mode="simple"|"double"|"full_scale")`
- Implemented model ranking in `frap2025/core/frap_advanced_fitting.py` with AIC/BIC and ΔAIC interpretation.
- Added BH FDR implementation and automatic test selection in `frap2025/stats/frap_group_stats.py`.
- Added image-analysis helpers for bleach radius, drift correction, pre-bleach photobleaching detection, and multi-ROI extraction.

### Fixed

- Diffusion formula audit validated by synthetic, noise-free fits.
- Build/install blocker (`setuptools.backends.legacy`) resolved.
- Generated reports/log/version marker files are now ignored and untracked from git.

### Repository Hygiene

- `.github/instructions/` removed from tracked content and ignored.
- Debug scripts moved to `scripts/dev/`.
- Added data policy docs for `sample_data/` and `data/`.

### Notes

- Legacy root modules and historical markdown files are still present for compatibility/reference and are being consolidated into `docs/` over subsequent cleanup iterations.
