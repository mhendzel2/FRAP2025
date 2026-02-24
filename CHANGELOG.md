# Changelog

## [0.4.0] - 2026-02-24

### Added

- Advanced analysis modules:
  - `frap2025/population_analysis.py` with multivariate Wasserstein comparison and permutation testing.
  - `frap2025/material_mechanics.py` with capillary-wave spectroscopy and equipartition-based tension estimation.
  - `frap2025/spt_models/bayesian.py` with `HDPHMM_Analyzer` (Pyro backend + fallback).
  - `frap2025/frap_fitting/fit_pinn.py` with `FRAP_PINN` and `fit_frap_pinn` (torch backend + fallback).
  - `frap2025/optical_flow_analysis.py` with backend-switchable deformation field computation (`farneback` / `raft`).
- New advanced test coverage in `tests/test_advanced_methods.py`.

### Changed

- Consolidated installation and dependency guidance across:
  - `pyproject.toml`, `requirements.txt`, `dependencies.txt`, `INSTALLATION.md`, and `README.md`.
- Added optional `advanced` installation path documentation (`pip install -e .[advanced]`).
- Aligned base image-processing dependencies to current floors:
  - `scipy>=1.13`, `scikit-image>=0.23`.

### Metadata

- Version bumped to `0.4.0` in package and citation metadata.
- Updated `.zenodo.json` software description/keywords to reflect advanced biophysics methods.

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
