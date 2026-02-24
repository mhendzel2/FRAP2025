# FRAP2025

[![CI](https://github.com/mhendzel2/FRAP2025/actions/workflows/ci.yml/badge.svg)](https://github.com/mhendzel2/FRAP2025/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

FRAP2025 is a quantitative fluorescence recovery after photobleaching (FRAP) analysis platform for cell biology and biophysics workflows. It provides normalization, model fitting, diffusion/recovery parameter estimation, image-stack helpers, statistical group comparison, and report generation for reproducible analysis of single-cell and grouped FRAP experiments.

## Installation Paths

### 1. Minimal

```bash
pip install -e .
```

### 2. Standard (recommended for development)

```bash
pip install -e .[dev]
```

### 3. Full (includes optional DB + viz extras)

```bash
pip install -e .[dev,db,viz]
```

### 4. Advanced Biophysics + ML

```bash
pip install -e .[advanced]
```

### 5. Full + Advanced

```bash
pip install -e .[dev,db,viz,advanced]
```

## Quick Start (5 steps)

1. Install dependencies (`pip install -e .[dev]`).
2. Launch Streamlit (`streamlit run app.py --server.port 5000`).
3. Load synthetic or experimental traces (`sample_data/` for reproducible examples).
4. Choose normalization mode explicitly (`simple`, `double`, `full_scale`).
5. Fit models, review quality flags, and export CSV/figures/reports.

## Advanced Analysis Modules (2025/2026)

- **Heterogeneity-aware population analysis**: Wasserstein/earth-mover population comparison with permutation testing in `frap2025.population_analysis.compare_populations_wasserstein`.
- **Condensate capillary wave spectroscopy**: FFT-based boundary mode analysis and equipartition-based relative surface tension estimation in `frap2025.material_mechanics.compute_capillary_waves`.
- **Non-parametric Bayesian SPT**: HDP-HMM-style analyzer with Pyro backend and automatic fallback in `frap2025.spt_models.HDPHMM_Analyzer`.
- **Physics-informed FRAP fitting**: reaction-diffusion PINN hooks with classical fallback in `frap2025.frap_fitting.fit_frap_pinn`.
- **Deep-learning optical flow hooks**: unified deformation-field interface with Farneback and optional RAFT backend in `frap2025.optical_flow_analysis.compute_deformation_field`.

All advanced modules use graceful degradation: when optional dependencies are missing, supported fallback methods are used and the application remains operational.

## Mathematical Models

| Model | Formula | Parameters | Assumptions | Primary Reference |
|---|---|---|---|---|
| Single exponential | `F(t)=baseline + M(1-exp(-t/tau_s))` | `M`, `tau_s`, `baseline` | Effective diffusion approximation, stationary geometry | Axelrod et al. 1976, DOI: 10.1016/S0006-3495(76)85755-4 |
| Double exponential | `F(t)=baseline + A1(1-exp(-t/tau1)) + A2(1-exp(-t/tau2))` | `A1,A2,tau1_s,tau2_s` | Two kinetic subpopulations | Axelrod-style extension |
| Soumpasis | `F(t)=baseline + M*exp(-tau_D/2t)[I0(z)+I1(z)]` | `tau_D_s`, `M`, `baseline` | Circular bleach, diffusion-dominant | Soumpasis 1983, DOI: 10.1016/S0006-3495(83)84410-5 |
| Reaction-diffusion | `F(t)=baseline + M[1 - (k_off/(k_on+k_off))e^{-(k_on+k_off)t} - (k_on/(k_on+k_off))e^{-k_D t}]` | `k_on_s,k_off_s,k_D_s,M` | Binding + diffusion terms separable | Sprague et al. 2004, DOI: 10.1529/biophysj.103.026765 |

Diffusion coefficient conventions used in code:

- `D_um2_per_s = w_um^2 / (4*tau_s)` (equivalently `D_um2_per_s = w_um^2*k_s/4`, `k_s=1/tau_s`)
- `D_um2_per_s = w_um^2*ln(2)/(4*t_half_s)` when converting from half-time
- Soumpasis half-time approximation: `D_um2_per_s ~= 0.2240 * w_um^2 / t_half_s`

## Known Limitations

- Circular bleach assumptions may fail for elliptical/nonuniform bleach profiles.
- Diffusion-only models are biased when strong binding/reaction kinetics dominate.
- Motion/drift artifacts can deflate apparent mobile fraction if uncorrected.
- Photobleaching during acquisition can bias plateau estimates without reference ROI correction.
- Model selection by AIC/BIC does not guarantee biological identifiability in low-SNR data.

## Documentation

- Methods:
  - `docs/methods/frap_models.md`
  - `docs/methods/normalization.md`
  - `docs/methods/statistics.md`
- User guides:
  - `docs/user_guide/getting_started.md`
  - `docs/user_guide/image_analysis.md`
  - `docs/user_guide/batch_processing.md`
- Development:
  - `docs/development/formula_audit.md`
  - `docs/development/call_graph.md`

## Citation

If you use FRAP2025 in published work, cite the software metadata in `CITATION.cff` and the relevant FRAP primary literature listed above.
