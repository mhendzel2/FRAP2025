# FRAP Single-Cell Analysis Implementation Summary

## Overview
This document summarizes the implementation of a comprehensive single-cell FRAP analysis system based on the provided specification.

## Implementation Status

### ✅ Completed Modules

#### 1. Dependencies (requirements.txt)
- Added all required packages: numpy, scipy, pandas, statsmodels, scikit-learn, scikit-image, matplotlib
- Single-cell extensions: dabest, joblib, filterpy, pymc, pytest, pyarrow
- All dependencies support Python ≥3.10

#### 2. Data Model (`frap_data_model.py`)
- **ROITrace** dataclass: Per-frame measurements with all required fields
- **CellFeatures** dataclass: Per-cell derived parameters
- **DataIO** class: Save/load functions for Parquet and CSV formats
- Validation functions for schema checking

#### 3. ROI Tracking (`frap_tracking.py`)
**§2.1-2.4: Core Tracking**
- `fit_gaussian_centroid()`: Sub-pixel centroid via 2D Gaussian fitting
- `ROIKalman` class: 2D Kalman filter with constant velocity model
  - Uses filterpy if available, custom implementation as fallback
  - Tracks innovation norm for QC
- `adapt_radius()`: Adaptive radius based on gradient energy
- `track_optical_flow()`: Lucas-Kanade optical flow fallback

**§2.5-2.6: Multi-ROI Support**
- `seed_rois()`: Automatic ROI detection via Otsu → watershed
- `hungarian_assignment()`: ROI linking across frames with cost matrix
- `evolve_mask()`: Active contour mask evolution
- `compute_mask_metrics()`: IoU and Hausdorff distance for QC

#### 4. Signal Extraction (`frap_signal.py`)
- `extract_signal_with_background()`: Annulus-based background with outlier-robust median
- `compute_pre_bleach_intensity()`: Pre-bleach averaging
- `find_bleach_frame()`: Multiple methods (min, derivative)
- `detect_motion_artifacts()`: Frame-to-frame displacement analysis
- Additional utilities: photobleaching correction, baseline correction, SNR

#### 5. Curve Fitting (`frap_fitting.py`)
- **FitResult** dataclass: Complete fit results with metadata
- `fit_recovery()`: Robust 1-exp fitting with soft_l1 loss
- `fit_recovery_2exp()`: Double exponential model
- `select_best_model()`: BIC-based model selection (ΔBIC > 10 threshold)
- `fit_cells_parallel()`: Parallel fitting with joblib
- `compute_mobile_fraction()`: With bounds checking
- `compute_fit_diagnostics()`: RMSE, Durbin-Watson, MAPE

#### 6. Population Analysis (`frap_populations.py`)
- `flag_outliers()`: Ensemble of IsolationForest + EllipticEnvelope
- `gmm_clusters()`: Gaussian Mixture Model with BIC selection
- `dbscan_fallback()`: DBSCAN when GMM finds single cluster
- `detect_outliers_and_clusters()`: Complete pipeline
- `compute_cluster_statistics()`: Per-cluster summaries
- `compute_separation_metrics()`: Silhouette, Calinski-Harabasz, Davies-Bouldin

#### 7. Statistics (`frap_statistics.py`)
- `lmm_param()`: Linear mixed-effects model using statsmodels
  - Random intercept by batch/experiment
  - Returns beta, SE, 95% CI, p-value, Hedges' g, omega-squared
- `bootstrap_bca_ci()`: Bias-corrected and accelerated bootstrap
- `multi_parameter_analysis()`: Multiple parameters with FDR correction (Benjamini-Hochberg)
- `compute_effect_size_ci()`: Bootstrap CI for effect sizes
- `permutation_test()`: Non-parametric alternative

#### 8. API Surface (`frap_singlecell_api.py`)
Unified public API:
- `track_movie()`: Complete ROI tracking pipeline
- `fit_cells()`: Batch curve fitting
- `flag_motion()`: Motion artifact flagging
- `analyze()`: Statistical analysis with FDR
- `analyze_frap_movie()`: End-to-end convenience function

#### 9. Test Data Generator (`test_synthetic.py`)
- `synth_movie()`: Generate movies with two subpopulations
  - Controlled drift, noise, recovery kinetics
  - Ground truth tracking
- `synth_multi_movie_dataset()`: Multi-experiment datasets
- `add_realistic_artifacts()`: Photobleaching, motion spikes
- `generate_test_movies_with_expectations()`: Pre-configured test cases

## Remaining Work

### High Priority

#### 1. Visualization Suite (`frap_visualizations.py`)
**Required functions:**
- `plot_spaghetti()`: Per-cell traces with mean ± bootstrap ribbon
- `plot_heatmap()`: Time × cell with hierarchical clustering
- `plot_estimation()`: Gardner-Altman plots using dabest
- `plot_pairplots()`: Scatter matrices colored by cluster
- `plot_qc_dashboard()`: Residuals, drift, IoU distributions

#### 2. Comprehensive Tests (`test_frap_singlecell.py`)
**Test coverage:**
- Tracking accuracy: MAE < 0.5 px on synthetic data
- Fitting accuracy: k RMSE < 10% of truth
- Clustering: ARI > 0.7 for known populations
- Statistics: Effect detection p < 0.01 for Δ = 0.15
- Visualization smoke tests

#### 3. UI Integration
**Streamlit tab additions:**
- Single-cell curve viewer with fit and residuals
- ROI trajectory sparklines
- Filters: outliers, clusters, batches
- Export: CSV for traces/features, PDF reports

#### 4. Reporting (`frap_singlecell_reports.py`)
**Generate PDF/HTML reports:**
- Tables: N, means ± SD, effect sizes, CIs, p, q
- Figures: All visualization types
- Methods: Tracking, fitting, clustering, stats details

### Medium Priority

#### 5. Advanced Features
- Hierarchical Bayesian models with PyMC (optional, if available)
- Nonlinear mixed effects via R/lme4 through rpy2 (optional)
- Curve-level comparison instead of parameter-level
- Additional QC metrics and automatic thresholding

#### 6. Performance Optimization
- Vectorize image operations further
- Cache gradient computations
- Implement timeouts for slow fits (500 ms threshold)
- Memory-efficient handling of large datasets

#### 7. Migration and Backfill
- Scripts to convert existing FRAP data to new schema
- Batch processing utilities
- Validation against old pipeline

## Key Features Implemented

### Robustness
- Multiple tracking methods with automatic fallback
- Robust curve fitting with soft_l1 loss
- Outlier detection before clustering
- QC flags at multiple levels (frame, cell, fit)

### Statistical Rigor
- Linear mixed-effects models for batch effects
- BCa bootstrap for accurate confidence intervals
- FDR correction for multiple testing
- Effect sizes (Hedges' g) with CIs

### Scalability
- Parallel processing with joblib
- Multi-ROI support
- Efficient data structures (Parquet)
- Designed for 10-100s of cells per movie

### Extensibility
- Clean module separation
- Type hints throughout
- Logging at all levels
- Comprehensive docstrings

## API Usage Examples

### Basic Single Movie Analysis
```python
from frap_singlecell_api import analyze_frap_movie
import numpy as np

# Load movie (T, H, W)
movie = np.load('movie.npy')
time_points = np.arange(movie.shape[0]) * 0.5  # seconds

results = analyze_frap_movie(
    movie=movie,
    time_points=time_points,
    exp_id='exp001',
    movie_id='movie001',
    condition='control',
    output_dir='./output'
)

print(f"Tracked {results['n_cells']} cells")
print(f"Found {results['n_clusters']} clusters")
print(f"Flagged {results['n_outliers']} outliers")
```

### Multi-Experiment Statistical Analysis
```python
from frap_singlecell_api import analyze
from frap_data_model import DataIO

# Load combined cell features from multiple experiments
cell_features, _ = DataIO.load_tables('./combined_data')

# Run statistical analysis
stats = analyze(
    cell_features=cell_features,
    group_col='condition',
    batch_col='exp_id',
    params=['mobile_frac', 'k', 't_half'],
    n_bootstrap=1000
)

# View results
print(stats['comparisons'])
# Shows: param, comparison, beta, p, q, hedges_g, CI
```

### Custom Pipeline
```python
from frap_singlecell_api import track_movie, fit_cells, flag_motion
from frap_populations import detect_outliers_and_clusters

# Track
roi_traces = track_movie(movie, time_points, 'exp1', 'mov1', 
                         use_kalman=True, adapt_radius_flag=True)

# Flag motion
roi_traces = flag_motion(roi_traces, threshold_px=5.0)

# Fit
cell_features = fit_cells(roi_traces, robust=True, try_2exp=True)

# Population analysis
cell_features = detect_outliers_and_clusters(
    cell_features, 
    contamination=0.07, 
    max_k=6
)
```

## Testing Strategy

### Unit Tests
- Individual function tests with known inputs/outputs
- Edge cases (empty data, single cell, failures)
- Numerical accuracy checks

### Integration Tests
- End-to-end on synthetic movies with ground truth
- Verify acceptance criteria from §14:
  - Tracking MAE < 0.5 px
  - k RMSE < 10%
  - Cluster ARI > 0.7
  - Effect detection p < 0.01 for Δ = 0.15

### Performance Tests
- Time limits for fitting (< 500 ms per cell)
- Memory usage for large datasets
- Parallel scaling efficiency

## File Structure
```
frap_data_model.py          # Data structures and I/O
frap_tracking.py            # ROI tracking (§2)
frap_signal.py              # Signal extraction (§3)
frap_fitting.py             # Curve fitting (§4)
frap_populations.py         # Outliers and clustering (§5)
frap_statistics.py          # Statistical analysis (§6)
frap_visualizations.py      # Plotting functions (§7) [TODO]
frap_singlecell_api.py      # Public API (§12)
frap_singlecell_reports.py  # Report generation (§9) [TODO]
test_synthetic.py           # Synthetic data generator (§11.1)
test_frap_singlecell.py     # Comprehensive tests (§11) [TODO]
requirements.txt            # Dependencies
```

## Configuration and Reproducibility

### Fixed Seeds
All random operations accept `random_state` parameter for reproducibility.

### Version Tracking
Report footer should include:
- Python version
- Package versions (numpy, scipy, scikit-learn, etc.)
- Analysis date
- Parameter settings

### Parameter Defaults
Key tunable parameters with sensible defaults:
- Kalman process variance: 1.0
- Kalman measurement variance: 2.0
- Outlier contamination: 0.07 (7%)
- Max clusters: 6
- BIC threshold for 2-exp: 10
- Motion threshold: 5 pixels
- Bootstrap iterations: 1000
- FDR method: 'fdr_bh'

## Next Steps

1. **Implement visualizations** (`frap_visualizations.py`)
2. **Write comprehensive tests** (`test_frap_singlecell.py`)
3. **Create reporting module** (`frap_singlecell_reports.py`)
4. **Integrate with existing Streamlit UI**
5. **Run validation on real data**
6. **Optimize performance** (profiling, caching)
7. **Documentation** (user guide, API reference)

## Acceptance Criteria Progress

- [x] End-to-end data structures (§1)
- [x] ROI tracking with QC (§2)
- [x] Signal extraction (§3)
- [x] Robust curve fitting with model selection (§4)
- [x] Outlier detection and clustering (§5)
- [x] LMM statistics with FDR (§6)
- [ ] Visualization suite (§7)
- [ ] UI integration (§8)
- [ ] Report generation (§9)
- [x] Synthetic test generator (§11.1)
- [ ] Comprehensive tests (§11)
- [x] Clean API surface (§12)

## Notes

- PyMC for hierarchical Bayesian models is optional; parameter-then-LMM approach is primary
- R integration via rpy2 is optional; statsmodels MixedLM is primary
- Filterpy for Kalman is optional; custom implementation included
- All critical functionality has fallbacks for missing optional dependencies
