# FRAP Single-Cell Analysis System

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from frap_singlecell_api import analyze_frap_movie
import numpy as np

# Load your movie data (T, H, W)
movie = np.load('your_movie.npy')
time_points = np.arange(movie.shape[0]) * 0.5  # time in seconds

# Run complete analysis
results = analyze_frap_movie(
    movie=movie,
    time_points=time_points,
    exp_id='experiment_001',
    movie_id='movie_001',
    condition='control',
    output_dir='./results'
)

print(f"Analyzed {results['n_cells']} cells")
print(f"Found {results['n_clusters']} distinct populations")
print(f"Flagged {results['n_outliers']} outliers")
```

## Features

### 1. Advanced ROI Tracking
- **Sub-pixel accuracy**: 2D Gaussian fitting for centroid localization
- **Kalman filtering**: Smooth trajectories with innovation-based QC
- **Adaptive radius**: Dynamic ROI sizing based on gradient energy
- **Multi-method**: Gaussian fitting + optical flow fallback
- **Multi-ROI**: Automatic detection and tracking of multiple cells

### 2. Robust Curve Fitting
- **Robust loss**: Soft-L1 loss for outlier resistance
- **Model selection**: Automatic 1-exp vs 2-exp using BIC (Δ>10)
- **Parallel processing**: Fast fitting with joblib
- **Derived parameters**: Mobile fraction, t½, diffusion coefficient

### 3. Population Analysis
- **Outlier detection**: Ensemble of IsolationForest + EllipticEnvelope
- **Clustering**: GMM with BIC selection, DBSCAN fallback
- **Subpopulations**: Identify heterogeneous kinetic behaviors
- **Quality metrics**: Silhouette score, Calinski-Harabasz, Davies-Bouldin

### 4. Statistical Analysis
- **Linear mixed models**: Account for batch effects with random intercepts
- **Bootstrap confidence intervals**: BCa method for accurate CIs
- **Effect sizes**: Hedges' g with confidence intervals
- **Multiple testing**: FDR correction (Benjamini-Hochberg)
- **Permutation tests**: Non-parametric alternative

### 5. Visualization
- **Spaghetti plots**: Individual traces + mean with bootstrap ribbon
- **Heatmaps**: Time × cell with hierarchical clustering
- **Estimation plots**: Gardner-Altman difference plots (dabest)
- **Pair plots**: Multi-dimensional feature space exploration
- **QC dashboard**: Comprehensive quality control metrics

## Module Overview

| Module | Purpose |
|--------|---------|
| `frap_data_model.py` | Data structures (ROITrace, CellFeatures) and I/O |
| `frap_tracking.py` | ROI tracking with Kalman filtering |
| `frap_signal.py` | Signal extraction and normalization |
| `frap_fitting.py` | Exponential curve fitting |
| `frap_populations.py` | Outlier detection and clustering |
| `frap_statistics.py` | LMM, bootstrap, effect sizes |
| `frap_visualizations.py` | Plotting functions |
| `frap_singlecell_api.py` | High-level API |
| `test_synthetic.py` | Synthetic data generator |
| `test_frap_singlecell.py` | Unit and integration tests |

## API Reference

### Core Functions

#### `track_movie(movie, time_points, exp_id, movie_id, **kwargs)`
Track ROIs across entire movie.

**Parameters:**
- `movie`: ndarray (T, H, W) - Movie frames
- `time_points`: ndarray - Time for each frame
- `exp_id`: str - Experiment identifier
- `movie_id`: str - Movie identifier
- `use_kalman`: bool - Use Kalman smoothing (default: True)
- `adapt_radius_flag`: bool - Adapt ROI radius (default: True)

**Returns:** DataFrame with per-frame ROI measurements

#### `fit_cells(roi_traces, robust=True, try_2exp=True, n_jobs=-1)`
Fit recovery curves for all cells.

**Parameters:**
- `roi_traces`: DataFrame - ROI traces from tracking
- `robust`: bool - Use robust fitting (soft_l1)
- `try_2exp`: bool - Try 2-exponential model
- `n_jobs`: int - Parallel jobs (-1 for all cores)

**Returns:** DataFrame with per-cell fitted parameters

#### `detect_outliers_and_clusters(cell_features, contamination=0.07, max_k=6)`
Population analysis pipeline.

**Parameters:**
- `cell_features`: DataFrame - Cell features
- `contamination`: float - Expected outlier fraction
- `max_k`: int - Maximum clusters to try

**Returns:** Updated DataFrame with outlier and cluster columns

#### `analyze(cell_features, group_col='condition', batch_col='exp_id', params=None)`
Complete statistical analysis with FDR correction.

**Parameters:**
- `cell_features`: DataFrame - Cell features with group/batch info
- `group_col`: str - Column for experimental groups
- `batch_col`: str - Column for batch/experiment ID
- `params`: list - Parameters to analyze

**Returns:** dict with comparison results and cluster statistics

## Data Structures

### ROI Traces Table
Per-frame measurements for each cell:

| Column | Type | Description |
|--------|------|-------------|
| exp_id | str | Experiment ID |
| movie_id | str | Movie ID |
| cell_id | int | Cell ID |
| frame | int | Frame number |
| t | float | Time (s) |
| x, y | float | ROI position (px) |
| radius | float | ROI radius (px) |
| signal_raw | float | Raw intensity |
| signal_bg | float | Background intensity |
| signal_corr | float | Background-corrected signal |
| signal_norm | float | Normalized signal |
| qc_motion | bool | Motion artifact flag |
| qc_reason | str | QC failure reason |

### Cell Features Table
Per-cell derived parameters:

| Column | Type | Description |
|--------|------|-------------|
| exp_id | str | Experiment ID |
| movie_id | str | Movie ID |
| cell_id | int | Cell ID |
| pre_bleach | float | Pre-bleach intensity |
| I0 | float | Post-bleach intensity |
| I_inf | float | Plateau intensity |
| k | float | Recovery rate (s⁻¹) |
| t_half | float | Half-time (s) |
| mobile_frac | float | Mobile fraction |
| r2 | float | Fit R² |
| sse | float | Sum of squared errors |
| drift_px | float | Total drift (px) |
| bleach_qc | bool | Pass quality control |
| roi_method | str | Tracking method used |
| outlier | bool | Outlier flag |
| cluster | int | Cluster assignment |
| cluster_prob | float | Cluster probability |

## Example Workflows

### Workflow 1: Single Movie Analysis

```python
from frap_singlecell_api import analyze_frap_movie

results = analyze_frap_movie(
    movie=my_movie,
    time_points=times,
    exp_id='exp1',
    movie_id='mov1',
    condition='control',
    output_dir='./output'
)

# Results saved to ./output/roi_traces.parquet and cell_features.parquet
```

### Workflow 2: Multi-Condition Comparison

```python
from frap_singlecell_api import track_movie, fit_cells, analyze
from frap_populations import detect_outliers_and_clusters
from frap_data_model import DataIO
import pandas as pd

# Process multiple movies
all_traces = []
all_features = []

for movie_data in my_dataset:
    traces = track_movie(movie_data['movie'], movie_data['times'], 
                        movie_data['exp_id'], movie_data['movie_id'])
    features = fit_cells(traces)
    features['condition'] = movie_data['condition']
    
    all_traces.append(traces)
    all_features.append(features)

# Combine
combined_features = pd.concat(all_features, ignore_index=True)

# Population analysis
combined_features = detect_outliers_and_clusters(combined_features)

# Statistical comparison
stats = analyze(
    combined_features,
    group_col='condition',
    batch_col='exp_id',
    params=['mobile_frac', 'k', 't_half']
)

# View results
print(stats['comparisons'])
```

### Workflow 3: Custom Pipeline with Visualization

```python
from frap_singlecell_api import track_movie, fit_cells
from frap_visualizations import plot_spaghetti, plot_qc_dashboard, save_all_figures

# Track and fit
traces = track_movie(movie, times, 'exp1', 'mov1')
features = fit_cells(traces)
features['condition'] = 'control'

# Generate plots
fig1 = plot_spaghetti(traces, features, 'control')
fig2 = plot_qc_dashboard(traces, features)

# Save all figures
save_all_figures('./figures', traces, features)
```

## Testing

Run unit tests:
```bash
pytest test_frap_singlecell.py -v
```

Generate synthetic test data:
```python
from test_synthetic import generate_test_movies_with_expectations

test_data = generate_test_movies_with_expectations()
movie = test_data['simple']['movie']
ground_truth = test_data['simple']['ground_truth']
```

## Configuration

Key parameters with defaults:

```python
# Tracking
use_kalman = True              # Kalman filtering
process_var = 1.0              # Kalman process variance
meas_var = 2.0                 # Kalman measurement variance
adapt_radius_flag = True       # Adaptive ROI radius

# Fitting
robust = True                  # Robust fitting (soft_l1)
try_2exp = True                # Try 2-exponential
bic_threshold = 10.0           # BIC threshold for model selection

# Population analysis
contamination = 0.07           # Expected outlier fraction
max_k = 6                      # Maximum clusters

# Statistics
n_bootstrap = 1000             # Bootstrap iterations
confidence = 0.95              # CI level
fdr_method = 'fdr_bh'          # FDR correction method
```

## Performance

- **Tracking**: ~10-50 ms per frame per cell
- **Fitting**: <500 ms per cell (with timeout)
- **Parallel**: Scales linearly with cores
- **Memory**: ~100 MB per 1000 frames at 512×512

## Troubleshooting

### Issue: Poor tracking accuracy
- Increase `window` parameter in Gaussian fitting
- Enable Kalman filtering: `use_kalman=True`
- Check for motion artifacts in source data

### Issue: Fit failures
- Check bleach frame detection
- Ensure sufficient post-bleach frames (≥10)
- Try disabling 2-exp: `try_2exp=False`

### Issue: No clusters found
- Check feature variance (may be single population)
- Reduce `max_k` if overfitting
- Try DBSCAN parameters manually

### Issue: High outlier rate
- Adjust `contamination` parameter
- Check data quality and QC flags
- Review fitting results (R²)

## Citation

If you use this software, please cite:
[Your publication details]

## License

[Your license]

## Support

For questions and bug reports, please open an issue on GitHub or contact [your contact].

## See Also

- `SINGLECELL_IMPLEMENTATION.md` - Detailed implementation notes
- Existing FRAP modules: `frap_core.py`, `frap_image_analysis.py`
- Streamlit UI: `streamlit_frap_final_clean.py`
