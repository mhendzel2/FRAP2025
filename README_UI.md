# FRAP Single-Cell Analysis - Streamlit UI

## Overview

A professional, interactive web application for exploring single-cell FRAP analysis results with:
- **Cohort management** - Build complex queries, save presets
- **Multi-level visualization** - Single-cell inspector, group analysis, multi-group comparisons
- **Interactive QC** - Live filtering by drift, R², motion artifacts
- **Statistical analysis** - Linear mixed models, bootstrap CIs, FDR correction
- **Reproducible exports** - Recipe stamping, figure presets, batch downloads

---

## Quick Start

### 1. Installation

Ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

Verify installation:

```bash
python verify_installation.py
```

### 2. Generate Example Data

Run the quick start examples to create test datasets:

```bash
python quick_start_singlecell.py
```

This creates several example datasets in `./output/`:
- `example1/` - Single synthetic movie (10 cells, 2 populations)
- `example2/` - Custom pipeline demonstration
- `example3/` - Multi-condition comparison (6 movies)
- `example4/` - Visualization showcase

### 3. Launch the UI

```bash
streamlit run streamlit_singlecell.py
```

The application will open in your browser at `http://localhost:8501`

### 4. Load Data

In the UI:
1. Click **"📂 Load Data"** in the left sidebar
2. Go to **"📊 Example Data"** tab
3. Select **"example3"** (multi-condition dataset)
4. Click **"📊 Load Example"**

You should now see:
- Summary header with cell counts
- Cohort builder with active filters
- Four main workspace tabs

---

## Features

### Left Rail Navigation

**Workflow Sections:**
- **📊 Data** → Load datasets, manage experiments
- **🔍 Cohort** → Build queries, save presets
- **✓ QC** → Quality control metrics
- **📈 Stats** → Statistical analysis
- **📤 Export** → Download results

**Quick Actions:**
- **📂 Load Data** → File upload or example selector
- **💾 Save Cohort** → Save current filters as preset
- **📋 Copy Recipe** → Export analysis configuration
- **🌙 Dark Mode** → Toggle theme

**Saved Cohorts:**
- Click to load
- Shows cell count
- Delete with 🗑 button

### Summary Header

Real-time metrics bar:
- **Cells** → Included count (with exclusions in delta)
- **Experiments** → Number of batches (exp_id)
- **Clusters** → Subpopulations identified
- **Outliers** → Count and percentage
- **Conditions** → Experimental groups
- **Recipe** → Configuration hash for reproducibility

### Cohort Builder

**Filters:**
- **Conditions** → Multi-select (e.g., control, treated)
- **Experiments** → Filter by batch
- **Clusters** → Select subpopulations (0, 1, 2...)
- **Include Outliers** → Show/hide statistical outliers
- **QC Passed Only** → Require good bleach detection
- **Include Noise** → Show cluster = -1 (unclustered)

**Active Filter Chips:**
Display current selections as removable tags

**Save Cohort:**
Enter name and click 💾 Save to create reusable preset

### Single-Cell Inspector

**Navigation Bar:**
```
[⬅ Prev]  [Cell Selector]  [Next ➡]  [🔖 Bookmark]  [📊 Compare]
```

**Left Pane - ROI Trajectory:**
- X-Y position plot colored by time
- QC badges: Drift, R², Motion flags
- Hover for frame details

**Right Pane - Recovery Curve:**
- Measured data (blue dots)
- Fitted exponential (red line)
- Residuals plot below

**Parameters Table:**
4-column display of mobile_frac, k, t½, I₀, I∞, model, AIC

### Group Workspace

**Spaghetti Plot:**
- Individual cells (gray, faint)
- Mean ± 95% CI (blue with ribbon)
- Normalize toggle

**Batch Effect Check:**
Small multiples showing each experiment separately

**Distribution Strips:**
For each parameter:
- Violin plot (density)
- Box plot (quartiles)
- Swarm (individual points)

### Multi-Group Comparisons

**Effect Size Heatmap:**
Matrix of Hedges' g for all parameter × comparison pairs
- Red → Positive effect
- Blue → Negative effect
- Color intensity → Magnitude

**Results Table:**
Detailed statistics with:
- Beta coefficients (LMM)
- Effect sizes (Hedges' g)
- p-values (raw)
- q-values (FDR-corrected)
- Significance markers

**Volcano Plot** (for ≥4 parameters):
Effect size vs. -log₁₀(p-value) with FDR threshold line

### QC Dashboard

**Metric Cards:**
- QC pass rate
- Median drift
- Median R²
- Motion artifact rate

**Interactive Histograms:**
Click to filter:
- Drift distribution (threshold at 10px)
- R² distribution (threshold at 0.8)
- Motion artifacts per cell

**Tracking Method Usage:**
Bar chart showing Gaussian vs LK vs Snake usage

### Export & Reporting

**Quick Exports:**
- Current cohort → CSV
- All traces → CSV
- Parquet format available

**Report Generation** (planned):
- PDF with tables and figures
- HTML with interactive plots
- Embedded recipe for reproducibility

**Figure Presets:**
Save current view configuration for reuse

**Analysis Recipe:**
JSON export of:
- Software versions
- Active filters
- Analysis parameters
- Timestamp and hash

---

## Workflows

### Workflow 1: Exploratory Analysis

1. **Load data** → Example3 (multi-condition)
2. **Check summary** → Verify cell counts, clusters
3. **QC dashboard** → Identify issues
4. **Filter outliers** → Uncheck "Include Outliers"
5. **Single-cell** → Inspect individual cells
6. **Group analysis** → Compare conditions
7. **Save cohort** → Name "qc_passed_only"

### Workflow 2: Statistical Comparison

1. **Load cohort** → "qc_passed_only"
2. **Multi-group** → Run comparisons
3. **Effect size heatmap** → Identify parameters
4. **Results table** → Check significance
5. **Export** → Download stats CSV
6. **Copy recipe** → Document analysis

### Workflow 3: QC Investigation

1. **QC dashboard** → Check distributions
2. **Click histogram** → Filter high-drift cells
3. **Single-cell** → Inspect failures
4. **Bookmark** → Mark interesting cases
5. **Compare** → Add to cell tray
6. **Export selection** → Save for re-analysis

### Workflow 4: Publication Figures

1. **Load cohort** → Final dataset
2. **Configure filters** → Desired subset
3. **Group workspace** → Generate spaghetti plot
4. **Save preset** → "Figure_2A_spaghetti"
5. **Multi-group** → Effect size heatmap
6. **Save preset** → "Figure_2B_heatmap"
7. **Generate report** → PDF with all figures
8. **Copy recipe** → Include in methods

---

## Data Requirements

### Required Files

**roi_traces.parquet** (or .csv)
Columns:
- `exp_id` (str) - Experiment identifier
- `movie_id` (str) - Movie identifier
- `cell_id` (int) - Unique cell ID
- `frame` (int) - Frame number
- `t` (float) - Time in seconds
- `x`, `y` (float) - ROI position
- `radius` (float) - ROI radius
- `signal_raw`, `signal_bg`, `signal_corr`, `signal_norm` (float) - Intensities
- `qc_motion` (bool) - Motion artifact flag
- `qc_reason` (str) - QC failure reason

**cell_features.parquet** (or .csv)
Columns:
- `cell_id` (int) - Unique cell ID
- `exp_id` (str) - Experiment identifier
- `condition` (str) - Experimental condition
- `pre_bleach`, `I0`, `I_inf` (float) - Intensity metrics
- `k`, `t_half`, `mobile_frac` (float) - Kinetic parameters
- `r2`, `sse` (float) - Fit quality
- `drift_px` (float) - Total drift
- `bleach_qc` (bool) - QC passed flag
- `outlier` (bool) - Statistical outlier flag
- `cluster` (int) - Subpopulation label
- `cluster_prob` (float) - Cluster confidence
- Additional: `A`, `B`, `fit_method`, `aic`, `bic`

### Generating from Raw Data

If you have raw FRAP movies:

```python
from frap_singlecell_api import analyze_frap_movie
import tifffile

# Load movie
movie = tifffile.imread('path/to/movie.tif')  # Shape: (T, H, W)
time_points = np.arange(len(movie)) * 0.5  # seconds

# Analyze
roi_traces, cell_features = analyze_frap_movie(
    movie=movie,
    time_points=time_points,
    exp_id='exp001',
    movie_id='movie01',
    condition='control',
    output_dir='./output/exp001'  # Saves parquet files
)

# Results automatically saved to:
#   ./output/exp001/roi_traces.parquet
#   ./output/exp001/cell_features.parquet
```

Then load `./output/exp001/` in the UI.

### Combining Multiple Movies

```python
from frap_data_model import DataIO
import pandas as pd

# Analyze multiple movies
all_traces = []
all_features = []

for movie_file, condition in zip(movie_files, conditions):
    traces, features = analyze_frap_movie(
        movie=tifffile.imread(movie_file),
        time_points=time_points,
        exp_id=f'exp_{i:03d}',
        movie_id=movie_file.stem,
        condition=condition
    )
    all_traces.append(traces)
    all_features.append(features)

# Combine
combined_traces = pd.concat(all_traces, ignore_index=True)
combined_features = pd.concat(all_features, ignore_index=True)

# Save
io = DataIO()
io.save_tables(combined_traces, combined_features, './combined_dataset')

# Load in UI: ./combined_dataset/
```

---

## Performance Tips

### For Large Datasets (>1000 cells)

1. **Pre-filter** before loading:
   ```python
   # Keep only QC-passed cells
   features = features[features['bleach_qc'] == True]
   traces = traces[traces['cell_id'].isin(features['cell_id'])]
   ```

2. **Use Parquet** format (10× faster than CSV):
   ```python
   io.save_tables(traces, features, output_dir, format='parquet')
   ```

3. **Down-sample** traces for display:
   ```python
   # Keep every 5th frame for visualization
   traces_display = traces.groupby('cell_id').apply(
       lambda g: g.iloc[::5]
   ).reset_index(drop=True)
   ```

4. **Reduce bootstrap** iterations:
   ```python
   analyze(..., n_bootstrap=200)  # Instead of 1000
   ```

### For Slow Loading

- Check file sizes (traces > 100MB may be slow)
- Close other browser tabs
- Use Chrome/Edge (better WebAssembly support)
- Increase RAM allocation if using Docker

### For Unresponsive UI

- Click "Stop" in top-right corner
- Refresh browser (F5)
- Check terminal for errors
- Reduce data size or filters

---

## Troubleshooting

### "Import streamlit could not be resolved"

**Solution:**
```bash
pip install streamlit plotly
```

Verify:
```bash
streamlit --version
```

### "No data loaded"

**Solution:**
1. Run `python quick_start_singlecell.py` to generate examples
2. Click "📂 Load Data" → "📊 Example Data" tab
3. Select "example3" and load

### "No module named 'frap_singlecell_api'"

**Solution:**
Ensure you're in the correct directory:
```bash
cd c:\Users\mjhen\Github\FRAP2025
streamlit run streamlit_singlecell.py
```

### Plots not rendering

**Solution:**
1. Check browser console (F12) for errors
2. Try different browser (Chrome recommended)
3. Disable browser extensions
4. Clear browser cache

### "Need at least 2 conditions"

**Solution:**
Multi-group comparisons require ≥2 conditions.

Check data:
```python
print(cell_features['condition'].unique())
```

Add conditions if missing:
```python
cell_features['condition'] = ['control'] * n_control + ['treated'] * n_treated
```

### App is slow

**Solutions:**
1. Reduce cohort size with filters
2. Use Parquet format
3. Reduce bootstrap iterations (edit code: n_bootstrap=200)
4. Close other Streamlit apps
5. Use `streamlit run --server.maxUploadSize=1000` for large files

### Data validation errors

**Solution:**
Check required columns:
```python
from frap_data_model import validate_roi_traces, validate_cell_features

valid, msg = validate_roi_traces(traces)
print(f"Traces valid: {valid}, {msg}")

valid, msg = validate_cell_features(features)
print(f"Features valid: {valid}, {msg}")
```

---

## Advanced Usage

### Custom Color Schemes (Planned)

Edit `streamlit_singlecell.py`:

```python
# Custom colors
CLUSTER_COLORS = ['#0173B2', '#DE8F05', '#029E73', '#CC78BC']
SIGNIFICANCE_COLOR = '#CC0000'
MEAN_COLOR = '#0066CC'
```

### Adding Custom Plots

Create new function in `frap_visualizations.py`:

```python
def plot_custom_metric(data, param):
    fig = go.Figure()
    # ... your plot code
    return fig
```

Then add to UI in `streamlit_singlecell.py`:

```python
with tab5:
    st.markdown("### Custom Analysis")
    fig = plot_custom_metric(df_cohort, 'my_param')
    st.plotly_chart(fig, use_container_width=True)
```

### Embedding in Other Apps

The UI can be embedded as an iframe:

```html
<iframe src="http://localhost:8501" width="100%" height="800px"></iframe>
```

Or use Streamlit's component API for deeper integration.

### Deployment

**Local network:**
```bash
streamlit run streamlit_singlecell.py --server.address 0.0.0.0 --server.port 8501
```

Access from other machines: `http://<your-ip>:8501`

**Cloud deployment:**
- Streamlit Cloud (free for public repos)
- Heroku
- AWS EC2 with Docker
- Azure Web Apps

See Streamlit documentation for details.

---

## Keyboard Shortcuts

- `Ctrl/Cmd + K` - Command palette (planned)
- `Tab` - Navigate between fields
- `Enter` - Submit forms, activate buttons
- `Esc` - Close modals
- `F5` - Refresh app
- `Ctrl/Cmd + R` - Rerun analysis

---

## API Reference

### Session State Variables

```python
st.session_state.roi_traces          # DataFrame of time series
st.session_state.cell_features       # DataFrame of fitted params
st.session_state.active_filters      # Dict of current filters
st.session_state.selected_cells      # List of selected cell IDs
st.session_state.bookmarked_cells    # List of bookmarked cells
st.session_state.cohorts             # Dict of saved cohorts
st.session_state.recipe              # Analysis configuration dict
```

### Key Functions

```python
build_cohort_query()                 # Apply filters, return DataFrame
save_cohort(name)                    # Save filters as preset
load_cohort(name)                    # Restore saved preset
compute_recipe_hash()                # Get 8-char hash of config
export_current_cohort(format='csv')  # Download filtered data
export_traces(cell_ids, format='csv') # Download trace data
```

---

## Contributing

### Planned Features

- [ ] Linked selection (brushing, lasso)
- [ ] Cell tray for comparison
- [ ] Video frame viewer with ROI overlay
- [ ] Time scrubber for re-fitting
- [ ] Command palette (Ctrl+K)
- [ ] PDF/HTML report generation
- [ ] Volcano plots for many parameters
- [ ] Tracking method timeline
- [ ] Cluster justification plots (silhouette, BIC)
- [ ] Figure preset templates
- [ ] Batch analysis mode

### Submitting Issues

Include:
- Streamlit version: `streamlit --version`
- Python version: `python --version`
- OS and browser
- Error message or screenshot
- Minimal reproducible example

### Pull Requests

1. Fork repository
2. Create feature branch
3. Add tests if applicable
4. Update documentation
5. Submit PR with description

---

## License

Same as parent project (see LICENSE file)

---

## Support

- **Documentation**: See `UI_GUIDE.md` for detailed usage
- **Examples**: Run `python quick_start_singlecell.py`
- **API Reference**: See `README_SINGLECELL.md`
- **Implementation**: See `SINGLECELL_IMPLEMENTATION.md`

---

*Last updated: 2025-10-04*
*UI Version: 1.0*
