# FRAP Single-Cell Analysis UI Guide

## Overview

The new interactive UI (`streamlit_singlecell.py`) provides a professional analysis platform with:
- **Cohort management** with saved presets
- **Linked selection** across all views
- **Interactive gating** for outliers and clusters
- **Multi-level workspaces** (single-cell, group, multi-group)
- **Comprehensive QC** with live filtering
- **Export capabilities** with reproducible analysis recipes

---

## Quick Start

### 1. Launch the Application

```bash
streamlit run streamlit_singlecell.py
```

The app will open in your browser at `http://localhost:8501`

### 2. Load Data

You have two options:

**Option A: Load existing analysis results**
```python
# In Python, save your analysis
from frap_data_model import DataIO

io = DataIO()
io.save_tables(roi_traces, cell_features, output_dir="./data")
```

Then use the "📂 Load Data" button in the UI sidebar.

**Option B: Generate synthetic test data**
```python
# Run the quick start examples first
python quick_start_singlecell.py

# This creates ./output/example1/ with roi_traces.parquet and cell_features.parquet
```

---

## Information Architecture

### Left Rail Navigation

The left sidebar provides workflow navigation:

- **📊 Data** → Load datasets and manage experiments
- **🔍 Cohort** → Build queries and filter cells
- **✓ QC** → Quality control metrics and gating
- **📈 Stats** → Statistical analysis and comparisons
- **📤 Export** → Download data and reports

### Persistent Summary Header

Shows real-time metrics for your current cohort:
- **Cells**: Included (with excluded count in delta)
- **Experiments**: Number of unique exp_ids
- **Clusters**: Number of identified subpopulations
- **Outliers**: Count and percentage
- **Conditions**: Number of experimental groups
- **Recipe**: Hash of current analysis configuration

### Three Main Workspaces

Top tabs provide different granularity:

1. **🔬 Single-cell** → Detailed inspector for individual cells
2. **📊 Group** → Condition-level aggregation with spaghetti plots
3. **🔬 Multi-group** → Pairwise statistical comparisons
4. **🔍 QC Dashboard** → Quality metrics and filtering

---

## Cohort Builder

### Query Interface

**Row 1: Primary Filters**
- **Conditions** → Multi-select experimental conditions
- **Experiments** → Filter by exp_id (batch)
- **Clusters** → Select specific subpopulations

**Row 2: Toggle Filters**
- **Include Outliers** → Show/hide statistical outliers
- **QC Passed Only** → Require bleach_qc = True
- **Include Noise** → Show cluster = -1 (unclustered)

### Active Filter Chips

Filters display as removable chips:
```
condition: control, treatment
No outliers
QC passed
```

Click "Clear All" to reset filters.

### Save & Load Cohorts

1. Apply your desired filters
2. Enter a name in "Cohort name" field
3. Click "💾 Save"

Saved cohorts appear in the sidebar with cell count:
```
[baseline (145)]  [🗑]
[treated (132)]   [🗑]
```

Click to load, click 🗑 to delete.

---

## Single-Cell Inspector

### Navigation

```
[⬅ Prev]  [Cell Selector ▼]  [Next ➡]  [🔖]  [📊 Compare]
```

- **Prev/Next** → Cycle through cohort cells
- **Bookmark 🔖** → Save interesting cells (turns to 📌)
- **Compare 📊** → Add to cell tray for side-by-side

### Two-Pane Layout

**Left Pane: ROI Trajectory**
- X-Y position plot with time-colored markers
- Hover shows frame number and coordinates
- QC metrics below:
  - **Drift** → Total displacement (⚠ if >10px)
  - **R²** → Fit quality (✓ if >0.8)
  - **Motion Flags** → Frames with detected artifacts

**Right Pane: Recovery Curve**
- Scatter: Measured intensities (blue dots)
- Line: Fitted exponential (red)
- Residuals plot below (gray dots around zero line)

### Time Scrubber

*(Planned feature)*
Brush a time window to re-fit only that region, useful for:
- Excluding photo-damage events
- Focusing on early/late recovery
- Testing model sensitivity

### Fitted Parameters

Four columns display:
- **Column 1**: Mobile Fraction, k (rate constant)
- **Column 2**: t½ (half-life), I₀ (bleach minimum)
- **Column 3**: I∞ (plateau), Pre-bleach intensity
- **Column 4**: Model (1-exp/2-exp), AIC

---

## Group Workspace

### Spaghetti Plot

Shows all cells in the selected condition:
- **Gray lines** (α=0.1) → Individual cells
- **Blue line** → Mean trajectory
- **Light blue ribbon** → 95% confidence interval (bootstrap)

**Controls:**
- [Normalize per cell] → Checkbox for per-cell (I-I₀)/(I∞-I₀)
- Hover → Shows time-aligned values

### Small Multiples (Batch Check)

Grid of mini-plots, one per experiment:
```
[exp_001]  [exp_002]  [exp_003]
```

Visual batch effect check:
- Similar shapes → Good
- Shifted means → Batch effect present
- Different dynamics → Experimental artifact

### Distribution Strips

For each parameter (mobile_frac, k, t½):
- **Violin plot** → Density distribution
- **Box plot overlay** → Quartiles and median line
- **Swarm points** → Individual cells with jitter

**Gardner-Altman Estimation** *(planned)*
Right panel shows:
- Effect size with 95% CI
- Raw difference distributions
- Unpaired/paired comparison

---

## Multi-Group Comparisons

### Pairwise Effect Size Matrix

Heatmap shows Hedges' g for all parameter × comparison pairs:
```
                control vs treated    control vs KO
mobile_frac          0.85*                1.23*
k                   -0.42                 0.67*
t_half              -0.55*               -0.98*
```

- **Red** → Positive effect (treatment increases)
- **Blue** → Negative effect (treatment decreases)
- **White** → No effect
- Numbers = Hedges' g with small sample correction

Click a cell to see underlying distributions.

### Detailed Results Table

Columns:
- **param** → Parameter name
- **comparison** → Group A vs Group B
- **beta** → LMM coefficient
- **hedges_g** → Effect size
- **p** → Raw p-value
- **q** → FDR-corrected q-value (Benjamini-Hochberg)
- **significant** → ✓ if q < 0.05

### Volcano Plot

For many parameters (≥4), shows:
- **X-axis**: Effect size (Hedges' g)
- **Y-axis**: -log₁₀(p-value)
- **Horizontal line**: FDR = 0.05 threshold
- **Red points**: Significant (q < 0.05)
- **Gray points**: Not significant

Hover for parameter name and exact values.

---

## Outliers and Subpopulations

### Current Implementation

Outliers and clusters are computed during analysis:

```python
from frap_populations import detect_outliers_and_clusters

detect_outliers_and_clusters(
    cell_features,
    contamination=0.07,  # Expected outlier fraction
    max_k=6              # Max clusters to try
)
```

Adds columns:
- `outlier` → Boolean flag
- `cluster` → Integer label (0, 1, 2, ... or -1 for noise)
- `cluster_prob` → Confidence score

### Planned UI Features

**Outlier Gate Slider**
```
Contamination: [====|=====] 0.07
               0.01      0.15
```
Adjust expected outlier fraction, re-compute live.

**Cluster Picker**
```
□ Cluster 0 (n=45) [mobile_frac: 0.42, k: 0.85]
☑ Cluster 1 (n=38) [mobile_frac: 0.68, k: 1.23]
□ Cluster 2 (n=12) [mobile_frac: 0.35, k: 0.45]
□ Noise (n=8)
```

Check/uncheck to include in analysis.

**Cluster Justification**
- **Silhouette sparkline** → Quality metric for each k
- **BIC curve** → Model selection justification
- Best k highlighted with marker

**Toggles**
- [Hide noise] → Exclude cluster = -1
- [Compare clusters] → Run pairwise tests between clusters

---

## ROI and Tracking QC

### QC Dashboard Metrics

Four interactive histograms (click to filter):

**1. Drift Distribution**
```
|     ███
|    █████
|   ███████
|  █████████
| ███████████
└─────────────
0px    5px   10px (threshold)
```

**2. R² Distribution**
```
|            ███
|          █████
|        ███████
|      █████████
|    ███████████
└─────────────────
0.0   0.5   0.8 (threshold)  1.0
```

**3. Motion Artifacts**
Count or percentage of frames flagged per cell.

**4. Tracking Method Usage**
```
Gaussian: ████████ 145
LK:       ███ 38
Snake:    █ 12
```

### Tracking Method Timeline

*(Planned)*
For each cell, show a colored timeline:
```
Frames: [GGGGGLLLGGGGSSSGGG]
         └─┘└─┘└─┘└─┘└─┘
          1  2  3  4  5
```
- **G** (blue) → Gaussian centroid
- **L** (orange) → Lucas-Kanade optical flow
- **S** (green) → Active contour (snake)
- **Red segments** → QC failed

### Click-to-Filter

Click any histogram bar to filter cells:
- Click "R² < 0.8" bar → Show only poor fits
- Click "Drift > 10px" → Show high-drift cells
- Use for QC gating and investigation

---

## Provenance and Reproducibility

### Analysis Recipe

Collapsible "Analysis recipe" panel shows:

```json
{
  "timestamp": "2025-10-04T10:23:45",
  "software_versions": {
    "python": "3.10.12",
    "numpy": "1.24.3",
    "scipy": "1.10.1",
    "sklearn": "1.3.0",
    "statsmodels": "0.14.0"
  },
  "filters": {
    "condition": ["control", "treated"],
    "outliers": false,
    "qc_pass": true
  },
  "parameters": {
    "tracking": {
      "use_kalman": true,
      "adapt_radius": true
    },
    "fitting": {
      "robust": true,
      "try_2exp": true
    },
    "clustering": {
      "contamination": 0.07,
      "max_k": 6
    }
  },
  "hash": "a3f2c891"
}
```

### Actions

- **Copy Recipe** → Copies JSON to clipboard
- **Download Recipe** → Saves as `analysis_recipe.json`
- **Apply Recipe** → Load JSON and restore exact settings

### Stamping

All exports include the recipe hash:
- CSV footer: `# Recipe: a3f2c891`
- Figure caption: `Analysis #a3f2c891`
- Report header: `Generated from recipe a3f2c891`

---

## Layout and Clarity

### Design Principles

**Consistent Two-Column Grid**
Most views use 50/50 split:
```
[Left Column]          [Right Column]
ROI trajectory         Recovery curve
Parameters 1-2         Parameters 3-4
```

**Fixed Gutters**
- Between columns: 20px
- Between sections: 40px
- Card padding: 16px

**Accent Colors**
- **Blue** (#0066CC) → Primary action, mean lines
- **Orange** (#FF8C00) → Cluster coding, warnings
- **Red** (#CC0000) → Errors, significance
- **Gray** (#808080) → Individual traces, secondary info

**Synchronized Cursors** *(planned)*
Hover on one time-series plot → vertical line appears on all aligned plots.

**Log-Scale Toggle**
For rate constants (k, t½):
```
[Linear] [Log]
```

**Clear Axis Ranges**
- Always show 0 for fractions (0-1)
- Auto-range with 5% padding for rates
- Fixed aspect ratio for X-Y position plots

### Typography

**Hierarchy:**
- Title: 24pt bold
- Section: 18pt bold
- Subsection: 14pt semibold
- Body: 12pt regular
- Caption: 10pt regular

**Accessibility:**
- Minimum 12pt for critical info
- High contrast ratios (4.5:1 for body, 3:1 for large)
- No information conveyed by color alone

---

## Performance Safeguards

### Current Implementation

**Efficient Data Structures**
- Load Parquet with `pyarrow` (10× faster than CSV)
- Use `pd.DataFrame` for vectorized operations
- Cache cohort queries in session state

**Reduced Bootstrap for UI**
```python
analyze(..., n_bootstrap=500)  # Instead of 1000
```

### Planned Optimizations

**Virtualized Lists**
For >1000 cells, render only visible rows:
```python
st.dataframe(df, height=400)  # Auto-virtualizes
```

**Lazy Video Loading**
Load frames on-demand when scrubbing timeline:
```python
@st.cache_data
def load_frame(movie_path, frame_idx):
    return tifffile.imread(movie_path)[frame_idx]
```

**Down-Sample for Interaction**
```python
# Show every 5th point for 1000+ frames
if len(trace) > 200:
    trace_display = trace[::5]
else:
    trace_display = trace
```

Full resolution preserved for export.

**Cached Derived Tables**
```python
@st.cache_data
def compute_cohort_stats(cohort_hash):
    # Expensive computation
    return stats_df
```

**Progress Indicators**
```python
with st.spinner("Running bootstrap (500 iterations)..."):
    results = analyze(...)

st.toast("✓ Analysis complete!", icon="✅")
```

Non-blocking toasts for long operations.

---

## Accessibility and Consistency

### Colorblind-Safe Palette

**Qualitative (Clusters, Categories):**
- #0173B2 (Blue)
- #DE8F05 (Orange)
- #029E73 (Teal)
- #CC78BC (Purple)
- #ECE133 (Yellow)
- #CA9161 (Brown)

**Sequential (Heatmaps):**
- Blues: #F0F9FF → #08519C
- Reds: #FFF5F0 → #67000D

**Diverging (Effect Sizes):**
- RdBu: #B2182B (red) → #F7F7F7 (white) → #2166AC (blue)

Tested with Coblis Color Blindness Simulator.

### Keyboard Navigation

**Supported Actions:**
- `Tab` → Move focus
- `Enter` → Activate button
- `Space` → Toggle checkbox
- `Arrow keys` → Navigate lists
- `Ctrl/Cmd + K` → Command palette *(planned)*

**Focus Indicators:**
```css
:focus {
    outline: 2px solid #0173B2;
    outline-offset: 2px;
}
```

### Tooltips

All icons have descriptive tooltips:
- 🔖 → "Bookmark this cell"
- 📊 → "Add to comparison tray"
- 🗑 → "Delete cohort"
- ⚠ → "QC warning: high drift"

### Dark and Light Themes

Toggle in sidebar:
```
[🌙 Dark Mode]
```

**Light theme (default):**
- Background: #FFFFFF
- Text: #262730
- Cards: #F0F2F6

**Dark theme:**
- Background: #0E1117
- Text: #FAFAFA
- Cards: #262730

Contrast ratios maintained across both.

### Reachable Actions

All critical actions have:
- Button (clickable)
- Keyboard shortcut
- API function (scriptable)

No hover-only controls.

---

## Export and Reporting

### One-Click Exports

**Current Cohort → CSV**
```python
# Includes all cell_features columns for filtered cells
# Footer stamp: # Recipe: a3f2c891, Date: 2025-10-04
```

**All Traces → CSV**
```python
# roi_traces for all cells in cohort
# Columns: exp_id, movie_id, cell_id, frame, t, x, y, radius, signals, QC flags
```

**Current Figure → PNG/SVG** *(planned)*
```python
# High-res export (300 DPI)
# Embedded metadata with recipe hash
```

### Figure Presets

**Save Preset:**
1. Configure cohort filters
2. Select desired plot type
3. Enter preset name
4. Click "💾 Save Current View as Preset"

**Load Preset:**
- Sidebar → "📁 Saved Presets" → Click name
- Restores exact filters and plot configuration

**Use Case:**
Create presets for:
- Weekly progress meetings (same view, updated data)
- Publication figures (identical styling)
- QC checks (standard metrics)

### PDF/HTML Reports *(planned)*

**"📄 Generate PDF Report"** will create:
- **Summary**: Cohort description, cell counts, date
- **Methods**: Tracking algorithm, fitting model, statistical tests
- **Results Tables**: Effect sizes, p-values, q-values
- **Figures**: Spaghetti, heatmap, pairplots, QC dashboard
- **Footer**: Software versions, recipe hash

**"🌐 Generate HTML Report"** similar structure with:
- Interactive Plotly figures
- Collapsible sections
- Copy-to-clipboard for tables
- Print-friendly CSS

### Append to Report

*(Planned)*
Each plot card will have:
```
[📸 Append to report]
```

Builds a figure queue, then "Generate Report" includes all queued items.

---

## Command Palette *(Planned)*

Press `Ctrl/Cmd + K` to open:

```
Search commands, cells, cohorts...
> _

Recent:
  Jump to cell 42
  Load cohort "baseline"
  Export current view

Commands:
  Show single-cell inspector
  Run multi-group analysis
  Generate report
  Apply recipe from file
```

**Fuzzy search** for quick navigation:
- `cell 42` → Jump to cell 42
- `coh base` → Load cohort "baseline"
- `exp csv` → Export cohort CSV

---

## Linked Selection and Cell Tray *(Planned)*

### Brushing and Lasso

On any plot with individual points:
1. Click-and-drag to brush rectangular region
2. Hold `Shift` + click to lasso freeform
3. Selected cells highlight in **yellow**
4. Selection propagates to all views

**Example:**
- Brush high-drift cells in QC histogram
- See them highlighted in spaghetti plot
- Inspect them in cell tray

### Cell Tray

Right sidebar panel shows selected cells:

```
╔═══════════════════════╗
║  CELL TRAY (3)        ║
╟───────────────────────╢
║ Cell 42               ║
║  k: 0.85, frac: 0.42  ║
║  [View] [Remove]      ║
╟───────────────────────╢
║ Cell 87               ║
║  k: 1.23, frac: 0.68  ║
║  [View] [Remove]      ║
╟───────────────────────╢
║ Cell 103              ║
║  k: 0.45, frac: 0.35  ║
║  [View] [Remove]      ║
╟───────────────────────╢
║ [Compare All]         ║
║ [Export Selection]    ║
╚═══════════════════════╝
```

**Actions:**
- **View** → Jump to single-cell inspector
- **Remove** → Remove from tray
- **Compare All** → Side-by-side overlay plot
- **Export Selection** → CSV of selected cells

---

## Integration with Existing Code

### Loading Real Data

If you have existing analysis results:

```python
# Your analysis script
from frap_singlecell_api import analyze_frap_movie
from frap_data_model import DataIO

# Run analysis
roi_traces, cell_features = analyze_frap_movie(
    movie, time_points, 
    exp_id='exp001',
    movie_id='movie01',
    condition='control',
    output_dir='./output'
)

# Data is automatically saved to ./output/roi_traces.parquet and cell_features.parquet
```

Then in the UI:
1. Click "📂 Load Data" in sidebar
2. Browse to `./output/`
3. Select `roi_traces.parquet` and `cell_features.parquet`
4. Data loads into session state

### Multi-Movie Datasets

For multiple experiments:

```python
# Analyze multiple movies
all_traces = []
all_features = []

for movie_file in movie_files:
    traces, features = analyze_frap_movie(...)
    all_traces.append(traces)
    all_features.append(features)

# Combine
combined_traces = pd.concat(all_traces, ignore_index=True)
combined_features = pd.concat(all_features, ignore_index=True)

# Save
io = DataIO()
io.save_tables(combined_traces, combined_features, output_dir='./combined')
```

Then load `./combined/` in the UI.

---

## Troubleshooting

### "No data loaded"

**Solution:**
Ensure you have `roi_traces.parquet` and `cell_features.parquet` files.

Generate test data:
```bash
python quick_start_singlecell.py
```

Then load from `./output/example1/`

### "Need at least 2 conditions"

**Solution:**
Multi-group comparisons require ≥2 unique values in `condition` column.

Ensure your data has:
```python
cell_features['condition'] = ['control', 'treated', 'control', ...]
```

### Plots not showing

**Solution:**
Check that cohort is not empty:
- Summary header should show "Cells: 0" if empty
- Adjust filters in Cohort Builder
- Verify "Include Outliers" and "QC Passed Only" settings

### Performance is slow

**Solution:**
- Reduce bootstrap iterations in code (n_bootstrap=200)
- Down-sample traces for display (every 5th frame)
- Close other browser tabs
- Use Parquet instead of CSV for data I/O

### Recipe hash keeps changing

**Solution:**
Recipe includes timestamp. To freeze a recipe:
1. Copy recipe JSON
2. Save to file: `analysis_recipe_v1.json`
3. Remove "timestamp" field
4. Use this as your reference version

---

## Next Steps

### Priority 1: Data Loading UI

Add file upload widget:
```python
uploaded_files = st.file_uploader(
    "Upload roi_traces and cell_features",
    type=['parquet', 'csv'],
    accept_multiple_files=True
)

if uploaded_files:
    # Load and parse
    ...
```

### Priority 2: Report Generation

Integrate with `frap_singlecell_reports.py`:
```python
from frap_singlecell_reports import build_report

if st.button("Generate PDF"):
    with st.spinner("Building report..."):
        build_report(
            results=stats_results,
            figs=st.session_state.report_figures,
            output_path='./report.pdf'
        )
    st.success("Report saved!")
```

### Priority 3: Linked Selection

Use Plotly's `selectedData` event:
```python
selected = plotly_chart.get("selectedData")
if selected:
    selected_cells = [p['customdata'][0] for p in selected['points']]
    st.session_state.selected_cells = selected_cells
```

### Priority 4: Video Frame Viewer

Add ROI overlay on original movie:
```python
frame = load_frame(movie_path, frame_idx)
overlay = draw_roi_circle(frame, x, y, radius)
st.image(overlay)
```

---

## FAQ

**Q: Can I use this with real microscopy data?**

A: Yes! After running your FRAP experiment:
1. Extract intensity traces with `frap_singlecell_api.track_movie()`
2. Fit curves with `fit_cells()`
3. Detect clusters with `detect_outliers_and_clusters()`
4. Save with `DataIO.save_tables()`
5. Load in UI

**Q: How do I export figures for publication?**

A: *(Planned)* Click "📸 Append to report" on desired plots, then "Generate PDF/HTML Report". Figures will be 300 DPI with embedded metadata.

**Q: Can I run this on a cluster?**

A: The analysis code (`frap_singlecell_api.py`) is command-line friendly:
```bash
python -c "from frap_singlecell_api import analyze_frap_movie; ..."
```

The UI is for interactive exploration after batch processing.

**Q: What if I have >10,000 cells?**

A: The UI will virtualize tables and down-sample plots automatically. For very large datasets, consider:
- Pre-filtering (e.g., QC passed only)
- Analyzing subsets (e.g., by experiment)
- Using the command-line API for statistics

**Q: How do I customize plot styles?**

A: *(Future)* Figure presets will allow saving custom color schemes, font sizes, etc.

---

## Support

For issues or feature requests:
- Check `README_SINGLECELL.md` for API documentation
- Review `SINGLECELL_IMPLEMENTATION.md` for technical details
- Open a GitHub issue with:
  - Streamlit version: `streamlit --version`
  - Python version: `python --version`
  - Error message (if any)
  - Minimal reproducible example

---

## Credits

**Design inspired by:**
- FlowJo (flow cytometry gating)
- Prism (statistical graphics)
- Observable Plot (linked visualizations)
- GitHub Copilot (AI-assisted development)

**Built with:**
- Streamlit (interactive UI)
- Plotly (interactive plots)
- Pandas (data wrangling)
- NumPy/SciPy (numerical computing)
- scikit-learn (machine learning)
- statsmodels (statistics)

---

*Last updated: 2025-10-04*
*UI Version: 1.0*
*Recipe-compatible since: v1.0*
