# FRAP Curve Alignment - Quick Reference

## What It Does

Aligns multiple FRAP recovery curves to a common time axis (t=0 at bleach) and interpolates them to enable accurate comparison even when experiments used different sampling rates.

---

## How To Use

### In Streamlit UI

1. Open your FRAP analysis application
2. Navigate to **Tab 2: Multi-Group Comparison**
3. Select a group with multiple files
4. Scroll to **Step 8: Visualize Aligned Group Curves**
5. Click **"Generate Aligned Curves Plot"** button
6. View the aligned curves plot

### In Python Code

```python
from frap_core import FRAPAnalysisCore
from frap_plots import FRAPPlots

# Prepare your curves
curves = [
    {'name': 'Cell1', 'time': time1, 'intensity': intensity1},
    {'name': 'Cell2', 'time': time2, 'intensity': intensity2},
    # ... more curves
]

# Align and interpolate
aligned_results = FRAPAnalysisCore.align_and_interpolate_curves(
    curves, 
    num_points=200  # Resolution of common time axis
)

# Plot the results
fig = FRAPPlots.plot_aligned_curves(aligned_results, height=500)

# In Streamlit:
st.plotly_chart(fig, use_container_width=True)
```

---

## Why It Matters

### Problem
❌ Different experiments → Different frame rates  
❌ Different bleach times → Curves not aligned  
❌ Direct comparison → Misleading results  

### Solution
✅ Time-shift all curves to start at t=0 (bleach event)  
✅ Interpolate all curves to common time grid  
✅ Accurate visual comparison of recovery kinetics  

---

## Key Features

- **Automatic alignment** to bleach point (t=0)
- **Handles different sampling rates** (0.3s, 0.5s, 0.8s, etc.)
- **Robust error handling** (skips problematic curves)
- **Interactive plots** with hover information
- **Publication-ready** visualization

---

## Functions

### `FRAPAnalysisCore.align_and_interpolate_curves()`

**Purpose:** Aligns curves to common time axis

**Input:**
- `list_of_curves`: List of dicts with 'name', 'time', 'intensity'
- `num_points`: Resolution (default: 200)

**Output:**
```python
{
    'common_time': array([0, 0.125, 0.25, ..., 45.0]),
    'interpolated_curves': [
        {'name': 'Cell1', 'intensity': array([...])},
        {'name': 'Cell2', 'intensity': array([...])},
        ...
    ]
}
```

### `FRAPPlots.plot_aligned_curves()`

**Purpose:** Visualizes aligned curves

**Input:**
- `aligned_results`: Output from alignment function
- `height`: Plot height in pixels (default: 500)

**Output:**
- Interactive Plotly figure

---

## Testing

Run the test suite:
```bash
python test_curve_alignment.py
```

**Expected output:**
```
🎉 ALL TESTS PASSED! 🎉

Total: 4/4 tests passed
```

---

## Common Issues

### No curves displayed
- **Cause:** All curves have bleach at t=0 or insufficient data
- **Fix:** Check that bleaching occurred mid-experiment

### Jagged curves
- **Cause:** Too few interpolation points or noisy data
- **Fix:** Increase `num_points` parameter or smooth raw data

### Memory error
- **Cause:** Too many curves or points
- **Fix:** Reduce `num_points` or process in batches

---

## Files

- **Core:** `frap_core.py` (line ~240)
- **Plots:** `frap_plots.py` (line ~1395)
- **UI:** `streamlit_frap_final_clean.py` (line ~2485)
- **Tests:** `test_curve_alignment.py`
- **Docs:** `CURVE_ALIGNMENT_IMPLEMENTATION.md`

---

## Example Use Cases

### 1. Quality Control
View all curves in a group to identify outliers or artifacts

### 2. Visual Comparison
Compare recovery kinetics between different experimental conditions

### 3. Publication Figures
Generate clean, aligned plots for manuscripts

### 4. Time Series Analysis
Ensure proper temporal alignment before statistical analysis

---

## Quick Tips

💡 **Use consistent normalization** before alignment  
💡 **Check raw data quality** if alignment fails  
💡 **Increase num_points** for smoother curves  
💡 **Hover over curves** to see detailed information  
💡 **Export plots** using Plotly's built-in tools  

---

## Status

✅ **Production Ready**  
✅ **Fully Tested**  
✅ **Documented**  
✅ **Integrated into UI**

---

**Version:** 1.0  
**Last Updated:** October 19, 2025
