# FRAP Plot Visualization Fixes - Summary

## Issues Identified and Fixed

### Problem 1: Red Fit Line Starting at Wrong Position
**Issue**: The red fitted curve was starting at time=0 instead of at the interpolated bleach point, causing misalignment with the blue data points.

**Root Cause**: The `get_post_bleach_data()` function returns timepoints starting from 0 (for fitting purposes), but the plotting code wasn't converting these back to the original time scale.

**Fix**: 
- Calculate the interpolated bleach time: `(time[bleach_idx-1] + time[bleach_idx]) / 2.0`
- Convert fitted timepoints back to original scale: `t_fit_original_scale = t_fit + interpolated_bleach_time`
- Plot both blue points and red curve using the converted timepoints

### Problem 2: Pre-bleach Data Not Shown for Context
**Issue**: The original plot wasn't clearly distinguishing between pre-bleach and post-bleach data.

**Fix**: 
- Add pre-bleach data as gray markers (not fitted)
- Add post-bleach data as blue markers (fitted)
- Clearly label each series in the legend

### Problem 3: Y-axis Not Starting from Zero
**Issue**: Y-axis scaling was automatic, not providing proper context for recovery measurements.

**Fix**: 
- Set `yaxis=dict(range=[0, None])` in plot layout
- Ensure y-axis always starts from zero for proper scaling context

### Problem 4: Bleach Event Line Position
**Issue**: The orange bleach event line was positioned at the raw bleach frame time, not the interpolated bleach point.

**Fix**: 
- Position the bleach line at the calculated interpolated bleach time
- Update annotation to clearly mark "Bleach Event"

## Code Changes Made

### 1. Updated Single File Analysis Plot (`streamlit_frap_final.py`, lines ~900-980)
```python
# Calculate interpolated bleach time
if bleach_idx > 0:
    interpolated_bleach_time = (file_data['time'][bleach_idx-1] + file_data['time'][bleach_idx]) / 2.0
else:
    interpolated_bleach_time = file_data['time'][bleach_idx]

# Convert fitted timepoints back to original time scale for plotting
t_fit_original_scale = t_fit + interpolated_bleach_time

# Plot pre-bleach data (gray), post-bleach data (blue), and fit (red)
# All using correct timepoints
```

### 2. Updated Group Plotting Functions (`streamlit_frap_final.py`, lines ~430-470)
```python
# Added y-axis range starting from zero
yaxis=dict(range=[0, None])
```

### 3. Enhanced Plot Explanation
- Updated explanatory text to accurately describe the visualization
- Clarified the role of each data series and visual element

## Verification

### Test Results
- ✅ Timepoint conversion logic verified with synthetic data
- ✅ Red fit line now starts at interpolated bleach point (not t=0)
- ✅ Blue data points and red line are properly aligned
- ✅ Pre-bleach data shown for context (gray markers)
- ✅ Y-axis starts from zero for proper scaling
- ✅ Bleach event line positioned at correct interpolated time

### Expected Visual Improvements
1. **Aligned Curves**: Red and blue lines/points now converge properly at the interpolated bleach point
2. **Context Visualization**: Pre-bleach data visible as gray points for experimental context
3. **Proper Scaling**: Y-axis starts at zero, providing better perspective on recovery magnitude
4. **Clear Event Marking**: Orange dashed line accurately marks the interpolated bleach event
5. **Professional Appearance**: Clear legend and explanatory text for interpretation

## Files Modified
- `streamlit_frap_final.py`: Main plotting logic and group visualization functions
- `test_simple_plot_fix.py`: Verification test (created)

## Compatibility
- All changes are backward compatible
- No changes to core fitting algorithms
- Existing analysis results remain unchanged
- Only visualization improvements implemented

---

*These fixes ensure that FRAP recovery plots accurately represent the underlying analysis methodology, with proper alignment between fitted curves and data points, clear visual distinction between pre- and post-bleach phases, and appropriate scaling for quantitative interpretation.*
