# Interactive Features Quick Reference

## 🎯 Interactive Parameter Dashboard
**Location:** Tab 2 → Group Analysis → Step 5

### Quick Start
1. Select X-axis parameter (e.g., mobile_fraction)
2. Select Y-axis parameter (e.g., half_time_fast)
3. **Click any point** in scatter plot
4. View recovery curve(s) below

### Common Analyses

| X-Axis | Y-Axis | Purpose |
|--------|--------|---------|
| mobile_fraction | r2 | **Quality Control** - Find bad fits |
| rate_constant_fast | rate_constant_slow | **Kinetic Regimes** - Identify populations |
| mobile_fraction | half_time_fast | **Mechanism** - Speed vs. completeness |
| half_time_fast | immobile_fraction | **Binding** - Slow recovery = more binding? |

### Tips
💡 **Click clusters** to see if they share kinetic behavior  
💡 **Hover first** to see which file before clicking  
💡 **Maximum 10 curves** displayed at once  
💡 **Color indicates** Y-axis value (viridis scale)  

---

## 📊 Comprehensive Multi-Panel Plot
**Location:** Tab 1 → Single File Analysis → Comprehensive Analysis View

### Quick Start
1. Select any file from dropdown
2. Scroll to "Comprehensive Analysis View"
3. Click **"Show Comprehensive Analysis Plot"** button
4. View integrated plot with 3 panels

### Panel Guide

```
┌─────────────────────────────┐
│ Top: Recovery Curve         │ ← Data + Fit + Parameters
├─────────────────────────────┤
│ Middle: Residuals           │ ← Fit Quality (should be random)
├─────────────────────────────┤
│ Bottom: Components          │ ← Individual exponentials
└─────────────────────────────┘
```

### Reading Residuals

| Pattern | Meaning | Action |
|---------|---------|--------|
| Random scatter | ✅ Good fit | Accept model |
| Wave pattern | ❌ Wrong model | Try different model |
| Trend (up/down) | ❌ Systematic error | Check data quality |
| Outliers at edges | ⚠️ Edge effects | Inspect time range |

### Component Interpretation

**Fast Component (green):**
- k > 1.0 s⁻¹ → Diffusion-dominated
- Large amplitude → Major population

**Slow Component (purple):**
- k < 0.1 s⁻¹ → Binding-dominated  
- Large amplitude → Major population

**Baseline (gray):**
- Level of immobile fraction

---

## 🚀 Workflow Examples

### Example 1: Find Outliers
```
1. Tab 2 → Step 5: Interactive Dashboard
2. X = mobile_fraction, Y = r2
3. Click on low R² points (<0.9)
4. Examine recovery curves
5. Identify noisy data or poor bleaching
```

### Example 2: Identify Subpopulations
```
1. Tab 2 → Step 5: Interactive Dashboard
2. X = rate_constant_fast, Y = rate_constant_slow
3. Notice two clusters in scatter plot
4. Click cluster 1: Fast diffusion, minimal binding
5. Click cluster 2: Moderate diffusion, strong binding
→ Heterogeneous population confirmed!
```

### Example 3: Validate Fit Quality
```
1. Tab 1 → Select file with unusual mobile fraction
2. View standard recovery curve
3. Click "Show Comprehensive Analysis Plot"
4. Check residuals panel:
   - Random scatter? → Fit is good
   - Pattern visible? → Try different model
5. Check components panel:
   - Do components make biological sense?
```

### Example 4: Understand Mechanism
```
1. Tab 1 → Select file
2. View comprehensive plot
3. Components panel shows:
   - Fast (20%): k = 1.5 s⁻¹ → Diffusion
   - Slow (80%): k = 0.05 s⁻¹ → Binding
→ Protein is primarily bound to chromatin!
```

---

## 🎨 Visual Cues

### Interactive Dashboard
- **Blue dots**: Data points (clickable)
- **Color gradient**: Viridis scale maps to Y-axis value
- **Hover tooltip**: Shows file name + parameters
- **Selected curves**: Each has unique color below scatter

### Comprehensive Plot
- **Blue markers**: Experimental data
- **Red line**: Fitted model
- **Green/Orange/Purple dashed**: Individual components
- **Gray dotted**: Baseline (immobile fraction)
- **White box**: Parameter annotations

---

## ⚡ Keyboard Shortcuts & Navigation

### Interactive Dashboard
- **Single click**: Select point
- **Shift+click**: Multi-select (up to 10)
- **Click scatter background**: Deselect all
- **Scroll**: Zoom in/out on scatter plot
- **Drag**: Pan scatter plot

### Comprehensive Plot
- **Hover**: See exact values
- **Click legend**: Toggle trace visibility
- **Drag**: Pan along time axis
- **Scroll**: Zoom in/out
- **Double-click**: Reset view

---

## 🔧 Troubleshooting

### Dashboard Issues

**"Nothing happens when I click"**
- Make sure you're clicking directly on a data point (dot)
- Check that group has analyzed files

**"No recovery curves shown"**
- Click on actual scatter points (not empty space)
- Verify files have time/intensity data

**"Too many curves, can't see anything"**
- Limit: Only first 10 selected points shown
- Click scatter background to clear, reselect fewer points

### Comprehensive Plot Issues

**"Button does nothing"**
- File must be successfully fitted first
- Check that file has valid fit_result

**"No components panel"**
- Only shown for double/triple exponential fits
- Single exponential has no separate components

**"Residuals look terrible"**
- Check data quality (noise, artifacts)
- May need different model type
- Consider outlier removal

---

## 💡 Pro Tips

### Dashboard
✨ **Start broad, then zoom**: mobile_fraction vs. r2 first, then explore kinetics  
✨ **Look for clusters**: Groups of similar cells often share biology  
✨ **Color patterns**: Gradient in colorscale reveals trends  
✨ **Compare groups**: Switch groups to see if patterns change  

### Comprehensive Plot
✨ **Residuals first**: Always check fit quality before trusting parameters  
✨ **Component separation**: Well-separated components = robust fit  
✨ **Annotation box**: Contains all key metrics at a glance  
✨ **Bottom panel**: Shows which component dominates  

---

## 📖 Common Questions

**Q: How do I know if my fit is good?**  
A: Check comprehensive plot residuals panel. Random scatter around zero = good fit. Patterns = problems.

**Q: Which parameters should I compare in dashboard?**  
A: Start with mobile_fraction vs. r2 (quality control), then explore rate constants (kinetics).

**Q: Can I export the comprehensive plot?**  
A: Yes! Hover over plot, click camera icon (top right) to save as PNG.

**Q: What if I see two clusters in scatter plot?**  
A: Click both clusters separately. If recovery curves differ = subpopulations! Consider analyzing separately.

**Q: How many points can I select in dashboard?**  
A: Technically unlimited, but only first 10 recovery curves displayed to avoid clutter.

---

## 🎓 Learning Path

### Beginner
1. Use interactive dashboard to explore mobile fraction vs. R²
2. Click outlier points to understand why they differ
3. View comprehensive plot to see all analysis components

### Intermediate  
1. Compare kinetic rates in dashboard (fast vs. slow)
2. Use comprehensive plot to validate multi-component fits
3. Identify subpopulations from scatter plot clusters

### Advanced
1. Design custom parameter comparisons for specific hypotheses
2. Use residuals patterns to diagnose data quality issues
3. Combine dashboard exploration with holistic group comparison

---

## 📚 Related Features

**Holistic Group Comparison** (Tab 2, bottom)
- Compare entire groups accounting for population distributions
- Complements single-cell dashboard exploration

**Multi-Group Comparison** (Tab 3)
- Statistical tests across multiple groups
- Use dashboard to explore individual cells behind group differences

**Image Analysis** (Tab 4)
- Measure bleach spot parameters
- Verify assumptions used in kinetic interpretations

---

**Quick Reference Version 1.0**  
**Last Updated:** October 19, 2025  
**Compatible With:** FRAP2025 Platform v3.0+
