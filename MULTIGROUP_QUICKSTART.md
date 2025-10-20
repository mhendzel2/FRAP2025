# FRAP Multi-Group Analysis: Quick Start Guide

## When to Use What?

### 🎯 Quick Decision Tree

```
Do you want to compare TREATMENT CONDITIONS?
    ├─ YES → Use Tab 2: Multi-Group Comparison
    │         └─ Focus on Mean Recovery Curves
    │
    └─ NO → Do you want to explore a SINGLE CONDITION in depth?
              └─ YES → Use Tab 1: Individual Group Analysis
                        └─ Analyze cell-level heterogeneity
```

---

## Tab 1: Individual Group Analysis

**Use for:** Deep-dive exploration of a single experimental condition

### What You Get
- Individual cell recovery curves
- Distribution of kinetic parameters
- Outlier detection
- Quality control metrics
- Cell-to-cell variability analysis

### When to Use
✅ Characterizing heterogeneity within a condition  
✅ Quality control and outlier removal  
✅ Understanding parameter distributions  
✅ Detailed exploration before comparison  

❌ Comparing treatment effects (use Tab 2 instead)

---

## Tab 2: Multi-Group Comparison

**Use for:** Comparing recovery dynamics across experimental conditions

### Section 1: Mean Recovery Curves (START HERE!)

**What You Get:**
- Averaged recovery profiles (2+ groups)
- Quantitative similarity metrics
- Advanced biophysical model fitting (optional)

**Workflow:**
1. Select 2+ experimental groups
2. View averaged recovery curves plot
3. Check quantitative metrics (RMSD, max difference)
4. (Optional) Click "Advanced Curve Fitting" expander
   - Select model or use "auto"
   - Click "Fit Advanced Models"
   - Review fitted parameters and interpretation

**When to Use:**
✅ **"Does Drug X slow recovery compared to control?"**  
✅ **"What are the biophysical parameters (D, k_on, k_off)?"**  
✅ **"Which treatment has the biggest effect?"**  

### Section 2: Population-Based Analysis

**What You Get:**
- Kinetic population distributions (% diffusion/binding/intermediate)
- Population shifts between groups
- Weighted kinetics
- Biological interpretation

**When to Use:**
✅ **"Did the treatment shift cells from binding to diffusion regimes?"**  
✅ **"Is there a change in the proportion of fast vs. slow cells?"**  

---

## Tab 3: Multi-Group Statistical Comparison

**Use for:** Statistical testing and parameter distributions across all groups

### What You Get
- Parameter visualizations (estimation plots, box plots, violin plots)
- Statistical tests (t-tests, ANOVA)
- Group-wise parameter distributions

### When to Use
✅ Statistical significance testing  
✅ Visualizing parameter distributions  
✅ Multi-group ANOVA (3+ groups)  
✅ Exploring different parameters systematically  

---

## Common Workflows

### Workflow 1: Basic Treatment Comparison

**Goal:** "Does my treatment affect FRAP recovery?"

```
1. Load data and create groups (Sidebar)
2. Tab 2: Multi-Group Comparison
   - Select control + treatment groups
   - View Mean Recovery Curves
   - Check if curves differ visually
   - Review quantitative metrics (RMSD)
3. If different:
   - Expand "Advanced Curve Fitting"
   - Fit models to extract parameters
   - Read biological interpretation
4. Tab 3: Statistical Comparison
   - Run t-test on mobile fraction
   - Check significance
```

**Expected Time:** 5-10 minutes

---

### Workflow 2: Mechanistic Comparison

**Goal:** "What biophysical mechanism changed?"

```
1. Tab 2: Multi-Group Comparison
   - Select 2 groups to compare
   - Expand "Advanced Curve Fitting"
   - Select model: "auto" (or specific model)
   - Enter bleach radius
   - Click "Fit Advanced Models"
2. Review Results:
   - Compare fitted parameters (D, k_on, k_off, etc.)
   - Check parameter fold changes
   - Read biological interpretation
   - Examine fitted curves plot
3. Tab 2: Population-Based Analysis
   - Check if population distributions changed
   - Review population shifts
   - Confirm mechanistic interpretation
```

**Expected Time:** 10-15 minutes

---

### Workflow 3: Multi-Treatment Screen

**Goal:** "Compare 3+ treatments to find the strongest effect"

```
1. Tab 2: Multi-Group Comparison
   - Select all groups (3+)
   - View all Mean Recovery Curves on one plot
   - Identify visually different groups
2. For each interesting comparison:
   - Select just 2 groups
   - Expand Advanced Curve Fitting
   - Fit models and extract parameters
3. Tab 3: Statistical Comparison
   - Select parameter (e.g., mobile fraction)
   - Run ANOVA
   - Check post-hoc pairwise comparisons
   - View estimation plot
```

**Expected Time:** 15-30 minutes

---

### Workflow 4: Exploring Heterogeneity

**Goal:** "How variable are cells within my treatment?"

```
1. Tab 1: Individual Group Analysis
   - Select specific group
   - View distribution of recovery curves
   - Check parameter distributions (histograms)
   - Identify outliers
   - Examine quality metrics
2. Tab 2: Population-Based Analysis
   - Check kinetic population percentages
   - See if cells cluster into subpopulations
3. Consider:
   - Are there distinct subpopulations?
   - Is heterogeneity biological or technical?
   - Should outliers be excluded?
```

**Expected Time:** 10-20 minutes

---

## Key Concepts Explained

### Mean Recovery Curves vs. Individual Cell Fitting

| Aspect | Mean Recovery Curves | Individual Cell Fitting |
|--------|---------------------|------------------------|
| **Signal-to-Noise** | High (averaging reduces noise) | Low (single cell = noisy) |
| **Models** | Complex (anomalous, reaction-diffusion) | Simple (exponential only) |
| **Use Case** | Compare conditions | Explore heterogeneity |
| **Best For** | Treatment effects | Distribution analysis |
| **Location** | Tab 2 (primary) | Tab 1 (individual group) |

### Population-Based Analysis

**Key Insight:** Not all cells recover the same way!

Cells are categorized into kinetic regimes:
- **Diffusion** (k > 1.0 s⁻¹): Fast recovery, mobile protein
- **Intermediate** (0.1 < k < 1.0 s⁻¹): Moderate recovery
- **Binding** (k < 0.1 s⁻¹): Slow recovery, bound protein

**Example:**
- **Control:** 40% diffusion, 20% intermediate, 40% binding
- **Mutant:** 85% diffusion, 10% intermediate, 5% binding

→ **Interpretation:** Mutant lost chromatin binding capability!

### Advanced Curve Fitting Models

1. **Anomalous Diffusion**
   - Use when: Recovery is non-exponential
   - Extracts: Anomalous exponent α (subdiffusive if α < 1)

2. **Reaction-Diffusion (Simple)**
   - Use when: Two-component recovery (fast + slow)
   - Extracts: Fast rate (diffusion), slow rate (binding)

3. **Reaction-Diffusion (Full)**
   - Use when: Complex multi-state dynamics
   - Extracts: D (diffusion), k_on, k_off (binding kinetics)

4. **Auto**
   - Use when: Unsure which model fits best
   - Selects: Best model by AIC

---

## Tips & Best Practices

### ✅ Do's

- **Start with mean curves** when comparing conditions
- **Use Tab 1** for detailed exploration of individual conditions
- **Apply advanced fitting** to extract mechanistic parameters
- **Check population distributions** for shifts in kinetic regimes
- **Read biological interpretations** for guidance

### ❌ Don'ts

- ❌ Don't fit individual cells with complex models (low SNR!)
- ❌ Don't compare conditions using individual cell parameters only
- ❌ Don't ignore the visual inspection of mean curves
- ❌ Don't over-interpret small parameter differences without statistical testing

### 💡 Pro Tips

1. **Bleach Radius Matters**
   - Advanced fitting requires accurate bleach radius
   - Measure from your imaging setup
   - Typical values: 0.5-2.0 μm

2. **Model Selection**
   - Start with "auto" to see which model fits best
   - If AIC values are close, simpler model is often better
   - Check residuals (future feature) to assess fit quality

3. **Interpreting R²**
   - R² > 0.95: Excellent fit
   - R² > 0.90: Good fit
   - R² < 0.90: Model may not be appropriate

4. **Sample Size**
   - Minimum 10-15 cells per group for reliable mean curves
   - More cells = better averaging, less noise
   - 20-30 cells recommended for robust comparisons

---

## Troubleshooting

### Problem: "My mean curves look identical"

**Possible Causes:**
- Treatment had no effect
- Sample size too small (high SEM)
- Time resolution too coarse

**Solutions:**
- Check Tab 3 statistical tests (may still be significant)
- Review individual cell distributions (Tab 1)
- Consider increasing sample size

### Problem: "Advanced fitting failed"

**Possible Causes:**
- Insufficient data points
- Inappropriate model for your data
- Invalid bleach radius

**Solutions:**
- Try simpler model (e.g., reaction-diffusion simple)
- Check that bleach radius is reasonable (0.1-10 μm)
- Ensure curves have at least 10-15 time points

### Problem: "Population analysis shows no shifts"

**Possible Causes:**
- Treatment affects kinetics within populations, not population composition
- Thresholds may not match your system

**Solutions:**
- Focus on mean curve comparison and advanced fitting
- Check parameter distributions in Tab 3
- Consider that subtle effects may not shift populations

---

## Summary: Decision Flowchart

```
┌─────────────────────────────────────────┐
│  What is your analysis goal?            │
└─────────────┬───────────────────────────┘
              │
              ├─ Compare Treatment Conditions?
              │  └─> Tab 2: Mean Recovery Curves
              │      ├─ Visual comparison
              │      ├─ Advanced fitting (mechanistic)
              │      └─ Population shifts
              │
              ├─ Explore Single Condition Heterogeneity?
              │  └─> Tab 1: Individual Group Analysis
              │      ├─ Cell-level distributions
              │      ├─ Outlier detection
              │      └─ Quality control
              │
              ├─ Statistical Testing Across 3+ Groups?
              │  └─> Tab 3: Multi-Group Statistical
              │      ├─ ANOVA
              │      ├─ Post-hoc tests
              │      └─ Parameter visualizations
              │
              └─ Extract Biophysical Parameters?
                 └─> Tab 2: Advanced Curve Fitting
                     ├─ Select model (or "auto")
                     ├─ Fit mean curves
                     └─ Review parameters & interpretation
```

---

**Need more help?** See `UI_REORGANIZATION_SUMMARY.md` for technical details.

**Version:** 1.0  
**Last Updated:** 2025-01-XX
