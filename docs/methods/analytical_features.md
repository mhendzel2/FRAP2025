# New Analytical Features for FRAP Analysis

This document describes the new analytical capabilities added to the FRAP analysis software, implemented from pull requests #13, #14, #15, and #18.

## Table of Contents
1. [Advanced Model Selection](#advanced-model-selection)
2. [Confidence Intervals](#confidence-intervals)
3. [Advanced Statistical Tests](#advanced-statistical-tests)
4. [Quality Control](#quality-control)
5. [Usage Examples](#usage-examples)

---

## Advanced Model Selection

### AICc (Corrected Akaike Information Criterion)

**What it is**: AICc is a modification of AIC that provides better model selection for small sample sizes (when n/k < 40, where n = sample size, k = parameters).

**Formula**: 
```
AICc = AIC + (2k²+ 2k) / (n - k - 1)
```

**When to use**: Always recommended over AIC for FRAP data, as typical datasets have 20-50 points.

**Implementation**: 
- Now the **default criterion** in `select_best_fit()`
- Automatically calculated for all models in `fit_all_models()`

### Anomalous Diffusion Model

**What it is**: A stretched exponential model that can capture subdiffusive behavior common in crowded cellular environments.

**Formula**: 
```
I(t) = A * (1 - exp(-(t/τ)^β)) + C
```

Where:
- **A**: Amplitude
- **τ** (tau): Characteristic time  
- **β** (beta): Anomalous diffusion exponent
  - β < 1: Subdiffusive (hindered diffusion)
  - β = 1: Normal diffusion (reduces to single exponential)
- **C**: Offset

**Interpretation**:
- When β < 1, the model is labeled as `'anomalous_diffusion_subdiffusive'`
- Subdiffusion indicates obstacles, binding interactions, or molecular crowding

### Delta AIC/BIC Values

**What they are**: Differences between each model's AIC/BIC and the best model's value.

**Interpretation**:
- Δ = 0: Best model
- Δ < 2: Substantial support (almost as good)
- Δ = 4-7: Considerably less support
- Δ > 10: Essentially no support

**Usage**: Use delta values to assess model uncertainty and identify competing models.

---

## Confidence Intervals

### Two Methods Available

#### 1. Parametric CIs (Fast, Built-in)

**Method**: Uses the covariance matrix from `curve_fit()` and t-distribution.

**Pros**:
- Fast (instant)
- Directly from fitting algorithm
- Reliable for well-behaved data

**Cons**:
- Assumes normally distributed errors
- May underestimate uncertainty for complex models

**Access**: Automatically included in `fit_all_models()` output as `fit['ci']`

#### 2. Bootstrap CIs (Robust, Computationally Intensive)

**Method**: Parametric bootstrap with residual resampling (default: 1000 iterations).

**Pros**:
- More robust to non-normal errors
- Better for complex models
- Confidence intervals for derived quantities (D, MW)

**Cons**:
- Computationally expensive (~10-30 seconds)
- CPU-limited systems automatically reduce to 200 iterations

**Usage**:
```python
from frap_bootstrap import run_bootstrap

ci_results = run_bootstrap(
    best_fit=best_fit,
    t_fit=t_fit,
    intensity_fit=intensity_fit,
    bleach_radius_um=1.0,
    n_bootstrap=1000
)

# Returns:
# {
#     'koff_median': 0.048,
#     'koff_ci_low': 0.042,
#     'koff_ci_high': 0.055,
#     'D_median': 0.03,
#     'D_ci_low': 0.026,
#     'D_ci_high': 0.035,
#     ...
# }
```

### Parameters with CIs

1. **koff**: Dissociation rate constant (1/s)
2. **D**: Diffusion coefficient (μm²/s)
3. **mobile_fraction**: Percentage of mobile molecules (%)
4. **app_mw**: Apparent molecular weight (kDa)

---

## Advanced Statistical Tests

Implemented in `frap_group_stats.py` for comparing experimental groups.

### 1. Welch's t-test

**What it is**: Student's t-test for unequal variances (more robust than standard t-test).

**When to use**: Comparing two groups with potentially different variances.

### 2. One-way ANOVA

**What it is**: Analysis of variance for comparing 3+ groups.

**When to use**: Comparing multiple experimental conditions.

**Note**: Followed by pairwise tests if significant.

### 3. Permutation Tests

**What it is**: Non-parametric test that doesn't assume normal distribution.

**Method**: 
- Randomly shuffle group labels 10,000 times
- Calculate how often shuffled data is as extreme as observed

**Advantage**: Valid even with non-normal data or small samples.

### 4. Effect Sizes

#### Cohen's d
**Interpretation**:
- |d| < 0.2: Small effect
- |d| = 0.5: Medium effect  
- |d| > 0.8: Large effect

#### Cliff's Delta
**Range**: -1 to +1

**Interpretation**:
- |δ| < 0.147: Negligible
- |δ| < 0.33: Small
- |δ| < 0.474: Medium
- |δ| ≥ 0.474: Large

### 5. FDR Correction (Benjamini-Hochberg)

**What it is**: Controls the false discovery rate when performing multiple tests.

**Output**: q-values (corrected p-values)

**Threshold**: q < 0.05 typically used for significance after correction.

### 6. TOST (Two One-Sided Tests)

**What it is**: Test for equivalence (proving groups are similar).

**Usage**: Set equivalence bounds, test if groups differ by less than that.

**Output**: "Equivalent" or "Not Equivalent"

### 7. Mixed-Effects Models

**What they are**: Account for hierarchical/nested data structure.

**Example**: Multiple FOVs per experiment, multiple cells per FOV.

**Usage**:
```python
from frap_group_stats import calculate_group_stats

stats_df = calculate_group_stats(
    data=combined_df,
    metrics=['mobile_fraction', 'koff', 'D'],
    group_order=['Control', 'Treatment'],
    tost_thresholds={'D': (-0.2, 0.2)},
    use_mixed_effects=True,
    random_effect_col='experiment_id'
)
```

**Returns**: DataFrame with all test results and q-values.

---

## Quality Control

### Automated QC Checks

#### Pre-fit QC
- Minimum 10 data points required
- Check for NaN values in raw data

#### Fit QC  
- R² threshold (default: 0.8)
- Check for unbounded confidence intervals
- Validate parameter ranges

#### Feature QC
- Mobile fraction: 0% ≤ MF ≤ 110%
- Diffusion coefficient: Must be positive
- Rate constants: Must be positive

### FOV ID Parsing

**What it does**: Automatically extracts field-of-view numbers from filenames.

**Patterns recognized**:
- `fov1`, `fov_1`, `FOV-1`
- `fld2`, `field3`
- `view4`

**Usage**:
```python
fov_id = FRAPAnalysisCore.parse_fov_id('experiment_fov12_data.csv')
# Returns: 12
```

**Purpose**: Track spatial location for clustering analysis and mixed-effects models.

---

## Usage Examples

### Example 1: Full Analysis with New Features

```python
from frap_core import FRAPAnalysisCore
import pandas as pd

# Load data
df = pd.read_csv('frap_data.csv')
time = df['time'].values
intensity = df['normalized_intensity'].values

# Fit all models (includes anomalous diffusion)
fits = FRAPAnalysisCore.fit_all_models(time, intensity)

# Select best model using AICc (now default)
best_fit = FRAPAnalysisCore.select_best_fit(fits, criterion='aicc')

# Examine model selection
print(f"Best model: {best_fit['model']}")
print(f"AICc: {best_fit['aicc']:.2f}")
print(f"Delta AICc: {best_fit['delta_aicc']:.2f}")
print(f"Delta BIC: {best_fit['delta_bic']:.2f}")

# Check for competing models (delta < 2)
for fit in fits:
    if fit['delta_aicc'] < 2:
        print(f"Competing model: {fit['model']} (ΔAIC={fit['delta_aicc']:.2f})")

# Get confidence intervals (parametric)
if 'ci' in best_fit:
    print("\nParametric 95% CIs:")
    for param, (low, high) in best_fit['ci'].items():
        print(f"{param}: [{low:.4f}, {high:.4f}]")
```

### Example 2: Bootstrap Confidence Intervals

```python
from frap_core import FRAPAnalysisCore
from frap_bootstrap import run_bootstrap

# After fitting...
t_fit, i_fit, _ = FRAPAnalysisCore.get_post_bleach_data(time, intensity)

ci_results = run_bootstrap(
    best_fit=best_fit,
    t_fit=t_fit,
    intensity_fit=i_fit,
    bleach_radius_um=1.0,  # Your bleach spot radius in μm
    n_bootstrap=1000,
    gfp_mw=27.0,  # GFP reference
    gfp_d=25.0    # GFP diffusion coefficient
)

print(f"koff: {ci_results['koff_median']:.4f} "
      f"[{ci_results['koff_ci_low']:.4f}, {ci_results['koff_ci_high']:.4f}]")
print(f"D: {ci_results['D_median']:.3f} "
      f"[{ci_results['D_ci_low']:.3f}, {ci_results['D_ci_high']:.3f}] μm²/s")
```

### Example 3: Group Comparison with Advanced Stats

```python
from frap_group_stats import calculate_group_stats
import pandas as pd

# Prepare data with group labels
data = pd.DataFrame({
    'group': ['Control']*10 + ['Treatment']*10,
    'experiment_id': [1, 1, 1, 2, 2, 2, 3, 3, 3, 3] * 2,  # For mixed effects
    'mobile_fraction': [...],  # Your data
    'koff': [...],
    'D': [...]
})

# Run comprehensive statistical analysis
stats_df = calculate_group_stats(
    data=data,
    metrics=['mobile_fraction', 'koff', 'D'],
    group_order=['Control', 'Treatment'],
    tost_thresholds={
        'D': (-0.2, 0.2),  # Equivalence bounds for diffusion
        'mobile_fraction': (-5, 5)  # Within 5% is equivalent
    },
    use_mixed_effects=True,
    random_effect_col='experiment_id'
)

# Display results
print(stats_df[['metric', 'test', 'p_value', 'q_value', 'cohen_d']])

# Check for significant differences after FDR correction
significant = stats_df[stats_df['q_value'] < 0.05]
print(f"\n{len(significant)} significant differences after FDR correction")
```

### Example 4: Quality Control

```python
from frap_core import FRAPAnalysisCore
import pandas as pd

raw_df = pd.read_csv('raw_data.csv')
processed_df = pd.read_csv('processed_data.csv')

qc_results = FRAPAnalysisCore.perform_quality_control(
    raw_df=raw_df,
    processed_df=processed_df,
    best_fit=best_fit,
    features=extracted_features
)

if not qc_results['qc_fit_pass']:
    print(f"WARNING: Fit QC failed - {qc_results['qc_fit_reason']}")
    
if not qc_results['qc_feat_pass']:
    print(f"WARNING: Feature QC failed - {qc_results['qc_feat_reason']}")
```

---

## References

### Statistical Methods
- Burnham, K. P., & Anderson, D. R. (2004). *Multimodel inference: understanding AIC and BIC in model selection*. Sociological methods & research, 33(2), 261-304.
- Welch, B. L. (1947). *The generalization of "Student's" problem when several different population variances are involved*. Biometrika, 34(1/2), 28-35.
- Benjamini, Y., & Hochberg, Y. (1995). *Controlling the false discovery rate: a practical and powerful approach to multiple testing*. Journal of the Royal Statistical Society: Series B, 57(1), 289-300.

### FRAP Analysis
- Sprague, B. L., & McNally, J. G. (2005). *FRAP analysis of binding: proper and fitting*. Trends in cell biology, 15(2), 84-91.
- Kang, M., Day, C. A., Kenworthy, A. K., & DiBenedetto, E. (2012). *Simplified equation to extract diffusion coefficients from confocal FRAP data*. Traffic, 13(12), 1589-1600.

### Bootstrap Methods
- Efron, B., & Tibshirani, R. J. (1994). *An introduction to the bootstrap*. CRC press.

---

## Implementation Notes

1. **Backwards Compatible**: All new features are optional. Existing code will continue to work.

2. **Performance**: 
   - Parametric CIs: Instant
   - Bootstrap CIs: 10-30 seconds (1000 iterations)
   - Group stats: 1-5 seconds per comparison

3. **Dependencies Added**:
   - `pingouin>=0.5.3` (for effect sizes and TOST)
   - `statsmodels>=0.14.0` (already required, used for mixed-effects)
   - `joblib>=1.3.0` (already required, used for bootstrap parallelization)

4. **Best Practices**:
   - Use AICc for model selection (now default)
   - Report delta AIC/BIC for transparency
   - Use bootstrap CIs for final publication values
   - Apply FDR correction for multiple comparisons
   - Check QC flags before accepting results

---

## Troubleshooting

### Bootstrap is slow
- Reduce `n_bootstrap` to 200-500
- System automatically reduces to 200 on ≤2 CPU cores

### Models not converging
- Check that time starts at 0 (post-bleach)
- Ensure intensity is normalized (0-1 range)
- Check for sufficient recovery (at least 10 post-bleach points)

### Effect sizes return NaN
- Need at least 2 samples per group
- Check for zero variance in groups
- Ensure data doesn't contain NaN/Inf values

### Mixed-effects model fails
- Verify `experiment_id` or random effect column exists
- Need at least 2 levels in random effect
- Check for sufficient replication within levels
