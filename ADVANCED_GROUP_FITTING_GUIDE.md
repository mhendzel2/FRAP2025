# Advanced Group-Level Curve Fitting

## Overview

This feature enables sophisticated biophysical modeling of mean recovery profiles for comparing groups. Instead of relying only on simple exponential fits to individual cells, you can now:

1. **Average recovery profiles** across all cells in a group
2. **Fit advanced models** (anomalous diffusion, reaction-diffusion) to the mean curves
3. **Compare fitted parameters** between conditions to gain mechanistic insights

## Why Use Advanced Group-Level Fitting?

### Traditional Approach Limitations
- Fits simple exponentials to individual cells
- Averages fitted parameters across cells
- May miss complex biophysical phenomena
- Assumes all cells follow same simple model

### Advanced Group-Level Approach Benefits
- Fits sophisticated models to averaged data (higher SNR)
- Can detect anomalous diffusion, binding kinetics
- Provides mechanistic interpretation
- Compares groups at the biophysical level

## Available Models

### 1. Anomalous Diffusion (Stretched Exponential)

**Model:**
```
I(t) = A * (1 - exp(-(t/τ)^β)) + C
```

**Parameters:**
- **A**: Amplitude (mobile fraction - immobile fraction)
- **C**: Baseline (immobile fraction)
- **τ (tau)**: Characteristic diffusion time
- **β (beta)**: Anomalous exponent
  - β = 1: Normal (Brownian) diffusion
  - β < 1: Subdiffusion (hindered by obstacles/crowding)
  - β > 1: Superdiffusion (rare, directed transport)

**Use When:**
- Investigating diffusion in crowded/heterogeneous environments
- Comparing mobility regimes between conditions
- Studying effects of molecular crowding or chromatin structure

**Biological Insights:**
- Changes in β indicate altered diffusion regime
- Subdiffusion (β < 1) suggests molecular crowding or obstacles
- τ reflects overall recovery timescale

### 2. Reaction-Diffusion (Simple)

**Model:**
Combines free diffusion and reversible binding components.

**Parameters:**
- **F_imm**: Truly immobile fraction
- **F_b**: Bound fraction (recovers via binding kinetics)
- **F_f**: Free fraction (recovers via diffusion)
- **k_eff**: Effective binding rate (combines k_on and k_off)
- **D_app**: Apparent diffusion coefficient
- **w**: Bleach spot radius (fixed)

**Use When:**
- Protein binds reversibly to chromatin or other structures
- Want to separate diffusion vs. binding contributions
- Studying effects of mutations on binding

**Biological Insights:**
- Changes in F_b indicate altered binding capacity
- Changes in k_eff show faster/slower exchange kinetics
- D_app reflects mobility of free population

### 3. Reaction-Diffusion (Full)

**Model:**
Explicit binding and unbinding rates.

**Parameters:**
- **F_imm**: Truly immobile fraction
- **F_mobile**: Mobile fraction (free + transiently bound)
- **k_on**: Binding rate to chromatin (s⁻¹)
- **k_off**: Unbinding rate from chromatin (s⁻¹)
- **D**: Free diffusion coefficient (μm²/s)
- **w**: Bleach spot radius (fixed)

**Use When:**
- Need explicit binding/unbinding kinetics
- Investigating effects on association vs. dissociation
- Calculating equilibrium binding constants (K_d = k_off / k_on)

**Biological Insights:**
- k_on changes: altered binding rate
- k_off changes: altered residence time on chromatin
- K_d: overall binding affinity

## Usage

### Basic Usage: Single Group

```python
from frap_advanced_fitting import fit_mean_recovery_profile

# Fit anomalous diffusion to mean profile
result = fit_mean_recovery_profile(
    time=time_array,
    intensity_mean=mean_intensity,
    intensity_sem=sem_intensity,  # Optional, for weighted fitting
    bleach_radius_um=1.0,
    model='anomalous'
)

if result['success']:
    print(f"R² = {result['r2']:.4f}")
    print("Parameters:", result['params'])
    print("Interpretation:", result['interpretation'])
```

### Group Comparison

```python
from frap_group_comparison import compare_recovery_profiles

# Compare two groups with advanced fitting
comparison = compare_recovery_profiles(
    group1_data=wt_data,
    group2_data=mut_data,
    group1_name="Wild Type",
    group2_name="Mutant",
    use_advanced_fitting=True,
    bleach_radius_um=1.0,
    advanced_model='all'  # Try all models, return best
)

# Access results
if comparison['advanced_fitting']['success']:
    adv = comparison['advanced_fitting']
    
    # Model info
    print(f"Best model: {adv['model_used']}")
    print(f"R² (WT): {adv['r2_group1']:.4f}")
    print(f"R² (Mutant): {adv['r2_group2']:.4f}")
    
    # Parameter comparison
    for param, data in adv['parameter_comparison'].items():
        print(f"{param}: {data['fold_change']:.2f}x change")
    
    # Biological interpretation
    print(adv['interpretation'])
```

### Model Selection

```python
# Fit specific model
result = fit_mean_recovery_profile(
    time, intensity, sem,
    model='anomalous'  # or 'reaction_diffusion_simple', 'reaction_diffusion_full'
)

# Try all models, automatically select best (by AIC)
result = fit_mean_recovery_profile(
    time, intensity, sem,
    model='all'
)

print(f"Best model: {result['model_name']}")
print(f"Models tested: {result['n_models_tested']}")

# See all results
for model_result in result['all_results']:
    print(f"{model_result['model_name']}: AIC={model_result['aic']:.2f}")
```

## Interpretation Examples

### Example 1: Loss of Binding

**Scenario:** Comparing WT vs. DNA-binding mutant

**Results:**
```
Reaction-Diffusion (Simple) Model:
  Bound Fraction:
    WT: 65.2%
    Mutant: 18.3%
    Difference: -46.9%
  
  Effective Rate (k_eff):
    WT: 0.085 s⁻¹
    Mutant: 0.312 s⁻¹
    Fold change: 3.67x
```

**Interpretation:**
- Mutant lost ability to bind chromatin
- Faster effective rate indicates predominantly diffusive recovery
- Biological conclusion: Mutation disrupts DNA binding domain

### Example 2: Altered Diffusion Regime

**Scenario:** Comparing normal vs. crowded conditions

**Results:**
```
Anomalous Diffusion Model:
  Beta (β):
    Normal: 0.92 ± 0.03
    Crowded: 0.61 ± 0.04
    Difference: -0.31
  
  Tau (τ):
    Normal: 3.2 s
    Crowded: 6.8 s
    Fold change: 2.13x
```

**Interpretation:**
- Crowding causes shift from near-Brownian to strong subdiffusion
- Recovery is slower (higher τ)
- Biological conclusion: Molecular crowding creates obstacles to diffusion

### Example 3: Changed Binding Kinetics

**Scenario:** Comparing WT vs. phosphorylation mimic

**Results:**
```
Reaction-Diffusion (Full) Model:
  k_on:
    WT: 0.125 s⁻¹
    Phosphomimic: 0.048 s⁻¹
    Fold change: 0.38x (slower binding)
  
  k_off:
    WT: 0.082 s⁻¹
    Phosphomimic: 0.156 s⁻¹
    Fold change: 1.90x (faster unbinding)
  
  K_d:
    WT: 0.656
    Phosphomimic: 3.25
    Fold change: 4.95x (weaker binding)
```

**Interpretation:**
- Phosphorylation reduces binding rate (k_on)
- Phosphorylation increases unbinding rate (k_off)
- Overall: 5-fold reduction in binding affinity
- Biological conclusion: Phosphorylation destabilizes chromatin association

## Visualization

### Plot Fitted Curves

```python
from frap_plots import FRAPPlots

# Plot advanced group comparison
fig = FRAPPlots.plot_advanced_group_comparison(
    comparison_results,
    height=600
)

# Shows:
# - Data points with error bars (both groups)
# - Fitted curves (both groups)
# - Model name and R² values in title
```

### Plot Parameter Comparison

```python
fig = FRAPPlots.plot_parameter_comparison(
    comparison_results,
    height=500
)

# Shows:
# - Bar chart comparing fitted parameters
# - Side-by-side comparison for each parameter
# - Values labeled on bars
```

## Best Practices

### 1. Data Quality
- **Minimum cells per group:** 10-15 for reliable average
- **Time resolution:** At least 20 points post-bleach
- **Signal-to-noise:** Higher SNR improves fitting

### 2. Model Selection
- **Start with 'all':** Let algorithm select best model
- **Check R²:** Should be > 0.95 for good fits
- **Compare AIC:** Lower is better (accounts for model complexity)

### 3. Biological Validation
- **Parameter ranges:** Check if values are physically reasonable
- **Interpret carefully:** Statistical significance ≠ biological significance
- **Validate findings:** Use complementary techniques when possible

### 4. Reporting Results
Include in reports:
- Model used and selection criteria
- R² values for all groups
- Fitted parameters with standard errors
- Fold changes and biological interpretation

## Troubleshooting

### Poor Fits (Low R²)

**Problem:** R² < 0.90

**Solutions:**
1. Check data quality (outliers, noise)
2. Try different models
3. Ensure sufficient time resolution
4. Check for systematic errors (photobleaching, drift)

### Failed Convergence

**Problem:** Fitting returns `success: False`

**Solutions:**
1. Check for NaN or infinite values in data
2. Ensure time starts at 0 (post-bleach)
3. Normalize intensity (0-1 range)
4. Increase number of time points

### Unrealistic Parameters

**Problem:** Parameters outside physical range

**Solutions:**
1. Check bleach_radius_um value (affects D calculations)
2. Review normalization (baseline should be near 0)
3. Try simpler model
4. Check for pre-bleach artifacts

## Technical Details

### Fitting Algorithm
- Uses **lmfit** library (Levenberg-Marquardt)
- Weighted least squares (if SEM provided)
- Bounded parameters to ensure physical validity

### Model Selection
- **AIC** (Akaike Information Criterion): Balances fit quality vs. complexity
- Formula: `AIC = n*ln(SSR/n) + 2*k`
  - n: number of data points
  - SSR: sum of squared residuals
  - k: number of fitted parameters

### Statistical Metrics
- **R²**: Coefficient of determination (0-1, higher is better)
- **AIC**: Information criterion (lower is better)
- **BIC**: Bayesian information criterion (alternative to AIC)

## Dependencies

### Required
- **numpy**: Array operations
- **scipy**: Statistical functions
- **lmfit**: Advanced curve fitting

### Optional
- **plotly**: Interactive plotting (for UI)
- **matplotlib**: Static plotting (for reports)

### Installation

```bash
pip install lmfit
```

## References

### Anomalous Diffusion
- Höfling & Franosch (2013). "Anomalous transport in the crowded world of biological cells." *Rep. Prog. Phys.* 76:046602

### FRAP Theory
- Sprague et al. (2004). "Analysis of binding reactions by fluorescence recovery after photobleaching." *Biophys J.* 86:3473-3495

### Reaction-Diffusion
- Mueller et al. (2008). "FRAP and kinetic modeling in the analysis of nuclear protein dynamics." *Curr. Opin. Cell Biol.* 20:390-395

## Support

For questions or issues:
1. Check test script: `test_advanced_group_fitting.py`
2. Review function docstrings in `frap_advanced_fitting.py`
3. Consult this documentation

## Future Enhancements

Potential additions:
- Confidence intervals on parameters (bootstrap)
- Global fitting across multiple groups
- Additional models (power-law, double anomalous)
- Automatic outlier detection
- Report generation with fitted parameters

---

**Last Updated:** October 19, 2025  
**Version:** 1.0  
**Status:** Production Ready ✅
