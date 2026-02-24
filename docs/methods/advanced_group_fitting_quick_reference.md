# Advanced Group Fitting - Quick Reference

## One-Line Summary
**Fit sophisticated biophysical models to mean recovery profiles to compare groups mechanistically.**

---

## Quick Start

```python
from frap_group_comparison import compare_recovery_profiles

# Compare groups with advanced fitting
comparison = compare_recovery_profiles(
    group1_data, group2_data,
    group1_name="WT", group2_name="Mutant",
    use_advanced_fitting=True,
    bleach_radius_um=1.0,
    advanced_model='all'  # Try all models, pick best
)

# Get results
adv = comparison['advanced_fitting']
print(adv['interpretation'])  # Biological interpretation
```

---

## Available Models

| Model | Use When | Key Parameters |
|-------|----------|----------------|
| **Anomalous Diffusion** | Studying diffusion regime | β (anomalous exponent), τ (time), D_eff |
| **Reaction-Diffusion (Simple)** | Separating diffusion vs binding | F_b (bound), k_eff (rate), D_app |
| **Reaction-Diffusion (Full)** | Need k_on/k_off explicitly | k_on, k_off, K_d, D |

---

## Model Selection

```python
model='all'                        # Automatic (best AIC)
model='anomalous'                  # Anomalous diffusion only
model='reaction_diffusion_simple'  # Simple RD
model='reaction_diffusion_full'    # Full RD with k_on/k_off
```

---

## Interpreting β (Beta)

| β Value | Regime | Meaning |
|---------|--------|---------|
| **β = 1** | Normal | Brownian diffusion |
| **β < 1** | Subdiffusion | Hindered by obstacles/crowding |
| **β > 1** | Superdiffusion | Directed transport (rare) |

---

## Common Scenarios

### Scenario 1: Loss of Binding
```
Model: Reaction-Diffusion (Simple)
Finding: F_b decreases, k_eff increases
Interpretation: Mutant lost chromatin binding
```

### Scenario 2: Changed Diffusion Regime
```
Model: Anomalous Diffusion
Finding: β decreases (e.g., 0.95 → 0.60)
Interpretation: Increased molecular crowding
```

### Scenario 3: Altered Binding Kinetics
```
Model: Reaction-Diffusion (Full)
Finding: k_on decreases, k_off increases
Interpretation: Reduced binding affinity
```

---

## Accessing Results

```python
adv = comparison['advanced_fitting']

# Model info
adv['model_used']        # Which model was selected
adv['r2_group1']         # R² for group 1
adv['r2_group2']         # R² for group 2

# Parameters
adv['parameter_comparison']  # Fold changes
adv['metric_comparison']     # Derived metrics

# Interpretation
adv['interpretation']    # Full narrative text

# Fitted curves
adv['group1_fit']['fitted_values']
adv['group2_fit']['fitted_values']
```

---

## Plotting

```python
from frap_plots import FRAPPlots

# Plot data + fitted curves
fig = FRAPPlots.plot_advanced_group_comparison(comparison)

# Plot parameter comparison bars
fig = FRAPPlots.plot_parameter_comparison(comparison)
```

---

## Quality Checks

| Metric | Good | Acceptable | Poor |
|--------|------|------------|------|
| **R²** | > 0.98 | 0.95 - 0.98 | < 0.95 |
| **AIC** | Lower is better | Compare models | - |
| **Parameters** | Physical range | Check units | Unrealistic |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Low R² | Try different model, check data quality |
| Failed fit | Check normalization, ensure time starts at 0 |
| Unrealistic params | Verify bleach_radius_um value |
| Missing lmfit | `pip install lmfit` |

---

## Parameter Units

| Parameter | Unit | Typical Range |
|-----------|------|---------------|
| **β** | dimensionless | 0.3 - 1.5 |
| **τ** | seconds | 0.5 - 30 s |
| **D** | μm²/s | 0.01 - 10 |
| **k_on, k_off** | s⁻¹ | 0.001 - 10 |
| **K_d** | dimensionless | 0.1 - 10 |

---

## When to Use This Feature

✅ **Use when:**
- Comparing mechanistic differences between conditions
- Need to separate diffusion vs. binding
- Want to detect anomalous diffusion
- Have good quality averaged data (n ≥ 10 cells)

❌ **Don't use when:**
- Single cells or small sample size (n < 5)
- Poor data quality (high noise)
- Simple comparison sufficient
- lmfit not installed

---

## Dependencies

```bash
pip install lmfit
```

Required: numpy, scipy, lmfit  
Optional: plotly, matplotlib

---

## Example Output

```
Model: anomalous_diffusion
R² (WT): 0.9994, R² (Mutant): 0.9992

Parameter Comparison:
  Beta (β):
    WT: 0.960
    Mutant: 0.607
    Fold change: 0.63x
  
  Tau (τ):
    WT: 2.98 s
    Mutant: 4.97 s
    Fold change: 1.67x

Interpretation:
  Mutant shows more hindered diffusion (increased subdiffusion).
  This suggests increased molecular crowding or obstacles.
```

---

## Files

- **Source:** `frap_advanced_fitting.py`, `frap_group_comparison.py`
- **Plots:** `frap_plots.py`
- **Test:** `test_advanced_group_fitting.py`
- **Guide:** `ADVANCED_GROUP_FITTING_GUIDE.md`
- **Summary:** `ADVANCED_GROUP_FITTING_SUMMARY.md`

---

**Last Updated:** October 19, 2025  
**Version:** 1.0  
**Status:** Production Ready ✅
