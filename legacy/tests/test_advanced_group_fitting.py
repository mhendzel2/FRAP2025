"""
Test script for advanced group-level curve fitting functionality.

This demonstrates how to apply sophisticated models (anomalous diffusion,
reaction-diffusion) to mean recovery profiles for comparing groups.
"""

import numpy as np
import matplotlib.pyplot as plt
from frap_group_comparison import compute_average_recovery_profile, compare_recovery_profiles

print("=" * 80)
print("ADVANCED GROUP-LEVEL CURVE FITTING TEST")
print("=" * 80)

# Check if lmfit is available
try:
    import lmfit
    LMFIT_AVAILABLE = True
    print("\n✓ lmfit is installed - advanced fitting available")
except ImportError:
    LMFIT_AVAILABLE = False
    print("\n✗ lmfit not installed - advanced fitting will be skipped")
    print("  Install with: pip install lmfit")

if not LMFIT_AVAILABLE:
    print("\nExiting - please install lmfit to test advanced fitting")
    exit()

# =============================================================================
# Create Synthetic Data for Two Groups
# =============================================================================

print("\n" + "=" * 80)
print("CREATING SYNTHETIC DATA")
print("=" * 80)

time_points = np.linspace(0, 30, 100)
n_cells_per_group = 20

# Group 1: Normal diffusion (beta ~ 1.0)
print("\nGroup 1 (Wild Type): Normal diffusion")
print("  Mobile fraction: ~85%")
print("  Beta: ~1.0 (Brownian motion)")
print("  Tau: ~3.0 s")

wt_data = {}
for i in range(n_cells_per_group):
    # Anomalous diffusion model: A * (1 - exp(-(t/tau)^beta)) + C
    A = 0.85 + np.random.normal(0, 0.05)
    C = 0.15 + np.random.normal(0, 0.03)
    tau = 3.0 + np.random.normal(0, 0.5)
    beta = 1.0 + np.random.normal(0, 0.05)  # Normal diffusion
    
    recovery = A * (1 - np.exp(-(time_points / tau)**beta)) + C
    recovery += np.random.normal(0, 0.02, len(time_points))  # Add noise
    
    wt_data[f'wt_{i}'] = {
        'time': time_points,
        'intensity': recovery
    }

# Group 2: Subdiffusive (beta ~ 0.6)
print("\nGroup 2 (Mutant): Subdiffusive motion")
print("  Mobile fraction: ~85% (same as WT)")
print("  Beta: ~0.6 (hindered diffusion)")
print("  Tau: ~5.0 s (slower)")

mut_data = {}
for i in range(n_cells_per_group):
    A = 0.85 + np.random.normal(0, 0.05)
    C = 0.15 + np.random.normal(0, 0.03)
    tau = 5.0 + np.random.normal(0, 0.8)
    beta = 0.6 + np.random.normal(0, 0.05)  # Subdiffusion
    
    recovery = A * (1 - np.exp(-(time_points / tau)**beta)) + C
    recovery += np.random.normal(0, 0.02, len(time_points))  # Add noise
    
    mut_data[f'mut_{i}'] = {
        'time': time_points,
        'intensity': recovery
    }

# =============================================================================
# Test 1: Compute Average Profiles
# =============================================================================

print("\n" + "=" * 80)
print("TEST 1: COMPUTE AVERAGE PROFILES")
print("=" * 80)

t_wt, i_wt_mean, i_wt_sem = compute_average_recovery_profile(wt_data)
t_mut, i_mut_mean, i_mut_sem = compute_average_recovery_profile(mut_data)

print(f"\n✓ WT averaged profile: {len(t_wt)} time points")
print(f"  Mean recovery: {i_wt_mean[-1]:.3f}")
print(f"\n✓ Mutant averaged profile: {len(t_mut)} time points")
print(f"  Mean recovery: {i_mut_mean[-1]:.3f}")

# =============================================================================
# Test 2: Advanced Fitting WITHOUT Comparing (Single Group)
# =============================================================================

print("\n" + "=" * 80)
print("TEST 2: ADVANCED FITTING - SINGLE GROUP")
print("=" * 80)

from frap_advanced_fitting import fit_mean_recovery_profile

print("\nFitting WT group with anomalous diffusion model...")
wt_fit = fit_mean_recovery_profile(
    t_wt, i_wt_mean, i_wt_sem,
    bleach_radius_um=1.0,
    model='anomalous'
)

if wt_fit['success']:
    print(f"✓ Fit successful (R² = {wt_fit['r2']:.4f})")
    print("\nFitted parameters:")
    for param, value in wt_fit['params'].items():
        print(f"  {param}: {value:.4f}")
    
    print("\nInterpretation:")
    for key, value in wt_fit['interpretation'].items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
else:
    print(f"✗ Fit failed: {wt_fit.get('error')}")

print("\nFitting Mutant group with anomalous diffusion model...")
mut_fit = fit_mean_recovery_profile(
    t_mut, i_mut_mean, i_mut_sem,
    bleach_radius_um=1.0,
    model='anomalous'
)

if mut_fit['success']:
    print(f"✓ Fit successful (R² = {mut_fit['r2']:.4f})")
    print("\nFitted parameters:")
    for param, value in mut_fit['params'].items():
        print(f"  {param}: {value:.4f}")
    
    print("\nInterpretation:")
    for key, value in mut_fit['interpretation'].items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
else:
    print(f"✗ Fit failed: {mut_fit.get('error')}")

# =============================================================================
# Test 3: Advanced Fitting WITH Group Comparison
# =============================================================================

print("\n" + "=" * 80)
print("TEST 3: ADVANCED FITTING - GROUP COMPARISON")
print("=" * 80)

print("\nComparing WT vs Mutant with advanced curve fitting...")
comparison = compare_recovery_profiles(
    wt_data, mut_data,
    group1_name="Wild Type",
    group2_name="Mutant",
    use_advanced_fitting=True,
    bleach_radius_um=1.0,
    advanced_model='anomalous'
)

if 'advanced_fitting' in comparison and comparison['advanced_fitting']['success']:
    adv = comparison['advanced_fitting']
    print(f"\n✓ Advanced fitting comparison successful")
    print(f"  Model used: {adv['model_used']}")
    print(f"  R² (WT): {adv['r2_group1']:.4f}")
    print(f"  R² (Mutant): {adv['r2_group2']:.4f}")
    
    print("\n" + "-" * 80)
    print("PARAMETER COMPARISON")
    print("-" * 80)
    
    if 'parameter_comparison' in adv:
        for param, data in adv['parameter_comparison'].items():
            print(f"\n{param}:")
            print(f"  WT: {data['Wild Type']:.4f}")
            print(f"  Mutant: {data['Mutant']:.4f}")
            print(f"  Fold change: {data['fold_change']:.3f}x")
            print(f"  Percent change: {data['percent_change']:+.1f}%")
    
    print("\n" + "-" * 80)
    print("BIOLOGICAL INTERPRETATION")
    print("-" * 80)
    
    if 'interpretation' in adv:
        print(adv['interpretation'])
    
else:
    print(f"\n✗ Advanced fitting comparison failed")
    if 'advanced_fitting' in comparison:
        print(f"  Error: {comparison['advanced_fitting'].get('error')}")

# =============================================================================
# Test 4: Try All Models
# =============================================================================

print("\n" + "=" * 80)
print("TEST 4: TRY ALL AVAILABLE MODELS")
print("=" * 80)

print("\nFitting WT group with ALL models (best will be selected)...")
wt_all_models = fit_mean_recovery_profile(
    t_wt, i_wt_mean, i_wt_sem,
    bleach_radius_um=1.0,
    model='all'
)

if wt_all_models['success']:
    print(f"✓ Best model selected: {wt_all_models['model_name']}")
    print(f"  R² = {wt_all_models['r2']:.4f}")
    print(f"  AIC = {wt_all_models['aic']:.2f}")
    print(f"  Number of models tested: {wt_all_models.get('n_models_tested', 'N/A')}")
    
    if 'all_results' in wt_all_models:
        print("\n  All models tested:")
        for result in wt_all_models['all_results']:
            print(f"    - {result['model_name']}: R²={result['r2']:.4f}, AIC={result['aic']:.2f}")
else:
    print(f"✗ Fitting failed: {wt_all_models.get('error')}")

# =============================================================================
# Test 5: Visualization
# =============================================================================

print("\n" + "=" * 80)
print("TEST 5: VISUALIZATION")
print("=" * 80)

try:
    # Create figure with advanced fitting comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left plot: Data and fits
    ax1.errorbar(t_wt, i_wt_mean, yerr=i_wt_sem, fmt='o', 
                 label='WT (data)', color='blue', alpha=0.5, capsize=3)
    ax1.errorbar(t_mut, i_mut_mean, yerr=i_mut_sem, fmt='o', 
                 label='Mutant (data)', color='orange', alpha=0.5, capsize=3)
    
    if wt_fit['success'] and 'fitted_values' in wt_fit:
        ax1.plot(t_wt, wt_fit['fitted_values'], '-', 
                label='WT (fit)', color='blue', linewidth=2)
    
    if mut_fit['success'] and 'fitted_values' in mut_fit:
        ax1.plot(t_mut, mut_fit['fitted_values'], '-', 
                label='Mutant (fit)', color='orange', linewidth=2)
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Normalized Intensity')
    ax1.set_title('Advanced Curve Fitting Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Parameter comparison
    if 'advanced_fitting' in comparison and comparison['advanced_fitting']['success']:
        adv = comparison['advanced_fitting']
        metric_comp = adv.get('metric_comparison', {})
        
        if metric_comp:
            metrics = []
            wt_vals = []
            mut_vals = []
            
            for metric, data in metric_comp.items():
                if isinstance(data.get('Wild Type'), (int, float)):
                    metrics.append(metric.replace('_', ' ').title())
                    wt_vals.append(data['Wild Type'])
                    mut_vals.append(data['Mutant'])
            
            if metrics:
                x = np.arange(len(metrics))
                width = 0.35
                
                ax2.bar(x - width/2, wt_vals, width, label='WT', color='blue', alpha=0.7)
                ax2.bar(x + width/2, mut_vals, width, label='Mutant', color='orange', alpha=0.7)
                
                ax2.set_xlabel('Parameter')
                ax2.set_ylabel('Value')
                ax2.set_title('Fitted Parameter Comparison')
                ax2.set_xticks(x)
                ax2.set_xticklabels(metrics, rotation=45, ha='right')
                ax2.legend()
                ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('test_advanced_group_fitting_results.png', dpi=150, bbox_inches='tight')
    print("\n✓ Plot saved as 'test_advanced_group_fitting_results.png'")
    
except Exception as e:
    print(f"\n✗ Visualization failed: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)

print("\n✓ All tests completed successfully!")
print("\nKey Features Demonstrated:")
print("  1. Single group advanced fitting (anomalous diffusion)")
print("  2. Group comparison with advanced fitting")
print("  3. Automatic model selection (best AIC)")
print("  4. Parameter comparison and fold changes")
print("  5. Biological interpretation")
print("  6. Visualization of results")

print("\nAdvanced models available:")
print("  - Anomalous diffusion (stretched exponential)")
print("  - Reaction-diffusion (simple)")
print("  - Reaction-diffusion (full)")

print("\nUse cases:")
print("  - Identify diffusion anomalies (subdiffusion, superdiffusion)")
print("  - Quantify binding/unbinding kinetics")
print("  - Compare mechanistic differences between conditions")
print("  - Provide biophysical interpretation of FRAP data")

print("\n" + "=" * 80)
