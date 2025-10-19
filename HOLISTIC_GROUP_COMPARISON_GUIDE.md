# Holistic FRAP Group Comparison Guide

## Date: October 18, 2025

---

## The Problem with Traditional Comparisons

### What We Used to Do ❌
```python
# Compare individual components
Group1_fast_k = 0.8 s⁻¹  (60% of mobile population)
Group2_fast_k = 1.2 s⁻¹  (25% of mobile population)

# Conclusion: "Group 2 has faster kinetics"
# BUT THIS MISSES THE BIOLOGICAL STORY!
```

**What's wrong:** Group 2 might have actually **lost binding capability**, so most cells are now purely diffusing (which is fast). The mutation didn't "speed things up" - it eliminated specific interactions!

### What We Should Do ✅
```python
# Look at the whole population distribution
Group1: 40% diffusion, 60% binding
Group2: 75% diffusion, 25% binding

# Weighted kinetics
Group1: weighted_k = 0.5 s⁻¹ (mixture of fast and slow)
Group2: weighted_k = 0.9 s⁻¹ (dominated by fast)

# Conclusion: "Group 2 lost binding, shifted to diffusion"
# THIS TELLS THE REAL BIOLOGICAL STORY!
```

---

## Solution: Holistic Group Comparison

### Module: `frap_group_comparison.py`

This new module provides three levels of comparison:

1. **Averaged Recovery Profiles** - Direct comparison of curves
2. **Population Distributions** - How components are distributed
3. **Weighted Kinetics** - Abundance-weighted rate constants

---

## Usage Guide

### 1. Comparing Averaged Recovery Profiles

**Why:** This is the most direct comparison - just average all the curves and compare them.

```python
from frap_group_comparison import compute_average_recovery_profile, compare_recovery_profiles

# Prepare data for each group
group1_data = {
    file_path: {'time': t_array, 'intensity': i_array}
    for file_path in group1_files
}

group2_data = {
    file_path: {'time': t_array, 'intensity': i_array}
    for file_path in group2_files
}

# Compute averages
t1, i1_mean, i1_sem = compute_average_recovery_profile(group1_data)
t2, i2_mean, i2_sem = compute_average_recovery_profile(group2_data)

# Compare
comparison = compare_recovery_profiles(
    group1_data, group2_data,
    group1_name="WT", group2_name="Mutant"
)

# Plot
plt.figure(figsize=(10, 6))
plt.errorbar(t1, i1_mean, yerr=i1_sem, label="WT", capsize=3)
plt.errorbar(t2, i2_mean, yerr=i2_sem, label="Mutant", capsize=3)
plt.xlabel("Time (s)")
plt.ylabel("Normalized Intensity")
plt.legend()
plt.title("Averaged Recovery Profiles")
plt.show()
```

**Output:**
- Direct visual comparison
- No model assumptions
- Shows raw kinetic differences

---

### 2. Population Distribution Analysis

**Why:** Different mutations affect different populations (e.g., lose binding but keep diffusion).

```python
from frap_group_comparison import HolisticGroupComparator

# Initialize comparator
comparator = HolisticGroupComparator(
    bleach_radius_um=1.0,
    pixel_size=0.3
)

# Compute weighted kinetics for each group
group1_metrics = comparator.compute_weighted_kinetics(group1_features_df)
group2_metrics = comparator.compute_weighted_kinetics(group2_features_df)

print(f"Group 1 Population Distribution:")
print(f"  Diffusion: {group1_metrics['population_diffusion']:.1f}%")
print(f"  Binding: {group1_metrics['population_binding']:.1f}%")
print(f"  Intermediate: {group1_metrics['population_intermediate']:.1f}%")

print(f"\nGroup 2 Population Distribution:")
print(f"  Diffusion: {group2_metrics['population_diffusion']:.1f}%")
print(f"  Binding: {group2_metrics['population_binding']:.1f}%")
print(f"  Intermediate: {group2_metrics['population_intermediate']:.1f}%")
```

**Example Output:**
```
Group 1 Population Distribution:
  Diffusion: 35.2%
  Binding: 58.3%
  Intermediate: 6.5%

Group 2 Population Distribution:
  Diffusion: 72.1%
  Binding: 18.9%
  Intermediate: 9.0%

→ Group 2 lost ~40% of binding population, shifted to diffusion!
```

---

### 3. Weighted Kinetic Metrics

**Why:** Compare rate constants while accounting for how abundant each component is.

```python
# Compare multiple groups
group_features = {
    "WT": wt_features_df,
    "Mutant-A": mutA_features_df,
    "Mutant-B": mutB_features_df
}

comparison_table = comparator.compare_groups(group_features)
print(comparison_table)
```

**Example Output:**
```
group      n_cells  mobile_fraction_mean  weighted_k_fast  weighted_k_slow  population_diffusion  population_binding
WT         45       82.3 ± 2.1            0.65 s⁻¹        0.08 s⁻¹         35.2%                 58.3%
Mutant-A   38       78.9 ± 3.2            0.92 s⁻¹        0.07 s⁻¹         72.1%                 18.9%
Mutant-B   42       85.1 ± 2.8            0.61 s⁻¹        0.12 s⁻¹         28.3%                 68.7%
```

**Interpretation:**
- **Mutant-A:** Lost binding (18.9% vs 58.3%), shifted to diffusion
- **Mutant-B:** Enhanced binding (68.7% vs 58.3%), slower off-rate

---

### 4. Statistical Comparison with Interpretation

**Why:** Get automated statistical tests AND biological interpretation.

```python
# Statistical comparison
stats_results = comparator.statistical_comparison(
    wt_features_df, 
    mutant_features_df,
    group1_name="WT",
    group2_name="Mutant"
)

# Get narrative interpretation
interpretation = comparator.interpret_differences(stats_results)
print(interpretation)
```

**Example Output:**
```
📊 **Mobile Fraction Comparison:**
   WT: 82.3% ± 2.1%
   Mutant: 78.9% ± 3.2%
   ✗ No significant difference in overall mobility (p=0.3421)

🔬 **Population Distribution Analysis:**
   WT:
      Diffusion: 35.2%
      Binding: 58.3%
      Intermediate: 6.5%
   Mutant:
      Diffusion: 72.1%
      Binding: 18.9%
      Intermediate: 9.0%

   💡 **Key Finding:** Mutant shows increased diffusing population (36.9% shift)
   💡 **Key Finding:** Mutant shows decreased binding population (39.4% shift)

⚡ **Kinetic Rate Comparison:**
   Fast component (weighted by abundance):
      WT: k = 0.652 s⁻¹
      Mutant: k = 0.921 s⁻¹
      Fold change: 1.41x
   Slow component (weighted by abundance):
      WT: k = 0.082 s⁻¹
      Mutant: k = 0.074 s⁻¹
      Fold change: 0.90x

🧬 **Biological Interpretation:**
   → Mutant appears to have LOST BINDING CAPABILITY
      Population shifted from binding to diffusion/intermediate states
      This suggests mutation disrupts chromatin association
```

---

## Real-World Examples

### Example 1: DNA Binding Domain Mutation

**Scenario:** You mutated a DNA binding domain and want to know if it affects FRAP kinetics.

```python
# WT has two populations: fast diffusion (40%) and slow binding (60%)
# Mutant lost DNA binding - everything is now fast diffusion

comparator = HolisticGroupComparator()

wt_metrics = comparator.compute_weighted_kinetics(wt_df)
mutant_metrics = comparator.compute_weighted_kinetics(mutant_df)

# Compare
print(f"WT binding population: {wt_metrics['population_binding']:.1f}%")
print(f"Mutant binding population: {mutant_metrics['population_binding']:.1f}%")

# Result:
# WT binding population: 60.3%
# Mutant binding population: 8.2%
# → Mutation eliminated DNA binding!
```

### Example 2: Comparing Fast Rates When Abundance Differs

**Scenario:** Both WT and mutant have a "fast component", but it represents different fractions of the population.

```python
# WT: fast component (k=1.2 s⁻¹) in 30% of cells
# Mutant: fast component (k=1.5 s⁻¹) in 85% of cells

# Traditional comparison would say "mutant is slightly faster"
# Holistic comparison reveals:

wt_metrics = comparator.compute_weighted_kinetics(wt_df)
mutant_metrics = comparator.compute_weighted_kinetics(mutant_df)

print(f"WT weighted k_fast: {wt_metrics['weighted_k_fast']:.3f} s⁻¹")
print(f"WT diffusion population: {wt_metrics['population_diffusion']:.1f}%")

print(f"Mutant weighted k_fast: {mutant_metrics['weighted_k_fast']:.3f} s⁻¹")
print(f"Mutant diffusion population: {mutant_metrics['population_diffusion']:.1f}%")

# Result:
# WT weighted k_fast: 1.200 s⁻¹, 30% diffusion
# Mutant weighted k_fast: 1.500 s⁻¹, 85% diffusion
# → Mutant didn't get "faster" - it lost binding and became pure diffusion!
```

### Example 3: Subtle Kinetic Shifts

**Scenario:** Populations are similar, but rates within populations differ.

```python
# Both have ~50% binding, but mutant binds/unbinds faster

stats = comparator.statistical_comparison(wt_df, mutant_df, "WT", "Mutant")

if 'kinetics_comparison' in stats:
    k_fold = stats['kinetics_comparison']['k_slow_fold_change']
    if k_fold > 1.5:
        print("Mutant has faster on/off kinetics within binding population")
    elif k_fold < 0.67:
        print("Mutant has slower on/off kinetics within binding population")
```

---

## Integration with Streamlit UI

### Add to Group Analysis Tab

```python
import streamlit as st
from frap_group_comparison import HolisticGroupComparator, compute_average_recovery_profile

# In Tab 2: Group Analysis
st.subheader("Holistic Group Comparison")

# Select groups to compare
selected_groups = st.multiselect(
    "Select groups to compare:",
    options=list(dm.groups.keys()),
    default=list(dm.groups.keys())[:2] if len(dm.groups) >= 2 else []
)

if len(selected_groups) >= 2:
    comparator = HolisticGroupComparator()
    
    # Prepare features for each group
    group_features = {}
    for group_name in selected_groups:
        features_df = dm.groups[group_name].get('features_df')
        if features_df is not None:
            group_features[group_name] = features_df
    
    # Show comparison table
    st.write("### Population Distribution Comparison")
    comparison_df = comparator.compare_groups(group_features)
    st.dataframe(comparison_df)
    
    # Pairwise comparison if exactly 2 groups
    if len(selected_groups) == 2:
        group1, group2 = selected_groups
        
        # Statistical comparison
        stats = comparator.statistical_comparison(
            group_features[group1],
            group_features[group2],
            group1, group2
        )
        
        # Show interpretation
        st.write("### Biological Interpretation")
        interpretation = comparator.interpret_differences(stats)
        st.markdown(interpretation)
        
        # Plot averaged recovery profiles
        st.write("### Averaged Recovery Profiles")
        
        # Get data for each group
        group1_data = {fp: dm.files[fp] for fp in dm.groups[group1]['files']}
        group2_data = {fp: dm.files[fp] for fp in dm.groups[group2]['files']}
        
        # Compute averages
        t1, i1_mean, i1_sem = compute_average_recovery_profile(group1_data)
        t2, i2_mean, i2_sem = compute_average_recovery_profile(group2_data)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.errorbar(t1, i1_mean, yerr=i1_sem, label=group1, capsize=3, alpha=0.7)
        ax.errorbar(t2, i2_mean, yerr=i2_sem, label=group2, capsize=3, alpha=0.7)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Normalized Intensity")
        ax.legend()
        ax.set_title(f"Averaged Recovery: {group1} vs {group2}")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
```

---

## Key Concepts

### 1. Kinetic Regimes

The module categorizes rate constants into three regimes:

- **Diffusion** (k > 1.0 s⁻¹): Fast, unrestricted movement
- **Intermediate** (0.1 < k < 1.0 s⁻¹): Mixed behavior
- **Binding** (k < 0.1 s⁻¹): Slow, binding-dominated

These thresholds can be adjusted based on bleach spot size.

### 2. Weighted Kinetics

Instead of treating all components equally, we weight by abundance:

```python
# Traditional: average all k values
k_avg = (k1 + k2) / 2

# Weighted: account for how common each is
k_weighted = (k1 * prop1 + k2 * prop2) / (prop1 + prop2)
```

### 3. Population Distribution

For each cell, we classify each kinetic component:
- Single-component fit → 100% in one regime
- Double-component fit → split between two regimes
- Triple-component fit → split across three regimes

Then aggregate across all cells to get group-level percentages.

---

## Advantages Over Traditional Comparison

| Traditional Approach | Holistic Approach |
|---------------------|-------------------|
| Compare k1 vs k1 separately | Compare weighted k values |
| Compare k2 vs k2 separately | Compare population distributions |
| Miss population shifts | Identify biological mechanisms |
| "Mutant is faster" | "Mutant lost binding" |
| Hard to interpret | Clear biological story |

---

## Best Practices

1. **Always look at averaged curves first** - This is model-independent
2. **Check population distributions** - Identifies major shifts
3. **Use weighted kinetics** - Accounts for abundance differences
4. **Read the interpretation** - Connects to biology

5. **Don't just compare components** - A "faster k" might mean "lost binding"
6. **Consider biological context** - What does the mutation target?

---

## Summary

✅ **Use averaged recovery profiles** for direct comparison  
✅ **Use population distributions** to identify shifts  
✅ **Use weighted kinetics** to compare rates properly  
✅ **Use interpretation** to understand biology  

❌ **Don't compare k1 vs k1 without context**  
❌ **Don't ignore population abundances**  
❌ **Don't over-interpret small kinetic differences**  

---

**Module:** `frap_group_comparison.py`  
**Status:** ✅ Ready to use  
**Integration:** Add to Streamlit Tab 2 (Group Analysis)
