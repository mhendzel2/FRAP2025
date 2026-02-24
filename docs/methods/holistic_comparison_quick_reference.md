# 🔬 Holistic Group Comparison - Quick Reference

## What It Does

Compares FRAP groups by analyzing **population distributions** and **weighted kinetics** instead of just comparing individual components.

---

## Why Use It?

### ❌ Traditional Comparison Says:
"Mutant has faster k₁ (1.5 vs 1.2 s⁻¹)"

### ✅ Holistic Comparison Says:
"Mutant lost 40% of binding population, shifted to pure diffusion"

**The second one reveals the biological mechanism!**

---

## How to Use

### 1. Navigate to Interface
```
Open App → Tab 2 (Group Analysis) → Scroll Down → "🔬 Holistic Group Comparison"
```

### 2. Select Groups
- Choose 2 or more groups from dropdown
- For detailed comparison: select exactly 2 groups

### 3. View Results

#### Population Distribution Table
Shows what percentage of each group is in:
- **Diffusion** (k > 1.0 s⁻¹): Fast, unrestricted movement
- **Intermediate** (0.1 < k < 1.0 s⁻¹): Mixed behavior
- **Binding** (k < 0.1 s⁻¹): Slow, chromatin-bound

#### Population Distribution Chart
Stacked bar chart showing visual representation of population shifts

#### Statistical Comparison (2 groups only)
- Mobile fraction comparison
- Population shift metrics
- Biological interpretation

#### Averaged Recovery Profiles
- Direct curve comparison (no model assumptions)
- Quantitative difference metrics

---

## Interpreting Results

### Population Shifts

**Example Output:**
```
WT:     Diffusion: 35%  |  Binding: 60%
Mutant: Diffusion: 75%  |  Binding: 20%

Interpretation: Mutant LOST BINDING CAPABILITY
→ 40% shift from binding to diffusion
```

### Weighted Kinetics

**Why It Matters:**
```
WT:     k_fast = 1.2 s⁻¹ (30% of cells)
        k_slow = 0.05 s⁻¹ (70% of cells)
        → Weighted k_overall = 0.4 s⁻¹

Mutant: k_fast = 1.5 s⁻¹ (85% of cells)
        k_slow = 0.08 s⁻¹ (15% of cells)
        → Weighted k_overall = 1.3 s⁻¹

Fold Change: 3.3x faster (accounting for populations!)
```

### Biological Interpretation

The system automatically generates narratives like:

✅ **"Group shows ENHANCED BINDING"**
- If binding population increases >15%

✅ **"Group shows LOST BINDING CAPABILITY"**
- If binding population decreases >15%

✅ **"Groups are KINETICALLY SIMILAR"**
- If weighted rates differ <1.5-fold

---

## Common Use Cases

### 1. DNA Binding Domain Mutation
**Question:** Does mutation affect DNA binding?

**Check:**
- Binding population percentage
- Population shift from WT to mutant
- Interpretation narrative

**Expected Result:** 
- Binding ↓ = Mutation disrupted binding
- Binding ↑ = Mutation enhanced binding

### 2. Post-Translational Modification
**Question:** Does phosphorylation change chromatin interaction?

**Check:**
- Population distribution comparison
- Weighted k values
- Profile overlay

**Expected Result:**
- Population shift reveals mechanism
- Weighted k shows functional impact

### 3. Drug Treatment Effect
**Question:** Does drug alter protein dynamics?

**Check:**
- All population metrics
- Statistical significance
- Profile differences

**Expected Result:**
- Population changes show target engagement
- Kinetic changes show functional effect

---

## Tips & Best Practices

### ✅ Do This

1. **Start with population distribution** - This gives the big picture
2. **Check statistical significance** - Don't over-interpret small changes
3. **Read the interpretation** - The narrative explains what shifts mean
4. **View averaged profiles** - Visual confirmation of differences
5. **Compare multiple groups** - See patterns across conditions

### ❌ Avoid This

1. **Don't compare just k values** - They're meaningless without abundance
2. **Don't ignore small populations** - Even 10% can be significant
3. **Don't skip the profile view** - It's model-independent validation
4. **Don't compare single cells** - This is for group-level analysis
5. **Don't trust p-values alone** - Look at effect sizes (Cohen's d)

---

## Understanding the Metrics

### Mobile Fraction
- **Mean ± SEM**: Average mobile fraction across group
- **p-value**: Statistical significance
- **Cohen's d**: Effect size (>0.8 = large effect)

### Population Distribution
- **Diffusion %**: Fast-moving, freely diffusing component
- **Binding %**: Slow-moving, chromatin-bound component
- **Intermediate %**: Everything in between

### Weighted Kinetics
- **weighted_k_fast**: Fast rate weighted by abundance
- **weighted_k_slow**: Slow rate weighted by abundance
- **Fold Change**: Ratio of weighted rates

### Profile Metrics
- **Max Difference**: Largest gap between averaged curves
- **Mean Difference**: Average gap across all time points
- **RMSD**: Overall deviation between curves

---

## Troubleshooting

### "No groups available"
→ Create groups first (sidebar → "Create New Group")

### "Select at least 2 groups"
→ Choose multiple groups from dropdown

### "Could not import holistic comparison module"
→ Make sure `frap_group_comparison.py` is in project folder

### "Error computing population comparison"
→ Check that groups have valid fitted data (Tab 2 → Analyze group first)

### Population shows "100% diffusion, 0% binding" for all groups
→ Check bleach radius setting (may need adjustment for your data)

---

## Keyboard Shortcuts

- **Ctrl + F**: Find text in interpretation
- **Shift + Scroll**: Zoom plot vertically
- **Drag**: Pan plot
- **Double-click**: Reset zoom

---

## Export Options

### Current Capabilities
- 📊 Screenshot plots (click camera icon)
- 📋 Copy interpretation text
- 💾 Download comparison table (coming soon)

### Future Features
- PDF report generation
- Excel export of all metrics
- Batch comparison reports

---

## Examples by Research Question

### "Did I knock out binding?"
1. Select WT + Knockout groups
2. Look at binding %: 
   - WT: 60% → KO: 10% = ✅ YES, binding knocked out
   - WT: 60% → KO: 55% = ❌ NO, binding intact

### "Is my mutant faster or just less bound?"
1. Check population distribution:
   - If diffusion ↑ → Less bound (population shift)
   - If diffusion ≈ same → Actually faster (kinetic change)

### "Did drug treatment work?"
1. Select Untreated + Treated groups
2. Check interpretation:
   - Population shift = Target engaged
   - No change = Drug ineffective or off-target

---

## Statistical Thresholds

### Significance Levels
- **p < 0.001**: *** (highly significant)
- **p < 0.01**: ** (very significant)
- **p < 0.05**: * (significant)
- **p ≥ 0.05**: ns (not significant)

### Effect Sizes (Cohen's d)
- **|d| > 0.8**: Large effect
- **|d| > 0.5**: Medium effect
- **|d| > 0.2**: Small effect
- **|d| < 0.2**: Negligible effect

### Population Shift Thresholds
- **>15%**: Major shift (biological significance)
- **5-15%**: Moderate shift
- **<5%**: Minor shift (may not be meaningful)

---

## Quick Checklist

Before interpreting results, verify:

- [ ] Both groups analyzed successfully (Tab 2)
- [ ] Outliers removed if needed
- [ ] Bleach radius set correctly (Settings)
- [ ] Sufficient cells per group (n ≥ 20 recommended)
- [ ] Groups are from same experimental batch
- [ ] Same microscope settings for all groups

---

## When to Use Each Analysis

| Analysis Type | When to Use | What It Reveals |
|--------------|-------------|-----------------|
| **Population Distribution** | Always start here | Big picture mechanism |
| **Weighted Kinetics** | After seeing population shift | Functional impact |
| **Statistical Comparison** | 2 groups only | Confidence in differences |
| **Averaged Profiles** | Visual learners | Model-independent validation |
| **All Together** | Final interpretation | Complete biological story |

---

## Remember

**The goal is not just to find differences, but to understand biological mechanisms.**

- Population shifts reveal **what changed** (binding loss, diffusion gain)
- Weighted kinetics reveal **how much** it changed functionally
- Profiles reveal **when** differences appear during recovery
- Interpretation reveals **what it means** biologically

---

## Quick Reference Table

| You Want To... | Look At... |
|----------------|-----------|
| See big picture difference | Population distribution chart |
| Quantify binding loss | Binding % shift |
| Know if it's significant | p-value and Cohen's d |
| Understand what it means | Biological interpretation |
| Validate without models | Averaged recovery profiles |
| Compare many groups | Population distribution table |
| Get detailed statistics | Pairwise comparison metrics |

---

**Need More Help?**

📖 See: `HOLISTIC_GROUP_COMPARISON_GUIDE.md` for detailed examples  
🔧 See: `HOLISTIC_COMPARISON_IMPLEMENTATION_SUMMARY.md` for technical details  
💬 Ask: Include "holistic comparison" in your question

---

**Last Updated:** October 18, 2025  
**Version:** 1.0  
**Status:** Production Ready ✅
