# Estimation Plots for Multi-Group Comparisons

## Date: October 19, 2025

---

## What Are Estimation Plots?

Estimation plots are a modern, statistically rigorous way to visualize group comparisons. They **show the complete data story** instead of hiding information in summary statistics.

### Traditional Bar Chart ❌
```
┌────┐
│ 80%│    ← Shows only mean and error bar
└────┘    ← Hides distribution, outliers, sample size
```

**Problems:**
- Hides individual data points
- Doesn't show distribution shape
- Makes sample size invisible
- Can mislead with small samples

### Estimation Plot ✅
```
Left Panel:              Right Panel:
○ ○ ○ ○                 Effect Size
○ ○○ ○    ◆            ◆────●────
  ○○                    (difference)
Group 1    Mean±CI      vs. Control
```

**Advantages:**
- Shows all raw data
- Displays distribution
- Visualizes uncertainty (CI)
- Reveals effect sizes

---

## How to Use in FRAP2025

### Location
**Tab 3: Multi-Group Comparison → Visualization Type → "Estimation Plot"**

### Quick Start
1. Load multiple groups in Tab 2
2. Go to Tab 3: Multi-Group Comparison
3. Select parameter (e.g., `mobile_fraction`)
4. Choose "Estimation Plot" from dropdown
5. View comprehensive visualization

---

## Understanding the Plot

### Left Panel: Raw Data with Mean ± 95% CI

#### Elements
- **Small circles** (semi-transparent): Individual measurements
  - Each dot = one cell/measurement
  - Jittered horizontally for visibility
  - Color-coded by group

- **Large diamonds**: Group means
  - Solid fill
  - Black outline
  - Offset to the right of raw data

- **Error bars**: 95% Confidence Intervals
  - Vertical lines through diamonds
  - **Not** standard error bars
  - **Not** standard deviation
  - **95% CI** = "We're 95% confident the true mean lies within this range"

#### Reading the Left Panel

**Example:**
```
Group 1:
  ○ ○○○           ◆
  ○ ○ ○ ○      ───┼───
   ○○              ↑
                95% CI

Interpretation:
- 10 individual measurements (10 circles)
- Mean ≈ 75% (diamond position)
- 95% CI: roughly 72-78%
- Data are fairly tight (small CI)
```

**What to Look For:**
1. **Distribution shape**: Are points clustered or spread out?
2. **Sample size**: Count the dots
3. **Outliers**: Any dots far from the pack?
4. **Overlap**: Do CIs from different groups overlap?

---

### Right Panel: Mean Differences (Effect Size)

This panel shows **how much each group differs from the reference group** (usually the first group or control).

#### Elements
- **Y-axis**: Lists comparisons (e.g., "Mutant - WT")
- **X-axis**: Magnitude of difference
- **Diamonds**: Mean difference
- **Error bars**: 95% CI of the difference
- **Dashed vertical line**: Zero (no difference)

#### Reading the Right Panel

**Example:**
```
            Difference from Control
Mutant-A    ◆────●────
            -15%
            
Mutant-B         ◆────●────
                 +5%
                 
     ←──────┊──────→
    Decrease│Increase
            0
```

**Interpretation:**
- **Mutant-A**: 15% **lower** than control, CI doesn't cross zero → significant decrease
- **Mutant-B**: 5% **higher** than control, CI crosses zero → not significant

**What to Look For:**
1. **Direction**: Left of zero = decrease, right = increase
2. **Magnitude**: How big is the effect?
3. **Significance**: Does CI cross zero?
   - **Crosses zero**: Not statistically significant
   - **Doesn't cross zero**: Statistically significant
4. **Precision**: Narrow CI = precise estimate

---

## Interpreting Estimation Plots

### Case 1: Clear Difference

```
Group 1:  ○○○  ◆──●──
Group 2:        ○○○  ◆──●──

Right panel: Group2 - Group1
             ◆────●────
             (doesn't cross zero)
```

**Interpretation:**
✅ Groups are clearly different  
✅ Effect size is visible (separation)  
✅ Statistically significant (CI doesn't cross zero)  
✅ **Conclusion**: Real biological difference detected

---

### Case 2: Overlapping but Different

```
Group 1:  ○○○○ ◆────●────
Group 2:      ○○○○ ◆────●────

Right panel: Group2 - Group1
                ◆───●───
             (just barely doesn't cross zero)
```

**Interpretation:**
⚠️ Groups have overlapping distributions  
⚠️ But means are statistically different  
⚠️ Effect size is modest  
⚠️ **Conclusion**: Significant but small effect - biological relevance?

---

### Case 3: No Difference

```
Group 1:  ○○○○ ◆────●────
Group 2:  ○○○○ ◆────●────

Right panel: Group2 - Group1
               ◆────●────
             (centered on zero, crosses it)
```

**Interpretation:**
✗ Groups have similar distributions  
✗ Means are not significantly different  
✗ CI crosses zero  
✗ **Conclusion**: No evidence of difference

---

### Case 4: High Variability

```
Group 1:  ○  ○  ○ ◆──────●──────
          ○      ○
Group 2:     ○  ○ ◆──────●──────
          ○    ○   ○
```

**Interpretation:**
⚠️ Wide spread in data (biological variability)  
⚠️ Large error bars (uncertainty)  
⚠️ May need more samples  
⚠️ **Conclusion**: Trends present but high noise

---

### Case 5: Outlier Detection

```
Group 1:  ○○○○○ ◆──●──
          ○
              ○  ← outlier far from cluster
```

**Interpretation:**
⚠️ Most data clustered  
⚠️ One or two outliers  
⚠️ **Action**: Investigate outliers - experimental error or biological?

---

## Real-World FRAP Examples

### Example 1: Mobile Fraction Comparison

**Scenario**: Wild-type vs. DNA-binding mutant

```
WT:     ○○○○○○ ◆──●──
        (mobile fraction ≈ 80%)

Mutant:       ○○○○○○ ◆──●──
              (mobile fraction ≈ 60%)

Right Panel: Mutant - WT
             ◆────●────
             -20%
```

**Interpretation:**
- Mutant has **20% lower mobile fraction**
- Effect is **highly significant** (CI doesn't cross zero)
- Raw data show clear separation
- **Biology**: Mutation increases immobile population (more binding)

---

### Example 2: Recovery Rate Comparison

**Scenario**: Three treatments with different kinetics

```
Control:   ○○○  ◆──●──       (k ≈ 0.5 s⁻¹)
Drug A:      ○○○  ◆──●──     (k ≈ 0.7 s⁻¹)
Drug B:         ○○○  ◆──●──  (k ≈ 0.9 s⁻¹)

Right Panel:
Drug A - Control   ◆───●───  (+0.2 s⁻¹)
Drug B - Control      ◆───●───  (+0.4 s⁻¹)
```

**Interpretation:**
- Both drugs **increase recovery rate**
- Drug B has **larger effect** than Drug A
- Both effects are **significant** (CIs don't cross zero)
- Dose-response relationship visible
- **Biology**: Drugs reduce binding, increase diffusion

---

### Example 3: Half-Time Analysis

**Scenario**: Testing if mutation affects recovery speed

```
WT:     ○○○○ ◆────●────   (t½ = 15s ± 3s)
Mutant: ○○○○ ◆────●────   (t½ = 16s ± 3s)

Right Panel: Mutant - WT
               ◆──────●──────
             (crosses zero widely)
```

**Interpretation:**
- Distributions largely overlap
- Mean difference is **small** (+1s)
- CI **crosses zero** → not significant
- Wide CIs indicate variability
- **Biology**: Mutation doesn't affect kinetics significantly

---

## Statistical Considerations

### Confidence Intervals vs. Error Bars

Traditional plots often show:
- **Standard Deviation (SD)**: Describes data spread
- **Standard Error (SEM)**: Describes mean precision

Estimation plots show:
- **95% Confidence Interval**: "We're 95% confident true mean is in this range"

**Why 95% CI is better:**
- Direct statistical inference
- Can see significance by checking if CIs overlap zero (in difference plot)
- More honest about uncertainty than SEM

### Sample Size Visibility

**Estimation plots make sample size obvious:**
- **5 dots**: Small sample, wide CIs
- **50 dots**: Large sample, narrow CIs
- **Traditional bar chart**: No idea if N=3 or N=30!

**Example:**
```
Small Sample (N=5):
○ ○ ○       ◆────────●────────
    ○ ○
(Wide CI = high uncertainty)

Large Sample (N=50):
○○○○○○○○○   ◆──●──
○○○○○○○○○
(Narrow CI = low uncertainty)
```

---

## Best Practices

### When to Use Estimation Plots

✅ **Comparing 2-6 groups** - Clear visualization  
✅ **Showing effect sizes** - Magnitude matters  
✅ **Publication figures** - Shows raw data  
✅ **Initial exploration** - Spot patterns and outliers  
✅ **Transparent reporting** - All data visible  

### When to Use Other Plots

**Box Plot**: Quartiles and outliers important  
**Violin Plot**: Distribution shape critical  
**Bar Chart**: Simple mean comparison (less informative)  

### Design Choices in FRAP2025

1. **Jittering**: Points spread horizontally for visibility
   - Seed=42 for reproducibility
   - Jitter width = 0.1 (adjustable)

2. **Colors**: Consistent across panels
   - Same color for group in both left and right panels
   - Transparent raw data (0.4 opacity)
   - Solid means (1.0 opacity)

3. **Reference Group**: First group in list
   - Usually control or wild-type
   - All comparisons relative to this

---

## Common Questions

**Q: Why do the dots overlap even though they're jittered?**  
A: With many samples (N>20), some overlap is unavoidable. The jitter helps but doesn't eliminate all overlap.

**Q: Can I click on individual dots to see which file?**  
A: Not currently - use the interactive dashboard in Tab 2, Step 5 for that feature.

**Q: What if my CI doesn't cross zero but groups look similar?**  
A: With large samples, small differences can be statistically significant but not biologically meaningful. Consider effect size!

**Q: Why do my groups have different numbers of points?**  
A: That's real - different groups may have different sample sizes. Estimation plots make this transparent.

**Q: Should I always use estimation plots?**  
A: They're great for 2-6 groups. For >6 groups, box plots might be clearer.

**Q: What does it mean if the right panel diamond is ON the zero line?**  
A: The mean difference is exactly zero (or very close). Groups are equivalent.

---

## Integration with Other Features

### Combine with Interactive Dashboard (Tab 2, Step 5)
1. Use estimation plot to spot group differences
2. Identify outlier points
3. Use dashboard to click and view their recovery curves
4. Investigate why outliers differ

### Combine with Holistic Comparison (Tab 2, bottom)
1. Use estimation plot for overall parameter comparison
2. Use holistic comparison for population-weighted analysis
3. Get biological interpretation

### Export for Publications
1. Generate estimation plot
2. Hover over plot → camera icon (top right)
3. Save as high-res PNG
4. Use in papers/presentations

---

## Technical Details

### Implementation

**Function**: `FRAPPlots.plot_estimation_plot()`  
**Location**: `frap_plots.py`, lines 1119-1346  
**Dependencies**: 
- `plotly.subplots.make_subplots`
- `scipy.stats` (for t-distribution)
- `numpy` (for jittering)

### Calculation Details

**95% Confidence Interval:**
```python
# For each group
mean = data.mean()
sem = data.sem()
n = len(data)

# t-distribution critical value (two-tailed, α=0.05)
t_critical = stats.t.ppf(0.975, df=n-1)

# 95% CI
ci_95 = t_critical * sem
```

**Mean Difference CI:**
```python
# Pooled SEM for difference
sem_diff = sqrt(sem1² + sem2²)

# Degrees of freedom
df = n1 + n2 - 2

# 95% CI for difference
ci_diff = t_critical(df) * sem_diff
```

---

## Comparison with Traditional Methods

| Feature | Bar Chart | Box Plot | Violin Plot | Estimation Plot |
|---------|-----------|----------|-------------|-----------------|
| Shows raw data | ❌ | Optional | Optional | ✅ Always |
| Shows mean | ✅ | Optional | Optional | ✅ Always |
| Shows distribution | ❌ | Quartiles | Full shape | Individual points |
| Shows CI | ❌ Usually SEM | ❌ | ❌ | ✅ 95% CI |
| Effect size panel | ❌ | ❌ | ❌ | ✅ Yes |
| Sample size visible | ❌ | ❌ | ❌ | ✅ Count dots |
| Statistical inference | ❌ | ❌ | ❌ | ✅ CI vs zero |
| Publication quality | ⚠️ Outdated | ✅ | ✅ | ✅ Modern |

---

## Summary

### Key Takeaways

1. **Estimation plots show everything**: Raw data + mean + uncertainty + effect size
2. **Left panel**: Raw data with means and 95% CIs
3. **Right panel**: Differences from control with CIs
4. **Statistical significance**: Check if CI crosses zero in right panel
5. **Sample size**: Count the dots - it's visible!
6. **Use for**: 2-6 group comparisons, publication figures, transparent reporting

### When Estimation Plots Changed Your Conclusion

**Example from real data:**

**Bar Chart showed:**
```
Group 1: 75% ± 5%  }  Looks similar
Group 2: 80% ± 5%  }  p=0.06 (not sig)
```

**Estimation Plot revealed:**
```
Group 1: ○○   ○○○     ◆  (N=5, wide CI)
Group 2: ○○○○○○○○○○ ◆  (N=10, narrow CI)
```

**New insight**: Sample sizes were very different! Group 1 needs more data before concluding "no difference."

---

**Documentation Version:** 1.0  
**Last Updated:** October 19, 2025  
**Compatible With:** FRAP2025 Platform v3.0+
