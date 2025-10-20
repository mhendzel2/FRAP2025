# Multi-Group Analysis UI: Visual Guide

## Overview Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         FRAP Analysis Application                        │
├─────────────────────────────────────────────────────────────────────────┤
│  Sidebar: Data Management                                                │
│  ├─ Load Files                                                          │
│  ├─ Create Groups                                                        │
│  └─ Settings                                                             │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────────┬──────────────────┬──────────────────┬──────────────┐
│   Tab 1          │   Tab 2          │   Tab 3          │   Tab 4      │
│   Individual     │   Multi-Group    │   Statistical    │   Export     │
│   Group Analysis │   Comparison     │   Comparison     │   & Reports  │
└──────────────────┴──────────────────┴──────────────────┴──────────────┘
```

---

## Tab 1: Individual Group Analysis (UNCHANGED)

**Purpose:** Deep-dive exploration of a single experimental condition

```
┌─────────────────────────────────────────────────────────────┐
│  Tab 1: Individual Group Analysis                           │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  📂 Select Group: [Dropdown ▼]                              │
│                                                               │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ Step 1: Data Quality Overview                          │  │
│  │  - Number of cells: 25                                 │  │
│  │  - Quality metrics table                               │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                               │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ Step 2: Outlier Detection                             │  │
│  │  - Statistical outlier identification                  │  │
│  │  - Manual exclusion controls                           │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                               │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ Step 3: Parameter Distributions                        │  │
│  │  - Histograms of fitted parameters                     │  │
│  │  - Mobile fraction, k_fast, k_slow                     │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                               │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ Step 4-6: Individual Cell Curves & Fitting            │  │
│  │  - Raw curves visualization                            │  │
│  │  - Exponential fitting (simple models)                 │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                               │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ Step 7: Group Average Curve                           │  │
│  │  - Mean ± SEM recovery profile                         │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                               │
└─────────────────────────────────────────────────────────────┘

🎯 Use Case: "What is the variability within my control group?"
```

---

## Tab 2: Multi-Group Comparison (REORGANIZED!)

**Purpose:** Compare recovery dynamics across experimental conditions

### NEW Layout

```
┌──────────────────────────────────────────────────────────────────────┐
│  Tab 2: Multi-Group Comparison                                       │
├──────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  🔬 Multi-Group Comparison                                           │
│  Compare recovery dynamics across conditions using mean curves       │
│                                                                        │
│  💡 Recommended Workflow:                                            │
│   - Start here for comparing treatment conditions                    │
│   - Use Individual Group Analysis (Tab 1) for cell-level analysis   │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  ▶ Why Focus on Mean Recovery Curves? [Click to expand]      │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                        │
│  ──────────────────────────────────────────────────────────────────  │
│                                                                        │
│  📋 Select Groups to Compare:                                        │
│  [☑ Control] [☑ Treatment A] [☐ Treatment B]                        │
│                                                                        │
├══════════════════════════════════════════════════════════════════════┤
│  📉 SECTION 1: MEAN RECOVERY CURVE ANALYSIS ⭐ PRIMARY              │
├══════════════════════════════════════════════════════════════════════┤
│                                                                        │
│  Mean Recovery Profiles Comparison                                   │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  1.0 ┤                                                       │    │
│  │      │     ╱────  Control (blue)                            │    │
│  │  0.8 ┤   ╱                                                   │    │
│  │      │ ╱                                                     │    │
│  │  0.6 ┤╱         ╱────  Treatment A (orange)                │    │
│  │      │       ╱                                               │    │
│  │  0.4 ┤     ╱                                                 │    │
│  │      │   ╱                                                   │    │
│  │  0.2 ┤ ╱                                                     │    │
│  │      │╱                                                      │    │
│  │  0.0 ┤───────────────────────────────────────────           │    │
│  │      0    10   20   30   40   50  Time (s)                  │    │
│  └────────────────────────────────────────────────────────────┘    │
│  (Shaded areas show ±SEM)                                           │
│                                                                        │
│  📊 Profile Similarity Metrics (2 groups):                          │
│  ┌──────────────┬──────────────┬──────────────┐                    │
│  │ Max Diff     │ Mean Diff    │ RMSD         │                    │
│  │ 0.234        │ 0.112        │ 0.145        │                    │
│  └──────────────┴──────────────┴──────────────┘                    │
│                                                                        │
│  ──────────────────────────────────────────────────────────────────  │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  ▶ 🔬 Advanced Curve Fitting (Mechanistic Models)            │  │
│  │     [Click to expand for biophysical model fitting]          │  │
│  │                                                                │  │
│  │  When expanded:                                               │  │
│  │  ┌────────────────────────────────────────────────────────┐ │  │
│  │  │ ⚙️ Configuration                                        │ │  │
│  │  │                                                          │ │  │
│  │  │ Select Model: [auto ▼]                                  │ │  │
│  │  │ Bleach Radius: [1.0] μm                                 │ │  │
│  │  │                                                          │ │  │
│  │  │ [Fit Advanced Models to Mean Curves]                    │ │  │
│  │  └────────────────────────────────────────────────────────┘ │  │
│  │                                                                │  │
│  │  After fitting:                                               │  │
│  │  ┌────────────────────────────────────────────────────────┐ │  │
│  │  │ 📈 Fitted Parameters                                    │ │  │
│  │  │                                                          │ │  │
│  │  │ Control:                    Treatment A:                │ │  │
│  │  │ - Model: reaction_diff      - Model: reaction_diff     │ │  │
│  │  │ - R²: 0.998                 - R²: 0.996                │ │  │
│  │  │ - D: 2.34 μm²/s             - D: 1.89 μm²/s            │ │  │
│  │  │ - k_on: 0.45 s⁻¹            - k_on: 0.12 s⁻¹           │ │  │
│  │  │ - k_off: 0.08 s⁻¹           - k_off: 0.15 s⁻¹          │ │  │
│  │  └────────────────────────────────────────────────────────┘ │  │
│  │                                                                │  │
│  │  ┌────────────────────────────────────────────────────────┐ │  │
│  │  │ 🔍 Parameter Fold Changes                              │ │  │
│  │  │ ┌──────────┬─────────────────────────┐                 │ │  │
│  │  │ │Parameter │ Fold Change (Trt / Ctl) │                 │ │  │
│  │  │ ├──────────┼─────────────────────────┤                 │ │  │
│  │  │ │ D        │ 0.81 (19% decrease)     │                 │ │  │
│  │  │ │ k_on     │ 0.27 (73% decrease)     │  ← Big change! │ │  │
│  │  │ │ k_off    │ 1.88 (88% increase)     │                 │ │  │
│  │  │ └──────────┴─────────────────────────┘                 │ │  │
│  │  └────────────────────────────────────────────────────────┘ │  │
│  │                                                                │  │
│  │  ┌────────────────────────────────────────────────────────┐ │  │
│  │  │ 🧬 Biological Interpretation                           │ │  │
│  │  │                                                          │ │  │
│  │  │ Treatment A shows reduced binding affinity (k_on      │ │  │
│  │  │ decreased 73%) and increased dissociation rate         │ │  │
│  │  │ (k_off increased 88%), suggesting weakened chromatin  │ │  │
│  │  │ binding. Diffusion coefficient decreased 19%,          │ │  │
│  │  │ indicating possible structural changes...              │ │  │
│  │  └────────────────────────────────────────────────────────┘ │  │
│  │                                                                │  │
│  │  ┌────────────────────────────────────────────────────────┐ │  │
│  │  │ 📊 Fitted Curves Visualization                         │ │  │
│  │  │ [Interactive Plotly plot showing data + fitted curves]│ │  │
│  │  └────────────────────────────────────────────────────────┘ │  │
│  │                                                                │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                        │
├══════════════════════════════════════════════════════════════════════┤
│  📊 SECTION 2: POPULATION-BASED ANALYSIS (Secondary)                │
├══════════════════════════════════════════════════════════════════════┤
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  ▶ What is Population-Based Analysis? [Click to expand]      │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                        │
│  Population Distribution Comparison                                  │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │ Group       │ Diffusion │ Intermediate │ Binding │ Wt k_fast│  │
│  ├─────────────┼───────────┼──────────────┼─────────┼──────────┤  │
│  │ Control     │   42.3%   │    18.5%     │  39.2%  │  0.845   │  │
│  │ Treatment A │   78.6%   │    15.2%     │   6.2%  │  1.123   │  │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                        │
│  Population Distribution Visualization                               │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │ 100% ┤███████████████████████  Diffusion                    │    │
│  │      │░░░░░░░░  Intermediate                                 │    │
│  │  75% ┤▓▓▓▓▓▓▓▓▓▓▓▓▓  Binding                                │    │
│  │      │                                                        │    │
│  │  50% ┤                                                        │    │
│  │      │                                                        │    │
│  │  25% ┤                                                        │    │
│  │      │                                                        │    │
│  │   0% ┤────────────────────────                              │    │
│  │      Control    Treatment A                                  │    │
│  └────────────────────────────────────────────────────────────┘    │
│  ← Notice shift from Binding to Diffusion!                          │
│                                                                        │
│  ──────────────────────────────────────────────────────────────────  │
│                                                                        │
│  📈 Pairwise Statistical Comparison (2 groups)                       │
│  ┌────────────────────────┬────────────────────────┐                │
│  │ Mobile Fraction        │ Population Shifts       │                │
│  │                        │                         │                │
│  │ Control:               │ Diffusion Shift:        │                │
│  │ 85.2% ± 3.1%          │ +36.3% ↑                │                │
│  │                        │                         │                │
│  │ Treatment A:           │ Binding Shift:          │                │
│  │ 88.7% ± 2.8%          │ -33.0% ↓                │                │
│  │                        │                         │                │
│  │ ✓ Significant          │ Major shift from        │                │
│  │ (p=0.0234)            │ binding to diffusion!   │                │
│  └────────────────────────┴────────────────────────┘                │
│                                                                        │
│  ──────────────────────────────────────────────────────────────────  │
│                                                                        │
│  🧬 Biological Interpretation                                        │
│  Treatment A caused a major shift in kinetic populations, with       │
│  33% of cells moving from binding to diffusion regimes. This         │
│  suggests the treatment disrupted chromatin binding capability...    │
│                                                                        │
└──────────────────────────────────────────────────────────────────────┘

🎯 Use Case: "Does Drug X affect recovery compared to control?"
```

---

## Tab 3: Multi-Group Statistical Comparison (UNCHANGED)

**Purpose:** Statistical testing across multiple groups

```
┌─────────────────────────────────────────────────────────────┐
│  Tab 3: Multi-Group Statistical Comparison                   │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  📊 Parameter Visualization                                  │
│                                                               │
│  Select Parameter: [mobile_fraction ▼]                      │
│  Visualization Type: [Estimation Plot ▼]                    │
│                                                               │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  [Interactive Estimation Plot]                         │  │
│  │  - Raw data points (swarm plot)                        │  │
│  │  - Group means ± 95% CI                                │  │
│  │  - Effect sizes                                         │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                               │
│  ──────────────────────────────────────────────────────────  │
│                                                               │
│  📊 Statistical Testing                                      │
│                                                               │
│  For 2 Groups:                 For 3+ Groups:               │
│  [Perform t-test]              [Perform ANOVA]              │
│                                                               │
│  Results:                      Results:                      │
│  P-value: 0.0023              F-statistic: 12.34            │
│  ✓ Significant (p<0.05)       P-value: 0.0001              │
│                                ✓ Significant                 │
│                                                               │
│                                Post-hoc:                     │
│                                [Tukey HSD pairwise tests]    │
│                                                               │
└─────────────────────────────────────────────────────────────┘

🎯 Use Case: "Are differences statistically significant?"
```

---

## Workflow Comparison

### OLD Workflow (Before Reorganization)

```
User opens Tab 2
    ↓
Sees "Holistic Group Comparison" (confusing name)
    ↓
Reads about population distributions (complex concept)
    ↓
Scrolls through population tables
    ↓
Finally finds averaged curves at the bottom
    ↓
No access to advanced fitting
    ↓
Goes to Tab 3 for statistics
    ↓
❌ Confused about which analysis to use
```

### NEW Workflow (After Reorganization)

```
User opens Tab 2
    ↓
Sees "Multi-Group Comparison" (clear name)
    ↓
Reads clear guidance: "Start here for comparing conditions"
    ↓
Immediately sees mean recovery curves (robust!)
    ↓
Can optionally expand advanced fitting
    ↓
Fits biophysical models in one click
    ↓
Gets parameter comparison + interpretation
    ↓
Scrolls down for population analysis (if interested)
    ↓
Goes to Tab 3 for detailed statistics (if needed)
    ↓
✅ Clear, logical workflow
```

---

## Key Visual Changes

### Before: Buried Important Features

```
Tab 2 (Old)
┌────────────────────┐
│ Population Table   │  ← Complex, shown first
│ (hard to interpret)│
├────────────────────┤
│ Statistical Tests  │
├────────────────────┤
│ ...scroll...       │
├────────────────────┤
│ Averaged Curves    │  ← Important, but at bottom!
│ (simple, useful)   │
└────────────────────┘
        ↓
❌ Users miss the best analysis method
```

### After: Prioritized Workflow

```
Tab 2 (New)
┌────────────────────┐
│ 🎯 Guidance        │  ← Educational
├────────────────────┤
│ 📉 MEAN CURVES     │  ← PRIMARY (shown first!)
│ (robust, clear)    │
├────────────────────┤
│ 🔬 Advanced Fit    │  ← Optional (in expander)
│ [Expand to see]    │
├────────────────────┤
│ 📊 Population      │  ← Secondary (still available)
│ (complementary)    │
└────────────────────┘
        ↓
✅ Users start with best practice
```

---

## Color Coding & Icons

### Section Priorities

```
⭐ PRIMARY   = Start here! (Mean recovery curves)
🔬 OPTIONAL  = Advanced users (Mechanistic fitting)
📊 SECONDARY = Complementary (Population analysis)
ℹ️ INFO      = Educational (Expandable explanations)
```

### Visual Hierarchy

```
Large Headers (##)     = Major sections
Medium Headers (###)   = Subsections
Small Headers (####)   = Details
Expanders (▶)         = Optional content
Boxes (┌─┐)           = Important content
Separators (─────)    = Visual breaks
```

---

## Mobile/Responsive Considerations

### Desktop (Wide Screen)

```
┌─────────────┬─────────────┐
│  Left Plot  │ Right Metrics│
│             │              │
│  Mean       │ Max Diff     │
│  Curves     │ RMSD         │
│  (large)    │ (compact)    │
└─────────────┴─────────────┘
```

### Tablet/Mobile (Narrow Screen)

```
┌─────────────┐
│  Mean       │
│  Curves     │
│  (full      │
│  width)     │
├─────────────┤
│ Metrics     │
│ (stacked    │
│  below)     │
└─────────────┘
```

*Note: Streamlit handles responsive layout automatically*

---

## User Journey Map

### Beginner User

```
1. Opens app → Sees Tab 1 first
2. Loads data, creates groups
3. Explores individual group (Tab 1)
4. Wants to compare → Tab 2
5. Sees mean curves immediately
6. Understands visual difference
7. Reads biological interpretation
8. Done! (Simple workflow)
```

### Advanced User

```
1. Opens app → Familiar with interface
2. Loads data, creates groups
3. Goes directly to Tab 2
4. Views mean curves
5. Expands Advanced Fitting
6. Selects sophisticated model
7. Analyzes fitted parameters
8. Checks population shifts
9. Confirms with Tab 3 statistics
10. Exports results
```

### Expert/Developer

```
1. Uses Python API directly
2. Calls compare_groups_advanced_fitting()
3. Programmatically analyzes batches
4. Generates custom reports
5. Integrates into pipelines
```

---

## Summary: Before vs. After

| Aspect | Before | After |
|--------|--------|-------|
| **First thing user sees** | Population distributions (complex) | Mean recovery curves (intuitive) |
| **Advanced fitting** | Not available in UI | Integrated with one-click access |
| **Workflow clarity** | Unclear, no guidance | Clear priorities, educational |
| **Visual hierarchy** | Flat, everything equal | Structured sections (1, 2, 3) |
| **Educational content** | Minimal, buried in expander | Prominent, helpful expandable sections |
| **User confusion** | High ("What do I use?") | Low ("Start with mean curves!") |
| **Time to insight** | Long (scroll to find curves) | Fast (curves shown first) |

---

**Conclusion:** The reorganization transforms Tab 2 from a confusing collection of analyses into a **clear, prioritized workflow** that guides users to the most robust methods while keeping advanced features accessible.

---

**Version:** 1.0  
**Last Updated:** 2025-01-XX  
**See Also:**
- `UI_REORGANIZATION_SUMMARY.md` - Technical details
- `MULTIGROUP_QUICKSTART.md` - User guide
- `IMPLEMENTATION_SUMMARY.md` - Change log
