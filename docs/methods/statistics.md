# Statistical Comparison Strategy

Implemented in `frap2025/stats/frap_group_stats.py`.

## Automatic Test Selection

`choose_test_automatically(groups)` chooses:

- Two groups:
  - unpaired t-test if all groups pass Shapiro-Wilk (`p > 0.05`) and `N >= 10`
  - Mann-Whitney U otherwise
- Three or more groups:
  - one-way ANOVA if all groups are approximately normal and `N >= 10`
  - Kruskal-Wallis otherwise

If any group has `N < 5`, a warning is emitted for low statistical power.

## Multiple-Testing Correction

`apply_multiple_testing_correction(..., method=...)` supports:

- `none`
- `bonferroni`
- `fdr_bh` (Benjamini-Hochberg step-up)

Adjusted p-values are capped at 1.0.

## Effect Sizes

Pairwise outputs include:

- Cohen's d for parametric two-group tests
- Rank-biserial correlation for Mann-Whitney U

For multi-group contexts, eta-squared is reported from omnibus ANOVA metadata.
