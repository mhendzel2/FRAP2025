# FRAP Normalization Modes

All normalization is routed through `normalize_frap_curve(..., mode=...)`.

## `mode="simple"`

- Equation:
  - `F_norm(t) = [F(t) - F_bg] / [F_pre - F_bg]`
- Required inputs:
  - background scalar (`F_bg`)
- Validation:
  - raises `NormalizationError` if `F_pre <= F_bg`
  - raises `NormalizationError` if first post-bleach value exceeds 85% of pre-bleach

## `mode="double"`

- Equation:
  - `F_norm(t) = [F_ROI(t)/F_ref(t)] / [F_ROI_pre/F_ref_pre]`
- Required inputs:
  - `F_ref`, `F_roi_pre`, `F_ref_pre`
- Use case:
  - compensates monotonic acquisition photobleaching

## `mode="full_scale"`

- Equation:
  - `F_norm(t) = [F(t) - F_post_min] / [F_pre - F_post_min]`
- Use case:
  - scales immediate post-bleach minimum to 0 and pre-bleach mean to 1
- Validation:
  - raises `NormalizationError` if there is no post-bleach data or denominator is non-positive

## Reporting Requirement

- The selected normalization mode is persisted in every `FRAPFitResult.normalization_mode`.
