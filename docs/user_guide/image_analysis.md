# Image Analysis Workflow

Use `frap2025.image.frap_image_analysis` utilities for frame-stack workflows.

## Recommended Steps

1. Optional drift correction:
   - `apply_drift_correction(frames)`
2. Measure bleach radius from pre/post frames:
   - `measure_bleach_radius_from_image(post_bleach_frame, pre_bleach_frame, pixel_size_um)`
3. Extract traces for bleach/reference/background ROIs:
   - `extract_multi_roi_timeseries(image_stack, bleach_roi_mask, reference_roi_mask, background_roi_mask)`
4. Screen for acquisition photobleaching:
   - `detect_photobleaching(pre_bleach_frames)`

## Quality Flags

- Ellipticity > 1.2: circular-model assumptions may be violated.
- Pre-bleach slope > 1%/frame: acquisition photobleaching likely significant.

## Release Notes 0.4.0 (Image Analysis)

- Added capillary-wave spectroscopy support for condensate contours via `frap2025.material_mechanics.compute_capillary_waves`.
- Added backend-switchable deformation-field computation via `frap2025.optical_flow_analysis.compute_deformation_field`:
   - `method="farneback"` (classical default)
   - `method="raft"` (deep-learning backend when optional dependencies are available)
- Added advanced optional install profile for image-biophysics tooling:
   - `pip install -e .[advanced]`

If optional advanced dependencies are missing (e.g., torch/torchvision), FRAP2025 automatically falls back to classical methods and remains fully usable.
