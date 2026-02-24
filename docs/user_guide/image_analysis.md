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
