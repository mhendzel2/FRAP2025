# Sample Data

This folder contains small synthetic FRAP datasets for validation and demos.

## Files

- `synthetic_single_exp.csv`
  - Model: single exponential
  - Ground truth: `mobile_fraction = 0.78`, `tau_s = 0.40 s`
  - With `w_um = 1.0`, expected `D_um2_per_s = 1/(4*0.40) = 0.625`

- `synthetic_double_exp.csv`
  - Model: double exponential
  - Ground truth: `A1 = 0.50`, `tau1_s = 3.0 s`; `A2 = 0.28`, `tau2_s = 35.0 s`
  - Total mobile fraction: `0.78`

- `synthetic_frap_stack.tif`
  - Small synthetic image stack (20 frames, 48x48 px)
  - Includes a central bleach event and recovery trend
  - Intended for image analysis pipeline smoke tests (ROI extraction, bleach radius estimate, normalization)

## Usage

These files are designed to be committed and used in automated tests, tutorials, and CLI validation.
