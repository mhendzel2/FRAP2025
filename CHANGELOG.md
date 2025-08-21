# Changelog - FRAP Analysis Platform

## Version: Corrected Release

### Critical Fixes
- **FIXED**: Diffusion coefficient calculation now uses correct formula D = (w² × k) / 4
- **REMOVED**: Erroneous np.log(2) factor from diffusion calculations
- **UNIFIED**: All modules now use centralized kinetics interpretation function
- **VERIFIED**: Mathematical accuracy against published literature

### Dependency Updates
- Added missing dependencies: scikit-learn, sqlalchemy, psycopg2-binary
- Updated requirements.txt with specific version ranges
- Fixed installation scripts for cross-platform compatibility

### Code Quality Improvements
- Eliminated code duplication in kinetics interpretation
- Centralized mathematical functions in frap_core_corrected.py.py
- Enhanced error handling and validation
- Improved documentation and comments

### Scientific Accuracy
- Diffusion coefficient calculations now publication-ready
- Molecular weight estimations corrected
- Kinetic interpretations mathematically consistent
- All formulas verified against FRAP literature

### Files Modified
- streamlit_frap_final.py: Updated kinetics interpretation
- frap_core_corrected.py.py: Added centralized kinetics function
- frap_pdf_reports.py: Uses centralized kinetics function
- All installation scripts: Updated dependencies

### Testing
- Mathematical formulas verified
- Cross-platform installation tested
- Sample data analysis validated
- Report generation confirmed

### Mobile/Immobile Fraction Correction (Unreleased)
- Corrected definitions: mobile fraction now equals plateau (final normalized recovery) × 100; immobile fraction = 100 − mobile.
- Added `plateau_reached` flag; if late-curve slope > 0.01 (still recovering), both fractions suppressed (NaN) and UI shows warning.
- Updated PDF/HTML reporting and Streamlit single-file metrics to reflect new logic.

---
Release Date: 2025-08-20 (auto-generated earlier; update on cut release)
