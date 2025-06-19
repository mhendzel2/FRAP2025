# FRAP Analysis Platform - Corrected Version

## Critical Mathematical Fixes Applied

### 1. Diffusion Coefficient Calculation
- **CORRECTED**: D = (w² × k) / 4
- **REMOVED**: Erroneous np.log(2) factor that was mathematically incorrect
- **CONSISTENCY**: All modules now use the same verified formula

### 2. Centralized Kinetics Interpretation
- Mathematical consistency across all modules
- Verified against published FRAP equations
- Dual interpretation: diffusion vs binding processes

## Installation Instructions

### Requirements
- Python 3.11 or higher
- pip package manager

### Quick Start
```bash
# Install dependencies
pip install streamlit pandas numpy scipy matplotlib plotly seaborn scikit-image opencv-python tifffile reportlab xlsxwriter openpyxl scikit-learn sqlalchemy psycopg2-binary

# Run the application
streamlit run streamlit_frap_final.py --server.port 5000
```

## Key Features

### Advanced Analysis
- Multi-component exponential fitting (1, 2, and 3 components)
- Model selection using AIC, BIC, and adjusted R²
- Comprehensive statistical analysis and outlier detection

### Image Processing
- TIFF stack analysis with automated bleach detection
- PSF calibration and ROI tracking
- Complete pipeline from raw microscopy to kinetic parameters

### Professional Output
- Automated PDF report generation
- Excel export with detailed statistics
- Markdown reports for documentation

### Mathematical Verification
- Corrected diffusion coefficient formula: D = (w² × k) / 4
- Proper molecular weight estimation using Stokes-Einstein relation
- Verified against published FRAP literature

## Usage Example

1. Upload your FRAP data files (XLS, XLSX, CSV supported)
2. Organize files into experimental groups
3. Configure analysis parameters (bleach radius, model selection criteria)
4. Review automated outlier detection
5. Generate comprehensive reports

## File Structure

- `streamlit_frap_final.py` - Main application interface
- `frap_core.py` - Core mathematical functions with corrected formulas
- `frap_pdf_reports.py` - Professional report generation
- `frap_image_analysis.py` - Image processing capabilities
- `sample_data/` - Example FRAP data files

## Scientific Accuracy

This version addresses critical mathematical errors identified in previous releases:
- Diffusion coefficient calculation now mathematically correct
- All modules use consistent kinetics interpretation
- Verified against peer-reviewed FRAP analysis publications

## Contact & Support

For technical issues or scientific questions about FRAP analysis,
refer to the comprehensive documentation included with the application.

---
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
