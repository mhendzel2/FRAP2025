# FRAP2025 Installation Guide

## Quick Installation (Windows)

### Automated Setup

Run the installation script in PowerShell:

```powershell
.\install_venv.ps1
```

This will:
1. Check Python version (≥3.10 required)
2. Create virtual environment in `.\venv\`
3. Activate the environment
4. Upgrade pip
5. Install all dependencies from `requirements.txt`
6. Verify installation

**Installation time:** 5-10 minutes (first time)

---

## Manual Installation

### Prerequisites

- **Python 3.10 or higher**
- **pip** (included with Python)
- **Git** (optional, for cloning repository)

### Step 1: Create Virtual Environment

```powershell
# In PowerShell
python -m venv venv
```

### Step 2: Activate Virtual Environment

```powershell
# Windows PowerShell
.\venv\Scripts\Activate.ps1

# Windows CMD
venv\Scripts\activate.bat

# Linux/Mac
source venv/bin/activate
```

You should see `(venv)` at the start of your command prompt.

### Step 3: Upgrade pip

```powershell
python -m pip install --upgrade pip
```

### Step 4: Install Dependencies

```powershell
pip install -r requirements.txt
```

**Note:** This installs ~30 packages including:
- Core: numpy, scipy, pandas, matplotlib
- Analysis: scikit-learn, scikit-image, statsmodels
- UI: streamlit, plotly
- Reports: reportlab, jinja2
- Optional: dabest, filterpy, pymc

### Step 5: Verify Installation

```powershell
python verify_installation.py
```

This checks:
- Python version ≥3.10
- All required modules importable
- Core functionality working

---

## What Gets Installed

### Core Dependencies (Required)

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥1.24.0 | Numerical computing |
| scipy | ≥1.10.0 | Scientific computing, optimization |
| pandas | ≥2.0.0 | Data structures and I/O |
| matplotlib | ≥3.7.0 | Plotting |
| seaborn | ≥0.12.0 | Statistical visualizations |
| scikit-learn | ≥1.3.0 | Machine learning (clustering, outliers) |
| scikit-image | ≥0.21.0 | Image processing (tracking, segmentation) |
| opencv-python | ≥4.8.0 | Optical flow tracking |
| statsmodels | ≥0.14.0 | Statistical models (LMM) |
| joblib | ≥1.3.0 | Parallel processing |
| pyarrow | ≥14.0.0 | Parquet file format |

### UI Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| streamlit | ≥1.28.0 | Interactive web UI |
| plotly | ≥5.15.0 | Interactive plots |
| jinja2 | ≥3.1.0 | HTML templating |

### Report Generation

| Package | Version | Purpose |
|---------|---------|---------|
| reportlab | ≥4.0.0 | PDF generation |

### Optional Dependencies

| Package | Version | Purpose | Fallback |
|---------|---------|---------|----------|
| dabest | ≥2023.2.14 | Gardner-Altman plots | Custom implementation |
| filterpy | ≥1.4.5 | Kalman filter | NumPy implementation |
| pymc | ≥5.10.0 | Bayesian models | Not required for core |
| pytest | ≥7.4.0 | Testing framework | Manual testing OK |

---

## Troubleshooting

### Python Not Found

**Error:**
```
python : The term 'python' is not recognized...
```

**Solution:**
1. Install Python from https://www.python.org/downloads/
2. During installation, check "Add Python to PATH"
3. Restart PowerShell
4. Verify: `python --version`

### Python Version Too Old

**Error:**
```
ERROR: Python 3.10 or higher required
```

**Solution:**
1. Download Python 3.10+ from python.org
2. Install (allow it to update PATH)
3. Restart terminal
4. Verify: `python --version`

### Execution Policy Error (PowerShell)

**Error:**
```
.\install_venv.ps1 : File cannot be loaded because running scripts is disabled...
```

**Solution:**
```powershell
# Run as Administrator
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then try again
.\install_venv.ps1
```

### Package Installation Fails

**Error:**
```
ERROR: Could not find a version that satisfies the requirement...
```

**Solution:**
1. Ensure pip is up to date:
   ```powershell
   python -m pip install --upgrade pip
   ```

2. If specific package fails (e.g., pymc):
   - Check if it's optional
   - Try installing without it
   - Install dependencies manually

3. For C++ compiler errors (pymc, filterpy):
   - These are optional packages
   - Core functionality works without them
   - To fix: Install Visual Studio Build Tools

### Virtual Environment Won't Activate

**Error:**
```
Activate.ps1 is not digitally signed...
```

**Solution:**
```powershell
# Option 1: Bypass for this script
powershell -ExecutionPolicy Bypass -File .\install_venv.ps1

# Option 2: Change policy
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Import Errors After Installation

**Error:**
```
ModuleNotFoundError: No module named 'streamlit'
```

**Solution:**
1. Ensure virtual environment is activated: `.\venv\Scripts\Activate.ps1`
2. Check you're using the right Python: `python --version`
3. Reinstall in venv: `pip install streamlit`

### Disk Space Issues

**Error:**
```
No space left on device
```

**Solution:**
- Installation requires ~2-3 GB
- Clear temporary files
- Use different drive
- Install fewer optional packages

---

## Post-Installation

### Quick Start

```powershell
# 1. Activate virtual environment (if not already active)
.\venv\Scripts\Activate.ps1

# 2. Generate test data
python quick_start_singlecell.py

# 3. Launch UI
python launch_ui.py
# OR
streamlit run streamlit_singlecell.py
```

### Run Tests

```powershell
# Run all tests
pytest test_frap_singlecell.py

# Run specific test
pytest test_frap_singlecell.py::test_acceptance_criteria

# Run with verbose output
pytest test_frap_singlecell.py -v
```

### Generate Reports

```powershell
# Test report generation
python frap_singlecell_reports.py

# This creates:
#   test_report.pdf
#   test_report.html
```

### Using in Scripts

```python
# Your analysis script
from frap_singlecell_api import analyze_frap_movie
import tifffile
import numpy as np

# Load movie
movie = tifffile.imread('my_movie.tif')
time_points = np.arange(len(movie)) * 0.5  # seconds

# Analyze
traces, features = analyze_frap_movie(
    movie, time_points,
    exp_id='exp001',
    movie_id='movie01',
    condition='control',
    output_dir='./output'
)

# Results saved to:
#   ./output/roi_traces.parquet
#   ./output/cell_features.parquet
```

---

## Virtual Environment Management

### Activating

**Every time** you start a new terminal session:

```powershell
# PowerShell
.\venv\Scripts\Activate.ps1

# CMD
venv\Scripts\activate.bat

# Linux/Mac
source venv/bin/activate
```

You'll see `(venv)` in your prompt when active.

### Deactivating

```powershell
deactivate
```

### Recreating

If your venv gets corrupted:

```powershell
# Remove old venv
Remove-Item -Recurse -Force venv

# Run installer again
.\install_venv.ps1
```

### Updating Dependencies

When `requirements.txt` changes:

```powershell
# Activate venv first
.\venv\Scripts\Activate.ps1

# Update packages
pip install --upgrade -r requirements.txt
```

---

## Directory Structure After Installation

```
FRAP2025/
├── venv/                           # Virtual environment (created)
│   ├── Scripts/                    # Executables
│   │   ├── python.exe             # Python interpreter
│   │   ├── Activate.ps1           # Activation script
│   │   └── ...
│   └── Lib/                        # Installed packages
│       └── site-packages/
│           ├── numpy/
│           ├── scipy/
│           ├── streamlit/
│           └── ...
├── requirements.txt                # Package list
├── install_venv.ps1               # Installation script
├── verify_installation.py         # Verification script
├── launch_ui.py                   # UI launcher
├── quick_start_singlecell.py      # Example generator
├── frap_singlecell_api.py         # Main API
├── frap_singlecell_reports.py     # Report generation
├── streamlit_singlecell.py        # UI application
├── frap_*.py                      # Analysis modules
├── test_*.py                      # Test files
└── *.md                           # Documentation
```

---

## System Requirements

### Minimum

- **OS:** Windows 10, macOS 10.14, or Linux (Ubuntu 18.04+)
- **CPU:** 2 cores
- **RAM:** 4 GB
- **Disk:** 5 GB free space
- **Python:** 3.10 or higher

### Recommended

- **OS:** Windows 11, macOS 12+, or Linux (Ubuntu 22.04+)
- **CPU:** 4+ cores (for parallel processing)
- **RAM:** 8 GB (for large datasets)
- **Disk:** 10 GB free space
- **Python:** 3.11 or 3.12

### For Large Datasets (>1000 cells)

- **RAM:** 16 GB
- **CPU:** 8+ cores
- **Disk:** SSD recommended

---

## Uninstallation

### Remove Virtual Environment

```powershell
# Deactivate if active
deactivate

# Delete venv folder
Remove-Item -Recurse -Force venv
```

### Complete Removal

```powershell
# Remove entire project
cd ..
Remove-Item -Recurse -Force FRAP2025
```

### Keep Code, Remove Dependencies

```powershell
# Just delete venv
Remove-Item -Recurse -Force venv

# To reinstall later
.\install_venv.ps1
```

---

## Getting Help

### Documentation

- **README_UI.md** - UI quick start
- **UI_GUIDE.md** - Comprehensive UI guide
- **README_SINGLECELL.md** - API reference
- **SINGLECELL_IMPLEMENTATION.md** - Technical details
- **UI_IMPLEMENTATION_SUMMARY.md** - Features and status

### Common Issues

1. **"Module not found" errors** → Activate venv
2. **"Python not recognized"** → Add to PATH
3. **"Permission denied"** → Run as Administrator
4. **"Package conflicts"** → Recreate venv
5. **"Out of memory"** → Reduce dataset size or use 64-bit Python

### Support

- Check documentation files
- Run `python verify_installation.py`
- Check error logs in console
- Review `requirements.txt` for package versions

---

## Next Steps

After successful installation:

1. ✅ **Verify:** `python verify_installation.py`
2. 📊 **Generate Examples:** `python quick_start_singlecell.py`
3. 🚀 **Launch UI:** `python launch_ui.py`
4. 📖 **Read Guides:** Start with `README_UI.md`
5. 🧪 **Run Tests:** `pytest test_frap_singlecell.py`
6. 🔬 **Analyze Data:** See `README_SINGLECELL.md` for API usage

---

*Last updated: 2025-10-04*
*Installation script version: 1.0*
