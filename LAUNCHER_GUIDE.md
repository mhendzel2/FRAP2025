# FRAP Analysis Platform - Launcher Guide

## Overview

The FRAP Analysis Platform now includes multiple Windows batch file launchers to make it easy to start the various analysis interfaces.

**Date Created:** October 15, 2025

---

## Available Launchers

### 1. `start.bat` - Single-Cell Analysis (Recommended)
**Quick Launch:** Double-click `start.bat`

Launches the **Single-Cell Analysis UI** - the newest and most comprehensive interface for FRAP single-cell analysis.

- **Port:** 8501
- **URL:** http://localhost:8501
- **Features:**
  - Single-cell FRAP analysis
  - Batch processing of multiple files
  - FDR correction for multiple comparisons
  - Advanced statistical visualizations (volcano plots, forest plots, etc.)
  - Linear Mixed Models (LMM) analysis
  - Interactive data exploration

---

### 2. `start_classic.bat` - Classic FRAP Analysis
**Quick Launch:** Double-click `start_classic.bat`

Launches the **Classic FRAP Analysis UI** - the original FRAP analysis interface.

- **Port:** 8502
- **URL:** http://localhost:8502
- **Features:**
  - Traditional FRAP curve analysis
  - Ensemble analysis
  - Time-series visualization
  - Basic statistical analysis

---

### 3. `start_microirradiation.bat` - Microirradiation Analysis
**Quick Launch:** Double-click `start_microirradiation.bat`

Launches the **Microirradiation Analysis UI** - specialized interface for microirradiation experiments.

- **Port:** 8503
- **URL:** http://localhost:8503
- **Features:**
  - Microirradiation-specific analysis
  - Damage response tracking
  - Recruitment kinetics

---

### 4. `start_menu.bat` - Interactive Menu (All Options)
**Quick Launch:** Double-click `start_menu.bat`

Provides an **interactive menu** to choose which interface to launch.

**Menu Options:**
1. Single-Cell Analysis UI (Port 8501)
2. Classic FRAP Analysis UI (Port 8502)
3. Microirradiation Analysis UI (Port 8503)
4. Launch UI (Python-based launcher)
5. Run Verification Tests
6. Exit

This is ideal if you frequently switch between different interfaces.

---

## What Each Launcher Does

### Automatic Checks
All launchers perform these validation checks:
- ✅ **Python Installation:** Verifies Python is installed and accessible
- ✅ **Directory Verification:** Ensures you're in the correct FRAP2025 directory
- ✅ **File Verification:** Confirms required Python files exist
- ✅ **Dependency Check:** Verifies Streamlit and other dependencies are installed

### Smart Installation
If dependencies are missing, the launchers will:
1. Detect the missing packages
2. Prompt you to install them automatically
3. Run `pip install -r requirements.txt` with your permission
4. Proceed with launching the UI

### Error Handling
Each launcher provides:
- Clear error messages if something goes wrong
- Troubleshooting tips for common issues
- Graceful exit with informative feedback

---

## Port Assignments

To allow running multiple interfaces simultaneously, each uses a different port:

| Interface | Port | URL |
|-----------|------|-----|
| Single-Cell Analysis | 8501 | http://localhost:8501 |
| Classic FRAP Analysis | 8502 | http://localhost:8502 |
| Microirradiation Analysis | 8503 | http://localhost:8503 |

**Tip:** You can run all three interfaces at the same time in separate terminal windows!

---

## Usage Instructions

### Method 1: Double-Click (Easiest)
1. Navigate to the `FRAP2025` folder in Windows Explorer
2. Double-click the desired `.bat` file
3. Follow any prompts (e.g., to install dependencies)
4. The UI will open automatically in your browser

### Method 2: Command Line
1. Open PowerShell or Command Prompt
2. Navigate to the FRAP2025 directory:
   ```cmd
   cd C:\Users\mjhen\Github\FRAP2025
   ```
3. Run the desired launcher:
   ```cmd
   start.bat
   ```

### Method 3: Right-Click Menu
1. Right-click the `.bat` file in Windows Explorer
2. Select "Run as administrator" (if needed for permissions)

---

## Stopping the Application

To stop a running Streamlit application:
- Press **Ctrl+C** in the terminal window
- Or simply close the terminal window

The browser tab can be closed independently.

---

## Troubleshooting

### Issue: "Python is not installed or not in PATH"
**Solution:**
1. Install Python 3.11+ from https://www.python.org/
2. During installation, check "Add Python to PATH"
3. Restart your terminal and try again

### Issue: "Cannot find [filename].py"
**Solution:**
1. Make sure you're in the FRAP2025 directory
2. Run the launcher from the correct location
3. Check that the repository is complete (not partially downloaded)

### Issue: "Streamlit is not installed"
**Solution:**
1. When prompted, choose "Y" to auto-install dependencies
2. Or manually run: `pip install -r requirements.txt`
3. If installation fails, try: `python -m pip install --upgrade pip`

### Issue: "Port already in use"
**Solution:**
1. Another instance is already running on that port
2. Close the other instance first
3. Or use a different launcher (different port)
4. Or manually specify a different port in the `.bat` file

### Issue: "Failed to launch Streamlit"
**Solution:**
1. Verify all dependencies are installed: `pip install -r requirements.txt`
2. Try the alternative command: `python -m streamlit run streamlit_singlecell.py`
3. Check the terminal output for specific error messages

---

## Advanced Usage

### Running Multiple Interfaces Simultaneously
You can run multiple interfaces at the same time:

1. Open 3 separate PowerShell windows
2. In Window 1: Run `start.bat` (Single-Cell on port 8501)
3. In Window 2: Run `start_classic.bat` (Classic on port 8502)
4. In Window 3: Run `start_microirradiation.bat` (Microirradiation on port 8503)

All three UIs will be accessible in your browser simultaneously!

### Customizing Launch Settings
You can edit any `.bat` file to customize settings:

**Change Port:**
```batch
--server.port=8501
```

**Change Theme:**
```batch
--theme.base=dark
```

**Disable Auto-Browser:**
```batch
--server.headless=true
```

---

## Files Overview

| File | Purpose | Entry Point |
|------|---------|-------------|
| `start.bat` | Single-Cell UI | `streamlit_singlecell.py` |
| `start_classic.bat` | Classic FRAP UI | `streamlit_frap_final.py` |
| `start_microirradiation.bat` | Microirradiation UI | `streamlit_microirradiation.py` |
| `start_menu.bat` | Interactive menu | Multiple options |

---

## Quick Reference

### Recommended Workflow
1. **First Time:** Run `start_menu.bat` → Choose option 5 to verify installation
2. **Daily Use:** Use `start.bat` for single-cell analysis (most common)
3. **Multiple Analyses:** Run all three launchers in separate windows

### Keyboard Shortcuts
- **Ctrl+C** - Stop Streamlit application
- **Alt+F4** - Close terminal window
- **Ctrl+Shift+R** - Reload browser page (refresh UI)

---

## Questions or Issues?

If you encounter problems not covered in this guide:
1. Check that Python 3.11+ is installed
2. Verify all requirements are installed: `pip install -r requirements.txt`
3. Look at the terminal output for error messages
4. Consult the main README.md for additional troubleshooting

---

**Last Updated:** October 15, 2025
