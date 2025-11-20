# ==============================================================================
# FRAP Analysis Platform - Windows PowerShell Launcher
# ==============================================================================
# This script activates the virtual environment and launches the FRAP Analysis UI
# Date: October 22, 2025
# ==============================================================================

Write-Host ""
Write-Host "=" -NoNewline
Write-Host ("=" * 78)
Write-Host "            FRAP Analysis Platform - Launcher"
Write-Host "=" -NoNewline
Write-Host ("=" * 78)
Write-Host ""

# Check if virtual environment exists
if (Test-Path "venv\Scripts\python.exe") {
    Write-Host "[INFO] Activating virtual environment..." -ForegroundColor Cyan
    & .\venv\Scripts\Activate.ps1
    
    if ($LASTEXITCODE -eq 0 -or $?) {
        Write-Host "[INFO] Virtual environment activated" -ForegroundColor Green
    } else {
        Write-Host "[WARNING] Failed to activate virtual environment, using system Python" -ForegroundColor Yellow
    }
    Write-Host ""
} else {
    Write-Host "[WARNING] Virtual environment not found, using system Python" -ForegroundColor Yellow
    Write-Host "[INFO] To create a virtual environment, run: .\install_venv.ps1" -ForegroundColor Cyan
    Write-Host ""
}

# Check Python version
Write-Host "[INFO] Python environment:" -ForegroundColor Cyan
python --version
Write-Host ""

# Check if we're in the correct directory
if (-not (Test-Path "app.py")) {
    Write-Host "[ERROR] Cannot find app.py" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please run this script from the FRAP2025 directory" -ForegroundColor Yellow
    Write-Host "Current directory: $(Get-Location)" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "[INFO] Checking required files..." -ForegroundColor Cyan
if (Test-Path "app.py") { Write-Host "  [OK] app.py" -ForegroundColor Green }
if (Test-Path "frap_singlecell_api.py") { Write-Host "  [OK] frap_singlecell_api.py" -ForegroundColor Green }
if (Test-Path "frap_data_model.py") { Write-Host "  [OK] frap_data_model.py" -ForegroundColor Green }
Write-Host ""

# Check if Streamlit is installed
python -c "import streamlit" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[WARNING] Streamlit is not installed in the current environment" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "[INFO] Installing required dependencies from requirements.txt..." -ForegroundColor Cyan
    python -m pip install -r requirements.txt
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to install dependencies" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
    Write-Host "[INFO] Dependencies installed successfully!" -ForegroundColor Green
    Write-Host ""
}

# Create output directory if it doesn't exist
if (-not (Test-Path "output")) {
    New-Item -ItemType Directory -Path "output" | Out-Null
}

Write-Host "=" -NoNewline
Write-Host ("=" * 78)
Write-Host "                       Launching FRAP Analysis UI"
Write-Host "=" -NoNewline
Write-Host ("=" * 78)
Write-Host ""
Write-Host "[INFO] Starting Streamlit application..." -ForegroundColor Cyan
Write-Host "[INFO] The UI will open in your default web browser" -ForegroundColor Cyan
Write-Host "[INFO] Default URL: http://localhost:8501" -ForegroundColor Cyan
Write-Host ""
Write-Host "[TIP] To stop the application, press Ctrl+C in this window" -ForegroundColor Yellow
Write-Host ""
Write-Host "=" -NoNewline
Write-Host ("=" * 78)
Write-Host ""

# Launch Streamlit with optimal settings
streamlit run app.py `
    --server.port=8501 `
    --server.address=localhost `
    --browser.gatherUsageStats=false `
    --theme.base=light

# If streamlit command failed
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[ERROR] Failed to launch Streamlit" -ForegroundColor Red
    Write-Host ""
    Write-Host "Troubleshooting:" -ForegroundColor Yellow
    Write-Host "1. Make sure all dependencies are installed: pip install -r requirements.txt"
    Write-Host "2. Check that no other application is using port 8501"
    Write-Host "3. Try running: python -m streamlit run app.py"
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "[INFO] Streamlit application closed" -ForegroundColor Cyan
Read-Host "Press Enter to exit"
