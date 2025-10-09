# FRAP2025 Installation Script for Windows PowerShell
# Creates virtual environment and installs all dependencies

Write-Host "=" -NoNewline
Write-Host ("=" * 59)
Write-Host "FRAP Single-Cell Analysis - Installation Script"
Write-Host "=" -NoNewline
Write-Host ("=" * 59)
Write-Host ""

# Check Python version
Write-Host "Step 1/5: Checking Python version..." -ForegroundColor Cyan
try {
    $pythonVersion = python --version 2>&1
    Write-Host "  Found: $pythonVersion" -ForegroundColor Green
    
    # Extract version numbers
    if ($pythonVersion -match "Python (\d+)\.(\d+)") {
        $major = [int]$matches[1]
        $minor = [int]$matches[2]
        
        if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 10)) {
            Write-Host "  ERROR: Python 3.10 or higher required" -ForegroundColor Red
            Write-Host "  Please install from: https://www.python.org/downloads/" -ForegroundColor Yellow
            exit 1
        }
    }
} catch {
    Write-Host "  ERROR: Python not found in PATH" -ForegroundColor Red
    Write-Host "  Please install from: https://www.python.org/downloads/" -ForegroundColor Yellow
    exit 1
}

Write-Host ""

# Check if venv already exists
Write-Host "Step 2/5: Setting up virtual environment..." -ForegroundColor Cyan
if (Test-Path "venv") {
    Write-Host "  Virtual environment 'venv' already exists" -ForegroundColor Yellow
    $response = Read-Host "  Delete and recreate? (y/N)"
    if ($response -eq "y" -or $response -eq "Y") {
        Write-Host "  Removing existing venv..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force venv
    } else {
        Write-Host "  Using existing venv" -ForegroundColor Green
        Write-Host ""
        Write-Host "Step 3/5: Activating virtual environment..." -ForegroundColor Cyan
        & .\venv\Scripts\Activate.ps1
        Write-Host "  Activated!" -ForegroundColor Green
        Write-Host ""
        
        Write-Host "Step 4/5: Upgrading pip..." -ForegroundColor Cyan
        python -m pip install --upgrade pip | Out-Null
        Write-Host "  Done!" -ForegroundColor Green
        Write-Host ""
        
        Write-Host "Step 5/5: Installing/updating dependencies..." -ForegroundColor Cyan
        Write-Host "  This may take several minutes..." -ForegroundColor Yellow
        python -m pip install -r requirements.txt
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  All packages installed successfully!" -ForegroundColor Green
        } else {
            Write-Host "  Some packages failed to install" -ForegroundColor Red
            Write-Host "  Check errors above for details" -ForegroundColor Yellow
        }
        
        Write-Host ""
        Write-Host "=" -NoNewline
        Write-Host ("=" * 59)
        Write-Host "Installation Complete!" -ForegroundColor Green
        Write-Host "=" -NoNewline
        Write-Host ("=" * 59)
        Write-Host ""
        Write-Host "Virtual environment is activated and ready to use." -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Quick Start:" -ForegroundColor Yellow
        Write-Host "  1. Generate test data:    python quick_start_singlecell.py"
        Write-Host "  2. Launch UI:             python launch_ui.py"
        Write-Host "  3. Run tests:             pytest test_frap_singlecell.py"
        Write-Host ""
        Write-Host "To activate venv in new sessions:" -ForegroundColor Yellow
        Write-Host "  .\venv\Scripts\Activate.ps1"
        Write-Host ""
        Write-Host "To deactivate:" -ForegroundColor Yellow
        Write-Host "  deactivate"
        Write-Host ""
        exit 0
    }
}

# Create new virtual environment
Write-Host "  Creating virtual environment..." -ForegroundColor Yellow
python -m venv venv

if ($LASTEXITCODE -ne 0) {
    Write-Host "  ERROR: Failed to create virtual environment" -ForegroundColor Red
    exit 1
}

Write-Host "  Created: venv\" -ForegroundColor Green
Write-Host ""

# Activate virtual environment
Write-Host "Step 3/5: Activating virtual environment..." -ForegroundColor Cyan
& .\venv\Scripts\Activate.ps1

if ($LASTEXITCODE -ne 0) {
    Write-Host "  ERROR: Failed to activate virtual environment" -ForegroundColor Red
    Write-Host "  Try manually: .\venv\Scripts\Activate.ps1" -ForegroundColor Yellow
    exit 1
}

Write-Host "  Activated!" -ForegroundColor Green
Write-Host ""

# Upgrade pip
Write-Host "Step 4/5: Upgrading pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip | Out-Null
Write-Host "  Done!" -ForegroundColor Green
Write-Host ""

# Install dependencies
Write-Host "Step 5/5: Installing dependencies..." -ForegroundColor Cyan
Write-Host "  This may take 5-10 minutes on first install..." -ForegroundColor Yellow
Write-Host ""

$startTime = Get-Date

python -m pip install -r requirements.txt

$endTime = Get-Date
$duration = ($endTime - $startTime).TotalSeconds

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "  All packages installed successfully!" -ForegroundColor Green
    Write-Host "  Installation took $([math]::Round($duration, 1)) seconds" -ForegroundColor Cyan
} else {
    Write-Host ""
    Write-Host "  WARNING: Some packages failed to install" -ForegroundColor Red
    Write-Host "  Check errors above for details" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Common issues:" -ForegroundColor Yellow
    Write-Host "  - pymc requires C++ compiler (install Visual Studio Build Tools)"
    Write-Host "  - filterpy is optional (will use fallback implementation)"
    Write-Host "  - dabest is optional (will use custom plots)"
    Write-Host ""
}

Write-Host ""

# Verify installation
Write-Host "Verifying installation..." -ForegroundColor Cyan
python verify_installation.py

Write-Host ""
Write-Host "=" -NoNewline
Write-Host ("=" * 59)
Write-Host "Installation Complete!" -ForegroundColor Green
Write-Host "=" -NoNewline
Write-Host ("=" * 59)
Write-Host ""

Write-Host "Virtual environment created at: " -NoNewline -ForegroundColor Cyan
Write-Host "$(Get-Location)\venv" -ForegroundColor White
Write-Host ""

Write-Host "Quick Start Guide:" -ForegroundColor Yellow
Write-Host "  1. Generate test data:" -ForegroundColor White
Write-Host "     python quick_start_singlecell.py" -ForegroundColor Gray
Write-Host ""
Write-Host "  2. Launch interactive UI:" -ForegroundColor White
Write-Host "     python launch_ui.py" -ForegroundColor Gray
Write-Host "     OR" -ForegroundColor White
Write-Host "     streamlit run streamlit_singlecell.py" -ForegroundColor Gray
Write-Host ""
Write-Host "  3. Run unit tests:" -ForegroundColor White
Write-Host "     pytest test_frap_singlecell.py" -ForegroundColor Gray
Write-Host ""
Write-Host "  4. Generate example report:" -ForegroundColor White
Write-Host "     python frap_singlecell_reports.py" -ForegroundColor Gray
Write-Host ""

Write-Host "Documentation:" -ForegroundColor Yellow
Write-Host "  - README_UI.md                    UI quick start"
Write-Host "  - UI_GUIDE.md                     Comprehensive UI guide"
Write-Host "  - README_SINGLECELL.md            API documentation"
Write-Host "  - SINGLECELL_IMPLEMENTATION.md    Technical details"
Write-Host ""

Write-Host "For new PowerShell sessions:" -ForegroundColor Yellow
Write-Host "  Activate venv:   " -NoNewline -ForegroundColor White
Write-Host ".\venv\Scripts\Activate.ps1" -ForegroundColor Gray
Write-Host "  Deactivate:      " -NoNewline -ForegroundColor White
Write-Host "deactivate" -ForegroundColor Gray
Write-Host ""

Write-Host "Enjoy analyzing FRAP data!" -ForegroundColor Green
Write-Host ""
