@echo off
REM ==============================================================================
REM FRAP Analysis Platform - Windows Launcher
REM ==============================================================================
REM This batch file launches the FRAP Analysis Streamlit UI
REM Date: October 22, 2025
REM ==============================================================================

echo.
echo ================================================================================
echo             FRAP Analysis Platform - Launcher
echo ================================================================================
echo.

REM Set Python executable path
if exist "venv\Scripts\python.exe" (
    echo [INFO] Using virtual environment Python
    set PYTHON_EXE=venv\Scripts\python.exe
    set STREAMLIT_CMD=venv\Scripts\streamlit.exe
    echo.
) else (
    echo [WARNING] Virtual environment not found, using system Python
    echo [INFO] To create a virtual environment, run: install_venv.ps1
    set PYTHON_EXE=python
    set STREAMLIT_CMD=streamlit
    echo.
)

REM Check Python version
echo [INFO] Python environment:
%PYTHON_EXE% --version
echo.

REM Check if we're in the correct directory
if not exist "streamlit_frap_final_restored.py" (
    echo [ERROR] Cannot find streamlit_frap_final_restored.py
    echo.
    echo Please run this script from the FRAP2025 directory
    echo Current directory: %CD%
    pause
    exit /b 1
)

echo [INFO] Checking required files...
if exist "streamlit_frap_final_restored.py" echo   [OK] streamlit_frap_final_restored.py
if exist "frap_singlecell_api.py" echo   [OK] frap_singlecell_api.py
if exist "frap_data_model.py" echo   [OK] frap_data_model.py
echo.

REM Check if Streamlit is installed
%PYTHON_EXE% -c "import streamlit" 2>nul
if %errorlevel% neq 0 (
    echo [WARNING] Streamlit is not installed in the current environment
    echo.
    echo [INFO] Installing required dependencies from requirements.txt...
    %PYTHON_EXE% -m pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to install dependencies
        pause
        exit /b 1
    )
    echo [INFO] Dependencies installed successfully!
    echo.
)

REM Create output directory if it doesn't exist
if not exist "output" mkdir output

echo ================================================================================
echo                        Launching FRAP Analysis UI
echo ================================================================================
echo.
echo [INFO] Starting Streamlit application...
echo [INFO] The UI will open in your default web browser
echo [INFO] Default URL: http://localhost:8501
echo.
echo [TIP] To stop the application, press Ctrl+C in this window
echo.
echo ================================================================================
echo.

REM Launch Streamlit with optimal settings
%STREAMLIT_CMD% run streamlit_frap_final_restored.py ^
    --server.port=8501 ^
    --server.address=localhost ^
    --browser.gatherUsageStats=false ^
    --theme.base=light

REM If streamlit command failed
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Failed to launch Streamlit
    echo.
    echo Troubleshooting:
    echo 1. Make sure all dependencies are installed: %PYTHON_EXE% -m pip install -r requirements.txt
    echo 2. Check that no other application is using port 8501
    echo 3. Try running: %PYTHON_EXE% -m streamlit run streamlit_frap_final_restored.py
    echo.
    pause
    exit /b 1
)

echo.
echo [INFO] Streamlit application closed
pause
