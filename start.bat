@echo off
REM ==============================================================================
REM FRAP Analysis Platform - Windows Launcher
REM ==============================================================================
REM This batch file launches the FRAP Single-Cell Analysis Streamlit UI
REM Date: October 15, 2025
REM ==============================================================================

echo.
echo ================================================================================
echo             FRAP Single-Cell Analysis Platform - Launcher
echo ================================================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH
    echo.
    echo Please install Python 3.11+ from https://www.python.org/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo [INFO] Python found: 
python --version
echo.

REM Check if we're in the correct directory
if not exist "streamlit_singlecell.py" (
    echo [ERROR] Cannot find streamlit_singlecell.py
    echo.
    echo Please run this script from the FRAP2025 directory
    echo Current directory: %CD%
    pause
    exit /b 1
)

echo [INFO] Checking required files...
if exist "streamlit_singlecell.py" echo   [OK] streamlit_singlecell.py
if exist "frap_singlecell_api.py" echo   [OK] frap_singlecell_api.py
if exist "frap_data_model.py" echo   [OK] frap_data_model.py
echo.

REM Check if Streamlit is installed
python -c "import streamlit" 2>nul
if %errorlevel% neq 0 (
    echo [WARNING] Streamlit is not installed
    echo.
    echo Would you like to install required dependencies now? (Y/N)
    set /p install_deps="Enter choice: "
    
    if /i "%install_deps%"=="Y" (
        echo.
        echo [INFO] Installing dependencies from requirements.txt...
        pip install -r requirements.txt
        if %errorlevel% neq 0 (
            echo [ERROR] Failed to install dependencies
            pause
            exit /b 1
        )
        echo [INFO] Dependencies installed successfully!
        echo.
    ) else (
        echo [ERROR] Cannot launch without Streamlit installed
        echo Please run: pip install -r requirements.txt
        pause
        exit /b 1
    )
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
streamlit run streamlit_singlecell.py ^
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
    echo 1. Make sure all dependencies are installed: pip install -r requirements.txt
    echo 2. Check that no other application is using port 8501
    echo 3. Try running: python -m streamlit run streamlit_singlecell.py
    echo.
    pause
    exit /b 1
)

echo.
echo [INFO] Streamlit application closed
pause
