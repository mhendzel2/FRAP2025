@echo off
REM ==============================================================================
REM FRAP Analysis Platform - Classic FRAP UI Launcher
REM ==============================================================================
REM This batch file launches the Classic FRAP Analysis Streamlit UI
REM Date: October 15, 2025
REM ==============================================================================

echo.
echo ================================================================================
echo             Classic FRAP Analysis UI - Launcher
echo ================================================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH
    echo.
    echo Please install Python 3.11+ from https://www.python.org/
    pause
    exit /b 1
)

echo [INFO] Python found: 
python --version
echo.

REM Check if we're in the correct directory
if not exist "streamlit_frap_final.py" (
    echo [ERROR] Cannot find streamlit_frap_final.py
    echo.
    echo Please run this script from the FRAP2025 directory
    pause
    exit /b 1
)

echo [INFO] Checking required files...
if exist "streamlit_frap_final.py" echo   [OK] streamlit_frap_final.py
if exist "frap_core.py" echo   [OK] frap_core.py
if exist "frap_data_model.py" echo   [OK] frap_data_model.py
echo.

REM Check if Streamlit is installed
python -c "import streamlit" 2>nul
if %errorlevel% neq 0 (
    echo [WARNING] Streamlit is not installed
    echo.
    set /p install_deps="Would you like to install dependencies now? (Y/N): "
    if /i "%install_deps%"=="Y" (
        echo [INFO] Installing dependencies...
        pip install -r requirements.txt
        if %errorlevel% neq 0 (
            echo [ERROR] Failed to install dependencies
            pause
            exit /b 1
        )
    ) else (
        echo [ERROR] Cannot launch without Streamlit
        pause
        exit /b 1
    )
)

REM Create output directory if it doesn't exist
if not exist "output" mkdir output

echo ================================================================================
echo                    Launching Classic FRAP Analysis UI
echo ================================================================================
echo.
echo [INFO] Starting Streamlit application...
echo [INFO] The UI will open in your default web browser
echo [INFO] Default URL: http://localhost:8502
echo.
echo [TIP] To stop the application, press Ctrl+C in this window
echo.
echo ================================================================================
echo.

REM Launch Streamlit with optimal settings
streamlit run streamlit_frap_final.py ^
    --server.port=8502 ^
    --server.address=localhost ^
    --browser.gatherUsageStats=false ^
    --theme.base=light

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Failed to launch Streamlit
    echo.
    echo Troubleshooting:
    echo 1. Make sure all dependencies are installed: pip install -r requirements.txt
    echo 2. Check that no other application is using port 8502
    echo 3. Try running: python -m streamlit run streamlit_frap_final.py
    echo.
    pause
    exit /b 1
)

echo.
echo [INFO] Application closed
pause
