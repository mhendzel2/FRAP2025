@echo off
REM ==============================================================================
REM FRAP Analysis Platform - Main Menu Launcher
REM ==============================================================================
REM This batch file provides a menu to launch different FRAP analysis interfaces
REM Date: October 15, 2025
REM ==============================================================================

:MENU
cls
echo.
echo ================================================================================
echo             FRAP Analysis Platform - Main Menu
echo ================================================================================
echo.
echo Please select which interface to launch:
echo.
echo   1. Single-Cell Analysis UI (Recommended)
echo   2. Classic FRAP Analysis UI
echo   3. Microirradiation Analysis UI
echo   4. Launch UI (Python-based launcher)
echo   5. Run Verification Tests
echo   6. Exit
echo.
echo ================================================================================
echo.

set /p choice="Enter your choice (1-6): "

if "%choice%"=="1" goto SINGLECELL
if "%choice%"=="2" goto CLASSIC
if "%choice%"=="3" goto MICROIRRADIATION
if "%choice%"=="4" goto LAUNCHUI
if "%choice%"=="5" goto VERIFY
if "%choice%"=="6" goto END
echo Invalid choice. Please try again.
timeout /t 2 >nul
goto MENU

:SINGLECELL
cls
echo.
echo ================================================================================
echo                   Launching Single-Cell Analysis UI
echo ================================================================================
echo.
call :CHECK_PYTHON
call :CHECK_FILES streamlit_singlecell.py
call :CHECK_STREAMLIT
echo [INFO] Starting Single-Cell Analysis UI...
echo [INFO] URL: http://localhost:8501
echo.
streamlit run streamlit_singlecell.py --server.port=8501 --server.address=localhost --browser.gatherUsageStats=false
goto MENU

:CLASSIC
cls
echo.
echo ================================================================================
echo                   Launching Classic FRAP Analysis UI
echo ================================================================================
echo.
call :CHECK_PYTHON
call :CHECK_FILES streamlit_frap_final_clean.py
call :CHECK_STREAMLIT
echo [INFO] Starting Classic FRAP Analysis UI...
echo [INFO] URL: http://localhost:8502
echo.
streamlit run streamlit_frap_final_clean.py --server.port=8502 --server.address=localhost --browser.gatherUsageStats=false
goto MENU

:MICROIRRADIATION
cls
echo.
echo ================================================================================
echo                   Launching Microirradiation Analysis UI
echo ================================================================================
echo.
call :CHECK_PYTHON
call :CHECK_FILES streamlit_microirradiation.py
call :CHECK_STREAMLIT
echo [INFO] Starting Microirradiation Analysis UI...
echo [INFO] URL: http://localhost:8503
echo.
streamlit run streamlit_microirradiation.py --server.port=8503 --server.address=localhost --browser.gatherUsageStats=false
goto MENU

:LAUNCHUI
cls
echo.
echo ================================================================================
echo                   Launching Python-based UI Launcher
echo ================================================================================
echo.
call :CHECK_PYTHON
call :CHECK_FILES launch_ui.py
echo [INFO] Starting Python launcher...
echo.
python launch_ui.py
goto MENU

:VERIFY
cls
echo.
echo ================================================================================
echo                   Running Verification Tests
echo ================================================================================
echo.
call :CHECK_PYTHON
echo [INFO] Running installation verification...
echo.
if exist "verify_installation.py" (
    python verify_installation.py
) else (
    echo [WARNING] verify_installation.py not found
)
echo.
echo Press any key to return to menu...
pause >nul
goto MENU

:CHECK_PYTHON
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH
    echo.
    echo Please install Python 3.11+ from https://www.python.org/
    pause
    exit /b 1
)
exit /b 0

:CHECK_FILES
if not exist "%~1" (
    echo [ERROR] Cannot find %~1
    echo.
    echo Please run this script from the FRAP2025 directory
    pause
    exit /b 1
)
exit /b 0

:CHECK_STREAMLIT
python -c "import streamlit" 2>nul
if %errorlevel% neq 0 (
    echo [WARNING] Streamlit is not installed
    echo.
    set /p install_deps="Would you like to install dependencies now? (Y/N): "
    if /i "!install_deps!"=="Y" (
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
exit /b 0

:END
echo.
echo Thank you for using FRAP Analysis Platform!
echo.
exit /b 0
