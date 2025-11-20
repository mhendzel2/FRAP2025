@echo off
setlocal

echo ==================================================
echo FRAP Analysis Platform - Launcher
echo ==================================================

REM Check if venv exists
if not exist "venv" (
    echo [ERROR] Virtual environment 'venv' not found.
    echo Please run install.bat first to set up the environment.
    pause
    exit /b 1
)

REM Activate Virtual Environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate

REM Run the Application
echo [INFO] Starting FRAP Analysis App (app.py)...
python -m streamlit run app.py

if %errorlevel% neq 0 (
    echo [ERROR] Application exited with an error.
    pause
)

echo [INFO] Application closed.
