@echo off
setlocal
echo ==================================================
echo FRAP Analysis Platform - Installation Script
echo ==================================================

REM Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not found in your PATH. Please install Python 3.9 or higher.
    pause
    exit /b 1
)

REM Create Virtual Environment
if not exist "venv" (
    echo [INFO] Creating virtual environment 'venv'...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
) else (
    echo [INFO] Virtual environment 'venv' already exists.
)

REM Activate Virtual Environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    exit /b 1
)

REM Upgrade pip
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip

REM Install Requirements
if exist "requirements.txt" (
    echo [INFO] Installing dependencies from requirements.txt...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to install dependencies.
        pause
        exit /b 1
    )
) else (
    echo [WARNING] requirements.txt not found. Skipping dependency installation.
)

REM Add venv to .gitignore
if exist ".gitignore" (
    findstr /C:"venv/" .gitignore >nul
    if %errorlevel% neq 0 (
        echo [INFO] Adding venv/ to .gitignore...
        echo.>> .gitignore
        echo venv/>> .gitignore
    ) else (
        echo [INFO] venv/ is already in .gitignore.
    )
) else (
    echo [INFO] Creating .gitignore and adding venv/...
    echo venv/> .gitignore
)

echo.
echo ==================================================
echo Installation Complete!
echo You can now run the application using start.bat
echo ==================================================
pause
