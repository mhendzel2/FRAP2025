@echo off
echo ================================================================================
echo FRAP Analysis Platform - Installation Script
echo ================================================================================
echo.

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.10+ from https://www.python.org/
    pause
    exit /b 1
)

echo [INFO] Python found:
python --version
echo.

echo [INFO] Installing dependencies from requirements.txt...
echo This may take several minutes...
echo.

pip install -r requirements.txt

if %errorlevel% equ 0 (
    echo.
    echo ================================================================================
    echo Installation complete!
    echo ================================================================================
    echo.
    echo To launch the application:
    echo   start.bat              - Launch main FRAP UI
    echo   start_singlecell.bat   - Launch single-cell analysis UI
    echo.
) else (
    echo.
    echo [ERROR] Installation failed. Please check the error messages above.
    echo.
)

pause
