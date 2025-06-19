@echo off
echo Installing FRAP Analysis Platform (Corrected Version)...
echo This version includes critical mathematical fixes for diffusion coefficient calculation

python --version >nul 2>&1 || (
    echo Python required. Please install Python 3.11+
    pause
    exit /b 1
)

pip install streamlit pandas numpy scipy matplotlib plotly seaborn scikit-image opencv-python tifffile reportlab xlsxwriter openpyxl scikit-learn sqlalchemy psycopg2-binary

echo.
echo Installation complete!
echo CRITICAL: This corrected version uses D = (w^2 * k) / 4 for diffusion coefficient
echo Run with: streamlit run streamlit_frap_final.py --server.port 5000
pause
