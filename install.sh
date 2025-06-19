#!/bin/bash
# FRAP Analysis Platform - Corrected Installation Script

echo "Installing FRAP Analysis Platform (Corrected Version)..."
echo "This version includes critical mathematical fixes for diffusion coefficient calculation"

# Check Python version
python3 --version || { echo "Python 3 required"; exit 1; }

# Install dependencies
pip install streamlit pandas numpy scipy matplotlib plotly seaborn \
    scikit-image opencv-python tifffile reportlab xlsxwriter openpyxl \
    scikit-learn sqlalchemy psycopg2-binary

echo ""
echo "Installation complete!"
echo "CRITICAL: This corrected version uses D = (w² × k) / 4 for diffusion coefficient"
echo "Run with: streamlit run streamlit_frap_final.py --server.port 5000"
