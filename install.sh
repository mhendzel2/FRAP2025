#!/bin/bash
# FRAP Analysis Platform - Installation Script

echo "================================================================================"
echo "FRAP Analysis Platform - Installation Script"
echo "================================================================================"
echo ""

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is not installed"
    echo "Please install Python 3.10+ from https://www.python.org/"
    exit 1
fi

echo "[INFO] Python found:"
python3 --version
echo ""

echo "[INFO] Installing dependencies from requirements.txt..."
echo "This may take several minutes..."
echo ""

pip3 install -r requirements.txt

if [ $? -eq 0 ]; then
    echo ""
    echo "================================================================================"
    echo "Installation complete!"
    echo "================================================================================"
    echo ""
    echo "To launch the application:"
    echo "  streamlit run streamlit_frap_final_restored.py"
    echo "  streamlit run streamlit_singlecell.py"
    echo ""
else
    echo ""
    echo "[ERROR] Installation failed. Please check the error messages above."
    echo ""
fi
