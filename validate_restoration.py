"""
Quick validation script to check if streamlit_frap_final_clean.py can be imported
"""
import sys
import os

print("Checking imports for streamlit_frap_final_clean.py...")
print("-" * 60)

# Test core imports
try:
    import streamlit as st
    print("[OK] streamlit")
except ImportError as e:
    print(f"[FAIL] streamlit: {e}")

try:
    import pandas as pd
    print("[OK] pandas")
except ImportError as e:
    print(f"[FAIL] pandas: {e}")

try:
    import numpy as np
    print("[OK] numpy")
except ImportError as e:
    print(f"[FAIL] numpy: {e}")

try:
    from scipy.optimize import curve_fit
    print("[OK] scipy.optimize")
except ImportError as e:
    print(f"[FAIL] scipy.optimize: {e}")

try:
    import plotly.graph_objects as go
    print("[OK] plotly")
except ImportError as e:
    print(f"[FAIL] plotly: {e}")

# Test custom module imports
try:
    from frap_core import FRAPAnalysisCore
    print("[OK] frap_core (CRITICAL FIX)")
except ImportError as e:
    print(f"[FAIL] frap_core: {e}")

try:
    from frap_pdf_reports import generate_pdf_report
    print("[OK] frap_pdf_reports")
except ImportError as e:
    print(f"[FAIL] frap_pdf_reports: {e}")

try:
    from frap_image_analysis import FRAPImageAnalyzer
    print("[OK] frap_image_analysis")
except ImportError as e:
    print(f"[FAIL] frap_image_analysis: {e}")

print("-" * 60)
print("\nChecking if streamlit_frap_final_clean.py syntax is valid...")

# Try to compile the file to check for syntax errors
try:
    with open('streamlit_frap_final_clean.py', 'r', encoding='utf-8') as f:
        code = f.read()
    compile(code, 'streamlit_frap_final_clean.py', 'exec')
    print("[OK] No syntax errors detected")
except SyntaxError as e:
    print(f"[FAIL] Syntax error: {e}")
except FileNotFoundError:
    print("[FAIL] File not found")

print("\n" + "=" * 60)
print("Validation complete!")
print("=" * 60)
