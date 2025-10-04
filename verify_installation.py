#!/usr/bin/env python3
"""
Installation Verification Script for FRAP Single-Cell Analysis

Run this script to verify that all dependencies are installed correctly
and the system is ready to use.

Usage:
    python verify_installation.py
"""

import sys
import importlib
from typing import Tuple, List


def check_python_version() -> Tuple[bool, str]:
    """Check Python version >= 3.10"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10:
        return True, f"Python {version.major}.{version.minor}.{version.micro}"
    else:
        return False, f"Python {version.major}.{version.minor}.{version.micro} (requires >= 3.10)"


def check_module(module_name: str, optional: bool = False) -> Tuple[bool, str]:
    """Check if a module can be imported"""
    try:
        mod = importlib.import_module(module_name)
        version = getattr(mod, '__version__', 'unknown')
        return True, f"{module_name} {version}"
    except ImportError as e:
        if optional:
            return True, f"{module_name} (optional, not installed)"
        else:
            return False, f"{module_name} - MISSING: {str(e)}"


def check_custom_modules() -> Tuple[bool, List[str]]:
    """Check custom FRAP modules"""
    modules = [
        'frap_data_model',
        'frap_tracking',
        'frap_signal',
        'frap_fitting',
        'frap_populations',
        'frap_statistics',
        'frap_visualizations',
        'frap_singlecell_api',
        'test_synthetic'
    ]
    
    results = []
    all_ok = True
    
    for module in modules:
        try:
            importlib.import_module(module)
            results.append(f"✓ {module}")
        except ImportError as e:
            results.append(f"✗ {module} - {str(e)}")
            all_ok = False
    
    return all_ok, results


def run_verification():
    """Run complete verification"""
    
    print("=" * 70)
    print("FRAP Single-Cell Analysis - Installation Verification")
    print("=" * 70)
    
    # Check Python version
    print("\n1. Python Version:")
    ok, msg = check_python_version()
    print(f"   {'✓' if ok else '✗'} {msg}")
    
    if not ok:
        print("\n⚠ WARNING: Python 3.10 or higher is required!")
        print("   Please upgrade Python before proceeding.")
        return False
    
    # Required dependencies
    print("\n2. Required Dependencies:")
    required = [
        'numpy',
        'scipy',
        'pandas',
        'matplotlib',
        'seaborn',
        'sklearn',
        'skimage',
        'cv2',
        'statsmodels',
        'joblib'
    ]
    
    all_required_ok = True
    for module in required:
        ok, msg = check_module(module)
        print(f"   {'✓' if ok else '✗'} {msg}")
        if not ok:
            all_required_ok = False
    
    # Optional dependencies
    print("\n3. Optional Dependencies:")
    optional = [
        'dabest',
        'filterpy',
        'pymc',
        'pytest',
        'pyarrow'
    ]
    
    for module in optional:
        ok, msg = check_module(module, optional=True)
        print(f"   {'○' if ok else '✗'} {msg}")
    
    # Custom modules
    print("\n4. FRAP Single-Cell Modules:")
    modules_ok, module_results = check_custom_modules()
    for result in module_results:
        print(f"   {result}")
    
    # Quick functionality test
    print("\n5. Quick Functionality Test:")
    try:
        from test_synthetic import synth_movie
        from frap_singlecell_api import track_movie, fit_cells
        
        print("   ✓ Generating synthetic movie...")
        movie, gt = synth_movie(n_cells=3, T=20, random_state=42)
        
        print("   ✓ Testing tracking...")
        import numpy as np
        time_points = np.arange(movie.shape[0])
        traces = track_movie(movie, time_points, 'test', 'test')
        
        print("   ✓ Testing fitting...")
        features = fit_cells(traces)
        
        print(f"   ✓ Successfully tracked {len(traces['cell_id'].unique())} cells")
        print(f"   ✓ Successfully fitted {len(features)} curves")
        
        functionality_ok = True
    except Exception as e:
        print(f"   ✗ Functionality test failed: {e}")
        functionality_ok = False
    
    # Summary
    print("\n" + "=" * 70)
    if all_required_ok and modules_ok and functionality_ok:
        print("✓ INSTALLATION VERIFIED SUCCESSFULLY")
        print("\nYou are ready to use the FRAP single-cell analysis system!")
        print("\nNext steps:")
        print("1. Try the quick start examples:")
        print("   python quick_start_singlecell.py")
        print("\n2. Run the test suite:")
        print("   pytest test_frap_singlecell.py -v")
        print("\n3. See README_SINGLECELL.md for documentation")
        return True
    else:
        print("✗ INSTALLATION INCOMPLETE")
        print("\nIssues detected:")
        if not all_required_ok:
            print("- Some required dependencies are missing")
            print("  → Run: pip install -r requirements.txt")
        if not modules_ok:
            print("- Some custom modules could not be imported")
            print("  → Check that all .py files are in the same directory")
        if not functionality_ok:
            print("- Functionality test failed")
            print("  → Check error messages above for details")
        return False


if __name__ == '__main__':
    success = run_verification()
    sys.exit(0 if success else 1)
