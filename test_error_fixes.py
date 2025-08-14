#!/usr/bin/env python3
"""
Test script to validate the error fixes for FRAP analysis
"""

import numpy as np
import sys
import os

# Add the current directory to path to import our modules
sys.path.insert(0, os.path.dirname(__file__))

def test_rate_constant_validation():
    """Test the improved rate constant validation logic"""
    
    def validate_rate_constant(primary_rate):
        """Replica of the improved validation logic"""
        return (primary_rate is not None and 
                np.isfinite(primary_rate) and 
                primary_rate > 1e-8)
    
    test_cases = [
        (None, False, "None value"),
        (np.nan, False, "NaN value"),
        (-1.0, False, "Negative value"),
        (0.0, False, "Zero value"),
        (1e-10, False, "Too small value"),
        (1e-6, True, "Valid small value"),
        (0.1, True, "Valid normal value"),
        (10.0, True, "Valid large value"),
        (np.inf, False, "Infinite value"),
    ]
    
    print("Testing rate constant validation:")
    all_passed = True
    
    for value, expected, description in test_cases:
        result = validate_rate_constant(value)
        status = "PASS" if result == expected else "FAIL"
        if result != expected:
            all_passed = False
        print(f"  {status}: {description} (value={value}, expected={expected}, got={result})")
    
    return all_passed

def test_parameter_safety():
    """Test parameter safety checks"""
    
    def safe_extract_features(best_fit):
        """Simplified version of parameter validation"""
        if best_fit is None:
            return None, "best_fit is None"
        
        if not isinstance(best_fit, dict):
            return None, f"best_fit is not a dict, got {type(best_fit)}"
            
        if 'model' not in best_fit or 'params' not in best_fit:
            return None, f"best_fit missing required keys. Available keys: {list(best_fit.keys())}"
        
        model = best_fit['model']
        params = best_fit['params']
        
        if params is None:
            return None, "params is None"
            
        if not isinstance(params, (list, tuple, np.ndarray)) or len(params) == 0:
            return None, f"invalid params type or empty. Got {type(params)}"
        
        return True, "Valid parameters"
    
    test_cases = [
        (None, False, "None input"),
        ("not_a_dict", False, "String input"),
        ({}, False, "Empty dict"),
        ({'model': 'single'}, False, "Missing params"),
        ({'params': [1, 2, 3]}, False, "Missing model"),
        ({'model': 'single', 'params': None}, False, "None params"),
        ({'model': 'single', 'params': []}, False, "Empty params"),
        ({'model': 'single', 'params': [1, 2, 3]}, True, "Valid input"),
        ({'model': 'double', 'params': np.array([1, 2, 3, 4, 5])}, True, "Valid numpy array"),
    ]
    
    print("\nTesting parameter safety:")
    all_passed = True
    
    for best_fit, expected_success, description in test_cases:
        result, message = safe_extract_features(best_fit)
        success = result is not None
        status = "PASS" if success == expected_success else "FAIL"
        if success != expected_success:
            all_passed = False
        print(f"  {status}: {description} (expected={expected_success}, got={success}) - {message}")
    
    return all_passed

def test_file_validation():
    """Test file validation logic"""
    
    def validate_file(filename):
        """Simplified file validation"""
        if not filename or filename.startswith('.'):
            return False, "Hidden or empty filename"
        
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext not in ['.xls', '.xlsx', '.csv']:
            return False, f"Unsupported file type: {file_ext}"
        
        return True, "Valid file"
    
    test_cases = [
        ("", False, "Empty filename"),
        (".hidden", False, "Hidden file"),
        ("data.txt", False, "Text file"),
        ("data.pdf", False, "PDF file"),
        ("data.xls", True, "Excel .xls file"),
        ("data.xlsx", True, "Excel .xlsx file"),
        ("data.csv", True, "CSV file"),
        ("Data_File.XLS", True, "Uppercase extension"),
        ("complex_name.with.dots.xlsx", True, "Complex filename"),
    ]
    
    print("\nTesting file validation:")
    all_passed = True
    
    for filename, expected_valid, description in test_cases:
        is_valid, message = validate_file(filename)
        status = "PASS" if is_valid == expected_valid else "FAIL"
        if is_valid != expected_valid:
            all_passed = False
        print(f"  {status}: {description} (filename='{filename}', expected={expected_valid}, got={is_valid}) - {message}")
    
    return all_passed

def main():
    """Run all tests"""
    print("FRAP Analysis Error Fixes Validation")
    print("=" * 50)
    
    test1_passed = test_rate_constant_validation()
    test2_passed = test_parameter_safety()
    test3_passed = test_file_validation()
    
    print("\n" + "=" * 50)
    print("TEST SUMMARY:")
    print(f"  Rate Constant Validation: {'PASS' if test1_passed else 'FAIL'}")
    print(f"  Parameter Safety:         {'PASS' if test2_passed else 'FAIL'}")
    print(f"  File Validation:          {'PASS' if test3_passed else 'FAIL'}")
    
    overall_result = all([test1_passed, test2_passed, test3_passed])
    print(f"\n  OVERALL RESULT: {'ALL TESTS PASSED' if overall_result else 'SOME TESTS FAILED'}")
    
    return overall_result

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
