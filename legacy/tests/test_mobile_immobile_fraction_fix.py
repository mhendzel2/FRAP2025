#!/usr/bin/env python3
"""
Test script to verify that mobile and immobile fractions add up to 100% correctly.
This tests the fix for the bug where mobile + immobile != 100%.
"""

import numpy as np
from frap_core import FRAPAnalysisCore

def test_mobile_immobile_fractions():
    """Test that mobile and immobile fractions add up to 100%"""
    
    print("Testing mobile/immobile fraction calculation fix...")
    
    # Test cases with different scenarios
    test_cases = [
        {
            'name': 'Normal single exponential fit',
            'best_fit': {
                'model': 'single',
                'params': [0.7, 0.5, 0.2],  # A=0.7, k=0.5, C=0.2 -> mobile = (0.7+0.2)*100 = 90%
                'fitted_values': np.array([0.2, 0.4, 0.6, 0.8, 0.9, 0.9, 0.9])  # plateaued
            }
        },
        {
            'name': 'Double exponential fit',
            'best_fit': {
                'model': 'double',
                'params': [0.4, 1.0, 0.3, 0.2, 0.15],  # A1=0.4, k1=1.0, A2=0.3, k2=0.2, C=0.15 -> mobile = (0.4+0.3+0.15)*100 = 85%
                'fitted_values': np.array([0.15, 0.3, 0.5, 0.7, 0.85, 0.85, 0.85])  # plateaued
            }
        },
        {
            'name': 'Edge case: very low mobile fraction',
            'best_fit': {
                'model': 'single',
                'params': [0.1, 0.3, 0.85],  # A=0.1, k=0.3, C=0.85 -> mobile = (0.1+0.85)*100 = 95%
                'fitted_values': np.array([0.85, 0.88, 0.91, 0.94, 0.95, 0.95, 0.95])  # plateaued
            }
        },
        {
            'name': 'Edge case: very high mobile fraction',
            'best_fit': {
                'model': 'single',
                'params': [0.95, 0.8, 0.03],  # A=0.95, k=0.8, C=0.03 -> mobile = (0.95+0.03)*100 = 98%
                'fitted_values': np.array([0.03, 0.2, 0.5, 0.8, 0.95, 0.98, 0.98])  # plateaued
            }
        },
        {
            'name': 'Invalid case: should result in NaN fractions',
            'best_fit': {
                'model': 'single',
                'params': [np.nan, 0.5, 0.2],  # Invalid A parameter
                'fitted_values': np.array([0.2, 0.4, 0.6, 0.8, 0.9, 0.9, 0.9])
            }
        }
    ]
    
    print(f"{'Test Case':<30} {'Mobile (%)':<12} {'Immobile (%)':<14} {'Sum (%)':<10} {'Status':<10}")
    print("-" * 80)
    
    all_passed = True
    
    for test_case in test_cases:
        name = test_case['name']
        best_fit = test_case['best_fit']
        
        # Extract features using the fixed function
        features = FRAPAnalysisCore.extract_clustering_features(best_fit)
        
        if features is None:
            print(f"{name:<30} {'N/A':<12} {'N/A':<14} {'N/A':<10} {'FAILED':<10}")
            all_passed = False
            continue
            
        mobile = features.get('mobile_fraction', np.nan)
        immobile = features.get('immobile_fraction', np.nan)
        
        # Format values for display
        mobile_str = f"{mobile:.2f}" if np.isfinite(mobile) else "NaN"
        immobile_str = f"{immobile:.2f}" if np.isfinite(immobile) else "NaN"
        
        # Check if the sum is approximately 100% (or both are NaN)
        if np.isfinite(mobile) and np.isfinite(immobile):
            total = mobile + immobile
            total_str = f"{total:.2f}"
            status = "PASS" if abs(total - 100.0) < 0.01 else "FAIL"
            if status == "FAIL":
                all_passed = False
        elif np.isnan(mobile) and np.isnan(immobile):
            total_str = "NaN"
            status = "PASS"  # Both NaN is acceptable for invalid cases
        else:
            total_str = "MISMATCH"
            status = "FAIL"
            all_passed = False
            
        print(f"{name:<30} {mobile_str:<12} {immobile_str:<14} {total_str:<10} {status:<10}")
    
    print("-" * 80)
    if all_passed:
        print("✅ ALL TESTS PASSED! Mobile and immobile fractions now add up to 100% correctly.")
    else:
        print("❌ SOME TESTS FAILED! There may still be issues with the fraction calculations.")
    
    return all_passed

if __name__ == "__main__":
    test_mobile_immobile_fractions()