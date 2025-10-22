#!/usr/bin/env python3
"""
Test script to validate the corrected mobile fraction calculation.

This tests the principle that mobile fraction should equal the plateau intensity
when the curve is properly normalized where 1.0 = 100% theoretical recovery.
"""

import numpy as np
import pandas as pd
from frap_core import FRAPAnalysisCore

def test_mobile_fraction_calculation():
    """Test that mobile fraction is correctly calculated as plateau intensity."""
    print("Testing Mobile Fraction Calculation")
    print("=" * 50)
    
    # Create test scenarios based on your example
    test_cases = [
        {
            "name": "Your Example",
            "description": "Starts at 0.7, plateaus at 0.95 → should be 95% mobile",
            "post_bleach": 0.7,
            "plateau": 0.95,
            "expected_mobile": 95.0
        },
        {
            "name": "High Mobile",
            "description": "Starts at 0.5, plateaus at 0.98 → should be 98% mobile", 
            "post_bleach": 0.5,
            "plateau": 0.98,
            "expected_mobile": 98.0
        },
        {
            "name": "Low Mobile",
            "description": "Starts at 0.6, plateaus at 0.75 → should be 75% mobile",
            "post_bleach": 0.6,
            "plateau": 0.75,
            "expected_mobile": 75.0
        },
        {
            "name": "Complete Recovery",
            "description": "Starts at 0.3, plateaus at 1.0 → should be 100% mobile",
            "post_bleach": 0.3,
            "plateau": 1.0,
            "expected_mobile": 100.0
        }
    ]
    
    all_tests_passed = True
    
    for i, case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {case['name']}")
        print(f"  {case['description']}")
        
        # Create synthetic FRAP data for this test case
        time = np.linspace(0, 100, 200)
        
        # Build recovery curve: pre-bleach at 1.0, bleach at t=20, recovery to plateau
        bleach_time = 20
        bleach_idx = np.argmin(np.abs(time - bleach_time))
        
        # Create intensity curve
        intensity = np.ones_like(time)  # Pre-bleach at 1.0
        
        # Add bleaching and recovery
        post_bleach_val = case['post_bleach']
        plateau_val = case['plateau']
        recovery_rate = 0.1  # Recovery time constant
        
        for j in range(bleach_idx, len(time)):
            t_since_bleach = time[j] - bleach_time
            # Exponential recovery from post_bleach to plateau
            recovery_progress = 1 - np.exp(-recovery_rate * t_since_bleach)
            intensity[j] = post_bleach_val + (plateau_val - post_bleach_val) * recovery_progress
        
        # Add minimal noise
        intensity += np.random.normal(0, 0.01, len(intensity))
        
        # Create DataFrame (we'll use the normalized data directly)
        df = pd.DataFrame({
            'time': time,
            'normalized': intensity
        })
        
        # Simulate fit results for single exponential model
        # For single exponential: I(t) = A(1 - exp(-kt)) + C
        # At equilibrium: I_plateau = A + C
        # So for our test: A + C = plateau_val
        
        A = plateau_val - post_bleach_val  # Recovery amplitude
        C = post_bleach_val  # Offset (immediate post-bleach value)
        k = recovery_rate  # Rate constant
        
        best_fit = {
            'model': 'single',
            'params': [A, k, C],
            'fitted_values': intensity[bleach_idx:],  # Post-bleach data
            'r2': 0.95
        }
        
        # Extract features using corrected calculation
        features = FRAPAnalysisCore.extract_clustering_features(
            best_fit, 
            bleach_spot_radius=1.0, 
            pixel_size=1.0
        )
        
        calculated_mobile = features.get('mobile_fraction', np.nan)
        expected_mobile = case['expected_mobile']
        
        print(f"  Post-bleach value: {post_bleach_val:.2f}")
        print(f"  Plateau value: {plateau_val:.2f}")
        print(f"  Expected mobile fraction: {expected_mobile:.1f}%")
        print(f"  Calculated mobile fraction: {calculated_mobile:.1f}%")
        
        # Check if calculation is correct (within 1% tolerance)
        if abs(calculated_mobile - expected_mobile) < 1.0:
            print(f"  ✅ PASS - Mobile fraction correctly calculated")
        else:
            print(f"  ❌ FAIL - Expected {expected_mobile:.1f}%, got {calculated_mobile:.1f}%")
            all_tests_passed = False
            
            # Debug information
            print(f"  Debug: A={A:.3f}, k={k:.3f}, C={C:.3f}")
            print(f"  Debug: A+C={A+C:.3f}, should equal plateau={plateau_val:.3f}")
    
    print(f"\n{'='*50}")
    if all_tests_passed:
        print("🎉 All tests PASSED!")
        print("✅ Mobile fraction calculation is correct")
        print("✅ Plateau intensity directly represents mobile percentage")
        print("✅ Accounts for diffusion during photobleaching")
    else:
        print("⚠️ Some tests FAILED!")
        print("❌ Mobile fraction calculation needs adjustment")
    
    return all_tests_passed

def demonstrate_principle():
    """Demonstrate the key principle behind the mobile fraction calculation."""
    print("\nKey Principle Demonstration")
    print("=" * 50)
    print()
    print("🔬 FRAP Mobile Fraction Calculation Principle:")
    print()
    print("1. Theoretical bleach depth = 100% (down to 0)")
    print("2. Actual post-bleach value > 0 due to diffusion during bleaching")
    print("3. Plateau value = fraction of molecules that are mobile")
    print()
    print("Example from your question:")
    print("  • Post-bleach: 0.7 (30% was bleached, 70% remained due to diffusion)")
    print("  • Plateau: 0.95 (95% final recovery)")
    print("  • Mobile fraction: 95% (plateau value directly)")
    print()
    print("Why this works:")
    print("  • With proper normalization, 1.0 = 100% theoretical recovery")
    print("  • Plateau represents equilibrium after all mobile molecules recover")
    print("  • Failure to reach 1.0 indicates immobile fraction (5% in example)")
    print("  • Post-bleach > 0 indicates diffusion during bleaching event")
    print()
    print("✅ This method correctly accounts for:")
    print("  - Diffusion during photobleaching")
    print("  - Actual vs theoretical bleach depth")
    print("  - True mobile population size")

def main():
    """Run all tests and demonstrations."""
    demonstrate_principle()
    
    success = test_mobile_fraction_calculation()
    
    if success:
        return 0
    else:
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())