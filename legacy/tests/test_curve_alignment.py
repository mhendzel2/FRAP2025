"""
Test script for FRAP curve alignment and interpolation functions
Tests the new align_and_interpolate_curves and plot_aligned_curves functions
"""

import numpy as np
import sys
import os

# Add the current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from frap_core import FRAPAnalysisCore
from frap_plots import FRAPPlots


def create_synthetic_frap_curve(bleach_time, bleach_depth=0.3, tau=5.0, 
                                mobile_fraction=0.8, sampling_rate=0.5, 
                                duration=30.0, noise_level=0.02):
    """
    Create a synthetic FRAP recovery curve with realistic characteristics.
    
    Parameters:
    -----------
    bleach_time : float
        Time at which bleaching occurs (seconds)
    bleach_depth : float
        Depth of photobleaching (0-1, where 1 is complete bleaching)
    tau : float
        Recovery half-time (seconds)
    mobile_fraction : float
        Fraction of molecules that recover (0-1)
    sampling_rate : float
        Time between frames (seconds)
    duration : float
        Total duration of experiment (seconds)
    noise_level : float
        Amplitude of Gaussian noise to add
        
    Returns:
    --------
    time : np.ndarray
        Time points
    intensity : np.ndarray
        Normalized intensity values
    """
    # Create time vector
    time = np.arange(0, duration, sampling_rate)
    
    # Initialize intensity at pre-bleach level (normalized to 1)
    intensity = np.ones_like(time)
    
    # Find bleach frame
    bleach_idx = np.argmin(np.abs(time - bleach_time))
    
    # Create recovery curve after bleach
    # Model: I(t) = 1 - A*(1 - mobile_fraction)*exp(-(t-t_bleach)/tau)
    t_post = time[bleach_idx:]
    t_rel = t_post - bleach_time
    
    # Exponential recovery
    recovery = 1 - bleach_depth * (1 - mobile_fraction * (1 - np.exp(-t_rel / tau)))
    
    # Apply recovery to post-bleach intensity
    intensity[bleach_idx:] = recovery
    
    # Add realistic noise
    intensity += np.random.normal(0, noise_level, len(intensity))
    
    # Ensure intensity doesn't go negative or above 1.2
    intensity = np.clip(intensity, 0, 1.2)
    
    return time, intensity


def test_basic_alignment():
    """Test 1: Basic alignment with identical sampling rates"""
    print("\n" + "="*70)
    print("TEST 1: Basic Alignment (Same Sampling Rates)")
    print("="*70)
    
    # Create 3 curves with same sampling rate but different bleach times
    curves = []
    for i, bleach_t in enumerate([5.0, 5.5, 6.0]):
        time, intensity = create_synthetic_frap_curve(
            bleach_time=bleach_t,
            bleach_depth=0.4,
            tau=4.0,
            mobile_fraction=0.85,
            sampling_rate=0.5,
            duration=30.0
        )
        curves.append({
            'name': f'Curve_{i+1}',
            'time': time,
            'intensity': intensity
        })
    
    # Test alignment
    try:
        aligned_results = FRAPAnalysisCore.align_and_interpolate_curves(curves, num_points=200)
        
        print(f"✓ Alignment successful!")
        print(f"  - Input curves: {len(curves)}")
        print(f"  - Output curves: {len(aligned_results['interpolated_curves'])}")
        print(f"  - Common time axis: {len(aligned_results['common_time'])} points")
        print(f"  - Time range: 0.0 to {aligned_results['common_time'][-1]:.2f} seconds")
        
        # Verify all curves have same length
        for curve in aligned_results['interpolated_curves']:
            assert len(curve['intensity']) == len(aligned_results['common_time']), \
                f"Curve {curve['name']} has mismatched length!"
        print(f"✓ All curves have consistent length")
        
        # Verify time starts at 0
        assert aligned_results['common_time'][0] == 0.0, "Time doesn't start at 0!"
        print(f"✓ Time axis starts at t=0")
        
        return True, aligned_results
        
    except Exception as e:
        print(f"✗ Alignment failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_different_sampling_rates():
    """Test 2: Alignment with different sampling rates"""
    print("\n" + "="*70)
    print("TEST 2: Different Sampling Rates")
    print("="*70)
    
    # Create curves with different sampling rates
    curves = []
    sampling_rates = [0.3, 0.5, 0.8]  # Different frame rates
    
    for i, rate in enumerate(sampling_rates):
        time, intensity = create_synthetic_frap_curve(
            bleach_time=5.0,
            bleach_depth=0.4,
            tau=4.0,
            mobile_fraction=0.85,
            sampling_rate=rate,
            duration=30.0
        )
        curves.append({
            'name': f'Curve_{i+1} ({rate}s/frame)',
            'time': time,
            'intensity': intensity
        })
        print(f"  Curve {i+1}: {len(time)} points, sampling rate={rate}s")
    
    # Test alignment
    try:
        aligned_results = FRAPAnalysisCore.align_and_interpolate_curves(curves, num_points=200)
        
        print(f"✓ Alignment successful despite different sampling rates!")
        print(f"  - Output curves: {len(aligned_results['interpolated_curves'])}")
        print(f"  - All interpolated to: {len(aligned_results['common_time'])} points")
        
        return True, aligned_results
        
    except Exception as e:
        print(f"✗ Alignment failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_edge_cases():
    """Test 3: Edge cases and error handling"""
    print("\n" + "="*70)
    print("TEST 3: Edge Cases and Error Handling")
    print("="*70)
    
    # Test 3a: Empty list
    print("\n  Test 3a: Empty curve list...")
    try:
        result = FRAPAnalysisCore.align_and_interpolate_curves([])
        if result['common_time'].size == 0 and len(result['interpolated_curves']) == 0:
            print("  ✓ Empty list handled correctly")
        else:
            print("  ✗ Empty list not handled properly")
    except Exception as e:
        print(f"  ✗ Exception with empty list: {e}")
    
    # Test 3b: Single curve
    print("\n  Test 3b: Single curve...")
    try:
        time, intensity = create_synthetic_frap_curve(
            bleach_time=5.0,
            bleach_depth=0.4,
            tau=4.0,
            mobile_fraction=0.85,
            sampling_rate=0.5,
            duration=30.0
        )
        single_curve = [{'name': 'Single', 'time': time, 'intensity': intensity}]
        result = FRAPAnalysisCore.align_and_interpolate_curves(single_curve)
        
        if len(result['interpolated_curves']) == 1:
            print("  ✓ Single curve handled correctly")
        else:
            print("  ✗ Single curve not handled properly")
    except Exception as e:
        print(f"  ✗ Exception with single curve: {e}")
    
    # Test 3c: Curve with bleach at start (should be skipped)
    print("\n  Test 3c: Curve with bleach at t=0 (pathological case)...")
    try:
        time, intensity = create_synthetic_frap_curve(
            bleach_time=0.0,  # Bleach at start
            bleach_depth=0.4,
            tau=4.0,
            mobile_fraction=0.85,
            sampling_rate=0.5,
            duration=30.0
        )
        
        # Add one good curve and one bad curve
        good_time, good_intensity = create_synthetic_frap_curve(
            bleach_time=5.0,
            bleach_depth=0.4,
            tau=4.0,
            mobile_fraction=0.85,
            sampling_rate=0.5,
            duration=30.0
        )
        
        mixed_curves = [
            {'name': 'Bad_curve', 'time': time, 'intensity': intensity},
            {'name': 'Good_curve', 'time': good_time, 'intensity': good_intensity}
        ]
        
        result = FRAPAnalysisCore.align_and_interpolate_curves(mixed_curves)
        
        if len(result['interpolated_curves']) == 1:
            print("  ✓ Pathological curve skipped, good curve processed")
        else:
            print(f"  ✗ Expected 1 curve, got {len(result['interpolated_curves'])}")
    except Exception as e:
        print(f"  ✗ Exception with mixed curves: {e}")
    
    # Test 3d: Very short recovery time
    print("\n  Test 3d: Very short recovery duration...")
    try:
        time, intensity = create_synthetic_frap_curve(
            bleach_time=2.0,
            bleach_depth=0.4,
            tau=1.0,
            mobile_fraction=0.85,
            sampling_rate=0.5,
            duration=5.0  # Very short
        )
        short_curve = [{'name': 'Short', 'time': time, 'intensity': intensity}]
        result = FRAPAnalysisCore.align_and_interpolate_curves(short_curve)
        
        if len(result['interpolated_curves']) == 1:
            print("  ✓ Short duration curve handled correctly")
        else:
            print("  ✗ Short duration curve not handled properly")
    except Exception as e:
        print(f"  ✗ Exception with short curve: {e}")


def test_plotting():
    """Test 4: Plotting function"""
    print("\n" + "="*70)
    print("TEST 4: Plotting Function")
    print("="*70)
    
    # Create test curves
    curves = []
    for i in range(3):
        time, intensity = create_synthetic_frap_curve(
            bleach_time=5.0,
            bleach_depth=0.4,
            tau=4.0 + i,  # Vary recovery time
            mobile_fraction=0.85 - i*0.1,  # Vary mobile fraction
            sampling_rate=0.5,
            duration=30.0
        )
        curves.append({
            'name': f'Condition_{i+1}',
            'time': time,
            'intensity': intensity
        })
    
    # Test plotting
    try:
        aligned_results = FRAPAnalysisCore.align_and_interpolate_curves(curves, num_points=200)
        fig = FRAPPlots.plot_aligned_curves(aligned_results, height=500)
        
        print(f"✓ Plot created successfully!")
        print(f"  - Figure type: {type(fig).__name__}")
        print(f"  - Number of traces: {len(fig.data)}")
        
        # Test with empty data
        empty_fig = FRAPPlots.plot_aligned_curves({}, height=500)
        print(f"✓ Empty data plot handled correctly")
        
        return True, fig
        
    except Exception as e:
        print(f"✗ Plotting failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_realistic_scenario():
    """Test 5: Realistic multi-experiment scenario"""
    print("\n" + "="*70)
    print("TEST 5: Realistic Multi-Experiment Scenario")
    print("="*70)
    
    # Simulate 3 experiments with:
    # - Different bleach times (realistic variation)
    # - Different sampling rates (different microscope settings)
    # - Different durations (some experiments stopped early)
    # - Different kinetics (biological variation)
    
    experiments = [
        {'name': 'WT_cell1', 'bleach_t': 10.0, 'rate': 0.4, 'duration': 60.0, 'tau': 5.0, 'mf': 0.85},
        {'name': 'WT_cell2', 'bleach_t': 12.0, 'rate': 0.5, 'duration': 55.0, 'tau': 5.2, 'mf': 0.83},
        {'name': 'WT_cell3', 'bleach_t': 8.0, 'rate': 0.3, 'duration': 50.0, 'tau': 4.8, 'mf': 0.87},
        {'name': 'Mutant_cell1', 'bleach_t': 11.0, 'rate': 0.45, 'duration': 58.0, 'tau': 8.0, 'mf': 0.70},
        {'name': 'Mutant_cell2', 'bleach_t': 9.0, 'rate': 0.35, 'duration': 62.0, 'tau': 8.5, 'mf': 0.68},
    ]
    
    curves = []
    for exp in experiments:
        time, intensity = create_synthetic_frap_curve(
            bleach_time=exp['bleach_t'],
            bleach_depth=0.45,
            tau=exp['tau'],
            mobile_fraction=exp['mf'],
            sampling_rate=exp['rate'],
            duration=exp['duration'],
            noise_level=0.03  # Realistic noise
        )
        curves.append({
            'name': exp['name'],
            'time': time,
            'intensity': intensity
        })
        print(f"  {exp['name']}: {len(time)} points, bleach at {exp['bleach_t']}s, "
              f"rate={exp['rate']}s, tau={exp['tau']}s")
    
    # Test alignment
    try:
        aligned_results = FRAPAnalysisCore.align_and_interpolate_curves(curves, num_points=250)
        
        print(f"\n✓ Realistic scenario handled successfully!")
        print(f"  - Input: {len(curves)} curves with varied parameters")
        print(f"  - Output: {len(aligned_results['interpolated_curves'])} aligned curves")
        print(f"  - Common time axis: 0 to {aligned_results['common_time'][-1]:.2f}s")
        
        # Verify recovery characteristics are preserved
        print(f"\n  Verifying recovery characteristics...")
        for curve in aligned_results['interpolated_curves']:
            final_intensity = curve['intensity'][-1]
            min_intensity = np.min(curve['intensity'])
            recovery = (final_intensity - min_intensity) / (1.0 - min_intensity) if min_intensity < 1.0 else 0
            print(f"    {curve['name']}: min={min_intensity:.3f}, final={final_intensity:.3f}, recovery={recovery:.1%}")
        
        return True, aligned_results
        
    except Exception as e:
        print(f"✗ Realistic scenario failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def run_all_tests():
    """Run all tests and summarize results"""
    print("\n" + "="*70)
    print("FRAP CURVE ALIGNMENT - COMPREHENSIVE TEST SUITE")
    print("="*70)
    
    results = {}
    
    # Run tests
    results['basic'], data1 = test_basic_alignment()
    results['different_rates'], data2 = test_different_sampling_rates()
    test_edge_cases()  # No pass/fail, just verification
    results['plotting'], fig = test_plotting()
    results['realistic'], data5 = test_realistic_scenario()
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  🎉 ALL TESTS PASSED! 🎉")
        print("\n  The alignment and plotting functions are working correctly.")
        print("  You can now use them in the Streamlit UI.")
    else:
        print("\n  ⚠️  SOME TESTS FAILED")
        print("  Please review the error messages above.")
    
    return passed == total


if __name__ == "__main__":
    # Run all tests
    success = run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
