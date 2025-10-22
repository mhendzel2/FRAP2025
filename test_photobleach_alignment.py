#!/usr/bin/env python3
"""
Test script to verify photobleach alignment in multi-group comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
from frap_group_comparison import compute_average_recovery_profile

def create_test_data():
    """Create synthetic FRAP data with different bleaching timepoints."""
    
    # Create two groups with different bleaching times
    group1_data = {}
    group2_data = {}
    
    # Group 1: Bleaching at t=10s
    for i in range(3):
        t = np.linspace(0, 50, 100)
        # Create FRAP curve: high intensity, then bleach at t=10, then recovery
        intensity = np.ones_like(t)
        bleach_idx = np.argmin(np.abs(t - 10))  # Bleach at t=10
        
        # Add bleaching dip
        intensity[bleach_idx:] = 0.3 + 0.7 * (1 - np.exp(-(t[bleach_idx:] - 10) / 8))
        
        # Add some noise
        intensity += np.random.normal(0, 0.02, len(intensity))
        
        group1_data[f'file1_{i}'] = {
            'time': t,
            'intensity': intensity
        }
    
    # Group 2: Bleaching at t=15s (different time!)
    for i in range(3):
        t = np.linspace(0, 50, 100)
        intensity = np.ones_like(t)
        bleach_idx = np.argmin(np.abs(t - 15))  # Bleach at t=15
        
        # Add bleaching dip with slower recovery (treatment effect)
        intensity[bleach_idx:] = 0.2 + 0.6 * (1 - np.exp(-(t[bleach_idx:] - 15) / 12))
        
        # Add some noise
        intensity += np.random.normal(0, 0.02, len(intensity))
        
        group2_data[f'file2_{i}'] = {
            'time': t,
            'intensity': intensity
        }
    
    return group1_data, group2_data

def test_alignment():
    """Test that photobleach alignment works correctly."""
    print("Testing photobleach alignment in multi-group comparison...")
    
    # Create test data
    group1_data, group2_data = create_test_data()
    
    print(f"Group 1: {len(group1_data)} curves with bleaching at t=10s")
    print(f"Group 2: {len(group2_data)} curves with bleaching at t=15s")
    
    # Test alignment
    try:
        # Compute average profiles (should align to t=0)
        t1, i1_mean, i1_sem = compute_average_recovery_profile(group1_data)
        t2, i2_mean, i2_sem = compute_average_recovery_profile(group2_data)
        
        print(f"✅ Group 1 aligned profile: {len(t1)} points, t_min={t1[0]:.3f}s, t_max={t1[-1]:.3f}s")
        print(f"✅ Group 2 aligned profile: {len(t2)} points, t_min={t2[0]:.3f}s, t_max={t2[-1]:.3f}s")
        
        # Check that both start at t=0 (bleaching aligned)
        if abs(t1[0]) < 0.001 and abs(t2[0]) < 0.001:
            print("✅ Photobleach alignment working: Both groups start at t≈0")
        else:
            print(f"❌ Alignment failed: Group 1 starts at t={t1[0]:.3f}, Group 2 starts at t={t2[0]:.3f}")
            return False
        
        # Check that recovery patterns are preserved
        if i1_mean[0] < i1_mean[-1] and i2_mean[0] < i2_mean[-1]:
            print("✅ Recovery patterns preserved: Both groups show recovery from bleaching")
        else:
            print("❌ Recovery patterns lost during alignment")
            return False
        
        # Visual comparison plot
        plt.figure(figsize=(12, 5))
        
        # Plot 1: Raw data (before alignment)
        plt.subplot(1, 2, 1)
        for i, (name, data) in enumerate(group1_data.items()):
            plt.plot(data['time'], data['intensity'], 'b-', alpha=0.3, 
                    label='Group 1' if i == 0 else "")
        for i, (name, data) in enumerate(group2_data.items()):
            plt.plot(data['time'], data['intensity'], 'r-', alpha=0.3,
                    label='Group 2' if i == 0 else "")
        plt.axvline(10, color='blue', linestyle='--', alpha=0.5, label='Group 1 bleach (t=10s)')
        plt.axvline(15, color='red', linestyle='--', alpha=0.5, label='Group 2 bleach (t=15s)')
        plt.title('Before Alignment')
        plt.xlabel('Time (s)')
        plt.ylabel('Intensity')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Aligned averages
        plt.subplot(1, 2, 2)
        plt.plot(t1, i1_mean, 'b-', linewidth=3, label='Group 1 (aligned)')
        plt.fill_between(t1, i1_mean - i1_sem, i1_mean + i1_sem, alpha=0.3, color='blue')
        plt.plot(t2, i2_mean, 'r-', linewidth=3, label='Group 2 (aligned)')
        plt.fill_between(t2, i2_mean - i2_sem, i2_mean + i2_sem, alpha=0.3, color='red')
        plt.axvline(0, color='black', linestyle='--', alpha=0.5, label='Aligned bleach (t=0)')
        plt.title('After Alignment - Mean Profiles')
        plt.xlabel('Time since bleaching (s)')
        plt.ylabel('Intensity')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('photobleach_alignment_test.png', dpi=150, bbox_inches='tight')
        print("📊 Visualization saved as 'photobleach_alignment_test.png'")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the alignment test."""
    print("Photobleach Alignment Test for Multi-Group Comparison\n")
    
    success = test_alignment()
    
    if success:
        print("\n🎉 Photobleach alignment is working correctly!")
        print("✅ All curves are aligned to start at t=0 (bleaching timepoint)")
        print("✅ Recovery patterns are preserved")
        print("✅ Groups can now be meaningfully compared")
    else:
        print("\n⚠️ Alignment test failed - please check the implementation")
    
    return 0 if success else 1

if __name__ == "__main__":
    import sys
    sys.exit(main())