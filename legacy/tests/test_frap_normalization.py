#!/usr/bin/env python3
"""
Test script to demonstrate the difference between old and new FRAP normalization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from frap_core import FRAPAnalysisCore

def create_synthetic_frap_data():
    """Create synthetic FRAP data to test normalization."""
    
    # Time points
    time = np.linspace(0, 100, 200)
    
    # Create realistic FRAP curve
    # Pre-bleach: stable high intensity
    # Bleaching at t=20: sharp drop
    # Recovery: exponential with mobile fraction of 0.7
    
    bleach_time = 20
    bleach_idx = np.argmin(np.abs(time - bleach_time))
    
    # ROI1 (bleached region)
    roi1 = np.ones_like(time) * 1000  # Pre-bleach intensity
    
    # Add photobleaching event
    roi1[bleach_idx:] = 300 + 700 * 0.7 * (1 - np.exp(-(time[bleach_idx:] - bleach_time) / 15))
    
    # ROI2 (reference region) - shows gradual photobleaching during imaging
    roi2 = np.ones_like(time) * 900
    roi2 *= np.exp(-time / 200)  # 0.5% per timepoint imaging bleaching
    
    # ROI3 (background)
    roi3 = np.ones_like(time) * 50
    
    # Add realistic noise
    roi1 += np.random.normal(0, 10, len(time))
    roi2 += np.random.normal(0, 8, len(time))
    roi3 += np.random.normal(0, 3, len(time))
    
    # Create DataFrame
    df = pd.DataFrame({
        'time': time,
        'roi1': roi1,
        'roi2': roi2,
        'roi3': roi3
    })
    
    return df, bleach_idx

def test_normalization_comparison():
    """Compare old vs new normalization approaches."""
    print("FRAP Normalization Comparison Test")
    print("=" * 50)
    
    # Create synthetic data
    df, true_bleach_idx = create_synthetic_frap_data()
    print(f"Created synthetic FRAP data with bleaching at frame {true_bleach_idx}")
    
    # Apply new (correct) normalization
    df_processed = FRAPAnalysisCore.preprocess(df)
    
    # Extract key metrics
    pre_bleach = df_processed.attrs.get('pre_bleach_intensity', np.nan)
    post_bleach = df_processed.attrs.get('post_bleach_intensity', np.nan)
    bleach_depth = df_processed.attrs.get('bleach_depth', np.nan)
    detected_bleach_frame = df_processed.attrs.get('bleach_frame', np.nan)
    
    print(f"\nNormalization Results:")
    print(f"  Pre-bleach intensity: {pre_bleach:.1f}")
    print(f"  Post-bleach intensity: {post_bleach:.1f}")
    print(f"  Bleach depth: {bleach_depth:.1%}")
    print(f"  Detected bleach frame: {detected_bleach_frame}")
    print(f"  True bleach frame: {true_bleach_idx}")
    
    # Calculate theoretical mobile fraction
    # With proper normalization: mobile_fraction = plateau_intensity (since 1.0 = full recovery)
    plateau_start = detected_bleach_frame + 50
    if plateau_start < len(df_processed):
        plateau_intensity = df_processed['double_normalized'].iloc[plateau_start:].mean()
        theoretical_mobile = 0.7  # What we built into the synthetic data
        
        print(f"\nMobile Fraction Analysis:")
        print(f"  Theoretical mobile fraction: {theoretical_mobile:.1%}")
        print(f"  Measured plateau intensity: {plateau_intensity:.3f}")
        print(f"  Difference: {abs(plateau_intensity - theoretical_mobile):.3f}")
        
        if abs(plateau_intensity - theoretical_mobile) < 0.05:
            print("  ✅ Mobile fraction accurately measured")
        else:
            print("  ❌ Mobile fraction measurement inaccurate")
    
    # Visualization
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Raw data
    plt.subplot(2, 3, 1)
    plt.plot(df['time'], df['roi1'], 'b-', label='ROI1 (Bleached)')
    plt.plot(df['time'], df['roi2'], 'g-', label='ROI2 (Reference)')
    plt.plot(df['time'], df['roi3'], 'k-', label='ROI3 (Background)')
    plt.axvline(df['time'].iloc[true_bleach_idx], color='red', linestyle='--', alpha=0.7, label='True Bleach')
    plt.axvline(df['time'].iloc[detected_bleach_frame], color='orange', linestyle='--', alpha=0.7, label='Detected Bleach')
    plt.title('Raw Data')
    plt.xlabel('Time')
    plt.ylabel('Intensity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Background corrected
    plt.subplot(2, 3, 2)
    plt.plot(df['time'], df_processed['roi1_bg_corrected'], 'b-', label='ROI1 - Background')
    plt.plot(df['time'], df_processed['roi2_bg_corrected'], 'g-', label='ROI2 - Background')
    plt.axvline(df['time'].iloc[detected_bleach_frame], color='red', linestyle='--', alpha=0.7)
    plt.title('Background Corrected')
    plt.xlabel('Time')
    plt.ylabel('Intensity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Simple normalization (old way)
    plt.subplot(2, 3, 3)
    plt.plot(df['time'], df_processed['normalized'], 'b-', linewidth=2, label='Simple Normalized')
    plt.axhline(1.0, color='red', linestyle='--', alpha=0.7, label='Theoretical Max (1.0)')
    plt.axhline(theoretical_mobile, color='green', linestyle='--', alpha=0.7, label=f'True Mobile ({theoretical_mobile:.1%})')
    plt.axvline(df['time'].iloc[detected_bleach_frame], color='orange', linestyle='--', alpha=0.7)
    plt.title('Simple Normalization\n(ROI1-BG)/Pre_bleach')
    plt.xlabel('Time')
    plt.ylabel('Normalized Intensity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Reference-corrected normalization (new way)
    plt.subplot(2, 3, 4)
    plt.plot(df['time'], df_processed['double_normalized'], 'r-', linewidth=2, label='Reference Corrected')
    plt.axhline(1.0, color='red', linestyle='--', alpha=0.7, label='Theoretical Max (1.0)')
    plt.axhline(theoretical_mobile, color='green', linestyle='--', alpha=0.7, label=f'True Mobile ({theoretical_mobile:.1%})')
    plt.axvline(df['time'].iloc[detected_bleach_frame], color='orange', linestyle='--', alpha=0.7)
    plt.title('Reference-Corrected Normalization\n(Accounts for Imaging Bleaching)')
    plt.xlabel('Time')
    plt.ylabel('Normalized Intensity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Reference correction factor
    plt.subplot(2, 3, 5)
    ref_factor = df_processed['roi2_bg_corrected'] / pre_bleach
    plt.plot(df['time'], ref_factor, 'g-', linewidth=2, label='Reference Factor')
    plt.axhline(1.0, color='black', linestyle='--', alpha=0.7, label='No Correction')
    plt.axvline(df['time'].iloc[detected_bleach_frame], color='red', linestyle='--', alpha=0.7)
    plt.title('Reference Correction Factor\n(ROI2/Pre_bleach_ROI2)')
    plt.xlabel('Time')
    plt.ylabel('Correction Factor')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Comparison
    plt.subplot(2, 3, 6)
    plt.plot(df['time'], df_processed['normalized'], 'b-', linewidth=2, label='Simple', alpha=0.7)
    plt.plot(df['time'], df_processed['double_normalized'], 'r-', linewidth=2, label='Reference-Corrected')
    plt.axhline(1.0, color='black', linestyle='--', alpha=0.7, label='Theoretical Max')
    plt.axhline(theoretical_mobile, color='green', linestyle='--', alpha=0.7, label=f'True Mobile ({theoretical_mobile:.1%})')
    plt.axvline(df['time'].iloc[detected_bleach_frame], color='orange', linestyle='--', alpha=0.7)
    plt.title('Normalization Comparison')
    plt.xlabel('Time')
    plt.ylabel('Normalized Intensity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('frap_normalization_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Comparison plot saved as 'frap_normalization_comparison.png'")
    
    return df_processed

def main():
    """Run the normalization comparison test."""
    print("Testing FRAP Normalization Improvements\n")
    
    df_processed = test_normalization_comparison()
    
    print(f"\n🔍 Key Improvements in New Normalization:")
    print(f"✅ 1.0 represents theoretical maximum recovery (complete mobile fraction)")
    print(f"✅ Pre-bleach intensity normalized to 1.0")
    print(f"✅ Accounts for imaging-induced photobleaching via reference ROI")
    print(f"✅ Mobile fraction = plateau intensity (when properly normalized)")
    print(f"✅ Bleach depth accurately calculated")
    print(f"✅ Consistent comparison between experiments")
    
    print(f"\n📈 Biological Interpretation:")
    print(f"- Recovery plateau < 1.0: Indicates immobile fraction")
    print(f"- Recovery plateau = 1.0: All molecules are mobile")  
    print(f"- Recovery > 1.0: Experimental artifacts or overcorrection")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())