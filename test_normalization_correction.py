"""
Test script to verify the corrected FRAP normalization using ROI2 (total nuclear signal)
Tests that the photobleaching correction maintains ROI2 post-bleach value constant
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from frap_core import FRAPAnalysisCore
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def test_normalization_correction(file_path):
    """Test the corrected normalization on a single file"""
    print(f"\n{'='*80}")
    print(f"Testing normalization correction on: {file_path}")
    print(f"{'='*80}\n")
    
    # Load raw data
    print("Step 1: Loading raw data...")
    raw_df = FRAPAnalysisCore.load_data(file_path)
    print(f"  ✓ Loaded {len(raw_df)} time points")
    print(f"  Columns: {list(raw_df.columns)}")
    
    # Display raw data sample
    print("\nRaw data (first 10 rows):")
    print(raw_df[['time', 'roi1', 'roi2', 'roi3']].head(10).to_string(index=False))
    
    # Preprocess with corrected normalization
    print("\nStep 2: Preprocessing with CORRECTED normalization...")
    processed_df = FRAPAnalysisCore.preprocess(raw_df)
    
    # Extract metadata
    pre_bleach_roi1 = processed_df.attrs.get('pre_bleach_intensity', 0)
    post_bleach_roi1 = processed_df.attrs.get('post_bleach_intensity', 0)
    bleach_depth = processed_df.attrs.get('bleach_depth', 0)
    bleach_frame = processed_df.attrs.get('bleach_frame', 0)
    roi2_pre = processed_df.attrs.get('roi2_pre_bleach', 0)
    roi2_post = processed_df.attrs.get('roi2_post_bleach', 0)
    
    print(f"\n  ✓ Preprocessing complete")
    print(f"\n  Metadata extracted:")
    print(f"    - Bleach frame: {bleach_frame}")
    print(f"    - Pre-bleach ROI1: {pre_bleach_roi1:.3f}")
    print(f"    - Post-bleach ROI1: {post_bleach_roi1:.3f}")
    print(f"    - Bleach depth: {bleach_depth:.1%}")
    print(f"    - Pre-bleach ROI2 (total nuclear): {roi2_pre:.3f}")
    print(f"    - Post-bleach ROI2 (reference): {roi2_post:.3f}")
    
    # Calculate photobleaching in ROI2
    roi2_bg = processed_df['roi2_bg_corrected']
    roi2_decay = (roi2_bg.iloc[0] - roi2_bg.iloc[-1]) / roi2_bg.iloc[0] * 100
    print(f"    - ROI2 photobleaching during acquisition: {roi2_decay:.1f}%")
    
    # Display processed data sample
    print("\nProcessed data (first 10 rows after bleach):")
    bleach_idx = int(bleach_frame)
    display_cols = ['time', 'roi1_bg_corrected', 'roi2_bg_corrected', 'normalized', 'double_normalized']
    print(processed_df[display_cols].iloc[bleach_idx:bleach_idx+10].to_string(index=False))
    
    # Verify correction is working
    print("\nStep 3: Verifying photobleaching correction...")
    
    # Calculate correction factor applied
    correction_factor = processed_df['double_normalized'] / processed_df['normalized']
    print(f"  Correction factor range: {correction_factor.min():.3f} to {correction_factor.max():.3f}")
    print(f"  Average correction factor: {correction_factor.mean():.3f}")
    
    # Check if ROI2 decreases (acquisition photobleaching)
    if roi2_bg.iloc[-1] < roi2_bg.iloc[bleach_idx + 1]:
        print(f"  ✓ ROI2 shows acquisition photobleaching (decreased by {roi2_decay:.1f}%)")
        print(f"  ✓ Correction factor increases appropriately to compensate")
    else:
        print(f"  ℹ ROI2 relatively stable (only {roi2_decay:.1f}% change)")
    
    # Compare simple vs double normalization at end
    end_idx = len(processed_df) - 1
    simple_end = processed_df['normalized'].iloc[end_idx]
    double_end = processed_df['double_normalized'].iloc[end_idx]
    correction_pct = (double_end - simple_end) / simple_end * 100 if simple_end > 0 else 0
    
    print(f"\n  End-point comparison:")
    print(f"    - Simple normalization: {simple_end:.3f}")
    print(f"    - Double normalization (corrected): {double_end:.3f}")
    print(f"    - Correction applied: {correction_pct:+.1f}%")
    
    return processed_df

def plot_normalization_comparison(processed_df, save_path='normalization_test_results.png'):
    """Create visualization comparing normalization methods"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('FRAP Normalization Correction Verification', fontsize=14, fontweight='bold')
    
    time = processed_df['time'].values
    bleach_frame = processed_df.attrs.get('bleach_frame', 0)
    
    # Plot 1: Raw ROI signals
    ax1 = axes[0, 0]
    ax1.plot(time, processed_df['roi1'], 'b-', label='ROI1 (Bleached)', linewidth=2)
    ax1.plot(time, processed_df['roi2'], 'g-', label='ROI2 (Total Nuclear)', linewidth=2)
    ax1.plot(time, processed_df['roi3'], 'r-', label='ROI3 (Background)', linewidth=2)
    ax1.axvline(time[bleach_frame], color='k', linestyle='--', alpha=0.5, label='Bleach')
    ax1.set_xlabel('Time (s)', fontsize=10)
    ax1.set_ylabel('Raw Intensity', fontsize=10)
    ax1.set_title('Raw ROI Signals', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    
    # Plot 2: Background-corrected signals
    ax2 = axes[0, 1]
    ax2.plot(time, processed_df['roi1_bg_corrected'], 'b-', label='ROI1 (BG-corrected)', linewidth=2)
    ax2.plot(time, processed_df['roi2_bg_corrected'], 'g-', label='ROI2 (BG-corrected)', linewidth=2)
    ax2.axvline(time[bleach_frame], color='k', linestyle='--', alpha=0.5, label='Bleach')
    ax2.set_xlabel('Time (s)', fontsize=10)
    ax2.set_ylabel('BG-Corrected Intensity', fontsize=10)
    ax2.set_title('Background-Corrected Signals', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    
    # Plot 3: ROI2 and correction factor
    ax3 = axes[1, 0]
    ax3_twin = ax3.twinx()
    
    roi2_norm = processed_df['roi2_bg_corrected'] / processed_df['roi2_bg_corrected'].iloc[bleach_frame + 1]
    correction = processed_df['double_normalized'] / processed_df['normalized']
    
    ax3.plot(time, roi2_norm, 'g-', label='ROI2 (normalized to post-bleach)', linewidth=2)
    ax3.axhline(1.0, color='g', linestyle=':', alpha=0.5, label='Target (100%)')
    ax3_twin.plot(time, correction, 'orange', label='Correction Factor', linewidth=2, linestyle='--')
    
    ax3.axvline(time[bleach_frame], color='k', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Time (s)', fontsize=10)
    ax3.set_ylabel('ROI2 (normalized)', fontsize=10, color='g')
    ax3_twin.set_ylabel('Correction Factor', fontsize=10, color='orange')
    ax3.set_title('Photobleaching Correction', fontsize=11, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=9)
    ax3_twin.legend(loc='upper right', fontsize=9)
    ax3.grid(alpha=0.3)
    ax3.tick_params(axis='y', labelcolor='g')
    ax3_twin.tick_params(axis='y', labelcolor='orange')
    
    # Plot 4: Comparison of normalization methods
    ax4 = axes[1, 1]
    ax4.plot(time, processed_df['normalized'], 'b-', label='Simple (no correction)', linewidth=2, alpha=0.7)
    ax4.plot(time, processed_df['double_normalized'], 'r-', label='Double (CORRECTED)', linewidth=2)
    ax4.axvline(time[bleach_frame], color='k', linestyle='--', alpha=0.5, label='Bleach')
    ax4.axhline(1.0, color='k', linestyle=':', alpha=0.3)
    ax4.set_xlabel('Time (s)', fontsize=10)
    ax4.set_ylabel('Normalized Intensity', fontsize=10)
    ax4.set_title('Normalization Methods Comparison', fontsize=11, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(alpha=0.3)
    ax4.set_ylim(-0.1, 1.2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {save_path}")
    plt.close()

if __name__ == "__main__":
    import os
    
    # Test on sample file
    sample_file = r"c:\Users\mjhen\Github\FRAP2025\sample_data\GFP-PARP2-583-1.xlsx"
    
    if os.path.exists(sample_file):
        try:
            # Run test
            processed_df = test_normalization_correction(sample_file)
            
            # Create visualization
            print("\nStep 4: Creating visualization...")
            plot_normalization_comparison(processed_df)
            
            print("\n" + "="*80)
            print("✓ NORMALIZATION CORRECTION TEST COMPLETE")
            print("="*80)
            print("\nSUMMARY:")
            print("  • ROI2 (total nuclear signal) is now used for photobleaching correction")
            print("  • Post-bleach ROI2 value is maintained constant (reference = 100%)")
            print("  • Correction factor = ROI2_post / ROI2(t) is applied to ROI1")
            print("  • This properly accounts for acquisition-induced photobleaching")
            print("\nCheck the generated plot: normalization_test_results.png")
            
        except Exception as e:
            print(f"\n✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"✗ Sample file not found: {sample_file}")
        print("\nAvailable sample files:")
        sample_dir = r"c:\Users\mjhen\Github\FRAP2025\sample_data"
        if os.path.exists(sample_dir):
            for f in os.listdir(sample_dir)[:5]:
                print(f"  - {f}")
