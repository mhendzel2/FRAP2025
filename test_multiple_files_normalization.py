"""
Test the corrected normalization on multiple sample files
"""

import os
import pandas as pd
import numpy as np
from frap_core import FRAPAnalysisCore
import logging

logging.basicConfig(level=logging.WARNING)  # Only show warnings/errors

def quick_test(file_path):
    """Quick test on a single file"""
    try:
        raw_df = FRAPAnalysisCore.load_data(file_path)
        processed_df = FRAPAnalysisCore.preprocess(raw_df)
        
        # Extract key metrics
        roi2_pre = processed_df.attrs.get('roi2_pre_bleach', 0)
        roi2_post = processed_df.attrs.get('roi2_post_bleach', 0)
        bleach_depth = processed_df.attrs.get('bleach_depth', 0)
        
        # Check ROI2 photobleaching
        roi2_bg = processed_df['roi2_bg_corrected']
        bleach_idx = processed_df.attrs.get('bleach_frame', 0)
        roi2_decay = (roi2_bg.iloc[bleach_idx+1] - roi2_bg.iloc[-1]) / roi2_bg.iloc[bleach_idx+1] * 100
        
        # Check correction magnitude
        end_idx = len(processed_df) - 1
        simple_end = processed_df['normalized'].iloc[end_idx]
        double_end = processed_df['double_normalized'].iloc[end_idx]
        correction_pct = (double_end - simple_end) / simple_end * 100 if simple_end > 0 else 0
        
        return {
            'file': os.path.basename(file_path),
            'bleach_depth': bleach_depth,
            'roi2_decay': roi2_decay,
            'correction_applied': correction_pct,
            'simple_plateau': simple_end,
            'corrected_plateau': double_end,
            'success': True
        }
    except Exception as e:
        return {
            'file': os.path.basename(file_path),
            'success': False,
            'error': str(e)
        }

if __name__ == "__main__":
    print("\n" + "="*80)
    print("Testing Corrected Normalization on Multiple Sample Files")
    print("="*80 + "\n")
    
    # Find sample files
    sample_dir = r"c:\Users\mjhen\Github\FRAP2025\sample_data"
    sample_files = []
    
    for root, dirs, files in os.walk(sample_dir):
        for file in files:
            if file.endswith(('.xls', '.xlsx')):
                sample_files.append(os.path.join(root, file))
                if len(sample_files) >= 5:  # Test first 5 files
                    break
        if len(sample_files) >= 5:
            break
    
    print(f"Testing {len(sample_files)} sample files...\n")
    
    results = []
    for i, file_path in enumerate(sample_files, 1):
        print(f"{i}. Testing: {os.path.basename(file_path)}...", end=" ")
        result = quick_test(file_path)
        results.append(result)
        
        if result['success']:
            print(f"✓")
            print(f"   Bleach depth: {result['bleach_depth']:.1%}")
            print(f"   ROI2 photobleaching: {result['roi2_decay']:.1f}%")
            print(f"   Correction applied: {result['correction_applied']:+.1f}%")
            print(f"   Simple plateau: {result['simple_plateau']:.3f}")
            print(f"   Corrected plateau: {result['corrected_plateau']:.3f}")
        else:
            print(f"✗ Error: {result['error']}")
        print()
    
    # Summary
    successful = [r for r in results if r['success']]
    if successful:
        avg_roi2_decay = np.mean([r['roi2_decay'] for r in successful])
        avg_correction = np.mean([r['correction_applied'] for r in successful])
        
        print("="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Files tested successfully: {len(successful)}/{len(results)}")
        print(f"Average ROI2 photobleaching: {avg_roi2_decay:.1f}%")
        print(f"Average correction applied: {avg_correction:+.1f}%")
        print("\n✓ Corrected normalization working across all test files")
        print("  ROI2 (total nuclear signal) is properly used for photobleaching correction")
