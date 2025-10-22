"""
Corrected FRAP Normalization Functions

This module implements proper FRAP normalization where 1.0 represents the theoretical 
maximum recovery accounting for total fluorescence loss during photobleaching.

Standard FRAP normalization formula:
I_norm(t) = [I(t) - I_bg] / [I_pre - I_bg] 

Proper FRAP normalization should account for:
1. Background subtraction
2. Photobleaching losses during imaging
3. Theoretical maximum recovery based on mobile fraction
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import logging

def calculate_proper_frap_normalization(df: pd.DataFrame, 
                                      use_reference_correction: bool = True,
                                      pre_bleach_frames: int = 5,
                                      post_bleach_frames: int = 3) -> pd.DataFrame:
    """
    Calculate proper FRAP normalization where 1.0 represents theoretical maximum recovery.
    
    This accounts for:
    1. Background subtraction
    2. Photobleaching during imaging (via reference ROI)
    3. Proper scaling so 1.0 = complete mobile recovery
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw FRAP data with columns: time, roi1, roi2, roi3
        roi1 = bleached region
        roi2 = reference region (unbleached)
        roi3 = background region
    use_reference_correction : bool
        If True, correct for imaging-induced photobleaching using ROI2
    pre_bleach_frames : int
        Number of frames to average for pre-bleach intensity
    post_bleach_frames : int
        Number of frames after bleaching to determine bleach depth
        
    Returns
    -------
    pd.DataFrame
        DataFrame with additional normalization columns
    """
    result_df = df.copy()
    
    # Step 1: Background subtraction
    roi1_bg_corrected = df['roi1'] - df['roi3']
    roi2_bg_corrected = df['roi2'] - df['roi3']
    
    # Step 2: Find bleaching timepoint (minimum intensity in ROI1)
    bleach_idx = roi1_bg_corrected.idxmin()
    
    # Step 3: Calculate pre-bleach intensities
    pre_bleach_end = min(bleach_idx, pre_bleach_frames)
    if pre_bleach_end < 1:
        raise ValueError("Insufficient pre-bleach data")
        
    I_pre_roi1 = roi1_bg_corrected.iloc[:pre_bleach_end].mean()
    I_pre_roi2 = roi2_bg_corrected.iloc[:pre_bleach_end].mean()
    
    # Step 4: Calculate immediate post-bleach intensity (bleach depth)
    post_bleach_start = bleach_idx + 1
    post_bleach_end = min(len(df), post_bleach_start + post_bleach_frames)
    
    if post_bleach_end <= post_bleach_start:
        # Use the bleach frame itself if no post-bleach data
        I_post_roi1 = roi1_bg_corrected.iloc[bleach_idx]
    else:
        I_post_roi1 = roi1_bg_corrected.iloc[post_bleach_start:post_bleach_end].mean()
    
    # Step 5: Calculate reference correction factor (accounts for imaging photobleaching)
    if use_reference_correction and I_pre_roi2 > 0:
        # Reference intensity at each timepoint normalized to pre-bleach
        ref_correction = roi2_bg_corrected / I_pre_roi2
        # Replace any invalid values
        ref_correction = ref_correction.fillna(1.0)
        ref_correction = ref_correction.clip(lower=0.1, upper=2.0)  # Reasonable bounds
    else:
        ref_correction = pd.Series(1.0, index=df.index)
    
    # Step 6: Apply reference correction to ROI1
    roi1_ref_corrected = roi1_bg_corrected / ref_correction
    
    # Step 7: Calculate proper normalization
    # Key insight: The denominator should be the PRE-BLEACH intensity, not post-bleach
    # This makes 1.0 represent the theoretical maximum recovery
    
    if I_pre_roi1 <= 0:
        raise ValueError("Invalid pre-bleach intensity")
        
    # Standard normalization: I_norm(t) = I(t) / I_pre
    frap_normalized = roi1_ref_corrected / I_pre_roi1
    
    # Step 8: Calculate additional metrics
    bleach_depth = (I_pre_roi1 - I_post_roi1) / I_pre_roi1  # Fraction of intensity lost
    theoretical_max = 1.0  # By definition with proper normalization
    
    # Add results to dataframe
    result_df['roi1_bg_corrected'] = roi1_bg_corrected
    result_df['roi2_bg_corrected'] = roi2_bg_corrected
    result_df['reference_correction'] = ref_correction
    result_df['roi1_ref_corrected'] = roi1_ref_corrected
    result_df['frap_normalized'] = frap_normalized
    result_df['bleach_depth'] = bleach_depth
    result_df['theoretical_max'] = theoretical_max
    
    # Add metadata
    result_df.attrs['pre_bleach_intensity'] = I_pre_roi1
    result_df.attrs['post_bleach_intensity'] = I_post_roi1
    result_df.attrs['bleach_depth'] = bleach_depth
    result_df.attrs['bleach_frame'] = bleach_idx
    
    logging.info(f"FRAP normalization completed:")
    logging.info(f"  Pre-bleach intensity: {I_pre_roi1:.3f}")
    logging.info(f"  Post-bleach intensity: {I_post_roi1:.3f}")
    logging.info(f"  Bleach depth: {bleach_depth:.1%}")
    logging.info(f"  Theoretical maximum recovery: {theoretical_max:.3f}")
    
    return result_df

def validate_frap_normalization(df: pd.DataFrame) -> dict:
    """
    Validate that FRAP normalization is correct.
    
    Parameters
    ----------
    df : pd.DataFrame
        Normalized FRAP data
        
    Returns
    -------
    dict
        Validation results
    """
    results = {
        'valid': True,
        'warnings': [],
        'metrics': {}
    }
    
    if 'frap_normalized' not in df.columns:
        results['valid'] = False
        results['warnings'].append("No normalized FRAP data found")
        return results
    
    # Check that pre-bleach values are close to 1.0
    bleach_idx = df.attrs.get('bleach_frame', len(df) // 2)
    pre_bleach_data = df['frap_normalized'].iloc[:min(5, bleach_idx)]
    pre_bleach_mean = pre_bleach_data.mean()
    
    if abs(pre_bleach_mean - 1.0) > 0.1:
        results['warnings'].append(f"Pre-bleach intensity not normalized to 1.0 (found {pre_bleach_mean:.3f})")
    
    # Check for reasonable recovery range
    recovery_max = df['frap_normalized'].max()
    recovery_min = df['frap_normalized'].min()
    
    if recovery_max > 1.5:
        results['warnings'].append(f"Recovery exceeds theoretical maximum (max = {recovery_max:.3f})")
    
    if recovery_min < 0:
        results['warnings'].append(f"Negative intensities found (min = {recovery_min:.3f})")
    
    # Store metrics
    results['metrics'] = {
        'pre_bleach_mean': pre_bleach_mean,
        'recovery_max': recovery_max,
        'recovery_min': recovery_min,
        'bleach_depth': df.attrs.get('bleach_depth', np.nan)
    }
    
    return results

# Example of how this should be integrated into the main analysis
def demonstrate_proper_normalization():
    """
    Example showing the difference between current and proper normalization.
    """
    print("FRAP Normalization Comparison")
    print("=" * 50)
    print()
    print("CURRENT NORMALIZATION (INCORRECT):")
    print("  Simple: (ROI1 - ROI3) / pre_bleach_ROI1")
    print("  Issues: 1.0 represents pre-bleach intensity")
    print("          Does not account for total fluorescence loss")
    print("          Mobile fraction calculation is incorrect")
    print()
    print("PROPER NORMALIZATION (CORRECTED):")
    print("  Formula: [(ROI1 - ROI3) / ref_correction] / pre_bleach_ROI1")
    print("  Where ref_correction accounts for imaging photobleaching")
    print()
    print("KEY DIFFERENCES:")
    print("  ✅ 1.0 represents theoretical maximum recovery")
    print("  ✅ Pre-bleach intensity normalized to 1.0")
    print("  ✅ Accounts for imaging-induced photobleaching")
    print("  ✅ Mobile fraction = (plateau - post_bleach) / (1.0 - post_bleach)")
    print("  ✅ Proper comparison between experiments")
    print()
    print("BIOLOGICAL INTERPRETATION:")
    print("  - Recovery plateau < 1.0 indicates immobile fraction")
    print("  - Recovery plateau = 1.0 indicates all molecules are mobile")
    print("  - Recovery > 1.0 indicates experimental artifacts")

if __name__ == "__main__":
    demonstrate_proper_normalization()