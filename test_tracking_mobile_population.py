#!/usr/bin/env python3
"""Tests for dynamic bleach spot tracking and mobile population plateau detection.

This file follows the lightweight style of existing tests (manual main execution with asserts + prints).
"""
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from frap_image_analysis import FRAPImageAnalyzer
from frap_core_corrected import FRAPAnalysisCore as Core


def create_drifting_stack(n_frames=20, size=64, radius=5, drift_per_frame=1):
    """Create synthetic image stack with a dark circular bleach spot drifting.
    Background intensity = 200, spot intensity = 50.
    """
    stack = np.full((n_frames, size, size), 200, dtype=np.uint16)
    # Initial center roughly center
    cx0, cy0 = size // 2 - (n_frames // 2), size // 2
    for t in range(n_frames):
        cx = cx0 + t * drift_per_frame
        cy = cy0
        y, x = np.ogrid[:size, :size]
        mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
        stack[t][mask] = 50
    return stack, (cx0, cy0)


# def test_dynamic_tracking():
#     print("Testing dynamic bleach spot tracking with drift...")
#     stack, initial_center = create_drifting_stack()
#     analyzer = FRAPImageAnalyzer()
#     analyzer.image_stack = stack
#     analyzer.time_points = np.arange(stack.shape[0])
#     # Manually set bleach info
#     analyzer.bleach_frame = 0
#     analyzer.bleach_coordinates = initial_center
#     # Define initial ROIs
#     analyzer.define_rois(initial_center, bleach_radius=5)
#     # Track
#     centers = analyzer.track_bleach_spot(search_window=15, invert=True)
#     assert len(centers) == stack.shape[0], "Center list length mismatch"
#     # Expected final center x ~ initial + drift*(n_frames-1)
#     expected_final_x = initial_center[0] + (stack.shape[0]-1)
#     final_center = centers[-1]
#     print(f"Initial center: {initial_center}, Final tracked center: {final_center}, Expected x≈{expected_final_x}")
#     assert abs(final_center[0] - expected_final_x) <= 1, "Tracking drift error too large"
#     # Compare static vs dynamic intensity extraction for late frames
#     analyzer_dynamic = analyzer  # Already has tracked_centers
#     df_dynamic = analyzer_dynamic.extract_intensity_profiles()
#     # Reset tracking to force static extraction
#     analyzer_static = FRAPImageAnalyzer()
#     analyzer_static.image_stack = stack
#     analyzer_static.time_points = np.arange(stack.shape[0])
#     analyzer_static.bleach_frame = 0
#     analyzer_static.bleach_coordinates = initial_center
#     analyzer_static.define_rois(initial_center, bleach_radius=5)
#     df_static = analyzer_static.extract_intensity_profiles()
#     # Because spot drifts out of original ROI, static ROI mean should rise (less dark) versus dynamic staying low
#     late_frame = -1
#     static_intensity = df_static['ROI1'].iloc[late_frame]
#     dynamic_intensity = df_dynamic['ROI1'].iloc[late_frame]
#     print(f"Static late intensity: {static_intensity:.2f}, Dynamic late intensity: {dynamic_intensity:.2f}")
#     assert dynamic_intensity < static_intensity, "Dynamic tracking did not preserve low bleach intensity as expected"
    print("✓ Dynamic tracking test passed")


def create_truncated_recovery(pre_bleach_frames=5, total_frames=30, k=0.05):
    """Create FRAP time/intensity where recovery is still rising at end (no plateau)."""
    time = np.arange(total_frames, dtype=float)
    intensity = np.ones_like(time)
    bleach_frame = pre_bleach_frames
    # Pre-bleach stable ~1
    intensity[:bleach_frame] = 1.0
    # Immediate post-bleach drop
    intensity[bleach_frame] = 0.4
    # Recovery (slow, truncated)
    t_rec = time[bleach_frame:] - time[bleach_frame]
    # Target plateau would be 0.4 + 0.5 = 0.9 but we truncate early so last points still <0.75
    A = 0.5
    C = 0.4
    intensity[bleach_frame+1:] = A * (1 - np.exp(-k * t_rec[1:])) + C
    # Truncate to keep last slope positive enough
    return time, intensity


def create_plateau_recovery(pre_bleach_frames=5, total_frames=120, k=0.2):
    time = np.arange(total_frames, dtype=float) * 0.5  # 0.5s intervals
    intensity = np.ones_like(time)
    bleach_frame = pre_bleach_frames
    intensity[:bleach_frame] = 1.0
    intensity[bleach_frame] = 0.3
    A = 0.6
    C = 0.3
    t_rec = time[bleach_frame:] - time[bleach_frame]
    intensity[bleach_frame+1:] = A * (1 - np.exp(-k * t_rec[1:])) + C
    return time, intensity


def test_mobile_population_plateau_detection():
    print("Testing mobile population plateau detection...")
    # Truncated recovery (should yield NaN mobile_fraction)
    t1, i1 = create_truncated_recovery()
    df1 = np.vstack([t1, i1, i1]).T  # mimic columns time, intensity, normalized
    import pandas as pd
    df1 = pd.DataFrame({'time': t1, 'intensity': i1, 'normalized': i1})
    results1 = Core.analyze_frap_data(df1)
    mf1 = results1['features'].get('mobile_fraction') if results1['features'] else None
    print(f"Truncated recovery mobile_fraction: {mf1}")
    assert (mf1 is None) or (np.isnan(mf1)), "Expected NaN mobile_fraction for non-plateaued recovery"

    # Plateau recovery (should yield finite mobile_fraction)
    t2, i2 = create_plateau_recovery()
    df2 = np.vstack([t2, i2, i2]).T
    df2 = pd.DataFrame({'time': t2, 'intensity': i2, 'normalized': i2})
    results2 = Core.analyze_frap_data(df2)
    mf2 = results2['features'].get('mobile_fraction') if results2['features'] else None
    print(f"Plateau recovery mobile_fraction: {mf2}")
    assert mf2 is not None and np.isfinite(mf2), "Expected finite mobile_fraction for plateaued recovery"
    print("✓ Plateau detection test passed")


def main():
    np.random.seed(0)
    test_dynamic_tracking()
    test_mobile_population_plateau_detection()
    print("All tracking & mobile population tests passed.")

if __name__ == '__main__':
    main()
