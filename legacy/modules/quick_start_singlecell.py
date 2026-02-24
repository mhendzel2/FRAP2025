#!/usr/bin/env python3
"""
Quick Start Example for FRAP Single-Cell Analysis

This script demonstrates the complete workflow from raw movie to statistical analysis.
Modify paths and parameters as needed for your data.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Import the single-cell API
from frap_singlecell_api import (
    track_movie,
    fit_cells,
    flag_motion,
    analyze,
    analyze_frap_movie
)
from frap_populations import detect_outliers_and_clusters
from frap_visualizations import save_all_figures
from frap_data_model import DataIO


def example_1_single_movie():
    """Example 1: Analyze a single FRAP movie"""
    print("\n=== Example 1: Single Movie Analysis ===\n")
    
    # Generate synthetic test data
    from test_synthetic import synth_movie
    
    movie, ground_truth = synth_movie(
        n_cells=15,
        T=100,
        drift_px=0.2,
        noise=0.02,
        random_state=42
    )
    
    time_points = np.arange(movie.shape[0]) * 0.5  # 0.5s intervals
    
    # Run complete analysis
    results = analyze_frap_movie(
        movie=movie,
        time_points=time_points,
        exp_id='exp001',
        movie_id='movie001',
        condition='control',
        output_dir='./output/example1',
        use_kalman=True,
        adapt_radius_flag=True,
        robust=True,
        try_2exp=True
    )
    
    print(f"✓ Tracked {results['n_cells']} cells")
    print(f"✓ Found {results['n_clusters']} populations")
    print(f"✓ Flagged {results['n_outliers']} outliers")
    print(f"✓ Results saved to ./output/example1/")
    
    return results


def example_2_custom_pipeline():
    """Example 2: Custom pipeline with step-by-step control"""
    print("\n=== Example 2: Custom Pipeline ===\n")
    
    # Generate test data
    from test_synthetic import synth_movie
    
    movie, gt = synth_movie(n_cells=20, T=80, random_state=123)
    time_points = np.arange(movie.shape[0])
    
    # Step 1: Track ROIs
    print("Step 1: Tracking ROIs...")
    roi_traces = track_movie(
        movie=movie,
        time_points=time_points,
        exp_id='exp002',
        movie_id='movie002',
        use_kalman=True,
        dt=1.0
    )
    print(f"  → Tracked {roi_traces['cell_id'].nunique()} cells")
    
    # Step 2: Flag motion artifacts
    print("Step 2: Flagging motion artifacts...")
    roi_traces = flag_motion(roi_traces, threshold_px=5.0)
    n_flagged = roi_traces['qc_motion'].sum()
    print(f"  → Flagged {n_flagged} frames")
    
    # Step 3: Fit recovery curves
    print("Step 3: Fitting recovery curves...")
    cell_features = fit_cells(
        roi_traces,
        pre_bleach_window=5,
        robust=True,
        try_2exp=True,
        n_jobs=-1
    )
    n_passed = cell_features['bleach_qc'].sum()
    print(f"  → {n_passed}/{len(cell_features)} cells passed QC")
    
    # Step 4: Population analysis
    print("Step 4: Population analysis...")
    cell_features = detect_outliers_and_clusters(
        cell_features,
        contamination=0.07,
        max_k=6
    )
    n_clusters = len(cell_features['cluster'].unique()) - (1 if -1 in cell_features['cluster'].values else 0)
    print(f"  → Found {n_clusters} clusters, {cell_features['outlier'].sum()} outliers")
    
    # Step 5: Save results
    print("Step 5: Saving results...")
    DataIO.save_tables(
        roi_traces,
        cell_features,
        './output/example2',
        format='both'
    )
    print("  → Saved to ./output/example2/")
    
    # Step 6: Generate figures
    print("Step 6: Generating figures...")
    cell_features['condition'] = 'test'
    saved_figs = save_all_figures(
        './output/example2/figures',
        roi_traces,
        cell_features
    )
    print(f"  → Saved {len(saved_figs)} figures")
    
    return roi_traces, cell_features


def example_3_multi_condition():
    """Example 3: Multi-condition statistical comparison"""
    print("\n=== Example 3: Multi-Condition Analysis ===\n")
    
    # Generate synthetic dataset with two conditions
    from test_synthetic import synth_multi_movie_dataset
    
    movies, ground_truth = synth_multi_movie_dataset(
        n_movies=6,
        n_cells_per_movie=15,
        T=100,
        conditions=['control', 'control', 'control', 'treatment', 'treatment', 'treatment'],
        random_state=42
    )
    
    # Process each movie
    all_features = []
    
    for i, (movie, condition) in enumerate(zip(movies, ['control']*3 + ['treatment']*3)):
        print(f"Processing movie {i+1}/6 ({condition})...")
        
        time_points = np.arange(movie.shape[0]) * 0.5
        
        # Quick analysis
        roi_traces = track_movie(movie, time_points, f'exp{i//2}', f'movie{i}')
        cell_features = fit_cells(roi_traces)
        cell_features['condition'] = condition
        cell_features['exp_id'] = f'exp{i//2}'
        
        all_features.append(cell_features)
    
    # Combine all data
    combined_features = pd.concat(all_features, ignore_index=True)
    
    # Population analysis on combined data
    print("\nPopulation analysis on combined dataset...")
    combined_features = detect_outliers_and_clusters(combined_features)
    
    # Statistical comparison
    print("\nStatistical analysis...")
    stats = analyze(
        combined_features,
        group_col='condition',
        batch_col='exp_id',
        params=['mobile_frac', 'k', 't_half', 'pre_bleach'],
        n_bootstrap=500  # Reduced for speed
    )
    
    print("\n=== Statistical Results ===")
    if 'comparisons' in stats:
        results_df = stats['comparisons']
        print(results_df[['param', 'comparison', 'beta', 'p', 'q', 'hedges_g']])
        
        # Highlight significant results
        sig_results = results_df[results_df['significant']]
        if len(sig_results) > 0:
            print(f"\n✓ Found {len(sig_results)} significant differences (FDR-adjusted)")
        else:
            print("\n○ No significant differences detected")
    
    # Save combined results
    DataIO.save_tables(
        pd.DataFrame(),  # Not saving traces to save space
        combined_features,
        './output/example3',
        format='parquet'
    )
    
    return combined_features, stats


def example_4_visualization():
    """Example 4: Comprehensive visualization"""
    print("\n=== Example 4: Visualization ===\n")
    
    from test_synthetic import synth_movie
    from frap_visualizations import (
        plot_spaghetti,
        plot_heatmap,
        plot_pairplot,
        plot_qc_dashboard
    )
    import matplotlib.pyplot as plt
    
    # Generate data
    movie, gt = synth_movie(n_cells=20, T=100, random_state=99)
    time_points = np.arange(movie.shape[0])
    
    # Analyze
    roi_traces = track_movie(movie, time_points, 'exp', 'mov')
    cell_features = fit_cells(roi_traces)
    cell_features['condition'] = 'test'
    cell_features = detect_outliers_and_clusters(cell_features)
    
    # Generate plots
    print("Generating visualizations...")
    
    fig1 = plot_spaghetti(roi_traces, cell_features, 'test')
    fig1.savefig('./output/example4/spaghetti.png', dpi=300, bbox_inches='tight')
    print("  ✓ Spaghetti plot")
    
    fig2 = plot_heatmap(roi_traces, 'test')
    fig2.savefig('./output/example4/heatmap.png', dpi=300, bbox_inches='tight')
    print("  ✓ Heatmap")
    
    fig3 = plot_pairplot(cell_features)
    fig3.savefig('./output/example4/pairplot.png', dpi=300, bbox_inches='tight')
    print("  ✓ Pair plot")
    
    fig4 = plot_qc_dashboard(roi_traces, cell_features)
    fig4.savefig('./output/example4/qc_dashboard.png', dpi=300, bbox_inches='tight')
    print("  ✓ QC dashboard")
    
    plt.close('all')
    print("\n✓ All figures saved to ./output/example4/")


def run_all_examples():
    """Run all examples"""
    import os
    os.makedirs('./output', exist_ok=True)
    
    print("=" * 60)
    print("FRAP Single-Cell Analysis - Quick Start Examples")
    print("=" * 60)
    
    try:
        # Example 1
        results1 = example_1_single_movie()
        
        # Example 2
        traces2, features2 = example_2_custom_pipeline()
        
        # Example 3
        features3, stats3 = example_3_multi_condition()
        
        # Example 4
        example_4_visualization()
        
        print("\n" + "=" * 60)
        print("✓ ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("\nResults saved to ./output/")
        print("\nNext steps:")
        print("1. Examine the output files")
        print("2. Try with your own data")
        print("3. Adjust parameters as needed")
        print("4. See README_SINGLECELL.md for more details")
        
    except Exception as e:
        print(f"\n✗ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    # Run all examples
    run_all_examples()
    
    # Or run individual examples:
    # example_1_single_movie()
    # example_2_custom_pipeline()
    # example_3_multi_condition()
    # example_4_visualization()
