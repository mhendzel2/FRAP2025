"""
Test script to verify mobile fraction calculation fixes with sample PARGi E558A data
"""
import os
import sys
import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from frap_core import FRAPAnalysisCore

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

def test_sample_data():
    """Test the fixed mobile fraction calculation on sample data"""
    
    sample_dir = Path("sample_data/265 PARGi E558A")
    
    if not sample_dir.exists():
        print(f"ERROR: Sample directory not found: {sample_dir}")
        return
    
    print("=" * 80)
    print("TESTING FIXED MOBILE FRACTION CALCULATION")
    print("Sample Data: PARGi E558A dataset")
    print("=" * 80)
    print()
    
    # Get all Excel files
    excel_files = sorted(list(sample_dir.glob("*.xlsx")))
    
    if not excel_files:
        print(f"ERROR: No Excel files found in {sample_dir}")
        return
    
    print(f"Found {len(excel_files)} files to process")
    print()
    
    # Process a subset for testing
    test_files = excel_files[:10]  # Test first 10 files
    
    results = []
    
    print("Processing files...")
    print("-" * 80)
    
    for file_path in test_files:
        try:
            # Load data using FRAPAnalysisCore
            raw_df = FRAPAnalysisCore.load_data(str(file_path))
            if raw_df is None or raw_df.empty:
                print(f"SKIP {file_path.name}: Failed to load")
                continue
            
            # Preprocess
            processed_df = FRAPAnalysisCore.preprocess(raw_df)
            if processed_df is None or 'normalized' not in processed_df.columns:
                print(f"SKIP {file_path.name}: Preprocessing failed")
                continue
            
            # Fit models
            time = processed_df['time'].values
            intensity = processed_df['normalized'].values
            
            fits = FRAPAnalysisCore.fit_all_models(time, intensity)
            
            if not fits:
                print(f"FAIL {file_path.name}: No models fit successfully")
                results.append({
                    'file': file_path.name,
                    'status': 'NO_FIT',
                    'model': 'none',
                    'mobile_fraction': np.nan,
                    'r2': np.nan
                })
                continue
            
            # Get best fit (by R²)
            best_fit = max(fits, key=lambda x: x.get('r2', 0))
            model = best_fit.get('model', 'unknown')
            r2 = best_fit.get('r2', 0)
            
            # Extract mobile fraction
            features = FRAPAnalysisCore.extract_clustering_features(best_fit)
            if features:
                mobile = features.get('mobile_fraction', np.nan)
                flags = features.get('mobile_fraction_flags', [])
                
                # Determine status
                if 'anomalous' in model:
                    status = 'ANOMALOUS'
                elif mobile > 100:
                    status = 'OVER_RECOVERY'
                elif mobile > 95:
                    status = 'HIGH'
                elif np.isfinite(mobile):
                    status = 'OK'
                else:
                    status = 'INVALID'
                
                results.append({
                    'file': file_path.name,
                    'status': status,
                    'model': model,
                    'mobile_fraction': mobile,
                    'r2': r2,
                    'flags': ','.join(flags) if flags else ''
                })
                
                # Print summary
                status_icon = "✓" if status == 'OK' else "⚠" if status in ['HIGH', 'ANOMALOUS'] else "✗"
                print(f"{status_icon} {file_path.name:25} | {model:10} | Mobile: {mobile:6.1f}% | R²: {r2:.3f} | {status}")
            else:
                print(f"✗ {file_path.name:25} | Feature extraction failed")
                results.append({
                    'file': file_path.name,
                    'status': 'NO_FEATURES',
                    'model': model,
                    'mobile_fraction': np.nan,
                    'r2': r2
                })
                
        except Exception as e:
            print(f"✗ {file_path.name:25} | ERROR: {str(e)[:40]}")
            results.append({
                'file': file_path.name,
                'status': 'ERROR',
                'model': 'error',
                'mobile_fraction': np.nan,
                'r2': np.nan
            })
    
    print()
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    df = pd.DataFrame(results)
    
    # Count by status
    print("\nFit Status:")
    status_counts = df['status'].value_counts()
    for status, count in status_counts.items():
        print(f"  {status:15} : {count:2} files")
    
    # Count by model
    print("\nModel Types:")
    model_counts = df['model'].value_counts()
    for model, count in model_counts.items():
        print(f"  {model:15} : {count:2} files")
    
    # Mobile fraction statistics (excluding invalid values)
    valid_mobiles = df[df['mobile_fraction'].notna() & np.isfinite(df['mobile_fraction'])]['mobile_fraction']
    
    if len(valid_mobiles) > 0:
        print(f"\nMobile Fraction Statistics (n={len(valid_mobiles)}):")
        print(f"  Mean:   {valid_mobiles.mean():.1f}%")
        print(f"  Median: {valid_mobiles.median():.1f}%")
        print(f"  Min:    {valid_mobiles.min():.1f}%")
        print(f"  Max:    {valid_mobiles.max():.1f}%")
        print(f"  Std:    {valid_mobiles.std():.1f}%")
        
        # Check for over-recovery
        over_recovery = (valid_mobiles > 100).sum()
        if over_recovery > 0:
            print(f"\n  ⚠ WARNING: {over_recovery} curves with mobile fraction > 100%")
            print(f"    Max over-recovery: {valid_mobiles.max():.1f}%")
        else:
            print(f"\n  ✓ No over-recovery detected (all mobile fractions ≤ 100%)")
    
    print()
    print("=" * 80)
    print("FIX VALIDATION")
    print("=" * 80)
    
    # Check if fixes are working
    exponential_fits = len(df[df['model'].isin(['single', 'double', 'triple'])])
    anomalous_fits = len(df[df['model'].str.contains('anomalous', na=False)])
    extreme_over_recovery = len(df[df['mobile_fraction'] > 200])
    
    print(f"\n✓ Exponential models succeeded: {exponential_fits}/{len(df)} files")
    print(f"  (Previously: many would fail and default to anomalous)")
    
    print(f"\n✓ Anomalous diffusion models: {anomalous_fits}/{len(df)} files")
    print(f"  (Expected: only for genuinely anomalous data)")
    
    if extreme_over_recovery > 0:
        print(f"\n✗ PROBLEM: {extreme_over_recovery} files still show extreme over-recovery (>200%)")
        print("  This indicates the fix may not be fully working")
    else:
        print(f"\n✓ No extreme over-recovery detected (was seeing >5000% before fix)")
    
    print()
    
    # Save results
    output_file = "test_results_sample_data.csv"
    df.to_csv(output_file, index=False)
    print(f"Detailed results saved to: {output_file}")
    print()

if __name__ == "__main__":
    test_sample_data()
