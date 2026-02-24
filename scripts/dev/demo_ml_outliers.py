#!/usr/bin/env python3
"""
Demo script for ML-based FRAP outlier detection

This script demonstrates how the machine learning outlier detection
works with synthetic FRAP data and shows the various features.

Author: FRAP2025 Analysis Platform
Date: October 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import os

# Add the current directory to the path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from frap_ml_outliers import FRAPOutlierDetector
    print("✅ Successfully imported FRAPOutlierDetector")
except ImportError as e:
    print(f"❌ Failed to import FRAPOutlierDetector: {e}")
    print("Please ensure scikit-learn is installed: pip install scikit-learn")
    sys.exit(1)


def generate_synthetic_frap_data(n_curves=50, outlier_fraction=0.15):
    """Generate synthetic FRAP data with known outliers."""
    
    np.random.seed(42)  # For reproducible results
    
    time_points = np.linspace(0, 120, 61)  # 2 minutes, 2-second intervals
    curves_data = []
    labels = []  # 0 = normal, 1 = outlier
    
    n_outliers = int(n_curves * outlier_fraction)
    n_normal = n_curves - n_outliers
    
    print(f"Generating {n_normal} normal curves and {n_outliers} outlier curves...")
    
    # Generate normal curves
    for i in range(n_normal):
        # Normal FRAP parameters
        mobile_fraction = np.random.normal(0.75, 0.1)  # 75% ± 10%
        mobile_fraction = np.clip(mobile_fraction, 0.4, 1.0)
        
        half_time = np.random.lognormal(np.log(15), 0.3)  # ~15s half-time with log-normal distribution
        half_time = np.clip(half_time, 5, 60)
        
        rate_constant = np.log(2) / half_time
        
        # Generate curve with noise
        pre_bleach = 1.0 + np.random.normal(0, 0.02, 10)  # Pre-bleach baseline
        bleach_depth = 0.8 + np.random.normal(0, 0.05)  # ~80% bleaching
        bleach_depth = np.clip(bleach_depth, 0.6, 0.95)
        
        # Recovery curve (single exponential)
        recovery_time = time_points[10:] - time_points[10]  # Start recovery at timepoint 10
        recovery = (1 - mobile_fraction) + mobile_fraction * (1 - np.exp(-rate_constant * recovery_time))
        recovery = recovery * (1 - bleach_depth) + bleach_depth
        
        # Add realistic noise
        noise_level = 0.02
        recovery += np.random.normal(0, noise_level, len(recovery))
        
        # Combine pre-bleach and recovery
        intensity = np.concatenate([pre_bleach, recovery])
        
        # Create curve data
        curve_data = {
            'time': time_points,
            'intensity': intensity,
            'name': f'normal_curve_{i+1}',
            'features': {
                'mobile_fraction': mobile_fraction * 100,
                'half_time': half_time,
                'rate_constant': rate_constant,
                'r2': np.random.normal(0.95, 0.03)
            }
        }
        
        curves_data.append(curve_data)
        labels.append(0)  # Normal
    
    # Generate outlier curves
    for i in range(n_outliers):
        outlier_type = np.random.choice(['low_mobile', 'fast_recovery', 'noisy', 'incomplete_bleach', 'drift'])
        
        if outlier_type == 'low_mobile':
            # Very low mobile fraction
            mobile_fraction = np.random.uniform(0.1, 0.3)
            half_time = np.random.lognormal(np.log(20), 0.5)
            
        elif outlier_type == 'fast_recovery':
            # Unusually fast recovery
            mobile_fraction = np.random.normal(0.8, 0.05)
            half_time = np.random.uniform(1, 4)  # Very fast
            
        elif outlier_type == 'noisy':
            # Normal parameters but very noisy data
            mobile_fraction = np.random.normal(0.7, 0.1)
            half_time = np.random.lognormal(np.log(15), 0.3)
            
        elif outlier_type == 'incomplete_bleach':
            # Poor bleaching efficiency
            mobile_fraction = np.random.normal(0.75, 0.1)
            half_time = np.random.lognormal(np.log(15), 0.3)
            
        elif outlier_type == 'drift':
            # Baseline drift
            mobile_fraction = np.random.normal(0.7, 0.1)
            half_time = np.random.lognormal(np.log(15), 0.3)
        
        mobile_fraction = np.clip(mobile_fraction, 0.05, 1.0)
        half_time = np.clip(half_time, 0.5, 120)
        rate_constant = np.log(2) / half_time
        
        # Generate outlier curve
        pre_bleach = 1.0 + np.random.normal(0, 0.02, 10)
        
        if outlier_type == 'incomplete_bleach':
            bleach_depth = np.random.uniform(0.2, 0.4)  # Poor bleaching
        else:
            bleach_depth = 0.8 + np.random.normal(0, 0.05)
            bleach_depth = np.clip(bleach_depth, 0.6, 0.95)
        
        recovery_time = time_points[10:] - time_points[10]
        recovery = (1 - mobile_fraction) + mobile_fraction * (1 - np.exp(-rate_constant * recovery_time))
        recovery = recovery * (1 - bleach_depth) + bleach_depth
        
        # Add outlier-specific modifications
        if outlier_type == 'noisy':
            noise_level = 0.08  # Very noisy
            recovery += np.random.normal(0, noise_level, len(recovery))
        elif outlier_type == 'drift':
            # Add linear drift
            drift = np.linspace(0, 0.15, len(recovery))
            recovery += drift
            noise_level = 0.02
            recovery += np.random.normal(0, noise_level, len(recovery))
        else:
            noise_level = 0.02
            recovery += np.random.normal(0, noise_level, len(recovery))
        
        intensity = np.concatenate([pre_bleach, recovery])
        
        curve_data = {
            'time': time_points,
            'intensity': intensity,
            'name': f'outlier_curve_{i+1}_{outlier_type}',
            'features': {
                'mobile_fraction': mobile_fraction * 100,
                'half_time': half_time,
                'rate_constant': rate_constant,
                'r2': np.random.normal(0.85, 0.1) if outlier_type == 'noisy' else np.random.normal(0.95, 0.03)
            }
        }
        
        curves_data.append(curve_data)
        labels.append(1)  # Outlier
    
    return curves_data, np.array(labels)


def demo_feature_extraction():
    """Demonstrate feature extraction capabilities."""
    
    print("\n" + "="*60)
    print("🔬 FEATURE EXTRACTION DEMO")
    print("="*60)
    
    # Generate sample data
    curves_data, true_labels = generate_synthetic_frap_data(n_curves=20, outlier_fraction=0.2)
    
    # Initialize detector
    detector = FRAPOutlierDetector()
    
    # Extract features from all curves
    print("\nExtracting features from FRAP curves...")
    features_list = []
    curve_names = []
    
    for curve_data in curves_data:
        features = detector.extract_features(curve_data)
        if len(features) > 0:
            features_list.append(features)
            curve_names.append(curve_data['name'])
    
    features_matrix = np.array(features_list)
    print(f"✅ Extracted {features_matrix.shape[1]} features from {len(features_list)} curves")
    
    # Show feature categories
    print("\n📊 Feature Categories:")
    print("- Curve Shape Features (12): Recovery dynamics, smoothness, asymmetry")
    print("- Kinetic Parameters (12): Mobile fraction, rate constants, fit quality")
    print("- Quality Metrics (12): Signal-to-noise, bleaching efficiency, baseline stability")
    print("- Statistical Features (12): Distribution properties, autocorrelation, trends")
    
    return features_matrix, true_labels, curve_names


def demo_unsupervised_detection():
    """Demonstrate unsupervised outlier detection methods."""
    
    print("\n" + "="*60)
    print("🤖 UNSUPERVISED OUTLIER DETECTION DEMO")
    print("="*60)
    
    # Get data
    features_matrix, true_labels, curve_names = demo_feature_extraction()
    detector = FRAPOutlierDetector()
    
    # Test different methods
    methods = ['isolation_forest', 'one_class_svm', 'lof']
    
    for method in methods:
        print(f"\n🔍 Testing {method.replace('_', ' ').title()}...")
        
        results = detector.fit_unsupervised(
            features_matrix, 
            method=method, 
            contamination=0.2  # Expect 20% outliers
        )
        
        if 'error' in results:
            print(f"❌ Error: {results['error']}")
            continue
        
        predicted_outliers = results['outliers']
        n_detected = np.sum(predicted_outliers)
        
        # Calculate accuracy
        true_positives = np.sum((true_labels == 1) & predicted_outliers)
        false_positives = np.sum((true_labels == 0) & predicted_outliers)
        true_negatives = np.sum((true_labels == 0) & ~predicted_outliers)
        false_negatives = np.sum((true_labels == 1) & ~predicted_outliers)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        accuracy = (true_positives + true_negatives) / len(true_labels)
        
        print(f"   📈 Results: {n_detected} outliers detected ({n_detected/len(true_labels):.1%})")
        print(f"   🎯 Accuracy: {accuracy:.3f}")
        print(f"   🔍 Precision: {precision:.3f}")
        print(f"   📊 Recall: {recall:.3f}")
        
        # Show detected outliers
        outlier_indices = np.where(predicted_outliers)[0]
        print(f"   🚨 Detected outliers:")
        for idx in outlier_indices[:5]:  # Show first 5
            name = curve_names[idx]
            score = results['scores'][idx]
            is_true_outlier = "✅" if true_labels[idx] == 1 else "❌"
            print(f"      {is_true_outlier} {name} (score: {score:.3f})")
        
        if len(outlier_indices) > 5:
            print(f"      ... and {len(outlier_indices) - 5} more")


def demo_supervised_learning():
    """Demonstrate supervised learning with labeled data."""
    
    print("\n" + "="*60)
    print("🎯 SUPERVISED LEARNING DEMO")
    print("="*60)
    
    # Generate larger dataset for training
    curves_data, true_labels = generate_synthetic_frap_data(n_curves=100, outlier_fraction=0.15)
    
    detector = FRAPOutlierDetector()
    
    # Extract features
    print("\nExtracting features for supervised learning...")
    features_list = []
    curve_names = []
    
    for curve_data in curves_data:
        features = detector.extract_features(curve_data)
        if len(features) > 0:
            features_list.append(features)
            curve_names.append(curve_data['name'])
    
    features_matrix = np.array(features_list)
    
    print(f"✅ Dataset: {len(features_list)} curves, {np.sum(true_labels)} outliers ({np.mean(true_labels):.1%})")
    
    # Train supervised model
    print("\n🎯 Training Random Forest classifier...")
    results = detector.fit_supervised(features_matrix, true_labels, method='random_forest')
    
    if 'error' in results:
        print(f"❌ Error: {results['error']}")
        return
    
    # Show results
    print(f"✅ Model trained successfully!")
    print(f"📊 Cross-validation AUC: {results['mean_cv_score']:.3f}")
    print(f"🎯 Training AUC: {results['auc_score']:.3f}")
    
    # Show feature importance
    importance_results = detector.analyze_feature_importance('supervised_random_forest')
    
    if 'feature_importance' in importance_results:
        print(f"\n🔑 Top 5 Most Important Features:")
        top_features = importance_results['top_features'].head(5)
        for idx, row in top_features.iterrows():
            print(f"   {row['feature']}: {row['importance']:.3f}")
    
    # Show predictions vs true labels
    predictions = results['predictions']
    probabilities = results['probabilities']
    
    accuracy = np.mean(predictions == true_labels)
    print(f"\n📈 Training Performance:")
    print(f"   Accuracy: {accuracy:.3f}")
    
    # Show high-confidence predictions
    high_conf_outliers = np.where((probabilities > 0.8) & (predictions == 1))[0]
    print(f"\n🚨 High-confidence outlier predictions (p > 0.8):")
    for idx in high_conf_outliers[:5]:
        name = curve_names[idx]
        prob = probabilities[idx]
        is_correct = "✅" if true_labels[idx] == 1 else "❌"
        print(f"   {is_correct} {name} (p = {prob:.3f})")


def demo_practical_workflow():
    """Demonstrate a practical ML outlier detection workflow."""
    
    print("\n" + "="*60)
    print("🔄 PRACTICAL WORKFLOW DEMO")
    print("="*60)
    
    print("""
This demonstrates how you would use ML outlier detection in practice:

1. Start with automatic outlier detection (unsupervised)
2. Manually review and refine the selections 
3. Use manual selections to train a supervised model
4. Apply the trained model to new data automatically

Let's walk through this workflow...
    """)
    
    # Step 1: Initial automatic detection
    print("\n📋 Step 1: Initial Automatic Detection")
    curves_data, true_labels = generate_synthetic_frap_data(n_curves=50, outlier_fraction=0.2)
    detector = FRAPOutlierDetector()
    
    features_list = []
    for curve_data in curves_data:
        features = detector.extract_features(curve_data)
        if len(features) > 0:
            features_list.append(features)
    
    features_matrix = np.array(features_list)
    
    # Automatic detection
    auto_results = detector.fit_unsupervised(features_matrix, method='isolation_forest', contamination=0.2)
    auto_outliers = auto_results['outliers']
    
    print(f"🤖 Automatic detection found {np.sum(auto_outliers)} outliers")
    
    # Step 2: Simulate manual refinement
    print("\n🖱️ Step 2: Manual Refinement (Simulated)")
    print("In the real interface, you would:")
    print("- Review the automatically detected outliers")
    print("- Use checkboxes to include/exclude specific curves")
    print("- Look at the plots to make informed decisions")
    
    # Simulate some manual corrections (add some false negatives, remove some false positives)
    manual_labels = auto_outliers.copy()
    
    # Find false negatives and add some
    false_negatives = np.where((true_labels == 1) & ~auto_outliers)[0]
    if len(false_negatives) > 0:
        manual_labels[false_negatives[:2]] = True  # Add 2 missed outliers
    
    # Find false positives and remove some  
    false_positives = np.where((true_labels == 0) & auto_outliers)[0]
    if len(false_positives) > 0:
        manual_labels[false_positives[:1]] = False  # Remove 1 false positive
    
    print(f"✋ After manual review: {np.sum(manual_labels)} outliers selected")
    
    # Step 3: Train supervised model
    print("\n🎯 Step 3: Train Supervised Model")
    supervised_results = detector.fit_supervised(features_matrix, manual_labels.astype(int), method='random_forest')
    
    if 'error' not in supervised_results:
        print(f"✅ Supervised model trained! AUC: {supervised_results['auc_score']:.3f}")
        
        # Step 4: Apply to new data
        print("\n🔮 Step 4: Apply to New Data")
        new_curves, new_true_labels = generate_synthetic_frap_data(n_curves=20, outlier_fraction=0.25)
        
        new_features = []
        for curve_data in new_curves:
            features = detector.extract_features(curve_data)
            if len(features) > 0:
                new_features.append(features)
        
        new_features_matrix = np.array(new_features)
        
        # Predict on new data
        new_predictions = detector.predict_outliers(new_features_matrix, 'supervised_random_forest')
        
        if 'error' not in new_predictions:
            predicted_outliers = new_predictions['outliers']
            probabilities = new_predictions['probabilities']
            
            # Calculate performance on new data
            accuracy = np.mean(predicted_outliers == new_true_labels)
            print(f"🎯 Performance on new data: {accuracy:.3f} accuracy")
            print(f"📊 Predicted {np.sum(predicted_outliers)} outliers in {len(new_true_labels)} new curves")
            
            # Show high-confidence predictions
            high_conf = np.where(probabilities > 0.7)[0]
            print(f"🔍 High-confidence predictions: {len(high_conf)} curves")
    
    print(f"\n✨ Workflow complete! The trained model can now automatically detect outliers in new FRAP data.")


if __name__ == "__main__":
    print("🧬 FRAP ML Outlier Detection Demo")
    print("=" * 40)
    print("This demo shows how machine learning can automatically")
    print("detect atypical FRAP curves using extracted features.")
    print()
    
    try:
        # Run all demos
        demo_feature_extraction()
        demo_unsupervised_detection()
        demo_supervised_learning()
        demo_practical_workflow()
        
        print("\n" + "="*60)
        print("🎉 DEMO COMPLETE!")
        print("="*60)
        print("""
Key Benefits of ML-Based Outlier Detection:

🤖 AUTOMATED: Reduces manual effort in identifying problematic curves
🎯 ACCURATE: Learns from your selections to improve over time  
🔍 COMPREHENSIVE: Uses 48 features across multiple categories
⚡ FAST: Processes large datasets quickly
🧠 ADAPTIVE: Can be retrained as you collect more data
📊 TRANSPARENT: Shows feature importance and confidence scores

Next Steps:
1. Install scikit-learn if not already available
2. Use the interactive interface in the main FRAP application
3. Start with unsupervised detection for initial screening
4. Refine selections manually and train supervised models
5. Apply trained models to new datasets automatically

Happy analyzing! 🔬
        """)
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()