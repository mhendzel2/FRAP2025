#!/usr/bin/env python3
"""
Machine Learning Outlier Detection for FRAP Data

This module implements ML-based approaches for automatically detecting
atypical/outlier FRAP recovery curves using features extracted from
curve characteristics and user-labeled training data.

Author: FRAP2025 Analysis Platform
Date: October 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# ML imports with fallbacks
try:
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.svm import OneClassSVM
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    from sklearn.cluster import DBSCAN
    from sklearn.decomposition import PCA
    from sklearn.neighbors import LocalOutlierFactor
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class FRAPOutlierDetector:
    """
    Machine Learning-based outlier detection for FRAP recovery curves.
    
    This class provides multiple approaches:
    1. Unsupervised outlier detection (Isolation Forest, One-Class SVM, LOF)
    2. Supervised classification using user-labeled data
    3. Feature-based anomaly detection
    4. Ensemble methods combining multiple approaches
    """
    
    def __init__(self):
        """Initialize the outlier detector."""
        self.feature_extractors = {
            'curve_shape': self._extract_curve_shape_features,
            'kinetic_params': self._extract_kinetic_features,
            'quality_metrics': self._extract_quality_features,
            'statistical': self._extract_statistical_features
        }
        
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.training_history = []
        
    def extract_features(self, frap_data: Dict) -> np.ndarray:
        """
        Extract comprehensive features from FRAP curve data.
        
        Parameters:
        -----------
        frap_data : Dict
            Dictionary containing time, intensity, and analysis results
            
        Returns:
        --------
        np.ndarray
            Feature vector for the curve
        """
        
        if not frap_data or 'time' not in frap_data or 'intensity' not in frap_data:
            return np.array([])
        
        time = np.array(frap_data['time'])
        intensity = np.array(frap_data['intensity'])
        features = frap_data.get('features', {})
        
        all_features = []
        
        # Extract features from each category
        for category, extractor in self.feature_extractors.items():
            try:
                category_features = extractor(time, intensity, features)
                all_features.extend(category_features)
            except Exception as e:
                # If extraction fails, fill with NaN
                all_features.extend([np.nan] * 10)  # Assume 10 features per category
        
        return np.array(all_features)
    
    def _extract_curve_shape_features(self, time: np.ndarray, intensity: np.ndarray, features: Dict) -> List[float]:
        """Extract features related to curve shape and recovery dynamics."""
        
        try:
            # Basic curve characteristics
            min_intensity = np.min(intensity)
            max_intensity = np.max(intensity)
            intensity_range = max_intensity - min_intensity
            
            # Find bleaching point
            bleach_idx = np.argmin(intensity)
            
            # Pre-bleach characteristics
            if bleach_idx > 0:
                pre_bleach_mean = np.mean(intensity[:bleach_idx])
                pre_bleach_std = np.std(intensity[:bleach_idx])
                pre_bleach_trend = np.polyfit(time[:bleach_idx], intensity[:bleach_idx], 1)[0] if bleach_idx > 1 else 0
            else:
                pre_bleach_mean = intensity[0]
                pre_bleach_std = 0
                pre_bleach_trend = 0
            
            # Recovery characteristics
            if bleach_idx < len(intensity) - 1:
                recovery_data = intensity[bleach_idx:]
                recovery_time = time[bleach_idx:] - time[bleach_idx]
                
                # Recovery rate (early slope)
                if len(recovery_data) > 5:
                    early_recovery = recovery_data[:min(5, len(recovery_data))]
                    early_time = recovery_time[:len(early_recovery)]
                    recovery_rate = np.polyfit(early_time, early_recovery, 1)[0] if len(early_time) > 1 else 0
                else:
                    recovery_rate = 0
                
                # Final recovery level
                final_recovery = np.mean(recovery_data[-5:]) if len(recovery_data) >= 5 else recovery_data[-1]
                
                # Recovery completeness
                recovery_fraction = (final_recovery - min_intensity) / (pre_bleach_mean - min_intensity) if pre_bleach_mean != min_intensity else 0
                
                # Time to half recovery
                half_recovery_level = min_intensity + 0.5 * (final_recovery - min_intensity)
                half_recovery_idx = np.argmin(np.abs(recovery_data - half_recovery_level))
                time_to_half = recovery_time[half_recovery_idx] if half_recovery_idx < len(recovery_time) else np.inf
                
            else:
                recovery_rate = 0
                final_recovery = min_intensity
                recovery_fraction = 0
                time_to_half = np.inf
            
            # Curve smoothness (second derivative)
            if len(intensity) > 4:
                second_deriv = np.diff(intensity, n=2)
                curve_smoothness = np.std(second_deriv)
            else:
                curve_smoothness = 0
            
            # Asymmetry of recovery
            if bleach_idx < len(intensity) - 1 and bleach_idx > 0:
                pre_bleach_duration = time[bleach_idx] - time[0]
                post_bleach_duration = time[-1] - time[bleach_idx]
                asymmetry = pre_bleach_duration / (pre_bleach_duration + post_bleach_duration) if (pre_bleach_duration + post_bleach_duration) > 0 else 0.5
            else:
                asymmetry = 0.5
            
            return [
                min_intensity, max_intensity, intensity_range,
                pre_bleach_mean, pre_bleach_std, pre_bleach_trend,
                recovery_rate, final_recovery, recovery_fraction,
                time_to_half, curve_smoothness, asymmetry
            ]
            
        except Exception:
            return [np.nan] * 12
    
    def _extract_kinetic_features(self, time: np.ndarray, intensity: np.ndarray, features: Dict) -> List[float]:
        """Extract features from kinetic analysis results."""
        
        try:
            return [
                features.get('mobile_fraction', np.nan),
                features.get('immobile_fraction', np.nan),
                features.get('rate_constant', np.nan),
                features.get('rate_constant_fast', np.nan),
                features.get('rate_constant_slow', np.nan),
                features.get('half_time', np.nan),
                features.get('half_time_fast', np.nan),
                features.get('half_time_slow', np.nan),
                features.get('r2', np.nan),
                features.get('adj_r2', np.nan),
                features.get('aic', np.nan),
                features.get('bic', np.nan)
            ]
        except Exception:
            return [np.nan] * 12
    
    def _extract_quality_features(self, time: np.ndarray, intensity: np.ndarray, features: Dict) -> List[float]:
        """Extract data quality and experimental condition features."""
        
        try:
            # Signal-to-noise ratio
            if len(intensity) > 10:
                signal = np.mean(intensity)
                noise = np.std(intensity)
                snr = signal / noise if noise > 0 else np.inf
            else:
                snr = 1
            
            # Bleaching efficiency
            bleach_idx = np.argmin(intensity)
            if bleach_idx > 0:
                pre_bleach = np.mean(intensity[:bleach_idx])
                bleach_depth = pre_bleach - np.min(intensity)
                bleach_efficiency = bleach_depth / pre_bleach if pre_bleach > 0 else 0
            else:
                bleach_efficiency = 0
            
            # Data completeness
            nan_fraction = np.sum(np.isnan(intensity)) / len(intensity)
            
            # Temporal sampling quality
            time_intervals = np.diff(time)
            sampling_regularity = np.std(time_intervals) / np.mean(time_intervals) if np.mean(time_intervals) > 0 else np.inf
            
            # Intensity range and dynamic range
            intensity_range = np.max(intensity) - np.min(intensity)
            dynamic_range = intensity_range / np.mean(intensity) if np.mean(intensity) > 0 else 0
            
            # Baseline stability (pre-bleach variance)
            if bleach_idx > 5:
                baseline_variance = np.var(intensity[:bleach_idx])
            else:
                baseline_variance = np.var(intensity[:min(5, len(intensity))])
            
            return [
                snr, bleach_efficiency, nan_fraction,
                sampling_regularity, intensity_range, dynamic_range,
                baseline_variance, len(intensity), np.mean(time_intervals),
                np.max(time) - np.min(time), 0, 0  # Padding to 12 features
            ]
        except Exception:
            return [np.nan] * 12
    
    def _extract_statistical_features(self, time: np.ndarray, intensity: np.ndarray, features: Dict) -> List[float]:
        """Extract statistical features from the curve."""
        
        try:
            # Basic statistics
            mean_intensity = np.mean(intensity)
            std_intensity = np.std(intensity)
            skewness = self._calculate_skewness(intensity)
            kurtosis = self._calculate_kurtosis(intensity)
            
            # Percentiles
            q25 = np.percentile(intensity, 25)
            q50 = np.percentile(intensity, 50)
            q75 = np.percentile(intensity, 75)
            iqr = q75 - q25
            
            # Autocorrelation
            if len(intensity) > 10:
                autocorr = np.corrcoef(intensity[:-1], intensity[1:])[0, 1]
            else:
                autocorr = 0
            
            # Trend
            if len(time) > 1:
                trend = np.polyfit(time, intensity, 1)[0]
            else:
                trend = 0
            
            return [
                mean_intensity, std_intensity, skewness, kurtosis,
                q25, q50, q75, iqr, autocorr, trend, 0, 0  # Padding to 12 features
            ]
        except Exception:
            return [np.nan] * 12
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        try:
            n = len(data)
            if n < 3:
                return 0
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0
            return n / ((n-1) * (n-2)) * np.sum(((data - mean) / std) ** 3)
        except:
            return 0
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        try:
            n = len(data)
            if n < 4:
                return 0
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0
            kurt = np.sum(((data - mean) / std) ** 4) / n
            return ((n+1) * (n-1) * kurt - 3 * (n-1)**2) / ((n-2) * (n-3))
        except:
            return 0
    
    def fit_unsupervised(self, features: np.ndarray, method: str = 'isolation_forest', **kwargs) -> Dict:
        """
        Fit unsupervised outlier detection model.
        
        Parameters:
        -----------
        features : np.ndarray
            Feature matrix (n_samples, n_features)
        method : str
            Method to use ('isolation_forest', 'one_class_svm', 'lof')
        
        Returns:
        --------
        Dict
            Results including predictions and model info
        """
        
        if not ML_AVAILABLE:
            return {'error': 'Scikit-learn not available'}
        
        # Handle NaN values
        features_clean = self._handle_missing_values(features)
        
        # Scale features
        scaler = RobustScaler()
        features_scaled = scaler.fit_transform(features_clean)
        
        # Select and fit model
        if method == 'isolation_forest':
            model = IsolationForest(
                contamination=kwargs.get('contamination', 0.1),
                random_state=42,
                n_estimators=100
            )
        elif method == 'one_class_svm':
            model = OneClassSVM(
                nu=kwargs.get('nu', 0.1),
                kernel='rbf',
                gamma='scale'
            )
        elif method == 'lof':
            model = LocalOutlierFactor(
                n_neighbors=kwargs.get('n_neighbors', 20),
                contamination=kwargs.get('contamination', 0.1)
            )
        else:
            return {'error': f'Unknown method: {method}'}
        
        # Fit and predict
        if method == 'lof':
            predictions = model.fit_predict(features_scaled)
            scores = model.negative_outlier_factor_
        else:
            model.fit(features_scaled)
            predictions = model.predict(features_scaled)
            scores = model.score_samples(features_scaled) if hasattr(model, 'score_samples') else model.decision_function(features_scaled)
        
        # Convert predictions to boolean outliers
        outliers = predictions == -1
        
        # Store model and scaler
        self.models[f'unsupervised_{method}'] = model
        self.scalers[f'unsupervised_{method}'] = scaler
        
        return {
            'outliers': outliers,
            'scores': scores,
            'predictions': predictions,
            'method': method,
            'model': model,
            'scaler': scaler,
            'n_outliers': np.sum(outliers),
            'outlier_fraction': np.mean(outliers)
        }
    
    def fit_supervised(self, features: np.ndarray, labels: np.ndarray, method: str = 'random_forest') -> Dict:
        """
        Fit supervised classification model using user-labeled data.
        
        Parameters:
        -----------
        features : np.ndarray
            Feature matrix (n_samples, n_features)
        labels : np.ndarray
            Binary labels (0=normal, 1=outlier)
        method : str
            Classification method to use
        
        Returns:
        --------
        Dict
            Results including model performance and predictions
        """
        
        if not ML_AVAILABLE:
            return {'error': 'Scikit-learn not available'}
        
        # Handle NaN values
        features_clean = self._handle_missing_values(features)
        
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_clean)
        
        # Select model
        if method == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced'
            )
        else:
            return {'error': f'Unknown method: {method}'}
        
        # Cross-validation
        cv_scores = cross_val_score(model, features_scaled, labels, cv=5, scoring='roc_auc')
        
        # Fit final model
        model.fit(features_scaled, labels)
        
        # Predictions and probabilities
        predictions = model.predict(features_scaled)
        probabilities = model.predict_proba(features_scaled)[:, 1]
        
        # Store model and scaler
        self.models[f'supervised_{method}'] = model
        self.scalers[f'supervised_{method}'] = scaler
        
        # Performance metrics
        try:
            auc_score = roc_auc_score(labels, probabilities)
            class_report = classification_report(labels, predictions, output_dict=True)
        except:
            auc_score = 0.5
            class_report = {}
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'cv_scores': cv_scores,
            'mean_cv_score': np.mean(cv_scores),
            'auc_score': auc_score,
            'classification_report': class_report,
            'model': model,
            'scaler': scaler,
            'method': method
        }
    
    def predict_outliers(self, features: np.ndarray, model_name: str) -> Dict:
        """
        Predict outliers using a trained model.
        
        Parameters:
        -----------
        features : np.ndarray
            Feature matrix for new data
        model_name : str
            Name of the trained model to use
        
        Returns:
        --------
        Dict
            Predictions and confidence scores
        """
        
        if model_name not in self.models:
            return {'error': f'Model {model_name} not found'}
        
        model = self.models[model_name]
        scaler = self.scalers[model_name]
        
        # Handle NaN values and scale
        features_clean = self._handle_missing_values(features)
        features_scaled = scaler.transform(features_clean)
        
        # Predict
        if 'supervised' in model_name:
            predictions = model.predict(features_scaled)
            probabilities = model.predict_proba(features_scaled)[:, 1]
            return {
                'outliers': predictions.astype(bool),
                'probabilities': probabilities,
                'predictions': predictions
            }
        else:
            predictions = model.predict(features_scaled)
            outliers = predictions == -1
            if hasattr(model, 'score_samples'):
                scores = model.score_samples(features_scaled)
            else:
                scores = model.decision_function(features_scaled)
            
            return {
                'outliers': outliers,
                'scores': scores,
                'predictions': predictions
            }
    
    def _handle_missing_values(self, features: np.ndarray) -> np.ndarray:
        """Handle missing values in feature matrix."""
        # Replace NaN with median values
        features_clean = features.copy()
        for i in range(features_clean.shape[1]):
            col = features_clean[:, i]
            if np.any(np.isnan(col)):
                median_val = np.nanmedian(col)
                features_clean[np.isnan(col), i] = median_val if not np.isnan(median_val) else 0
        
        return features_clean
    
    def analyze_feature_importance(self, model_name: str) -> Dict:
        """Analyze feature importance for a trained model."""
        
        if model_name not in self.models:
            return {'error': f'Model {model_name} not found'}
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            
            # Create feature names if not available
            if not self.feature_names:
                self.feature_names = [f'feature_{i}' for i in range(len(importance))]
            
            # Sort by importance
            importance_df = pd.DataFrame({
                'feature': self.feature_names[:len(importance)],
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            return {
                'feature_importance': importance_df,
                'top_features': importance_df.head(10)
            }
        else:
            return {'error': 'Model does not support feature importance analysis'}


def create_ml_outlier_interface(group_data: Dict, data_manager, group_name: str):
    """
    Create Streamlit interface for ML-based outlier detection.
    
    Parameters:
    -----------
    group_data : Dict
        Group data from data manager
    data_manager : FRAPDataManager
        Data manager instance
    group_name : str
        Name of the current group
    """
    
    st.markdown("---")
    st.markdown("### 🤖 Machine Learning Outlier Detection")
    
    if not ML_AVAILABLE:
        st.error("⚠️ Machine Learning features require scikit-learn")
        st.code("pip install scikit-learn", language="bash")
        return
    
    st.markdown("""
    Use machine learning to automatically detect atypical FRAP curves based on:
    - **Curve shape characteristics** (recovery dynamics, smoothness, asymmetry)
    - **Kinetic parameters** (mobile fraction, rate constants, fit quality)
    - **Data quality metrics** (signal-to-noise, bleaching efficiency, baseline stability)
    - **Statistical features** (distribution properties, autocorrelation, trends)
    """)
    
    # Initialize detector
    detector = FRAPOutlierDetector()
    
    # Extract features from all curves in the group
    st.markdown("#### Feature Extraction")
    
    with st.spinner("Extracting features from FRAP curves..."):
        features_list = []
        file_paths = []
        file_names = []
        
        for file_path in group_data['files']:
            file_data = data_manager.files[file_path]
            features = detector.extract_features(file_data)
            
            if len(features) > 0:
                features_list.append(features)
                file_paths.append(file_path)
                file_names.append(file_data['name'])
        
        if not features_list:
            st.error("No features could be extracted from the curves.")
            return
        
        features_matrix = np.array(features_list)
    
    st.success(f"✅ Extracted {features_matrix.shape[1]} features from {len(features_list)} curves")
    
    # ML Method Selection
    st.markdown("#### ML Method Selection")
    
    ml_tabs = st.tabs(["🔍 Unsupervised Detection", "🎯 Supervised Learning", "📊 Feature Analysis"])
    
    with ml_tabs[0]:
        st.markdown("**Unsupervised Anomaly Detection**")
        st.markdown("Automatically detect outliers without labeled training data.")
        
        unsupervised_method = st.selectbox(
            "Choose detection method:",
            ['isolation_forest', 'one_class_svm', 'lof'],
            format_func=lambda x: {
                'isolation_forest': 'Isolation Forest (Recommended)',
                'one_class_svm': 'One-Class SVM',
                'lof': 'Local Outlier Factor'
            }[x]
        )
        
        col1, col2 = st.columns(2)
        with col1:
            contamination = st.slider(
                "Expected outlier fraction:",
                0.05, 0.3, 0.1, 0.01,
                help="Proportion of curves expected to be outliers"
            )
        
        with col2:
            if unsupervised_method == 'lof':
                n_neighbors = st.slider("Number of neighbors:", 5, 50, 20)
                method_params = {'contamination': contamination, 'n_neighbors': n_neighbors}
            else:
                method_params = {'contamination': contamination}
        
        if st.button("🔍 Run Unsupervised Detection", type="primary"):
            with st.spinner(f"Running {unsupervised_method} detection..."):
                results = detector.fit_unsupervised(
                    features_matrix, 
                    method=unsupervised_method,
                    **method_params
                )
            
            if 'error' in results:
                st.error(results['error'])
            else:
                outlier_indices = np.where(results['outliers'])[0]
                outlier_files = [file_names[i] for i in outlier_indices]
                
                st.success(f"✅ Detected {len(outlier_files)} outliers ({results['outlier_fraction']:.1%})")
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🚨 Detected Outliers:**")
                    if outlier_files:
                        for i, (idx, file) in enumerate(zip(outlier_indices, outlier_files)):
                            score = results['scores'][idx]
                            st.markdown(f"{i+1}. `{file}` (score: {score:.3f})")
                    else:
                        st.info("No outliers detected")
                
                with col2:
                    st.markdown("**📊 Detection Statistics:**")
                    st.metric("Total Curves", len(file_names))
                    st.metric("Outliers Found", len(outlier_files))
                    st.metric("Outlier Rate", f"{results['outlier_fraction']:.1%}")
                
                # Visualization
                st.markdown("**📈 Outlier Scores Visualization**")
                
                fig = go.Figure()
                
                # Plot normal curves
                normal_indices = np.where(~results['outliers'])[0]
                fig.add_trace(go.Scatter(
                    x=normal_indices,
                    y=[results['scores'][i] for i in normal_indices],
                    mode='markers',
                    name='Normal Curves',
                    marker=dict(color='green', size=8),
                    text=[file_names[i] for i in normal_indices],
                    hovertemplate="<b>%{text}</b><br>Score: %{y:.3f}<extra></extra>"
                ))
                
                # Plot outliers
                if len(outlier_indices) > 0:
                    fig.add_trace(go.Scatter(
                        x=outlier_indices,
                        y=[results['scores'][i] for i in outlier_indices],
                        mode='markers',
                        name='Detected Outliers',
                        marker=dict(color='red', size=12, symbol='x'),
                        text=[file_names[i] for i in outlier_indices],
                        hovertemplate="<b>%{text}</b><br>Score: %{y:.3f}<extra></extra>"
                    ))
                
                fig.update_layout(
                    title=f"Outlier Detection Results - {unsupervised_method.title()}",
                    xaxis_title="Curve Index",
                    yaxis_title="Outlier Score",
                    height=400
                )
                
                st.plotly_chart(fig, width="stretch")
                
                # Apply outliers button
                if st.button("💾 Apply ML Outlier Detection", help="Update group analysis with detected outliers"):
                    # Update the interactive excluded files
                    outlier_paths = [file_paths[i] for i in outlier_indices]
                    st.session_state.interactive_excluded_files = set(outlier_paths)
                    
                    # Update the group analysis
                    data_manager.update_group_analysis(group_name, excluded_files=outlier_paths)
                    
                    st.success(f"✅ Applied ML detection! Excluded {len(outlier_files)} outliers from analysis.")
                    st.rerun()
    
    with ml_tabs[1]:
        st.markdown("**Supervised Outlier Classification**")
        st.markdown("Train a model using your manual selections to learn outlier patterns.")
        
        # Check if user has made selections
        if 'interactive_excluded_files' in st.session_state:
            excluded_paths = st.session_state.interactive_excluded_files
            
            # Create labels based on user selections
            labels = np.array([1 if path in excluded_paths else 0 for path in file_paths])
            
            n_outliers = np.sum(labels)
            n_normal = len(labels) - n_outliers
            
            st.info(f"📊 Training data: {n_normal} normal curves, {n_outliers} outliers")
            
            if n_outliers > 0 and n_normal > 0:
                supervised_method = st.selectbox(
                    "Choose classification method:",
                    ['random_forest'],
                    format_func=lambda x: {'random_forest': 'Random Forest Classifier'}[x]
                )
                
                if st.button("🎯 Train Supervised Model", type="primary"):
                    with st.spinner("Training supervised classification model..."):
                        results = detector.fit_supervised(
                            features_matrix,
                            labels,
                            method=supervised_method
                        )
                    
                    if 'error' in results:
                        st.error(results['error'])
                    else:
                        st.success(f"✅ Model trained! Cross-validation AUC: {results['mean_cv_score']:.3f}")
                        
                        # Display performance metrics
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**📊 Model Performance:**")
                            st.metric("Cross-Validation AUC", f"{results['mean_cv_score']:.3f}")
                            st.metric("Training AUC", f"{results['auc_score']:.3f}")
                            
                            if 'classification_report' in results and results['classification_report']:
                                accuracy = results['classification_report'].get('accuracy', 0)
                                st.metric("Training Accuracy", f"{accuracy:.3f}")
                        
                        with col2:
                            st.markdown("**🔍 Predictions vs. Labels:**")
                            
                            # Confusion matrix
                            cm = confusion_matrix(labels, results['predictions'])
                            
                            fig_cm = px.imshow(
                                cm,
                                text_auto=True,
                                aspect="auto",
                                labels=dict(x="Predicted", y="Actual"),
                                x=['Normal', 'Outlier'],
                                y=['Normal', 'Outlier'],
                                title="Confusion Matrix"
                            )
                            
                            st.plotly_chart(fig_cm, width="stretch")
                        
                        # Feature importance
                        importance_results = detector.analyze_feature_importance(f'supervised_{supervised_method}')
                        
                        if 'feature_importance' in importance_results:
                            st.markdown("**🔑 Most Important Features:**")
                            
                            top_features = importance_results['top_features']
                            
                            fig_importance = px.bar(
                                top_features.head(10),
                                x='importance',
                                y='feature',
                                orientation='h',
                                title="Top 10 Feature Importances"
                            )
                            fig_importance.update_layout(height=400)
                            
                            st.plotly_chart(fig_importance, width="stretch")
                        
                        # Probability visualization
                        st.markdown("**📈 Outlier Probabilities**")
                        
                        fig_prob = go.Figure()
                        
                        # Plot by true labels
                        normal_indices = np.where(labels == 0)[0]
                        outlier_indices = np.where(labels == 1)[0]
                        
                        if len(normal_indices) > 0:
                            fig_prob.add_trace(go.Scatter(
                                x=normal_indices,
                                y=[results['probabilities'][i] for i in normal_indices],
                                mode='markers',
                                name='Normal (User Labeled)',
                                marker=dict(color='green', size=8),
                                text=[file_names[i] for i in normal_indices],
                                hovertemplate="<b>%{text}</b><br>Outlier Probability: %{y:.3f}<extra></extra>"
                            ))
                        
                        if len(outlier_indices) > 0:
                            fig_prob.add_trace(go.Scatter(
                                x=outlier_indices,
                                y=[results['probabilities'][i] for i in outlier_indices],
                                mode='markers',
                                name='Outliers (User Labeled)',
                                marker=dict(color='red', size=12, symbol='x'),
                                text=[file_names[i] for i in outlier_indices],
                                hovertemplate="<b>%{text}</b><br>Outlier Probability: %{y:.3f}<extra></extra>"
                            ))
                        
                        # Add threshold line
                        fig_prob.add_hline(y=0.5, line_dash="dash", line_color="gray", 
                                         annotation_text="Decision Threshold (0.5)")
                        
                        fig_prob.update_layout(
                            title="Outlier Probability Predictions",
                            xaxis_title="Curve Index",
                            yaxis_title="Outlier Probability",
                            height=400
                        )
                        
                        st.plotly_chart(fig_prob, width="stretch")
            
            else:
                st.warning("⚠️ Need both normal and outlier examples for supervised learning. Please select some curves as outliers first.")
        
        else:
            st.info("💡 Make some manual curve selections first to create training data for supervised learning.")
    
    with ml_tabs[2]:
        st.markdown("**Feature Analysis and Visualization**")
        
        # Feature correlation heatmap
        st.markdown("**🔥 Feature Correlation Heatmap**")
        
        # Create feature names
        feature_categories = ['curve_shape'] * 12 + ['kinetic_params'] * 12 + ['quality_metrics'] * 12 + ['statistical'] * 12
        feature_names = [f"{cat}_{i}" for cat, i in zip(feature_categories, range(len(feature_categories)))]
        
        # Calculate correlation matrix
        features_clean = detector._handle_missing_values(features_matrix)
        correlation_matrix = np.corrcoef(features_clean.T)
        
        fig_corr = px.imshow(
            correlation_matrix,
            labels=dict(x="Features", y="Features"),
            title="Feature Correlation Matrix",
            color_continuous_scale="RdBu_r"
        )
        fig_corr.update_layout(height=500)
        
        st.plotly_chart(fig_corr, width="stretch")
        
        # Principal Component Analysis
        st.markdown("**🎯 Principal Component Analysis (PCA)**")
        
        if st.button("Run PCA Analysis"):
            with st.spinner("Running PCA..."):
                pca = PCA(n_components=2)
                features_pca = pca.fit_transform(features_clean)
                
                # Create PCA visualization
                fig_pca = go.Figure()
                
                # Color by user selections if available
                if 'interactive_excluded_files' in st.session_state:
                    excluded_paths = st.session_state.interactive_excluded_files
                    colors = ['red' if path in excluded_paths else 'blue' for path in file_paths]
                    labels_pca = ['Excluded' if path in excluded_paths else 'Included' for path in file_paths]
                else:
                    colors = ['blue'] * len(file_names)
                    labels_pca = ['Curve'] * len(file_names)
                
                fig_pca.add_trace(go.Scatter(
                    x=features_pca[:, 0],
                    y=features_pca[:, 1],
                    mode='markers',
                    marker=dict(color=colors, size=10),
                    text=file_names,
                    hovertemplate="<b>%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>",
                    showlegend=False
                ))
                
                fig_pca.update_layout(
                    title=f"PCA Visualization (Explained Variance: PC1={pca.explained_variance_ratio_[0]:.1%}, PC2={pca.explained_variance_ratio_[1]:.1%})",
                    xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
                    yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",
                    height=500
                )
                
                st.plotly_chart(fig_pca, width="stretch")
                
                st.info(f"📊 First 2 components explain {pca.explained_variance_ratio_[:2].sum():.1%} of the variance")


if __name__ == "__main__":
    # Test feature extraction
    detector = FRAPOutlierDetector()
    
    # Create dummy FRAP data
    time = np.linspace(0, 100, 101)
    intensity = 0.5 + 0.4 * (1 - np.exp(-time/20)) + np.random.normal(0, 0.02, len(time))
    
    test_data = {
        'time': time,
        'intensity': intensity,
        'features': {
            'mobile_fraction': 80,
            'rate_constant': 0.05,
            'r2': 0.95
        }
    }
    
    features = detector.extract_features(test_data)
    print(f"Extracted {len(features)} features")
    print(f"Features: {features[:10]}...")  # Show first 10