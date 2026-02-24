#!/usr/bin/env python3
"""
Statistical Outlier Detection for FRAP Analysis

This module implements robust statistical methods for outlier detection,
focusing on IQR-based filtering and other statistical approaches to ensure
robust population averages in FRAP analysis.

Author: FRAP2025 Analysis Platform
Date: October 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
warnings.filterwarnings('ignore')

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from scipy import stats


class FRAPStatisticalOutlierDetector:
    """
    Statistical outlier detection for FRAP recovery curves using IQR and other robust methods.
    """
    
    def __init__(self):
        """Initialize the statistical outlier detector."""
        self.methods = {
            'iqr': self._iqr_outlier_detection,
            'zscore': self._zscore_outlier_detection,
            'modified_zscore': self._modified_zscore_outlier_detection,
            'grubbs': self._grubbs_outlier_detection,
            'dixon': self._dixon_outlier_detection
        }
        
        self.default_params = {
            'iqr_multiplier': 1.5,  # Standard IQR multiplier
            'zscore_threshold': 3.0,  # Standard Z-score threshold
            'modified_zscore_threshold': 3.5,  # Modified Z-score threshold
            'grubbs_alpha': 0.05,  # Grubbs test significance level
            'dixon_alpha': 0.05,  # Dixon test significance level
            'min_samples': 3,  # Minimum samples for statistical tests
            'robust_center': 'median',  # Use median instead of mean for robustness
        }
    
    def detect_outliers_by_parameter(self, 
                                   parameter_values: np.ndarray, 
                                   method: str = 'iqr',
                                   params: Optional[Dict] = None) -> Dict:
        """
        Detect outliers in a single parameter using statistical methods.
        
        Parameters:
        -----------
        parameter_values : np.ndarray
            Array of parameter values (e.g., mobile fractions, half-times)
        method : str
            Statistical method to use ('iqr', 'zscore', 'modified_zscore', 'grubbs', 'dixon')
        params : Dict, optional
            Method-specific parameters
            
        Returns:
        --------
        Dict
            Outlier detection results
        """
        
        if params is None:
            params = self.default_params.copy()
        
        # Remove NaN values
        clean_values = parameter_values[~np.isnan(parameter_values)]
        
        if len(clean_values) < params['min_samples']:
            return {
                'outliers': np.array([]),
                'outlier_indices': np.array([]),
                'inliers': clean_values,
                'inlier_indices': np.array([]),
                'method': method,
                'n_outliers': 0,
                'n_inliers': len(clean_values),
                'error': 'Insufficient samples for statistical analysis'
            }
        
        try:
            # Apply the selected method
            detector_func = self.methods.get(method, self._iqr_outlier_detection)
            result = detector_func(clean_values, params)
            
            # Map back to original indices (handling NaN values)
            original_indices = np.where(~np.isnan(parameter_values))[0]
            
            outlier_original_indices = original_indices[result['outlier_indices']]
            inlier_original_indices = original_indices[result['inlier_indices']]
            
            result['outlier_original_indices'] = outlier_original_indices
            result['inlier_original_indices'] = inlier_original_indices
            result['method'] = method
            
            return result
            
        except Exception as e:
            return {
                'outliers': np.array([]),
                'outlier_indices': np.array([]),
                'inliers': clean_values,
                'inlier_indices': np.arange(len(clean_values)),
                'method': method,
                'n_outliers': 0,
                'n_inliers': len(clean_values),
                'error': str(e)
            }
    
    def _iqr_outlier_detection(self, values: np.ndarray, params: Dict) -> Dict:
        """IQR-based outlier detection (most robust and widely used)."""
        
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        
        multiplier = params.get('iqr_multiplier', 1.5)
        
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        
        outlier_mask = (values < lower_bound) | (values > upper_bound)
        outlier_indices = np.where(outlier_mask)[0]
        inlier_indices = np.where(~outlier_mask)[0]
        
        return {
            'outliers': values[outlier_indices],
            'outlier_indices': outlier_indices,
            'inliers': values[inlier_indices],
            'inlier_indices': inlier_indices,
            'n_outliers': len(outlier_indices),
            'n_inliers': len(inlier_indices),
            'bounds': {'lower': lower_bound, 'upper': upper_bound},
            'statistics': {'q1': q1, 'q3': q3, 'iqr': iqr, 'multiplier': multiplier}
        }
    
    def _zscore_outlier_detection(self, values: np.ndarray, params: Dict) -> Dict:
        """Z-score based outlier detection."""
        
        if params.get('robust_center') == 'median':
            center = np.median(values)
            scale = np.median(np.abs(values - center)) * 1.4826  # MAD to std conversion
        else:
            center = np.mean(values)
            scale = np.std(values)
        
        z_scores = np.abs((values - center) / scale) if scale > 0 else np.zeros_like(values)
        threshold = params.get('zscore_threshold', 3.0)
        
        outlier_mask = z_scores > threshold
        outlier_indices = np.where(outlier_mask)[0]
        inlier_indices = np.where(~outlier_mask)[0]
        
        return {
            'outliers': values[outlier_indices],
            'outlier_indices': outlier_indices,
            'inliers': values[inlier_indices],
            'inlier_indices': inlier_indices,
            'n_outliers': len(outlier_indices),
            'n_inliers': len(inlier_indices),
            'z_scores': z_scores,
            'threshold': threshold,
            'statistics': {'center': center, 'scale': scale}
        }
    
    def _modified_zscore_outlier_detection(self, values: np.ndarray, params: Dict) -> Dict:
        """Modified Z-score using median absolute deviation (more robust)."""
        
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        
        if mad == 0:
            # If MAD is 0, use a small value or fallback to std
            mad = np.std(values) / 1.4826 if np.std(values) > 0 else 1e-10
        
        modified_z_scores = 0.6745 * (values - median) / mad
        threshold = params.get('modified_zscore_threshold', 3.5)
        
        outlier_mask = np.abs(modified_z_scores) > threshold
        outlier_indices = np.where(outlier_mask)[0]
        inlier_indices = np.where(~outlier_mask)[0]
        
        return {
            'outliers': values[outlier_indices],
            'outlier_indices': outlier_indices,
            'inliers': values[inlier_indices],
            'inlier_indices': inlier_indices,
            'n_outliers': len(outlier_indices),
            'n_inliers': len(inlier_indices),
            'modified_z_scores': modified_z_scores,
            'threshold': threshold,
            'statistics': {'median': median, 'mad': mad}
        }
    
    def _grubbs_outlier_detection(self, values: np.ndarray, params: Dict) -> Dict:
        """Grubbs test for outliers (assumes normal distribution)."""
        
        n = len(values)
        if n < 3:
            return self._iqr_outlier_detection(values, params)  # Fallback
        
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1)
        
        if std_val == 0:
            return {
                'outliers': np.array([]),
                'outlier_indices': np.array([]),
                'inliers': values,
                'inlier_indices': np.arange(len(values)),
                'n_outliers': 0,
                'n_inliers': len(values),
                'statistics': {'mean': mean_val, 'std': std_val}
            }
        
        # Calculate Grubbs statistic for each point
        g_values = np.abs((values - mean_val) / std_val)
        max_g = np.max(g_values)
        max_g_idx = np.argmax(g_values)
        
        # Critical value for Grubbs test
        alpha = params.get('grubbs_alpha', 0.05)
        t_critical = stats.t.ppf(1 - alpha / (2 * n), n - 2)
        g_critical = ((n - 1) / np.sqrt(n)) * np.sqrt(t_critical**2 / (n - 2 + t_critical**2))
        
        if max_g > g_critical:
            outlier_indices = np.array([max_g_idx])
            inlier_indices = np.array([i for i in range(n) if i != max_g_idx])
        else:
            outlier_indices = np.array([])
            inlier_indices = np.arange(n)
        
        return {
            'outliers': values[outlier_indices] if len(outlier_indices) > 0 else np.array([]),
            'outlier_indices': outlier_indices,
            'inliers': values[inlier_indices],
            'inlier_indices': inlier_indices,
            'n_outliers': len(outlier_indices),
            'n_inliers': len(inlier_indices),
            'g_values': g_values,
            'g_critical': g_critical,
            'statistics': {'mean': mean_val, 'std': std_val, 'alpha': alpha}
        }
    
    def _dixon_outlier_detection(self, values: np.ndarray, params: Dict) -> Dict:
        """Dixon Q-test for outliers."""
        
        n = len(values)
        if n < 3:
            return self._iqr_outlier_detection(values, params)  # Fallback
        
        sorted_values = np.sort(values)
        
        # Dixon Q-test critical values (approximate)
        dixon_critical = {
            3: 0.970, 4: 0.829, 5: 0.710, 6: 0.625, 7: 0.568,
            8: 0.526, 9: 0.493, 10: 0.466
        }
        
        q_critical = dixon_critical.get(min(n, 10), 0.466)  # Use 0.466 for n > 10
        
        # Test lowest value
        if n >= 3:
            q_low = (sorted_values[1] - sorted_values[0]) / (sorted_values[-1] - sorted_values[0])
        else:
            q_low = 0
        
        # Test highest value  
        if n >= 3:
            q_high = (sorted_values[-1] - sorted_values[-2]) / (sorted_values[-1] - sorted_values[0])
        else:
            q_high = 0
        
        outlier_indices = []
        
        # Check if lowest value is outlier
        if q_low > q_critical:
            outlier_val = sorted_values[0]
            outlier_idx = np.where(values == outlier_val)[0][0]
            outlier_indices.append(outlier_idx)
        
        # Check if highest value is outlier
        if q_high > q_critical:
            outlier_val = sorted_values[-1]
            outlier_idx = np.where(values == outlier_val)[0][0]
            if outlier_idx not in outlier_indices:
                outlier_indices.append(outlier_idx)
        
        outlier_indices = np.array(outlier_indices)
        inlier_indices = np.array([i for i in range(n) if i not in outlier_indices])
        
        return {
            'outliers': values[outlier_indices] if len(outlier_indices) > 0 else np.array([]),
            'outlier_indices': outlier_indices,
            'inliers': values[inlier_indices],
            'inlier_indices': inlier_indices,
            'n_outliers': len(outlier_indices),
            'n_inliers': len(inlier_indices),
            'q_values': {'q_low': q_low, 'q_high': q_high},
            'q_critical': q_critical,
            'statistics': {'sorted_values': sorted_values}
        }
    
    def analyze_group_parameters(self, group_data: Dict, data_manager, parameters: List[str] = None) -> Dict:
        """
        Analyze multiple parameters across a group to detect statistical outliers.
        
        Parameters:
        -----------
        group_data : Dict
            Group data containing file paths
        data_manager : FRAPDataManager
            Data manager with analysis results
        parameters : List[str], optional
            Parameters to analyze (default: common FRAP parameters)
            
        Returns:
        --------
        Dict
            Comprehensive outlier analysis results
        """
        
        if parameters is None:
            parameters = ['mobile_fraction', 'half_time', 'rate_constant', 'immobile_fraction']
        
        # Extract parameter values for all files in group
        param_data = {}
        file_paths = []
        file_names = []
        
        for file_path in group_data['files']:
            try:
                file_data = data_manager.files[file_path]
                
                # Check if analysis results exist
                if 'analysis_results' in file_data and file_data['analysis_results']:
                    results = file_data['analysis_results']
                    
                    # Collect parameter values
                    param_values = {}
                    for param in parameters:
                        if param in results:
                            param_values[param] = results[param]
                        else:
                            param_values[param] = np.nan
                    
                    # Store data
                    file_paths.append(file_path)
                    file_names.append(file_data['name'])
                    
                    for param in parameters:
                        if param not in param_data:
                            param_data[param] = []
                        param_data[param].append(param_values[param])
                        
            except Exception as e:
                continue
        
        if not param_data or not file_paths:
            return {
                'error': 'No analysis results found for statistical outlier detection',
                'parameters': parameters,
                'n_files': 0
            }
        
        # Convert to arrays
        for param in param_data:
            param_data[param] = np.array(param_data[param])
        
        return {
            'param_data': param_data,
            'file_paths': file_paths,
            'file_names': file_names,
            'parameters': parameters,
            'n_files': len(file_paths)
        }
    
    def create_statistical_outlier_plot(self, 
                                      param_values: np.ndarray, 
                                      outlier_result: Dict, 
                                      param_name: str,
                                      file_names: List[str] = None) -> go.Figure:
        """
        Create visualization of statistical outlier detection results.
        """
        
        if file_names is None:
            file_names = [f"File {i+1}" for i in range(len(param_values))]
        
        fig = go.Figure()
        
        # Get outlier information
        outlier_indices = outlier_result.get('outlier_original_indices', [])
        inlier_indices = outlier_result.get('inlier_original_indices', [])
        
        # Plot inliers
        if len(inlier_indices) > 0:
            fig.add_trace(go.Scatter(
                x=inlier_indices,
                y=param_values[inlier_indices],
                mode='markers',
                name='Inliers',
                marker=dict(color='blue', size=8),
                text=[file_names[i] for i in inlier_indices],
                hovertemplate="<b>%{text}</b><br>Value: %{y:.3f}<br>Status: Inlier<extra></extra>"
            ))
        
        # Plot outliers
        if len(outlier_indices) > 0:
            fig.add_trace(go.Scatter(
                x=outlier_indices,
                y=param_values[outlier_indices],
                mode='markers',
                name='Outliers',
                marker=dict(color='red', size=12, symbol='x'),
                text=[file_names[i] for i in outlier_indices],
                hovertemplate="<b>%{text}</b><br>Value: %{y:.3f}<br>Status: Outlier<extra></extra>"
            ))
        
        # Add bounds if available (for IQR method)
        if 'bounds' in outlier_result:
            bounds = outlier_result['bounds']
            
            # Lower bound
            fig.add_hline(
                y=bounds['lower'],
                line_dash="dash",
                line_color="red",
                annotation_text=f"Lower Bound ({bounds['lower']:.3f})"
            )
            
            # Upper bound
            fig.add_hline(
                y=bounds['upper'],
                line_dash="dash",
                line_color="red",
                annotation_text=f"Upper Bound ({bounds['upper']:.3f})"
            )
        
        # Customize layout
        method_name = outlier_result.get('method', 'Unknown').upper()
        n_outliers = outlier_result.get('n_outliers', 0)
        
        fig.update_layout(
            title=f"Statistical Outlier Detection: {param_name.replace('_', ' ').title()}<br>"
                  f"<sub>Method: {method_name} | Outliers: {n_outliers}</sub>",
            xaxis_title="File Index",
            yaxis_title=param_name.replace('_', ' ').title(),
            height=400,
            showlegend=True
        )
        
        return fig
    
    def create_outlier_summary_table(self, param_data: Dict, outlier_results: Dict, file_names: List[str]) -> pd.DataFrame:
        """
        Create a summary table of outlier detection results across parameters.
        """
        
        summary_data = []
        
        for i, file_name in enumerate(file_names):
            row = {'File Name': file_name}
            
            is_outlier_any = False
            outlier_params = []
            
            for param in param_data:
                if param in outlier_results:
                    result = outlier_results[param]
                    outlier_indices = result.get('outlier_original_indices', [])
                    
                    if i in outlier_indices:
                        row[f'{param}_outlier'] = '🚨 OUTLIER'
                        is_outlier_any = True
                        outlier_params.append(param)
                    else:
                        row[f'{param}_outlier'] = '✅ Normal'
                    
                    # Add parameter value
                    row[f'{param}_value'] = param_data[param][i] if not np.isnan(param_data[param][i]) else 'N/A'
            
            row['Overall Status'] = '🚨 OUTLIER' if is_outlier_any else '✅ Normal'
            row['Outlier Parameters'] = ', '.join(outlier_params) if outlier_params else 'None'
            
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)


def create_statistical_outlier_interface(group_data: Dict, data_manager, group_name: str):
    """
    Create Streamlit interface for statistical outlier detection.
    
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
    st.markdown("### 📊 Statistical Outlier Detection")
    
    st.markdown("""
    **Robust statistical methods for identifying outliers in FRAP parameters to ensure reliable population averages.**
    
    Available methods:
    - **📈 IQR (Interquartile Range)**: Most robust, works without assuming distribution
    - **📏 Z-Score**: Good for normally distributed data
    - **🔍 Modified Z-Score**: Uses median absolute deviation, more robust than Z-score
    - **🧪 Grubbs Test**: Tests for single outliers, assumes normal distribution
    - **🎯 Dixon Q-Test**: Good for small samples, tests extreme values
    """)
    
    # Initialize detector
    detector = FRAPStatisticalOutlierDetector()
    
    # Get parameter data
    with st.spinner("Extracting parameter data..."):
        analysis_data = detector.analyze_group_parameters(group_data, data_manager)
    
    if 'error' in analysis_data:
        st.error(f"❌ {analysis_data['error']}")
        st.info("💡 Make sure to run curve fitting analysis first (Step 2) before statistical outlier detection.")
        return
    
    param_data = analysis_data['param_data']
    file_paths = analysis_data['file_paths']
    file_names = analysis_data['file_names']
    available_params = list(param_data.keys())
    
    st.success(f"✅ Found analysis results for {len(file_paths)} curves")
    
    # Parameter and method selection
    col1, col2 = st.columns(2)
    
    with col1:
        selected_params = st.multiselect(
            "Select parameters to analyze:",
            options=available_params,
            default=available_params[:3] if len(available_params) >= 3 else available_params,
            help="Choose kinetic parameters for outlier detection"
        )
    
    with col2:
        method = st.selectbox(
            "Statistical method:",
            options=['iqr', 'zscore', 'modified_zscore', 'grubbs', 'dixon'],
            index=0,  # Default to IQR
            format_func=lambda x: {
                'iqr': '📈 IQR (Recommended)',
                'zscore': '📏 Z-Score',
                'modified_zscore': '🔍 Modified Z-Score',
                'grubbs': '🧪 Grubbs Test',
                'dixon': '🎯 Dixon Q-Test'
            }[x]
        )
    
    # Method-specific parameters
    with st.expander("🔧 Method Parameters", expanded=False):
        if method == 'iqr':
            iqr_multiplier = st.slider("IQR Multiplier:", 1.0, 3.0, 1.5, 0.1,
                                     help="1.5 = standard, 2.0 = conservative, 1.0 = aggressive")
            method_params = {'iqr_multiplier': iqr_multiplier}
            
        elif method in ['zscore', 'modified_zscore']:
            threshold = st.slider("Z-Score Threshold:", 2.0, 4.0, 3.0 if method == 'zscore' else 3.5, 0.1)
            robust_center = st.checkbox("Use robust center (median)", value=True,
                                      help="Use median instead of mean for robustness")
            method_params = {
                'zscore_threshold' if method == 'zscore' else 'modified_zscore_threshold': threshold,
                'robust_center': 'median' if robust_center else 'mean'
            }
            
        elif method in ['grubbs', 'dixon']:
            alpha = st.slider("Significance Level (α):", 0.01, 0.10, 0.05, 0.01,
                            help="Lower values = more conservative detection")
            method_params = {f'{method}_alpha': alpha}
        
        else:
            method_params = {}
    
    if not selected_params:
        st.warning("⚠️ Please select at least one parameter to analyze.")
        return
    
    # Run outlier detection
    if st.button("🔍 Detect Statistical Outliers", type="primary"):
        
        outlier_results = {}
        overall_outliers = set()
        
        with st.spinner(f"Running {method.upper()} outlier detection..."):
            
            for param in selected_params:
                if param in param_data:
                    param_values = param_data[param]
                    
                    result = detector.detect_outliers_by_parameter(
                        param_values, 
                        method=method, 
                        params=method_params
                    )
                    
                    outlier_results[param] = result
                    
                    # Collect outlier file indices
                    if 'outlier_original_indices' in result:
                        overall_outliers.update(result['outlier_original_indices'])
        
        # Display results
        st.markdown("#### 📋 Detection Results")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Parameters Analyzed", len(selected_params))
        
        with col2:
            st.metric("Total Curves", len(file_paths))
        
        with col3:
            n_outliers = len(overall_outliers)
            st.metric("Outlier Curves", n_outliers, 
                     delta=f"{n_outliers/len(file_paths):.1%} of total")
        
        with col4:
            n_clean = len(file_paths) - n_outliers
            st.metric("Clean Curves", n_clean,
                     delta=f"{n_clean/len(file_paths):.1%} of total")
        
        # Individual parameter results
        st.markdown("#### 📊 Parameter-wise Analysis")
        
        param_tabs = st.tabs([param.replace('_', ' ').title() for param in selected_params])
        
        for i, (param, tab) in enumerate(zip(selected_params, param_tabs)):
            with tab:
                if param in outlier_results:
                    result = outlier_results[param]
                    
                    if 'error' in result:
                        st.error(f"Error analyzing {param}: {result['error']}")
                        continue
                    
                    # Parameter statistics
                    param_values = param_data[param]
                    clean_values = param_values[~np.isnan(param_values)]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**📈 {param.replace('_', ' ').title()} Statistics:**")
                        st.write(f"Mean: {np.mean(clean_values):.4f}")
                        st.write(f"Median: {np.median(clean_values):.4f}")
                        st.write(f"Std Dev: {np.std(clean_values):.4f}")
                        st.write(f"IQR: {np.percentile(clean_values, 75) - np.percentile(clean_values, 25):.4f}")
                    
                    with col2:
                        st.markdown(f"**🎯 {method.upper()} Results:**")
                        st.write(f"Outliers: {result['n_outliers']}")
                        st.write(f"Inliers: {result['n_inliers']}")
                        st.write(f"Outlier Rate: {result['n_outliers']/(result['n_outliers']+result['n_inliers']):.1%}")
                        
                        if 'statistics' in result:
                            stats = result['statistics']
                            for key, value in stats.items():
                                if isinstance(value, (int, float)):
                                    st.write(f"{key.title()}: {value:.4f}")
                    
                    # Visualization
                    fig = detector.create_statistical_outlier_plot(
                        param_values, result, param, file_names
                    )
                    st.plotly_chart(fig, width="stretch")
                    
                    # Show outliers for this parameter
                    if result['n_outliers'] > 0:
                        st.markdown("**🚨 Detected Outliers:**")
                        outlier_indices = result.get('outlier_original_indices', [])
                        
                        for idx in outlier_indices:
                            value = param_values[idx]
                            st.write(f"• {file_names[idx]}: {value:.4f}")
        
        # Overall summary table
        st.markdown("#### 📋 Overall Summary")
        
        summary_df = detector.create_outlier_summary_table(param_data, outlier_results, file_names)
        
        # Style the dataframe
        def highlight_outliers(val):
            if '🚨 OUTLIER' in str(val):
                return 'background-color: #ffcccb'
            elif '✅ Normal' in str(val):
                return 'background-color: #90EE90'
            return ''
        
        styled_df = summary_df.style.applymap(highlight_outliers)
        st.dataframe(styled_df, width="stretch")
        
        # Apply outliers button
        if overall_outliers:
            st.markdown("#### 🎯 Apply Statistical Outlier Detection")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Files to be excluded:**")
                outlier_file_paths = [file_paths[i] for i in overall_outliers]
                for i, idx in enumerate(sorted(overall_outliers)):
                    st.write(f"{i+1}. {file_names[idx]}")
            
            with col2:
                if st.button("🚫 Apply Statistical Exclusion", type="secondary",
                           help="Exclude detected outliers from group analysis"):
                    
                    # Update session state
                    if 'interactive_excluded_files' not in st.session_state:
                        st.session_state.interactive_excluded_files = set()
                    
                    # Add statistical outliers to exclusion set
                    st.session_state.interactive_excluded_files.update(outlier_file_paths)
                    
                    # Update group analysis
                    data_manager.update_group_analysis(group_name, 
                                                     excluded_files=list(st.session_state.interactive_excluded_files))
                    
                    st.success(f"✅ Excluded {len(outlier_file_paths)} statistical outliers!")
                    st.info(f"Population averages will now be based on {len(file_paths) - len(overall_outliers)} curves.")
                    st.rerun()
        
        else:
            st.info("✅ No statistical outliers detected! Population appears robust.")


if __name__ == "__main__":
    # Test the statistical outlier detector
    detector = FRAPStatisticalOutlierDetector()
    
    # Generate test data with outliers
    np.random.seed(42)
    normal_data = np.random.normal(75, 10, 20)  # Mobile fractions around 75%
    outliers = [45, 95]  # Clear outliers
    test_data = np.concatenate([normal_data, outliers])
    
    print("Testing Statistical Outlier Detection")
    print("=====================================")
    
    for method in ['iqr', 'zscore', 'modified_zscore']:
        result = detector.detect_outliers_by_parameter(test_data, method=method)
        print(f"\n{method.upper()}:")
        print(f"  Outliers detected: {result['n_outliers']}")
        if result['n_outliers'] > 0:
            print(f"  Outlier values: {result['outliers']}")
        print(f"  Method successful: {'error' not in result}")
    
    print(f"\nTest completed successfully!")