#!/usr/bin/env python3
"""
Automatic Detection of Downward Slopes in FRAP Recovery Curves

This module provides simple, effective detection of curves that show
downward trends in the final recovery phase, which typically indicates
technical problems like photobleaching, cell movement, or sample drift.

Author: FRAP2025 Analysis Platform
Date: October 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st


class FRAPSlopeDetector:
    """
    Simple and effective detection of problematic downward slopes
    in FRAP recovery curves.
    """
    
    def __init__(self):
        """Initialize the slope detector with default parameters."""
        self.default_params = {
            'recovery_start_fraction': 0.3,  # Start analyzing after 30% of recovery
            'analysis_window_fraction': 0.4,  # Analyze final 40% of recovery
            'slope_threshold': -0.001,  # Negative slope threshold (per second)
            'min_recovery_points': 10,  # Minimum points needed for reliable analysis
            'r2_threshold': 0.7,  # Minimum R² for linear fit to be considered reliable
        }
    
    def detect_downward_slope(self, 
                             time: np.ndarray, 
                             intensity: np.ndarray, 
                             params: Optional[Dict] = None) -> Dict:
        """
        Detect if a FRAP curve shows problematic downward slope in recovery phase.
        
        Parameters:
        -----------
        time : np.ndarray
            Time points of the FRAP curve
        intensity : np.ndarray  
            Intensity values of the FRAP curve
        params : Dict, optional
            Detection parameters (uses defaults if not provided)
            
        Returns:
        --------
        Dict
            Detection results including slope, confidence, and recommendation
        """
        
        if params is None:
            params = self.default_params.copy()
        
        try:
            # Basic validation
            if len(time) != len(intensity) or len(time) < params['min_recovery_points']:
                return {
                    'has_downward_slope': False,
                    'slope': 0.0,
                    'r2': 0.0,
                    'confidence': 0.0,
                    'recommendation': 'insufficient_data',
                    'analysis_region': None,
                    'error': 'Insufficient data points'
                }
            
            # Find bleaching point (minimum intensity)
            bleach_idx = np.argmin(intensity)
            
            if bleach_idx >= len(intensity) - params['min_recovery_points']:
                return {
                    'has_downward_slope': False,
                    'slope': 0.0,
                    'r2': 0.0,
                    'confidence': 0.0,
                    'recommendation': 'no_recovery_data',
                    'analysis_region': None,
                    'error': 'No recovery phase detected'
                }
            
            # Define recovery phase
            recovery_time = time[bleach_idx:] - time[bleach_idx]
            recovery_intensity = intensity[bleach_idx:]
            total_recovery_time = recovery_time[-1]
            
            # Define analysis window (final portion of recovery)
            start_time = total_recovery_time * params['recovery_start_fraction']
            analysis_start_idx = np.argmin(np.abs(recovery_time - start_time))
            
            # Use final portion for slope analysis
            window_fraction = params['analysis_window_fraction']
            window_start_time = total_recovery_time * (1 - window_fraction)
            window_start_idx = np.argmin(np.abs(recovery_time - window_start_time))
            
            # Analysis region
            analysis_time = recovery_time[window_start_idx:]
            analysis_intensity = recovery_intensity[window_start_idx:]
            
            if len(analysis_time) < 3:  # Need at least 3 points for linear fit
                return {
                    'has_downward_slope': False,
                    'slope': 0.0,
                    'r2': 0.0,
                    'confidence': 0.0,
                    'recommendation': 'insufficient_analysis_points',
                    'analysis_region': None,
                    'error': 'Insufficient points in analysis window'
                }
            
            # Linear fit to analysis region
            try:
                # Fit linear regression: intensity = slope * time + intercept
                coeffs = np.polyfit(analysis_time, analysis_intensity, 1)
                slope = coeffs[0]
                intercept = coeffs[1]
                
                # Calculate R²
                predicted = slope * analysis_time + intercept
                ss_res = np.sum((analysis_intensity - predicted) ** 2)
                ss_tot = np.sum((analysis_intensity - np.mean(analysis_intensity)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
            except Exception:
                return {
                    'has_downward_slope': False,
                    'slope': 0.0,
                    'r2': 0.0,
                    'confidence': 0.0,
                    'recommendation': 'fit_failed',
                    'analysis_region': None,
                    'error': 'Linear fit failed'
                }
            
            # Determine if slope is problematically negative
            has_downward_slope = (slope < params['slope_threshold'] and 
                                r2 > params['r2_threshold'])
            
            # Calculate confidence based on slope magnitude and fit quality
            slope_magnitude = abs(slope) if slope < 0 else 0
            confidence = min(1.0, (slope_magnitude / abs(params['slope_threshold'])) * r2)
            
            # Generate recommendation
            if has_downward_slope:
                if slope < params['slope_threshold'] * 3:  # Very steep decline
                    recommendation = 'exclude_strong'
                elif r2 > 0.9:  # High confidence linear decline
                    recommendation = 'exclude_confident'
                else:
                    recommendation = 'exclude_moderate'
            else:
                if slope > 0:  # Positive slope is good
                    recommendation = 'include_good'
                elif abs(slope) < abs(params['slope_threshold']) / 2:  # Very small slope
                    recommendation = 'include_stable'
                else:
                    recommendation = 'review_manual'
            
            # Prepare analysis region info
            analysis_region = {
                'time_start': recovery_time[window_start_idx] + time[bleach_idx],
                'time_end': recovery_time[-1] + time[bleach_idx],
                'intensity_start': analysis_intensity[0],
                'intensity_end': analysis_intensity[-1],
                'n_points': len(analysis_time),
                'window_fraction': window_fraction
            }
            
            return {
                'has_downward_slope': has_downward_slope,
                'slope': slope,
                'r2': r2,
                'confidence': confidence,
                'recommendation': recommendation,
                'analysis_region': analysis_region,
                'fit_params': {
                    'slope': slope,
                    'intercept': intercept,
                    'analysis_time': analysis_time + time[bleach_idx],  # Convert back to absolute time
                    'analysis_intensity': analysis_intensity,
                    'predicted_intensity': predicted
                }
            }
            
        except Exception as e:
            return {
                'has_downward_slope': False,
                'slope': 0.0,
                'r2': 0.0,
                'confidence': 0.0,
                'recommendation': 'error',
                'analysis_region': None,
                'error': str(e)
            }
    
    def analyze_group_slopes(self, group_data: Dict, data_manager) -> Dict:
        """
        Analyze slopes for all curves in a group.
        
        Parameters:
        -----------
        group_data : Dict
            Group data containing file paths
        data_manager : FRAPDataManager
            Data manager with loaded files
            
        Returns:
        --------
        Dict
            Results for all curves in the group
        """
        
        results = {}
        slope_issues = []
        
        for file_path in group_data['files']:
            try:
                file_data = data_manager.files[file_path]
                time = np.array(file_data['time'])
                intensity = np.array(file_data['intensity'])
                
                result = self.detect_downward_slope(time, intensity)
                results[file_path] = result
                
                # Track problematic curves
                if result['has_downward_slope']:
                    slope_issues.append({
                        'file_path': file_path,
                        'file_name': file_data['name'],
                        'slope': result['slope'],
                        'confidence': result['confidence'],
                        'recommendation': result['recommendation']
                    })
                    
            except Exception as e:
                results[file_path] = {
                    'has_downward_slope': False,
                    'slope': 0.0,
                    'error': str(e)
                }
        
        return {
            'individual_results': results,
            'slope_issues': slope_issues,
            'total_curves': len(group_data['files']),
            'problematic_curves': len(slope_issues),
            'exclusion_rate': len(slope_issues) / len(group_data['files']) if group_data['files'] else 0
        }
    
    def create_slope_visualization(self, file_data: Dict, result: Dict) -> go.Figure:
        """
        Create visualization showing the slope analysis for a single curve.
        
        Parameters:
        -----------
        file_data : Dict
            FRAP curve data
        result : Dict
            Slope detection result
            
        Returns:
        --------
        go.Figure
            Plotly figure showing the analysis
        """
        
        time = np.array(file_data['time'])
        intensity = np.array(file_data['intensity'])
        
        fig = go.Figure()
        
        # Plot full curve
        fig.add_trace(go.Scatter(
            x=time,
            y=intensity,
            mode='lines+markers',
            name='FRAP Curve',
            line=dict(color='blue', width=2),
            marker=dict(size=4)
        ))
        
        # Highlight analysis region if available
        if result.get('fit_params') and result.get('analysis_region'):
            fit_params = result['fit_params']
            analysis_region = result['analysis_region']
            
            # Show analysis window
            fig.add_trace(go.Scatter(
                x=fit_params['analysis_time'],
                y=fit_params['analysis_intensity'],
                mode='markers',
                name='Analysis Region',
                marker=dict(color='orange', size=6, symbol='square')
            ))
            
            # Show linear fit
            fig.add_trace(go.Scatter(
                x=fit_params['analysis_time'],
                y=fit_params['predicted_intensity'],
                mode='lines',
                name=f'Linear Fit (slope={result["slope"]:.4f})',
                line=dict(color='red', width=3, dash='dash')
            ))
            
            # Add shaded region
            fig.add_vrect(
                x0=analysis_region['time_start'],
                x1=analysis_region['time_end'],
                fillcolor="rgba(255, 0, 0, 0.1)",
                layer="below",
                line_width=0,
            )
        
        # Customize layout
        color = 'red' if result['has_downward_slope'] else 'green'
        title = f"Slope Analysis: {file_data['name']}"
        if result['has_downward_slope']:
            title += f" ⚠️ DOWNWARD SLOPE DETECTED"
        else:
            title += f" ✅ OK"
        
        fig.update_layout(
            title=title,
            xaxis_title="Time (s)",
            yaxis_title="Normalized Intensity",
            height=400,
            showlegend=True
        )
        
        return fig


def create_slope_detection_interface(group_data: Dict, data_manager, group_name: str):
    """
    Create Streamlit interface for automatic slope detection.
    
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
    st.markdown("### 📉 Automatic Downward Slope Detection")
    
    st.markdown("""
    **Automatically detect and exclude curves with problematic downward slopes in the recovery phase.**
    
    This typically indicates:
    - 📸 **Photobleaching** during recovery imaging
    - 🏃 **Cell movement** or sample drift
    - 🔬 **Focus drift** during acquisition
    - ⚡ **Continued bleaching** from imaging laser
    """)
    
    # Initialize detector
    detector = FRAPSlopeDetector()
    
    # Parameter customization
    with st.expander("🔧 Detection Parameters", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            slope_threshold = st.number_input(
                "Slope Threshold (per second):",
                value=-0.001,
                step=0.0001,
                format="%.4f",
                help="More negative = stricter detection"
            )
            
            recovery_start = st.slider(
                "Recovery Start Analysis (%):",
                10, 50, 30,
                help="Skip early recovery, analyze later phase"
            )
        
        with col2:
            analysis_window = st.slider(
                "Analysis Window (%):",
                20, 60, 40,
                help="Percentage of recovery phase to analyze"
            )
            
            r2_threshold = st.slider(
                "R² Threshold:",
                0.5, 0.95, 0.7, 0.05,
                help="Minimum fit quality for reliable detection"
            )
    
    # Custom parameters
    custom_params = {
        'slope_threshold': slope_threshold,
        'recovery_start_fraction': recovery_start / 100,
        'analysis_window_fraction': analysis_window / 100,
        'r2_threshold': r2_threshold,
        'min_recovery_points': 10
    }
    
    # Run slope analysis
    if st.button("🔍 Analyze Slopes", type="primary"):
        with st.spinner("Analyzing recovery slopes..."):
            group_results = detector.analyze_group_slopes(group_data, data_manager)
        
        st.success(f"✅ Analyzed {group_results['total_curves']} curves")
        
        # Summary results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Total Curves", 
                group_results['total_curves']
            )
        
        with col2:
            st.metric(
                "Problematic Slopes", 
                group_results['problematic_curves'],
                delta=f"{group_results['exclusion_rate']:.1%} of total"
            )
        
        with col3:
            curves_to_keep = group_results['total_curves'] - group_results['problematic_curves']
            st.metric(
                "Curves to Keep",
                curves_to_keep,
                delta=f"{curves_to_keep/group_results['total_curves']:.1%} of total"
            )
        
        # Show problematic curves
        if group_results['slope_issues']:
            st.markdown("#### 🚨 Curves with Downward Slopes")
            
            issues_df = pd.DataFrame(group_results['slope_issues'])
            issues_df['slope_per_min'] = issues_df['slope'] * 60  # Convert to per minute
            
            # Format the dataframe for display
            display_df = issues_df[['file_name', 'slope_per_min', 'confidence', 'recommendation']].copy()
            display_df.columns = ['File Name', 'Slope (per min)', 'Confidence', 'Recommendation']
            display_df['Slope (per min)'] = display_df['Slope (per min)'].round(4)
            display_df['Confidence'] = display_df['Confidence'].round(3)
            
            st.dataframe(display_df, width="stretch")
            
            # Apply automatic exclusion button
            if st.button("🚫 Auto-Exclude Slope Problems", type="secondary", 
                        help="Automatically exclude all curves with downward slopes"):
                # Get problematic file paths
                problematic_paths = [item['file_path'] for item in group_results['slope_issues']]
                
                # Update session state
                if 'interactive_excluded_files' not in st.session_state:
                    st.session_state.interactive_excluded_files = set()
                
                # Add problematic curves to exclusion set
                st.session_state.interactive_excluded_files.update(problematic_paths)
                
                # Update the group analysis
                data_manager.update_group_analysis(group_name, excluded_files=list(st.session_state.interactive_excluded_files))
                
                st.success(f"✅ Excluded {len(problematic_paths)} curves with downward slopes!")
                st.rerun()
        
        else:
            st.info("✅ No curves with problematic downward slopes detected!")
        
        # Individual curve inspection
        if st.checkbox("🔍 Inspect Individual Curves", help="View detailed slope analysis for each curve"):
            selected_file = st.selectbox(
                "Select curve to inspect:",
                options=list(group_data['files']),
                format_func=lambda x: data_manager.files[x]['name']
            )
            
            if selected_file:
                file_data = data_manager.files[selected_file]
                time = np.array(file_data['time'])
                intensity = np.array(file_data['intensity'])
                
                # Analyze this specific curve
                result = detector.detect_downward_slope(time, intensity, custom_params)
                
                # Create visualization
                fig = detector.create_slope_visualization(file_data, result)
                st.plotly_chart(fig, width="stretch")
                
                # Show detailed results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📊 Analysis Results:**")
                    st.write(f"**Slope:** {result['slope']:.6f} per second")
                    st.write(f"**Slope:** {result['slope']*60:.4f} per minute")
                    st.write(f"**R²:** {result['r2']:.3f}")
                    st.write(f"**Confidence:** {result['confidence']:.3f}")
                
                with col2:
                    st.markdown("**🎯 Recommendation:**")
                    
                    recommendation_text = {
                        'exclude_strong': '🚫 **EXCLUDE** - Strong downward slope',
                        'exclude_confident': '🚫 **EXCLUDE** - Confident downward trend', 
                        'exclude_moderate': '⚠️ **EXCLUDE** - Moderate downward slope',
                        'include_good': '✅ **INCLUDE** - Good recovery curve',
                        'include_stable': '✅ **INCLUDE** - Stable final phase',
                        'review_manual': '🤔 **REVIEW** - Manual inspection recommended',
                        'error': '❌ **ERROR** - Analysis failed'
                    }
                    
                    rec_text = recommendation_text.get(result['recommendation'], 'Unknown')
                    st.markdown(rec_text)
                    
                    if result.get('analysis_region'):
                        ar = result['analysis_region']
                        st.write(f"Analysis window: {ar['time_start']:.1f}-{ar['time_end']:.1f}s")
                        st.write(f"Points analyzed: {ar['n_points']}")


if __name__ == "__main__":
    # Test the slope detector
    detector = FRAPSlopeDetector()
    
    # Create test curve with downward slope
    time = np.linspace(0, 120, 61)
    
    # Normal recovery
    normal_curve = 0.3 + 0.6 * (1 - np.exp(-time/20))
    normal_curve[:10] = 1.0  # Pre-bleach
    normal_curve[10] = 0.2   # Bleach point
    
    # Add downward slope in final phase
    slope_curve = normal_curve.copy()
    slope_curve[45:] -= (time[45:] - time[45]) * 0.002  # Downward slope
    
    # Test detection
    normal_result = detector.detect_downward_slope(time, normal_curve)
    slope_result = detector.detect_downward_slope(time, slope_curve)
    
    print("Normal curve:", normal_result['has_downward_slope'], f"slope: {normal_result['slope']:.6f}")
    print("Slope curve:", slope_result['has_downward_slope'], f"slope: {slope_result['slope']:.6f}")