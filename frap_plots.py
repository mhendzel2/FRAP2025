"""
FRAP Plots Module
Generate interactive plots for FRAP data analysis
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import io
import base64
import logging

logger = logging.getLogger(__name__)

class FRAPPlots:
    @staticmethod
    def plot_raw_data(df, height=400):
        """
        Plot raw FRAP data (all ROIs)
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with raw FRAP data
        height : int
            Height of the plot
            
        Returns:
        --------
        plotly.graph_objects.Figure
            Plotly figure with raw data plot
        """
        try:
            fig = go.Figure()
            
            # Plot ROI1 (Signal)
            fig.add_trace(go.Scatter(
                x=df['time'], 
                y=df['ROI1'],
                mode='lines+markers',
                name='ROI1 (Signal)',
                line=dict(color='#4f46e5', width=2),
                marker=dict(size=6)
            ))
            
            # Plot ROI2 (Reference)
            fig.add_trace(go.Scatter(
                x=df['time'], 
                y=df['ROI2'],
                mode='lines+markers',
                name='ROI2 (Reference)',
                line=dict(color='#8b5cf6', width=2),
                marker=dict(size=6)
            ))
            
            # Plot ROI3 (Background)
            fig.add_trace(go.Scatter(
                x=df['time'], 
                y=df['ROI3'],
                mode='lines+markers',
                name='ROI3 (Background)',
                line=dict(color='#94a3b8', width=2, dash='dot'),
                marker=dict(size=6)
            ))
            
            # Add a vertical line at t=0 (bleach point)
            fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="gray")
            
            fig.update_layout(
                title="Raw FRAP Data",
                xaxis_title="Time",
                yaxis_title="Intensity",
                height=height,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                margin=dict(l=10, r=10, t=50, b=10),
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting raw data: {e}")
            return None
    
    @staticmethod
    def plot_corrected_data(df, height=400):
        """
        Plot background-corrected FRAP data
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with processed FRAP data
        height : int
            Height of the plot
            
        Returns:
        --------
        plotly.graph_objects.Figure
            Plotly figure with corrected data plot
        """
        try:
            fig = go.Figure()
            
            # Plot ROI1 corrected
            fig.add_trace(go.Scatter(
                x=df['time'], 
                y=df['ROI1_corr'],
                mode='lines+markers',
                name='ROI1 (corrected)',
                line=dict(color='#4f46e5', width=2),
                marker=dict(size=6)
            ))
            
            # Plot ROI2 corrected
            fig.add_trace(go.Scatter(
                x=df['time'], 
                y=df['ROI2_corr'],
                mode='lines+markers',
                name='ROI2 (corrected)',
                line=dict(color='#8b5cf6', width=2),
                marker=dict(size=6)
            ))
            
            # Add a vertical line at t=0 (bleach point)
            fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="gray")
            
            fig.update_layout(
                title="Background-Corrected FRAP Data",
                xaxis_title="Time",
                yaxis_title="Intensity (background subtracted)",
                height=height,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                margin=dict(l=10, r=10, t=50, b=10),
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting corrected data: {e}")
            return None
    
    @staticmethod
    def plot_normalized_data(df, height=400):
        """
        Plot normalized FRAP data
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with processed FRAP data
        height : int
            Height of the plot
            
        Returns:
        --------
        plotly.graph_objects.Figure
            Plotly figure with normalized data plot
        """
        try:
            fig = go.Figure()
            
            # Plot normalized data
            fig.add_trace(go.Scatter(
                x=df['time'], 
                y=df['normalized'],
                mode='lines+markers',
                name='Normalized intensity',
                line=dict(color='#4f46e5', width=2),
                marker=dict(size=6)
            ))
            
            # Add a vertical line at t=0 (bleach point)
            fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="gray")
            
            # Add a horizontal line at y=1 (pre-bleach level)
            fig.add_hline(y=1, line_width=1, line_dash="dash", line_color="gray")
            
            fig.update_layout(
                title="Normalized FRAP Data",
                xaxis_title="Time",
                yaxis_title="Normalized Intensity",
                height=height,
                margin=dict(l=10, r=10, t=50, b=10),
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting normalized data: {e}")
            return None
    
    @staticmethod
    def plot_fit_curves(time, intensity, fits, height=500, highlight_model=None):
        """
        Plot FRAP data with fitted curves
        
        Parameters:
        -----------
        time : numpy.ndarray
            Time values
        intensity : numpy.ndarray
            Intensity values
        fits : list
            List of fit dictionaries
        height : int
            Height of the plot
        highlight_model : str
            Model name to highlight (e.g., 'single', 'double', 'triple')
            
        Returns:
        --------
        plotly.graph_objects.Figure
            Plotly figure with fit curves
        """
        try:
            # Get post-bleach data for fitting
            from frap_core_corrected import FRAPAnalysisCore
            t_post, i_post, i_min = FRAPAnalysisCore.get_post_bleach_data(time, intensity)
            
            # Create figure
            fig = go.Figure()
            
            # Plot original data
            fig.add_trace(go.Scatter(
                x=time, 
                y=intensity,
                mode='markers',
                name='Data',
                marker=dict(size=8, color='#4f46e5')
            ))
            
            # Plot extrapolated starting value (connecting pre-bleach and post-bleach)
            if i_min > 0:
                # Connect the last pre-bleach point to the first post-bleach point
                fig.add_trace(go.Scatter(
                    x=[time[i_min-1], time[i_min]],
                    y=[intensity[i_min-1], intensity[i_min]],
                    mode='lines',
                    line=dict(color='#94a3b8', width=2, dash='dash'),
                    name='Bleach event'
                ))
            
            # Plot each fit
            colors = ['#4f46e5', '#8b5cf6', '#ec4899', '#f97316']
            for i, fit in enumerate(fits):
                # Generate smooth curve for plotting
                t_smooth = np.linspace(min(t_post), max(t_post), 100)
                y_smooth = fit['func'](t_smooth, *fit['params'])
                
                # Add model curve
                fig.add_trace(go.Scatter(
                    x=t_smooth,
                    y=y_smooth,
                    mode='lines',
                    name=f"{fit['model']} component fit<br>R² = {fit['r2']:.4f}, AIC = {fit['aic']:.2f}",
                    line=dict(color=colors[i % len(colors)], width=3)
                ))
            
            # Add a vertical line at t=0 (bleach point)
            fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="gray")
            
            # Add a horizontal line at y=1 (pre-bleach level)
            fig.add_hline(y=1, line_width=1, line_dash="dash", line_color="gray")
            
            fig.update_layout(
                title="FRAP Recovery Curves with Model Fits",
                xaxis_title="Time",
                yaxis_title="Normalized Intensity",
                height=height,
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="right",
                    x=1.15
                ),
                margin=dict(l=10, r=120, t=50, b=10),
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting fit curves: {e}")
            return None
    
    @staticmethod
    def plot_residuals(time, intensity, best_fit, height=300):
        """
        Plot residuals for the best fit model
        
        Parameters:
        -----------
        time : numpy.ndarray
            Time values
        intensity : numpy.ndarray
            Intensity values
        best_fit : dict
            Dictionary with best fit information
        height : int
            Height of the plot
            
        Returns:
        --------
        plotly.graph_objects.Figure
            Plotly figure with residuals plot
        """
        try:
            # Get post-bleach data for fitting
            from frap_core_corrected import FRAPAnalysisCore
            t_post, i_post, i_min = FRAPAnalysisCore.get_post_bleach_data(time, intensity)
            
            # Calculate residuals
            y_fit = best_fit['fitted_values']
            residuals = i_post - y_fit
            
            # Create figure
            fig = go.Figure()
            
            # Plot residuals
            fig.add_trace(go.Scatter(
                x=t_post,
                y=residuals,
                mode='markers',
                name='Residuals',
                marker=dict(size=8, color='#4f46e5')
            ))
            
            # Add a horizontal line at y=0
            fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="gray")
            
            fig.update_layout(
                title=f"Residuals for {best_fit['model']} component fit",
                xaxis_title="Time",
                yaxis_title="Residual",
                height=height,
                margin=dict(l=10, r=10, t=50, b=10),
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting residuals: {e}")
            return None
    
    @staticmethod
    def plot_group_comparison(data_manager, group_names, feature='half_time', height=500):
        """
        Plot comparison of a specific feature across groups
        
        Parameters:
        -----------
        data_manager : FRAPData
            Data manager object
        group_names : list
            List of group names to compare
        feature : str
            Feature to compare
        height : int
            Height of the plot
            
        Returns:
        --------
        plotly.graph_objects.Figure
            Plotly figure with group comparison
        """
        try:
            # Get data for each group
            data = []
            
            for group_name in group_names:
                if group_name in data_manager.groups:
                    group = data_manager.groups[group_name]
                    if group['features_df'] is not None and feature in group['features_df'].columns:
                        group_data = group['features_df'][feature].dropna()
                        for value in group_data:
                            data.append({
                                'Group': group_name,
                                'Value': value
                            })
            
            if not data:
                logger.error(f"No data available for feature {feature} in the selected groups")
                return None
                
            # Create DataFrame
            df = pd.DataFrame(data)
            
            # Create figure
            fig = px.box(
                df, 
                x='Group', 
                y='Value',
                title=f"Comparison of {feature} across groups",
                color='Group',
                points='all',
                height=height,
            )
            
            fig.update_layout(
                yaxis_title=feature,
                margin=dict(l=10, r=10, t=50, b=10),
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting group comparison: {e}")
            return None
    
    @staticmethod
    def plot_multiple_curves(data_manager, group_name, height=500):
        """
        Plot multiple FRAP curves from a group
        
        Parameters:
        -----------
        data_manager : FRAPData
            Data manager object
        group_name : str
            Name of the group
        height : int
            Height of the plot
            
        Returns:
        --------
        plotly.graph_objects.Figure
            Plotly figure with multiple curves
        """
        try:
            if group_name not in data_manager.groups:
                logger.error(f"Group {group_name} does not exist")
                return None
                
            group = data_manager.groups[group_name]
            
            # Create figure
            fig = go.Figure()
            
            # Plot each curve
            for file_path in group['files']:
                if file_path in data_manager.files:
                    file_data = data_manager.files[file_path]
                    
                    # Plot normalized data
                    fig.add_trace(go.Scatter(
                        x=file_data['time'], 
                        y=file_data['intensity'],
                        mode='lines+markers',
                        name=file_data['name'],
                        marker=dict(size=4),
                        line=dict(width=1.5)
                    ))
            
            # Add a vertical line at t=0 (bleach point)
            fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="gray")
            
            # Add a horizontal line at y=1 (pre-bleach level)
            fig.add_hline(y=1, line_width=1, line_dash="dash", line_color="gray")
            
            fig.update_layout(
                title=f"FRAP Recovery Curves for Group: {group_name}",
                xaxis_title="Time",
                yaxis_title="Normalized Intensity",
                height=height,
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.05
                ),
                margin=dict(l=10, r=120, t=50, b=10),
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting multiple curves: {e}")
            return None
    
    @staticmethod
    def plot_feature_distributions(data_manager, group_name, height=400):
        """
        Plot distributions of features from a group
        
        Parameters:
        -----------
        data_manager : FRAPData
            Data manager object
        group_name : str
            Name of the group
        height : int
            Height of the plot
            
        Returns:
        --------
        plotly.graph_objects.Figure
            Plotly figure with feature distributions
        """
        try:
            if group_name not in data_manager.groups:
                logger.error(f"Group {group_name} does not exist")
                return None
                
            group = data_manager.groups[group_name]
            
            if group['features_df'] is None or len(group['features_df']) == 0:
                logger.error(f"No feature data available for group {group_name}")
                return None
                
            # Select key features
            features = ['mobile_fraction', 'half_time']
            if 'amplitude' in group['features_df']:
                features.append('amplitude')
            if 'rate_constant' in group['features_df']:
                features.append('rate_constant')
            
            # Select only numerical features that exist in the DataFrame
            numerical_features = [f for f in features if f in group['features_df'].columns]
            
            if not numerical_features:
                logger.error(f"No numerical features available for group {group_name}")
                return None
            
            # Create subplots
            fig = make_subplots(
                rows=len(numerical_features), 
                cols=1,
                subplot_titles=[f.replace('_', ' ').title() for f in numerical_features]
            )
            
            for i, feature in enumerate(numerical_features):
                # Get values and filter out NaNs
                values = group['features_df'][feature].dropna()
                
                # Add histogram
                fig.add_trace(
                    go.Histogram(
                        x=values,
                        opacity=0.7,
                        marker_color='#4f46e5'
                    ),
                    row=i+1, 
                    col=1
                )
                
                # Add mean line
                mean_value = values.mean()
                fig.add_vline(
                    x=mean_value, 
                    line_width=2, 
                    line_dash="dash", 
                    line_color="red",
                    row=i+1, 
                    col=1
                )
            
            fig.update_layout(
                title=f"Feature Distributions for Group: {group_name}",
                height=height * len(numerical_features),
                showlegend=False,
                margin=dict(l=10, r=10, t=50, b=10),
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting feature distributions: {e}")
            return None
    
    @staticmethod
    def plot_group_average_curve(data_manager, group_name, height=500):
        """
        Plot average FRAP curve for a group with the optimal fit model
        
        Parameters:
        -----------
        data_manager : FRAPData
            Data manager object
        group_name : str
            Name of the group
        height : int
            Height of the plot
            
        Returns:
        --------
        plotly.graph_objects.Figure
            Plotly figure with average curve and optimal fit model
        """
        try:
            if group_name not in data_manager.groups:
                logger.error(f"Group {group_name} does not exist")
                return None
                
            group = data_manager.groups[group_name]
            
            if not group['files']:
                logger.error(f"No files in group {group_name}")
                return None
            
            # Get common time points (post-bleach only)
            common_times = None
            all_intensities = []
            
            # Track models used for best fits
            model_counts = {"single": 0, "double": 0, "triple": 0}
            
            for file_path in group['files']:
                if file_path in data_manager.files:
                    file_data = data_manager.files[file_path]
                    
                    # Get post-bleach data
                    from frap_core_corrected import FRAPAnalysisCore
                    t_post, i_post, _ = FRAPAnalysisCore.get_post_bleach_data(
                        file_data['time'], 
                        file_data['intensity']
                    )
                    
                    # Store data
                    if common_times is None:
                        common_times = t_post
                    else:
                        # Use the shortest time series as the common time base
                        if len(t_post) < len(common_times):
                            common_times = t_post
                    
                    all_intensities.append(i_post)
                    
                    # Count model types
                    if 'best_fit' in file_data and file_data['best_fit']:
                        model = file_data['best_fit']['model']
                        model_counts[model] = model_counts.get(model, 0) + 1
            
            if not all_intensities or common_times is None:
                logger.error(f"No intensity data available for group {group_name}")
                return None
            
            # Resize all intensity arrays to match common time points
            min_length = min(len(arr) for arr in all_intensities)
            if min_length < len(common_times):
                common_times = common_times[:min_length]
                all_intensities = [arr[:min_length] for arr in all_intensities]
            
            # Convert to numpy array
            all_intensities = np.array(all_intensities)
            
            # Calculate mean and standard deviation
            mean_intensity = np.mean(all_intensities, axis=0)
            std_intensity = np.std(all_intensities, axis=0)
            
            # Create figure
            fig = go.Figure()
            
            # Plot mean curve with markers
            fig.add_trace(go.Scatter(
                x=common_times, 
                y=mean_intensity,
                mode='markers+lines',
                name='Mean (Data)',
                line=dict(color='#4f46e5', width=2),
                marker=dict(size=6, color='#4f46e5')
            ))
            
            # Plot confidence interval
            fig.add_trace(go.Scatter(
                x=np.concatenate([common_times, common_times[::-1]]) if len(common_times) > 0 else [],
                y=np.concatenate([mean_intensity + std_intensity, (mean_intensity - std_intensity)[::-1]]) if len(mean_intensity) > 0 else [],
                fill='toself',
                fillcolor='rgba(79, 70, 229, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Mean ± SD'
            ))
            
            # Determine the dominant model
            dominant_model = None
            max_count = 0
            for model, count in model_counts.items():
                if count > max_count:
                    max_count = count
                    dominant_model = model
            
            # If we have a dominant model, fit the average data with it
            if dominant_model is not None and max_count > 0:
                from frap_core_corrected import FRAPAnalysisCore
                
                # Fit the average data with all models and select the dominant model type
                fits = FRAPAnalysisCore.fit_all_models(common_times, mean_intensity)
                
                # Find the fit that matches the dominant model
                dominant_fit = None
                for fit in fits:
                    if fit['model'] == dominant_model:
                        dominant_fit = fit
                        break
                
                if dominant_fit:
                    # Create smooth time series for plotting the fit
                    smooth_time = np.linspace(common_times[0], common_times[-1], 100)
                    
                    # Get fit function and parameters
                    fit_func = dominant_fit['func']
                    params = dominant_fit['params']
                    
                    # Calculate fitted values for smooth curve
                    smooth_fit = fit_func(smooth_time, *params)
                    
                    # Add fit curve
                    model_names = {
                        'single': 'Single-Component Model',
                        'double': 'Two-Component Model',
                        'triple': 'Three-Component Model'
                    }
                    
                    fig.add_trace(go.Scatter(
                        x=smooth_time,
                        y=smooth_fit,
                        mode='lines',
                        name=model_names.get(dominant_model, dominant_model),
                        line=dict(color='#ef4444', width=3)
                    ))
                    
                    # Extract model parameters for annotation
                    if dominant_model == 'single':
                        A, k, C = params
                        mobile_fraction = A / (1.0 - C) if C < 1.0 else np.nan
                        half_time = np.log(2) / k if k > 0 else np.nan
                        annotation_text = f"Mobile Fraction: {mobile_fraction*100:.1f}%<br>Half-time: {half_time:.2f}s"
                    elif dominant_model == 'double':
                        A1, k1, A2, k2, C = params
                        # Sort components (fast to slow)
                        if k1 < k2:
                            k1, k2 = k2, k1
                            A1, A2 = A2, A1
                        total_amp = A1 + A2
                        mobile_fraction = total_amp / (1.0 - C) if C < 1.0 else np.nan
                        fast_half_time = np.log(2) / k1 if k1 > 0 else np.nan
                        slow_half_time = np.log(2) / k2 if k2 > 0 else np.nan
                        annotation_text = f"Mobile Fraction: {mobile_fraction*100:.1f}%<br>Fast t½: {fast_half_time:.2f}s ({A1/total_amp*100:.1f}%)<br>Slow t½: {slow_half_time:.2f}s ({A2/total_amp*100:.1f}%)"
                    elif dominant_model == 'triple':
                        A1, k1, A2, k2, A3, k3, C = params
                        # Sort components by rate (fast to slow)
                        components = sorted([(k1, A1), (k2, A2), (k3, A3)], reverse=True)
                        ks = [comp[0] for comp in components]
                        As = [comp[1] for comp in components]
                        total_amp = sum(As)
                        mobile_fraction = total_amp / (1.0 - C) if C < 1.0 else np.nan
                        half_times = [np.log(2) / k if k > 0 else np.nan for k in ks]
                        props = [A / total_amp * 100 if total_amp > 0 else np.nan for A in As]
                        annotation_text = f"Mobile Fraction: {mobile_fraction*100:.1f}%<br>Fast t½: {half_times[0]:.2f}s ({props[0]:.1f}%)<br>Medium t½: {half_times[1]:.2f}s ({props[1]:.1f}%)<br>Slow t½: {half_times[2]:.2f}s ({props[2]:.1f}%)"
                    
                    # Add annotation with model parameters
                    fig.add_annotation(
                        x=0.95,
                        y=0.95,
                        xref="paper",
                        yref="paper",
                        text=f"<b>{model_names.get(dominant_model, dominant_model)}</b><br>{annotation_text}",
                        showarrow=False,
                        font=dict(
                            family="Arial",
                            size=12,
                            color="#ef4444"
                        ),
                        align="right",
                        bgcolor="rgba(255, 255, 255, 0.8)",
                        bordercolor="#ef4444",
                        borderwidth=1,
                        borderpad=4
                    )
            
            # Add a horizontal line at y=1 (pre-bleach level)
            fig.add_hline(y=1, line_width=1, line_dash="dash", line_color="gray")
            
            fig.update_layout(
                title=f"Average FRAP Recovery Curve for Group: {group_name}",
                xaxis_title="Time (s)",
                yaxis_title="Normalized Intensity",
                height=height,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                margin=dict(l=10, r=10, t=50, b=10),
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting group average curve: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
