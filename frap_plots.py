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
        Plot FRAP data with fitted curves, incorporating visualization fixes.
        
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
            from frap_core import FRAPAnalysisCore
            
            # Get post-bleach data for fitting
            t_post_fit, i_post_fit, bleach_idx = FRAPAnalysisCore.get_post_bleach_data(time, intensity)
            
            # Calculate interpolated bleach time for plotting on original time scale
            if bleach_idx > 0:
                interpolated_bleach_time = (time[bleach_idx-1] + time[bleach_idx]) / 2.0
            else:
                interpolated_bleach_time = time[bleach_idx]
            
            # Create figure
            fig = go.Figure()
            
            # 1. Plot pre-bleach data (gray markers)
            pre_bleach_time = time[:bleach_idx]
            pre_bleach_intensity = intensity[:bleach_idx]
            if len(pre_bleach_time) > 0:
                fig.add_trace(go.Scatter(
                    x=pre_bleach_time, 
                    y=pre_bleach_intensity,
                    mode='markers',
                    name='Pre-bleach (not fitted)',
                    marker=dict(color='lightgray', size=6, opacity=0.7)
                ))

            # Convert post-bleach time back to original scale for plotting
            t_post_plot = t_post_fit + interpolated_bleach_time

            # 2. Plot post-bleach data, highlighting the interpolated first point.
            # Plot the rest of the post-bleach data points as standard blue dots.
            if len(t_post_plot) > 1:
                fig.add_trace(go.Scatter(
                    x=t_post_plot[1:],
                    y=i_post_fit[1:],
                    mode='markers',
                    name='Post-bleach Data',
                    marker=dict(
                        size=8,
                        color='#4f46e5'  # Standard blue color
                    )
                ))

            # Plot the interpolated point with a distinct, highlighted marker.
            # This point is plotted last to appear on top.
            if len(t_post_plot) > 0:
                fig.add_trace(go.Scatter(
                    x=[t_post_plot[0]],
                    y=[i_post_fit[0]],
                    mode='markers',
                    name='Interpolated Start',
                    marker=dict(
                        symbol='diamond',
                        color='#10b981',  # A distinct green for highlighting
                        size=12,
                        line=dict(color='black', width=1)
                    )
                ))
            
            # Plot each fit
            colors = ['#ef4444', '#ec4899', '#f97316', '#8b5cf6']
            for i, fit in enumerate(fits):
                # Generate smooth curve for plotting using fitted_values if available
                if 'fitted_values' in fit and fit['fitted_values'] is not None:
                    # Use pre-computed fitted values if available
                    y_smooth = fit['fitted_values']
                    t_smooth_fit = t_post_fit
                else:
                    # Fallback: try to compute using function if available
                    if 'func' in fit and fit['func'] is not None:
                        t_smooth_fit = np.linspace(min(t_post_fit), max(t_post_fit), 200)
                        y_smooth = fit['func'](t_smooth_fit, *fit['params'])
                    else:
                        # Skip this fit if we can't generate the curve
                        continue
                
                # Convert smooth time to original scale for plotting
                t_smooth_plot = t_smooth_fit + interpolated_bleach_time
                
                # Highlight the selected model
                is_highlighted = highlight_model and fit['model'] == highlight_model
                line_width = 4 if is_highlighted else 2.5
                opacity = 1.0 if is_highlighted else 0.8
                
                # Add model curve
                fig.add_trace(go.Scatter(
                    x=t_smooth_plot,
                    y=y_smooth,
                    mode='lines',
                    name=f"{fit['model']} fit (R²={fit['r2']:.4f})",
                    line=dict(color=colors[i % len(colors)], width=line_width, dash='solid'),
                    opacity=opacity
                ))
            
            # Add a vertical line at the interpolated bleach time
            max_intensity_val = np.max(intensity) * 1.05
            fig.add_shape(
                type="line",
                x0=interpolated_bleach_time, y0=0,
                x1=interpolated_bleach_time, y1=max_intensity_val,
                line=dict(color="orange", width=2, dash="dash"),
            )
            fig.add_annotation(
                x=interpolated_bleach_time, y=max_intensity_val * 0.9,
                text="Bleach Event", showarrow=True, arrowhead=1, ax=20, ay=-30
            )
            
            fig.update_layout(
                title="FRAP Recovery Curves with Model Fits",
                xaxis_title="Time (s)",
                yaxis_title="Normalized Intensity",
                height=height,
                yaxis=dict(range=[0, max_intensity_val]), # Force y-axis to start at 0
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
            from frap_core import FRAPAnalysisCore
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
                    from frap_core import FRAPAnalysisCore
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
                from frap_core import FRAPAnalysisCore
                
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
                    
                    # Use fitted_values if available, otherwise try to reconstruct the fit
                    if 'fitted_values' in dominant_fit and dominant_fit['fitted_values'] is not None:
                        # Use pre-computed fitted values, interpolated to smooth_time
                        from scipy.interpolate import interp1d
                        interp_func = interp1d(common_times, dominant_fit['fitted_values'], 
                                             kind='linear', bounds_error=False, fill_value='extrapolate')
                        smooth_fit = interp_func(smooth_time)
                    elif 'func' in dominant_fit and dominant_fit['func'] is not None:
                        # Fallback: use function if available
                        fit_func = dominant_fit['func']
                        params = dominant_fit['params']
                        smooth_fit = fit_func(smooth_time, *params)
                    else:
                        # Skip plotting fit curve if we can't generate it
                        smooth_fit = None
                    
                    # Add fit curve if we have one
                    if smooth_fit is not None:
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
                        endpoint = A + C
                        mobile_fraction = (1 - endpoint) if np.isfinite(endpoint) else np.nan
                        half_time = np.log(2) / k if k > 0 else np.nan
                        annotation_text = f"Mobile Pop.: {mobile_fraction*100:.1f}%<br>Half-time: {half_time:.2f}s"
                    elif dominant_model == 'double':
                        A1, k1, A2, k2, C = params
                        # Sort components (fast to slow)
                        if k1 < k2:
                            k1, k2 = k2, k1
                            A1, A2 = A2, A1
                        total_amp = A1 + A2
                        endpoint = total_amp + C
                        mobile_fraction = (1 - endpoint) if np.isfinite(endpoint) else np.nan
                        fast_half_time = np.log(2) / k1 if k1 > 0 else np.nan
                        slow_half_time = np.log(2) / k2 if k2 > 0 else np.nan
                        annotation_text = f"Mobile Pop.: {mobile_fraction*100:.1f}%<br>Fast t½: {fast_half_time:.2f}s ({A1/total_amp*100:.1f}%)<br>Slow t½: {slow_half_time:.2f}s ({A2/total_amp*100:.1f}%)"
                    elif dominant_model == 'triple':
                        A1, k1, A2, k2, A3, k3, C = params
                        # Sort components by rate (fast to slow)
                        components = sorted([(k1, A1), (k2, A2), (k3, A3)], reverse=True)
                        ks = [comp[0] for comp in components]
                        As = [comp[1] for comp in components]
                        total_amp = sum(As)
                        endpoint = total_amp + C
                        mobile_fraction = (1 - endpoint) if np.isfinite(endpoint) else np.nan
                        half_times = [np.log(2) / k if k > 0 else np.nan for k in ks]
                        props = [A / total_amp * 100 if total_amp > 0 else np.nan for A in As]
                        annotation_text = f"Mobile Pop.: {mobile_fraction*100:.1f}%<br>Fast t½: {half_times[0]:.2f}s ({props[0]:.1f}%)<br>Medium t½: {half_times[1]:.2f}s ({props[1]:.1f}%)<br>Slow t½: {half_times[2]:.2f}s ({props[2]:.1f}%)"
                    
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

    @staticmethod
    def plot_comprehensive_fit(time, intensity, fit_result, file_name="", height=800):
        """
        Create a comprehensive multi-panel plot showing:
        - Top panel: Recovery curve with fitted line
        - Middle panel: Residuals
        - Bottom panel: Parameter information (simplified)
        
        Parameters:
        -----------
        time : np.ndarray
            Time points
        intensity : np.ndarray
            Normalized intensity values
        fit_result : dict
            Dictionary containing fit results with keys:
            - 'model': Model type 
            - 'fitted_values': Fitted curve values
            - 'r2': R-squared value
            - Additional fit parameters as available
        file_name : str
            Name of the file being analyzed
        height : int
            Total height of the plot
            
        Returns:
        --------
        plotly.graph_objects.Figure
            Multi-panel Plotly figure
        """
        try:
            if fit_result is None:
                logger.error("No fit_result provided")
                return None
                
            model_name = fit_result.get('model', 'Unknown')
            fitted_values = fit_result.get('fitted_values')
            r2 = fit_result.get('r2', 0)
            
            if fitted_values is None:
                logger.error("No fitted_values found in fit_result")
                return None
            
            # Create 2-panel subplot (data + residuals)
            fig = make_subplots(
                rows=2,
                cols=1,
                row_heights=[0.7, 0.3],
                subplot_titles=["Recovery Curve with Fit", "Residuals"],
                vertical_spacing=0.08,
                shared_xaxes=True
            )
            
            # Panel 1: Data and fit
            fig.add_trace(
                go.Scatter(x=time, y=intensity, mode='markers', name='Data',
                          marker=dict(size=8, color='#3b82f6', line=dict(width=1, color='white'))),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=time, y=fitted_values, mode='lines', name=f'{model_name.title()} Fit',
                          line=dict(color='#ef4444', width=3)),
                row=1, col=1
            )
            
            # Panel 2: Residuals (only if we have enough data)
            if len(fitted_values) == len(intensity):
                residuals = intensity - fitted_values
                fig.add_trace(
                    go.Scatter(x=time, y=residuals, mode='markers', name='Residuals',
                              marker=dict(size=6, color='#8b5cf6')),
                    row=2, col=1
                )
                # Add zero line for residuals
                fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="gray", row=2, col=1)
            else:
                # Add placeholder text if residuals can't be computed
                fig.add_annotation(
                    x=0.5, y=0.5,
                    xref="x domain", yref="y domain",
                    text="Residuals not available",
                    showarrow=False,
                    row=2, col=1
                )
            
            # Add parameter annotation to top panel
            aic = fit_result.get('aic', np.nan)
            bic = fit_result.get('bic', np.nan)
            
            annotation_text = (f"<b>{model_name.title()} Model</b><br>"
                             f"R² = {r2:.4f}<br>"
                             f"AIC = {aic:.2f}<br>" if np.isfinite(aic) else f"AIC = N/A<br>"
                             f"BIC = {bic:.2f}" if np.isfinite(bic) else f"BIC = N/A")
            
            fig.add_annotation(
                x=0.02, y=0.98,
                xref="x domain", yref="y domain",
                text=annotation_text,
                showarrow=False,
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor="gray",
                borderwidth=1,
                borderpad=8,
                align="left",
                font=dict(size=10),
                row=1, col=1
            )
            
            # Update layout
            fig.update_xaxes(title_text="Time (s)", row=2, col=1)
            fig.update_yaxes(title_text="Normalized Intensity", row=1, col=1)
            fig.update_yaxes(title_text="Residual", row=2, col=1)
            
            fig.update_layout(
                title=f"Comprehensive FRAP Analysis: {file_name}" if file_name else "Comprehensive FRAP Analysis",
                height=height,
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=0.99,
                    xanchor="right",
                    x=0.99,
                    bgcolor="rgba(255, 255, 255, 0.8)"
                ),
                margin=dict(l=60, r=20, t=80, b=50)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating comprehensive fit plot: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    @staticmethod
    def plot_estimation_plot(data_df, group_col='group', value_col='value', 
                            group_names=None, title="Estimation Plot", height=600):
        """
        Create an estimation plot showing raw data, means, and confidence intervals.
        
        An estimation plot is superior to traditional bar charts because it shows:
        1. All raw data points (transparency about sample distribution)
        2. Mean with confidence interval (statistical precision)
        3. Effect size visualization (magnitude of differences)
        
        Parameters:
        -----------
        data_df : pd.DataFrame
            DataFrame with group and value columns
        group_col : str
            Column name containing group labels
        value_col : str
            Column name containing numeric values
        group_names : list, optional
            Specific groups to plot (default: all groups)
        title : str
            Plot title
        height : int
            Plot height in pixels
            
        Returns:
        --------
        plotly.graph_objects.Figure
            Estimation plot figure
        """
        try:
            import scipy.stats as stats
            
            if group_names is None:
                group_names = data_df[group_col].unique().tolist()
            
            # Filter data
            plot_data = data_df[data_df[group_col].isin(group_names)].copy()
            
            # Create figure with two subplots: raw data on left, mean differences on right
            fig = make_subplots(
                rows=1, cols=2,
                column_widths=[0.6, 0.4],
                subplot_titles=["Raw Data with Mean ± 95% CI", "Mean Differences"],
                horizontal_spacing=0.15
            )
            
            # Color palette
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            
            # Calculate statistics for each group
            group_stats = {}
            for i, group_name in enumerate(group_names):
                group_data = plot_data[plot_data[group_col] == group_name][value_col].dropna()
                
                if len(group_data) > 0:
                    mean_val = group_data.mean()
                    sem_val = group_data.sem()
                    n = len(group_data)
                    
                    # 95% confidence interval
                    if n > 1:
                        ci_95 = stats.t.ppf(0.975, n-1) * sem_val
                    else:
                        ci_95 = 0
                    
                    group_stats[group_name] = {
                        'data': group_data,
                        'mean': mean_val,
                        'sem': sem_val,
                        'ci_95': ci_95,
                        'n': n,
                        'color': colors[i % len(colors)]
                    }
            
            # Left panel: Raw data with means and CIs
            x_offset = 0
            x_spacing = 1.5
            
            for i, group_name in enumerate(group_names):
                if group_name not in group_stats:
                    continue
                
                stats_dict = group_stats[group_name]
                x_pos = x_offset + i * x_spacing
                
                # Add jittered raw data points
                np.random.seed(42)  # For reproducibility
                jitter = np.random.normal(0, 0.1, len(stats_dict['data']))
                x_jittered = [x_pos] * len(stats_dict['data']) + jitter
                
                fig.add_trace(
                    go.Scatter(
                        x=x_jittered,
                        y=stats_dict['data'].values,
                        mode='markers',
                        name=f'{group_name} (raw)',
                        marker=dict(
                            size=8,
                            color=stats_dict['color'],
                            opacity=0.4,
                            line=dict(width=0.5, color='white')
                        ),
                        showlegend=True,
                        hovertemplate=f'<b>{group_name}</b><br>Value: %{{y:.3f}}<extra></extra>'
                    ),
                    row=1, col=1
                )
                
                # Add mean with CI error bar
                fig.add_trace(
                    go.Scatter(
                        x=[x_pos + 0.4],
                        y=[stats_dict['mean']],
                        error_y=dict(
                            type='data',
                            array=[stats_dict['ci_95']],
                            visible=True,
                            thickness=3,
                            width=8
                        ),
                        mode='markers',
                        name=f'{group_name} (mean ± 95% CI)',
                        marker=dict(
                            size=16,
                            color=stats_dict['color'],
                            symbol='diamond',
                            line=dict(width=2, color='black')
                        ),
                        showlegend=True,
                        hovertemplate=f'<b>{group_name} Mean</b><br>' + 
                                     f'Value: {stats_dict["mean"]:.3f}<br>' +
                                     f'95% CI: ±{stats_dict["ci_95"]:.3f}<br>' +
                                     f'N: {stats_dict["n"]}<extra></extra>'
                    ),
                    row=1, col=1
                )
            
            # Right panel: Effect size visualization (pairwise mean differences)
            if len(group_names) >= 2:
                # For simplicity, show differences relative to first group
                reference_group = group_names[0]
                
                if reference_group in group_stats:
                    ref_mean = group_stats[reference_group]['mean']
                    
                    y_positions = []
                    differences = []
                    ci_errors = []
                    labels = []
                    marker_colors = []
                    
                    for i, group_name in enumerate(group_names[1:], 1):
                        if group_name not in group_stats:
                            continue
                        
                        # Calculate difference
                        diff = group_stats[group_name]['mean'] - ref_mean
                        
                        # Pooled SEM for difference
                        sem_diff = np.sqrt(
                            group_stats[reference_group]['sem']**2 + 
                            group_stats[group_name]['sem']**2
                        )
                        
                        # 95% CI for difference
                        n_ref = group_stats[reference_group]['n']
                        n_comp = group_stats[group_name]['n']
                        df = n_ref + n_comp - 2
                        ci_diff = stats.t.ppf(0.975, df) * sem_diff if df > 0 else 0
                        
                        y_positions.append(i)
                        differences.append(diff)
                        ci_errors.append(ci_diff)
                        labels.append(f'{group_name} - {reference_group}')
                        marker_colors.append(group_stats[group_name]['color'])
                    
                    # Add difference plot
                    if len(differences) > 0:
                        fig.add_trace(
                            go.Scatter(
                                x=differences,
                                y=y_positions,
                                error_x=dict(
                                    type='data',
                                    array=ci_errors,
                                    visible=True,
                                    thickness=3,
                                    width=8
                                ),
                                mode='markers',
                                name='Mean Difference',
                                marker=dict(
                                    size=16,
                                    color=marker_colors,
                                    symbol='diamond',
                                    line=dict(width=2, color='black')
                                ),
                                showlegend=False,
                                hovertemplate='<b>%{text}</b><br>' +
                                             'Difference: %{x:.3f}<br>' +
                                             '95% CI: ±%{error_x.array:.3f}<extra></extra>',
                                text=labels
                            ),
                            row=1, col=2
                        )
                        
                        # Add zero line
                        fig.add_vline(
                            x=0, 
                            line_dash="dash", 
                            line_color="gray",
                            row=1, col=2
                        )
                        
                        # Update y-axis for right panel
                        fig.update_yaxes(
                            ticktext=labels,
                            tickvals=y_positions,
                            row=1, col=2
                        )
            
            # Update layout
            fig.update_xaxes(
                title_text="Group",
                ticktext=group_names,
                tickvals=[i * x_spacing for i in range(len(group_names))],
                row=1, col=1
            )
            
            fig.update_yaxes(
                title_text=value_col.replace('_', ' ').title(),
                row=1, col=1
            )
            
            fig.update_xaxes(
                title_text="Difference from Control",
                zeroline=True,
                row=1, col=2
            )
            
            fig.update_layout(
                title=title,
                height=height,
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.15
                ),
                hovermode='closest'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating estimation plot: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    @staticmethod
    def plot_aligned_curves(aligned_results: dict, height: int = 500):
        """
        Plots multiple, time-aligned, and interpolated FRAP curves.
        
        This visualization shows curves that have been properly aligned to t=0 
        (bleach event) and interpolated onto a common time axis, enabling direct
        comparison of recovery kinetics even when experiments used different
        sampling rates.

        Parameters
        ----------
        aligned_results : dict
            The output from FRAPAnalysisCore.align_and_interpolate_curves()
            Expected keys: 'common_time', 'interpolated_curves'
        height : int
            The height of the plot in pixels. Default: 500

        Returns
        -------
        plotly.graph_objects.Figure
            A Plotly figure showing all aligned recovery curves with legend
        """
        if not aligned_results or not aligned_results.get('interpolated_curves'):
            logger.warning("No data available for aligned plot")
            return go.Figure().update_layout(
                title="No data available for aligned plot",
                annotations=[{
                    'text': 'No valid curves to display',
                    'xref': 'paper',
                    'yref': 'paper',
                    'showarrow': False,
                    'font': {'size': 16}
                }]
            )

        fig = go.Figure()
        common_time = aligned_results['common_time']

        # Plot each aligned curve
        for i, curve in enumerate(aligned_results['interpolated_curves']):
            fig.add_trace(go.Scatter(
                x=common_time,
                y=curve['intensity'],
                mode='lines',
                name=curve['name'],
                line=dict(width=1.5),
                opacity=0.8,
                hovertemplate='<b>%{fullData.name}</b><br>' +
                             'Time: %{x:.2f} s<br>' +
                             'Intensity: %{y:.3f}<br>' +
                             '<extra></extra>'
            ))

        # Add a horizontal line at y=1 (pre-bleach normalized level)
        fig.add_hline(
            y=1, 
            line_width=1, 
            line_dash="dash", 
            line_color="gray",
            annotation_text="Pre-bleach level",
            annotation_position="right"
        )

        fig.update_layout(
            title="Time-Aligned and Interpolated FRAP Recovery Curves",
            xaxis_title="Time Since Bleach (s)",
            yaxis_title="Normalized Intensity",
            height=height,
            xaxis=dict(
                range=[0, None],  # Ensure x-axis starts at 0
                showgrid=True,
                gridcolor='lightgray'
            ),
            yaxis=dict(
                range=[0, None],  # Ensure y-axis starts at 0
                showgrid=True,
                gridcolor='lightgray'
            ),
            legend=dict(
                title="File",
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            ),
            hovermode='closest',
            template='plotly_white'
        )
        
        return fig

    @staticmethod
    def plot_advanced_group_comparison(comparison_results, height=600):
        """
        Plot advanced curve fitting comparison between groups.
        
        Parameters:
        -----------
        comparison_results : dict
            Results from compare_recovery_profiles with advanced fitting
        height : int
            Height of the plot
            
        Returns:
        --------
        plotly.graph_objects.Figure
            Plotly figure with data and fitted curves
        """
        try:
            if 'advanced_fitting' not in comparison_results or not comparison_results['advanced_fitting']['success']:
                logger.warning("No advanced fitting results available")
                return None
            
            adv = comparison_results['advanced_fitting']
            group1_name = adv['group1_name']
            group2_name = adv['group2_name']
            
            # Get profile data
            g1_prof = comparison_results['group1_profile']
            g2_prof = comparison_results['group2_profile']
            
            # Get fitted curves
            fit1 = adv['group1_fit']
            fit2 = adv['group2_fit']
            
            fig = go.Figure()
            
            # Plot Group 1 data and fit
            # Data with error bars
            fig.add_trace(go.Scatter(
                x=g1_prof['time'],
                y=g1_prof['intensity_mean'],
                error_y=dict(
                    type='data',
                    array=g1_prof['intensity_sem'],
                    visible=True,
                    color='rgba(31, 119, 180, 0.3)'
                ),
                mode='markers',
                name=f'{group1_name} (data)',
                marker=dict(size=6, color='#1f77b4'),
                showlegend=True
            ))
            
            # Fitted curve
            if 'fitted_values' in fit1:
                fig.add_trace(go.Scatter(
                    x=g1_prof['time'],
                    y=fit1['fitted_values'],
                    mode='lines',
                    name=f'{group1_name} (fit)',
                    line=dict(color='#1f77b4', width=3),
                    showlegend=True
                ))
            
            # Plot Group 2 data and fit
            # Data with error bars
            fig.add_trace(go.Scatter(
                x=g2_prof['time'],
                y=g2_prof['intensity_mean'],
                error_y=dict(
                    type='data',
                    array=g2_prof['intensity_sem'],
                    visible=True,
                    color='rgba(255, 127, 14, 0.3)'
                ),
                mode='markers',
                name=f'{group2_name} (data)',
                marker=dict(size=6, color='#ff7f0e'),
                showlegend=True
            ))
            
            # Fitted curve
            if 'fitted_values' in fit2:
                fig.add_trace(go.Scatter(
                    x=g2_prof['time'],
                    y=fit2['fitted_values'],
                    mode='lines',
                    name=f'{group2_name} (fit)',
                    line=dict(color='#ff7f0e', width=3),
                    showlegend=True
                ))
            
            # Add vertical line at t=0
            fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="gray")
            
            # Create title with model info
            model_name = adv['model_used'].replace('_', ' ').title()
            r2_text = f"R²: {group1_name}={fit1.get('r2', 0):.3f}, {group2_name}={fit2.get('r2', 0):.3f}"
            title_text = f"Advanced Fitting Comparison: {model_name}<br><sub>{r2_text}</sub>"
            
            fig.update_layout(
                title=title_text,
                xaxis_title="Time (s)",
                yaxis_title="Normalized Intensity",
                height=height,
                template='plotly_white',
                hovermode='closest',
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="right",
                    x=1
                )
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting advanced group comparison: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    @staticmethod
    def plot_parameter_comparison(comparison_results, height=500):
        """
        Create bar chart comparing fitted parameters between groups.
        
        Parameters:
        -----------
        comparison_results : dict
            Results from compare_groups_advanced_fitting
        height : int
            Height of the plot
            
        Returns:
        --------
        plotly.graph_objects.Figure
            Plotly figure with parameter comparison
        """
        try:
            if 'advanced_fitting' not in comparison_results or not comparison_results['advanced_fitting']['success']:
                return None
            
            adv = comparison_results['advanced_fitting']
            param_comp = adv.get('metric_comparison', {})
            
            if not param_comp:
                return None
            
            group1_name = adv['group1_name']
            group2_name = adv['group2_name']
            
            # Prepare data for plotting
            metrics = []
            group1_vals = []
            group2_vals = []
            
            for metric, data in param_comp.items():
                if isinstance(data.get(group1_name), (int, float)):
                    metrics.append(metric.replace('_', ' ').title())
                    group1_vals.append(data[group1_name])
                    group2_vals.append(data[group2_name])
            
            if not metrics:
                return None
            
            fig = go.Figure()
            
            # Add bars for each group
            fig.add_trace(go.Bar(
                name=group1_name,
                x=metrics,
                y=group1_vals,
                marker_color='#1f77b4',
                text=[f'{v:.3f}' for v in group1_vals],
                textposition='outside'
            ))
            
            fig.add_trace(go.Bar(
                name=group2_name,
                x=metrics,
                y=group2_vals,
                marker_color='#ff7f0e',
                text=[f'{v:.3f}' for v in group2_vals],
                textposition='outside'
            ))
            
            fig.update_layout(
                title="Fitted Parameter Comparison",
                xaxis_title="Parameter",
                yaxis_title="Value",
                height=height,
                barmode='group',
                template='plotly_white',
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting parameter comparison: {e}")
            return None
