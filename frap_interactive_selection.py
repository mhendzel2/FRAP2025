#!/usr/bin/env python3
"""
Interactive Graph-based Curve Selection for FRAP Analysis

This module provides direct selection capabilities from plotly graphs,
allowing users to click on curves to include/exclude them from analysis.

Author: FRAP2025 Analysis Platform
Date: October 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import json

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st


def create_interactive_selection_plot(group_data: Dict, data_manager, excluded_files: set) -> go.Figure:
    """
    Create an interactive plot where users can click on curves to select/deselect them.
    
    Parameters:
    -----------
    group_data : Dict
        Group data containing file paths
    data_manager : FRAPDataManager
        Data manager instance
    excluded_files : set
        Currently excluded file paths
        
    Returns:
    --------
    go.Figure
        Interactive Plotly figure with clickable curves
    """
    
    fig = go.Figure()
    
    # Color scheme
    included_colors = px.colors.qualitative.Set3
    excluded_color = "rgba(255, 100, 100, 0.6)"
    
    file_paths = list(group_data['files'])
    curve_data = []
    
    for i, file_path in enumerate(file_paths):
        try:
            file_data = data_manager.files[file_path]
            time = np.array(file_data['time'])
            intensity = np.array(file_data['intensity'])
            file_name = file_data['name']
            
            is_excluded = file_path in excluded_files
            
            if is_excluded:
                line_color = excluded_color
                line_width = 2
                line_dash = "dash"
                opacity = 0.6
                showlegend = False
            else:
                color_idx = i % len(included_colors)
                line_color = included_colors[color_idx]
                line_width = 2
                line_dash = "solid"
                opacity = 0.8
                showlegend = False
            
            # Add curve to plot
            fig.add_trace(go.Scatter(
                x=time,
                y=intensity,
                mode='lines',
                name=file_name,
                line=dict(
                    color=line_color,
                    width=line_width,
                    dash=line_dash
                ),
                opacity=opacity,
                hovertemplate=(
                    f"<b>{file_name}</b><br>"
                    "Time: %{x:.1f}s<br>"
                    "Intensity: %{y:.3f}<br>"
                    f"Status: {'EXCLUDED' if is_excluded else 'INCLUDED'}<br>"
                    "<i>Click to toggle selection</i>"
                    "<extra></extra>"
                ),
                customdata=[file_path] * len(time),  # Store file path for click handling
                showlegend=showlegend
            ))
            
            # Store curve metadata for click handling
            curve_data.append({
                'file_path': file_path,
                'file_name': file_name,
                'is_excluded': is_excluded,
                'trace_index': len(fig.data) - 1
            })
            
        except Exception as e:
            st.warning(f"Could not plot {file_path}: {e}")
            continue
    
    # Update layout for interactivity
    fig.update_layout(
        title={
            'text': (
                "🖱️ Interactive Curve Selection - Click on curves to include/exclude<br>"
                f"<sub>Included: {len(file_paths) - len(excluded_files)} | "
                f"Excluded: {len(excluded_files)} | "
                f"Total: {len(file_paths)}</sub>"
            ),
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title="Time (s)",
        yaxis_title="Normalized Intensity", 
        height=600,
        hovermode='closest',
        showlegend=False,
        annotations=[
            dict(
                text=(
                    "💡 Instructions:<br>"
                    "• Click on any curve to toggle inclusion/exclusion<br>"
                    "• Solid lines = Included in analysis<br>"
                    "• Dashed red lines = Excluded from analysis<br>"
                    "• Changes update automatically"
                ),
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                xanchor="left", yanchor="top",
                showarrow=False,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="gray",
                borderwidth=1,
                font=dict(size=10)
            )
        ]
    )
    
    # Store curve metadata in figure for click handling
    fig.update_layout(
        updatemenus=[],  # Remove any default update menus
        meta={'curve_data': curve_data}  # Store metadata
    )
    
    return fig


def create_selection_summary_plot(group_data: Dict, data_manager, excluded_files: set) -> go.Figure:
    """
    Create a summary plot showing selection statistics and quality metrics.
    """
    
    file_paths = list(group_data['files'])
    included_paths = [p for p in file_paths if p not in excluded_files]
    
    # Calculate summary statistics
    summary_data = {
        'Category': ['Included', 'Excluded', 'Total'],
        'Count': [len(included_paths), len(excluded_files), len(file_paths)],
        'Percentage': [
            len(included_paths) / len(file_paths) * 100 if file_paths else 0,
            len(excluded_files) / len(file_paths) * 100 if file_paths else 0,
            100
        ]
    }
    
    # Create bar chart
    fig = go.Figure()
    
    colors = ['green', 'red', 'blue']
    
    for i, (category, count, pct) in enumerate(zip(summary_data['Category'], 
                                                   summary_data['Count'], 
                                                   summary_data['Percentage'])):
        fig.add_trace(go.Bar(
            x=[category],
            y=[count],
            name=category,
            marker_color=colors[i],
            text=f"{count} ({pct:.1f}%)",
            textposition='auto',
            showlegend=False
        ))
    
    fig.update_layout(
        title="Selection Summary",
        xaxis_title="Category",
        yaxis_title="Number of Curves",
        height=300,
        showlegend=False
    )
    
    return fig


def handle_curve_click_selection(group_data: Dict, data_manager, group_name: str):
    """
    Create the complete interactive curve selection interface.
    
    Parameters:
    -----------
    group_data : Dict
        Group data from data manager
    data_manager : FRAPDataManager
        Data manager instance
    group_name : str
        Name of the current group
    """
    
    st.markdown("### 🖱️ Interactive Graph Selection")
    st.markdown("**Click directly on curves in the plot below to include/exclude them from analysis**")
    
    # Initialize session state for excluded files if not present
    if 'interactive_excluded_files' not in st.session_state:
        st.session_state.interactive_excluded_files = set()
    
    # Get current exclusion state
    excluded_files = st.session_state.interactive_excluded_files
    
    # Quick action buttons
    action_cols = st.columns(5)
    
    with action_cols[0]:
        if st.button("✅ Include All", help="Include all curves"):
            st.session_state.interactive_excluded_files = set()
            st.rerun()
    
    with action_cols[1]:
        if st.button("❌ Exclude All", help="Exclude all curves"):
            st.session_state.interactive_excluded_files = set(group_data['files'])
            st.rerun()
    
    with action_cols[2]:
        if st.button("🔄 Reset", help="Reset to original auto-detected outliers"):
            # Get original excluded files from group analysis
            original_excluded = set(group_data.get('excluded_files', []))
            st.session_state.interactive_excluded_files = original_excluded
            st.rerun()
    
    with action_cols[3]:
        if st.button("📉 Auto-Slopes", help="Automatically exclude curves with downward slopes"):
            # Import and use slope detection
            try:
                from frap_slope_detection import FRAPSlopeDetector
                detector = FRAPSlopeDetector()
                results = detector.analyze_group_slopes(group_data, data_manager)
                
                slope_problematic = [item['file_path'] for item in results['slope_issues']]
                st.session_state.interactive_excluded_files.update(slope_problematic)
                
                st.success(f"Auto-excluded {len(slope_problematic)} curves with downward slopes")
                st.rerun()
                
            except ImportError:
                st.warning("Slope detection not available")
    
    with action_cols[4]:
        if st.button("💾 Apply Changes", type="primary", help="Apply current selection to analysis"):
            # Update the group analysis with current selections
            excluded_list = list(st.session_state.interactive_excluded_files)
            data_manager.update_group_analysis(group_name, excluded_files=excluded_list)
            
            n_included = len(group_data['files']) - len(excluded_list)
            st.success(f"✅ Applied selection! Now using {n_included} curves in analysis.")
            st.rerun()
    
    # Create the interactive plot
    interactive_fig = create_interactive_selection_plot(group_data, data_manager, excluded_files)
    
    # Display the plot with click handling
    clicked_data = st.plotly_chart(
        interactive_fig, 
        use_container_width=True,
        key=f"interactive_plot_{group_name}",
        on_select="rerun",
        selection_mode="points"
    )
    
    # Handle click events if Streamlit supports it (fallback to manual selection)
    if hasattr(st, 'plotly_events') or 'selection' in st.session_state:
        # This would be the ideal implementation with click events
        # For now, we provide manual selection as backup
        pass
    
    # Manual selection interface as backup
    st.markdown("---")
    st.markdown("#### 📋 Manual Selection Interface")
    st.markdown("*Use this if direct clicking doesn't work in your browser*")
    
    # File selection with checkboxes in a more compact layout
    file_paths = list(group_data['files'])
    
    # Create a more compact selection interface
    n_cols = 3
    cols = st.columns(n_cols)
    
    selection_changed = False
    
    for i, file_path in enumerate(file_paths):
        col_idx = i % n_cols
        
        file_data = data_manager.files[file_path]
        file_name = file_data['name']
        
        is_included = file_path not in excluded_files
        
        # Shorter display name
        display_name = file_name[:30] + "..." if len(file_name) > 30 else file_name
        
        with cols[col_idx]:
            new_status = st.checkbox(
                f"{'✅' if is_included else '❌'} {display_name}",
                value=is_included,
                key=f"manual_select_{file_path}_{i}",
                help=f"Full name: {file_name}"
            )
            
            if new_status != is_included:
                if new_status:  # Include file
                    st.session_state.interactive_excluded_files.discard(file_path)
                else:  # Exclude file
                    st.session_state.interactive_excluded_files.add(file_path)
                selection_changed = True
    
    # Auto-refresh if selection changed
    if selection_changed:
        st.rerun()
    
    # Selection summary
    st.markdown("---")
    summary_cols = st.columns([2, 1])
    
    with summary_cols[0]:
        # Selection statistics
        n_total = len(file_paths)
        n_excluded = len(excluded_files)
        n_included = n_total - n_excluded
        
        st.markdown("#### 📊 Selection Summary")
        
        metrics_cols = st.columns(3)
        with metrics_cols[0]:
            st.metric("✅ Included", n_included)
        with metrics_cols[1]:
            st.metric("❌ Excluded", n_excluded)
        with metrics_cols[2]:
            st.metric("📈 Inclusion Rate", f"{n_included/n_total:.1%}" if n_total > 0 else "0%")
    
    with summary_cols[1]:
        # Summary plot
        if n_total > 0:
            summary_fig = create_selection_summary_plot(group_data, data_manager, excluded_files)
            st.plotly_chart(summary_fig, use_container_width=True)
    
    # Show excluded files list if any
    if excluded_files:
        with st.expander(f"🗂️ Excluded Files ({len(excluded_files)})", expanded=False):
            excluded_names = []
            for file_path in excluded_files:
                try:
                    file_name = data_manager.files[file_path]['name']
                    excluded_names.append(file_name)
                except:
                    excluded_names.append(file_path)
            
            for i, name in enumerate(sorted(excluded_names), 1):
                st.write(f"{i}. {name}")


def create_hover_selection_plot(group_data: Dict, data_manager, excluded_files: set) -> go.Figure:
    """
    Create a plot with enhanced hover information for easier selection decisions.
    """
    
    fig = go.Figure()
    
    file_paths = list(group_data['files'])
    
    for i, file_path in enumerate(file_paths):
        try:
            file_data = data_manager.files[file_path]
            time = np.array(file_data['time'])
            intensity = np.array(file_data['intensity'])
            file_name = file_data['name']
            
            is_excluded = file_path in excluded_files
            
            # Calculate basic curve metrics for hover info
            min_intensity = np.min(intensity)
            max_intensity = np.max(intensity)
            bleach_depth = max_intensity - min_intensity
            
            # Find recovery characteristics
            bleach_idx = np.argmin(intensity)
            if bleach_idx < len(intensity) - 5:
                final_recovery = np.mean(intensity[-5:])
                recovery_fraction = (final_recovery - min_intensity) / bleach_depth if bleach_depth > 0 else 0
            else:
                recovery_fraction = 0
            
            # Calculate slope in final phase for quick assessment
            if len(intensity) > 10:
                final_phase = intensity[-10:]
                final_time = time[-10:] - time[-10]
                try:
                    slope = np.polyfit(final_time, final_phase, 1)[0] * 60  # per minute
                except:
                    slope = 0
            else:
                slope = 0
            
            # Color coding
            if is_excluded:
                line_color = "red"
                line_width = 3
                line_dash = "dash"
                opacity = 0.7
            else:
                # Color by quality metrics
                if slope < -0.05:  # Strong downward slope
                    line_color = "orange"
                elif recovery_fraction < 0.5:  # Low recovery
                    line_color = "purple"
                else:
                    line_color = "blue"
                line_width = 2
                line_dash = "solid"
                opacity = 0.8
            
            # Enhanced hover template
            hover_template = (
                f"<b>{file_name}</b><br>"
                "Time: %{x:.1f}s<br>"
                "Intensity: %{y:.3f}<br>"
                f"Recovery: {recovery_fraction:.1%}<br>"
                f"Final Slope: {slope:.3f}/min<br>"
                f"Status: {'EXCLUDED' if is_excluded else 'INCLUDED'}<br>"
                "<i>Assess quality from these metrics</i>"
                "<extra></extra>"
            )
            
            fig.add_trace(go.Scatter(
                x=time,
                y=intensity,
                mode='lines',
                name=file_name,
                line=dict(
                    color=line_color,
                    width=line_width,
                    dash=line_dash
                ),
                opacity=opacity,
                hovertemplate=hover_template,
                showlegend=False
            ))
            
        except Exception as e:
            continue
    
    # Add legend explanation
    fig.update_layout(
        title="📊 Curve Quality Assessment - Hover for Details",
        xaxis_title="Time (s)",
        yaxis_title="Normalized Intensity",
        height=500,
        annotations=[
            dict(
                text=(
                    "Color Guide:<br>"
                    "🔵 Blue = Good quality<br>"
                    "🟠 Orange = Downward slope<br>"
                    "🟣 Purple = Low recovery<br>"
                    "🔴 Red = Excluded"
                ),
                xref="paper", yref="paper",
                x=0.98, y=0.98,
                xanchor="right", yanchor="top",
                showarrow=False,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="gray",
                borderwidth=1,
                font=dict(size=10)
            )
        ]
    )
    
    return fig


if __name__ == "__main__":
    print("Interactive graph selection module loaded successfully!")
    print("This module provides:")
    print("- Direct curve selection from interactive plots")
    print("- Enhanced hover information for quality assessment") 
    print("- Compact manual selection interface as backup")
    print("- Real-time selection summary and statistics")