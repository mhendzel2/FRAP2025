"""
FRAP Single-Cell Analysis - Modern Streamlit UI
Advanced interactive interface with linked views, cohort management, and live gating
"""

import streamlit as st

from streamlit_compat import patch_streamlit_width

patch_streamlit_width(st)
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json
import hashlib
from datetime import datetime
from typing import Optional, List, Dict, Any
import logging

# Import analysis modules
from frap_singlecell_api import track_movie, fit_cells, analyze
from frap_populations import detect_outliers_and_clusters
from frap_data_model import DataIO
from frap_visualizations import (
    plot_spaghetti, plot_heatmap, plot_pairplot, plot_qc_dashboard
)
from frap_data_loader import (
    render_data_loader, load_from_directory, export_current_cohort,
    export_traces, check_data_quality
)
from frap_singlecell_reports import build_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        # Data
        'roi_traces': pd.DataFrame(),
        'cell_features': pd.DataFrame(),
        'cohorts': {},
        'active_cohort': 'default',
        
        # Selection and filtering
        'selected_cells': [],
        'brushed_cells': [],
        'active_filters': {},
        'outlier_threshold': 0.07,
        'selected_clusters': [],
        
        # UI state
        'current_tab': 'Single-cell',
        'current_cell_id': 0,
        'show_outliers': True,
        'show_noise': True,
        'dark_mode': False,
        
        # Analysis recipe
        'recipe': {
            'timestamp': datetime.now().isoformat(),
            'filters': {},
            'parameters': {},
            'software_versions': get_software_versions()
        },
        
        # Bookmarks
        'bookmarked_cells': [],
        'figure_presets': {},
        
        # Export queue
        'report_figures': []
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_software_versions() -> dict:
    """Get versions of all analysis packages"""
    import sys
    versions = {
        'python': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    }
    
    for module_name in ['numpy', 'scipy', 'pandas', 'sklearn', 'statsmodels']:
        try:
            mod = __import__(module_name)
            versions[module_name] = getattr(mod, '__version__', 'unknown')
        except:
            versions[module_name] = 'not installed'
    
    return versions


def compute_recipe_hash() -> str:
    """Compute hash of current analysis recipe"""
    recipe_str = json.dumps(st.session_state.recipe, sort_keys=True)
    return hashlib.md5(recipe_str.encode()).hexdigest()[:8]


# ============================================================================
# COHORT MANAGEMENT
# ============================================================================

def build_cohort_query():
    """Build cohort based on active filters"""
    df = st.session_state.cell_features.copy()
    
    if df.empty:
        return df
    
    filters = st.session_state.active_filters
    
    # Apply filters
    if 'condition' in filters and filters['condition']:
        df = df[df['condition'].isin(filters['condition'])]
    
    if 'exp_id' in filters and filters['exp_id']:
        df = df[df['exp_id'].isin(filters['exp_id'])]
    
    if 'clusters' in filters and filters['clusters']:
        df = df[df['cluster'].isin(filters['clusters'])]
    
    if 'outliers' in filters:
        if not filters['outliers']:
            df = df[~df['outlier']]
    
    if 'qc_pass' in filters and filters['qc_pass']:
        df = df[df['bleach_qc'] == True]
    
    if 'date_range' in filters and filters['date_range']:
        # Assume date in exp_id or separate column
        pass
    
    return df


def save_cohort(name: str):
    """Save current cohort as a preset"""
    st.session_state.cohorts[name] = {
        'filters': st.session_state.active_filters.copy(),
        'timestamp': datetime.now().isoformat(),
        'n_cells': len(build_cohort_query())
    }
    st.success(f"✓ Cohort '{name}' saved")


def load_cohort(name: str):
    """Load a saved cohort preset"""
    if name in st.session_state.cohorts:
        st.session_state.active_filters = st.session_state.cohorts[name]['filters'].copy()
        st.session_state.active_cohort = name
        st.rerun()


# ============================================================================
# LEFT RAIL - NAVIGATION
# ============================================================================

def render_left_rail():
    """Render left navigation rail"""
    with st.sidebar:
        st.title("🔬 FRAP Analysis")
        
        # Navigation sections
        st.markdown("### 📊 Workflow")
        
        sections = ["Data", "Cohort", "QC", "Stats", "Export"]
        
        for section in sections:
            if st.button(f"{'▶' if section else '▷'} {section}", 
                        key=f"nav_{section}",
                        width="stretch"):
                st.session_state.current_section = section
        
        st.divider()
        
        # Quick actions
        st.markdown("### ⚡ Quick Actions")
        
        if st.button("📂 Load Data", width="stretch"):
            st.session_state.show_data_loader = True
        
        # Show data loader modal
        if st.session_state.get('show_data_loader', False):
            with st.expander("📂 Data Loader", expanded=True):
                render_data_loader()
                if st.button("✕ Close", key="close_loader"):
                    st.session_state.show_data_loader = False
                    st.rerun()
        
        if st.button("💾 Save Cohort", width="stretch"):
            st.session_state.show_save_cohort = True
        
        if st.button("📋 Copy Recipe", width="stretch"):
            recipe_json = json.dumps(st.session_state.recipe, indent=2)
            st.code(recipe_json, language='json')
            st.success("Recipe copied to clipboard!")
        
        st.divider()
        
        # Theme toggle
        dark_mode = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
        if dark_mode != st.session_state.dark_mode:
            st.session_state.dark_mode = dark_mode
            st.rerun()
        
        # Saved cohorts
        if st.session_state.cohorts:
            st.markdown("### 📁 Saved Cohorts")
            for name, info in st.session_state.cohorts.items():
                col1, col2 = st.columns([3, 1])
                with col1:
                    if st.button(f"{name} ({info['n_cells']})", 
                               key=f"load_{name}",
                               width="stretch"):
                        load_cohort(name)
                with col2:
                    if st.button("🗑", key=f"del_{name}"):
                        del st.session_state.cohorts[name]
                        st.rerun()


# ============================================================================
# SUMMARY HEADER
# ============================================================================

def render_summary_header():
    """Render persistent summary header with counts"""
    df = build_cohort_query()
    
    total_cells = len(st.session_state.cell_features)
    included_cells = len(df)
    excluded_cells = total_cells - included_cells
    
    n_experiments = df['exp_id'].nunique() if not df.empty else 0
    n_clusters = len(df['cluster'].unique()) - (1 if -1 in df['cluster'].values else 0) if not df.empty else 0
    n_outliers = df['outlier'].sum() if not df.empty else 0
    
    # Create columns for metrics
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("Cells", f"{included_cells:,}", 
                 delta=f"-{excluded_cells}" if excluded_cells > 0 else None,
                 delta_color="off")
    
    with col2:
        st.metric("Experiments", f"{n_experiments}")
    
    with col3:
        st.metric("Clusters", f"{n_clusters}")
    
    with col4:
        st.metric("Outliers", f"{n_outliers}",
                 delta=f"{n_outliers/included_cells*100:.1f}%" if included_cells > 0 else None)
    
    with col5:
        conditions = df['condition'].unique() if 'condition' in df.columns else []
        st.metric("Conditions", f"{len(conditions)}")
    
    with col6:
        recipe_hash = compute_recipe_hash()
        st.metric("Recipe", f"#{recipe_hash}")
    
    st.divider()


# ============================================================================
# COHORT BUILDER
# ============================================================================

def render_cohort_builder():
    """Render compact cohort query builder"""
    st.markdown("### 🔍 Cohort Builder")
    
    df = st.session_state.cell_features
    
    if df.empty:
        st.warning("No data loaded. Please load data first.")
        return
    
    # Query bar
    col1, col2, col3 = st.columns(3)
    
    with col1:
        conditions = ['All'] + sorted(df['condition'].unique().tolist()) if 'condition' in df.columns else ['All']
        selected_conditions = st.multiselect(
            "Conditions",
            conditions,
            default=['All'],
            key="filter_conditions"
        )
        if 'All' not in selected_conditions and selected_conditions:
            st.session_state.active_filters['condition'] = selected_conditions
        elif 'condition' in st.session_state.active_filters:
            del st.session_state.active_filters['condition']
    
    with col2:
        experiments = ['All'] + sorted(df['exp_id'].unique().tolist())
        selected_exps = st.multiselect(
            "Experiments",
            experiments,
            default=['All'],
            key="filter_experiments"
        )
        if 'All' not in selected_exps and selected_exps:
            st.session_state.active_filters['exp_id'] = selected_exps
        elif 'exp_id' in st.session_state.active_filters:
            del st.session_state.active_filters['exp_id']
    
    with col3:
        clusters = sorted(df['cluster'].unique().tolist())
        clusters = [c for c in clusters if c != -1]
        selected_clusters = st.multiselect(
            "Clusters",
            clusters,
            default=clusters,
            key="filter_clusters"
        )
        if selected_clusters != clusters:
            st.session_state.active_filters['clusters'] = selected_clusters
        elif 'clusters' in st.session_state.active_filters:
            del st.session_state.active_filters['clusters']
    
    # Additional filters
    col4, col5, col6 = st.columns(3)
    
    with col4:
        show_outliers = st.checkbox("Include Outliers", value=True, key="show_outliers_cb")
        st.session_state.active_filters['outliers'] = show_outliers
    
    with col5:
        qc_only = st.checkbox("QC Passed Only", value=False, key="qc_only_cb")
        st.session_state.active_filters['qc_pass'] = qc_only
    
    with col6:
        show_noise = st.checkbox("Include Noise (-1)", value=True, key="show_noise_cb")
        st.session_state.show_noise = show_noise
    
    # Active filters as chips
    if st.session_state.active_filters:
        st.markdown("**Active Filters:**")
        
        filter_chips = []
        for key, value in st.session_state.active_filters.items():
            if key in ['condition', 'exp_id', 'clusters'] and value:
                filter_chips.append(f"{key}: {', '.join(map(str, value[:2]))}" + 
                                  (f" +{len(value)-2}" if len(value) > 2 else ""))
            elif key == 'outliers' and not value:
                filter_chips.append("No outliers")
            elif key == 'qc_pass' and value:
                filter_chips.append("QC passed")
        
        cols = st.columns(len(filter_chips) + 1)
        for i, chip in enumerate(filter_chips):
            with cols[i]:
                st.markdown(f"`{chip}`")
        
        with cols[-1]:
            if st.button("Clear All", key="clear_filters"):
                st.session_state.active_filters = {}
                st.rerun()
    
    # Save cohort
    col_save1, col_save2 = st.columns([3, 1])
    with col_save1:
        cohort_name = st.text_input("Cohort name", key="cohort_name_input", 
                                    placeholder="Enter name to save...")
    with col_save2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("💾 Save", key="save_cohort_btn", disabled=not cohort_name):
            save_cohort(cohort_name)


# ============================================================================
# SINGLE-CELL INSPECTOR
# ============================================================================

def render_single_cell_inspector():
    """Render detailed single-cell inspector"""
    st.markdown("## 🔬 Single-Cell Inspector")
    
    df_cohort = build_cohort_query()
    
    if df_cohort.empty:
        st.warning("No cells in current cohort")
        return
    
    cell_ids = sorted(df_cohort['cell_id'].unique())
    
    # Cell selector with navigation
    col1, col2, col3, col4, col5 = st.columns([1, 3, 1, 1, 1])
    
    with col1:
        if st.button("⬅ Prev"):
            current_idx = cell_ids.index(st.session_state.current_cell_id) if st.session_state.current_cell_id in cell_ids else 0
            st.session_state.current_cell_id = cell_ids[(current_idx - 1) % len(cell_ids)]
            st.rerun()
    
    with col2:
        selected_cell = st.selectbox(
            "Cell",
            cell_ids,
            index=cell_ids.index(st.session_state.current_cell_id) if st.session_state.current_cell_id in cell_ids else 0,
            key="cell_selector"
        )
        st.session_state.current_cell_id = selected_cell
    
    with col3:
        if st.button("Next ➡"):
            current_idx = cell_ids.index(st.session_state.current_cell_id) if st.session_state.current_cell_id in cell_ids else 0
            st.session_state.current_cell_id = cell_ids[(current_idx + 1) % len(cell_ids)]
            st.rerun()
    
    with col4:
        is_bookmarked = selected_cell in st.session_state.bookmarked_cells
        if st.button("🔖" if not is_bookmarked else "📌", key="bookmark_cell"):
            if is_bookmarked:
                st.session_state.bookmarked_cells.remove(selected_cell)
            else:
                st.session_state.bookmarked_cells.append(selected_cell)
            st.rerun()
    
    with col5:
        if st.button("📊 Compare", key="add_to_tray"):
            if selected_cell not in st.session_state.selected_cells:
                st.session_state.selected_cells.append(selected_cell)
    
    # Get cell data
    cell_features = df_cohort[df_cohort['cell_id'] == selected_cell].iloc[0]
    cell_traces = st.session_state.roi_traces[
        st.session_state.roi_traces['cell_id'] == selected_cell
    ].sort_values('t')
    
    # Two-pane layout
    left_pane, right_pane = st.columns([1, 1])
    
    with left_pane:
        st.markdown("#### ROI Trajectory")
        
        # Trajectory plot
        fig_traj = go.Figure()
        
        fig_traj.add_trace(go.Scatter(
            x=cell_traces['x'],
            y=cell_traces['y'],
            mode='lines+markers',
            line=dict(color='blue', width=2),
            marker=dict(size=4, color=cell_traces['t'], colorscale='Viridis'),
            text=cell_traces['frame'],
            hovertemplate='Frame %{text}<br>x: %{x:.1f}<br>y: %{y:.1f}<extra></extra>'
        ))
        
        fig_traj.update_layout(
            title="ROI Position Over Time",
            xaxis_title="X (pixels)",
            yaxis_title="Y (pixels)",
            height=300,
            yaxis=dict(scaleanchor="x", scaleratio=1)
        )
        
        st.plotly_chart(fig_traj, width="stretch")
        
        # QC badges
        st.markdown("#### QC Metrics")
        
        qc_col1, qc_col2, qc_col3 = st.columns(3)
        
        with qc_col1:
            drift = cell_features.get('drift_px', 0)
            st.metric("Drift", f"{drift:.2f} px", 
                     delta="⚠" if drift > 10 else "✓",
                     delta_color="inverse")
        
        with qc_col2:
            r2 = cell_features.get('r2', 0)
            st.metric("R²", f"{r2:.3f}",
                     delta="✓" if r2 > 0.8 else "⚠",
                     delta_color="normal" if r2 > 0.8 else "inverse")
        
        with qc_col3:
            motion_frames = cell_traces['qc_motion'].sum() if not cell_traces.empty else 0
            st.metric("Motion Flags", motion_frames,
                     delta="✓" if motion_frames == 0 else "⚠",
                     delta_color="inverse")
    
    with right_pane:
        st.markdown("#### Recovery Curve")
        
        # Recovery curve with fit
        fig_recovery = go.Figure()
        
        # Raw data
        fig_recovery.add_trace(go.Scatter(
            x=cell_traces['t'],
            y=cell_traces['signal_corr'],
            mode='markers',
            name='Measured',
            marker=dict(size=6, color='lightblue')
        ))
        
        # Fit line
        if not np.isnan(cell_features.get('k', np.nan)):
            t_fit = np.linspace(cell_traces['t'].min(), cell_traces['t'].max(), 100)
            A = cell_features.get('A', 1)
            B = cell_features.get('B', 0.5)
            k = cell_features.get('k', 0.5)
            
            # Find bleach point
            bleach_idx = cell_traces['signal_corr'].idxmin()
            t_bleach = cell_traces.loc[bleach_idx, 't']
            t_post = t_fit[t_fit >= t_bleach] - t_bleach
            
            y_fit = A - B * np.exp(-k * t_post)
            
            fig_recovery.add_trace(go.Scatter(
                x=t_fit[t_fit >= t_bleach],
                y=y_fit,
                mode='lines',
                name='Fit',
                line=dict(color='red', width=2)
            ))
        
        fig_recovery.update_layout(
            title="Intensity Recovery",
            xaxis_title="Time (s)",
            yaxis_title="Signal (corrected)",
            height=300
        )
        
        st.plotly_chart(fig_recovery, width="stretch")
        
        # Residuals
        st.markdown("#### Residuals")
        
        if not np.isnan(cell_features.get('k', np.nan)):
            residuals = cell_traces['signal_corr'] - (A - B * np.exp(-k * (cell_traces['t'] - t_bleach)))
            
            fig_resid = go.Figure()
            fig_resid.add_trace(go.Scatter(
                x=cell_traces['t'],
                y=residuals,
                mode='markers',
                marker=dict(size=4, color='gray')
            ))
            fig_resid.add_hline(y=0, line_dash="dash", line_color="black")
            fig_resid.update_layout(
                xaxis_title="Time (s)",
                yaxis_title="Residual",
                height=150,
                margin=dict(t=10, b=30)
            )
            
            st.plotly_chart(fig_resid, width="stretch")
    
    # Parameters table
    st.markdown("#### Fitted Parameters")
    
    param_col1, param_col2, param_col3, param_col4 = st.columns(4)
    
    with param_col1:
        st.metric("Mobile Fraction", f"{cell_features.get('mobile_frac', 0):.3f}")
        st.metric("k (s⁻¹)", f"{cell_features.get('k', 0):.3f}")
    
    with param_col2:
        st.metric("t½ (s)", f"{cell_features.get('t_half', 0):.2f}")
        st.metric("I₀", f"{cell_features.get('I0', 0):.3f}")
    
    with param_col3:
        st.metric("I∞", f"{cell_features.get('I_inf', 0):.3f}")
        st.metric("Pre-bleach", f"{cell_features.get('pre_bleach', 0):.3f}")
    
    with param_col4:
        st.metric("Model", cell_features.get('fit_method', 'unknown'))
        st.metric("AIC", f"{cell_features.get('aic', 0):.1f}")


# ============================================================================
# GROUP WORKSPACE
# ============================================================================

def render_group_workspace():
    """Render group-level analysis workspace"""
    st.markdown("## 📊 Group Analysis")
    
    df_cohort = build_cohort_query()
    
    if df_cohort.empty:
        st.warning("No cells in current cohort")
        return
    
    # Check if we have condition column
    if 'condition' not in df_cohort.columns:
        st.error("No 'condition' column found in data")
        return
    
    conditions = sorted(df_cohort['condition'].unique())
    
    # Condition selector
    selected_condition = st.selectbox("Select condition", conditions, key="group_condition")
    
    # Normalization toggle
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### Recovery Curves")
    with col2:
        normalize = st.checkbox("Normalize per cell", value=False, key="normalize_curves")
    
    # Spaghetti plot
    condition_data = df_cohort[df_cohort['condition'] == selected_condition]
    condition_traces = st.session_state.roi_traces[
        st.session_state.roi_traces['cell_id'].isin(condition_data['cell_id'])
    ]
    
    fig_spaghetti = create_interactive_spaghetti(condition_traces, condition_data, normalize)
    st.plotly_chart(fig_spaghetti, width="stretch")
    
    # Small multiples by experiment
    st.markdown("### By Experiment (Batch Effect Check)")
    
    experiments = sorted(condition_data['exp_id'].unique())
    n_exp = len(experiments)
    
    if n_exp > 1:
        cols = st.columns(min(n_exp, 3))
        
        for i, exp in enumerate(experiments):
            with cols[i % 3]:
                exp_data = condition_data[condition_data['exp_id'] == exp]
                exp_traces = condition_traces[condition_traces['cell_id'].isin(exp_data['cell_id'])]
                
                fig_exp = create_mini_spaghetti(exp_traces, exp)
                st.plotly_chart(fig_exp, width="stretch")
    
    # Parameter distributions
    st.markdown("### Parameter Distributions")
    
    params = ['mobile_frac', 'k', 't_half']
    
    for param in params:
        fig_dist = create_distribution_strip(condition_data, param, selected_condition)
        st.plotly_chart(fig_dist, width="stretch")


def create_interactive_spaghetti(traces, features, normalize=False):
    """Create interactive spaghetti plot with selection"""
    fig = go.Figure()
    
    # Plot individual cells
    for cell_id in features['cell_id'].unique():
        cell_traces = traces[traces['cell_id'] == cell_id].sort_values('t')
        
        y_data = cell_traces['signal_norm'] if normalize else cell_traces['signal_corr']
        
        fig.add_trace(go.Scatter(
            x=cell_traces['t'],
            y=y_data,
            mode='lines',
            line=dict(color='gray', width=0.5),
            opacity=0.3,
            name=f'Cell {cell_id}',
            legendgroup='cells',
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Compute mean
    time_points = sorted(traces['t'].unique())
    means = []
    stds = []
    
    for t in time_points:
        t_data = traces[traces['t'] == t]['signal_corr' if not normalize else 'signal_norm']
        means.append(t_data.mean())
        stds.append(t_data.std())
    
    means = np.array(means)
    stds = np.array(stds)
    
    # Mean with CI
    fig.add_trace(go.Scatter(
        x=time_points + time_points[::-1],
        y=list(means + 1.96*stds/np.sqrt(len(features))) + list((means - 1.96*stds/np.sqrt(len(features)))[::-1]),
        fill='toself',
        fillcolor='rgba(0,100,200,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False,
        name='95% CI'
    ))
    
    fig.add_trace(go.Scatter(
        x=time_points,
        y=means,
        mode='lines',
        line=dict(color='blue', width=3),
        name='Mean'
    ))
    
    fig.update_layout(
        title=f"Recovery Curves (n={len(features)} cells)",
        xaxis_title="Time (s)",
        yaxis_title="Signal (normalized)" if normalize else "Signal (corrected)",
        height=400,
        hovermode='x unified'
    )
    
    return fig


def create_mini_spaghetti(traces, exp_name):
    """Create small multiple spaghetti plot"""
    fig = go.Figure()
    
    for cell_id in traces['cell_id'].unique():
        cell_data = traces[traces['cell_id'] == cell_id].sort_values('t')
        fig.add_trace(go.Scatter(
            x=cell_data['t'],
            y=cell_data['signal_corr'],
            mode='lines',
            line=dict(width=1),
            showlegend=False,
            opacity=0.5
        ))
    
    fig.update_layout(
        title=f"{exp_name}",
        xaxis_title="Time (s)",
        yaxis_title="Signal",
        height=200,
        margin=dict(l=40, r=20, t=40, b=30)
    )
    
    return fig


def create_distribution_strip(data, param, condition):
    """Create distribution strip with violin and swarm"""
    fig = go.Figure()
    
    # Violin
    fig.add_trace(go.Violin(
        y=data[param],
        name=condition,
        box_visible=True,
        meanline_visible=True,
        fillcolor='lightblue',
        opacity=0.6,
        x0=condition
    ))
    
    # Add individual points with jitter
    jitter = np.random.normal(0, 0.04, len(data))
    
    fig.add_trace(go.Scatter(
        x=[condition]*len(data) + jitter,
        y=data[param],
        mode='markers',
        marker=dict(size=4, color='darkblue', opacity=0.5),
        showlegend=False
    ))
    
    fig.update_layout(
        title=f"{param.replace('_', ' ').title()}",
        yaxis_title=param,
        height=250,
        showlegend=False
    )
    
    return fig


# ============================================================================
# MULTI-GROUP COMPARISONS
# ============================================================================

def render_multigroup_workspace():
    """Render multi-group comparison workspace"""
    st.markdown("## 🔬 Multi-Group Comparisons")
    
    df_cohort = build_cohort_query()
    
    if df_cohort.empty or 'condition' not in df_cohort.columns:
        st.warning("Need at least 2 conditions for comparisons")
        return
    
    conditions = sorted(df_cohort['condition'].unique())
    
    if len(conditions) < 2:
        st.warning("Need at least 2 conditions")
        return
    
    # Run statistical analysis
    with st.spinner("Computing statistics..."):
        stats_results = analyze(
            df_cohort,
            group_col='condition',
            batch_col='exp_id',
            params=['mobile_frac', 'k', 't_half', 'pre_bleach'],
            n_bootstrap=500  # Reduced for UI responsiveness
        )
    
    if 'comparisons' in stats_results:
        results_df = stats_results['comparisons']
        
        # Pairwise comparison matrix
        st.markdown("### Pairwise Comparisons")
        
        params = results_df['param'].unique()
        comparisons = results_df['comparison'].unique()
        
        # Create heatmap of effect sizes
        fig_heatmap = create_effect_size_heatmap(results_df, params, comparisons)
        st.plotly_chart(fig_heatmap, width="stretch")
        
        # Detailed results table
        st.markdown("### Detailed Results")
        
        # Format table
        display_df = results_df[[
            'param', 'comparison', 'beta', 'hedges_g', 
            'p', 'q', 'significant'
        ]].copy()
        
        display_df['beta'] = display_df['beta'].apply(lambda x: f"{x:.3f}")
        display_df['hedges_g'] = display_df['hedges_g'].apply(lambda x: f"{x:.3f}")
        display_df['p'] = display_df['p'].apply(lambda x: f"{x:.4f}")
        display_df['q'] = display_df['q'].apply(lambda x: f"{x:.4f}")
        display_df['significant'] = display_df['significant'].apply(lambda x: "✓" if x else "")
        
        st.dataframe(display_df, width="stretch", hide_index=True)
        
        # Volcano-style plot if many parameters
        if len(params) >= 4:
            st.markdown("### Volcano Plot")
            fig_volcano = create_volcano_plot(results_df)
            st.plotly_chart(fig_volcano, width="stretch")


def create_effect_size_heatmap(results_df, params, comparisons):
    """Create heatmap of effect sizes"""
    # Pivot data
    matrix = np.zeros((len(params), len(comparisons)))
    
    for i, param in enumerate(params):
        for j, comp in enumerate(comparisons):
            row = results_df[(results_df['param'] == param) & 
                           (results_df['comparison'] == comp)]
            if not row.empty:
                matrix[i, j] = row.iloc[0]['hedges_g']
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=comparisons,
        y=params,
        colorscale='RdBu',
        zmid=0,
        text=np.round(matrix, 2),
        texttemplate='%{text}',
        textfont={"size": 12},
        colorbar=dict(title="Hedges' g")
    ))
    
    fig.update_layout(
        title="Effect Sizes (Hedges' g)",
        xaxis_title="Comparison",
        yaxis_title="Parameter",
        height=300
    )
    
    return fig


def create_volcano_plot(results_df):
    """Create volcano plot for many parameters"""
    fig = go.Figure()
    
    # Plot each point
    for _, row in results_df.iterrows():
        color = 'red' if row['significant'] else 'gray'
        
        fig.add_trace(go.Scatter(
            x=[row['hedges_g']],
            y=[-np.log10(row['p'])],
            mode='markers',
            marker=dict(size=8, color=color),
            text=f"{row['param']}<br>{row['comparison']}",
            hovertemplate='%{text}<br>g=%{x:.3f}<br>p=%{y:.3f}<extra></extra>',
            showlegend=False
        ))
    
    # FDR threshold line
    fdr_threshold = 0.05
    fig.add_hline(y=-np.log10(fdr_threshold), line_dash="dash", 
                 annotation_text="FDR = 0.05")
    
    fig.update_layout(
        title="Effect Size vs Significance",
        xaxis_title="Effect Size (Hedges' g)",
        yaxis_title="-log₁₀(p-value)",
        height=400
    )
    
    return fig


# ============================================================================
# QC DASHBOARD
# ============================================================================

def render_qc_dashboard():
    """Render QC dashboard with interactive filtering"""
    st.markdown("## 🔍 Quality Control Dashboard")
    
    df_cohort = build_cohort_query()
    traces = st.session_state.roi_traces
    
    if df_cohort.empty:
        st.warning("No data to display")
        return
    
    # QC metrics overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        pass_rate = df_cohort['bleach_qc'].mean() * 100
        st.metric("QC Pass Rate", f"{pass_rate:.1f}%")
    
    with col2:
        median_drift = df_cohort['drift_px'].median()
        st.metric("Median Drift", f"{median_drift:.2f} px")
    
    with col3:
        median_r2 = df_cohort['r2'].median()
        st.metric("Median R²", f"{median_r2:.3f}")
    
    with col4:
        motion_rate = traces['qc_motion'].mean() * 100 if not traces.empty else 0
        st.metric("Motion Artifact Rate", f"{motion_rate:.1f}%")
    
    # Interactive histograms
    st.markdown("### QC Metric Distributions (Click to filter)")
    
    metric_col1, metric_col2 = st.columns(2)
    
    with metric_col1:
        fig_drift = create_interactive_histogram(
            df_cohort, 'drift_px', 'Drift (pixels)', 
            threshold=10, threshold_label="High drift"
        )
        st.plotly_chart(fig_drift, width="stretch")
    
    with metric_col2:
        fig_r2 = create_interactive_histogram(
            df_cohort, 'r2', 'R²',
            threshold=0.8, threshold_label="Good fit", invert=True
        )
        st.plotly_chart(fig_r2, width="stretch")
    
    # Tracking method usage
    st.markdown("### Tracking Method Usage")
    
    if 'roi_method' in df_cohort.columns:
        method_counts = df_cohort['roi_method'].value_counts()
        
        fig_methods = go.Figure(data=[go.Bar(
            x=method_counts.index,
            y=method_counts.values,
            marker_color=['blue', 'orange', 'green'][:len(method_counts)]
        )])
        
        fig_methods.update_layout(
            title="ROI Tracking Methods",
            xaxis_title="Method",
            yaxis_title="Count",
            height=300
        )
        
        st.plotly_chart(fig_methods, width="stretch")


def create_interactive_histogram(data, column, title, threshold=None, 
                                 threshold_label="", invert=False):
    """Create interactive histogram with threshold"""
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=data[column],
        nbinsx=30,
        marker_color='lightblue',
        name=title
    ))
    
    if threshold is not None:
        fig.add_vline(x=threshold, line_dash="dash", line_color="red",
                     annotation_text=threshold_label)
    
    fig.update_layout(
        title=title,
        xaxis_title=column,
        yaxis_title="Count",
        height=300
    )
    
    return fig


# ============================================================================
# EXPORT AND REPORTING
# ============================================================================

def render_export_panel():
    """Render export and reporting panel"""
    st.markdown("## 📤 Export & Reporting")
    
    # Export options
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Quick Exports")
        
        if st.button("📊 Export Current Cohort (CSV)", width="stretch"):
            try:
                data, filename = export_current_cohort(format='csv')
                st.download_button(
                    "⬇ Download CSV",
                    data,
                    filename,
                    "text/csv",
                    key="download_cohort_csv",
                    width="stretch"
                )
            except Exception as e:
                st.error(f"Export failed: {e}")
        
        if st.button("📈 Export All Traces (CSV)", width="stretch"):
            try:
                cohort_cells = build_cohort_query()['cell_id'].unique()
                data, filename = export_traces(cohort_cells, format='csv')
                st.download_button(
                    "⬇ Download Traces CSV",
                    data,
                    filename,
                    "text/csv",
                    key="download_traces_csv",
                    width="stretch"
                )
            except Exception as e:
                st.error(f"Export failed: {e}")
    
    with col2:
        st.markdown("### Report Generation")
        
        report_format = st.radio("Format", ["PDF", "HTML"], horizontal=True)
        report_name = st.text_input("Report filename", value=f"frap_report_{datetime.now().strftime('%Y%m%d')}")
        
        if st.button(f"📄 Generate {report_format} Report", width="stretch"):
            cohort_df = build_cohort_query()
            
            if cohort_df.empty:
                st.error("No data in cohort to report")
            else:
                with st.spinner(f"Generating {report_format} report..."):
                    try:
                        # Get statistics if available
                        stats_results = None
                        if 'condition' in cohort_df.columns and cohort_df['condition'].nunique() >= 2:
                            stats_results = analyze(
                                cohort_df,
                                params=['mobile_frac', 'k', 't_half'],
                                n_bootstrap=200  # Reduced for UI speed
                            )
                        
                        # Generate report
                        output_file = f"{report_name}.{report_format.lower()}"
                        
                        success = build_report(
                            cohort_df,
                            stats_results=stats_results,
                            figures=None,  # Add figures later
                            output_path=output_file,
                            format=report_format.lower(),
                            title=f"FRAP Analysis Report - {st.session_state.active_cohort}",
                            recipe=st.session_state.recipe
                        )
                        
                        if success:
                            st.success(f"✓ Report generated: {output_file}")
                            
                            # Offer download
                            with open(output_file, 'rb') as f:
                                file_data = f.read()
                                st.download_button(
                                    f"⬇ Download {report_format}",
                                    file_data,
                                    output_file,
                                    mime='application/pdf' if report_format == 'PDF' else 'text/html',
                                    width="stretch"
                                )
                        else:
                            st.error("Report generation failed. Check console for errors.")
                    
                    except Exception as e:
                        st.error(f"Error generating report: {e}")
                        logger.exception("Report generation error")
    
    # Figure presets
    st.markdown("### Figure Presets")
    
    preset_name = st.text_input("Preset name", key="preset_name_input")
    
    if st.button("💾 Save Current View as Preset"):
        if preset_name:
            st.session_state.figure_presets[preset_name] = {
                'cohort': st.session_state.active_cohort,
                'filters': st.session_state.active_filters.copy(),
                'timestamp': datetime.now().isoformat()
            }
            st.success(f"Preset '{preset_name}' saved!")
    
    # Analysis recipe export
    st.markdown("### Analysis Recipe")
    
    recipe = st.session_state.recipe
    recipe['filters'] = st.session_state.active_filters
    recipe['timestamp'] = datetime.now().isoformat()
    recipe['hash'] = compute_recipe_hash()
    
    recipe_json = json.dumps(recipe, indent=2)
    
    st.code(recipe_json, language='json')
    
    st.download_button(
        "📋 Download Recipe",
        recipe_json,
        "analysis_recipe.json",
        "application/json",
        key="download_recipe"
    )


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application"""
    st.set_page_config(
        page_title="FRAP Single-Cell Analysis",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize
    init_session_state()
    
    # Apply theme
    if st.session_state.dark_mode:
        st.markdown("""
        <style>
        .stApp {
            background-color: #0e1117;
            color: #fafafa;
        }
        </style>
        """, unsafe_allow_html=True)
    
    # Left rail
    render_left_rail()
    
    # Main content
    st.title("🔬 FRAP Single-Cell Analysis Platform")
    
    # Summary header
    render_summary_header()
    
    # Cohort builder
    with st.expander("🔍 Cohort Builder", expanded=True):
        render_cohort_builder()
    
    # Main workspace tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔬 Single-cell",
        "📊 Group",
        "🔬 Multi-group",
        "🔍 QC Dashboard"
    ])
    
    with tab1:
        render_single_cell_inspector()
    
    with tab2:
        render_group_workspace()
    
    with tab3:
        render_multigroup_workspace()
    
    with tab4:
        render_qc_dashboard()
    
    # Export panel
    with st.expander("📤 Export & Reporting"):
        render_export_panel()
    
    # Footer
    st.divider()
    st.markdown(
        f"**FRAP Analysis v1.0** | Recipe #{compute_recipe_hash()} | "
        f"Python {st.session_state.recipe['software_versions']['python']}"
    )


if __name__ == '__main__':
    main()
