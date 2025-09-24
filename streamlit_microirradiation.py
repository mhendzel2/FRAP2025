"""
Microirradiation Analysis Platform
A comprehensive analysis application for laser microirradiation experiments including:
- Protein recruitment kinetics to DSB sites
- Chromatin decondensation (ROI expansion) analysis  
- Combined microirradiation + photobleaching analysis
- Adaptive mask generation for expanding damage regions
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import io
from scipy.optimize import curve_fit
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, Optional, Tuple, List
import logging
import tempfile
import shutil

# Import our microirradiation modules
from microirradiation_core import (
    MicroirradiationResult, 
    analyze_recruitment_kinetics, 
    analyze_roi_expansion, 
    analyze_combined_experiment,
    single_exponential_recruitment,
    double_exponential_recruitment,
    sigmoidal_recruitment,
    exponential_expansion,
    linear_expansion,
    power_law_expansion
)
from microirradiation_image_analysis import MicroirradiationImageAnalyzer, ROIExpansionData

# Import existing FRAP infrastructure for combined analysis
from frap_manager import FRAPDataManager
from frap_pdf_reports import generate_pdf_report
from frap_core import FRAPAnalysisCore

# Page configuration
st.set_page_config(
    page_title="Microirradiation Analysis", 
    page_icon="⚡", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure data directory exists
os.makedirs('data', exist_ok=True)

# Initialize session state
if 'microirradiation_data' not in st.session_state:
    st.session_state.microirradiation_data = {}
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}

def create_recruitment_plot(time_data, intensity_data, fit_results=None, damage_frame=0):
    """Create plotly figure for recruitment kinetics"""
    fig = go.Figure()
    
    # Raw data
    fig.add_trace(go.Scatter(
        x=time_data,
        y=intensity_data,
        mode='markers',
        name='Raw Data',
        marker=dict(size=6, opacity=0.7)
    ))
    
    # Add fits if available
    if fit_results and fit_results['best_fit']:
        best_model = fit_results['best_model']
        best_fit = fit_results['best_fit']
        
        # Generate fitted curve
        t_fit = np.linspace(0, np.max(time_data), 200)
        
        if best_model == 'single_exp':
            y_fit = single_exponential_recruitment(t_fit, *best_fit['params'])
            model_name = f"Single Exponential (R² = {best_fit['r_squared']:.3f})"
        elif best_model == 'double_exp':
            y_fit = double_exponential_recruitment(t_fit, *best_fit['params'])
            model_name = f"Double Exponential (R² = {best_fit['r_squared']:.3f})"
        elif best_model == 'sigmoidal':
            y_fit = sigmoidal_recruitment(t_fit, *best_fit['params'])
            model_name = f"Sigmoidal (R² = {best_fit['r_squared']:.3f})"
        else:
            y_fit = None
            model_name = "Unknown Model"
        
        if y_fit is not None:
            fig.add_trace(go.Scatter(
                x=t_fit,
                y=y_fit,
                mode='lines',
                name=model_name,
                line=dict(width=3)
            ))
    
    # Mark damage event
    fig.add_vline(
        x=0, 
        line=dict(color="red", width=2, dash="dash"),
        annotation_text="Microirradiation"
    )
    
    fig.update_layout(
        title="Protein Recruitment Kinetics",
        xaxis_title="Time (s)",
        yaxis_title="Intensity (AU)",
        template="plotly_white",
        showlegend=True
    )
    
    return fig


def create_expansion_plot(time_data, area_data, fit_results=None):
    """Create plotly figure for ROI expansion"""
    fig = go.Figure()
    
    # Raw data
    fig.add_trace(go.Scatter(
        x=time_data,
        y=area_data,
        mode='markers+lines',
        name='ROI Area',
        marker=dict(size=6),
        line=dict(width=2)
    ))
    
    # Add fits if available
    if fit_results and fit_results['best_fit']:
        best_model = fit_results['best_model']
        best_fit = fit_results['best_fit']
        
        # Generate fitted curve
        t_fit = np.linspace(0, np.max(time_data), 200)
        
        if best_model == 'exponential':
            y_fit = exponential_expansion(t_fit, *best_fit['params'])
            model_name = f"Exponential (R² = {best_fit['r_squared']:.3f})"
        elif best_model == 'linear':
            y_fit = linear_expansion(t_fit, *best_fit['params'])
            model_name = f"Linear (R² = {best_fit['r_squared']:.3f})"
        elif best_model == 'power_law':
            y_fit = power_law_expansion(t_fit, *best_fit['params'])
            model_name = f"Power Law (R² = {best_fit['r_squared']:.3f})"
        else:
            y_fit = None
            model_name = "Unknown Model"
        
        if y_fit is not None:
            fig.add_trace(go.Scatter(
                x=t_fit,
                y=y_fit,
                mode='lines',
                name=model_name,
                line=dict(width=3, dash='dash')
            ))
    
    fig.update_layout(
        title="Chromatin Decondensation (ROI Expansion)",
        xaxis_title="Time (s)",
        yaxis_title="ROI Area (µm²)",
        template="plotly_white",
        showlegend=True
    )
    
    return fig


def analyze_data_file(df, file_info):
    """Analyze uploaded microirradiation data file"""
    
    st.subheader(f"📊 Analysis: {file_info['name']}")
    
    # Data validation
    required_cols = ['time', 'intensity']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        return None
    
    # Extract data
    time_data = df['time'].values
    intensity_data = df['intensity'].values
    
    # Optional ROI area data
    has_area_data = 'roi_area' in df.columns
    area_data = df['roi_area'].values if has_area_data else None
    
    # Analysis parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        damage_frame = st.number_input(
            "Damage frame index", 
            min_value=0, 
            max_value=len(time_data)-1, 
            value=0,
            help="Frame when microirradiation occurred"
        )
    
    with col2:
        recruitment_models = st.multiselect(
            "Recruitment models to fit",
            ['single_exp', 'double_exp', 'sigmoidal'],
            default=['single_exp', 'double_exp'],
            help="Mathematical models for recruitment kinetics"
        )
    
    with col3:
        expansion_models = st.multiselect(
            "Expansion models to fit",
            ['exponential', 'linear', 'power_law'],
            default=['exponential', 'linear'],
            help="Mathematical models for ROI expansion"
        ) if has_area_data else []
    
    if st.button(f"🔬 Analyze {file_info['name']}", key=f"analyze_{file_info['name']}"):
        
        with st.spinner("Analyzing recruitment kinetics..."):
            # Analyze recruitment kinetics
            recruitment_results = analyze_recruitment_kinetics(
                time_data, intensity_data, 
                damage_frame=damage_frame,
                models=recruitment_models
            )
        
        # Analyze ROI expansion if data available
        expansion_results = None
        if has_area_data and area_data is not None:
            with st.spinner("Analyzing ROI expansion..."):
                expansion_results = analyze_roi_expansion(
                    time_data, area_data,
                    damage_frame=damage_frame,
                    models=expansion_models
                )
        
        # Store results
        analysis_key = f"{file_info['name']}_analysis"
        st.session_state.analysis_results[analysis_key] = {
            'recruitment': recruitment_results,
            'expansion': expansion_results,
            'data': {
                'time': time_data,
                'intensity': intensity_data,
                'area': area_data
            },
            'params': {
                'damage_frame': damage_frame,
                'recruitment_models': recruitment_models,
                'expansion_models': expansion_models
            }
        }
        
        st.success("Analysis complete!")
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Recruitment Kinetics Results**")
            if recruitment_results['best_fit']:
                best_fit = recruitment_results['best_fit']
                st.write(f"- **Best model**: {recruitment_results['best_model']}")
                st.write(f"- **Recruitment rate**: {best_fit.get('rate', 'N/A'):.4f} s⁻¹")
                st.write(f"- **Half-time**: {best_fit.get('half_time', 'N/A'):.2f} s")
                st.write(f"- **Amplitude**: {best_fit.get('amplitude', 'N/A'):.2f}")
                st.write(f"- **R²**: {best_fit.get('r_squared', 'N/A'):.4f}")
                st.write(f"- **AIC**: {best_fit.get('aic', 'N/A'):.2f}")
            else:
                st.write("No successful recruitment fit")
        
        with col2:
            if expansion_results:
                st.write("**ROI Expansion Results**")
                if expansion_results['best_fit']:
                    best_fit = expansion_results['best_fit']
                    st.write(f"- **Best model**: {expansion_results['best_model']}")
                    st.write(f"- **Expansion rate**: {best_fit.get('rate', 'N/A'):.4f}")
                    st.write(f"- **Initial area**: {best_fit.get('initial_size', 'N/A'):.2f} µm²")
                    if 'max_expansion' in best_fit:
                        st.write(f"- **Max expansion**: {best_fit['max_expansion']:.2f} µm²")
                    st.write(f"- **R²**: {best_fit.get('r_squared', 'N/A'):.4f}")
                    st.write(f"- **AIC**: {best_fit.get('aic', 'N/A'):.2f}")
                else:
                    st.write("No successful expansion fit")
            else:
                st.write("**ROI Expansion**: No area data provided")
        
        # Create plots
        fig_recruitment = create_recruitment_plot(
            time_data[damage_frame:] - time_data[damage_frame],
            intensity_data[damage_frame:],
            recruitment_results
        )
        st.plotly_chart(fig_recruitment, use_container_width=True)
        
        if expansion_results:
            fig_expansion = create_expansion_plot(
                time_data[damage_frame:] - time_data[damage_frame],
                area_data[damage_frame:],
                expansion_results
            )
            st.plotly_chart(fig_expansion, use_container_width=True)
        
        return {
            'recruitment': recruitment_results,
            'expansion': expansion_results
        }
    
    return None


def create_comparison_plots(analysis_results):
    """Create comparison plots for multiple experiments"""
    
    if len(analysis_results) < 2:
        st.info("Need at least 2 analyzed files for comparison")
        return
    
    st.subheader("📈 Multi-Experiment Comparison")
    
    # Collect metrics from all experiments
    comparison_data = []
    
    for exp_name, results in analysis_results.items():
        if not exp_name.endswith('_analysis'):
            continue
            
        exp_name_clean = exp_name.replace('_analysis', '')
        
        # Recruitment metrics
        recruitment = results.get('recruitment', {})
        if recruitment.get('best_fit'):
            best_fit = recruitment['best_fit']
            
            row = {
                'Experiment': exp_name_clean,
                'Recruitment_Rate': best_fit.get('rate', np.nan),
                'Recruitment_Half_Time': best_fit.get('half_time', np.nan),
                'Recruitment_Amplitude': best_fit.get('amplitude', np.nan),
                'Recruitment_R_Squared': best_fit.get('r_squared', np.nan),
                'Recruitment_Model': recruitment.get('best_model', 'None')
            }
            
            # Expansion metrics
            expansion = results.get('expansion', {})
            if expansion and expansion.get('best_fit'):
                exp_fit = expansion['best_fit']
                row.update({
                    'Expansion_Rate': exp_fit.get('rate', np.nan),
                    'Initial_Area': exp_fit.get('initial_size', np.nan),
                    'Max_Expansion': exp_fit.get('max_expansion', np.nan),
                    'Expansion_R_Squared': exp_fit.get('r_squared', np.nan),
                    'Expansion_Model': expansion.get('best_model', 'None')
                })
            else:
                row.update({
                    'Expansion_Rate': np.nan,
                    'Initial_Area': np.nan,
                    'Max_Expansion': np.nan,
                    'Expansion_R_Squared': np.nan,
                    'Expansion_Model': 'None'
                })
            
            comparison_data.append(row)
    
    if not comparison_data:
        st.warning("No analysis results available for comparison")
        return
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Display summary table
    st.write("**Summary Statistics**")
    st.dataframe(comparison_df)
    
    # Create comparison plots
    metrics_to_plot = [
        ('Recruitment_Rate', 'Recruitment Rate (s⁻¹)'),
        ('Recruitment_Half_Time', 'Recruitment Half-Time (s)'),
        ('Recruitment_Amplitude', 'Recruitment Amplitude'),
        ('Expansion_Rate', 'Expansion Rate'),
        ('Max_Expansion', 'Max Expansion (µm²)')
    ]
    
    # Select metrics for plotting
    selected_metrics = st.multiselect(
        "Select metrics to plot",
        [metric[0] for metric in metrics_to_plot],
        default=[metric[0] for metric in metrics_to_plot[:3]]
    )
    
    if selected_metrics:
        # Create subplot
        n_metrics = len(selected_metrics)
        cols = min(3, n_metrics)
        rows = (n_metrics + cols - 1) // cols
        
        for i, metric in enumerate(selected_metrics):
            if i % cols == 0:
                plot_cols = st.columns(cols)
            
            with plot_cols[i % cols]:
                metric_label = next((label for m, label in metrics_to_plot if m == metric), metric)
                
                # Remove NaN values for plotting
                plot_data = comparison_df[['Experiment', metric]].dropna()
                
                if not plot_data.empty:
                    fig = px.bar(
                        plot_data, 
                        x='Experiment', 
                        y=metric,
                        title=metric_label
                    )
                    fig.update_layout(
                        xaxis_tickangle=-45,
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write(f"No data for {metric_label}")
    
    # Statistical comparison
    if len(comparison_df) >= 3:
        st.subheader("📊 Statistical Analysis")
        
        # Select metric for statistical analysis
        stat_metric = st.selectbox(
            "Select metric for statistical analysis",
            [metric[0] for metric in metrics_to_plot if metric[0] in comparison_df.columns]
        )
        
        if stat_metric and not comparison_df[stat_metric].isna().all():
            values = comparison_df[stat_metric].dropna()
            
            if len(values) >= 3:
                # Basic statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean", f"{np.mean(values):.4f}")
                with col2:
                    st.metric("Std Dev", f"{np.std(values):.4f}")
                with col3:
                    st.metric("Min", f"{np.min(values):.4f}")
                with col4:
                    st.metric("Max", f"{np.max(values):.4f}")
                
                # Distribution plot
                fig = px.histogram(
                    x=values,
                    nbins=min(10, len(values)),
                    title=f"Distribution of {stat_metric}"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Export comparison results
    if st.button("💾 Export Comparison Results"):
        csv = comparison_df.to_csv(index=False)
        st.download_button(
            label="Download comparison as CSV",
            data=csv,
            file_name="microirradiation_comparison.csv",
            mime="text/csv"
        )


# Main application
def main():
    st.title("⚡ Microirradiation Analysis Platform")
    
    st.markdown("""
    ### Comprehensive analysis for laser microirradiation experiments
    
    **Key Features:**
    - 🎯 **Protein recruitment kinetics** to DNA damage sites
    - 📏 **Chromatin decondensation** (ROI expansion) measurement
    - 🔄 **Combined microirradiation + photobleaching** analysis
    - 🖼️ **Direct image analysis** with adaptive masking
    - 📊 **Statistical comparison** across experiments
    """)
    
    # Sidebar controls
    with st.sidebar:
        st.header("🛠️ Analysis Options")
        
        analysis_mode = st.selectbox(
            "Analysis Mode",
            [
                "📁 Data File Analysis", 
                "🖼️ Image Stack Analysis",
                "📊 Results Comparison",
                "📚 Help & Documentation"
            ]
        )
    
    # Main content based on selected mode
    if analysis_mode == "📁 Data File Analysis":
        
        st.header("📁 Data File Analysis")
        st.markdown("Upload CSV/Excel files with time-series intensity and optional ROI area data")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload data files",
            type=['csv', 'xlsx', 'xls'],
            accept_multiple_files=True,
            help="Files should contain columns: 'time', 'intensity', and optionally 'roi_area'"
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                try:
                    # Read file
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                    
                    file_info = {
                        'name': uploaded_file.name,
                        'size': uploaded_file.size
                    }
                    
                    # Analyze file
                    with st.expander(f"📋 {uploaded_file.name}", expanded=True):
                        st.write(f"**Shape**: {df.shape}")
                        st.write(f"**Columns**: {list(df.columns)}")
                        
                        # Show data preview
                        st.write("**Data Preview:**")
                        st.dataframe(df.head())
                        
                        # Analyze data
                        analyze_data_file(df, file_info)
                        
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        
        # Results comparison
        if st.session_state.analysis_results:
            st.markdown("---")
            create_comparison_plots(st.session_state.analysis_results)
    
    elif analysis_mode == "🖼️ Image Stack Analysis":
        
        st.header("🖼️ Image Stack Analysis")
        st.markdown("Direct analysis of microscopy image stacks with automated damage detection and ROI tracking")
        
        # Initialize image analyzer
        if 'image_analyzer' not in st.session_state:
            st.session_state.image_analyzer = MicroirradiationImageAnalyzer()
        
        analyzer = st.session_state.image_analyzer
        
        # Create the image analysis interface
        analyzer.create_microirradiation_interface(None)
    
    elif analysis_mode == "📊 Results Comparison":
        
        st.header("📊 Results Comparison")
        
        if not st.session_state.analysis_results:
            st.info("No analysis results available. Analyze some data first!")
        else:
            create_comparison_plots(st.session_state.analysis_results)
            
            # Additional comparison tools
            st.subheader("🔧 Advanced Comparison Tools")
            
            # Clear results
            if st.button("🗑️ Clear All Results"):
                st.session_state.analysis_results = {}
                st.success("All results cleared!")
                st.experimental_rerun()
    
    elif analysis_mode == "📚 Help & Documentation":
        
        st.header("📚 Help & Documentation")
        
        st.markdown("""
        ## 🎯 Microirradiation Analysis Overview
        
        This platform analyzes laser microirradiation experiments to quantify:
        
        ### 1. Protein Recruitment Kinetics
        - Measures protein accumulation at DNA damage sites
        - Fits exponential and sigmoidal models
        - Calculates recruitment rates and half-times
        
        ### 2. Chromatin Decondensation
        - Tracks ROI expansion over time
        - Measures damage-induced chromatin relaxation
        - Fits multiple expansion models (exponential, linear, power-law)
        
        ### 3. Combined Analysis
        - Handles experiments with both microirradiation and photobleaching
        - Separates recruitment and recovery phases
        - Provides comprehensive kinetic analysis
        
        ## 📊 Data Format Requirements
        
        ### For File Analysis:
        **Required columns:**
        - `time`: Time points (seconds)
        - `intensity`: Fluorescence intensity at damage ROI
        
        **Optional columns:**
        - `roi_area`: ROI area measurements (µm²)
        - `background`: Background intensity
        
        ### For Image Analysis:
        - TIFF image stacks (time-lapse)
        - Calibrated pixel size and time intervals
        - Clear damage site visibility
        
        ## 🔬 Analysis Models
        
        ### Recruitment Kinetics:
        1. **Single Exponential**: I(t) = baseline + A(1 - e^(-kt))
        2. **Double Exponential**: Two-component recruitment
        3. **Sigmoidal**: Recruitment with lag phase
        
        ### ROI Expansion:
        1. **Exponential**: Area(t) = A₀ + A_max(1 - e^(-kt))
        2. **Linear**: Area(t) = A₀ + kt
        3. **Power Law**: Area(t) = A₀ + at^b
        
        ## 📈 Interpretation Guide
        
        ### Key Parameters:
        - **Recruitment Rate (k)**: Speed of protein accumulation (s⁻¹)
        - **Half-time (t₁/₂)**: Time to reach 50% of maximum recruitment
        - **Amplitude**: Maximum recruitment level
        - **Expansion Rate**: Speed of chromatin decondensation
        - **R²**: Goodness of fit (closer to 1 is better)
        - **AIC**: Model selection criterion (lower is better)
        
        ## 🛠️ Troubleshooting
        
        ### Common Issues:
        1. **Damage site not detected**: Try manual selection or adjust detection parameters
        2. **Poor fits**: Check data quality, try different models
        3. **No ROI expansion**: Verify damage actually causes chromatin decondensation
        4. **Noisy data**: Consider pre-processing or background correction
        
        ### Best Practices:
        - Use appropriate time resolution (not too sparse or dense)
        - Ensure good signal-to-noise ratio
        - Include sufficient pre-damage baseline
        - Monitor for photobleaching artifacts
        - Use consistent imaging conditions across experiments
        """)


if __name__ == "__main__":
    main()