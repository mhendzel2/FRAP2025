import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from io import BytesIO

# Import our new enhanced modules
from frap_input_handler import FRAPInputHandler, FRAPCurveData
from frap_analysis_enhanced import FRAPGroupAnalyzer, FRAPStatisticalComparator
from frap_visualizer import FRAPVisualizer
from frap_report_generator import FRAPReportGenerator

# Configure Page
st.set_page_config(
    page_title="FRAP Analysis Enhanced",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Session State Management ---
if 'data_groups' not in st.session_state:
    st.session_state.data_groups = {} # {group_name: FRAPGroupAnalyzer}
if 'current_group' not in st.session_state:
    st.session_state.current_group = None

# --- Sidebar Navigation ---
st.sidebar.title("FRAP Analysis 2.0")
page = st.sidebar.radio("Workflow", ["1. Import & Preprocess", "2. Model Fitting", "3. Subpopulations", "4. Compare Groups", "5. Report"])

# --- Helper Functions ---
def load_and_process_file(uploaded_file, bleach_frame_idx):
    # Save to temp file to load with our handler (or adapt handler to read bytes)
    # For now, let's assume CSV text
    try:
        # Create a temp file
        with open("temp_upload.csv", "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        # Load
        curve_data = FRAPInputHandler.load_csv("temp_upload.csv")
        
        # Preprocess
        curve_data = FRAPInputHandler.double_normalization(curve_data, bleach_frame_idx)
        curve_data = FRAPInputHandler.time_zero_correction(curve_data, bleach_frame_idx)
        
        return curve_data
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None

# --- Page 1: Import & Preprocess ---
if page == "1. Import & Preprocess":
    st.header("📂 Data Import & Preprocessing")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Upload Data")
        group_name = st.text_input("Group Name (e.g., 'Control', 'Mutant')", "Control")
        uploaded_files = st.file_uploader("Upload CSV Files", accept_multiple_files=True, type=['csv'])
        
        st.subheader("Settings")
        bleach_frame = st.number_input("Bleach Frame Index", min_value=1, value=10, help="Index of the frame where bleaching occurs (0-based)")
        
        if st.button("Process and Add to Group"):
            if not uploaded_files:
                st.warning("Please upload files.")
            else:
                if group_name not in st.session_state.data_groups:
                    st.session_state.data_groups[group_name] = FRAPGroupAnalyzer()
                
                analyzer = st.session_state.data_groups[group_name]
                count = 0
                progress_bar = st.progress(0)
                
                for i, file in enumerate(uploaded_files):
                    curve = load_and_process_file(file, bleach_frame)
                    if curve:
                        analyzer.add_curve(curve)
                        count += 1
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                st.success(f"Successfully added {count} curves to group '{group_name}'")
                st.session_state.current_group = group_name

    with col2:
        st.subheader("Current Data Groups")
        if st.session_state.data_groups:
            for name, analyzer in st.session_state.data_groups.items():
                st.info(f"**{name}**: {len(analyzer.curves)} curves loaded")
                
                # Preview first curve if available
                if analyzer.curves:
                    curve = analyzer.curves[0]
                    if curve.normalized_intensity is not None:
                        fig, ax = plt.subplots(figsize=(6, 2))
                        ax.plot(curve.time, curve.normalized_intensity, label='Normalized')
                        ax.axvline(curve.time[bleach_frame], color='r', linestyle='--', label='Bleach')
                        ax.set_title(f"Preview: First Curve in {name}")
                        ax.legend()
                        st.pyplot(fig)
        else:
            st.info("No data loaded yet.")

# --- Page 2: Model Fitting ---
elif page == "2. Model Fitting":
    st.header("📈 Model Fitting")
    
    if not st.session_state.data_groups:
        st.warning("Please import data first.")
    else:
        col1, col2 = st.columns([1, 3])
        
        with col1:
            group_select = st.selectbox("Select Group", list(st.session_state.data_groups.keys()))
            model_select = st.selectbox("Select Model", ["soumpasis", "single_exp", "double_exp"])
            
            if st.button("Fit Model"):
                with st.spinner("Fitting models..."):
                    analyzer = st.session_state.data_groups[group_select]
                    analyzer.analyze_group(model_name=model_select)
                    st.success("Fitting complete!")
        
        with col2:
            analyzer = st.session_state.data_groups[group_select]
            if analyzer.features is not None and not analyzer.features.empty:
                st.subheader("Fit Results")
                st.dataframe(analyzer.features.describe())
                
                st.subheader("Recovery Plot")
                # Prepare data for plotting
                times = analyzer.curves[0].time_post_bleach if analyzer.curves else None
                if times is not None:
                    data_intensities = [c.intensity_post_bleach for c in analyzer.curves if c.intensity_post_bleach is not None]
                    fitted_curves = [res.fitted_curve for res in analyzer.fit_results if res.success and res.fitted_curve is not None]
                    
                    fig = FRAPVisualizer.plot_recovery_curves(
                        times, data_intensities, fitted_curves, 
                        title=f"{group_select} - {model_select} Fit"
                    )
                    st.pyplot(fig)
                    
                    st.subheader("Residuals")
                    residuals = [res.residuals for res in analyzer.fit_results if res.success and res.residuals is not None]
                    if residuals:
                        fig_res = FRAPVisualizer.plot_residuals(times, residuals)
                        st.pyplot(fig_res)
            else:
                st.info("Run fitting to see results.")

# --- Page 3: Subpopulations ---
elif page == "3. Subpopulations":
    st.header("🔍 Subpopulation Analysis")
    
    if not st.session_state.data_groups:
        st.warning("Please import data first.")
    else:
        group_select = st.selectbox("Select Group", list(st.session_state.data_groups.keys()))
        analyzer = st.session_state.data_groups[group_select]
        
        if analyzer.features is None or analyzer.features.empty:
            st.warning("Please run model fitting first.")
        else:
            col1, col2 = st.columns([1, 3])
            with col1:
                st.subheader("Clustering")
                max_k = st.slider("Max Components", 2, 5, 3)
                if st.button("Detect Subpopulations"):
                    analyzer.detect_subpopulations(range(1, max_k + 1))
                    st.success("Clustering complete.")
                
                if st.button("Detect Outliers"):
                    analyzer.detect_outliers()
                    st.success("Outlier detection complete.")
            
            with col2:
                if 'subpopulation' in analyzer.features.columns:
                    st.subheader("Cluster Visualization")
                    # Dynamic parameter selection
                    params = [c for c in analyzer.features.columns if c not in ['subpopulation', 'is_outlier']]
                    x_axis = st.selectbox("X Axis", params, index=0)
                    y_axis = st.selectbox("Y Axis", params, index=min(1, len(params)-1))
                    
                    fig = FRAPVisualizer.plot_subpopulations(analyzer.features, x_axis, y_axis)
                    st.pyplot(fig)
                    
                    st.write("Subpopulation Counts:")
                    st.write(analyzer.features['subpopulation'].value_counts())

# --- Page 4: Compare Groups ---
elif page == "4. Compare Groups":
    st.header("⚖️ Statistical Comparison")
    
    if len(st.session_state.data_groups) < 2:
        st.warning("Need at least 2 groups to compare.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            group1 = st.selectbox("Group 1 (Control)", list(st.session_state.data_groups.keys()), index=0)
        with col2:
            group2 = st.selectbox("Group 2 (Test)", list(st.session_state.data_groups.keys()), index=1)
            
        if group1 == group2:
            st.error("Please select different groups.")
        else:
            analyzer1 = st.session_state.data_groups[group1]
            analyzer2 = st.session_state.data_groups[group2]
            
            if analyzer1.features is None or analyzer2.features is None:
                st.warning("Both groups must be fitted first.")
            else:
                param = st.selectbox("Parameter to Compare", analyzer1.features.columns)
                
                if st.button("Run Statistical Test"):
                    result = FRAPStatisticalComparator.compare_groups(analyzer1.features, analyzer2.features, param)
                    
                    st.subheader("Results")
                    st.json(result)
                    
                    st.subheader("Distribution Plot")
                    # Combine for plotting
                    df1 = analyzer1.features.copy()
                    df1['Group'] = group1
                    df2 = analyzer2.features.copy()
                    df2['Group'] = group2
                    combined = pd.concat([df1, df2])
                    
                    fig = FRAPVisualizer.plot_parameter_distribution(combined, param, group_col='Group')
                    st.pyplot(fig)

# --- Page 5: Report ---
elif page == "5. Report":
    st.header("📄 Report Generation")
    
    if st.button("Generate HTML Report"):
        if not st.session_state.data_groups:
            st.error("No data to report.")
        else:
            # Collect all figures and tables
            figures = {}
            all_features = pd.DataFrame()
            
            for name, analyzer in st.session_state.data_groups.items():
                if analyzer.features is not None:
                    df = analyzer.features.copy()
                    df['Group'] = name
                    all_features = pd.concat([all_features, df])
                    
                    # Generate recovery plot for report
                    times = analyzer.curves[0].time_post_bleach
                    data_intensities = [c.intensity_post_bleach for c in analyzer.curves if c.intensity_post_bleach is not None]
                    fitted_curves = [res.fitted_curve for res in analyzer.fit_results if res.success and res.fitted_curve is not None]
                    
                    fig = FRAPVisualizer.plot_recovery_curves(times, data_intensities, fitted_curves, title=f"{name} Recovery")
                    figures[f"{name}_Recovery"] = fig
            
            FRAPReportGenerator.generate_html_report(all_features, figures, "FRAP_Enhanced_Report.html")
            st.success("Report generated: FRAP_Enhanced_Report.html")
            
            with open("FRAP_Enhanced_Report.html", "rb") as f:
                st.download_button("Download Report", f, file_name="FRAP_Report.html")
