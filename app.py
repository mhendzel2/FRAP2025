import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from io import BytesIO
import zipfile
import tempfile
import os
import shutil
import openpyxl  # For Excel file support

# Import our new enhanced modules
from frap_input_handler import FRAPInputHandler, FRAPCurveData
from frap_analysis_enhanced import FRAPGroupAnalyzer, FRAPStatisticalComparator
from frap_visualizer import FRAPVisualizer
from frap_report_generator import EnhancedFRAPReportGenerator

# Import global fitting module
try:
    from frap_global_fitting import (
        UnifiedModelWorkflow, GlobalFitReporter, 
        convert_analyzer_to_dataframe, run_global_frap_analysis,
        LMFIT_AVAILABLE as GLOBAL_FITTING_AVAILABLE
    )
except ImportError:
    GLOBAL_FITTING_AVAILABLE = False

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
if 'processed_zip_files' not in st.session_state:
    st.session_state.processed_zip_files = set()
# Experimental parameters
if 'r2_threshold' not in st.session_state:
    st.session_state.r2_threshold = 0.5
if 'bleach_radius' not in st.session_state:
    st.session_state.bleach_radius = 1.0  # microns
if 'pixel_size' not in st.session_state:
    st.session_state.pixel_size = 0.065  # microns/pixel
# Multi-model comparison results
if 'model_comparison_results' not in st.session_state:
    st.session_state.model_comparison_results = {}  # {group_name: {'double': df, 'triple': df}}

# --- Sidebar Navigation ---
st.sidebar.title("FRAP Analysis 2.0")
page = st.sidebar.radio("Workflow", [
    "1. Import & Preprocess", 
    "2. Batch Process All Groups", 
    "3. Model Fitting", 
    "4. Subpopulations", 
    "5. Compare Groups",
    "6. Global Fitting",
    "7. Report"
])

# --- Sidebar Experimental Parameters ---
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Experimental Parameters")

with st.sidebar.expander("🔬 Physical Parameters", expanded=False):
    st.session_state.bleach_radius = st.number_input(
        "Bleach Region Radius (µm)",
        min_value=0.1,
        max_value=50.0,
        value=st.session_state.bleach_radius,
        step=0.1,
        help="Radius of the bleached region in micrometers"
    )
    
    st.session_state.pixel_size = st.number_input(
        "Pixel Size (µm/pixel)",
        min_value=0.001,
        max_value=1.0,
        value=st.session_state.pixel_size,
        step=0.001,
        format="%.3f",
        help="Size of one pixel in micrometers"
    )
    
    # Calculate bleach radius in pixels
    bleach_radius_pixels = st.session_state.bleach_radius / st.session_state.pixel_size
    st.info(f"📐 Bleach radius: **{bleach_radius_pixels:.1f} pixels**")

with st.sidebar.expander("📊 Quality Filters", expanded=True):
    st.session_state.r2_threshold = st.slider(
        "R² Threshold",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.r2_threshold,
        step=0.05,
        help="Minimum R² value for including fits in analysis. Fits below this threshold will be excluded."
    )
    
    st.caption(f"✅ Keep fits with R² ≥ {st.session_state.r2_threshold}")
    st.caption(f"❌ Remove fits with R² < {st.session_state.r2_threshold}")

# --- Helper Functions ---
def apply_r2_filter(features_df, r2_threshold):
    """
    Filter DataFrame to only include rows with R² >= threshold.
    Returns filtered DataFrame and count of removed rows.
    """
    if features_df is None or features_df.empty:
        return features_df, 0
    
    if 'r2' not in features_df.columns:
        logger.warning("R² column not found in features DataFrame")
        return features_df, 0
    
    original_count = len(features_df)
    filtered_df = features_df[features_df['r2'] >= r2_threshold].copy()
    removed_count = original_count - len(filtered_df)
    
    if removed_count > 0:
        logger.info(f"R² filter: Removed {removed_count}/{original_count} fits (R² < {r2_threshold})")
    
    return filtered_df, removed_count

def load_and_process_file(uploaded_file, bleach_frame_idx=None):
    # Save to temp file to load with our handler
    try:
        # Determine file extension
        file_ext = uploaded_file.name.lower().split('.')[-1]
        temp_filename = f"temp_upload.{file_ext}"
        
        # Create a temp file with correct extension
        with open(temp_filename, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        # Load using the generic load_file method
        curve_data = FRAPInputHandler.load_file(temp_filename)
        
        # Autodetect bleach frame if not provided
        if bleach_frame_idx is None:
            bleach_frame_idx = FRAPInputHandler.detect_bleach_frame(curve_data)
            # logger.info(f"Autodetected bleach frame at index {bleach_frame_idx} for {uploaded_file.name}")
        
        # Preprocess
        curve_data = FRAPInputHandler.double_normalization(curve_data, bleach_frame_idx)
        curve_data = FRAPInputHandler.time_zero_correction(curve_data, bleach_frame_idx)
        
        # Clean up temp file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        
        return curve_data
    except Exception as e:
        st.error(f"Error processing file {uploaded_file.name}: {e}")
        return None

def load_groups_from_zip(zip_file, bleach_frame_idx=None):
    """
    Load files from a ZIP archive where each subfolder becomes a group.
    Returns dict of {group_name: [list of FRAPCurveData]}
    """
    groups_data = {}
    success_count = 0
    error_count = 0
    error_details = []
    
    st.write("🔍 **Debug:** Starting ZIP processing...")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            st.write(f"📁 **Debug:** Created temp directory: {temp_dir}")
            
            # Extract ZIP
            try:
                with zipfile.ZipFile(BytesIO(zip_file.getbuffer())) as z:
                    file_list = z.namelist()
                    st.write(f"📦 **Debug:** ZIP contains {len(file_list)} files")
                    z.extractall(temp_dir)
                    st.write("✅ **Debug:** ZIP extracted successfully")
            except zipfile.BadZipFile as e:
                st.error(f"Invalid ZIP file format: {e}")
                return {}, 0, 1, [f"Bad ZIP file: {e}"]
            
            # Find all folders with CSV files
            folders_with_data = {}
            st.write(f"🔍 **Debug:** Walking directory tree...")
            
            for root, dirs, files in os.walk(temp_dir):
                data_files = [f for f in files 
                             if f.lower().endswith(('.csv', '.xls', '.xlsx')) 
                             and not f.startswith('.')]
                
                if data_files:
                    folder_name = os.path.basename(root)
                    if not folder_name.startswith('__') and not folder_name.startswith('.'):
                        folders_with_data[root] = {
                            'name': folder_name,
                            'files': data_files
                        }
                        st.write(f"📂 **Debug:** Found group '{folder_name}' with {len(data_files)} data files")
            
            if not folders_with_data:
                st.warning("⚠️ No folders with data files found in ZIP archive")
                return {}, 0, 0, ["No CSV, XLS, or XLSX files found in any subdirectories"]
            
            st.write(f"✅ **Debug:** Found {len(folders_with_data)} groups total")
            
            # Process each folder
            total_folders = len(folders_with_data)
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for folder_idx, (folder_path, folder_info) in enumerate(folders_with_data.items()):
                group_name = folder_info['name']
                status_text.text(f"Processing group {folder_idx + 1}/{total_folders}: {group_name}")
                progress_bar.progress((folder_idx + 1) / total_folders)
                
                if group_name not in groups_data:
                    groups_data[group_name] = []
                
                # Process each data file in this folder
                for data_file in folder_info['files']:
                    file_path = os.path.join(folder_path, data_file)
                    try:
                        curve_data = FRAPInputHandler.load_file(file_path)
                        
                        # Autodetect bleach frame if not provided
                        current_bleach_idx = bleach_frame_idx
                        if current_bleach_idx is None:
                            current_bleach_idx = FRAPInputHandler.detect_bleach_frame(curve_data)
                        
                        curve_data = FRAPInputHandler.double_normalization(curve_data, current_bleach_idx)
                        curve_data = FRAPInputHandler.time_zero_correction(curve_data, current_bleach_idx)
                        groups_data[group_name].append(curve_data)
                        success_count += 1
                    except Exception as e:
                        error_count += 1
                        error_msg = f"Error in {group_name}/{data_file}: {str(e)}"
                        error_details.append(error_msg)
                        st.write(f"❌ **Debug:** {error_msg}")
            
            progress_bar.empty()
            status_text.empty()
            st.write(f"✅ **Debug:** Processing complete! {success_count} files processed, {error_count} errors")
            
    except Exception as e:
        error_msg = f"Error processing ZIP archive: {e}"
        st.error(error_msg)
        st.write(f"❌ **Debug:** Exception: {type(e).__name__}: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return {}, 0, 1, [error_msg]
    
    return groups_data, success_count, error_count, error_details

# --- Page 1: Import & Preprocess ---
if page == "1. Import & Preprocess":
    st.header("📂 Data Import & Preprocessing")
    
    # Upload method selection
    upload_method = st.radio(
        "Choose upload method:",
        ["🚀 Bulk Upload (ZIP with Groups - Recommended)", "📄 Individual Files"],
        help="Bulk upload allows automatic group creation from ZIP folder structure"
    )
    
    st.markdown("---")
    
    if upload_method == "🚀 Bulk Upload (ZIP with Groups - Recommended)":
        st.subheader("📦 Bulk Upload: ZIP File with Group Subfolders")
        
        col_info1, col_info2 = st.columns([3, 2])
        with col_info1:
            st.markdown("""
            **Expected ZIP Structure:**
            ```
            your_archive.zip
            ├── Control/
            │   ├── cell1.csv
            │   ├── cell2.csv
            │   └── cell3.csv
            ├── Treatment_A/
            │   ├── cell1.csv
            │   └── cell2.csv
            └── Treatment_B/
                ├── cell1.csv
                └── cell2.csv
            ```
            Each subfolder becomes a separate group automatically.
            """)
        
        with col_info2:
            st.info("""
            **✨ Benefits:**
            - ⚡ Fast bulk processing
            - 📁 Auto group creation
            - 🗂️ Preserves organization
            - 🎯 Ready for analysis
            """)
        
        st.subheader("Settings")
        autodetect_bleach = st.checkbox("Autodetect Bleach Frame", value=True, help="Automatically detect the frame with minimum intensity as the bleach frame.")
        
        if not autodetect_bleach:
            bleach_frame_zip = st.number_input("Bleach Frame Index (for all files)", min_value=1, value=10, 
                                              help="Index of the frame where bleaching occurs (0-based)", key="bleach_zip")
        else:
            bleach_frame_zip = None
        
        uploaded_zip = st.file_uploader("📂 Select ZIP file containing grouped FRAP data", 
                                       type=['zip'], key="zip_uploader")
        
        if uploaded_zip:
            zip_file_id = f"{uploaded_zip.name}_{uploaded_zip.size}"
            st.info(f"📦 **Selected:** {uploaded_zip.name} ({uploaded_zip.size / 1024:.1f} KB)")
            
            col_btn1, col_btn2 = st.columns([1, 1])
            with col_btn1:
                if st.button("🚀 Process ZIP Archive", type="primary", use_container_width=True):
                    if zip_file_id in st.session_state.processed_zip_files:
                        st.warning("⚠️ This ZIP file has already been processed.")
                    else:
                        with st.spinner(f"Processing '{uploaded_zip.name}'..."):
                            groups_data, success, errors, error_details = load_groups_from_zip(
                                uploaded_zip, bleach_frame_zip
                            )
                            
                            if groups_data:
                                # Add curves to session state groups
                                for group_name, curves in groups_data.items():
                                    if group_name not in st.session_state.data_groups:
                                        st.session_state.data_groups[group_name] = FRAPGroupAnalyzer()
                                    
                                    analyzer = st.session_state.data_groups[group_name]
                                    for curve in curves:
                                        analyzer.add_curve(curve)
                                
                                st.session_state.processed_zip_files.add(zip_file_id)
                                st.success(f"✅ **Success!** Created {len(groups_data)} groups with {success} files total")
                                
                                with st.expander("📊 View Created Groups", expanded=True):
                                    for group_name in groups_data.keys():
                                        analyzer = st.session_state.data_groups[group_name]
                                        st.markdown(f"**📁 {group_name}**: {len(analyzer.curves)} curves")
                                
                                if errors > 0:
                                    st.warning(f"⚠️ {errors} files had errors")
                                    with st.expander("View errors"):
                                        for err in error_details:
                                            st.text(err)
                                
                                st.balloons()
                                st.info("🎯 **Next step:** Go to '2. Model Fitting' to analyze your data!")
            
            with col_btn2:
                if st.button("🔄 Upload Different File", use_container_width=True):
                    st.rerun()
    
    else:  # Individual Files mode
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Upload Data")
            group_name = st.text_input("Group Name (e.g., 'Control', 'Mutant')", "Control")
            uploaded_files = st.file_uploader("Upload Data Files (CSV, XLS, XLSX)", accept_multiple_files=True, type=['csv', 'xls', 'xlsx'])
            
            st.subheader("Settings")
            autodetect_bleach_ind = st.checkbox("Autodetect Bleach Frame", value=True, key="auto_bleach_ind")
            
            if not autodetect_bleach_ind:
                bleach_frame = st.number_input("Bleach Frame Index", min_value=1, value=10, 
                                              help="Index of the frame where bleaching occurs (0-based)")
            else:
                bleach_frame = None
            
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
                # Quick action button
                st.markdown("---")
                if st.button("⚡ Go to Batch Processing →", type="primary", use_container_width=True, key="goto_batch"):
                    st.session_state.page = "2. Batch Process All Groups"
                    st.rerun()
                st.markdown("---")
                
                for name, analyzer in st.session_state.data_groups.items():
                    st.info(f"**{name}**: {len(analyzer.curves)} curves loaded")
                    
                    # Preview first curve if available
                    if analyzer.curves:
                        curve = analyzer.curves[0]
                        if curve.normalized_intensity is not None:
                            fig, ax = plt.subplots(figsize=(6, 2))
                            ax.plot(curve.time, curve.normalized_intensity, label='Normalized')
                            bleach_idx = min(10, len(curve.time)-1)  # Safe bleach index for preview
                            if bleach_idx < len(curve.time):
                                ax.axvline(curve.time[bleach_idx], color='r', linestyle='--', label='Bleach')
                            ax.set_title(f"Preview: First Curve in {name}")
                            ax.legend()
                            st.pyplot(fig)
            else:
                st.info("No data loaded yet.")

# --- Page 2: Batch Process All Groups ---
elif page == "2. Batch Process All Groups":
    st.header("⚡ Batch Process All Groups")
    
    if not st.session_state.data_groups:
        st.warning("⚠️ Please import data first on the 'Import & Preprocess' page.")
    else:
        st.info(f"📊 **{len(st.session_state.data_groups)} groups** loaded with {sum(len(a.curves) for a in st.session_state.data_groups.values())} total curves")
        
        # Display current experimental parameters
        with st.expander("🔬 Current Experimental Parameters", expanded=False):
            col_p1, col_p2, col_p3 = st.columns(3)
            with col_p1:
                st.metric("Bleach Radius", f"{st.session_state.bleach_radius} µm")
            with col_p2:
                st.metric("Pixel Size", f"{st.session_state.pixel_size} µm/pixel")
            with col_p3:
                bleach_pixels = st.session_state.bleach_radius / st.session_state.pixel_size
                st.metric("Bleach Radius", f"{bleach_pixels:.1f} pixels")
            st.caption("💡 These parameters can be adjusted in the sidebar under 'Experimental Parameters'")
        
        # Configuration section
        st.subheader("⚙️ Processing Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔬 Model Fitting")
            fitting_mode = st.radio(
                "Fitting Strategy:",
                ["compare_2_vs_3", "fit_all", "single_model"],
                format_func=lambda x: {
                    "compare_2_vs_3": "📊 Compare 2-Component vs 3-Component (Recommended)",
                    "fit_all": "🚀 Fit All Models & Auto-Select Best",
                    "single_model": "🎯 Single Model Only"
                }[x],
                help="Compare 2 vs 3: Fits both double and triple exponential to all curves for direct comparison. Fit All: Tests all models and selects best by criterion."
            )
            
            if fitting_mode == "single_model":
                model_select = st.selectbox(
                    "Select Model",
                    ["single", "double", "triple", "anomalous_diffusion", "reaction_diffusion"],
                    format_func=lambda x: {
                        "single": "Single Exponential",
                        "double": "Double Exponential (2-Component)",
                        "triple": "Triple Exponential (3-Component)",
                        "anomalous_diffusion": "Anomalous Diffusion",
                        "reaction_diffusion": "Reaction-Diffusion"
                    }[x]
                )
            else:
                model_select = None
            
            criterion = st.selectbox(
                "Model Selection Criterion:",
                ["aicc", "aic", "bic", "adj_r2"],
                format_func=lambda x: {
                    "aicc": "AICc (Corrected AIC)",
                    "aic": "AIC",
                    "bic": "BIC",
                    "adj_r2": "Adjusted R²"
                }[x]
            )
        
        with col2:
            st.markdown("#### 🎯 Optional Analyses")
            
            st.info(f"📊 **Quality Filter:** R² ≥ {st.session_state.r2_threshold} (set in sidebar)")
            
            run_subpopulations = st.checkbox(
                "🔍 Detect Subpopulations",
                value=True,
                help="Use Gaussian Mixture Models to identify subpopulations in your data"
            )
            
            if run_subpopulations:
                max_k = st.slider("Max Subpopulations", 2, 5, 3, help="Maximum number of subpopulations to test")
            
            run_outliers = st.checkbox(
                "🎯 Detect Outliers",
                value=False,
                help="Use Isolation Forest to identify outlier curves"
            )
            
            exclude_outliers = st.checkbox(
                "🚫 Exclude Outliers from Results",
                value=False,
                disabled=not run_outliers,
                help="Remove detected outliers from final results and reports"
            )
        
        st.markdown("---")
        
        # Group selection
        st.subheader("📁 Select Groups to Process")
        select_all = st.checkbox("Select All Groups", value=True)
        
        if select_all:
            selected_groups = list(st.session_state.data_groups.keys())
        else:
            selected_groups = st.multiselect(
                "Choose groups:",
                list(st.session_state.data_groups.keys()),
                default=list(st.session_state.data_groups.keys())
            )
        
        st.info(f"✅ Will process **{len(selected_groups)}** group(s)")
        
        # Process button
        if st.button("🚀 Run Batch Processing", type="primary", use_container_width=True):
            if not selected_groups:
                st.error("Please select at least one group to process")
            else:
                # Create progress tracking
                overall_progress = st.progress(0)
                status_text = st.empty()
                
                results_summary = []
                
                for idx, group_name in enumerate(selected_groups):
                    analyzer = st.session_state.data_groups[group_name]
                    
                    status_text.markdown(f"### Processing: **{group_name}** ({idx+1}/{len(selected_groups)})")
                    
                    group_results = {
                        'group': group_name,
                        'n_curves': len(analyzer.curves),
                        'fit_success': 0,
                        'fit_failed': 0,
                        'r2_filtered': 0,
                        'subpopulations': 0,
                        'outliers': 0
                    }
                    
                    # Step 1: Model Fitting
                    with st.spinner(f"🔬 Fitting models for {group_name}..."):
                        try:
                            if fitting_mode == "compare_2_vs_3":
                                # Fit both 2-component (double) and 3-component (triple) separately
                                from frap_core import FRAPAnalysisCore
                                
                                double_results = []
                                triple_results = []
                                
                                for curve_idx, curve in enumerate(analyzer.curves):
                                    if curve.normalized_intensity is None:
                                        double_results.append({'curve_idx': curve_idx, 'success': False})
                                        triple_results.append({'curve_idx': curve_idx, 'success': False})
                                        continue
                                    
                                    time_data = curve.time
                                    intensity_data = curve.normalized_intensity
                                    
                                    # Fit double exponential (2-component)
                                    try:
                                        double_fit = FRAPAnalysisCore.fit_double_exponential(time_data, intensity_data)
                                        if double_fit and double_fit.get('success', False):
                                            double_results.append({
                                                'curve_idx': curve_idx,
                                                'success': True,
                                                'model': 'double',
                                                'r2': double_fit.get('r2', np.nan),
                                                'adj_r2': double_fit.get('adj_r2', np.nan),
                                                'aic': double_fit.get('aic', np.nan),
                                                'aicc': double_fit.get('aicc', np.nan),
                                                'bic': double_fit.get('bic', np.nan),
                                                'params': double_fit.get('params', []),
                                                'mobile_fraction': sum(double_fit.get('params', [0,0,0,0,0])[:3:2]) * 100 if len(double_fit.get('params', [])) >= 5 else np.nan,
                                                'k_fast': double_fit.get('params', [0,0])[1] if len(double_fit.get('params', [])) >= 2 else np.nan,
                                                'k_slow': double_fit.get('params', [0,0,0,0])[3] if len(double_fit.get('params', [])) >= 4 else np.nan
                                            })
                                        else:
                                            double_results.append({'curve_idx': curve_idx, 'success': False, 'model': 'double'})
                                    except Exception as e:
                                        double_results.append({'curve_idx': curve_idx, 'success': False, 'model': 'double', 'error': str(e)})
                                    
                                    # Fit triple exponential (3-component)
                                    try:
                                        triple_fit = FRAPAnalysisCore.fit_triple_exponential(time_data, intensity_data)
                                        if triple_fit and triple_fit.get('success', False):
                                            triple_results.append({
                                                'curve_idx': curve_idx,
                                                'success': True,
                                                'model': 'triple',
                                                'r2': triple_fit.get('r2', np.nan),
                                                'adj_r2': triple_fit.get('adj_r2', np.nan),
                                                'aic': triple_fit.get('aic', np.nan),
                                                'aicc': triple_fit.get('aicc', np.nan),
                                                'bic': triple_fit.get('bic', np.nan),
                                                'params': triple_fit.get('params', []),
                                                'mobile_fraction': sum(triple_fit.get('params', [0,0,0,0,0,0,0])[:5:2]) * 100 if len(triple_fit.get('params', [])) >= 7 else np.nan,
                                                'k_fast': triple_fit.get('params', [0,0])[1] if len(triple_fit.get('params', [])) >= 2 else np.nan,
                                                'k_medium': triple_fit.get('params', [0,0,0,0])[3] if len(triple_fit.get('params', [])) >= 4 else np.nan,
                                                'k_slow': triple_fit.get('params', [0,0,0,0,0,0])[5] if len(triple_fit.get('params', [])) >= 6 else np.nan
                                            })
                                        else:
                                            triple_results.append({'curve_idx': curve_idx, 'success': False, 'model': 'triple'})
                                    except Exception as e:
                                        triple_results.append({'curve_idx': curve_idx, 'success': False, 'model': 'triple', 'error': str(e)})
                                
                                # Create DataFrames for both models
                                double_df = pd.DataFrame([r for r in double_results if r.get('success', False)])
                                triple_df = pd.DataFrame([r for r in triple_results if r.get('success', False)])
                                
                                # Store comparison results
                                st.session_state.model_comparison_results[group_name] = {
                                    'double': double_df,
                                    'triple': triple_df
                                }
                                
                                # Use double as primary for analyzer.features (can be changed)
                                if not double_df.empty:
                                    analyzer.features = double_df.copy()
                                    group_results['fit_success'] = len(double_df) + len(triple_df)
                                elif not triple_df.empty:
                                    analyzer.features = triple_df.copy()
                                    group_results['fit_success'] = len(triple_df)
                                
                                st.success(f"✅ 2-Component: {len(double_df)} fits | 3-Component: {len(triple_df)} fits")
                                
                            else:
                                # Original behavior
                                analyzer.analyze_group(model_name=model_select)
                                
                                if analyzer.features is not None:
                                    group_results['fit_success'] = len(analyzer.features.dropna(subset=['r2']))
                                    group_results['fit_failed'] = len(analyzer.features) - group_results['fit_success']
                                    
                                    # Apply R² filter
                                    analyzer.features, removed_count = apply_r2_filter(
                                        analyzer.features, 
                                        st.session_state.r2_threshold
                                    )
                                    group_results['r2_filtered'] = removed_count
                                    
                                    if removed_count > 0:
                                        st.info(f"🔍 R² filter: Removed {removed_count} fits with R² < {st.session_state.r2_threshold}")
                                
                                st.success(f"✅ Model fitting complete: {group_results['fit_success']} successful fits")
                        except Exception as e:
                            st.error(f"❌ Error fitting models: {e}")
                            logger.error(f"Batch processing error for {group_name}: {e}")
                    
                    # Step 2: Subpopulation Detection
                    if run_subpopulations and analyzer.features is not None and not analyzer.features.empty:
                        with st.spinner(f"🔍 Detecting subpopulations for {group_name}..."):
                            try:
                                analyzer.detect_subpopulations(range(1, max_k + 1))
                                
                                if 'subpopulation' in analyzer.features.columns:
                                    group_results['subpopulations'] = analyzer.features['subpopulation'].nunique()
                                    st.success(f"✅ Found {group_results['subpopulations']} subpopulation(s)")
                            except Exception as e:
                                st.error(f"❌ Error detecting subpopulations: {e}")
                                logger.error(f"Subpopulation detection error for {group_name}: {e}")
                    
                    # Step 3: Outlier Detection
                    if run_outliers and analyzer.features is not None and not analyzer.features.empty:
                        with st.spinner(f"🎯 Detecting outliers for {group_name}..."):
                            try:
                                analyzer.detect_outliers()
                                
                                if 'is_outlier' in analyzer.features.columns:
                                    group_results['outliers'] = analyzer.features['is_outlier'].sum()
                                    st.success(f"✅ Detected {group_results['outliers']} outlier(s)")
                                    
                                    # Exclude outliers if requested
                                    if exclude_outliers and group_results['outliers'] > 0:
                                        original_count = len(analyzer.features)
                                        analyzer.features = analyzer.features[~analyzer.features['is_outlier']]
                                        st.info(f"🚫 Excluded {original_count - len(analyzer.features)} outlier(s) from results")
                            except Exception as e:
                                st.error(f"❌ Error detecting outliers: {e}")
                                logger.error(f"Outlier detection error for {group_name}: {e}")
                    
                    results_summary.append(group_results)
                    overall_progress.progress((idx + 1) / len(selected_groups))
                
                # Display summary
                status_text.empty()
                overall_progress.empty()
                
                st.markdown("---")
                st.subheader("✅ Batch Processing Complete!")
                
                # Create summary dataframe
                summary_df = pd.DataFrame(results_summary)
                
                col_summary1, col_summary2 = st.columns([2, 1])
                
                with col_summary1:
                    st.dataframe(summary_df, use_container_width=True)
                
                with col_summary2:
                    st.metric("Total Groups Processed", len(selected_groups))
                    st.metric("Total Curves", summary_df['n_curves'].sum())
                    st.metric("Successful Fits", summary_df['fit_success'].sum())
                    if 'r2_filtered' in summary_df.columns:
                        st.metric("Removed by R² Filter", summary_df['r2_filtered'].sum())
                    if run_outliers:
                        st.metric("Total Outliers", summary_df['outliers'].sum())
                
                st.success("🎉 All groups processed successfully! You can now view individual results, compare groups, or generate reports.")
                
                # Display Model Comparison Results (if compare_2_vs_3 mode was used)
                if fitting_mode == "compare_2_vs_3" and st.session_state.model_comparison_results:
                    st.markdown("---")
                    st.subheader("📊 2-Component vs 3-Component Model Comparison")
                    
                    comparison_tabs = st.tabs([f"📁 {group}" for group in st.session_state.model_comparison_results.keys()])
                    
                    for tab, group_name in zip(comparison_tabs, st.session_state.model_comparison_results.keys()):
                        with tab:
                            group_comparison = st.session_state.model_comparison_results[group_name]
                            double_df = group_comparison.get('double')
                            triple_df = group_comparison.get('triple')
                            
                            if double_df is not None and triple_df is not None and not double_df.empty and not triple_df.empty:
                                col_model1, col_model2 = st.columns(2)
                                
                                with col_model1:
                                    st.markdown("### 🔷 Double Exponential (2-Component)")
                                    st.metric("Curves Fitted", len(double_df))
                                    if 'r2' in double_df.columns:
                                        st.metric("Mean R²", f"{double_df['r2'].mean():.4f}")
                                        st.metric("R² Range", f"{double_df['r2'].min():.4f} - {double_df['r2'].max():.4f}")
                                    if 'mobile_fraction_total' in double_df.columns:
                                        st.metric("Mean Mobile Fraction", f"{double_df['mobile_fraction_total'].mean():.2%}")
                                    if 'aicc' in double_df.columns:
                                        st.metric("Mean AICc", f"{double_df['aicc'].mean():.2f}")
                                
                                with col_model2:
                                    st.markdown("### 🔶 Triple Exponential (3-Component)")
                                    st.metric("Curves Fitted", len(triple_df))
                                    if 'r2' in triple_df.columns:
                                        st.metric("Mean R²", f"{triple_df['r2'].mean():.4f}")
                                        st.metric("R² Range", f"{triple_df['r2'].min():.4f} - {triple_df['r2'].max():.4f}")
                                    if 'mobile_fraction_total' in triple_df.columns:
                                        st.metric("Mean Mobile Fraction", f"{triple_df['mobile_fraction_total'].mean():.2%}")
                                    if 'aicc' in triple_df.columns:
                                        st.metric("Mean AICc", f"{triple_df['aicc'].mean():.2f}")
                                
                                # Statistical Comparison
                                st.markdown("---")
                                st.markdown("### 📈 Statistical Comparison")
                                
                                # Compare AICc values per curve to determine better fit
                                if 'aicc' in double_df.columns and 'aicc' in triple_df.columns:
                                    # Match by curve identifier if available
                                    n_curves = min(len(double_df), len(triple_df))
                                    double_aicc = double_df['aicc'].values[:n_curves]
                                    triple_aicc = triple_df['aicc'].values[:n_curves]
                                    
                                    # Lower AICc is better
                                    double_better = np.sum(double_aicc < triple_aicc)
                                    triple_better = np.sum(triple_aicc < double_aicc)
                                    ties = n_curves - double_better - triple_better
                                    
                                    comp_col1, comp_col2, comp_col3 = st.columns(3)
                                    with comp_col1:
                                        st.metric("Double Exponential Better", f"{double_better} ({100*double_better/n_curves:.1f}%)")
                                    with comp_col2:
                                        st.metric("Triple Exponential Better", f"{triple_better} ({100*triple_better/n_curves:.1f}%)")
                                    with comp_col3:
                                        st.metric("Mean ΔAICc", f"{np.mean(double_aicc - triple_aicc):.2f}")
                                    
                                    # Interpretation
                                    if double_better > triple_better:
                                        st.info(f"🔷 **Double exponential fits better** for {double_better}/{n_curves} curves ({100*double_better/n_curves:.1f}%). The 3rd component may not be justified by the data.")
                                    elif triple_better > double_better:
                                        st.info(f"🔶 **Triple exponential fits better** for {triple_better}/{n_curves} curves ({100*triple_better/n_curves:.1f}%). The data supports 3 distinct recovery populations.")
                                    else:
                                        st.info("⚖️ **Models perform similarly**. Consider using the simpler double exponential model (parsimony principle).")
                                
                                # Create comparison plot
                                st.markdown("### 📊 Model Comparison Visualization")
                                
                                fig_compare = plt.figure(figsize=(14, 5))
                                
                                # R² comparison
                                ax1 = fig_compare.add_subplot(1, 3, 1)
                                if 'r2' in double_df.columns and 'r2' in triple_df.columns:
                                    positions = [1, 2]
                                    bp = ax1.boxplot([double_df['r2'].dropna(), triple_df['r2'].dropna()], 
                                                     positions=positions, widths=0.6, patch_artist=True)
                                    bp['boxes'][0].set_facecolor('steelblue')
                                    bp['boxes'][1].set_facecolor('darkorange')
                                    ax1.set_xticklabels(['Double\n(2-comp)', 'Triple\n(3-comp)'])
                                    ax1.set_ylabel('R²')
                                    ax1.set_title('Goodness of Fit (R²)')
                                    ax1.axhline(y=st.session_state.r2_threshold, color='red', linestyle='--', alpha=0.5, label=f'R² threshold ({st.session_state.r2_threshold})')
                                    ax1.legend(fontsize=8)
                                
                                # AICc comparison
                                ax2 = fig_compare.add_subplot(1, 3, 2)
                                if 'aicc' in double_df.columns and 'aicc' in triple_df.columns:
                                    n_curves = min(len(double_df), len(triple_df))
                                    delta_aicc = double_df['aicc'].values[:n_curves] - triple_df['aicc'].values[:n_curves]
                                    ax2.hist(delta_aicc, bins=20, color='purple', alpha=0.7, edgecolor='black')
                                    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Equal fit')
                                    ax2.axvline(x=2, color='green', linestyle=':', alpha=0.7, label='Substantial diff (±2)')
                                    ax2.axvline(x=-2, color='green', linestyle=':', alpha=0.7)
                                    ax2.set_xlabel('ΔAICc (Double - Triple)')
                                    ax2.set_ylabel('Count')
                                    ax2.set_title('AICc Difference Distribution\n(Negative = Triple better)')
                                    ax2.legend(fontsize=8)
                                
                                # Mobile fraction comparison
                                ax3 = fig_compare.add_subplot(1, 3, 3)
                                if 'mobile_fraction_total' in double_df.columns and 'mobile_fraction_total' in triple_df.columns:
                                    n_curves = min(len(double_df), len(triple_df))
                                    ax3.scatter(double_df['mobile_fraction_total'].values[:n_curves], 
                                               triple_df['mobile_fraction_total'].values[:n_curves],
                                               alpha=0.6, c='purple', edgecolors='black')
                                    max_val = max(double_df['mobile_fraction_total'].max(), triple_df['mobile_fraction_total'].max())
                                    ax3.plot([0, max_val], [0, max_val], 'r--', label='1:1 line')
                                    ax3.set_xlabel('Double Exp Mobile Fraction')
                                    ax3.set_ylabel('Triple Exp Mobile Fraction')
                                    ax3.set_title('Mobile Fraction Comparison')
                                    ax3.legend(fontsize=8)
                                
                                plt.tight_layout()
                                st.pyplot(fig_compare)
                                plt.close(fig_compare)
                                
                                # Detailed data tables
                                with st.expander("📋 View Detailed Comparison Data"):
                                    st.markdown("**Double Exponential Results:**")
                                    st.dataframe(double_df, use_container_width=True)
                                    
                                    st.markdown("**Triple Exponential Results:**")
                                    st.dataframe(triple_df, use_container_width=True)
                            else:
                                st.warning(f"Comparison data incomplete for {group_name}")

# --- Page 3: Model Fitting ---
elif page == "3. Model Fitting":
    st.header("📈 Model Fitting & Analysis")
    
    if not st.session_state.data_groups:
        st.warning("⚠️ Please import data first on the 'Import & Preprocess' page.")
    else:
        # Sidebar controls
        st.sidebar.markdown("---")
        st.sidebar.subheader("⚙️ Fitting Options")
        
        group_select = st.selectbox("📁 Select Group", list(st.session_state.data_groups.keys()))
        analyzer = st.session_state.data_groups[group_select]
        
        # Fitting mode selection
        fitting_mode = st.radio(
            "Fitting Strategy:",
            ["🚀 Fit All Models (Recommended)", "🎯 Single Model"],
            help="Fit All Models: Fits single, double, triple exponential and anomalous diffusion models, then selects best based on AICc. Single Model: Fit only one specific model."
        )
        
        if fitting_mode == "🎯 Single Model":
            model_select = st.selectbox(
                "Select Model",
                ["single", "double", "triple", "anomalous_diffusion", "reaction_diffusion"],
                format_func=lambda x: {
                    "single": "Single Exponential (Simple Diffusion)",
                    "double": "Double Exponential (Two Populations)",
                    "triple": "Triple Exponential (Three Populations)",
                    "anomalous_diffusion": "Anomalous Diffusion (Subdiffusive)",
                    "reaction_diffusion": "Reaction-Diffusion (Binding/Unbinding)"
                }[x]
            )
        
        # Model selection criterion
        criterion = st.selectbox(
            "Model Selection Criterion:",
            ["aicc", "aic", "bic", "adj_r2"],
            format_func=lambda x: {
                "aicc": "AICc (Corrected AIC - Best for small samples)",
                "aic": "AIC (Akaike Information Criterion)",
                "bic": "BIC (Bayesian Information Criterion)",
                "adj_r2": "Adjusted R² (Penalized R-squared)"
            }[x],
            help="AICc is recommended for most cases as it accounts for small sample sizes"
        )
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            fit_button = st.button("🔬 Fit Models", type="primary", use_container_width=True)
        with col_btn2:
            if st.button("🔄 Clear Results", use_container_width=True):
                analyzer.features = None
                st.rerun()
        
        if fit_button:
            with st.spinner("🔬 Fitting models to recovery curves..."):
                if fitting_mode == "🚀 Fit All Models (Recommended)":
                    # Fit all models for comparison
                    analyzer.analyze_group(model_name=None)  # None triggers all models
                    
                    # Apply R² filter
                    if analyzer.features is not None:
                        original_count = len(analyzer.features)
                        analyzer.features, removed_count = apply_r2_filter(
                            analyzer.features, 
                            st.session_state.r2_threshold
                        )
                        
                        st.success(f"✅ Successfully fitted all models to {len(analyzer.curves)} curves!")
                        if removed_count > 0:
                            st.info(f"🔍 R² filter: Removed {removed_count}/{original_count} fits with R² < {st.session_state.r2_threshold}")
                    else:
                        st.success(f"✅ Successfully fitted all models to {len(analyzer.curves)} curves!")
                else:
                    # Fit single selected model
                    analyzer.analyze_group(model_name=model_select)
                    
                    # Apply R² filter
                    if analyzer.features is not None:
                        original_count = len(analyzer.features)
                        analyzer.features, removed_count = apply_r2_filter(
                            analyzer.features, 
                            st.session_state.r2_threshold
                        )
                        
                        st.success(f"✅ Successfully fitted {model_select} model to {len(analyzer.curves)} curves!")
                        if removed_count > 0:
                            st.info(f"🔍 R² filter: Removed {removed_count}/{original_count} fits with R² < {st.session_state.r2_threshold}")
                    else:
                        st.success(f"✅ Successfully fitted {model_select} model to {len(analyzer.curves)} curves!")
                
                st.balloons()
        
        # Display results
        if analyzer.features is not None and not analyzer.features.empty:
            st.markdown("---")
            st.subheader("📊 Fit Results Summary")
            
            # Summary statistics
            col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
            with col_sum1:
                st.metric("📈 Curves Analyzed", len(analyzer.curves))
            with col_sum2:
                if 'model' in analyzer.features.columns:
                    best_models = analyzer.features['model'].value_counts()
                    st.metric("🏆 Most Common Model", best_models.index[0].title() if len(best_models) > 0 else "N/A")
            with col_sum3:
                if 'mobile_fraction' in analyzer.features.columns:
                    avg_mobile = analyzer.features['mobile_fraction'].mean()
                    st.metric("📊 Avg Mobile Fraction", f"{avg_mobile:.1f}%" if np.isfinite(avg_mobile) else "N/A")
            with col_sum4:
                if 'r2' in analyzer.features.columns:
                    avg_r2 = analyzer.features['r2'].mean()
                    st.metric("✨ Avg R²", f"{avg_r2:.3f}" if np.isfinite(avg_r2) else "N/A")
            
            # Model comparison table
            st.markdown("### 🔬 Model Comparison by Curve")
            
            # Create a display dataframe with selected columns
            display_cols = []
            for col in ['model', 'r2', 'adj_r2', 'aic', 'aicc', 'bic', 'mobile_fraction', 'half_time_fast', 'k_fast']:
                if col in analyzer.features.columns:
                    display_cols.append(col)
            
            if display_cols:
                display_df = analyzer.features[display_cols].copy()
                
                # Format columns for better display
                if 'model' in display_df.columns:
                    display_df['model'] = display_df['model'].apply(lambda x: x.title() if isinstance(x, str) else x)
                
                st.dataframe(
                    display_df.style.format({
                        'r2': '{:.4f}',
                        'adj_r2': '{:.4f}',
                        'aic': '{:.2f}',
                        'aicc': '{:.2f}',
                        'bic': '{:.2f}',
                        'mobile_fraction': '{:.1f}',
                        'half_time_fast': '{:.2f}',
                        'k_fast': '{:.4f}'
                    }),
                    use_container_width=True,
                    height=400
                )
                
                # Download button
                csv = display_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name=f"{group_select}_fit_results.csv",
                    mime="text/csv"
                )
            
            # Detailed statistical summary
            st.markdown("### 📈 Statistical Summary")
            st.dataframe(analyzer.features.describe(), use_container_width=True)
            
            # Visualizations
            st.markdown("### 📊 Fit Quality Visualizations")
            
            tab_r2, tab_models, tab_mobile, tab_kinetics = st.tabs([
                "R² Distribution", "Model Selection", "Mobile Fractions", "Kinetics"
            ])
            
            with tab_r2:
                if 'r2' in analyzer.features.columns:
                    fig_r2, ax_r2 = plt.subplots(figsize=(10, 5))
                    ax_r2.hist(analyzer.features['r2'].dropna(), bins=20, edgecolor='black', alpha=0.7)
                    ax_r2.axvline(analyzer.features['r2'].mean(), color='red', linestyle='--', 
                                 label=f'Mean: {analyzer.features["r2"].mean():.3f}')
                    ax_r2.set_xlabel('R² Value')
                    ax_r2.set_ylabel('Frequency')
                    ax_r2.set_title('Distribution of R² Values (Goodness of Fit)')
                    ax_r2.legend()
                    ax_r2.grid(alpha=0.3)
                    st.pyplot(fig_r2)
                    
                    st.caption(f"**Interpretation:** Higher R² values (closer to 1.0) indicate better fits. "
                             f"Average R² = {analyzer.features['r2'].mean():.3f}")
            
            with tab_models:
                if 'model' in analyzer.features.columns:
                    model_counts = analyzer.features['model'].value_counts()
                    fig_models, ax_models = plt.subplots(figsize=(10, 5))
                    model_counts.plot(kind='bar', ax=ax_models, color='skyblue', edgecolor='black')
                    ax_models.set_xlabel('Model Type')
                    ax_models.set_ylabel('Count')
                    ax_models.set_title(f'Best-Fit Model Distribution (by {criterion.upper()})')
                    ax_models.set_xticklabels([m.title() for m in model_counts.index], rotation=45)
                    ax_models.grid(axis='y', alpha=0.3)
                    st.pyplot(fig_models)
                    
                    st.caption("**Interpretation:** Shows which models best describe your data. "
                             "Multiple populations (double/triple) suggest heterogeneous protein behavior.")
            
            with tab_mobile:
                if 'mobile_fraction' in analyzer.features.columns:
                    fig_mob, ax_mob = plt.subplots(figsize=(10, 5))
                    mobile_data = analyzer.features['mobile_fraction'].dropna()
                    ax_mob.hist(mobile_data, bins=20, edgecolor='black', alpha=0.7, color='green')
                    ax_mob.axvline(mobile_data.mean(), color='red', linestyle='--',
                                  label=f'Mean: {mobile_data.mean():.1f}%')
                    ax_mob.set_xlabel('Mobile Fraction (%)')
                    ax_mob.set_ylabel('Frequency')
                    ax_mob.set_title('Distribution of Mobile Fractions')
                    ax_mob.legend()
                    ax_mob.grid(alpha=0.3)
                    st.pyplot(fig_mob)
                    
                    st.caption("**Interpretation:** Mobile fraction indicates the percentage of protein that recovers. "
                             f"Average = {mobile_data.mean():.1f}%. Values <100% suggest immobile binding.")
            
            with tab_kinetics:
                if 'k_fast' in analyzer.features.columns and 'half_time_fast' in analyzer.features.columns:
                    fig_kin, (ax_k, ax_t) = plt.subplots(1, 2, figsize=(14, 5))
                    
                    # Rate constant distribution
                    k_data = analyzer.features['k_fast'].dropna()
                    ax_k.hist(k_data, bins=20, edgecolor='black', alpha=0.7, color='orange')
                    ax_k.axvline(k_data.mean(), color='red', linestyle='--',
                               label=f'Mean: {k_data.mean():.4f} s⁻¹')
                    ax_k.set_xlabel('Rate Constant k (s⁻¹)')
                    ax_k.set_ylabel('Frequency')
                    ax_k.set_title('Distribution of Recovery Rate Constants')
                    ax_k.legend()
                    ax_k.grid(alpha=0.3)
                    
                    # Half-time distribution
                    t_data = analyzer.features['half_time_fast'].dropna()
                    ax_t.hist(t_data, bins=20, edgecolor='black', alpha=0.7, color='purple')
                    ax_t.axvline(t_data.mean(), color='red', linestyle='--',
                               label=f'Mean: {t_data.mean():.2f} s')
                    ax_t.set_xlabel('Half-time (s)')
                    ax_t.set_ylabel('Frequency')
                    ax_t.set_title('Distribution of Recovery Half-times')
                    ax_t.legend()
                    ax_t.grid(alpha=0.3)
                    
                    st.pyplot(fig_kin)
                    
                    st.caption("**Interpretation:** Faster rate constants (higher k) and shorter half-times indicate "
                             "rapid diffusion. Slower kinetics suggest larger complexes or binding interactions.")
            
            # Biological interpretation guide
            with st.expander("🧬 Guide to Biological Interpretation"):
                st.markdown("""
                ### Understanding Your FRAP Results
                
                #### Model Selection:
                - **Single Exponential**: Simple diffusion of a homogeneous population
                - **Double Exponential**: Two populations with different mobilities (e.g., free + bound)
                - **Triple Exponential**: Three distinct populations (e.g., free + weakly bound + strongly bound)
                - **Anomalous Diffusion**: Subdiffusive behavior in crowded/constrained environments
                - **Reaction-Diffusion**: Combined diffusion and binding kinetics (e.g., nuclear proteins with transient DNA binding)
                
                #### Key Parameters:
                - **Mobile Fraction**: % of protein that recovers
                  - 100%: Freely diffusing, no stable binding
                  - 50-90%: Partial binding, dynamic exchange
                  - <50%: Significant immobile population
                
                - **Rate Constant (k)**: Speed of recovery
                  - High k: Fast diffusion (small molecules, free protein)
                  - Low k: Slow diffusion (large complexes, binding)
                
                - **Half-time**: Time to reach 50% recovery
                  - Faster = more mobile protein
                  - Use to estimate diffusion coefficient or binding kinetics
                
                #### Diffusion vs. Binding:
                - Compare calculated molecular weight from diffusion to known protein MW
                - If calculated MW >> actual MW: Binding is slowing recovery
                - If calculated MW ≈ actual MW: Pure diffusion dominates
                - Multiple exponentials: Heterogeneous populations with different binding states
                
                #### Next Steps:
                1. Check R² values (should be >0.95 for good fits)
                2. Compare model selection across replicates (should be consistent)
                3. Use "Compare Groups" tab to test statistical differences
                4. Generate report with detailed biophysical parameters
                """)
        else:
            st.info("👆 Click 'Fit Models' to analyze your data and see comprehensive fit statistics.")

# --- Page 4: Subpopulations ---
elif page == "4. Subpopulations":
    st.header("🔍 Subpopulation Analysis")
    
    if not st.session_state.data_groups:
        st.warning("Please import data first.")
    else:
        group_select = st.selectbox("Select Group", list(st.session_state.data_groups.keys()))
        analyzer = st.session_state.data_groups[group_select]
        
        if analyzer.features is None or analyzer.features.empty:
            st.warning("⚠️ Please run model fitting first on the 'Model Fitting' page.")
        else:
            # Display current data info
            st.info(f"📊 **{len(analyzer.curves)} curves** loaded with **{len(analyzer.features)} fitted results**")
            
            # Show current features
            with st.expander("📋 View Fitted Parameters", expanded=False):
                st.dataframe(analyzer.features, use_container_width=True)
            
            col1, col2 = st.columns([1, 3])
            with col1:
                st.subheader("Clustering")
                
                # Get numerical columns only
                numerical_cols = analyzer.features.select_dtypes(include=[np.number]).columns.tolist()
                if len(numerical_cols) < 2:
                    st.warning("Need at least 2 numerical parameters for clustering")
                else:
                    max_k = st.slider("Max Components", 2, min(5, len(analyzer.features)), 3)
                    if st.button("🔍 Detect Subpopulations"):
                        with st.spinner("Running clustering analysis..."):
                            analyzer.detect_subpopulations(range(1, max_k + 1))
                        st.success("✅ Clustering complete!")
                        st.rerun()
                    
                    if st.button("🎯 Detect Outliers"):
                        with st.spinner("Running outlier detection..."):
                            analyzer.detect_outliers()
                        st.success("✅ Outlier detection complete!")
                        st.rerun()
            
            with col2:
                if 'subpopulation' in analyzer.features.columns:
                    st.subheader("🎨 Cluster Visualization")
                    
                    # Check how many curves have cluster assignments
                    clustered_data = analyzer.features.dropna(subset=['subpopulation'])
                    n_clustered = len(clustered_data)
                    n_total = len(analyzer.features)
                    
                    if n_clustered == 0:
                        st.warning("⚠️ No curves were successfully clustered. This may occur if all curves have missing values.")
                    else:
                        st.info(f"✅ Successfully clustered **{n_clustered}/{n_total}** curves")
                        
                        # Dynamic parameter selection - only numerical
                        numerical_params = [c for c in analyzer.features.select_dtypes(include=[np.number]).columns 
                                           if c not in ['subpopulation', 'is_outlier']]
                        
                        if len(numerical_params) >= 2:
                            x_axis = st.selectbox("X Axis", numerical_params, index=0)
                            y_axis = st.selectbox("Y Axis", numerical_params, index=min(1, len(numerical_params)-1))
                            
                            try:
                                fig = FRAPVisualizer.plot_subpopulations(clustered_data, x_axis, y_axis)
                                st.pyplot(fig)
                            except Exception as e:
                                st.error(f"Error creating visualization: {e}")
                            
                            st.write("**Subpopulation Counts:**")
                            counts = clustered_data['subpopulation'].value_counts().sort_index()
                            
                            # Display in columns for better layout
                            cols = st.columns(min(len(counts), 4))
                            for idx, (pop, count) in enumerate(counts.items()):
                                with cols[idx % len(cols)]:
                                    st.metric(f"Cluster {int(pop)}", f"{count} curves", 
                                            delta=f"{count/n_clustered*100:.1f}%")
                        else:
                            st.warning("Need at least 2 numerical parameters for visualization")
                else:
                    st.info("👆 Click 'Detect Subpopulations' to cluster your data")

# --- Page 5: Compare Groups ---
elif page == "5. Compare Groups":
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
                # Filter to only numerical columns
                numerical_cols = analyzer1.features.select_dtypes(include=[np.number]).columns.tolist()
                if not numerical_cols:
                    st.error("No numerical parameters available for comparison.")
                else:
                    param = st.selectbox("Parameter to Compare", numerical_cols)
                    
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

# --- Page 6: Global Fitting ---
elif page == "6. Global Fitting":
    st.header("🔬 Global Fitting & Unified Model Selection")
    
    if not GLOBAL_FITTING_AVAILABLE:
        st.error("⚠️ Global fitting requires lmfit. Install with: `pip install lmfit`")
        st.stop()
    
    st.markdown("""
    **Global fitting** simultaneously fits all curves in a group while sharing kinetic parameters.
    This provides more robust parameter estimates and enables rigorous statistical model comparisons.
    
    ### Workflow:
    1. **Phase 1**: Fit all candidate models to each group
    2. **Phase 2**: Statistically determine the unified model (most complex model required by any group)
    3. **Phase 3**: Refit all groups with the unified model for "apples-to-apples" comparison
    """)
    
    if len(st.session_state.data_groups) < 1:
        st.warning("⚠️ Please import and process data first (Pages 1-2)")
        st.stop()
    
    # Configuration
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔧 Global Fitting Options")
    
    mf_is_local = st.sidebar.checkbox(
        "Local Mobile Fraction (Mf)",
        value=True,
        help="If checked, Mf is fitted independently for each curve. "
             "If unchecked, Mf is shared across all curves in a group."
    )
    
    alpha_level = st.sidebar.slider(
        "Significance Level (α)",
        min_value=0.01,
        max_value=0.10,
        value=0.05,
        step=0.01,
        help="P-value threshold for F-test model comparison"
    )
    
    normalize_data = st.sidebar.checkbox(
        "Apply Double Normalization",
        value=True,
        help="Normalize curves so pre-bleach=1, post-bleach start=0"
    )
    
    # Group selection
    st.subheader("📁 Select Groups for Global Analysis")
    
    available_groups = list(st.session_state.data_groups.keys())
    selected_groups = st.multiselect(
        "Groups to include:",
        available_groups,
        default=available_groups,
        help="Select at least 2 groups for comparative analysis"
    )
    
    if len(selected_groups) < 2:
        st.warning("⚠️ Select at least 2 groups for unified model selection")
    
    # Convert data
    if selected_groups:
        st.subheader("📊 Data Summary")
        
        summary_data = []
        for group_name in selected_groups:
            analyzer = st.session_state.data_groups[group_name]
            n_curves = len(analyzer.curves) if hasattr(analyzer, 'curves') else 0
            summary_data.append({
                'Group': group_name,
                'Curves': n_curves,
                'Status': '✅' if n_curves > 0 else '❌'
            })
        
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
    
    # Run analysis button
    st.markdown("---")
    
    if st.button("🚀 Run Global Fitting Analysis", type="primary", disabled=len(selected_groups) < 2):
        
        with st.spinner("Converting data..."):
            # Convert analyzer data to DataFrame format
            try:
                df = convert_analyzer_to_dataframe(
                    st.session_state.data_groups,
                    group_names=selected_groups
                )
                
                if df.empty:
                    st.error("❌ Could not extract curve data. Ensure groups have been processed.")
                    st.stop()
                
                st.success(f"✅ Converted {len(df)} data points from {df['GroupID'].nunique()} groups")
                
            except Exception as e:
                st.error(f"❌ Data conversion failed: {e}")
                st.stop()
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(current, total, message):
            progress_bar.progress(current / total if total > 0 else 0)
            status_text.text(message)
        
        # Run the workflow
        try:
            workflow = UnifiedModelWorkflow(mf_is_local=mf_is_local, alpha=alpha_level)
            
            result = workflow.run_full_analysis(
                df,
                time_col='Time',
                intensity_col='Intensity', 
                curve_col='CurveID',
                group_col='GroupID',
                normalize=normalize_data,
                progress_callback=update_progress
            )
            
            progress_bar.progress(1.0)
            status_text.text("✅ Analysis complete!")
            
            # Store results in session state
            st.session_state.global_fit_result = result
            
        except Exception as e:
            st.error(f"❌ Global fitting failed: {e}")
            import traceback
            st.code(traceback.format_exc())
            st.stop()
        
        # Display results
        st.markdown("---")
        st.header("📈 Results")
        
        # Phase 1 Results
        st.subheader("Phase 1: Model Exploration")
        
        if result.model_selection_table is not None:
            st.dataframe(
                result.model_selection_table.style.format({
                    'Chi-squared': '{:.4f}',
                    'Reduced Chi-sq': '{:.4f}',
                    'AIC': '{:.2f}',
                    'BIC': '{:.2f}'
                }),
                use_container_width=True
            )
        
        # Phase 2 Results
        st.subheader("Phase 2: Unified Model Selection")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.metric("Unified Model", result.unified_model.upper())
            
            st.markdown("**Per-group preferences:**")
            for group, model in result.group_preferred_models.items():
                emoji = "✅" if model == result.unified_model else "⬆️"
                st.markdown(f"- {group}: {model} {emoji}")
        
        with col2:
            with st.expander("📋 Selection Rationale", expanded=False):
                st.text(result.unification_rationale)
        
        # Phase 3 Results
        st.subheader("Phase 3: Final Comparative Results")
        
        if result.comparison_table is not None:
            st.dataframe(
                result.comparison_table.style.format(
                    {col: '{:.4f}' for col in result.comparison_table.columns 
                     if result.comparison_table[col].dtype in [np.float64, np.float32]}
                ),
                use_container_width=True
            )
        
        # Visualizations
        st.subheader("📊 Visualizations")
        
        tab1, tab2, tab3 = st.tabs(["Fitted Curves", "Residuals", "Parameter Comparison"])
        
        with tab1:
            # Prepare grouped data for plotting
            grouped_data = workflow.prepare_data(df, 'Time', 'Intensity', 'CurveID', 'GroupID')
            if normalize_data:
                grouped_data = {gid: workflow.normalize_curves(curves) 
                              for gid, curves in grouped_data.items()}
            
            fig_fits = GlobalFitReporter.plot_group_fits(grouped_data, result.final_fits)
            st.pyplot(fig_fits)
        
        with tab2:
            fig_residuals = GlobalFitReporter.plot_residuals(grouped_data, result.final_fits)
            st.pyplot(fig_residuals)
        
        with tab3:
            # Get available global parameters
            if result.final_fits:
                first_fit = list(result.final_fits.values())[0]
                global_params = [p for p, d in first_fit.parameters.items() 
                                if d.get('is_global') and d.get('vary', True) and '_global' in p]
                
                if global_params:
                    selected_param = st.selectbox("Select parameter to compare:", global_params)
                    fig_param = GlobalFitReporter.plot_parameter_comparison(result, selected_param)
                    st.pyplot(fig_param)
                else:
                    st.info("No global parameters available for comparison")
        
        # Download report
        st.markdown("---")
        st.subheader("📥 Download Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Text report
            report_text = GlobalFitReporter.generate_model_selection_report(result)
            st.download_button(
                "📄 Download Text Report",
                data=report_text,
                file_name="global_fitting_report.txt",
                mime="text/plain"
            )
        
        with col2:
            # CSV comparison table
            if result.comparison_table is not None:
                csv_data = result.comparison_table.to_csv(index=False)
                st.download_button(
                    "📊 Download Comparison Table (CSV)",
                    data=csv_data,
                    file_name="global_fitting_comparison.csv",
                    mime="text/csv"
                )

# --- Page 7: Report ---
elif page == "7. Report":
    st.header("📄 Report Generation")
    
    if not st.session_state.data_groups:
        st.warning("⚠️ No data available. Please import and process data first.")
    else:
        st.info(f"📊 **{len(st.session_state.data_groups)} groups** available for reporting")
        
        # Report configuration
        st.subheader("⚙️ Report Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            report_format = st.radio(
                "Report Format:",
                ["html", "pdf"],
                format_func=lambda x: "📄 HTML (Interactive)" if x == "html" else "📑 PDF (Printable)"
            )
            
            include_plots = st.checkbox("Include Recovery Plots", value=True)
            include_distributions = st.checkbox("Include Parameter Distributions", value=True)
            include_subpopulations = st.checkbox("Include Subpopulation Analysis", value=True)
        
        with col2:
            # Group selection
            select_all_groups = st.checkbox("Include All Groups", value=True)
            
            if not select_all_groups:
                selected_report_groups = st.multiselect(
                    "Select Groups:",
                    list(st.session_state.data_groups.keys()),
                    default=list(st.session_state.data_groups.keys())
                )
            else:
                selected_report_groups = list(st.session_state.data_groups.keys())
        
        st.markdown("---")
        
        # Preview section
        with st.expander("📊 Data Preview", expanded=False):
            for name in selected_report_groups:
                analyzer = st.session_state.data_groups[name]
                st.markdown(f"### {name}")
                
                if analyzer.features is not None and not analyzer.features.empty:
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Curves", len(analyzer.curves))
                    with col_b:
                        st.metric("Successful Fits", len(analyzer.features.dropna(subset=['r2'])))
                    with col_c:
                        if 'subpopulation' in analyzer.features.columns:
                            st.metric("Subpopulations", analyzer.features['subpopulation'].nunique())
                        else:
                            st.metric("Subpopulations", "N/A")
                    
                    st.dataframe(analyzer.features.head(), use_container_width=True)
                else:
                    st.warning(f"⚠️ No fitted data available for {name}. Please run fitting first.")
        
        # Generate report button
        if st.button("🎨 Generate Report", type="primary", use_container_width=True):
            if not selected_report_groups:
                st.error("Please select at least one group for the report")
            else:
                with st.spinner("📝 Generating report..."):
                    try:
                        # Collect all data
                        figures = {}
                        all_features = pd.DataFrame()
                        
                        for name in selected_report_groups:
                            analyzer = st.session_state.data_groups[name]
                            
                            if analyzer.features is not None and not analyzer.features.empty:
                                df = analyzer.features.copy()
                                df['Group'] = name
                                all_features = pd.concat([all_features, df], ignore_index=True)
                                
                                # Generate recovery plot
                                if include_plots and analyzer.curves:
                                    try:
                                        # Get time data from first curve
                                        if analyzer.curves[0].time_post_bleach is not None:
                                            times = analyzer.curves[0].time_post_bleach
                                        else:
                                            times = analyzer.curves[0].time
                                        
                                        # Collect data intensities
                                        data_intensities = []
                                        for c in analyzer.curves[:10]:  # Limit to first 10 for clarity
                                            if c.intensity_post_bleach is not None:
                                                data_intensities.append(c.intensity_post_bleach)
                                            elif c.normalized_intensity is not None:
                                                data_intensities.append(c.normalized_intensity)
                                        
                                        # Collect fitted curves
                                        fitted_curves = []
                                        for res in analyzer.fit_results[:10]:
                                            if res.success and res.fitted_curve is not None:
                                                fitted_curves.append(res.fitted_curve)
                                        
                                        if data_intensities:
                                            fig = FRAPVisualizer.plot_recovery_curves(
                                                times, 
                                                data_intensities, 
                                                fitted_curves, 
                                                title=f"{name} Recovery Curves"
                                            )
                                            figures[f"{name}_Recovery"] = fig
                                    except Exception as e:
                                        logger.warning(f"Could not generate recovery plot for {name}: {e}")
                                
                                # Generate parameter distributions
                                if include_distributions:
                                    try:
                                        numerical_params = df.select_dtypes(include=[np.number]).columns
                                        for param in ['r2', 'mobile_fraction', 'k_fast']:
                                            if param in numerical_params:
                                                fig = FRAPVisualizer.plot_parameter_distribution(
                                                    df, 
                                                    param, 
                                                    group_col='Group'
                                                )
                                                figures[f"{name}_{param}_dist"] = fig
                                    except Exception as e:
                                        logger.warning(f"Could not generate distribution plots for {name}: {e}")
                                
                                # Generate subpopulation plots
                                if include_subpopulations and 'subpopulation' in df.columns:
                                    try:
                                        numerical_cols = [c for c in df.select_dtypes(include=[np.number]).columns 
                                                         if c not in ['subpopulation', 'is_outlier']]
                                        if len(numerical_cols) >= 2:
                                            fig = FRAPVisualizer.plot_subpopulations(
                                                df, 
                                                numerical_cols[0], 
                                                numerical_cols[min(1, len(numerical_cols)-1)]
                                            )
                                            figures[f"{name}_subpopulations"] = fig
                                    except Exception as e:
                                        logger.warning(f"Could not generate subpopulation plot for {name}: {e}")
                        
                        if all_features.empty:
                            st.error("❌ No fitted data available. Please run model fitting on at least one group.")
                        else:
                            # Generate report
                            report_filename = f"FRAP_Report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.{report_format}"
                            
                            if report_format == "html":
                                reporter = EnhancedFRAPReportGenerator(output_dir=".")
                                reporter.generate_report(
                                    data_groups=st.session_state.data_groups,
                                    selected_groups=selected_report_groups,
                                    filename=report_filename
                                )
                            else:
                                st.warning("⚠️ PDF reports are not supported in the enhanced analytical engine yet. Please select HTML.")

                            
                            st.success(f"✅ Report generated successfully: {report_filename}")
                            
                            # Download button
                            if os.path.exists(report_filename):
                                with open(report_filename, "rb") as f:
                                    mime_type = "text/html" if report_format == "html" else "application/pdf"
                                    st.download_button(
                                        "📥 Download Report", 
                                        f, 
                                        file_name=report_filename,
                                        mime=mime_type,
                                        use_container_width=True
                                    )
                            
                            # Summary statistics
                            st.markdown("---")
                            st.subheader("📊 Report Summary")
                            
                            col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
                            with col_sum1:
                                st.metric("Groups", len(selected_report_groups))
                            with col_sum2:
                                st.metric("Total Curves", len(all_features))
                            with col_sum3:
                                if 'model' in all_features.columns:
                                    st.metric("Most Common Model", all_features['model'].mode()[0] if not all_features['model'].mode().empty else "N/A")
                            with col_sum4:
                                if 'r2' in all_features.columns:
                                    st.metric("Avg R²", f"{all_features['r2'].mean():.3f}")
                    
                    except Exception as e:
                        st.error(f"❌ Error generating report: {e}")
                        logger.error(f"Report generation error: {e}")
                        import traceback
                        st.code(traceback.format_exc())
