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
                ["comprehensive", "single_model"],
                format_func=lambda x: {
                    "comprehensive": "🧬 Comprehensive Analysis (All Models - Recommended)",
                    "single_model": "🎯 Single Model Only"
                }[x],
                help="Comprehensive: Fits ALL models to each curve, determines best fit, then compares groups using each model. Single: Fit only one specific model."
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
                
                # Initialize comprehensive results storage
                if 'all_model_results' not in st.session_state:
                    st.session_state.all_model_results = {}
                
                from frap_core import FRAPAnalysisCore
                from scipy import stats as scipy_stats
                
                # Define all models to fit
                all_models = ['single', 'double', 'triple', 'reaction_diffusion']
                
                results_summary = []
                
                for idx, group_name in enumerate(selected_groups):
                    analyzer = st.session_state.data_groups[group_name]
                    
                    status_text.markdown(f"### Processing: **{group_name}** ({idx+1}/{len(selected_groups)})")
                    
                    group_results = {
                        'group': group_name,
                        'n_curves': len(analyzer.curves),
                        'fit_success': 0,
                        'best_model_counts': {},
                    }
                    
                    # Initialize storage for this group's model results
                    st.session_state.all_model_results[group_name] = {}
                    
                    # Step 1: Fit ALL models to each curve
                    if fitting_mode == "comprehensive":
                        with st.spinner(f"🔬 Fitting all models for {group_name}..."):
                            try:
                                # Storage for each model's results
                                model_results = {model: [] for model in all_models}
                                best_model_per_curve = []
                                
                                for curve_idx, curve in enumerate(analyzer.curves):
                                    if curve.normalized_intensity is None:
                                        for model in all_models:
                                            model_results[model].append({'curve_idx': curve_idx, 'success': False})
                                        continue
                                    
                                    time_data = curve.time
                                    intensity_data = curve.normalized_intensity
                                    
                                    curve_fits = {}
                                    
                                    # Fit Single Exponential
                                    try:
                                        single_fit = FRAPAnalysisCore.fit_single_exponential(time_data, intensity_data)
                                        if single_fit and single_fit.get('success', False):
                                            params = single_fit.get('params', [])
                                            A, k, C = params[0], params[1], params[2] if len(params) >= 3 else (np.nan, np.nan, np.nan)
                                            result = {
                                                'curve_idx': curve_idx, 'success': True, 'model': 'single',
                                                'r2': single_fit.get('r2', np.nan),
                                                'adj_r2': single_fit.get('adj_r2', np.nan),
                                                'aic': single_fit.get('aic', np.nan),
                                                'aicc': single_fit.get('aicc', np.nan),
                                                'bic': single_fit.get('bic', np.nan),
                                                'mobile_fraction': A * 100,
                                                'A1': A, 'k1': k, 'C': C,
                                                't_half': np.log(2) / k if k > 0 else np.nan,
                                            }
                                            model_results['single'].append(result)
                                            curve_fits['single'] = result
                                        else:
                                            model_results['single'].append({'curve_idx': curve_idx, 'success': False})
                                    except Exception as e:
                                        model_results['single'].append({'curve_idx': curve_idx, 'success': False, 'error': str(e)})
                                    
                                    # Fit Double Exponential
                                    try:
                                        double_fit = FRAPAnalysisCore.fit_double_exponential(time_data, intensity_data)
                                        if double_fit and double_fit.get('success', False):
                                            params = double_fit.get('params', [])
                                            if len(params) >= 5:
                                                A1, k1, A2, k2, C = params[:5]
                                                result = {
                                                    'curve_idx': curve_idx, 'success': True, 'model': 'double',
                                                    'r2': double_fit.get('r2', np.nan),
                                                    'adj_r2': double_fit.get('adj_r2', np.nan),
                                                    'aic': double_fit.get('aic', np.nan),
                                                    'aicc': double_fit.get('aicc', np.nan),
                                                    'bic': double_fit.get('bic', np.nan),
                                                    'mobile_fraction': (A1 + A2) * 100,
                                                    'A1': A1, 'k1': k1, 'A2': A2, 'k2': k2, 'C': C,
                                                    'fraction_fast': A1 / (A1 + A2) * 100 if (A1 + A2) > 0 else np.nan,
                                                    'fraction_slow': A2 / (A1 + A2) * 100 if (A1 + A2) > 0 else np.nan,
                                                    't_half_fast': np.log(2) / k1 if k1 > 0 else np.nan,
                                                    't_half_slow': np.log(2) / k2 if k2 > 0 else np.nan,
                                                }
                                                model_results['double'].append(result)
                                                curve_fits['double'] = result
                                        else:
                                            model_results['double'].append({'curve_idx': curve_idx, 'success': False})
                                    except Exception as e:
                                        model_results['double'].append({'curve_idx': curve_idx, 'success': False, 'error': str(e)})
                                    
                                    # Fit Triple Exponential
                                    try:
                                        triple_fit = FRAPAnalysisCore.fit_triple_exponential(time_data, intensity_data)
                                        if triple_fit and triple_fit.get('success', False):
                                            params = triple_fit.get('params', [])
                                            if len(params) >= 7:
                                                A1, k1, A2, k2, A3, k3, C = params[:7]
                                                total_A = A1 + A2 + A3
                                                result = {
                                                    'curve_idx': curve_idx, 'success': True, 'model': 'triple',
                                                    'r2': triple_fit.get('r2', np.nan),
                                                    'adj_r2': triple_fit.get('adj_r2', np.nan),
                                                    'aic': triple_fit.get('aic', np.nan),
                                                    'aicc': triple_fit.get('aicc', np.nan),
                                                    'bic': triple_fit.get('bic', np.nan),
                                                    'mobile_fraction': total_A * 100,
                                                    'A1': A1, 'k1': k1, 'A2': A2, 'k2': k2, 'A3': A3, 'k3': k3, 'C': C,
                                                    'fraction_fast': A1 / total_A * 100 if total_A > 0 else np.nan,
                                                    'fraction_medium': A2 / total_A * 100 if total_A > 0 else np.nan,
                                                    'fraction_slow': A3 / total_A * 100 if total_A > 0 else np.nan,
                                                    't_half_fast': np.log(2) / k1 if k1 > 0 else np.nan,
                                                    't_half_medium': np.log(2) / k2 if k2 > 0 else np.nan,
                                                    't_half_slow': np.log(2) / k3 if k3 > 0 else np.nan,
                                                }
                                                model_results['triple'].append(result)
                                                curve_fits['triple'] = result
                                        else:
                                            model_results['triple'].append({'curve_idx': curve_idx, 'success': False})
                                    except Exception as e:
                                        model_results['triple'].append({'curve_idx': curve_idx, 'success': False, 'error': str(e)})
                                    
                                    # Fit Reaction-Diffusion
                                    try:
                                        rxn_fit = FRAPAnalysisCore.fit_reaction_diffusion(time_data, intensity_data)
                                        if rxn_fit and rxn_fit.get('success', False):
                                            params = rxn_fit.get('params', [])
                                            if len(params) >= 5:
                                                A_diff, k_diff, A_bind, k_bind, C = params[:5]
                                                total_A = A_diff + A_bind
                                                result = {
                                                    'curve_idx': curve_idx, 'success': True, 'model': 'reaction_diffusion',
                                                    'r2': rxn_fit.get('r2', np.nan),
                                                    'adj_r2': rxn_fit.get('adj_r2', np.nan),
                                                    'aic': rxn_fit.get('aic', np.nan),
                                                    'aicc': rxn_fit.get('aicc', np.nan),
                                                    'bic': rxn_fit.get('bic', np.nan),
                                                    'mobile_fraction': total_A * 100,
                                                    'A_diff': A_diff, 'k_diff': k_diff, 'A_bind': A_bind, 'k_bind': k_bind, 'C': C,
                                                    'fraction_diffusion': A_diff / total_A * 100 if total_A > 0 else np.nan,
                                                    'fraction_binding': A_bind / total_A * 100 if total_A > 0 else np.nan,
                                                    't_half_diff': np.log(2) / k_diff if k_diff > 0 else np.nan,
                                                    't_half_bind': np.log(2) / k_bind if k_bind > 0 else np.nan,
                                                }
                                                model_results['reaction_diffusion'].append(result)
                                                curve_fits['reaction_diffusion'] = result
                                        else:
                                            model_results['reaction_diffusion'].append({'curve_idx': curve_idx, 'success': False})
                                    except Exception as e:
                                        model_results['reaction_diffusion'].append({'curve_idx': curve_idx, 'success': False, 'error': str(e)})
                                    
                                    # Determine best model for this curve based on criterion
                                    if curve_fits:
                                        if criterion == 'adj_r2':
                                            best_model = max(curve_fits.keys(), key=lambda m: curve_fits[m].get('adj_r2', -np.inf))
                                        else:  # aicc, aic, bic - lower is better
                                            best_model = min(curve_fits.keys(), key=lambda m: curve_fits[m].get(criterion, np.inf))
                                        best_model_per_curve.append(best_model)
                                    else:
                                        best_model_per_curve.append(None)
                                
                                # Store results for each model
                                for model in all_models:
                                    df = pd.DataFrame([r for r in model_results[model] if r.get('success', False)])
                                    if not df.empty:
                                        # Apply R² filter
                                        df, _ = apply_r2_filter(df, st.session_state.r2_threshold)
                                    st.session_state.all_model_results[group_name][model] = df
                                
                                # Count best models
                                best_counts = pd.Series([m for m in best_model_per_curve if m is not None]).value_counts().to_dict()
                                group_results['best_model_counts'] = best_counts
                                group_results['fit_success'] = sum(len(st.session_state.all_model_results[group_name][m]) for m in all_models)
                                
                                # Use reaction_diffusion as default for analyzer.features
                                if not st.session_state.all_model_results[group_name]['reaction_diffusion'].empty:
                                    analyzer.features = st.session_state.all_model_results[group_name]['reaction_diffusion'].copy()
                                elif not st.session_state.all_model_results[group_name]['double'].empty:
                                    analyzer.features = st.session_state.all_model_results[group_name]['double'].copy()
                                
                                st.success(f"✅ All models fitted. Best model distribution: {best_counts}")
                                
                            except Exception as e:
                                st.error(f"❌ Error fitting models: {e}")
                                logger.error(f"Batch processing error for {group_name}: {e}")
                    else:
                        # Single model mode
                        with st.spinner(f"🔬 Fitting {model_select} model for {group_name}..."):
                            analyzer.analyze_group(model_name=model_select)
                            if analyzer.features is not None:
                                group_results['fit_success'] = len(analyzer.features.dropna(subset=['r2']))
                                analyzer.features, removed = apply_r2_filter(analyzer.features, st.session_state.r2_threshold)
                            st.success(f"✅ {model_select} model fitted: {group_results['fit_success']} successful fits")
                    
                    # Subpopulation and outlier detection (optional)
                    if run_subpopulations and analyzer.features is not None and not analyzer.features.empty:
                        with st.spinner(f"🔍 Detecting subpopulations for {group_name}..."):
                            try:
                                analyzer.detect_subpopulations(range(1, max_k + 1))
                                if 'subpopulation' in analyzer.features.columns:
                                    group_results['subpopulations'] = analyzer.features['subpopulation'].nunique()
                            except Exception as e:
                                logger.error(f"Subpopulation detection error: {e}")
                    
                    if run_outliers and analyzer.features is not None and not analyzer.features.empty:
                        with st.spinner(f"🎯 Detecting outliers for {group_name}..."):
                            try:
                                analyzer.detect_outliers()
                                if 'is_outlier' in analyzer.features.columns:
                                    group_results['outliers'] = analyzer.features['is_outlier'].sum()
                                    if exclude_outliers:
                                        analyzer.features = analyzer.features[~analyzer.features['is_outlier']]
                            except Exception as e:
                                logger.error(f"Outlier detection error: {e}")
                    
                    results_summary.append(group_results)
                    overall_progress.progress((idx + 1) / len(selected_groups))
                
                # Display summary
                status_text.empty()
                overall_progress.empty()
                
                st.markdown("---")
                st.subheader("✅ Batch Processing Complete!")
                
                summary_df = pd.DataFrame(results_summary)
                
                col_summary1, col_summary2 = st.columns([2, 1])
                with col_summary1:
                    st.dataframe(summary_df, use_container_width=True)
                with col_summary2:
                    st.metric("Total Groups", len(selected_groups))
                    st.metric("Total Curves", summary_df['n_curves'].sum())
                
                # ============================================================
                # COMPREHENSIVE MODEL COMPARISON AND ANALYSIS
                # ============================================================
                if fitting_mode == "comprehensive" and len(selected_groups) >= 1:
                    st.markdown("---")
                    st.header("📊 Comprehensive Model Analysis")
                    
                    # --- Section 1: Best Model Selection Statistics ---
                    st.subheader("🏆 Best Model Selection (by " + criterion.upper() + ")")
                    
                    best_model_data = []
                    for group_name in selected_groups:
                        if group_name in st.session_state.all_model_results:
                            for result in results_summary:
                                if result['group'] == group_name:
                                    counts = result.get('best_model_counts', {})
                                    total = sum(counts.values())
                                    row = {'Group': group_name}
                                    for model in all_models:
                                        count = counts.get(model, 0)
                                        row[f'{model}'] = count
                                        row[f'{model} (%)'] = f"{100*count/total:.1f}%" if total > 0 else "0%"
                                    best_model_data.append(row)
                    
                    if best_model_data:
                        best_model_df = pd.DataFrame(best_model_data)
                        st.dataframe(best_model_df, use_container_width=True)
                        
                        # Plot best model distribution
                        fig_best, ax_best = plt.subplots(figsize=(10, 5))
                        x = np.arange(len(selected_groups))
                        width = 0.2
                        colors = {'single': '#1f77b4', 'double': '#ff7f0e', 'triple': '#2ca02c', 'reaction_diffusion': '#d62728'}
                        
                        for i, model in enumerate(all_models):
                            counts = [best_model_df[best_model_df['Group'] == g][model].values[0] if g in best_model_df['Group'].values else 0 for g in selected_groups]
                            ax_best.bar(x + i*width, counts, width, label=model.replace('_', ' ').title(), color=colors[model])
                        
                        ax_best.set_xlabel('Group')
                        ax_best.set_ylabel('Number of Curves')
                        ax_best.set_title(f'Best Model Distribution by Group (Based on {criterion.upper()})')
                        ax_best.set_xticks(x + width * 1.5)
                        ax_best.set_xticklabels(selected_groups, rotation=45, ha='right')
                        ax_best.legend()
                        plt.tight_layout()
                        st.pyplot(fig_best)
                        plt.close(fig_best)
                    
                    # --- Section 2: R² Comparison Across Models ---
                    st.subheader("📈 Model Fit Quality (R²) Comparison")
                    
                    r2_comparison_data = []
                    for group_name in selected_groups:
                        if group_name in st.session_state.all_model_results:
                            for model in all_models:
                                df = st.session_state.all_model_results[group_name].get(model, pd.DataFrame())
                                if not df.empty and 'r2' in df.columns:
                                    r2_values = df['r2'].dropna()
                                    if len(r2_values) > 0:
                                        r2_comparison_data.append({
                                            'Group': group_name,
                                            'Model': model.replace('_', ' ').title(),
                                            'Mean R²': r2_values.mean(),
                                            'Std R²': r2_values.std(),
                                            'N': len(r2_values)
                                        })
                    
                    if r2_comparison_data:
                        r2_df = pd.DataFrame(r2_comparison_data)
                        
                        # Create R² comparison plot
                        fig_r2, ax_r2 = plt.subplots(figsize=(12, 5))
                        
                        groups = selected_groups
                        models = [m.replace('_', ' ').title() for m in all_models]
                        x = np.arange(len(groups))
                        width = 0.2
                        
                        for i, model in enumerate(models):
                            model_data = r2_df[r2_df['Model'] == model]
                            means = [model_data[model_data['Group'] == g]['Mean R²'].values[0] if g in model_data['Group'].values else 0 for g in groups]
                            stds = [model_data[model_data['Group'] == g]['Std R²'].values[0] if g in model_data['Group'].values else 0 for g in groups]
                            ax_r2.bar(x + i*width, means, width, yerr=stds, label=model, capsize=3, color=list(colors.values())[i])
                        
                        ax_r2.set_ylabel('R²')
                        ax_r2.set_xlabel('Group')
                        ax_r2.set_title('Model Fit Quality Comparison (Mean R² ± SD)')
                        ax_r2.set_xticks(x + width * 1.5)
                        ax_r2.set_xticklabels(groups, rotation=45, ha='right')
                        ax_r2.legend()
                        ax_r2.axhline(y=st.session_state.r2_threshold, color='red', linestyle='--', alpha=0.5, label=f'R² threshold')
                        plt.tight_layout()
                        st.pyplot(fig_r2)
                        plt.close(fig_r2)
                        
                        st.dataframe(r2_df.pivot(index='Group', columns='Model', values='Mean R²').round(4), use_container_width=True)
                    
                    # --- Section 3: Per-Model Group Comparisons ---
                    st.markdown("---")
                    st.header("🔬 Group Comparisons by Model")
                    st.info("Each tab shows kinetic parameter comparisons across groups using a specific model.")
                    
                    model_tabs = st.tabs([f"📊 {m.replace('_', ' ').title()}" for m in all_models])
                    
                    for tab, model in zip(model_tabs, all_models):
                        with tab:
                            st.subheader(f"{model.replace('_', ' ').title()} Model Analysis")
                            
                            # Collect data for this model across all groups
                            model_group_data = []
                            for group_name in selected_groups:
                                if group_name in st.session_state.all_model_results:
                                    df = st.session_state.all_model_results[group_name].get(model, pd.DataFrame())
                                    if not df.empty:
                                        df_copy = df.copy()
                                        df_copy['Group'] = group_name
                                        model_group_data.append(df_copy)
                            
                            if model_group_data:
                                combined_model_df = pd.concat(model_group_data, ignore_index=True)
                                
                                # Define parameters for each model
                                if model == 'single':
                                    params = [
                                        ('mobile_fraction', 'Mobile Fraction (%)'),
                                        ('k1', 'Rate Constant k (s⁻¹)'),
                                        ('t_half', 'Half-time (s)'),
                                    ]
                                elif model == 'double':
                                    params = [
                                        ('mobile_fraction', 'Mobile Fraction (%)'),
                                        ('k1', 'Fast Rate k₁ (s⁻¹)'),
                                        ('k2', 'Slow Rate k₂ (s⁻¹)'),
                                        ('fraction_fast', 'Fast Population (%)'),
                                        ('fraction_slow', 'Slow Population (%)'),
                                        ('t_half_fast', 'Fast t½ (s)'),
                                        ('t_half_slow', 'Slow t½ (s)'),
                                    ]
                                elif model == 'triple':
                                    params = [
                                        ('mobile_fraction', 'Mobile Fraction (%)'),
                                        ('k1', 'Fast Rate k₁ (s⁻¹)'),
                                        ('k2', 'Medium Rate k₂ (s⁻¹)'),
                                        ('k3', 'Slow Rate k₃ (s⁻¹)'),
                                        ('fraction_fast', 'Fast Population (%)'),
                                        ('fraction_medium', 'Medium Population (%)'),
                                        ('fraction_slow', 'Slow Population (%)'),
                                    ]
                                elif model == 'reaction_diffusion':
                                    params = [
                                        ('mobile_fraction', 'Mobile Fraction (%)'),
                                        ('k_diff', 'Diffusion Rate (s⁻¹)'),
                                        ('k_bind', 'Exchange Rate (s⁻¹)'),
                                        ('fraction_diffusion', 'Diffusion Population (%)'),
                                        ('fraction_binding', 'Binding Population (%)'),
                                        ('t_half_diff', 'Diffusion t½ (s)'),
                                        ('t_half_bind', 'Binding t½ (s)'),
                                    ]
                                
                                # Filter to available parameters
                                available_params = [(col, label) for col, label in params if col in combined_model_df.columns]
                                
                                if available_params:
                                    # Summary statistics table
                                    st.markdown("#### 📋 Summary Statistics")
                                    summary_rows = []
                                    for group_name in selected_groups:
                                        group_df = combined_model_df[combined_model_df['Group'] == group_name]
                                        row = {'Group': group_name, 'N': len(group_df)}
                                        for col, label in available_params:
                                            values = group_df[col].dropna()
                                            if len(values) > 0:
                                                row[label] = f"{values.mean():.3f} ± {values.sem():.3f}"
                                        summary_rows.append(row)
                                    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)
                                    
                                    # Parameter distribution plots
                                    st.markdown("#### 📈 Parameter Distributions")
                                    
                                    n_params = len(available_params)
                                    n_cols = min(3, n_params)
                                    n_rows = (n_params + n_cols - 1) // n_cols
                                    
                                    fig_model, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
                                    if n_params == 1:
                                        axes = np.array([[axes]])
                                    elif n_rows == 1:
                                        axes = axes.reshape(1, -1)
                                    elif n_cols == 1:
                                        axes = axes.reshape(-1, 1)
                                    
                                    group_colors = plt.cm.Set2(np.linspace(0, 1, len(selected_groups)))
                                    
                                    for p_idx, (col, label) in enumerate(available_params):
                                        row_idx = p_idx // n_cols
                                        col_idx = p_idx % n_cols
                                        ax = axes[row_idx, col_idx]
                                        
                                        group_data = [combined_model_df[combined_model_df['Group'] == g][col].dropna().values for g in selected_groups]
                                        positions = range(1, len(selected_groups) + 1)
                                        
                                        bp = ax.boxplot(group_data, positions=positions, widths=0.6, patch_artist=True)
                                        for patch, color in zip(bp['boxes'], group_colors):
                                            patch.set_facecolor(color)
                                            patch.set_alpha(0.7)
                                        
                                        for pos, data, color in zip(positions, group_data, group_colors):
                                            if len(data) > 0:
                                                jitter = np.random.normal(0, 0.08, len(data))
                                                ax.scatter(np.full_like(data, pos) + jitter, data, alpha=0.5, s=15, c=[color], edgecolors='black', linewidths=0.3)
                                        
                                        ax.set_xticklabels([g[:12] + '..' if len(g) > 12 else g for g in selected_groups], rotation=45, ha='right', fontsize=8)
                                        ax.set_ylabel(label, fontsize=9)
                                        ax.set_title(label, fontsize=10)
                                    
                                    # Hide unused axes
                                    for p_idx in range(n_params, n_rows * n_cols):
                                        row_idx = p_idx // n_cols
                                        col_idx = p_idx % n_cols
                                        axes[row_idx, col_idx].set_visible(False)
                                    
                                    plt.tight_layout()
                                    st.pyplot(fig_model)
                                    plt.close(fig_model)
                                    
                                    # Statistical comparisons
                                    if len(selected_groups) >= 2:
                                        st.markdown("#### 📊 Statistical Comparisons")
                                        
                                        stat_results = []
                                        for col, label in available_params:
                                            group_data = [combined_model_df[combined_model_df['Group'] == g][col].dropna().values for g in selected_groups]
                                            
                                            if all(len(d) >= 2 for d in group_data):
                                                if len(selected_groups) > 2:
                                                    h_stat, pval = scipy_stats.kruskal(*group_data)
                                                    test = "Kruskal-Wallis"
                                                else:
                                                    h_stat, pval = scipy_stats.mannwhitneyu(group_data[0], group_data[1], alternative='two-sided')
                                                    test = "Mann-Whitney U"
                                                
                                                sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
                                                stat_results.append({
                                                    'Parameter': label,
                                                    'Test': test,
                                                    'Statistic': f"{h_stat:.3f}",
                                                    'p-value': f"{pval:.2e}" if pval < 0.0001 else f"{pval:.4f}",
                                                    'Sig': sig
                                                })
                                        
                                        if stat_results:
                                            st.dataframe(pd.DataFrame(stat_results), use_container_width=True)
                                            st.caption("*** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
                                            
                                            sig_params = [r['Parameter'] for r in stat_results if r['Sig'] != 'ns']
                                            if sig_params:
                                                st.success(f"🔬 Significant differences in: {', '.join(sig_params)}")
                            else:
                                st.warning(f"No successful fits for {model.replace('_', ' ').title()} model in any group.")
                
                st.success("🎉 Comprehensive analysis complete! All models fitted, compared, and statistics generated.")

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
            ["🧬 Reaction-Diffusion (Recommended)", "🚀 Fit All Models", "🎯 Single Model"],
            help="Reaction-Diffusion: Best for nuclear proteins with binding/unbinding dynamics. Reports mobile fraction, diffusion rate, and exchange rate. Fit All: Tests all models and selects best based on AICc."
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
                if fitting_mode == "🧬 Reaction-Diffusion (Recommended)":
                    # Fit reaction-diffusion model (default for nuclear proteins)
                    from frap_core import FRAPAnalysisCore
                    
                    rxn_diff_results = []
                    for curve_idx, curve in enumerate(analyzer.curves):
                        if curve.normalized_intensity is None:
                            rxn_diff_results.append({})
                            continue
                        
                        time_data = curve.time
                        intensity_data = curve.normalized_intensity
                        
                        try:
                            rxn_fit = FRAPAnalysisCore.fit_reaction_diffusion(time_data, intensity_data)
                            if rxn_fit and rxn_fit.get('success', False):
                                params = rxn_fit.get('params', [])
                                if len(params) >= 5:
                                    A_diff, k_diff, A_bind, k_bind, C = params[:5]
                                    mobile_fraction = (A_diff + A_bind) * 100
                                    diffusion_fraction = A_diff / (A_diff + A_bind) if (A_diff + A_bind) > 0 else np.nan
                                else:
                                    A_diff, k_diff, A_bind, k_bind, C = np.nan, np.nan, np.nan, np.nan, np.nan
                                    mobile_fraction = np.nan
                                    diffusion_fraction = np.nan
                                
                                rxn_diff_results.append({
                                    'model': 'reaction_diffusion',
                                    'r2': rxn_fit.get('r2', np.nan),
                                    'adj_r2': rxn_fit.get('adj_r2', np.nan),
                                    'aic': rxn_fit.get('aic', np.nan),
                                    'aicc': rxn_fit.get('aicc', np.nan),
                                    'bic': rxn_fit.get('bic', np.nan),
                                    'mobile_fraction': mobile_fraction,
                                    'diffusion_fraction': diffusion_fraction * 100 if not np.isnan(diffusion_fraction) else np.nan,
                                    'A_diff': A_diff,
                                    'k_diff': k_diff,
                                    'A_bind': A_bind,
                                    'k_bind': k_bind,
                                    'C': C,
                                    't_half_diff': np.log(2) / k_diff if k_diff > 0 else np.nan,
                                    't_half_bind': np.log(2) / k_bind if k_bind > 0 else np.nan,
                                    't_half': np.log(2) / k_diff if k_diff > 0 else np.nan,  # Primary half-time
                                    'k_fast': k_diff,  # Alias for compatibility
                                    'rate_constant_fast': k_diff
                                })
                            else:
                                rxn_diff_results.append({})
                        except Exception as e:
                            logger.error(f"Reaction-diffusion fit failed for curve {curve_idx}: {e}")
                            rxn_diff_results.append({})
                    
                    analyzer.features = pd.DataFrame([r for r in rxn_diff_results if r])
                    
                    # Apply R² filter
                    if analyzer.features is not None and not analyzer.features.empty:
                        original_count = len(analyzer.features)
                        analyzer.features, removed_count = apply_r2_filter(
                            analyzer.features, 
                            st.session_state.r2_threshold
                        )
                        
                        st.success(f"✅ Successfully fitted Reaction-Diffusion model to {len(analyzer.curves)} curves!")
                        if removed_count > 0:
                            st.info(f"🔍 R² filter: Removed {removed_count}/{original_count} fits with R² < {st.session_state.r2_threshold}")
                    else:
                        st.success(f"✅ Successfully fitted Reaction-Diffusion model to {len(analyzer.curves)} curves!")
                        
                elif fitting_mode == "🚀 Fit All Models":
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
                    st.metric("🏆 Model", best_models.index[0].replace('_', '-').title() if len(best_models) > 0 else "N/A")
            with col_sum3:
                if 'mobile_fraction' in analyzer.features.columns:
                    avg_mobile = analyzer.features['mobile_fraction'].mean()
                    st.metric("📊 Avg Mobile Fraction", f"{avg_mobile:.1f}%" if np.isfinite(avg_mobile) else "N/A")
            with col_sum4:
                if 'r2' in analyzer.features.columns:
                    avg_r2 = analyzer.features['r2'].mean()
                    st.metric("✨ Avg R²", f"{avg_r2:.3f}" if np.isfinite(avg_r2) else "N/A")
            
            # Additional reaction-diffusion metrics
            if 'k_diff' in analyzer.features.columns or 'k_bind' in analyzer.features.columns:
                col_rd1, col_rd2, col_rd3, col_rd4 = st.columns(4)
                with col_rd1:
                    if 'k_diff' in analyzer.features.columns:
                        avg_k_diff = analyzer.features['k_diff'].mean()
                        st.metric("⚡ Avg k_diff (s⁻¹)", f"{avg_k_diff:.3f}" if np.isfinite(avg_k_diff) else "N/A")
                with col_rd2:
                    if 'k_bind' in analyzer.features.columns:
                        avg_k_bind = analyzer.features['k_bind'].mean()
                        st.metric("🔗 Avg k_bind (s⁻¹)", f"{avg_k_bind:.3f}" if np.isfinite(avg_k_bind) else "N/A")
                with col_rd3:
                    if 't_half_diff' in analyzer.features.columns:
                        avg_t_half_diff = analyzer.features['t_half_diff'].mean()
                        st.metric("⏱️ Avg t½ Diffusion (s)", f"{avg_t_half_diff:.2f}" if np.isfinite(avg_t_half_diff) else "N/A")
                with col_rd4:
                    if 't_half_bind' in analyzer.features.columns:
                        avg_t_half_bind = analyzer.features['t_half_bind'].mean()
                        st.metric("⏱️ Avg t½ Binding (s)", f"{avg_t_half_bind:.2f}" if np.isfinite(avg_t_half_bind) else "N/A")
            
            # Model comparison table
            st.markdown("### 🔬 Model Comparison by Curve")
            
            # Create a display dataframe with selected columns
            display_cols = []
            # Include reaction-diffusion specific columns
            for col in ['model', 'r2', 'adj_r2', 'aic', 'aicc', 'bic', 'mobile_fraction', 
                        'k_diff', 'k_bind', 't_half_diff', 't_half_bind', 'diffusion_fraction',
                        'half_time_fast', 'k_fast', 'k_slow']:
                if col in analyzer.features.columns:
                    display_cols.append(col)
            
            if display_cols:
                display_df = analyzer.features[display_cols].copy()
                
                # Format columns for better display
                if 'model' in display_df.columns:
                    display_df['model'] = display_df['model'].apply(lambda x: x.title() if isinstance(x, str) else x)
                
                # Build format dict based on available columns
                format_dict = {}
                if 'r2' in display_df.columns: format_dict['r2'] = '{:.4f}'
                if 'adj_r2' in display_df.columns: format_dict['adj_r2'] = '{:.4f}'
                if 'aic' in display_df.columns: format_dict['aic'] = '{:.2f}'
                if 'aicc' in display_df.columns: format_dict['aicc'] = '{:.2f}'
                if 'bic' in display_df.columns: format_dict['bic'] = '{:.2f}'
                if 'mobile_fraction' in display_df.columns: format_dict['mobile_fraction'] = '{:.1f}'
                if 'diffusion_fraction' in display_df.columns: format_dict['diffusion_fraction'] = '{:.1f}'
                if 'k_diff' in display_df.columns: format_dict['k_diff'] = '{:.4f}'
                if 'k_bind' in display_df.columns: format_dict['k_bind'] = '{:.4f}'
                if 't_half_diff' in display_df.columns: format_dict['t_half_diff'] = '{:.2f}'
                if 't_half_bind' in display_df.columns: format_dict['t_half_bind'] = '{:.2f}'
                if 'half_time_fast' in display_df.columns: format_dict['half_time_fast'] = '{:.2f}'
                if 'k_fast' in display_df.columns: format_dict['k_fast'] = '{:.4f}'
                if 'k_slow' in display_df.columns: format_dict['k_slow'] = '{:.4f}'
                
                st.dataframe(
                    display_df.style.format(format_dict),
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
    st.header("🔬 Comprehensive Model Fitting & Group Comparison")
    
    st.markdown("""
    **Comprehensive Analysis** fits all models to all curves and provides statistical comparisons between groups.
    
    ### Features:
    - **All models fitted**: Single, Double, Triple exponential, and Reaction-Diffusion
    - **Same model applied across conditions**: Compare groups using identical model parameters
    - **Statistical comparisons**: Each kinetic parameter compared between groups
    - **Population analysis**: Component fractions and kinetic properties
    - **Subpopulation detection**: GMM clustering within each group
    """)
    
    if len(st.session_state.data_groups) < 1:
        st.warning("⚠️ Please import and process data first (Pages 1-2)")
        st.stop()
    
    # Configuration sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔧 Analysis Options")
    
    alpha_level = st.sidebar.slider(
        "Significance Level (α)",
        min_value=0.01,
        max_value=0.10,
        value=0.05,
        step=0.01,
        help="P-value threshold for statistical tests"
    )
    
    r2_threshold = st.sidebar.slider(
        "Minimum R² for inclusion",
        min_value=0.5,
        max_value=0.99,
        value=0.8,
        step=0.05,
        help="Curves with R² below this threshold are excluded"
    )
    
    max_mobile_fraction = st.sidebar.slider(
        "Maximum Mobile Fraction (%)",
        min_value=100,
        max_value=150,
        value=105,
        step=5,
        help="Exclude curves with mobile fraction above this threshold (over-recovery indicates normalization issues)"
    )
    
    min_mobile_fraction = st.sidebar.slider(
        "Minimum Mobile Fraction (%)",
        min_value=0,
        max_value=50,
        value=5,
        step=5,
        help="Exclude curves with mobile fraction below this threshold (may indicate failed bleaching or fitting)"
    )
    
    run_subpopulations = st.sidebar.checkbox(
        "Detect Subpopulations (GMM)",
        value=True,
        help="Use Gaussian Mixture Models to identify subpopulations"
    )
    
    if run_subpopulations:
        max_subpops = st.sidebar.slider("Max Subpopulations", 2, 5, 3)
    
    # Group selection
    st.subheader("📁 Select Groups for Analysis")
    
    available_groups = list(st.session_state.data_groups.keys())
    selected_groups = st.multiselect(
        "Groups to include:",
        available_groups,
        default=available_groups,
        help="Select groups to compare"
    )
    
    if len(selected_groups) < 1:
        st.warning("⚠️ Select at least 1 group for analysis")
        st.stop()
    
    # Data summary
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
    
    st.markdown("---")
    
    # Define model labels (needed both inside and outside the analysis block)
    all_models = ['single', 'double', 'triple', 'reaction_diffusion']
    model_labels = {
        'single': 'Single Exponential',
        'double': 'Double Exponential',
        'triple': 'Triple Exponential',
        'reaction_diffusion': 'Reaction-Diffusion'
    }
    
    # Run analysis button
    if st.button("🚀 Run Comprehensive Analysis", type="primary", use_container_width=True):
        
        from frap_core import FRAPAnalysisCore
        from scipy import stats as scipy_stats
        from sklearn.mixture import GaussianMixture
        
        # Initialize results storage
        if 'global_model_results' not in st.session_state:
            st.session_state.global_model_results = {}
        
        overall_progress = st.progress(0)
        status_text = st.empty()
        
        # ============================================================
        # STEP 1: Fit all models to all curves in all groups
        # ============================================================
        status_text.markdown("### Step 1: Fitting all models to all curves...")
        
        st.session_state.global_model_results = {}
        st.session_state.global_raw_curves = {}  # Store raw curve data for plotting
        best_model_summary = []
        
        for g_idx, group_name in enumerate(selected_groups):
            status_text.markdown(f"### Fitting models for **{group_name}** ({g_idx+1}/{len(selected_groups)})")
            
            analyzer = st.session_state.data_groups[group_name]
            st.session_state.global_model_results[group_name] = {}
            st.session_state.global_raw_curves[group_name] = []  # Store curves for this group
            
            model_results = {model: [] for model in all_models}
            best_per_curve = []
            
            for curve_idx, curve in enumerate(analyzer.curves):
                if curve.normalized_intensity is None:
                    continue
                
                time_data = curve.time
                intensity_data = curve.normalized_intensity
                
                # Store raw curve data
                st.session_state.global_raw_curves[group_name].append({
                    'curve_idx': curve_idx,
                    'time': time_data.copy(),
                    'intensity': intensity_data.copy()
                })
                
                curve_fits = {}
                
                # Fit Single Exponential
                try:
                    fit = FRAPAnalysisCore.fit_single_exponential(time_data, intensity_data)
                    if fit and fit.get('success', False):
                        params = fit.get('params', [])
                        if len(params) >= 3:
                            A, k, C = params[:3]
                            # Mobile fraction = plateau intensity = (A + C) × 100%
                            endpoint = A + C
                            result = {
                                'curve_idx': curve_idx, 'success': True, 'model': 'single',
                                'r2': fit.get('r2', np.nan), 'adj_r2': fit.get('adj_r2', np.nan),
                                'aicc': fit.get('aicc', np.nan), 'aic': fit.get('aic', np.nan),
                                'mobile_fraction': endpoint * 100,
                                'k1': k, 't_half': np.log(2) / k if k > 0 else np.nan,
                                'pop1_fraction': 100.0,  # Single component = 100%
                            }
                            # Apply R² and mobile fraction filters
                            mf = result['mobile_fraction']
                            if result['r2'] >= r2_threshold and min_mobile_fraction <= mf <= max_mobile_fraction:
                                model_results['single'].append(result)
                                curve_fits['single'] = result
                except Exception:
                    pass
                
                # Fit Double Exponential
                try:
                    fit = FRAPAnalysisCore.fit_double_exponential(time_data, intensity_data)
                    if fit and fit.get('success', False):
                        params = fit.get('params', [])
                        if len(params) >= 5:
                            A1, k1, A2, k2, C = params[:5]
                            total_A = A1 + A2
                            # Mobile fraction = plateau intensity = (A1 + A2 + C) × 100%
                            endpoint = total_A + C
                            result = {
                                'curve_idx': curve_idx, 'success': True, 'model': 'double',
                                'r2': fit.get('r2', np.nan), 'adj_r2': fit.get('adj_r2', np.nan),
                                'aicc': fit.get('aicc', np.nan), 'aic': fit.get('aic', np.nan),
                                'mobile_fraction': endpoint * 100,
                                'k1': k1, 'k2': k2,
                                't_half_fast': np.log(2) / k1 if k1 > 0 else np.nan,
                                't_half_slow': np.log(2) / k2 if k2 > 0 else np.nan,
                                'pop1_fraction': A1 / total_A * 100 if total_A > 0 else np.nan,
                                'pop2_fraction': A2 / total_A * 100 if total_A > 0 else np.nan,
                            }
                            # Apply R² and mobile fraction filters
                            mf = result['mobile_fraction']
                            if result['r2'] >= r2_threshold and min_mobile_fraction <= mf <= max_mobile_fraction:
                                model_results['double'].append(result)
                                curve_fits['double'] = result
                except Exception:
                    pass
                
                # Fit Triple Exponential
                try:
                    fit = FRAPAnalysisCore.fit_triple_exponential(time_data, intensity_data)
                    if fit and fit.get('success', False):
                        params = fit.get('params', [])
                        if len(params) >= 7:
                            A1, k1, A2, k2, A3, k3, C = params[:7]
                            total_A = A1 + A2 + A3
                            # Mobile fraction = plateau intensity = (A1 + A2 + A3 + C) × 100%
                            endpoint = total_A + C
                            result = {
                                'curve_idx': curve_idx, 'success': True, 'model': 'triple',
                                'r2': fit.get('r2', np.nan), 'adj_r2': fit.get('adj_r2', np.nan),
                                'aicc': fit.get('aicc', np.nan), 'aic': fit.get('aic', np.nan),
                                'mobile_fraction': endpoint * 100,
                                'k1': k1, 'k2': k2, 'k3': k3,
                                't_half_fast': np.log(2) / k1 if k1 > 0 else np.nan,
                                't_half_medium': np.log(2) / k2 if k2 > 0 else np.nan,
                                't_half_slow': np.log(2) / k3 if k3 > 0 else np.nan,
                                'pop1_fraction': A1 / total_A * 100 if total_A > 0 else np.nan,
                                'pop2_fraction': A2 / total_A * 100 if total_A > 0 else np.nan,
                                'pop3_fraction': A3 / total_A * 100 if total_A > 0 else np.nan,
                            }
                            # Apply R² and mobile fraction filters
                            mf = result['mobile_fraction']
                            if result['r2'] >= r2_threshold and min_mobile_fraction <= mf <= max_mobile_fraction:
                                model_results['triple'].append(result)
                                curve_fits['triple'] = result
                except Exception:
                    pass
                
                # Fit Reaction-Diffusion
                try:
                    fit = FRAPAnalysisCore.fit_reaction_diffusion(time_data, intensity_data)
                    if fit and fit.get('success', False):
                        params = fit.get('params', [])
                        if len(params) >= 5:
                            A_diff, k_diff, A_bind, k_bind, C = params[:5]
                            total_A = A_diff + A_bind
                            # Mobile fraction = plateau intensity = (A_diff + A_bind + C) × 100%
                            endpoint = total_A + C
                            result = {
                                'curve_idx': curve_idx, 'success': True, 'model': 'reaction_diffusion',
                                'r2': fit.get('r2', np.nan), 'adj_r2': fit.get('adj_r2', np.nan),
                                'aicc': fit.get('aicc', np.nan), 'aic': fit.get('aic', np.nan),
                                'mobile_fraction': endpoint * 100,
                                'k_diff': k_diff, 'k_bind': k_bind,
                                't_half_diff': np.log(2) / k_diff if k_diff > 0 else np.nan,
                                't_half_bind': np.log(2) / k_bind if k_bind > 0 else np.nan,
                                'pop_diffusion': A_diff / total_A * 100 if total_A > 0 else np.nan,
                                'pop_binding': A_bind / total_A * 100 if total_A > 0 else np.nan,
                            }
                            # Apply R² and mobile fraction filters
                            mf = result['mobile_fraction']
                            if result['r2'] >= r2_threshold and min_mobile_fraction <= mf <= max_mobile_fraction:
                                model_results['reaction_diffusion'].append(result)
                                curve_fits['reaction_diffusion'] = result
                except Exception:
                    pass
                
                # Determine best model for this curve
                if curve_fits:
                    best = min(curve_fits.keys(), key=lambda m: curve_fits[m].get('aicc', np.inf))
                    best_per_curve.append(best)
            
            # Store results
            for model in all_models:
                df = pd.DataFrame(model_results[model])
                st.session_state.global_model_results[group_name][model] = df
            
            # Best model summary
            best_counts = pd.Series(best_per_curve).value_counts().to_dict() if best_per_curve else {}
            for model in all_models:
                best_model_summary.append({
                    'Group': group_name,
                    'Model': model_labels[model],
                    'N_fits': len(model_results[model]),
                    'Best_model_count': best_counts.get(model, 0),
                    'Mean_R2': np.nanmean([r['r2'] for r in model_results[model]]) if model_results[model] else np.nan,
                    'Mean_AICc': np.nanmean([r['aicc'] for r in model_results[model]]) if model_results[model] else np.nan,
                })
            
            overall_progress.progress((g_idx + 1) / len(selected_groups) * 0.5)
        
        # ============================================================
        # STEP 2: Subpopulation Analysis (if enabled)
        # ============================================================
        if run_subpopulations:
            status_text.markdown("### Step 2: Detecting subpopulations...")
            
            for group_name in selected_groups:
                for model in all_models:
                    df = st.session_state.global_model_results[group_name].get(model, pd.DataFrame())
                    if df.empty or len(df) < 5:
                        continue
                    
                    # Select clustering features based on model
                    if model == 'single':
                        cluster_cols = ['mobile_fraction', 'k1']
                    elif model == 'double':
                        cluster_cols = ['mobile_fraction', 'k1', 'k2', 'pop1_fraction']
                    elif model == 'triple':
                        cluster_cols = ['mobile_fraction', 'k1', 'k2', 'k3', 'pop1_fraction', 'pop2_fraction']
                    else:  # reaction_diffusion
                        cluster_cols = ['mobile_fraction', 'k_diff', 'k_bind', 'pop_diffusion']
                    
                    cluster_cols = [c for c in cluster_cols if c in df.columns]
                    if not cluster_cols:
                        continue
                    
                    X = df[cluster_cols].dropna()
                    if len(X) < 5:
                        continue
                    
                    # Standardize
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    # Find optimal number of clusters
                    best_n = 1
                    best_bic = np.inf
                    for n in range(1, min(max_subpops + 1, len(X))):
                        gmm = GaussianMixture(n_components=n, random_state=42, n_init=3)
                        gmm.fit(X_scaled)
                        bic = gmm.bic(X_scaled)
                        if bic < best_bic:
                            best_bic = bic
                            best_n = n
                    
                    # Assign subpopulations
                    gmm = GaussianMixture(n_components=best_n, random_state=42, n_init=3)
                    labels = gmm.fit_predict(X_scaled)
                    
                    df.loc[X.index, 'subpopulation'] = labels
                    df.loc[X.index, 'n_subpopulations'] = best_n
                    st.session_state.global_model_results[group_name][model] = df
        
        overall_progress.progress(0.7)
        
        # ============================================================
        # DISPLAY RESULTS
        # ============================================================
        status_text.empty()
        overall_progress.empty()
        
        st.markdown("---")
        st.header("📈 Analysis Results")
        
        # --- Section 1: Model Fit Quality Overview ---
        st.subheader("🏆 Model Fit Quality Overview")
        
        summary_df = pd.DataFrame(best_model_summary)
        
        # Store summary in session state for HTML report generation
        st.session_state.global_summary_df = summary_df
        st.session_state.global_selected_groups = selected_groups
        st.session_state.global_analysis_settings = {
            'r2_threshold': r2_threshold,
            'min_mobile_fraction': min_mobile_fraction,
            'max_mobile_fraction': max_mobile_fraction,
            'alpha_level': alpha_level,
            'run_subpopulations': run_subpopulations
        }
        
        # Pivot table: Groups vs Models
        r2_pivot = summary_df.pivot(index='Group', columns='Model', values='Mean_R2')
        st.markdown("**Mean R² by Group and Model:**")
        st.dataframe(r2_pivot.style.format("{:.4f}").background_gradient(cmap='RdYlGn', vmin=0.7, vmax=1.0), use_container_width=True)
        
        # Best model distribution
        st.markdown("**Best Model Selection (by AICc):**")
        best_pivot = summary_df.pivot(index='Group', columns='Model', values='Best_model_count')
        st.dataframe(best_pivot, use_container_width=True)
        
        # Plot R² comparison
        fig_r2, ax_r2 = plt.subplots(figsize=(12, 5))
        x = np.arange(len(selected_groups))
        width = 0.2
        colors = {'Single Exponential': '#1f77b4', 'Double Exponential': '#ff7f0e', 
                  'Triple Exponential': '#2ca02c', 'Reaction-Diffusion': '#d62728'}
        
        for i, model in enumerate(model_labels.values()):
            means = [summary_df[(summary_df['Group'] == g) & (summary_df['Model'] == model)]['Mean_R2'].values[0] 
                     if len(summary_df[(summary_df['Group'] == g) & (summary_df['Model'] == model)]) > 0 else 0 
                     for g in selected_groups]
            ax_r2.bar(x + i*width, means, width, label=model, color=colors.get(model, 'gray'))
        
        ax_r2.set_ylabel('Mean R²')
        ax_r2.set_xlabel('Group')
        ax_r2.set_title('Model Fit Quality Comparison')
        ax_r2.set_xticks(x + width * 1.5)
        ax_r2.set_xticklabels(selected_groups, rotation=45, ha='right')
        ax_r2.legend(loc='lower right')
        ax_r2.axhline(y=r2_threshold, color='red', linestyle='--', alpha=0.5, label=f'R² threshold')
        ax_r2.set_ylim(0.5, 1.05)
        plt.tight_layout()
        st.pyplot(fig_r2)
        plt.close(fig_r2)
        
        # --- Section 2: Per-Model Group Comparisons ---
        st.markdown("---")
        st.header("🔬 Statistical Comparisons by Model")
        st.info("Each tab compares all kinetic parameters between groups using the same model.")
        
        model_tabs = st.tabs([f"📊 {model_labels[m]}" for m in all_models])
        
        for tab, model in zip(model_tabs, all_models):
            with tab:
                st.subheader(f"{model_labels[model]} Analysis")
                
                # Collect data across groups
                model_group_data = []
                for group_name in selected_groups:
                    df = st.session_state.global_model_results.get(group_name, {}).get(model, pd.DataFrame())
                    if not df.empty:
                        df_copy = df.copy()
                        df_copy['Group'] = group_name
                        model_group_data.append(df_copy)
                
                if not model_group_data:
                    st.warning(f"No successful fits for {model_labels[model]}")
                    continue
                
                combined_df = pd.concat(model_group_data, ignore_index=True)
                
                # Define parameters for this model
                if model == 'single':
                    params = [
                        ('mobile_fraction', 'Mobile Fraction (%)', 'kinetic'),
                        ('k1', 'Rate Constant k (s⁻¹)', 'kinetic'),
                        ('t_half', 'Half-time (s)', 'kinetic'),
                    ]
                    pop_params = []
                elif model == 'double':
                    params = [
                        ('mobile_fraction', 'Mobile Fraction (%)', 'kinetic'),
                        ('k1', 'Fast Rate k₁ (s⁻¹)', 'kinetic'),
                        ('k2', 'Slow Rate k₂ (s⁻¹)', 'kinetic'),
                        ('t_half_fast', 'Fast t½ (s)', 'kinetic'),
                        ('t_half_slow', 'Slow t½ (s)', 'kinetic'),
                    ]
                    pop_params = [
                        ('pop1_fraction', 'Fast Population (%)', 'population'),
                        ('pop2_fraction', 'Slow Population (%)', 'population'),
                    ]
                elif model == 'triple':
                    params = [
                        ('mobile_fraction', 'Mobile Fraction (%)', 'kinetic'),
                        ('k1', 'Fast Rate k₁ (s⁻¹)', 'kinetic'),
                        ('k2', 'Medium Rate k₂ (s⁻¹)', 'kinetic'),
                        ('k3', 'Slow Rate k₃ (s⁻¹)', 'kinetic'),
                    ]
                    pop_params = [
                        ('pop1_fraction', 'Fast Population (%)', 'population'),
                        ('pop2_fraction', 'Medium Population (%)', 'population'),
                        ('pop3_fraction', 'Slow Population (%)', 'population'),
                    ]
                else:  # reaction_diffusion
                    params = [
                        ('mobile_fraction', 'Mobile Fraction (%)', 'kinetic'),
                        ('k_diff', 'Diffusion Rate (s⁻¹)', 'kinetic'),
                        ('k_bind', 'Exchange Rate (s⁻¹)', 'kinetic'),
                        ('t_half_diff', 'Diffusion t½ (s)', 'kinetic'),
                        ('t_half_bind', 'Binding t½ (s)', 'kinetic'),
                    ]
                    pop_params = [
                        ('pop_diffusion', 'Diffusion Population (%)', 'population'),
                        ('pop_binding', 'Binding Population (%)', 'population'),
                    ]
                
                all_params = params + pop_params
                available_params = [(c, l, t) for c, l, t in all_params if c in combined_df.columns]
                
                if not available_params:
                    st.warning("No parameters available for comparison")
                    continue
                
                # Summary statistics
                st.markdown("#### 📋 Summary Statistics (Mean ± SEM)")
                summary_rows = []
                for group_name in selected_groups:
                    group_df = combined_df[combined_df['Group'] == group_name]
                    row = {'Group': group_name, 'N': len(group_df)}
                    for col, label, ptype in available_params:
                        values = group_df[col].dropna()
                        if len(values) > 0:
                            row[label] = f"{values.mean():.3f} ± {values.sem():.3f}"
                    summary_rows.append(row)
                st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)
                
                # Kinetic Parameters Plots
                kinetic_params = [(c, l) for c, l, t in available_params if t == 'kinetic']
                if kinetic_params:
                    st.markdown("#### 📈 Kinetic Parameter Distributions")
                    
                    n_params = len(kinetic_params)
                    n_cols = min(3, n_params)
                    n_rows = (n_params + n_cols - 1) // n_cols
                    
                    fig_kin, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
                    if n_params == 1:
                        axes = np.array([[axes]])
                    elif n_rows == 1:
                        axes = axes.reshape(1, -1)
                    elif n_cols == 1:
                        axes = axes.reshape(-1, 1)
                    
                    group_colors = plt.cm.Set2(np.linspace(0, 1, len(selected_groups)))
                    
                    for p_idx, (col, label) in enumerate(kinetic_params):
                        r_idx, c_idx = p_idx // n_cols, p_idx % n_cols
                        ax = axes[r_idx, c_idx]
                        
                        group_data = [combined_df[combined_df['Group'] == g][col].dropna().values for g in selected_groups]
                        positions = range(1, len(selected_groups) + 1)
                        
                        bp = ax.boxplot(group_data, positions=positions, widths=0.6, patch_artist=True)
                        for patch, color in zip(bp['boxes'], group_colors):
                            patch.set_facecolor(color)
                            patch.set_alpha(0.7)
                        
                        for pos, data, color in zip(positions, group_data, group_colors):
                            if len(data) > 0:
                                jitter = np.random.normal(0, 0.08, len(data))
                                ax.scatter(np.full_like(data, pos) + jitter, data, alpha=0.5, s=15, c=[color], edgecolors='black', linewidths=0.3)
                        
                        ax.set_xticklabels([g[:10] + '..' if len(g) > 10 else g for g in selected_groups], rotation=45, ha='right', fontsize=8)
                        ax.set_ylabel(label, fontsize=9)
                        ax.set_title(label, fontsize=10)
                    
                    for p_idx in range(n_params, n_rows * n_cols):
                        axes[p_idx // n_cols, p_idx % n_cols].set_visible(False)
                    
                    plt.tight_layout()
                    st.pyplot(fig_kin)
                    plt.close(fig_kin)
                
                # Population Fraction Plots
                pop_params_avail = [(c, l) for c, l, t in available_params if t == 'population']
                if pop_params_avail:
                    st.markdown("#### 🥧 Population Fractions by Group")
                    
                    # Stacked bar chart
                    fig_pop, ax_pop = plt.subplots(figsize=(10, 5))
                    
                    pop_cols = [c for c, l in pop_params_avail]
                    pop_labels = [l for c, l in pop_params_avail]
                    
                    x = np.arange(len(selected_groups))
                    width = 0.6
                    bottom = np.zeros(len(selected_groups))
                    colors_pop = plt.cm.Pastel1(np.linspace(0, 1, len(pop_cols)))
                    
                    for i, (col, label) in enumerate(zip(pop_cols, pop_labels)):
                        means = [combined_df[combined_df['Group'] == g][col].mean() for g in selected_groups]
                        ax_pop.bar(x, means, width, bottom=bottom, label=label, color=colors_pop[i])
                        bottom += np.array(means)
                    
                    ax_pop.set_ylabel('Population Fraction (%)')
                    ax_pop.set_xlabel('Group')
                    ax_pop.set_title('Component Population Sizes')
                    ax_pop.set_xticks(x)
                    ax_pop.set_xticklabels(selected_groups, rotation=45, ha='right')
                    ax_pop.legend(loc='upper right')
                    ax_pop.set_ylim(0, 110)
                    plt.tight_layout()
                    st.pyplot(fig_pop)
                    plt.close(fig_pop)
                
                # Statistical Tests
                if len(selected_groups) >= 2:
                    st.markdown("#### 📊 Statistical Comparisons")
                    
                    stat_results = []
                    for col, label, ptype in available_params:
                        group_data = [combined_df[combined_df['Group'] == g][col].dropna().values for g in selected_groups]
                        
                        if all(len(d) >= 2 for d in group_data):
                            if len(selected_groups) > 2:
                                stat, pval = scipy_stats.kruskal(*group_data)
                                test = "Kruskal-Wallis"
                            else:
                                stat, pval = scipy_stats.mannwhitneyu(group_data[0], group_data[1], alternative='two-sided')
                                test = "Mann-Whitney U"
                            
                            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < alpha_level else "ns"
                            stat_results.append({
                                'Parameter': label,
                                'Type': ptype.title(),
                                'Test': test,
                                'Statistic': f"{stat:.3f}",
                                'p-value': f"{pval:.2e}" if pval < 0.0001 else f"{pval:.4f}",
                                'Significance': sig
                            })
                    
                    if stat_results:
                        stat_df = pd.DataFrame(stat_results)
                        st.dataframe(stat_df, use_container_width=True)
                        st.caption(f"Significance: *** p<0.001, ** p<0.01, * p<{alpha_level}, ns = not significant")
                        
                        sig_params = [r['Parameter'] for r in stat_results if r['Significance'] != 'ns']
                        if sig_params:
                            st.success(f"🔬 **Significant differences detected in:** {', '.join(sig_params)}")
                        else:
                            st.info("📊 No statistically significant differences between groups for this model.")
                
                # Subpopulation Results
                if run_subpopulations and 'subpopulation' in combined_df.columns:
                    st.markdown("#### 🔍 Subpopulation Analysis")
                    
                    subpop_summary = []
                    for group_name in selected_groups:
                        group_df = combined_df[combined_df['Group'] == group_name]
                        if 'n_subpopulations' in group_df.columns:
                            n_subpops = int(group_df['n_subpopulations'].mode().iloc[0]) if not group_df['n_subpopulations'].isna().all() else 1
                            subpop_counts = group_df['subpopulation'].value_counts().to_dict()
                            subpop_summary.append({
                                'Group': group_name,
                                'N_Subpopulations': n_subpops,
                                'Distribution': str(subpop_counts)
                            })
                    
                    if subpop_summary:
                        st.dataframe(pd.DataFrame(subpop_summary), use_container_width=True)
        
        # --- Section 3: Cross-Model Summary ---
        st.markdown("---")
        st.header("📋 Summary: Recommended Model")
        
        # Determine best model overall
        avg_r2_by_model = summary_df.groupby('Model')['Mean_R2'].mean()
        best_overall = avg_r2_by_model.idxmax()
        
        total_best_counts = summary_df.groupby('Model')['Best_model_count'].sum()
        most_selected = total_best_counts.idxmax()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Highest Mean R²", best_overall, f"R² = {avg_r2_by_model[best_overall]:.4f}")
        with col2:
            st.metric("Most Selected (AICc)", most_selected, f"{total_best_counts[most_selected]} curves")
        with col3:
            # Recommendation based on both criteria
            if best_overall == most_selected:
                recommendation = best_overall
            else:
                recommendation = most_selected  # Prefer AICc selection
            st.metric("Recommended Model", recommendation)
        
        st.success("🎉 Comprehensive analysis complete! Use the tabs above to explore each model's results.")
        
        # ============================================================
        # SECTION 4: Mean ± SD Recovery Curves with Fits
        # ============================================================
        st.markdown("---")
        st.header("📉 Recovery Curves: Mean ± SD with Model Fits")
        
        st.info("**Recovery curves** show the mean ± SD of normalized intensity across all curves in each group, with the reaction-diffusion model fit overlaid.")
        
        # Define reaction-diffusion model function for plotting
        def reaction_diffusion_model(t, A_diff, k_diff, A_bind, k_bind, C):
            return A_diff * (1 - np.exp(-k_diff * t)) + A_bind * (1 - np.exp(-k_bind * t)) + C
        
        # Create tabs for different views
        curve_tabs = st.tabs(["📊 By Group", "🔬 By Subpopulation (if detected)"])
        
        with curve_tabs[0]:
            st.subheader("Mean ± SD Recovery Curves by Group")
            
            # Calculate mean and SD for each group
            group_curve_stats = {}
            
            for group_name in selected_groups:
                raw_curves = st.session_state.global_raw_curves.get(group_name, [])
                if not raw_curves:
                    continue
                
                # Find common time grid (use first curve as reference)
                ref_time = raw_curves[0]['time']
                
                # Interpolate all curves to common time grid
                interpolated = []
                for curve_data in raw_curves:
                    try:
                        interp_intensity = np.interp(ref_time, curve_data['time'], curve_data['intensity'])
                        interpolated.append(interp_intensity)
                    except Exception:
                        continue
                
                if len(interpolated) >= 2:
                    intensity_matrix = np.array(interpolated)
                    mean_curve = np.nanmean(intensity_matrix, axis=0)
                    std_curve = np.nanstd(intensity_matrix, axis=0)
                    sem_curve = std_curve / np.sqrt(len(interpolated))
                    
                    group_curve_stats[group_name] = {
                        'time': ref_time,
                        'mean': mean_curve,
                        'std': std_curve,
                        'sem': sem_curve,
                        'n': len(interpolated)
                    }
            
            if group_curve_stats:
                # Plot all groups on one figure
                n_groups = len(group_curve_stats)
                fig_curves, axes = plt.subplots(1, n_groups, figsize=(5*n_groups, 5), squeeze=False)
                
                group_colors = plt.cm.Set1(np.linspace(0, 1, n_groups))
                
                for idx, (group_name, stats) in enumerate(group_curve_stats.items()):
                    ax = axes[0, idx]
                    time = stats['time']
                    mean = stats['mean']
                    std = stats['std']
                    
                    # Plot mean ± SD
                    ax.fill_between(time, mean - std, mean + std, alpha=0.3, color=group_colors[idx], label='±1 SD')
                    ax.plot(time, mean, '-', color=group_colors[idx], linewidth=2, label=f'Mean (n={stats["n"]})')
                    
                    # Get reaction-diffusion fit parameters for this group
                    rd_df = st.session_state.global_model_results.get(group_name, {}).get('reaction_diffusion', pd.DataFrame())
                    if not rd_df.empty:
                        # Use median parameters for the fit line
                        A_diff_med = rd_df['pop_diffusion'].median() / 100 * rd_df['mobile_fraction'].median() / 100
                        k_diff_med = rd_df['k_diff'].median()
                        A_bind_med = rd_df['pop_binding'].median() / 100 * rd_df['mobile_fraction'].median() / 100
                        k_bind_med = rd_df['k_bind'].median()
                        
                        # Estimate C from mobile fraction
                        mf_med = rd_df['mobile_fraction'].median() / 100
                        C_med = mean[0]  # Use initial value as C
                        
                        # Generate fit curve
                        t_fit = np.linspace(time.min(), time.max(), 200)
                        try:
                            fit_curve = reaction_diffusion_model(t_fit, A_diff_med, k_diff_med, A_bind_med, k_bind_med, C_med)
                            ax.plot(t_fit, fit_curve, '--', color='darkred', linewidth=2, label='R-D Fit')
                        except Exception:
                            pass
                    
                    ax.set_xlabel('Time (s)', fontsize=10)
                    ax.set_ylabel('Normalized Intensity', fontsize=10)
                    ax.set_title(f'{group_name}', fontsize=11, fontweight='bold')
                    ax.legend(loc='lower right', fontsize=8)
                    ax.set_ylim(0, 1.1)
                    ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig_curves)
                plt.close(fig_curves)
                
                # Also create overlay plot
                st.markdown("#### All Groups Overlaid")
                fig_overlay, ax_overlay = plt.subplots(figsize=(10, 6))
                
                for idx, (group_name, stats) in enumerate(group_curve_stats.items()):
                    time = stats['time']
                    mean = stats['mean']
                    std = stats['std']
                    
                    ax_overlay.fill_between(time, mean - std, mean + std, alpha=0.2, color=group_colors[idx])
                    ax_overlay.plot(time, mean, '-', color=group_colors[idx], linewidth=2, label=f'{group_name} (n={stats["n"]})')
                
                ax_overlay.set_xlabel('Time (s)', fontsize=11)
                ax_overlay.set_ylabel('Normalized Intensity', fontsize=11)
                ax_overlay.set_title('Recovery Curves: All Groups (Mean ± SD)', fontsize=12, fontweight='bold')
                ax_overlay.legend(loc='lower right', fontsize=9)
                ax_overlay.set_ylim(0, 1.1)
                ax_overlay.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig_overlay)
                plt.close(fig_overlay)
        
        with curve_tabs[1]:
            st.subheader("Mean ± SD Recovery Curves by Subpopulation")
            
            if not run_subpopulations:
                st.warning("⚠️ Subpopulation detection was not enabled. Enable it in the sidebar and re-run analysis.")
            else:
                # Check if subpopulations were detected
                has_subpops = False
                for group_name in selected_groups:
                    rd_df = st.session_state.global_model_results.get(group_name, {}).get('reaction_diffusion', pd.DataFrame())
                    if not rd_df.empty and 'subpopulation' in rd_df.columns:
                        has_subpops = True
                        break
                
                if not has_subpops:
                    st.info("No subpopulations were detected in the data (BIC selected 1 component).")
                else:
                    # Plot subpopulation curves for each group
                    for group_name in selected_groups:
                        rd_df = st.session_state.global_model_results.get(group_name, {}).get('reaction_diffusion', pd.DataFrame())
                        raw_curves = st.session_state.global_raw_curves.get(group_name, [])
                        
                        if rd_df.empty or 'subpopulation' not in rd_df.columns or not raw_curves:
                            continue
                        
                        n_subpops = int(rd_df['n_subpopulations'].mode().iloc[0]) if 'n_subpopulations' in rd_df.columns else 1
                        
                        if n_subpops <= 1:
                            continue
                        
                        st.markdown(f"#### {group_name} - {n_subpops} Subpopulations Detected")
                        
                        # Map curve indices to subpopulation labels
                        curve_to_subpop = dict(zip(rd_df['curve_idx'], rd_df['subpopulation']))
                        
                        # Group curves by subpopulation
                        subpop_curves = {i: [] for i in range(n_subpops)}
                        ref_time = raw_curves[0]['time'] if raw_curves else None
                        
                        for curve_data in raw_curves:
                            curve_idx = curve_data['curve_idx']
                            if curve_idx in curve_to_subpop:
                                subpop = int(curve_to_subpop[curve_idx])
                                try:
                                    interp_intensity = np.interp(ref_time, curve_data['time'], curve_data['intensity'])
                                    subpop_curves[subpop].append(interp_intensity)
                                except Exception:
                                    continue
                        
                        # Calculate stats for each subpopulation
                        subpop_stats = {}
                        for subpop, curves in subpop_curves.items():
                            if len(curves) >= 2:
                                intensity_matrix = np.array(curves)
                                subpop_stats[subpop] = {
                                    'time': ref_time,
                                    'mean': np.nanmean(intensity_matrix, axis=0),
                                    'std': np.nanstd(intensity_matrix, axis=0),
                                    'n': len(curves)
                                }
                        
                        if subpop_stats:
                            fig_subpop, ax_subpop = plt.subplots(figsize=(10, 5))
                            subpop_colors = plt.cm.tab10(np.linspace(0, 1, n_subpops))
                            
                            for subpop, stats in subpop_stats.items():
                                time = stats['time']
                                mean = stats['mean']
                                std = stats['std']
                                
                                ax_subpop.fill_between(time, mean - std, mean + std, alpha=0.2, color=subpop_colors[subpop])
                                ax_subpop.plot(time, mean, '-', color=subpop_colors[subpop], linewidth=2, 
                                              label=f'Subpop {subpop+1} (n={stats["n"]})')
                                
                                # Add R-D fit for this subpopulation
                                subpop_df = rd_df[rd_df['subpopulation'] == subpop]
                                if len(subpop_df) >= 2:
                                    A_diff_med = subpop_df['pop_diffusion'].median() / 100 * subpop_df['mobile_fraction'].median() / 100
                                    k_diff_med = subpop_df['k_diff'].median()
                                    A_bind_med = subpop_df['pop_binding'].median() / 100 * subpop_df['mobile_fraction'].median() / 100
                                    k_bind_med = subpop_df['k_bind'].median()
                                    C_med = mean[0]
                                    
                                    t_fit = np.linspace(time.min(), time.max(), 200)
                                    try:
                                        fit_curve = reaction_diffusion_model(t_fit, A_diff_med, k_diff_med, A_bind_med, k_bind_med, C_med)
                                        ax_subpop.plot(t_fit, fit_curve, '--', color=subpop_colors[subpop], linewidth=1.5, alpha=0.8)
                                    except Exception:
                                        pass
                            
                            ax_subpop.set_xlabel('Time (s)', fontsize=11)
                            ax_subpop.set_ylabel('Normalized Intensity', fontsize=11)
                            ax_subpop.set_title(f'{group_name}: Subpopulation Recovery Curves', fontsize=12, fontweight='bold')
                            ax_subpop.legend(loc='lower right', fontsize=9)
                            ax_subpop.set_ylim(0, 1.1)
                            ax_subpop.grid(True, alpha=0.3)
                            plt.tight_layout()
                            st.pyplot(fig_subpop)
                            plt.close(fig_subpop)
                            
                            # Show subpopulation statistics
                            subpop_param_summary = []
                            for subpop in range(n_subpops):
                                subpop_df = rd_df[rd_df['subpopulation'] == subpop]
                                if len(subpop_df) >= 1:
                                    subpop_param_summary.append({
                                        'Subpopulation': f'Subpop {subpop+1}',
                                        'N': len(subpop_df),
                                        'Mobile Fraction (%)': f"{subpop_df['mobile_fraction'].mean():.1f} ± {subpop_df['mobile_fraction'].std():.1f}",
                                        'k_diff (s⁻¹)': f"{subpop_df['k_diff'].mean():.3f} ± {subpop_df['k_diff'].std():.3f}",
                                        'k_bind (s⁻¹)': f"{subpop_df['k_bind'].mean():.3f} ± {subpop_df['k_bind'].std():.3f}",
                                        'Diffusion Pop (%)': f"{subpop_df['pop_diffusion'].mean():.1f} ± {subpop_df['pop_diffusion'].std():.1f}",
                                    })
                            
                            if subpop_param_summary:
                                st.markdown("**Subpopulation Parameters (Mean ± SD):**")
                                st.dataframe(pd.DataFrame(subpop_param_summary), use_container_width=True)
        
        # Store raw curves in session state for HTML report
        st.session_state.global_group_curve_stats = group_curve_stats if 'group_curve_stats' in dir() else {}
        
        # Store recommendation in session state
        st.session_state.global_recommendation = recommendation
        st.session_state.global_best_overall = best_overall
        st.session_state.global_most_selected = most_selected
        st.session_state.global_avg_r2_by_model = avg_r2_by_model.to_dict()
        st.session_state.global_total_best_counts = total_best_counts.to_dict()
    
    # ============================================================
    # DOWNLOAD SECTION - Outside the analysis button block
    # ============================================================
    # Check if analysis results exist in session state
    if 'global_model_results' in st.session_state and st.session_state.global_model_results:
        st.markdown("---")
        st.subheader("📥 Download Results")
        
        # Retrieve stored data
        summary_df = st.session_state.get('global_summary_df', pd.DataFrame())
        stored_groups = st.session_state.get('global_selected_groups', [])
        recommendation = st.session_state.get('global_recommendation', 'Single Exponential')
        
        col1, col2 = st.columns(2)
        with col1:
            if not summary_df.empty:
                csv_summary = summary_df.to_csv(index=False)
                st.download_button(
                    "📊 Download Model Summary (CSV)",
                    data=csv_summary,
                    file_name="global_model_summary.csv",
                    mime="text/csv",
                    key="download_summary_csv"
                )
        
        with col2:
            # Download all results for recommended model
            rec_model_key = [k for k, v in model_labels.items() if v == recommendation]
            if rec_model_key:
                rec_model_key = rec_model_key[0]
                all_rec_data = []
                for group_name in stored_groups:
                    df = st.session_state.global_model_results.get(group_name, {}).get(rec_model_key, pd.DataFrame())
                    if not df.empty:
                        df_copy = df.copy()
                        df_copy['Group'] = group_name
                        all_rec_data.append(df_copy)
                
                if all_rec_data:
                    combined_rec = pd.concat(all_rec_data, ignore_index=True)
                    csv_rec = combined_rec.to_csv(index=False)
                    st.download_button(
                        f"📄 Download {recommendation} Results (CSV)",
                        data=csv_rec,
                        file_name=f"global_{rec_model_key}_results.csv",
                        mime="text/csv",
                        key="download_rec_csv"
                    )
        
        # HTML Report Generation
        st.markdown("---")
        st.subheader("📄 Generate HTML Report")
        
        # Generate HTML directly without a button (avoids the rerun issue)
        from datetime import datetime
        from scipy import stats as scipy_stats
        
        # Retrieve settings
        settings = st.session_state.get('global_analysis_settings', {})
        r2_thresh = settings.get('r2_threshold', 0.8)
        min_mf = settings.get('min_mobile_fraction', 5)
        max_mf = settings.get('max_mobile_fraction', 105)
        alpha = settings.get('alpha_level', 0.05)
        subpop_enabled = settings.get('run_subpopulations', False)
        
        best_overall = st.session_state.get('global_best_overall', 'N/A')
        most_selected = st.session_state.get('global_most_selected', 'N/A')
        avg_r2_dict = st.session_state.get('global_avg_r2_by_model', {})
        total_best_dict = st.session_state.get('global_total_best_counts', {})
        
        # Build HTML report
        html_parts = []
        
        # Header
        html_parts.append(f"""
        <h1>🔬 FRAP Global Fitting Analysis Report</h1>
        <p><b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><b>Groups Analyzed:</b> {', '.join(stored_groups)}</p>
        <p><b>Quality Filters:</b> R² ≥ {r2_thresh}, Mobile Fraction: {min_mf}% - {max_mf}%</p>
        <hr/>
        """)
        
        # Model Fit Quality Summary
        if not summary_df.empty:
            html_parts.append("<h2>📊 Model Fit Quality Summary</h2>")
            html_parts.append(summary_df.to_html(index=False, classes='summary-table'))
        
        # Best Model Recommendation
        html_parts.append(f"""
        <h2>🏆 Model Recommendation</h2>
        <table class='recommendation-table'>
            <tr><th>Criterion</th><th>Best Model</th><th>Value</th></tr>
            <tr><td>Highest Mean R²</td><td>{best_overall}</td><td>R² = {avg_r2_dict.get(best_overall, 0):.4f}</td></tr>
            <tr><td>Most Selected (AICc)</td><td>{most_selected}</td><td>{total_best_dict.get(most_selected, 0)} curves</td></tr>
            <tr style='background-color:#d4edda;font-weight:bold;'><td>Recommended</td><td colspan='2'>{recommendation}</td></tr>
        </table>
        """)
        
        # Per-Model Detailed Results
        for model in all_models:
            model_name = model_labels[model]
            html_parts.append(f"<h2>📈 {model_name} Results</h2>")
            
            # Collect all groups for this model
            model_all_data = []
            for group_name in stored_groups:
                df = st.session_state.global_model_results.get(group_name, {}).get(model, pd.DataFrame())
                if not df.empty:
                    df_copy = df.copy()
                    df_copy['Group'] = group_name
                    model_all_data.append(df_copy)
            
            if model_all_data:
                model_combined = pd.concat(model_all_data, ignore_index=True)
                
                # Summary statistics per group
                html_parts.append(f"<h3>Summary Statistics (Mean ± SEM)</h3>")
                
                # Get numeric columns for summary
                numeric_cols = model_combined.select_dtypes(include=[np.number]).columns.tolist()
                exclude_cols = ['curve_idx', 'success', 'subpopulation', 'n_subpopulations']
                summary_cols = [c for c in numeric_cols if c not in exclude_cols]
                
                if summary_cols:
                    summary_stats = model_combined.groupby('Group')[summary_cols].agg(['mean', 'sem']).round(4)
                    summary_stats.columns = [f"{col[0]} ({col[1]})" for col in summary_stats.columns]
                    html_parts.append(summary_stats.to_html(classes='stats-table'))
                
                # Statistical comparisons between groups
                if len(stored_groups) >= 2:
                    html_parts.append(f"<h3>Statistical Comparisons</h3>")
                    stat_rows = []
                    for col in summary_cols:
                        if col in ['r2', 'adj_r2', 'aicc', 'aic']:
                            continue  # Skip fit quality metrics
                        group_data = [model_combined[model_combined['Group'] == g][col].dropna().values 
                                     for g in stored_groups]
                        if all(len(d) >= 2 for d in group_data):
                            try:
                                if len(stored_groups) > 2:
                                    stat, pval = scipy_stats.kruskal(*group_data)
                                    test = "Kruskal-Wallis"
                                else:
                                    stat, pval = scipy_stats.mannwhitneyu(group_data[0], group_data[1], alternative='two-sided')
                                    test = "Mann-Whitney U"
                                sig = "Yes" if pval < alpha else "No"
                                sig_color = '#d4edda' if sig == 'Yes' else '#f8d7da'
                                stat_rows.append(f"<tr style='background:{sig_color}'><td>{col}</td><td>{test}</td><td>{stat:.3f}</td><td>{pval:.4e}</td><td>{sig}</td></tr>")
                            except Exception:
                                pass
                    
                    if stat_rows:
                        html_parts.append("""
                        <table class='stat-table'>
                            <thead><tr><th>Parameter</th><th>Test</th><th>Statistic</th><th>p-value</th><th>Significant</th></tr></thead>
                            <tbody>
                        """)
                        html_parts.extend(stat_rows)
                        html_parts.append("</tbody></table>")
                
                # Full data table
                html_parts.append(f"<h3>Detailed Results ({len(model_combined)} curves)</h3>")
                html_parts.append(model_combined.to_html(index=False, classes='data-table'))
            else:
                html_parts.append("<p><i>No successful fits for this model.</i></p>")
            
            html_parts.append("<hr/>")
        
        # Subpopulation summary if enabled
        if subpop_enabled:
            html_parts.append("<h2>🔍 Subpopulation Analysis Summary</h2>")
            subpop_html = []
            for group_name in stored_groups:
                for model in all_models:
                    df = st.session_state.global_model_results.get(group_name, {}).get(model, pd.DataFrame())
                    if not df.empty and 'n_subpopulations' in df.columns:
                        n_subpops = int(df['n_subpopulations'].mode().iloc[0]) if not df['n_subpopulations'].isna().all() else 1
                        subpop_html.append(f"<tr><td>{group_name}</td><td>{model_labels[model]}</td><td>{n_subpops}</td></tr>")
            
            if subpop_html:
                html_parts.append("""
                <table class='subpop-table'>
                    <thead><tr><th>Group</th><th>Model</th><th>Subpopulations Detected</th></tr></thead>
                    <tbody>
                """)
                html_parts.extend(subpop_html)
                html_parts.append("</tbody></table>")
        
        # Assemble full HTML
        body_html = '\n'.join(html_parts)
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset='utf-8'/>
            <title>FRAP Global Fitting Report</title>
            <style>
                body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 2em; background: #fafafa; }}
                h1 {{ color: #1a5276; border-bottom: 3px solid #1a5276; padding-bottom: 10px; }}
                h2 {{ color: #2874a6; margin-top: 2em; border-left: 4px solid #2874a6; padding-left: 10px; }}
                h3 {{ color: #5d6d7e; }}
                table {{ border-collapse: collapse; width: 100%; margin: 1em 0 2em 0; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
                th {{ background-color: #2874a6; color: white; padding: 10px 8px; text-align: left; font-size: 13px; }}
                td {{ border: 1px solid #ddd; padding: 8px; text-align: left; font-size: 12px; }}
                tr:nth-child(even) {{ background-color: #f8f9fa; }}
                tr:hover {{ background-color: #e8f4f8; }}
                .recommendation-table th {{ background-color: #27ae60; }}
                .stat-table th {{ background-color: #8e44ad; }}
                hr {{ border: none; border-top: 2px solid #eee; margin: 2em 0; }}
                p {{ color: #444; line-height: 1.6; }}
            </style>
        </head>
        <body>
            {body_html}
            <footer style='margin-top: 3em; padding-top: 1em; border-top: 1px solid #ddd; color: #888; font-size: 11px;'>
                <p>Generated by FRAP Analysis Suite - Global Fitting Module</p>
            </footer>
        </body>
        </html>
        """
        
        # Provide download button directly
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        st.download_button(
            "🌐 Download HTML Report",
            data=html_template,
            file_name=f"FRAP_Global_Fitting_Report_{timestamp}.html",
            mime="text/html",
            type="secondary",
            use_container_width=True,
            key="download_html_report"
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
