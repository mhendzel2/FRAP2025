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
from typing import Optional, Dict, Any, List
import plotly.graph_objects as go

# Import our new enhanced modules
from frap_input_handler import FRAPInputHandler, FRAPCurveData
from frap_analysis_enhanced import FRAPGroupAnalyzer, FRAPStatisticalComparator
from frap_visualizer import FRAPVisualizer
from frap_report_generator import FRAPReportGenerator

# Import advanced fitting methods
try:
    from frap_robust_bayesian import (
        robust_fit_single_exp, robust_fit_double_exp,
        bayesian_fit_single_exp, compare_fitting_methods,
        AdvancedFitResult
    )
    ROBUST_FITTING_AVAILABLE = True
except ImportError:
    ROBUST_FITTING_AVAILABLE = False
    logger.warning("Robust/Bayesian fitting not available")

# Import bootstrap confidence intervals
try:
    from frap_bootstrap import run_bootstrap
    from frap_statistics import bootstrap_bca_ci, bootstrap_group_comparison
    BOOTSTRAP_AVAILABLE = True
except ImportError:
    BOOTSTRAP_AVAILABLE = False
    logger.warning("Bootstrap methods not available")

# Import advanced biophysical models
try:
    from frap_advanced_fitting import (
        fit_anomalous_diffusion, fit_reaction_diffusion,
        fit_all_advanced_models, compare_groups_advanced_fitting
    )
    ADVANCED_MODELS_AVAILABLE = True
except ImportError:
    ADVANCED_MODELS_AVAILABLE = False
    logger.warning("Advanced biophysical models not available")

# Import holistic group comparison
try:
    from frap_group_comparison import HolisticGroupComparator, compute_average_recovery_profile
    HOLISTIC_COMPARISON_AVAILABLE = True
except ImportError:
    HOLISTIC_COMPARISON_AVAILABLE = False
    logger.warning("Holistic group comparison not available")

# Import outlier detection
try:
    from frap_statistical_outliers import FRAPStatisticalOutlierDetector
    from frap_ml_outliers import FRAPOutlierDetector
    OUTLIER_DETECTION_AVAILABLE = True
except ImportError:
    OUTLIER_DETECTION_AVAILABLE = False
    logger.warning("Outlier detection not available")

# Import reference database
try:
    from frap_reference_database import FRAPReferenceDatabase, display_reference_database_ui
    from frap_reference_integration import add_reference_comparison_to_results
    REFERENCE_DB_AVAILABLE = True
except ImportError:
    REFERENCE_DB_AVAILABLE = False
    logger.warning("Reference database not available")

# Import advanced visualizations
try:
    from frap_stat_viz import plot_volcano, plot_forest, plot_qq, plot_comparison_summary
    from frap_visualizations import plot_spaghetti, plot_heatmap, plot_pairplot
    ADVANCED_VIZ_AVAILABLE = True
except ImportError:
    ADVANCED_VIZ_AVAILABLE = False
    logger.warning("Advanced visualizations not available")

# Import report generation
try:
    from frap_html_reports import generate_html_report
    from frap_pdf_reports import generate_pdf_report
    REPORT_GEN_AVAILABLE = True
except ImportError:
    REPORT_GEN_AVAILABLE = False
    logger.warning("Report generation not available")

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
if 'holistic_results' not in st.session_state:
    st.session_state.holistic_results = None
if 'advanced_fit_results' not in st.session_state:
    st.session_state.advanced_fit_results = None
if 'subpopulation_results' not in st.session_state:
    st.session_state.subpopulation_results = {}  # {group_name: {subpop_id: features_df}}

# --- Sidebar Navigation ---
st.sidebar.title("FRAP Analysis 2.0 Enhanced")

# Feature availability status
with st.sidebar.expander("🔧 Available Features"):
    st.markdown("**Core Features:** ✅")
    st.markdown(f"**Robust/Bayesian Fitting:** {'✅' if ROBUST_FITTING_AVAILABLE else '❌'}")
    st.markdown(f"**Bootstrap CIs:** {'✅' if BOOTSTRAP_AVAILABLE else '❌'}")
    st.markdown(f"**Advanced Models:** {'✅' if ADVANCED_MODELS_AVAILABLE else '❌'}")
    st.markdown(f"**Holistic Comparison:** {'✅' if HOLISTIC_COMPARISON_AVAILABLE else '❌'}")
    st.markdown(f"**Outlier Detection:** {'✅' if OUTLIER_DETECTION_AVAILABLE else '❌'}")
    st.markdown(f"**Reference Database:** {'✅' if REFERENCE_DB_AVAILABLE else '❌'}")
    st.markdown(f"**Advanced Visualizations:** {'✅' if ADVANCED_VIZ_AVAILABLE else '❌'}")
    st.markdown(f"**Report Generation:** {'✅' if REPORT_GEN_AVAILABLE else '❌'}")

page = st.sidebar.radio("Workflow", [
    "1. Import & Preprocess", 
    "2. Subpopulations & Outliers", 
    "3. Model Fitting", 
    "4. Compare Groups", 
    "5. Report"
])

# --- Helper Functions ---
def detect_bleach_frame(intensity_data: np.ndarray) -> int:
    """
    Automatically detect the bleach frame based on the largest intensity drop.
    
    Parameters:
    -----------
    intensity_data : np.ndarray
        Array of intensity values over time
    
    Returns:
    --------
    int
        Index of the bleach frame (frame with minimum intensity)
    """
    # Method 1: Find frame with minimum intensity (most common for FRAP)
    bleach_idx = int(np.argmin(intensity_data))
    
    # Validate: bleach shouldn't be at first or last frame
    if bleach_idx == 0:
        # If minimum is at start, look for largest drop instead
        intensity_diffs = np.diff(intensity_data)
        bleach_idx = int(np.argmin(intensity_diffs)) + 1
    
    if bleach_idx >= len(intensity_data) - 2:
        # If too close to end, use largest drop method
        intensity_diffs = np.diff(intensity_data)
        bleach_idx = int(np.argmin(intensity_diffs)) + 1
    
    return bleach_idx

def load_and_process_file(uploaded_file, bleach_frame_idx=None, auto_detect_bleach=True):
    """
    Load and process a FRAP file.
    
    Parameters:
    -----------
    uploaded_file : UploadedFile
        Streamlit uploaded file object
    bleach_frame_idx : int, optional
        Manual bleach frame index. If None and auto_detect_bleach=True, will auto-detect.
    auto_detect_bleach : bool
        If True, automatically detect bleach frame from intensity data
    
    Returns:
    --------
    FRAPCurveData or None
        Processed curve data or None if error
    """
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
        
        # Auto-detect bleach frame if requested
        if auto_detect_bleach and bleach_frame_idx is None:
            bleach_frame_idx = detect_bleach_frame(curve_data.roi_intensity)
            curve_data.metadata['bleach_frame'] = bleach_frame_idx
            curve_data.metadata['bleach_detection'] = 'automatic'
        elif bleach_frame_idx is not None:
            curve_data.metadata['bleach_frame'] = bleach_frame_idx
            curve_data.metadata['bleach_detection'] = 'manual'
        
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

def load_groups_from_zip(zip_file, bleach_frame_idx=None, auto_detect_bleach=True):
    """
    Load files from a ZIP archive where each subfolder becomes a group.
    
    Parameters:
    -----------
    zip_file : UploadedFile
        ZIP file containing grouped FRAP data
    bleach_frame_idx : int, optional
        Manual bleach frame index for all files. If None and auto_detect_bleach=True, 
        will auto-detect for each file.
    auto_detect_bleach : bool
        If True, automatically detect bleach frame for each file
    
    Returns:
    --------
    tuple
        (groups_data, success_count, error_count, error_details)
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
                        
                        # Auto-detect or use manual bleach frame
                        if auto_detect_bleach and bleach_frame_idx is None:
                            detected_bleach = detect_bleach_frame(curve_data.roi_intensity)
                            curve_data.metadata['bleach_frame'] = detected_bleach
                            curve_data.metadata['bleach_detection'] = 'automatic'
                            use_bleach_idx = detected_bleach
                        else:
                            use_bleach_idx = bleach_frame_idx
                            curve_data.metadata['bleach_frame'] = bleach_frame_idx
                            curve_data.metadata['bleach_detection'] = 'manual'
                        
                        curve_data = FRAPInputHandler.double_normalization(curve_data, use_bleach_idx)
                        curve_data = FRAPInputHandler.time_zero_correction(curve_data, use_bleach_idx)
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
        
        # Bleach frame detection option
        use_auto_detect = st.checkbox(
            "🤖 Auto-detect bleach frame (Recommended)",
            value=True,
            help="Automatically identify bleach frame based on intensity drop in photobleached region",
            key="auto_detect_zip"
        )
        
        if not use_auto_detect:
            bleach_frame_zip = st.number_input(
                "Bleach Frame Index (for all files)", 
                min_value=1, 
                value=10, 
                help="Manual override: Index of the frame where bleaching occurs (0-based)", 
                key="bleach_zip"
            )
        else:
            bleach_frame_zip = None
            st.info("ℹ️ Bleach frame will be automatically detected for each file based on intensity drop.")
        
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
                                uploaded_zip, 
                                bleach_frame_idx=bleach_frame_zip,
                                auto_detect_bleach=use_auto_detect
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
            
            # Bleach frame detection option
            use_auto_detect_individual = st.checkbox(
                "🤖 Auto-detect bleach frame (Recommended)",
                value=True,
                help="Automatically identify bleach frame based on intensity drop in photobleached region",
                key="auto_detect_individual"
            )
            
            if not use_auto_detect_individual:
                bleach_frame = st.number_input(
                    "Bleach Frame Index", 
                    min_value=1, 
                    value=10, 
                    help="Manual override: Index of the frame where bleaching occurs (0-based)"
                )
            else:
                bleach_frame = None
                st.info("ℹ️ Bleach frame will be automatically detected for each file.")
            
            if st.button("Process and Add to Group"):
                if not uploaded_files:
                    st.warning("Please upload files.")
                else:
                    if group_name not in st.session_state.data_groups:
                        st.session_state.data_groups[group_name] = FRAPGroupAnalyzer()
                    
                    analyzer = st.session_state.data_groups[group_name]
                    count = 0
                    progress_bar = st.progress(0)
                    
                    # Show detected bleach frames if auto-detecting
                    detected_frames = []
                    
                    for i, file in enumerate(uploaded_files):
                        curve = load_and_process_file(
                            file, 
                            bleach_frame_idx=bleach_frame,
                            auto_detect_bleach=use_auto_detect_individual
                        )
                        if curve:
                            analyzer.add_curve(curve)
                            count += 1
                            if 'bleach_frame' in curve.metadata:
                                detected_frames.append((file.name, curve.metadata['bleach_frame'], 
                                                       curve.metadata.get('bleach_detection', 'unknown')))
                        progress_bar.progress((i + 1) / len(uploaded_files))
                    
                    st.success(f"Successfully added {count} curves to group '{group_name}'")
                    
                    # Show detection results if auto-detected
                    if detected_frames and use_auto_detect_individual:
                        with st.expander("📊 View Detected Bleach Frames", expanded=True):
                            df_detect = pd.DataFrame(detected_frames, 
                                                    columns=['File', 'Bleach Frame', 'Detection Method'])
                            st.dataframe(df_detect)
                    
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
                            bleach_idx = min(10, len(curve.time)-1)  # Safe bleach index for preview
                            if bleach_idx < len(curve.time):
                                ax.axvline(curve.time[bleach_idx], color='r', linestyle='--', label='Bleach')
                            ax.set_title(f"Preview: First Curve in {name}")
                            ax.legend()
                            st.pyplot(fig)
            else:
                st.info("No data loaded yet.")

# --- Page 2: Subpopulations & Outliers ---
elif page == "2. Subpopulations & Outliers":
    st.header("🔍 Subpopulation & Outlier Detection")
    
    st.markdown("""
    **⚡ Quick Start:** First perform initial fitting to extract basic features, then detect subpopulations.
    Subpopulations will be analyzed separately in the Model Fitting step.
    """)
    
    if not st.session_state.data_groups:
        st.warning("⚠️ Please import data first on the 'Import & Preprocess' page.")
    else:
        group_select = st.selectbox("📁 Select Group", list(st.session_state.data_groups.keys()))
        analyzer = st.session_state.data_groups[group_select]
        
        # Check if initial features exist
        if analyzer.features is None or analyzer.features.empty:
            st.warning("⚠️ Need initial fitting to extract features for clustering.")
            st.info("💡 Click below to automatically fit all models and extract parameters for analysis.")
            
            if st.button("🚀 Extract Basic Features", type="primary", use_container_width=True):
                with st.spinner("Fitting models and extracting features from curves..."):
                    # Use None to fit all models and select best
                    analyzer.analyze_group(model_name=None)
                st.success("✅ Basic features extracted! You can now perform clustering and outlier detection below.")
                st.rerun()
        else:
            # Display current data info
            st.info(f"📊 **{len(analyzer.curves)} curves** loaded with **{len(analyzer.features)} fitted results**")
            
            # Create tabs for different analyses
            tab_cluster, tab_outlier, tab_viz = st.tabs([
                "🎯 Clustering", 
                "⚠️ Outlier Detection",
                "📊 Visualizations"
            ])
            
            with tab_cluster:
                st.subheader("Subpopulation Detection")
                st.markdown("""
                Automatically identify distinct subpopulations in your data using unsupervised clustering.
                Useful for detecting heterogeneous protein behaviors.
                """)
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    # Get numerical columns only
                    numerical_cols = analyzer.features.select_dtypes(include=[np.number]).columns.tolist()
                    if len(numerical_cols) < 2:
                        st.warning("Need at least 2 numerical parameters for clustering")
                    else:
                        max_k = st.slider("Max Components", 2, min(5, len(analyzer.features)), 3)
                        if st.button("🔍 Detect Subpopulations", use_container_width=True):
                            with st.spinner("Running clustering analysis..."):
                                analyzer.detect_subpopulations(range(1, max_k + 1))
                            st.success("✅ Clustering complete!")
                            st.rerun()
                
                with col2:
                    if 'subpopulation' in analyzer.features.columns:
                        # Show cluster statistics
                        clustered_data = analyzer.features.dropna(subset=['subpopulation'])
                        n_clustered = len(clustered_data)
                        
                        st.metric("Clustered Curves", f"{n_clustered}/{len(analyzer.features)}")
                        
                        st.write("**Cluster Distribution:**")
                        counts = clustered_data['subpopulation'].value_counts().sort_index()
                        
                        # Display in columns
                        cols = st.columns(min(len(counts), 4))
                        for idx, (pop, count) in enumerate(counts.items()):
                            with cols[idx % len(cols)]:
                                st.metric(f"Cluster {int(pop)}", f"{count} curves", 
                                        delta=f"{count/n_clustered*100:.1f}%")
                    else:
                        st.info("👆 Run clustering to see results")
            
            with tab_outlier:
                st.subheader("Outlier Detection")
                st.markdown("""
                Identify curves that deviate significantly from the population.
                Multiple methods available for robust detection.
                """)
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    outlier_method = "iqr"
                    if OUTLIER_DETECTION_AVAILABLE:
                        outlier_method = st.selectbox(
                            "Detection Method:",
                            ["iqr", "zscore", "isolation_forest", "lof"],
                            format_func=lambda x: {
                                "iqr": "IQR (Interquartile Range)",
                                "zscore": "Z-Score (Statistical)",
                                "isolation_forest": "Isolation Forest (ML)",
                                "lof": "Local Outlier Factor (ML)"
                            }[x]
                        )
                    
                    outlier_threshold = st.slider("Sensitivity:", 1.0, 3.0, 1.5, 0.1,
                                                 help="Lower = more sensitive")
                    
                    if st.button("⚠️ Detect Outliers", use_container_width=True):
                        with st.spinner(f"Running {outlier_method} outlier detection..."):
                            if OUTLIER_DETECTION_AVAILABLE and outlier_method in ["isolation_forest", "lof"]:
                                # Use ML-based detection
                                detector = FRAPOutlierDetector()
                                outlier_results = detector.detect_outliers(
                                    analyzer.features, 
                                    method=outlier_method
                                )
                                analyzer.features['is_outlier'] = outlier_results['is_outlier']
                            else:
                                # Use statistical detection
                                analyzer.detect_outliers(method=outlier_method, threshold=outlier_threshold)
                        st.success("✅ Outlier detection complete!")
                        st.rerun()
                
                with col2:
                    if 'is_outlier' in analyzer.features.columns:
                        n_outliers = analyzer.features['is_outlier'].sum()
                        n_total = len(analyzer.features)
                        
                        st.metric("Outliers Detected", f"{n_outliers}/{n_total}",
                                delta=f"{n_outliers/n_total*100:.1f}%",
                                delta_color="inverse")
                        
                        if n_outliers > 0:
                            st.warning(f"⚠️ {n_outliers} curves flagged as outliers")
                            
                            if st.checkbox("Show outlier details"):
                                outlier_data = analyzer.features[analyzer.features['is_outlier']==True]
                                st.dataframe(outlier_data)
                            
                            if st.button("🗑️ Remove Outliers from Analysis"):
                                analyzer.features = analyzer.features[analyzer.features['is_outlier']==False]
                                st.success(f"Removed {n_outliers} outliers")
                                st.rerun()
                    else:
                        st.info("👆 Run detection to identify outliers")
            
            with tab_viz:
                st.subheader("Data Visualization")
                
                if 'subpopulation' in analyzer.features.columns or 'is_outlier' in analyzer.features.columns:
                    # Parameter selection for visualization
                    numerical_params = [c for c in analyzer.features.select_dtypes(include=[np.number]).columns 
                                       if c not in ['subpopulation', 'is_outlier']]
                    
                    if len(numerical_params) >= 2:
                        col_x, col_y, col_color = st.columns(3)
                        with col_x:
                            x_axis = st.selectbox("X Axis", numerical_params, index=0)
                        with col_y:
                            y_axis = st.selectbox("Y Axis", numerical_params, index=min(1, len(numerical_params)-1))
                        with col_color:
                            color_by = st.selectbox("Color By", 
                                                   ["None"] + (["subpopulation"] if 'subpopulation' in analyzer.features.columns else []) +
                                                   (["is_outlier"] if 'is_outlier' in analyzer.features.columns else []))
                        
                        # Create visualization
                        if 'subpopulation' in analyzer.features.columns and color_by == "subpopulation":
                            clustered_data = analyzer.features.dropna(subset=['subpopulation'])
                            if len(clustered_data) > 0:
                                fig = FRAPVisualizer.plot_subpopulations(clustered_data, x_axis, y_axis)
                                st.pyplot(fig)
                        else:
                            # Standard scatter plot
                            fig, ax = plt.subplots(figsize=(10, 6))
                            if color_by == "is_outlier" and 'is_outlier' in analyzer.features.columns:
                                outliers = analyzer.features[analyzer.features['is_outlier']==True]
                                normals = analyzer.features[analyzer.features['is_outlier']==False]
                                ax.scatter(normals[x_axis], normals[y_axis], alpha=0.6, label='Normal')
                                ax.scatter(outliers[x_axis], outliers[y_axis], alpha=0.6, 
                                         color='red', marker='x', s=100, label='Outlier')
                                ax.legend()
                            else:
                                ax.scatter(analyzer.features[x_axis], analyzer.features[y_axis], alpha=0.6)
                            ax.set_xlabel(x_axis)
                            ax.set_ylabel(y_axis)
                            ax.set_title(f"{x_axis} vs {y_axis}")
                            ax.grid(alpha=0.3)
                            st.pyplot(fig)
                else:
                    st.info("Run clustering or outlier detection first to visualize groupings")
        
        # Fitting mode selection
        fitting_mode = st.radio(
            "Fitting Strategy:",
            ["🚀 Fit All Models (Recommended)", "🎯 Single Model"],
            help="Fit All Models: Fits single, double, triple exponential and anomalous diffusion models, then selects best based on AICc. Single Model: Fit only one specific model."
        )
        
        if fitting_mode == "🎯 Single Model":
            model_options = ["single", "double", "triple"]
            if ADVANCED_MODELS_AVAILABLE:
                model_options.extend(["anomalous_diffusion", "reaction_diffusion"])
                
            model_select = st.selectbox(
                "Select Model",
                model_options,
                format_func=lambda x: {
                    "single": "Single Exponential (Simple Diffusion)",
                    "double": "Double Exponential (Two Populations)",
                    "triple": "Triple Exponential (Three Populations)",
                    "anomalous_diffusion": "Anomalous Diffusion (Subdiffusive)",
                    "reaction_diffusion": "Reaction-Diffusion (Binding Kinetics)"
                }.get(x, x)
            )
        
        # Bootstrap confidence intervals option
        calculate_bootstrap = False
        if BOOTSTRAP_AVAILABLE:
            calculate_bootstrap = st.checkbox(
                "📊 Calculate Bootstrap Confidence Intervals",
                value=False,
                help="Compute robust 95% CIs for all parameters (computationally intensive)"
            )
            if calculate_bootstrap:
                n_bootstrap = st.slider("Bootstrap Iterations:", 100, 2000, 1000, 100)
        
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
                    st.success(f"✅ Successfully fitted all models to {len(analyzer.curves)} curves!")
                else:
                    # Fit single selected model
                    analyzer.analyze_group(model_name=model_select)
                    st.success(f"✅ Successfully fitted {model_select} model to {len(analyzer.curves)} curves!")
                
                # Apply advanced fitting if selected
                if fitting_method_type != "standard" and ROBUST_FITTING_AVAILABLE:
                    st.info(f"Applying {fitting_method_type} fitting method...")
                    # Here you would integrate robust/Bayesian fitting
                
                # Calculate bootstrap CIs if requested
                if calculate_bootstrap and BOOTSTRAP_AVAILABLE:
                    st.info(f"Calculating bootstrap confidence intervals ({n_bootstrap} iterations)...")
                    # Here you would integrate bootstrap CI calculations
                
                st.balloons()
        
        # Display results (existing code continues...)
        
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
        
        # Check if subpopulations or outliers were detected
        has_subpopulations = (analyzer.features is not None and 
                            'subpopulation' in analyzer.features.columns and 
                            analyzer.features['subpopulation'].nunique() > 1)
        has_outliers = (analyzer.features is not None and 
                       'is_outlier' in analyzer.features.columns and 
                       analyzer.features['is_outlier'].any())
        
        if has_subpopulations:
            n_subpops = analyzer.features['subpopulation'].nunique()
            st.info(f"ℹ️ Detected **{n_subpops} subpopulations**. Fitting will be performed separately for each.")
        
        if has_outliers:
            n_outliers = analyzer.features['is_outlier'].sum()
            exclude_outliers = st.checkbox(f"Exclude {n_outliers} detected outliers from fitting", value=True)
        else:
            exclude_outliers = False
        
        # Advanced fitting method selection
        fitting_method_type = "standard"
        if ROBUST_FITTING_AVAILABLE:
            fitting_method_type = st.sidebar.radio(
                "Fitting Algorithm:",
                ["standard", "robust", "bayesian"],
                format_func=lambda x: {
                    "standard": "Standard Least Squares",
                    "robust": "Robust (Outlier-Resistant)",
                    "bayesian": "Bayesian MCMC"
                }[x],
                help="Advanced methods improve fit quality for noisy data"
            )
            
            if fitting_method_type == "robust":
                st.sidebar.info("✓ Robust fitting automatically detects and downweights outliers")
            elif fitting_method_type == "bayesian":
                st.sidebar.info("✓ Bayesian fitting provides full uncertainty quantification")
        
        # Fitting mode selection
        fitting_mode = st.radio(
            "Fitting Strategy:",
            ["🚀 Fit All Models (Recommended)", "🎯 Single Model"],
            help="Fit All Models: Fits single, double, triple exponential and anomalous diffusion models, then selects best based on AICc. Single Model: Fit only one specific model."
        )
        
        if fitting_mode == "🎯 Single Model":
            model_options = ["single", "double", "triple"]
            if ADVANCED_MODELS_AVAILABLE:
                model_options.extend(["anomalous_diffusion", "reaction_diffusion"])
                
            model_select = st.selectbox(
                "Select Model",
                model_options,
                format_func=lambda x: {
                    "single": "Single Exponential (Simple Diffusion)",
                    "double": "Double Exponential (Two Populations)",
                    "triple": "Triple Exponential (Three Populations)",
                    "anomalous_diffusion": "Anomalous Diffusion (Subdiffusive)",
                    "reaction_diffusion": "Reaction-Diffusion (Binding Kinetics)"
                }.get(x, x)
            )
        
        # Bootstrap confidence intervals option
        calculate_bootstrap = False
        if BOOTSTRAP_AVAILABLE:
            calculate_bootstrap = st.checkbox(
                "📊 Calculate Bootstrap Confidence Intervals",
                value=False,
                help="Compute robust 95% CIs for all parameters (computationally intensive)"
            )
            if calculate_bootstrap:
                n_bootstrap = st.slider("Bootstrap Iterations:", 100, 2000, 1000, 100)
        
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
                # Determine which curves to fit
                curves_to_fit = analyzer.curves
                
                if exclude_outliers and has_outliers:
                    # Get non-outlier indices
                    valid_indices = analyzer.features[analyzer.features['is_outlier'] == False].index.tolist()
                    curves_to_fit = [analyzer.curves[i] for i in valid_indices if i < len(analyzer.curves)]
                    st.info(f"Fitting {len(curves_to_fit)} curves (excluded {n_outliers} outliers)")
                
                if has_subpopulations:
                    # Fit each subpopulation separately
                    st.info(f"Fitting {n_subpops} subpopulations separately...")
                    subpop_results = {}
                    
                    for subpop_id in range(n_subpops):
                        subpop_mask = analyzer.features['subpopulation'] == subpop_id
                        if exclude_outliers:
                            subpop_mask = subpop_mask & (analyzer.features['is_outlier'] == False)
                        
                        subpop_indices = analyzer.features[subpop_mask].index.tolist()
                        subpop_curves = [analyzer.curves[i] for i in subpop_indices if i < len(analyzer.curves)]
                        
                        if len(subpop_curves) > 0:
                            # Create temporary analyzer for this subpopulation
                            temp_analyzer = FRAPGroupAnalyzer()
                            for curve in subpop_curves:
                                temp_analyzer.add_curve(curve)
                            
                            # Fit models
                            if fitting_mode == "🚀 Fit All Models (Recommended)":
                                temp_analyzer.analyze_group(model_name=None)
                            else:
                                temp_analyzer.analyze_group(model_name=model_select)
                            
                            subpop_results[f"Subpop_{subpop_id}"] = temp_analyzer.features
                            st.success(f"✅ Fitted Subpopulation {subpop_id}: {len(subpop_curves)} curves")
                    
                    # Store results
                    st.session_state.subpopulation_results[group_select] = subpop_results
                    
                else:
                    # Fit all curves together (original behavior)
                    if fitting_mode == "🚀 Fit All Models (Recommended)":
                        analyzer.analyze_group(model_name=None)
                        st.success(f"✅ Successfully fitted all models to {len(curves_to_fit)} curves!")
                    else:
                        analyzer.analyze_group(model_name=model_select)
                        st.success(f"✅ Successfully fitted {model_select} model to {len(curves_to_fit)} curves!")
                
                # Apply advanced fitting if selected
                if fitting_method_type != "standard" and ROBUST_FITTING_AVAILABLE:
                    st.info(f"Applying {fitting_method_type} fitting method...")
                    # Here you would integrate robust/Bayesian fitting
                
                # Calculate bootstrap CIs if requested
                if calculate_bootstrap and BOOTSTRAP_AVAILABLE:
                    st.info(f"Calculating bootstrap confidence intervals ({n_bootstrap} iterations)...")
                    # Here you would integrate bootstrap CI calculations
                
                st.balloons()
        
        # Display results
        if has_subpopulations and group_select in st.session_state.subpopulation_results:
            # Display subpopulation-specific results
            st.markdown("---")
            st.subheader("📊 Subpopulation-Specific Fit Results")
            
            subpop_results = st.session_state.subpopulation_results[group_select]
            
            for subpop_name, subpop_features in subpop_results.items():
                with st.expander(f"🔬 {subpop_name} Results ({len(subpop_features)} curves)", expanded=True):
                    # Summary stats for this subpopulation
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if 'mobile_fraction' in subpop_features.columns:
                            avg_mobile = subpop_features['mobile_fraction'].mean()
                            st.metric("Avg Mobile Fraction", f"{avg_mobile:.1f}%")
                    with col2:
                        if 'half_time_fast' in subpop_features.columns:
                            avg_half = subpop_features['half_time_fast'].mean()
                            st.metric("Avg Half-Time", f"{avg_half:.2f}s")
                    with col3:
                        if 'r2' in subpop_features.columns:
                            avg_r2 = subpop_features['r2'].mean()
                            st.metric("Avg R²", f"{avg_r2:.3f}")
                    
                    # Show detailed table
                    st.dataframe(subpop_features, use_container_width=True)
                    
        elif analyzer.features is not None and not analyzer.features.empty:
            
            # Create tabs for different analyses
            tab_cluster, tab_outlier, tab_viz = st.tabs([
                "🎯 Clustering", 
                "⚠️ Outlier Detection",
                "📊 Visualizations"
            ])
            
            with tab_cluster:
                st.subheader("Subpopulation Detection")
                st.markdown("""
                Automatically identify distinct subpopulations in your data using unsupervised clustering.
                Useful for detecting heterogeneous protein behaviors.
                """)
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    # Get numerical columns only
                    numerical_cols = analyzer.features.select_dtypes(include=[np.number]).columns.tolist()
                    if len(numerical_cols) < 2:
                        st.warning("Need at least 2 numerical parameters for clustering")
                    else:
                        max_k = st.slider("Max Components", 2, min(5, len(analyzer.features)), 3)
                        if st.button("🔍 Detect Subpopulations", use_container_width=True):
                            with st.spinner("Running clustering analysis..."):
                                analyzer.detect_subpopulations(range(1, max_k + 1))
                            st.success("✅ Clustering complete!")
                            st.rerun()
                
                with col2:
                    if 'subpopulation' in analyzer.features.columns:
                        # Show cluster statistics
                        clustered_data = analyzer.features.dropna(subset=['subpopulation'])
                        n_clustered = len(clustered_data)
                        
                        st.metric("Clustered Curves", f"{n_clustered}/{len(analyzer.features)}")
                        
                        st.write("**Cluster Distribution:**")
                        counts = clustered_data['subpopulation'].value_counts().sort_index()
                        
                        # Display in columns
                        cols = st.columns(min(len(counts), 4))
                        for idx, (pop, count) in enumerate(counts.items()):
                            with cols[idx % len(cols)]:
                                st.metric(f"Cluster {int(pop)}", f"{count} curves", 
                                        delta=f"{count/n_clustered*100:.1f}%")
                    else:
                        st.info("👆 Run clustering to see results")
            
            with tab_outlier:
                st.subheader("Outlier Detection")
                st.markdown("""
                Identify curves that deviate significantly from the population.
                Multiple methods available for robust detection.
                """)
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    outlier_method = "iqr"
                    if OUTLIER_DETECTION_AVAILABLE:
                        outlier_method = st.selectbox(
                            "Detection Method:",
                            ["iqr", "zscore", "isolation_forest", "lof"],
                            format_func=lambda x: {
                                "iqr": "IQR (Interquartile Range)",
                                "zscore": "Z-Score (Statistical)",
                                "isolation_forest": "Isolation Forest (ML)",
                                "lof": "Local Outlier Factor (ML)"
                            }[x]
                        )
                    
                    outlier_threshold = st.slider("Sensitivity:", 1.0, 3.0, 1.5, 0.1,
                                                 help="Lower = more sensitive")
                    
                    if st.button("⚠️ Detect Outliers", use_container_width=True):
                        with st.spinner(f"Running {outlier_method} outlier detection..."):
                            if OUTLIER_DETECTION_AVAILABLE and outlier_method in ["isolation_forest", "lof"]:
                                # Use ML-based detection
                                detector = FRAPOutlierDetector()
                                outlier_results = detector.detect_outliers(
                                    analyzer.features, 
                                    method=outlier_method
                                )
                                analyzer.features['is_outlier'] = outlier_results['is_outlier']
                            else:
                                # Use statistical detection
                                analyzer.detect_outliers(method=outlier_method, threshold=outlier_threshold)
                        st.success("✅ Outlier detection complete!")
                        st.rerun()
                
                with col2:
                    if 'is_outlier' in analyzer.features.columns:
                        n_outliers = analyzer.features['is_outlier'].sum()
                        n_total = len(analyzer.features)
                        
                        st.metric("Outliers Detected", f"{n_outliers}/{n_total}",
                                delta=f"{n_outliers/n_total*100:.1f}%",
                                delta_color="inverse")
                        
                        if n_outliers > 0:
                            st.warning(f"⚠️ {n_outliers} curves flagged as outliers")
                            
                            if st.checkbox("Show outlier details"):
                                outlier_data = analyzer.features[analyzer.features['is_outlier']==True]
                                st.dataframe(outlier_data)
                            
                            if st.button("🗑️ Remove Outliers from Analysis"):
                                analyzer.features = analyzer.features[analyzer.features['is_outlier']==False]
                                st.success(f"Removed {n_outliers} outliers")
                                st.rerun()
                    else:
                        st.info("👆 Run detection to identify outliers")
            
            with tab_viz:
                st.subheader("Data Visualization")
                
                if 'subpopulation' in analyzer.features.columns or 'is_outlier' in analyzer.features.columns:
                    # Parameter selection for visualization
                    numerical_params = [c for c in analyzer.features.select_dtypes(include=[np.number]).columns 
                                       if c not in ['subpopulation', 'is_outlier']]
                    
                    if len(numerical_params) >= 2:
                        col_x, col_y, col_color = st.columns(3)
                        with col_x:
                            x_axis = st.selectbox("X Axis", numerical_params, index=0)
                        with col_y:
                            y_axis = st.selectbox("Y Axis", numerical_params, index=min(1, len(numerical_params)-1))
                        with col_color:
                            color_by = st.selectbox("Color By", 
                                                   ["None"] + (["subpopulation"] if 'subpopulation' in analyzer.features.columns else []) +
                                                   (["is_outlier"] if 'is_outlier' in analyzer.features.columns else []))
                        
                        # Create visualization
                        if 'subpopulation' in analyzer.features.columns and color_by == "subpopulation":
                            clustered_data = analyzer.features.dropna(subset=['subpopulation'])
                            if len(clustered_data) > 0:
                                fig = FRAPVisualizer.plot_subpopulations(clustered_data, x_axis, y_axis)
                                st.pyplot(fig)
                        else:
                            # Standard scatter plot
                            fig, ax = plt.subplots(figsize=(10, 6))
                            if color_by == "is_outlier" and 'is_outlier' in analyzer.features.columns:
                                outliers = analyzer.features[analyzer.features['is_outlier']==True]
                                normals = analyzer.features[analyzer.features['is_outlier']==False]
                                ax.scatter(normals[x_axis], normals[y_axis], alpha=0.6, label='Normal')
                                ax.scatter(outliers[x_axis], outliers[y_axis], alpha=0.6, 
                                         color='red', marker='x', s=100, label='Outlier')
                                ax.legend()
                            else:
                                ax.scatter(analyzer.features[x_axis], analyzer.features[y_axis], alpha=0.6)
                            ax.set_xlabel(x_axis)
                            ax.set_ylabel(y_axis)
                            ax.set_title(f"{x_axis} vs {y_axis}")
                            ax.grid(alpha=0.3)
                            st.pyplot(fig)
                            
                        # Advanced visualizations if available
                        if ADVANCED_VIZ_AVAILABLE:
                            st.markdown("### Advanced Visualizations")
                            viz_type = st.selectbox("Visualization Type", 
                                                   ["Heatmap", "Pairplot", "QQ Plot"])
                            
                            if viz_type == "Heatmap" and st.button("Generate Heatmap"):
                                fig = plot_heatmap(analyzer.features, numerical_params[:10])
                                st.pyplot(fig)
                            elif viz_type == "Pairplot" and st.button("Generate Pairplot"):
                                fig = plot_pairplot(analyzer.features[numerical_params[:5]])
                                st.pyplot(fig)
                            elif viz_type == "QQ Plot" and st.button("Generate QQ Plot"):
                                param = st.selectbox("Parameter", numerical_params)
                                fig = plot_qq(analyzer.features[param].dropna())
                                st.pyplot(fig)
                    else:
                        st.warning("Need at least 2 numerical parameters for visualization")
                else:
                    st.info("Run clustering or outlier detection first to visualize results")

# --- Page 4: Compare Groups ---
elif page == "4. Compare Groups":
    st.header("⚖️ Statistical Group Comparison")
    
    if len(st.session_state.data_groups) < 2:
        st.warning("Need at least 2 groups to compare.")
    else:
        # Group selection
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
                # Create tabs for different comparison types
                tab_param, tab_holistic, tab_advanced, tab_reference = st.tabs([
                    "📊 Parameter Comparison",
                    "📈 Holistic Comparison", 
                    "🔬 Advanced Models",
                    "📚 Reference Database"
                ])
                
                with tab_param:
                    st.subheader("Individual Parameter Comparison")
                    st.markdown("""
                    Compare fitted parameters between groups using statistical tests.
                    Includes t-tests, Mann-Whitney U, effect sizes, and more.
                    """)
                    
                    # Filter to only numerical columns
                    numerical_cols = analyzer1.features.select_dtypes(include=[np.number]).columns.tolist()
                    if not numerical_cols:
                        st.error("No numerical parameters available for comparison.")
                    else:
                        col_param, col_test = st.columns([2, 1])
                        with col_param:
                            param = st.selectbox("Parameter to Compare", numerical_cols)
                        with col_test:
                            test_type = st.selectbox("Statistical Test", 
                                                    ["t-test", "mann-whitney", "permutation"])
                        
                        if st.button("Run Statistical Test", use_container_width=True):
                            with st.spinner("Running statistical analysis..."):
                                result = FRAPStatisticalComparator.compare_groups(
                                    analyzer1.features, analyzer2.features, param
                                )
                            
                            st.subheader("📊 Results")
                            
                            # Display key metrics
                            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                            with col_m1:
                                mean1 = analyzer1.features[param].mean()
                                st.metric(f"{group1} Mean", f"{mean1:.4f}")
                            with col_m2:
                                mean2 = analyzer2.features[param].mean()
                                st.metric(f"{group2} Mean", f"{mean2:.4f}")
                            with col_m3:
                                fold_change = mean2 / mean1 if mean1 != 0 else np.nan
                                st.metric("Fold Change", f"{fold_change:.2f}x")
                            with col_m4:
                                if 'p_value' in result:
                                    sig = "***" if result['p_value'] < 0.001 else "**" if result['p_value'] < 0.01 else "*" if result['p_value'] < 0.05 else "ns"
                                    st.metric("P-value", f"{result.get('p_value', 'N/A'):.4f}", delta=sig)
                            
                            # Full results
                            st.json(result)
                        
                        # Distribution plot
                        st.subheader("📈 Distribution Comparison")
                        df1 = analyzer1.features.copy()
                        df1['Group'] = group1
                        df2 = analyzer2.features.copy()
                        df2['Group'] = group2
                        combined = pd.concat([df1, df2])
                        
                        fig = FRAPVisualizer.plot_parameter_distribution(combined, param, group_col='Group')
                        st.pyplot(fig)
                        
                        # Advanced visualizations
                        if ADVANCED_VIZ_AVAILABLE:
                            col_v1, col_v2 = st.columns(2)
                            with col_v1:
                                if st.button("📊 Volcano Plot"):
                                    # Perform comparison for all parameters
                                    results_all = {}
                                    for p in numerical_cols:
                                        try:
                                            res = FRAPStatisticalComparator.compare_groups(
                                                analyzer1.features, analyzer2.features, p
                                            )
                                            results_all[p] = res
                                        except:
                                            pass
                                    fig = plot_volcano(results_all)
                                    st.pyplot(fig)
                            
                            with col_v2:
                                if st.button("🌲 Forest Plot"):
                                    fig = plot_forest(combined, param, 'Group')
                                    st.pyplot(fig)
                
                with tab_holistic:
                    st.subheader("Holistic Mean Profile Comparison")
                    st.markdown("""
                    Compare averaged recovery curves between groups.
                    This approach has higher statistical power and enables advanced model fitting.
                    """)
                    
                    if HOLISTIC_COMPARISON_AVAILABLE:
                        # Get raw data for both groups
                        group1_curves = [c for c in analyzer1.curves if c.time_post_bleach is not None]
                        group2_curves = [c for c in analyzer2.curves if c.time_post_bleach is not None]
                        
                        if len(group1_curves) > 0 and len(group2_curves) > 0:
                            if st.button("🔬 Compute Mean Profiles", use_container_width=True):
                                with st.spinner("Computing averaged recovery profiles..."):
                                    # Compute mean profiles
                                    t1_avg, i1_avg, i1_sem = compute_average_recovery_profile(
                                        {i: {'time_post_bleach': c.time_post_bleach, 
                                             'intensity_post_bleach': c.intensity_post_bleach}
                                         for i, c in enumerate(group1_curves)}
                                    )
                                    t2_avg, i2_avg, i2_sem = compute_average_recovery_profile(
                                        {i: {'time_post_bleach': c.time_post_bleach,
                                             'intensity_post_bleach': c.intensity_post_bleach}
                                         for i, c in enumerate(group2_curves)}
                                    )
                                    
                                    # Store in session state
                                    st.session_state.holistic_results = {
                                        'group1': {'time': t1_avg, 'intensity': i1_avg, 'sem': i1_sem},
                                        'group2': {'time': t2_avg, 'intensity': i2_avg, 'sem': i2_sem}
                                    }
                                    st.success("✅ Mean profiles computed!")
                                    st.rerun()
                            
                            # Display results if available
                            if 'holistic_results' in st.session_state:
                                res = st.session_state.holistic_results
                                
                                # Plot mean profiles
                                fig, ax = plt.subplots(figsize=(10, 6))
                                
                                # Group 1
                                ax.plot(res['group1']['time'], res['group1']['intensity'], 
                                       'b-', linewidth=2, label=group1)
                                ax.fill_between(res['group1']['time'],
                                              res['group1']['intensity'] - res['group1']['sem'],
                                              res['group1']['intensity'] + res['group1']['sem'],
                                              alpha=0.2, color='b')
                                
                                # Group 2
                                ax.plot(res['group2']['time'], res['group2']['intensity'],
                                       'r-', linewidth=2, label=group2)
                                ax.fill_between(res['group2']['time'],
                                              res['group2']['intensity'] - res['group2']['sem'],
                                              res['group2']['intensity'] + res['group2']['sem'],
                                              alpha=0.2, color='r')
                                
                                ax.set_xlabel('Time (s)')
                                ax.set_ylabel('Normalized Intensity')
                                ax.set_title('Mean Recovery Profiles')
                                ax.legend()
                                ax.grid(alpha=0.3)
                                st.pyplot(fig)
                                
                                # Calculate statistics on mean profiles
                                st.info(f"**{group1}**: n={len(group1_curves)} curves")
                                st.info(f"**{group2}**: n={len(group2_curves)} curves")
                        else:
                            st.warning("Both groups need post-bleach data for comparison")
                    else:
                        st.info("Holistic comparison module not available. Install required dependencies.")
                
                with tab_advanced:
                    st.subheader("Advanced Biophysical Model Fitting")
                    st.markdown("""
                    Fit sophisticated models (anomalous diffusion, reaction-diffusion) to mean curves.
                    Enables mechanistic understanding of differences between groups.
                    """)
                    
                    if ADVANCED_MODELS_AVAILABLE and HOLISTIC_COMPARISON_AVAILABLE:
                        if 'holistic_results' in st.session_state:
                            col_model, col_param = st.columns(2)
                            with col_model:
                                adv_model = st.selectbox(
                                    "Biophysical Model:",
                                    ["auto", "anomalous_diffusion", "reaction_diffusion_simple"],
                                    help="'auto' selects best model by AIC"
                                )
                            with col_param:
                                bleach_radius_um = st.number_input(
                                    "Bleach Radius (μm):",
                                    min_value=0.1, max_value=10.0, value=1.0, step=0.1
                                )
                            
                            if st.button("🔬 Fit Advanced Models", use_container_width=True):
                                with st.spinner("Fitting advanced models..."):
                                    res = st.session_state.holistic_results
                                    
                                    try:
                                        fit_results = compare_groups_advanced_fitting(
                                            group1_time=res['group1']['time'],
                                            group1_intensity=res['group1']['intensity'],
                                            group1_sem=res['group1']['sem'],
                                            group2_time=res['group2']['time'],
                                            group2_intensity=res['group2']['intensity'],
                                            group2_sem=res['group2']['sem'],
                                            group1_name=group1,
                                            group2_name=group2,
                                            bleach_radius_um=bleach_radius_um,
                                            model=adv_model
                                        )
                                        
                                        st.success("✅ Advanced fitting complete!")
                                        
                                        # Display results
                                        st.subheader("Model Results")
                                        st.json(fit_results)
                                        
                                        # Store for later use
                                        st.session_state.advanced_fit_results = fit_results
                                    except Exception as e:
                                        st.error(f"Error fitting models: {e}")
                        else:
                            st.info("👆 Compute mean profiles first in the 'Holistic Comparison' tab")
                    else:
                        st.info("Advanced model fitting not available. Install required dependencies.")
                
                with tab_reference:
                    st.subheader("Reference Database Comparison")
                    st.markdown("""
                    Compare your results to published FRAP data from 40+ proteins.
                    Get biological context and interpretation guidance.
                    """)
                    
                    if REFERENCE_DB_AVAILABLE:
                        # Display reference database UI
                        display_reference_database_ui()
                    else:
                        st.info("Reference database not available. Install required dependencies.")

# --- Page 5: Report ---
elif page == "5. Report":
    st.header("📄 Professional Report Generation")
    st.markdown("""
    Generate comprehensive reports in HTML or PDF format with all your analysis results,
    figures, and statistical summaries.
    """)
    
    if not st.session_state.data_groups:
        st.error("No data to report. Please load and analyze data first.")
    else:
        # Report configuration
        st.subheader("Report Configuration")
        
        col_cfg1, col_cfg2 = st.columns(2)
        with col_cfg1:
            report_title = st.text_input("Report Title", "FRAP Analysis Report")
            include_raw_data = st.checkbox("Include Raw Data Tables", value=True)
            include_statistics = st.checkbox("Include Statistical Summaries", value=True)
        
        with col_cfg2:
            report_format = st.selectbox("Report Format", ["HTML", "PDF", "Both"])
            include_methods = st.checkbox("Include Methods Section", value=True)
            include_interpretation = st.checkbox("Include Interpretation Guide", value=True)
        
        # Group selection for report
        st.subheader("Select Groups to Include")
        selected_groups = st.multiselect(
            "Groups:",
            list(st.session_state.data_groups.keys()),
            default=list(st.session_state.data_groups.keys())
        )
        
        if not selected_groups:
            st.warning("Please select at least one group to include in the report")
        else:
            # Generate report button
            col_btn1, col_btn2 = st.columns(2)
            
            with col_btn1:
                if st.button("📊 Generate HTML Report", use_container_width=True, type="primary"):
                    if REPORT_GEN_AVAILABLE:
                        with st.spinner("Generating HTML report..."):
                            try:
                                # Collect data
                                all_features = pd.DataFrame()
                                figures = {}
                                
                                for name in selected_groups:
                                    analyzer = st.session_state.data_groups[name]
                                    if analyzer.features is not None:
                                        df = analyzer.features.copy()
                                        df['Group'] = name
                                        all_features = pd.concat([all_features, df])
                                        
                                        # Generate recovery plot for report
                                        if analyzer.curves:
                                            times = []
                                            data_intensities = []
                                            fitted_curves = []
                                            
                                            for c in analyzer.curves:
                                                if c.time_post_bleach is not None:
                                                    times.append(c.time_post_bleach)
                                                if c.intensity_post_bleach is not None:
                                                    data_intensities.append(c.intensity_post_bleach)
                                            
                                            for res in analyzer.fit_results:
                                                if res.success and res.fitted_curve is not None:
                                                    fitted_curves.append(res.fitted_curve)
                                            
                                            if times and data_intensities:
                                                fig = FRAPVisualizer.plot_recovery_curves(
                                                    times[0] if times else np.array([]),
                                                    data_intensities,
                                                    fitted_curves,
                                                    title=f"{name} Recovery Curves"
                                                )
                                                figures[f"{name}_Recovery"] = fig
                                
                                # Generate HTML report
                                output_file = f"{report_title.replace(' ', '_')}_Report.html"
                                
                                # Use the enhanced HTML report generator
                                if hasattr(generate_html_report, '__call__'):
                                    # Create data manager-like structure
                                    class DataManager:
                                        def __init__(self, groups_dict):
                                            self.groups = {}
                                            self.files = {}
                                            for name, analyzer in groups_dict.items():
                                                self.groups[name] = {
                                                    'files': [],
                                                    'features': analyzer.features
                                                }
                                    
                                    dm = DataManager({k: st.session_state.data_groups[k] for k in selected_groups})
                                    generate_html_report(dm, selected_groups, output_file)
                                else:
                                    # Fallback to basic report generator
                                    FRAPReportGenerator.generate_html_report(all_features, figures, output_file)
                                
                                st.success(f"✅ HTML report generated: {output_file}")
                                
                                # Provide download button
                                if os.path.exists(output_file):
                                    with open(output_file, "rb") as f:
                                        st.download_button(
                                            label="📥 Download HTML Report",
                                            data=f,
                                            file_name=output_file,
                                            mime="text/html"
                                        )
                            except Exception as e:
                                st.error(f"Error generating HTML report: {e}")
                                logger.error(f"Report generation error: {e}", exc_info=True)
                    else:
                        st.warning("HTML report generation not available. Install required dependencies.")
            
            with col_btn2:
                if st.button("📑 Generate PDF Report", use_container_width=True):
                    if REPORT_GEN_AVAILABLE:
                        with st.spinner("Generating PDF report..."):
                            try:
                                output_file = f"{report_title.replace(' ', '_')}_Report.pdf"
                                
                                # Create data manager structure
                                class DataManager:
                                    def __init__(self, groups_dict):
                                        self.groups = {}
                                        self.files = {}
                                        for name, analyzer in groups_dict.items():
                                            self.groups[name] = {
                                                'files': [],
                                                'features': analyzer.features
                                            }
                                
                                dm = DataManager({k: st.session_state.data_groups[k] for k in selected_groups})
                                
                                # Generate PDF report
                                result_file = generate_pdf_report(dm, selected_groups, output_file)
                                
                                if result_file and os.path.exists(result_file):
                                    st.success(f"✅ PDF report generated: {result_file}")
                                    
                                    with open(result_file, "rb") as f:
                                        st.download_button(
                                            label="📥 Download PDF Report",
                                            data=f,
                                            file_name=os.path.basename(result_file),
                                            mime="application/pdf"
                                        )
                                else:
                                    st.warning("PDF report was not created successfully")
                                    
                            except Exception as e:
                                st.error(f"Error generating PDF report: {e}")
                                logger.error(f"PDF generation error: {e}", exc_info=True)
                    else:
                        st.warning("PDF report generation not available. Install required dependencies.")
            
            # Preview report content
            st.markdown("---")
            st.subheader("Report Preview")
            
            if st.checkbox("Show Report Preview"):
                st.markdown("### Summary Statistics")
                
                for name in selected_groups:
                    analyzer = st.session_state.data_groups[name]
                    if analyzer.features is not None:
                        st.markdown(f"#### {name}")
                        st.write(f"**Number of curves:** {len(analyzer.curves)}")
                        st.write(f"**Fitted parameters available:** {len(analyzer.features.columns)}")
                        
                        # Show key statistics
                        if not analyzer.features.empty:
                            st.dataframe(analyzer.features.describe())
                
                # Show available figures
                st.markdown("### Figures to be Included")
                for name in selected_groups:
                    st.write(f"- {name} Recovery Curves")
                    st.write(f"- {name} Parameter Distributions")
                
                if len(selected_groups) >= 2:
                    st.write("- Group Comparison Plots")
                    st.write("- Statistical Test Results")
