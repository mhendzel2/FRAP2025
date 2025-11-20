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
if 'processed_zip_files' not in st.session_state:
    st.session_state.processed_zip_files = set()

# --- Sidebar Navigation ---
st.sidebar.title("FRAP Analysis 2.0")
page = st.sidebar.radio("Workflow", ["1. Import & Preprocess", "2. Model Fitting", "3. Subpopulations", "4. Compare Groups", "5. Report"])

# --- Helper Functions ---
def load_and_process_file(uploaded_file, bleach_frame_idx):
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

def load_groups_from_zip(zip_file, bleach_frame_idx):
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
                        curve_data = FRAPInputHandler.double_normalization(curve_data, bleach_frame_idx)
                        curve_data = FRAPInputHandler.time_zero_correction(curve_data, bleach_frame_idx)
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
        bleach_frame_zip = st.number_input("Bleach Frame Index (for all files)", min_value=1, value=10, 
                                          help="Index of the frame where bleaching occurs (0-based)", key="bleach_zip")
        
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
            bleach_frame = st.number_input("Bleach Frame Index", min_value=1, value=10, 
                                          help="Index of the frame where bleaching occurs (0-based)")
            
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
                            bleach_idx = min(10, len(curve.time)-1)  # Safe bleach index for preview
                            if bleach_idx < len(curve.time):
                                ax.axvline(curve.time[bleach_idx], color='r', linestyle='--', label='Bleach')
                            ax.set_title(f"Preview: First Curve in {name}")
                            ax.legend()
                            st.pyplot(fig)
            else:
                st.info("No data loaded yet.")

# --- Page 2: Model Fitting ---
elif page == "2. Model Fitting":
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
                ["single", "double", "triple", "anomalous_diffusion"],
                format_func=lambda x: {
                    "single": "Single Exponential (Simple Diffusion)",
                    "double": "Double Exponential (Two Populations)",
                    "triple": "Triple Exponential (Three Populations)",
                    "anomalous_diffusion": "Anomalous Diffusion (Subdiffusive)"
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
                    st.success(f"✅ Successfully fitted all models to {len(analyzer.curves)} curves!")
                else:
                    # Fit single selected model
                    analyzer.analyze_group(model_name=model_select)
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

# --- Page 3: Subpopulations ---
elif page == "3. Subpopulations":
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
