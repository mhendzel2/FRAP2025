"""
FRAP Analysis App - Final Verified Version
A comprehensive FRAP analysis application with supervised outlier removal, sequential group plots,
a detailed kinetics table showing proportions relative to both the mobile pool and the total population,
and all original settings functionality restored.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import io
from math import log
from scipy.optimize import curve_fit
from scipy.ndimage import minimum_position
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, Optional, Tuple, List
import logging
from PIL import Image
from frap_pdf_reports import generate_pdf_report
from frap_image_analysis import FRAPImageAnalyzer, create_image_analysis_interface
from frap_core_corrected import FRAPAnalysisCore as CoreFRAPAnalysis
import zipfile
import tempfile
import shutil
# --- Page and Logging Configuration ---
st.set_page_config(page_title="FRAP Analysis", page_icon="🔬", layout="wide", initial_sidebar_state="expanded")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
os.makedirs('data', exist_ok=True)

def validate_frap_data(df: pd.DataFrame, file_path: str = "") -> bool:
    """
    Validate the structure and quality of FRAP data.
    
    Args:
        df: DataFrame to validate
        file_path: Path to the file (for error reporting)
        
    Returns:
        bool: True if data is valid, False otherwise
    """
    try:
        # Check required columns
        required_cols = ['time', 'ROI1', 'ROI2', 'ROI3']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns {missing_cols} in {file_path}")
            return False
        
        # Check data types
        for col in required_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                logger.error(f"Column '{col}' is not numeric in {file_path}")
                return False
        
        # Check for sufficient data points
        if len(df) < 10:
            logger.error(f"Insufficient data points ({len(df)}) in {file_path}")
            return False
        
        # Check for monotonic time
        if not df['time'].is_monotonic_increasing:
            logger.warning(f"Time column is not monotonic in {file_path}")
        
        # Check for negative intensities (should be non-negative)
        for col in ['ROI1', 'ROI2', 'ROI3']:
            if (df[col] < 0).any():
                logger.warning(f"Negative intensities found in column '{col}' in {file_path}")
        
        # Check for NaN values
        try:
            has_nan = df[required_cols].isnull().sum().sum() > 0
            if has_nan:
                logger.warning(f"NaN values found in data from {file_path}")
        except Exception:
            logger.debug(f"Could not check for NaN values in {file_path}")
        
        logger.info(f"Data validation passed for {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Data validation failed for {file_path}: {e}")
        return False

def gaussian_2d(xy: Tuple[np.ndarray, np.ndarray], A: float, x0: float, y0: float, 
                sigma_x: float, sigma_y: float, theta: float, offset: float) -> np.ndarray:
    """A 2D Gaussian function for PSF fitting."""
    x, y = xy
    x0 = float(x0)
    y0 = float(y0)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + A * np.exp( - (a*((x-x0)**2) + 2*b*(x-x0)*(y-y0) + c*((y-y0)**2)))
    return g.ravel()

def fit_psf(image_data: np.ndarray) -> Dict[str, float]:
    """
    Fits a 2D Gaussian to image data to find the PSF.
    
    Parameters:
    -----------
    image_data : numpy.ndarray
        A 2D array representing the cropped image of a fluorescent bead.
        
    Returns:
    --------
    dict
        A dictionary containing the fitted parameters (sigma_x, sigma_y).
    """
    # Create x and y coordinate arrays
    x = np.arange(image_data.shape[1])
    y = np.arange(image_data.shape[0])
    x, y = np.meshgrid(x, y)

    # Initial guesses for the parameters
    initial_guess = (
        np.max(image_data),
        image_data.shape[1]/2,
        image_data.shape[0]/2,
        2.0,
        2.0,
        0,
        np.min(image_data)
    )

    try:
        # The curve_fit function requires the data to be flattened
        popt, _ = curve_fit(gaussian_2d, (x.ravel(), y.ravel()), image_data.ravel(), p0=initial_guess)
        
        # Extract the parameters
        A, x0, y0, sigma_x, sigma_y, theta, offset = popt
        
        return {
            'sigma_x': abs(sigma_x), 
            'sigma_y': abs(sigma_y),
            'amplitude': A,
            'center_x': x0,
            'center_y': y0,
            'theta': theta,
            'offset': offset
        }
    except Exception as e:
        logger.error(f"PSF fitting failed: {e}")
        return {'sigma_x': 2.0, 'sigma_y': 2.0}

def track_bleach_center(image_stack: np.ndarray, bleach_frame_index: int, 
                       search_radius: int = 5) -> List[Tuple[int, int]]:
    """
    Tracks the center of the photobleached region across an image stack.
    
    Parameters:
    -----------
    image_stack : numpy.ndarray
        The 3D FRAP data (time, y, x).
    bleach_frame_index : int
        The index of the frame immediately after photobleaching.
    search_radius : int
        The radius (in pixels) to search for the minimum in subsequent frames.
        
    Returns:
    --------
    list
        A list of (y, x) coordinates of the bleach center for each frame post-bleach.
    """
    post_bleach_stack = image_stack[bleach_frame_index:]
    centers = []

    # Find the initial center in the first post-bleach frame
    initial_center = tuple(minimum_position(post_bleach_stack[0]))
    centers.append(initial_center)

    last_center = initial_center
    for i in range(1, len(post_bleach_stack)):
        frame = post_bleach_stack[i]
        
        # Define a search area around the last known center
        y_min = max(0, int(last_center[0]) - search_radius)
        y_max = min(frame.shape[0], int(last_center[0]) + search_radius)
        x_min = max(0, int(last_center[1]) - search_radius)
        x_max = min(frame.shape[1], int(last_center[1]) + search_radius)
        
        search_area = frame[y_min:y_max, x_min:x_max]
        
        # Find the minimum in the search area
        local_min_pos = tuple(minimum_position(search_area))
        
        # Convert back to global frame coordinates
        current_center = (int(local_min_pos[0]) + y_min, int(local_min_pos[1]) + x_min)
        centers.append(current_center)
        last_center = current_center
        
    return centers

def interpret_kinetics(k, bleach_radius_um, gfp_d=25.0, gfp_rg=2.82, gfp_mw=27.0):
    """
    Centralized kinetics interpretation function with CORRECTED mathematics
    """
    if k <= 0:
        return {
            'k_off': k,
            'diffusion_coefficient': np.nan,
            'apparent_mw': np.nan,
            'half_time_diffusion': np.nan,
            'half_time_binding': np.nan
        }

    # 1. Interpretation as a binding/unbinding process
    k_off = k  # The rate is the dissociation constant
    half_time_binding = np.log(2) / k  # Time for 50% recovery via binding

    # 2. Interpretation as a diffusion process
    # CORRECTED FORMULA: For 2D diffusion: D = (w^2 * k) / 4 
    # where w is bleach radius and k is rate constant
    # This is the mathematically correct formula WITHOUT the erroneous np.log(2) factor
    diffusion_coefficient = (bleach_radius_um**2 * k) / 4.0  # CORRECTED: removed ln(2)
    half_time_diffusion = np.log(2) / k  # Half-time from rate constant

    # Estimate apparent molecular weight from diffusion coefficient
    # Using Stokes-Einstein relation: D ∝ 1/Rg ∝ 1/MW^(1/3)
    if diffusion_coefficient > 0:
        apparent_mw = gfp_mw * (gfp_d / diffusion_coefficient)**3
    else:
        apparent_mw = np.nan

    return {
        'k_off': k_off,
        'diffusion_coefficient': diffusion_coefficient,
        'apparent_mw': apparent_mw,
        'half_time_diffusion': half_time_diffusion,
        'half_time_binding': half_time_binding
    }

def generate_markdown_report(group_name, settings, summary_df, detailed_df, excluded_count, total_count):
    """
    Generates a comprehensive Markdown report from the analysis results.
    """
    from datetime import datetime
    
    # Calculate summary statistics
    included_data = detailed_df[detailed_df['Status'] == 'Included']
    
    report_str = f"""# FRAP Analysis Report

**Group Name:** {group_name}  
**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Analysis Software:** Enhanced FRAP Analysis Platform  

---

## Executive Summary

- **Total Files Analyzed:** {total_count}
- **Files Included in Analysis:** {total_count - excluded_count}
- **Outliers Excluded:** {excluded_count}
- **Primary Analysis:** Dual-interpretation kinetics (diffusion vs. binding)

---

## Analysis Settings

| Parameter | Value |
|-----------|-------|
| Model Selection Criterion | {settings.get('default_criterion', 'AIC').upper()} |
| Outlier Detection Method | IQR-based filtering |
| Bleach Radius | {settings.get('default_bleach_radius', 1.0):.2f} pixels |
| Pixel Size | {settings.get('default_pixel_size', 1.0):.3f} μm/pixel |
| Effective Bleach Size | {settings.get('default_bleach_radius', 1.0) * settings.get('default_pixel_size', 1.0):.3f} μm |
| Reference Protein | GFP (D = {settings.get('default_gfp_diffusion', 25.0):.1f} μm²/s, MW = 27 kDa) |

---

## Population-Level Results

### Averaged Kinetic Parameters
"""

    # Add the summary dataframe
    if not summary_df.empty:
        report_str += "\n" + summary_df.to_markdown(index=True) + "\n"
    
    report_str += """
### Biological Interpretation Summary

The kinetic rates can be interpreted in two ways:

**Diffusion-Limited Recovery:**
- Assumes fluorescence recovery is limited by molecular diffusion
- Provides apparent diffusion coefficients and molecular weight estimates
- Useful for comparing mobility between conditions

**Binding-Limited Recovery:**
- Assumes recovery is limited by molecular binding/unbinding kinetics
- Provides dissociation rate constants (k_off) and residence times
- Useful for studying protein-chromatin interactions

---

## Individual File Analysis

### Summary Statistics (Included Files Only)
"""

    if len(included_data) > 0:
        # Calculate summary statistics
        numeric_cols = ['Mobile (%)', 'Primary Rate (k)', 'App. D (μm²/s)', 'App. MW (kDa)', 'R²']
        stats_summary = included_data[numeric_cols].describe()
        
        report_str += "\n" + stats_summary.round(3).to_markdown() + "\n"
        
        # Add variability analysis
        report_str += f"""
### Data Quality Assessment

- **Mobile Fraction CV:** {(included_data['Mobile (%)'].std() / included_data['Mobile (%)'].mean() * 100):.1f}%
- **Rate Constant CV:** {(included_data['Primary Rate (k)'].std() / included_data['Primary Rate (k)'].mean() * 100):.1f}%
- **Average R²:** {included_data['R²'].mean():.3f} ± {included_data['R²'].std():.3f}
- **Data Consistency:** {'Excellent' if included_data['Primary Rate (k)'].std() / included_data['Primary Rate (k)'].mean() < 0.3 else 'Good' if included_data['Primary Rate (k)'].std() / included_data['Primary Rate (k)'].mean() < 0.5 else 'Variable'}

"""

    report_str += """
### Detailed Results Table

**Legend:** 
- ✅ Included in population analysis
- ❌ Excluded as outlier

| File | Status | Mobile (%) | Rate (k) | k_off (1/s) | App. D (μm²/s) | App. MW (kDa) | Model | R² |
|------|--------|------------|----------|-------------|----------------|---------------|-------|-----|
"""

    # Add detailed results
    for _, row in detailed_df.iterrows():
        status_icon = "✅" if row['Status'] == 'Included' else "❌"
        report_str += f"| {row['File Name']} | {status_icon} | {row['Mobile (%)']:.1f} | {row['Primary Rate (k)']:.4f} | {row['k_off (1/s)']:.4f} | {row['App. D (μm²/s)']:.3f} | {row['App. MW (kDa)']:.1f} | {row['Model']} | {row['R²']:.3f} |\n"

    report_str += """
---

## Experimental Recommendations

### Data Quality
"""
    
    if len(included_data) > 0:
        avg_r2 = included_data['R²'].mean()
        cv_rate = included_data['Primary Rate (k)'].std() / included_data['Primary Rate (k)'].mean()
        
        if avg_r2 > 0.95:
            report_str += "- **Excellent curve fits** (R² > 0.95) indicate high data quality\n"
        elif avg_r2 > 0.90:
            report_str += "- **Good curve fits** (R² > 0.90) suggest reliable data\n"
        else:
            report_str += "- **Consider data quality improvement** - some fits show R² < 0.90\n"
            
        if cv_rate < 0.3:
            report_str += "- **Low variability** in kinetic rates suggests consistent experimental conditions\n"
        else:
            report_str += "- **Higher variability** detected - consider experimental factors affecting recovery rates\n"

    report_str += """
### Biological Insights

1. **Diffusion vs. Binding:** Compare apparent diffusion coefficients with expected values for free diffusion
2. **Molecular Weight Estimates:** Use apparent MW calculations to assess protein complex formation
3. **Binding Kinetics:** Evaluate k_off values in the context of known protein-DNA binding affinities

---

## Methods Summary

**Curve Fitting:** Multi-component exponential models (1, 2, or 3 components) fitted using least-squares optimization with model selection based on AIC/BIC criteria.

**Dual Interpretation:** Each kinetic rate constant k is interpreted as both:
- Diffusion coefficient: D = (w² × k) / 4 (CORRECTED FORMULA)
- Binding dissociation rate: k_off = k

**Quality Control:** Statistical outlier detection using IQR-based filtering to ensure robust population averages.

---

*Report generated by Enhanced FRAP Analysis Platform*
"""

    return report_str

def import_imagej_roi(roi_data: bytes) -> Optional[Dict[str, Any]]:
    """
    Imports ROI data from ImageJ .roi file bytes.
    
    Parameters:
    -----------
    roi_data : bytes
        The .roi file data as bytes.
        
    Returns:
    --------
    dict or None
        A dictionary containing ROI information, or None if import fails.
    """
    try:
        # This is a simplified parser - in practice you'd use the roifile library
        # For now, return a placeholder structure
        return {
            'name': 'imported_roi',
            'type': 'rectangle',
            'coordinates': {'x': 0, 'y': 0, 'width': 50, 'height': 50}
        }
    except Exception as e:
        logger.error(f"ROI import failed: {e}")
        return None

# --- Session State Initialization ---
if 'settings' not in st.session_state:
    st.session_state.settings = {
        'default_criterion': 'aic', 'default_bleach_radius': 1.0, 'default_pixel_size': 0.3,
        'default_gfp_diffusion': 25.0, 'default_gfp_rg': 2.82, 'default_gfp_mw': 27.0,
        'default_scaling_alpha': 1.0, 'default_target_mw': 27.0, 'decimal_places': 2
    }
if "data_manager" not in st.session_state:
    st.session_state.data_manager = None
if 'selected_group_name' not in st.session_state:
    st.session_state.selected_group_name = None

# --- Plotting Helper Functions ---
def plot_all_curves(group_files_data):
    fig = go.Figure()
    for file_data in group_files_data.values():
        fig.add_trace(go.Scatter(x=file_data['time'], y=file_data['intensity'], mode='lines', name=file_data['name']))
    fig.update_layout(title="All Individual Recovery Curves", xaxis_title="Time (s)", yaxis_title="Normalized Intensity", legend_title="File")
    return fig

def plot_average_curve(group_files_data):
    if not group_files_data:
        return go.Figure()
    all_times = np.concatenate([d['time'] for d in group_files_data.values() if d['time'] is not None])
    if all_times.size == 0:
        return go.Figure().update_layout(title="Not enough data to generate average curve.")
    common_time = np.linspace(all_times.min(), all_times.max(), num=200)
    interpolated_intensities = [np.interp(common_time, fd['time'], fd['intensity'], left=np.nan, right=np.nan) for fd in group_files_data.values()]
    if not interpolated_intensities:
        return go.Figure().update_layout(title="Not enough valid data for averaging.")
    intensities_array = np.array(interpolated_intensities)
    mean_intensity = np.nanmean(intensities_array, axis=0)
    std_intensity = np.nanstd(intensities_array, axis=0)
    upper_bound, lower_bound = mean_intensity + std_intensity, mean_intensity - std_intensity
    fig = go.Figure([
        go.Scatter(x=np.concatenate([common_time, common_time[::-1]]), y=np.concatenate([upper_bound, lower_bound[::-1]]), fill='toself', fillcolor='rgba(220,20,60,0.2)', line=dict(color='rgba(255,255,255,0)'), hoverinfo="none", name='Std. Dev.'),
        go.Scatter(x=common_time, y=mean_intensity, mode='lines', name='Average Recovery', line=dict(color='crimson', width=3))
    ])
    fig.update_layout(title="Average FRAP Recovery Curve", xaxis_title="Time (s)", yaxis_title="Normalized Intensity")
    return fig

# --- Core Analysis and Data Logic ---

class FRAPDataManager:
    def __init__(self): 
        self.files,self.groups = {},{}
    
    def load_file(self,file_path,file_name):
        try:
            # Extract original extension before the hash suffix
            original_path = file_path
            if '_' in file_path and any(ext in file_path for ext in ['.xls_', '.xlsx_', '.csv_']):
                # Find the original extension and create a temporary file with correct extension
                import tempfile
                import shutil
                if '.xlsx_' in file_path:
                    temp_path = tempfile.mktemp(suffix='.xlsx')
                elif '.xls_' in file_path:
                    temp_path = tempfile.mktemp(suffix='.xls')
                elif '.csv_' in file_path:
                    temp_path = tempfile.mktemp(suffix='.csv')
                else:
                    temp_path = file_path
                
                if temp_path != file_path:
                    shutil.copy2(file_path, temp_path)
                    file_path = temp_path
            
            processed_df = CoreFRAPAnalysis.preprocess(CoreFRAPAnalysis.load_data(file_path))
            if 'normalized' in processed_df.columns and not processed_df['normalized'].isnull().all():
                time,intensity = processed_df['time'].values,processed_df['normalized'].values
                fits = CoreFRAPAnalysis.fit_all_models(time,intensity)
                best_fit = CoreFRAPAnalysis.select_best_fit(fits,st.session_state.settings['default_criterion'])
                params = CoreFRAPAnalysis.extract_clustering_features(best_fit)
                self.files[file_path]={
                    'name':file_name,'data':processed_df,'time':time,'intensity':intensity,
                    'fits':fits,'best_fit':best_fit,'features':params
                }
                logger.info(f"Loaded: {file_name}")
                return True
        except Exception as e: 
            st.error(f"Error loading {file_name}: {e}")
            return False
    
    def create_group(self,name):
        if name not in self.groups: 
            self.groups[name]={'name':name,'files':[],'features_df':None}
    
    def update_group_analysis(self,name,excluded_files=None):
        if name not in self.groups: return
        group=self.groups[name]
        features_list=[]
        for fp in group['files']:
            if fp not in (excluded_files or []) and fp in self.files and self.files[fp]['features']:
                ff=self.files[fp]['features'].copy()
                ff.update({'file_path':fp,'file_name':self.files[fp]['name']})
                features_list.append(ff)
        group['features_df'] = pd.DataFrame(features_list) if features_list else pd.DataFrame()

    def add_file_to_group(self, group_name, file_path):
        """
        Adds a file to an existing group.
        """
        if group_name in self.groups and file_path in self.files:
            if file_path not in self.groups[group_name]['files']:
                self.groups[group_name]['files'].append(file_path)
                return True
        return False

    def load_groups_from_zip_archive(self, zip_file):
        """
        Loads files from a ZIP archive containing subfolders, where each subfolder
        is treated as a new group.
        """
        try:
            # Create a temporary directory to extract the files
            with tempfile.TemporaryDirectory() as temp_dir:
                with zipfile.ZipFile(io.BytesIO(zip_file.getbuffer())) as z:
                    z.extractall(temp_dir)

                # Walk through the extracted directory to find subfolders (groups)
                for root, dirs, _ in os.walk(temp_dir):
                    # We are interested in the directories at the first level
                    if root == temp_dir:
                        for group_name in dirs:
                            if group_name.startswith('__'): # Ignore system folders like __MACOSX
                                continue
                            self.create_group(group_name)
                            group_path = os.path.join(root, group_name)
                            
                            # Now find the files in this group's subfolder
                            for file_in_group in os.listdir(group_path):
                                file_path_in_temp = os.path.join(group_path, file_in_group)
                                # Ignore subdirectories and hidden files
                                if os.path.isfile(file_path_in_temp) and not file_in_group.startswith('.'):
                                    with open(file_path_in_temp, 'rb') as f_content:
                                        file_content = f_content.read()
                                    
                                    file_name = os.path.basename(file_in_group)
                                    tp = f"data/{file_name}_{hash(file_content)}"
                                    
                                    if tp not in self.files:
                                        # Copy the file to the data directory to be loaded
                                        shutil.copy(file_path_in_temp, tp)
                                        if self.load_file(tp, file_name):
                                            self.add_file_to_group(group_name, tp)
                            self.update_group_analysis(group_name)
            return True
        except Exception as e:
            logger.error(f"Error processing ZIP archive with subfolders: {e}")
            return False

    def load_zip_archive_and_create_group(self, zip_file, group_name):
        """
        Loads files from a ZIP archive, creates a new group, and adds the files to it.
        """
        try:
            self.create_group(group_name)
            with zipfile.ZipFile(io.BytesIO(zip_file.read())) as z:
                file_list = z.namelist()
                for file_in_zip in file_list:
                    # Ignore directories and hidden files (like __MACOSX)
                    if not file_in_zip.endswith('/') and '__MACOSX' not in file_in_zip:
                        file_content = z.read(file_in_zip)
                        file_name = os.path.basename(file_in_zip)

                        # Create a temporary file to use with the existing load_file logic
                        tp = f"data/{file_name}_{hash(file_content)}"
                        if tp not in self.files:
                             with open(tp, "wb") as f:
                                 f.write(file_content)
                             if self.load_file(tp, file_name):
                                 self.add_file_to_group(group_name, tp)

            self.update_group_analysis(group_name)
            st.session_state.selected_group_name = group_name
            return True
        except Exception as e:
            logger.error(f"Error processing ZIP archive for group {group_name}: {e}")
            # Clean up group if creation failed
            if group_name in self.groups:
                del self.groups[group_name]
            return False

# --- Streamlit UI Application ---
st.title("🔬 FRAP Analysis Application")
st.markdown("**Fluorescence Recovery After Photobleaching with Supervised Outlier Removal**")
dm = st.session_state.data_manager = FRAPDataManager() if st.session_state.data_manager is None else st.session_state.data_manager

with st.sidebar:
    st.header("Data Management")
    
    # --- New ZIP Uploader for Groups ---
    st.subheader("Group Upload (from ZIP with subfolders)")
    st.markdown("Drag and drop a ZIP file containing subfolders. Each subfolder will be treated as a new group, and the files within will be added to it.")
    uploaded_zip = st.file_uploader("Upload a .zip file with group subfolders", type=['zip'], key="zip_uploader")

    if uploaded_zip:
        if st.button(f"Create Groups from '{uploaded_zip.name}'"):
            with st.spinner(f"Processing groups from '{uploaded_zip.name}'..."):
                if 'data_manager' in st.session_state:
                    success = dm.load_groups_from_zip_archive(uploaded_zip)
                    if success:
                        st.success(f"Successfully created groups from the ZIP archive.")
                        st.rerun()
                    else:
                        st.error("Failed to process ZIP archive with subfolders.")

    # --- Existing Single File Uploader ---
    st.subheader("Single File Upload")
    uploaded_files = st.file_uploader("Upload FRAP files", type=['xls', 'xlsx', 'csv'], accept_multiple_files=True, key="single_file_uploader")
    if uploaded_files:
        # ... keep the existing single file upload logic here ...
        new_files_loaded = False
        for uf in uploaded_files:
            tp=f"data/{uf.name}_{hash(uf.getvalue())}"
            if tp not in dm.files:
                with open(tp,"wb") as f:
                    f.write(uf.getbuffer())
                if dm.load_file(tp,uf.name):
                    st.success(f"✅ Successfully loaded: {uf.name}")
                    new_files_loaded = True
                else:
                    st.error(f"❌ Failed to load: {uf.name}")
        
        if new_files_loaded:
            st.info("📂 Files have been processed and are ready for analysis. Use the 'Add Selected Files' section below to assign them to groups.")
            if st.button("Refresh Interface", type="primary"):
                st.rerun()

    st.header("Group Setup")
    with st.form("new_group_form",clear_on_submit=True):
        new_group_name=st.text_input("Enter New Group Name")
        if st.form_submit_button("Create Group") and new_group_name:
            dm.create_group(new_group_name)
            st.session_state.selected_group_name=new_group_name
            st.success(f"Group '{new_group_name}' created!")
            st.rerun()
    
    if dm.groups:
        all_groups=list(dm.groups.keys())
        if st.session_state.selected_group_name not in all_groups: 
            st.session_state.selected_group_name = all_groups[0] if all_groups else None
        
        def on_group_change(): 
            st.session_state.selected_group_name=st.session_state.group_selector_widget
        
        st.selectbox(
            "Select Active Group",all_groups,key="group_selector_widget",on_change=on_group_change,
            index=all_groups.index(st.session_state.selected_group_name) if st.session_state.selected_group_name in all_groups else 0
        )
        
        selected_group_name = st.session_state.selected_group_name
        if selected_group_name:
            group = dm.groups[selected_group_name]
            
            # Enhanced group management UI with two-panel layout
            st.subheader("Group File Management")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Ungrouped Files**")
                ungrouped_files = [f for f in dm.files if not any(f in g['files'] for g in dm.groups.values())]
                if ungrouped_files:
                    selected_files = st.multiselect(
                        "Select files to add:",
                        ungrouped_files,
                        format_func=lambda p: dm.files[p]['name'],
                        key=f"add_files_to_{selected_group_name}"
                    )
                else:
                    st.info("No ungrouped files available.")
                    selected_files = []
            
            with col2:
                st.markdown(f"**Group: {selected_group_name}**")
                if group['files']:
                    st.write(f"Files in group ({len(group['files'])}):")
                    for file_path in group['files']:
                        st.write(f"• {dm.files[file_path]['name']}")
                    
                    files_to_remove = st.multiselect(
                        "Select files to remove:",
                        group['files'],
                        format_func=lambda p: dm.files[p]['name'],
                        key=f"remove_files_from_{selected_group_name}"
                    )
                else:
                    st.info("No files in this group yet.")
                    files_to_remove = []
            
            # Action buttons
            button_col1, button_col2 = st.columns(2)
            with button_col1:
                if st.button("Add Selected Files", disabled=len(selected_files) == 0, key=f"btn_add_{selected_group_name}"):
                    group['files'].extend(selected_files)
                    dm.update_group_analysis(selected_group_name)
                    st.success(f"Added {len(selected_files)} files to {selected_group_name}")
                    st.rerun()
            
            with button_col2:
                if st.button("Remove Selected Files", disabled=len(files_to_remove) == 0, key=f"btn_rm_{selected_group_name}"):
                    for file_path in files_to_remove:
                        group['files'].remove(file_path)
                    dm.update_group_analysis(selected_group_name)
                    st.success(f"Removed {len(files_to_remove)} files from {selected_group_name}")
                    st.rerun()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Single File Analysis", "📈 Group Analysis", "📊 Multi-Group Comparison", "🖼️ Image Analysis", "💾 Session Management", "⚙️ Settings"])

with tab1:
    st.header("Single File Analysis")
    if dm.files:
        selected_file_path = st.selectbox("Select file to analyze", list(dm.files.keys()), format_func=lambda p: dm.files[p]['name'])
        if selected_file_path and selected_file_path in dm.files:
            file_data=dm.files[selected_file_path]
            st.subheader(f"Results for: {file_data['name']}")
            if file_data['best_fit']:
                best_fit,params=file_data['best_fit'],file_data['features']
                t_fit,intensity_fit,_=CoreFRAPAnalysis.get_post_bleach_data(file_data['time'],file_data['intensity'])
                
                # Enhanced metrics display
                st.markdown("### Kinetic Analysis Results")
                col1,col2,col3,col4=st.columns(4)
                with col1:
                    st.metric("Mobile Fraction",f"{params.get('mobile_fraction',0):.1f}%")
                with col2:
                    st.metric("Half-time",f"{params.get('half_time_fast',params.get('half_time',0)):.1f} s")
                with col3:
                    st.metric("R²",f"{best_fit.get('r2',0):.3f}")
                with col4:
                    st.metric("Model",best_fit['model'].title())
                
                # Additional goodness-of-fit metrics
                col5,col6,col7,col8=st.columns(4)
                with col5:
                    st.metric("AIC",f"{best_fit.get('aic',0):.1f}")
                with col6:
                    st.metric("Adj. R²",f"{best_fit.get('adj_r2',0):.3f}")
                with col7:
                    st.metric("BIC",f"{best_fit.get('bic',0):.1f}")
                with col8:
                    st.metric("Red. χ²",f"{best_fit.get('red_chi2',0):.3f}")
                
                # Main recovery curve plot
                fig=go.Figure([
                    go.Scatter(x=file_data['time'],y=file_data['intensity'],mode='markers',name='Data',marker_color='blue',marker_size=6),
                    go.Scatter(x=t_fit,y=best_fit['fitted_values'],mode='lines',name=f"Fit: {best_fit['model'].title()}",line=dict(color='red',width=3))
                ])
                fig.update_layout(title='FRAP Recovery Curve',xaxis_title='Time (s)',yaxis_title='Normalized Intensity',height=450)
                st.plotly_chart(fig,use_container_width=True)
                
                # Residuals analysis
                st.markdown("### Residuals Analysis")
                st.markdown("Random scatter around zero indicates good fit quality. Patterns suggest model inadequacy.")
                
                if 'fitted_values' in best_fit:
                    residuals=intensity_fit-best_fit['fitted_values']
                    residual_std=np.std(residuals)
                    residual_mean=np.mean(residuals)
                    
                    res_fig=go.Figure()
                    res_fig.add_trace(go.Scatter(
                        x=t_fit,y=residuals,mode='markers',
                        marker=dict(color='orange',size=6),name='Residuals',
                        hovertemplate='Time: %{x:.1f}s<br>Residual: %{y:.4f}<extra></extra>'
                    ))
                    res_fig.add_hline(y=0,line_dash="dash",line_color="gray",annotation_text="Zero")
                    res_fig.add_hline(y=2*residual_std,line_dash="dot",line_color="red",annotation_text="+2σ")
                    res_fig.add_hline(y=-2*residual_std,line_dash="dot",line_color="red",annotation_text="-2σ")
                    
                    res_fig.update_layout(
                        title=f"Residuals - {best_fit['model'].title()} Model",
                        xaxis_title="Time (s)",yaxis_title="Residuals",height=350,
                        annotations=[dict(x=0.02,y=0.98,xref="paper",yref="paper",
                                         text=f"Mean: {residual_mean:.4f}<br>Std: {residual_std:.4f}",
                                         showarrow=False,bgcolor="white",bordercolor="gray",borderwidth=1)]
                    )
                    st.plotly_chart(res_fig,use_container_width=True)
                
                # Component-wise analysis for multi-component models
                if best_fit['model'] in ['double','triple']:
                    st.markdown("### Component Analysis")
                    st.markdown("Individual kinetic components contributing to overall recovery.")
                    
                    comp_fig=go.Figure()
                    fit_params=best_fit['params']
                    C=fit_params[-1]  # Offset
                    
                    # Plot total fit
                    comp_fig.add_trace(go.Scatter(
                        x=t_fit,y=best_fit['fitted_values'],mode='lines',
                        name="Total Fit",line=dict(color='red',width=3)
                    ))
                    
                    if best_fit['model']=='double':
                        A1,k1,A2,k2=fit_params[0],fit_params[1],fit_params[2],fit_params[3]
                        comp1=A1*(1-np.exp(-k1*t_fit))+C
                        comp2=A2*(1-np.exp(-k2*t_fit))+C
                        
                        comp_fig.add_trace(go.Scatter(
                            x=t_fit,y=comp1,mode='lines',name=f'Fast (k={k1:.4f})',
                            line=dict(dash='dot',color='blue')
                        ))
                        comp_fig.add_trace(go.Scatter(
                            x=t_fit,y=comp2,mode='lines',name=f'Slow (k={k2:.4f})',
                            line=dict(dash='dot',color='green')
                        ))
                        
                    elif best_fit['model']=='triple':
                        A1,k1,A2,k2,A3,k3=fit_params[0],fit_params[1],fit_params[2],fit_params[3],fit_params[4],fit_params[5]
                        comp1=A1*(1-np.exp(-k1*t_fit))+C
                        comp2=A2*(1-np.exp(-k2*t_fit))+C
                        comp3=A3*(1-np.exp(-k3*t_fit))+C
                        
                        comp_fig.add_trace(go.Scatter(
                            x=t_fit,y=comp1,mode='lines',name=f'Fast (k={k1:.4f})',
                            line=dict(dash='dot',color='blue')
                        ))
                        comp_fig.add_trace(go.Scatter(
                            x=t_fit,y=comp2,mode='lines',name=f'Medium (k={k2:.4f})',
                            line=dict(dash='dot',color='green')
                        ))
                        comp_fig.add_trace(go.Scatter(
                            x=t_fit,y=comp3,mode='lines',name=f'Slow (k={k3:.4f})',
                            line=dict(dash='dot',color='purple')
                        ))
                    
                    comp_fig.update_layout(
                        title="Component-wise Recovery Analysis",
                        xaxis_title="Time (s)",yaxis_title="Normalized Intensity",height=400
                    )
                    st.plotly_chart(comp_fig,use_container_width=True)
                
                # Biophysical parameters
                st.markdown("### Biophysical Interpretation")
                
                # Calculate diffusion coefficient and molecular weight estimates
                primary_rate=params.get('rate_constant_fast',params.get('rate_constant',0))
                if primary_rate>0:
                    bleach_radius=st.session_state.settings.get('default_bleach_radius',1.0)
                    pixel_size=st.session_state.settings.get('default_pixel_size',1.0)
                    effective_radius_um=bleach_radius*pixel_size
                    
                    # Diffusion interpretation: D = (r² × k) / 4 (CORRECTED FORMULA)
                    diffusion_coeff=(effective_radius_um**2*primary_rate)/4.0
                    
                    # Binding interpretation: k_off = k
                    k_off=primary_rate
                    
                    # Molecular weight estimation (relative to GFP)
                    gfp_d=25.0  # μm²/s
                    gfp_mw=27.0  # kDa
                    apparent_mw=gfp_mw*(gfp_d/diffusion_coeff) if diffusion_coeff>0 else 0
                    
                    col_bio1,col_bio2,col_bio3,col_bio4=st.columns(4)
                    with col_bio1:
                        st.metric("App. D (μm²/s)",f"{diffusion_coeff:.3f}")
                    with col_bio2:
                        st.metric("k_off (s⁻¹)",f"{k_off:.4f}")
                    with col_bio3:
                        st.metric("App. MW (kDa)",f"{apparent_mw:.1f}")
                    with col_bio4:
                        st.metric("Immobile (%)",f"{params.get('immobile_fraction',0):.1f}")
                
                else:
                    st.warning("Cannot calculate biophysical parameters - invalid rate constant")
            else: 
                st.error("Could not determine a best fit for this file.")
    else: 
        st.info("Upload files using the sidebar to begin.")

with tab2:
    st.header("Group Analysis")
    selected_group_name = st.session_state.get('selected_group_name')
    if selected_group_name and selected_group_name in dm.groups:
        st.subheader(f"Analysis for Group: {selected_group_name}")
        group = dm.groups[selected_group_name]
        if not group['files']: 
            st.warning("This group is empty. Add files from the sidebar.")
        else:
            st.markdown("---")
            st.markdown("### Step 1: Statistical Outlier Removal")
            dm.update_group_analysis(selected_group_name)
            features_df=group.get('features_df')
            excluded_paths=[]
            
            if features_df is not None and not features_df.empty:
                # Show automatic outlier detection results
                auto_outliers = group.get('auto_outliers', [])
                if auto_outliers:
                    st.info(f"🤖 **Automatic outlier detection** identified {len(auto_outliers)} potential outliers based on half-time analysis")
                    with st.expander("View auto-detected outliers"):
                        for outlier_path in auto_outliers:
                            if outlier_path in dm.files:
                                st.write(f"• {dm.files[outlier_path]['name']}")
                
                # Manual outlier selection with auto-outliers pre-selected
                opts=[c for c in features_df.select_dtypes(include=np.number).columns if 'fraction' in c or 'rate' in c]
                defaults=[c for c in ['mobile_fraction','immobile_fraction'] if c in opts]
                
                col_outlier1, col_outlier2 = st.columns(2)
                with col_outlier1:
                    outlier_check_features=st.multiselect("Check for outliers based on:",options=opts,default=defaults)
                with col_outlier2:
                    iqr_multiplier=st.slider("Outlier Sensitivity",1.0,3.0,1.5,0.1,help="Lower value = more sensitive.")
                
                # Combine automatic and manual outlier detection
                manual_identified = CoreFRAPAnalysis.identify_outliers(features_df,outlier_check_features,iqr_multiplier)
                all_identified = list(set(auto_outliers + manual_identified))
                
                excluded_paths=st.multiselect(
                    "Select files to EXCLUDE (auto-detected outliers are pre-selected):",
                    options=group['files'],
                    default=all_identified,
                    format_func=lambda p:dm.files[p]['name'],
                    help="Auto-detected outliers are pre-selected. You can add or remove files as needed."
                )
            
            dm.update_group_analysis(selected_group_name,excluded_files=excluded_paths)
            filtered_df=group.get('features_df')
            
            st.markdown("---")
            st.markdown("### Step 2: View Group Results")
            if filtered_df is not None and not filtered_df.empty:
                st.success(f"Displaying results for **{len(filtered_df)}** of {len(group['files'])} files.")
                mean_vals=filtered_df.mean(numeric_only=True)
                st.markdown("#### Overall Fractions")
                col1,col2=st.columns(2)
                col1.metric("Average Mobile Fraction",f"{mean_vals.get('mobile_fraction',0):.2f}%")
                col2.metric("Average Immobile Fraction",f"{mean_vals.get('immobile_fraction',0):.2f}%")
                
                st.markdown("#### Kinetic Parameters: Dual Interpretation Analysis")
                st.markdown("Each kinetic rate can represent either diffusion or binding processes:")
                
                # Get experimental parameters for interpretation
                bleach_radius = st.session_state.settings.get('default_bleach_radius', 1.0)
                pixel_size = st.session_state.settings.get('default_pixel_size', 1.0)
                effective_radius_um = bleach_radius * pixel_size
                
                summary_data=[]
                for name in ['fast','medium','slow']:
                    prop_mobile_key = f'proportion_of_mobile_{name}'
                    prop_total_key = f'proportion_of_total_{name}'
                    rate_key = f'rate_constant_{name}'
                    
                    if prop_mobile_key in mean_vals and not pd.isna(mean_vals.get(prop_mobile_key)):
                        k_val = mean_vals.get(rate_key, 0)
                        
                        # Get dual interpretation
                        kinetic_interp = interpret_kinetics(
                            k_val, 
                            bleach_radius_um=effective_radius_um,
                            gfp_d=25.0,  # GFP reference
                            gfp_mw=27.0
                        )
                        
                        summary_data.append([
                            name.capitalize(),
                            mean_vals.get(prop_mobile_key),
                            mean_vals.get(prop_total_key),
                            k_val,
                            mean_vals.get(f'half_time_{name}'),
                            kinetic_interp['k_off'],
                            kinetic_interp['diffusion_coefficient'],
                            kinetic_interp['apparent_mw'],
                            kinetic_interp['half_time_binding'],
                            kinetic_interp['half_time_diffusion']
                        ])
                
                summary_df = pd.DataFrame(summary_data, columns=[
                    'Component', 'Of Mobile Pool (%)', 'Of Total Pop. (%)', 
                    'Rate (k)', 'Half-time (s)',
                    'k_off (1/s)', 'App. D (μm²/s)', 'App. MW (kDa)',
                    't½ Binding (s)', 't½ Diffusion (s)'
                ])
                
                # Display with enhanced formatting
                st.dataframe(summary_df.set_index('Component').style.format({
                    'Of Mobile Pool (%)': '{:.1f}%',
                    'Of Total Pop. (%)': '{:.1f}%',
                    'Rate (k)': '{:.4f}',
                    'Half-time (s)': '{:.2f}',
                    'k_off (1/s)': '{:.4f}',
                    'App. D (μm²/s)': '{:.3f}',
                    'App. MW (kDa)': '{:.1f}',
                    't½ Binding (s)': '{:.2f}',
                    't½ Diffusion (s)': '{:.2f}'
                }, na_rep="-"))
                
                # Add interpretation guide
                with st.expander("📖 Interpretation Guide"):
                    st.markdown("""
                    **Diffusion Interpretation:**
                    - **App. D**: Apparent diffusion coefficient if recovery is purely diffusion-limited
                    - **App. MW**: Estimated molecular weight based on diffusion relative to GFP
                    - **t½ Diffusion**: Half-time for diffusion across the bleach spot
                    
                    **Binding Interpretation:**
                    - **k_off**: Dissociation rate constant if recovery is binding-limited
                    - **t½ Binding**: Half-time for molecular dissociation and rebinding
                    
                    **Experimental Parameters Used:**
                    - Bleach radius: {:.2f} μm
                    - Pixel size: {:.2f} μm/pixel
                    - Reference: GFP (D = 25 μm²/s, MW = 27 kDa)
                    """.format(effective_radius_um, pixel_size))
                
                # Add comparison metrics
                st.markdown("#### Biological Interpretation Metrics")
                if summary_data:
                    col_interp1, col_interp2, col_interp3 = st.columns(3)
                    
                    # Calculate average diffusion coefficient
                    avg_d = np.nanmean([row[6] for row in summary_data if not np.isnan(row[6])])
                    avg_mw = np.nanmean([row[7] for row in summary_data if not np.isnan(row[7])])
                    avg_koff = np.nanmean([row[5] for row in summary_data if not np.isnan(row[5])])
                    
                    with col_interp1:
                        st.metric(
                            "Avg. Apparent D", 
                            f"{avg_d:.3f} μm²/s" if not np.isnan(avg_d) else "N/A",
                            help="Average diffusion coefficient across all components"
                        )
                    
                    with col_interp2:
                        st.metric(
                            "Avg. Apparent MW", 
                            f"{avg_mw:.1f} kDa" if not np.isnan(avg_mw) else "N/A",
                            help="Average molecular weight estimate"
                        )
                    
                    with col_interp3:
                        st.metric(
                            "Avg. k_off", 
                            f"{avg_koff:.4f} s⁻¹" if not np.isnan(avg_koff) else "N/A",
                            help="Average dissociation rate constant"
                        )
                
                st.markdown("---")
                st.markdown("### Step 3: Individual Curve Analysis")
                
                # Enhanced plot of all individual curves with outliers highlighted
                st.markdown("#### All Individual Curves (Outliers Highlighted)")
                fig_indiv = go.Figure()
                
                group_files_data = {path: dm.files[path] for path in group['files']}
                
                for path, file_data in group_files_data.items():
                    is_outlier = path in excluded_paths
                    line_color = "rgba(255, 0, 0, 0.7)" if is_outlier else "rgba(100, 100, 100, 0.4)"
                    line_width = 2 if is_outlier else 1
                    
                    fig_indiv.add_trace(go.Scatter(
                        x=file_data['time'],
                        y=file_data['intensity'],
                        mode='lines',
                        name=file_data['name'],
                        line=dict(color=line_color, width=line_width),
                        legendgroup="outlier" if is_outlier else "included",
                        showlegend=False,
                        hovertemplate=f"<b>{file_data['name']}</b><br>" +
                                    f"Status: {'Outlier' if is_outlier else 'Included'}<br>" +
                                    "Time: %{x:.2f}s<br>" +
                                    "Intensity: %{y:.3f}<extra></extra>"
                    ))
                
                # Add legend entries manually
                fig_indiv.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
                                             line=dict(color="rgba(100, 100, 100, 0.8)", width=2),
                                             name="Included", showlegend=True))
                fig_indiv.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
                                             line=dict(color="rgba(255, 0, 0, 0.8)", width=2),
                                             name="Outliers", showlegend=True))
                
                fig_indiv.update_layout(
                    title="All Individual Recovery Curves in Group",
                    xaxis_title="Time (s)",
                    yaxis_title="Normalized Intensity",
                    legend_title="File Status",
                    height=500
                )
                st.plotly_chart(fig_indiv, use_container_width=True)
                
                # Detailed table of individual kinetics
                st.markdown("#### Kinetic Parameters for Each File")
                
                all_features_df = group.get('features_df')
                if all_features_df is not None and not all_features_df.empty:
                    # Create a complete dataframe with all files (before filtering)
                    all_files_data = []
                    for path in group['files']:
                        file_data = dm.files[path]
                        features = file_data.get('features', {})
                        
                        # Get dual interpretation for each file
                        bleach_radius = st.session_state.settings.get('default_bleach_radius', 1.0)
                        pixel_size = st.session_state.settings.get('default_pixel_size', 1.0)
                        effective_radius_um = bleach_radius * pixel_size
                        
                        # For the primary rate constant (usually fast component)
                        primary_rate = features.get('rate_constant_fast', features.get('rate_constant', 0))
                        kinetic_interp = interpret_kinetics(
                            primary_rate,
                            bleach_radius_um=effective_radius_um,
                            gfp_d=25.0,
                            gfp_mw=27.0
                        )
                        
                        all_files_data.append({
                            'File Name': file_data['name'],
                            'Status': 'Outlier' if path in excluded_paths else 'Included',
                            'Mobile (%)': features.get('mobile_fraction', np.nan),
                            'Immobile (%)': features.get('immobile_fraction', np.nan),
                            'Primary Rate (k)': primary_rate,
                            'Half-time (s)': features.get('half_time_fast', features.get('half_time', np.nan)),
                            'k_off (1/s)': kinetic_interp['k_off'],
                            'App. D (μm²/s)': kinetic_interp['diffusion_coefficient'],
                            'App. MW (kDa)': kinetic_interp['apparent_mw'],
                            'Model': features.get('model', 'Unknown'),
                            'R²': features.get('r2', np.nan)
                        })
                    
                    detailed_df = pd.DataFrame(all_files_data)
                    
                    # Style the dataframe with outliers highlighted
                    def highlight_outliers(row):
                        return ['background-color: #ffcccc' if row['Status'] == 'Outlier' else '' for _ in row]
                    
                    with st.expander("📊 Show Detailed Kinetics Table", expanded=True):
                        st.dataframe(
                            detailed_df.style.apply(highlight_outliers, axis=1).format({
                                'Mobile (%)': '{:.1f}',
                                'Immobile (%)': '{:.1f}',
                                'Primary Rate (k)': '{:.4f}',
                                'Half-time (s)': '{:.2f}',
                                'k_off (1/s)': '{:.4f}',
                                'App. D (μm²/s)': '{:.3f}',
                                'App. MW (kDa)': '{:.1f}',
                                'R²': '{:.3f}'
                            }, na_rep="-"),
                            use_container_width=True
                        )
                        
                        # Summary statistics
                        included_data = detailed_df[detailed_df['Status'] == 'Included']
                        st.markdown("##### Summary Statistics (Included Files Only)")
                        summary_cols = ['Mobile (%)', 'Primary Rate (k)', 'App. D (μm²/s)', 'App. MW (kDa)']
                        summary_stats = included_data[summary_cols].describe()
                        st.dataframe(summary_stats.round(3))
                
                st.markdown("---")
                st.markdown("### Step 4: Generate Comprehensive Report")
                
                col_report1, col_report2 = st.columns([3, 1])
                
                with col_report1:
                    st.markdown("Generate a detailed analysis report including:")
                    st.markdown("- Executive summary with outlier analysis")
                    st.markdown("- Dual-interpretation kinetics results")
                    st.markdown("- Individual file details with quality assessment")
                    st.markdown("- Experimental recommendations")
                
                with col_report2:
                    if st.button("📄 Generate Report", type="primary"):
                        try:
                            # Generate comprehensive report
                            report_content = generate_markdown_report(
                                group_name=selected_group_name,
                                settings=st.session_state.settings,
                                summary_df=summary_df,
                                detailed_df=detailed_df,
                                excluded_count=len(excluded_paths),
                                total_count=len(group['files'])
                            )
                            
                            # Provide download button
                            st.download_button(
                                label="⬇️ Download Report",
                                data=report_content,
                                file_name=f"FRAP_Report_{selected_group_name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
                                mime="text/markdown",
                                help="Download comprehensive analysis report as Markdown file"
                            )
                            
                            st.success("Report generated successfully! Click Download Report to save.")
                            
                        except Exception as e:
                            st.error(f"Error generating report: {e}")
                
                st.markdown("---")
                st.markdown("### Step 5: Parameter Distribution Analysis")
                
                # Parameter distribution visualization
                numeric_cols = [col for col in filtered_df.select_dtypes(include=[np.number]).columns 
                               if col not in ['file_path'] and not filtered_df[col].isna().all()]
                
                if numeric_cols:
                    selected_param = st.selectbox(
                        "Select parameter for distribution analysis:",
                        numeric_cols,
                        index=numeric_cols.index('mobile_fraction') if 'mobile_fraction' in numeric_cols else 0
                    )
                    
                    if selected_param:
                        # Create parameter distribution plot
                        param_data = filtered_df.dropna(subset=[selected_param])
                        
                        if len(param_data) > 0:
                            # Create combined histogram and box plot
                            from plotly.subplots import make_subplots
                            
                            fig = make_subplots(
                                rows=2, cols=1,
                                row_heights=[0.8, 0.2],
                                subplot_titles=[f"Distribution of {selected_param.replace('_', ' ').title()}", "Box Plot"],
                                vertical_spacing=0.1
                            )
                            
                            # Histogram
                            fig.add_trace(
                                go.Histogram(
                                    x=param_data[selected_param],
                                    nbinsx=min(20, len(param_data)//2 + 1),
                                    name="Distribution",
                                    marker_color="teal",
                                    opacity=0.7
                                ),
                                row=1, col=1
                            )
                            
                            # Box plot
                            fig.add_trace(
                                go.Box(
                                    x=param_data[selected_param],
                                    name="Statistics",
                                    marker_color="orange",
                                    boxpoints="all",
                                    jitter=0.3,
                                    pointpos=-1.8
                                ),
                                row=2, col=1
                            )
                            
                            # Add statistics
                            mean_val = param_data[selected_param].mean()
                            std_val = param_data[selected_param].std()
                            median_val = param_data[selected_param].median()
                            cv_val = (std_val / mean_val) * 100 if mean_val != 0 else 0
                            
                            fig.add_annotation(
                                x=0.02, y=0.98,
                                xref="paper", yref="paper",
                                text=f"Mean: {mean_val:.4f}<br>Std: {std_val:.4f}<br>Median: {median_val:.4f}<br>CV: {cv_val:.1f}%<br>N: {len(param_data)}",
                                showarrow=False,
                                bgcolor="white",
                                bordercolor="gray",
                                borderwidth=1
                            )
                            
                            fig.update_layout(
                                height=500,
                                showlegend=False
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Parameter statistics table
                            stats_data = {
                                'Statistic': ['Count', 'Mean', 'Std Dev', 'Min', 'Max', 'Median', 'CV (%)'],
                                'Value': [
                                    len(param_data),
                                    f"{mean_val:.4f}",
                                    f"{std_val:.4f}", 
                                    f"{param_data[selected_param].min():.4f}",
                                    f"{param_data[selected_param].max():.4f}",
                                    f"{median_val:.4f}",
                                    f"{cv_val:.1f}"
                                ]
                            }
                            
                            stats_df = pd.DataFrame(stats_data)
                            st.dataframe(stats_df, use_container_width=True)
                        else:
                            st.warning("No valid data for selected parameter")
                
                st.markdown("---")
                st.markdown("### Step 6: Global Simultaneous Fitting")
                st.markdown("Perform global fitting with shared kinetic parameters across all traces in the group")
                
                with st.expander("🌐 Global Simultaneous Fit", expanded=False):
                    st.markdown("""
                    **Global fitting** constrains kinetic rate constants to be identical across all traces 
                    while allowing individual amplitudes and offsets. This approach:
                    - Increases statistical power by pooling data
                    - Provides more robust parameter estimates
                    - Enables direct comparison of amplitudes between conditions
                    """)
                    
                    col_global1, col_global2 = st.columns(2)
                    
                    with col_global1:
                        global_model = st.selectbox(
                            "Select model for global fitting:",
                            ["single", "double", "triple"],
                            format_func=lambda x: f"{x.title()}-component exponential",
                            help="Choose the kinetic model for global fitting"
                        )
                        
                        include_outliers_global = st.checkbox(
                            "Include outliers in global fit",
                            value=False,
                            help="Whether to include previously excluded outliers in global fitting"
                        )
                    
                    with col_global2:
                        if st.button("🚀 Run Global Fit", type="primary"):
                            try:
                                with st.spinner(f"Performing global {global_model}-component fitting..."):
                                    # Determine which files to exclude
                                    files_to_exclude = [] if include_outliers_global else excluded_paths
                                    
                                    # Import the FRAPData class and perform global fitting
                                    from frap_data import FRAPData
                                    
                                    # Create a temporary FRAPData instance with current data
                                    temp_dm = FRAPData()
                                    temp_dm.files = dm.files
                                    temp_dm.groups = dm.groups
                                    
                                    # Perform global fitting
                                    global_result = temp_dm.fit_group_models(
                                        selected_group_name, 
                                        model=global_model,
                                        excluded_files=files_to_exclude
                                    )
                                    
                                    if global_result.get('success', False):
                                        st.success("✅ Global fitting completed successfully!")
                                        
                                        # Display global fit results
                                        st.markdown("#### Global Fit Results")
                                        
                                        # Shared parameters
                                        shared_params = global_result['shared_params']
                                        col_param1, col_param2, col_param3 = st.columns(3)
                                        
                                        if global_model == 'single':
                                            with col_param1:
                                                st.metric("Shared Rate (k)", f"{shared_params['k']:.4f} s⁻¹")
                                            with col_param2:
                                                st.metric("Mean R²", f"{global_result['mean_r2']:.3f}")
                                            with col_param3:
                                                st.metric("Global AIC", f"{global_result['aic']:.1f}")
                                        
                                        elif global_model == 'double':
                                            with col_param1:
                                                st.metric("Fast Rate (k₁)", f"{shared_params['k1']:.4f} s⁻¹")
                                            with col_param2:
                                                st.metric("Slow Rate (k₂)", f"{shared_params['k2']:.4f} s⁻¹")
                                            with col_param3:
                                                st.metric("Mean R²", f"{global_result['mean_r2']:.3f}")
                                        
                                        elif global_model == 'triple':
                                            with col_param1:
                                                st.metric("Fast Rate (k₁)", f"{shared_params['k1']:.4f} s⁻¹")
                                            with col_param2:
                                                st.metric("Medium Rate (k₂)", f"{shared_params['k2']:.4f} s⁻¹")
                                            with col_param3:
                                                st.metric("Slow Rate (k₃)", f"{shared_params['k3']:.4f} s⁻¹")
                                        
                                        # Individual amplitudes table
                                        st.markdown("#### Individual File Amplitudes")
                                        individual_data = []
                                        for i, (file_name, params, r2) in enumerate(zip(
                                            global_result['file_names'],
                                            global_result['individual_params'],
                                            global_result['r2_values']
                                        )):
                                            row_data = {'File': file_name, 'R²': r2}
                                            
                                            if global_model == 'single':
                                                row_data['Amplitude (A)'] = params['A']
                                            elif global_model == 'double':
                                                row_data['Fast Amplitude (A₁)'] = params['A1']
                                                row_data['Slow Amplitude (A₂)'] = params['A2']
                                                row_data['Total Amplitude'] = params['A1'] + params['A2']
                                            elif global_model == 'triple':
                                                row_data['Fast Amplitude (A₁)'] = params['A1']
                                                row_data['Medium Amplitude (A₂)'] = params['A2']
                                                row_data['Slow Amplitude (A₃)'] = params['A3']
                                                row_data['Total Amplitude'] = params['A1'] + params['A2'] + params['A3']
                                            
                                            individual_data.append(row_data)
                                        
                                        individual_df = pd.DataFrame(individual_data)
                                        st.dataframe(individual_df.style.format({
                                            col: '{:.4f}' for col in individual_df.columns if col not in ['File']
                                        }), use_container_width=True)
                                        
                                        # Global fit visualization
                                        st.markdown("#### Global Fit Visualization")
                                        
                                        # Create plot showing all traces with global fit
                                        fig_global = go.Figure()
                                        
                                        # Plot individual data and fits
                                        fitted_curves = global_result['fitted_curves']
                                        common_time = global_result['common_time']
                                        
                                        for i, (file_name, fitted_curve) in enumerate(zip(global_result['file_names'], fitted_curves)):
                                            # Find original file data
                                            file_path = None
                                            for fp in group['files']:
                                                if fp not in files_to_exclude and dm.files[fp]['name'] == file_name:
                                                    file_path = fp
                                                    break
                                            
                                            if file_path:
                                                file_data = dm.files[file_path]
                                                t_post, i_post, _ = CoreFRAPAnalysis.get_post_bleach_data(
                                                    file_data['time'], file_data['intensity']
                                                )
                                                
                                                # Plot original data
                                                fig_global.add_trace(go.Scatter(
                                                    x=t_post, y=i_post,
                                                    mode='markers',
                                                    name=f"{file_name} (data)",
                                                    marker=dict(size=4, opacity=0.6),
                                                    showlegend=False
                                                ))
                                                
                                                # Plot global fit
                                                fig_global.add_trace(go.Scatter(
                                                    x=common_time, y=fitted_curve,
                                                    mode='lines',
                                                    name=f"{file_name} (global fit)",
                                                    line=dict(width=2),
                                                    showlegend=False
                                                ))
                                        
                                        fig_global.update_layout(
                                            title=f"Global {global_model.title()}-Component Fit Results",
                                            xaxis_title="Time (s)",
                                            yaxis_title="Normalized Intensity",
                                            height=500
                                        )
                                        st.plotly_chart(fig_global, use_container_width=True)
                                        
                                        # Comparison with individual fits
                                        st.markdown("#### Comparison with Individual Fits")
                                        comparison_data = []
                                        
                                        for file_name in global_result['file_names']:
                                            # Find corresponding individual fit
                                            file_path = None
                                            for fp in group['files']:
                                                if fp not in files_to_exclude and dm.files[fp]['name'] == file_name:
                                                    file_path = fp
                                                    break
                                            
                                            if file_path and file_path in dm.files:
                                                file_data = dm.files[file_path]
                                                individual_fit = file_data.get('best_fit')
                                                
                                                if individual_fit and individual_fit['model'] == global_model:
                                                    individual_r2 = individual_fit.get('r2', np.nan)
                                                    global_r2 = global_result['r2_values'][global_result['file_names'].index(file_name)]
                                                    
                                                    comparison_data.append({
                                                        'File': file_name,
                                                        'Individual R²': individual_r2,
                                                        'Global R²': global_r2,
                                                        'Δ R²': global_r2 - individual_r2
                                                    })
                                        
                                        if comparison_data:
                                            comparison_df = pd.DataFrame(comparison_data)
                                            st.dataframe(comparison_df.style.format({
                                                'Individual R²': '{:.4f}',
                                                'Global R²': '{:.4f}',
                                                'Δ R²': '{:.4f}'
                                            }), use_container_width=True)
                                            
                                            avg_improvement = comparison_df['Δ R²'].mean()
                                            if avg_improvement > 0:
                                                st.success(f"Global fitting improved average R² by {avg_improvement:.4f}")
                                            else:
                                                st.info(f"Individual fits performed better on average (Δ R² = {avg_improvement:.4f})")
                                    
                                    else:
                                        st.error(f"❌ Global fitting failed: {global_result.get('error', 'Unknown error')}")
                                        
                            except Exception as e:
                                st.error(f"Error during global fitting: {e}")
                
                st.markdown("---")
                st.markdown("### Step 7: Group Recovery Plots")
                plot_data={path:dm.files[path] for path in filtered_df['file_path'].tolist()}
                st.markdown("##### Average Recovery Curve")
                avg_fig = plot_average_curve(plot_data)
                st.plotly_chart(avg_fig, use_container_width=True)
            else: 
                st.warning("No data to display. All files may have been excluded as outliers.")
    else: 
        st.info("Create and/or select a group from the sidebar to begin analysis.")

with tab3:
    st.header("📊 Multi-Group Statistical Comparison")
    st.markdown("Compare kinetic parameters across multiple experimental conditions")
    
    if len(dm.groups) < 2:
        st.warning("You need at least 2 groups to perform statistical comparisons.")
        st.info("Create groups in the sidebar to enable multi-group analysis.")
    else:
        # Combine data from all groups
        all_group_data = []
        for group_name, group_info in dm.groups.items():
            if group_info.get('files'):
                dm.update_group_analysis(group_name)
                features_df = group_info.get('features_df')
                if features_df is not None and not features_df.empty:
                    temp_df = features_df.copy()
                    temp_df['group'] = group_name
                    all_group_data.append(temp_df)
        
        if not all_group_data:
            st.warning("No processed groups available for comparison.")
        else:
            combined_df = pd.concat(all_group_data, ignore_index=True)
            
            st.markdown("### Parameter Visualization")
            
            # Parameter selection
            available_params = [col for col in combined_df.select_dtypes(include=[np.number]).columns 
                              if col not in ['file_path'] and not combined_df[col].isna().all()]
            
            col_vis1, col_vis2 = st.columns(2)
            with col_vis1:
                param_to_plot = st.selectbox(
                    "Select Parameter for Visualization:",
                    available_params,
                    index=available_params.index('mobile_fraction') if 'mobile_fraction' in available_params else 0
                )
                
            with col_vis2:
                plot_type = st.selectbox("Visualization Type:", ["Box Plot", "Violin Plot", "Bar Plot (Mean ± SEM)"])
            
            # Create visualization
            if plot_type == "Box Plot":
                fig = px.box(
                    combined_df, x='group', y=param_to_plot, color='group',
                    title=f'Distribution of {param_to_plot} Across Groups',
                    points="all"
                )
            elif plot_type == "Violin Plot":
                fig = px.violin(
                    combined_df, x='group', y=param_to_plot, color='group',
                    title=f'Distribution of {param_to_plot} Across Groups',
                    box=True, points="all"
                )
            else:  # Bar Plot
                group_stats = combined_df.groupby('group')[param_to_plot].agg(['mean', 'sem']).reset_index()
                fig = px.bar(
                    group_stats, x='group', y='mean', color='group',
                    error_y='sem',
                    title=f'Mean {param_to_plot} Across Groups (±SEM)'
                )
            
            fig.update_xaxes(title="Experimental Group")
            fig.update_yaxes(title=param_to_plot.replace('_', ' ').title())
            fig.update_layout(showlegend=False, height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### Statistical Testing")
            
            # Statistical comparison options
            col_stat1, col_stat2 = st.columns(2)
            
            with col_stat1:
                if len(dm.groups) == 2:
                    st.markdown("**Two-Group Comparison**")
                    group_names = list(dm.groups.keys())
                    group1_name = st.selectbox("Group 1:", group_names, key="stat_group1")
                    group2_name = st.selectbox("Group 2:", group_names, 
                                             index=1 if len(group_names) > 1 else 0, key="stat_group2")
                    
                    if st.button("Perform t-test"):
                        if group1_name != group2_name:
                            data1 = combined_df[combined_df['group'] == group1_name][param_to_plot].dropna()
                            data2 = combined_df[combined_df['group'] == group2_name][param_to_plot].dropna()
                            
                            if len(data1) > 1 and len(data2) > 1:
                                # Perform Shapiro-Wilk test for normality
                                from scipy import stats
                                _, p_norm1 = stats.shapiro(data1) if len(data1) <= 5000 else (None, 0.05)
                                _, p_norm2 = stats.shapiro(data2) if len(data2) <= 5000 else (None, 0.05)
                                
                                # Choose appropriate test
                                if p_norm1 > 0.05 and p_norm2 > 0.05:
                                    t_stat, p_value = stats.ttest_ind(data1, data2)
                                    test_used = "Student's t-test"
                                else:
                                    t_stat, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                                    test_used = "Mann-Whitney U test"
                                
                                st.success(f"**{test_used} Results:**")
                                st.metric("P-value", f"{p_value:.6f}")
                                st.metric("Test Statistic", f"{t_stat:.4f}")
                                
                                if p_value < 0.001:
                                    st.success("Highly significant difference (p < 0.001)")
                                elif p_value < 0.01:
                                    st.success("Very significant difference (p < 0.01)")
                                elif p_value < 0.05:
                                    st.success("Significant difference (p < 0.05)")
                                else:
                                    st.info("No significant difference (p ≥ 0.05)")
                            else:
                                st.error("Insufficient data for statistical testing")
                        else:
                            st.error("Please select different groups")
                
                else:
                    st.markdown("**Multi-Group Comparison**")
                    if st.button("Perform ANOVA"):
                        groups_data = []
                        group_labels = []
                        
                        for group_name in dm.groups.keys():
                            group_data = combined_df[combined_df['group'] == group_name][param_to_plot].dropna()
                            if len(group_data) > 1:
                                groups_data.append(group_data)
                                group_labels.append(group_name)
                        
                        if len(groups_data) >= 2:
                            from scipy import stats
                            
                            # Perform ANOVA
                            f_stat, p_anova = stats.f_oneway(*groups_data)
                            
                            st.success("**ANOVA Results:**")
                            st.metric("F-statistic", f"{f_stat:.4f}")
                            st.metric("P-value", f"{p_anova:.6f}")
                            
                            if p_anova < 0.05:
                                st.success("Significant group differences detected (p < 0.05)")
                                
                                # Post-hoc pairwise comparisons
                                st.markdown("**Post-hoc Pairwise Comparisons:**")
                                pairwise_results = []
                                
                                for i in range(len(groups_data)):
                                    for j in range(i+1, len(groups_data)):
                                        _, p_pair = stats.ttest_ind(groups_data[i], groups_data[j])
                                        # Bonferroni correction
                                        n_comparisons = len(groups_data) * (len(groups_data) - 1) // 2
                                        p_corrected = min(p_pair * n_comparisons, 1.0)
                                        
                                        pairwise_results.append({
                                            'Group 1': group_labels[i],
                                            'Group 2': group_labels[j],
                                            'P-value': p_pair,
                                            'P-corrected': p_corrected,
                                            'Significant': 'Yes' if p_corrected < 0.05 else 'No'
                                        })
                                
                                pairwise_df = pd.DataFrame(pairwise_results)
                                st.dataframe(pairwise_df.style.format({
                                    'P-value': '{:.6f}',
                                    'P-corrected': '{:.6f}'
                                }))
                            else:
                                st.info("No significant group differences detected (p ≥ 0.05)")
                        else:
                            st.error("Need at least 2 groups with sufficient data")
            
            with col_stat2:
                st.markdown("**Effect Size Analysis**")
                if len(dm.groups) >= 2:
                    group_summary = combined_df.groupby('group')[param_to_plot].agg(['count', 'mean', 'std']).round(4)
                    st.dataframe(group_summary)
                    
                    # Calculate Cohen's d for two-group comparison
                    if len(dm.groups) == 2:
                        groups = list(dm.groups.keys())
                        data1 = combined_df[combined_df['group'] == groups[0]][param_to_plot].dropna()
                        data2 = combined_df[combined_df['group'] == groups[1]][param_to_plot].dropna()
                        
                        if len(data1) > 1 and len(data2) > 1:
                            # Cohen's d calculation
                            pooled_std = np.sqrt(((len(data1)-1)*data1.var() + (len(data2)-1)*data2.var()) / (len(data1)+len(data2)-2))
                            cohens_d = (data1.mean() - data2.mean()) / pooled_std
                            
                            st.metric("Cohen's d", f"{cohens_d:.3f}")
                            
                            if abs(cohens_d) < 0.2:
                                effect_size = "Small"
                            elif abs(cohens_d) < 0.8:
                                effect_size = "Medium" 
                            else:
                                effect_size = "Large"
                            
                            st.info(f"Effect size: {effect_size}")
            
            st.markdown("### Summary Statistics Table")
            summary_stats = combined_df.groupby('group')[available_params].agg(['count', 'mean', 'std', 'sem']).round(4)
            st.dataframe(summary_stats)
            
            st.markdown("---")
            st.markdown("### Automated PDF Report Generation")
            
            col_pdf1, col_pdf2 = st.columns([2, 1])
            
            with col_pdf1:
                st.markdown("Generate a comprehensive statistical analysis report including:")
                st.markdown("- Executive summary with group comparisons")
                st.markdown("- Statistical test results (t-tests, ANOVA, effect sizes)")
                st.markdown("- Publication-ready visualizations and tables")
                st.markdown("- Detailed results for each experimental group")
                
            with col_pdf2:
                # Group selection for PDF report
                selected_groups_pdf = st.multiselect(
                    "Select groups for PDF report:",
                    options=list(dm.groups.keys()),
                    default=list(dm.groups.keys()),
                    help="Choose which groups to include in the comprehensive report"
                )
                
                if st.button("📄 Generate PDF Report", type="primary", disabled=len(selected_groups_pdf) < 2):
                    if len(selected_groups_pdf) >= 2:
                        try:
                            with st.spinner("Generating comprehensive PDF report..."):
                                # Generate the PDF report
                                timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                                pdf_filename = f"FRAP_Statistical_Report_{timestamp}.pdf"
                                
                                # Call the PDF report generator
                                output_file = generate_pdf_report(
                                    data_manager=dm,
                                    groups_to_compare=selected_groups_pdf,
                                    output_filename=pdf_filename,
                                    settings=st.session_state.settings
                                )
                                
                                # Read the generated PDF file
                                with open(output_file, 'rb') as pdf_file:
                                    pdf_data = pdf_file.read()
                                
                                # Provide download button
                                st.download_button(
                                    label="⬇️ Download PDF Report",
                                    data=pdf_data,
                                    file_name=pdf_filename,
                                    mime="application/pdf",
                                    help="Download comprehensive statistical analysis report"
                                )
                                
                                st.success("PDF report generated successfully!")
                                st.info(f"Report includes {len(selected_groups_pdf)} groups with comprehensive statistical analysis")
                                
                                # Clean up temporary file
                                os.remove(output_file)
                                
                        except Exception as e:
                            st.error(f"Error generating PDF report: {e}")
                            st.error("Please ensure all selected groups have processed data")
                    else:
                        st.warning("Select at least 2 groups for statistical comparison")

with tab4:
    # Use the comprehensive image analysis interface
    create_image_analysis_interface()

with tab5:
    st.header("💾 Session Management & Data Export")
    st.markdown("Save your analysis session and export results to various formats")
    
    col_session1, col_session2 = st.columns(2)
    
    with col_session1:
        st.subheader("Session Management")
        
        # Session save functionality
        if st.button("💾 Save Current Session", type="primary"):
            try:
                import pickle
                from datetime import datetime
                
                session_data = {
                    'files': dm.files,
                    'groups': dm.groups,
                    'settings': st.session_state.settings,
                    'timestamp': datetime.now().isoformat(),
                    'version': '1.0'
                }
                
                session_filename = f"FRAP_Session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                session_bytes = pickle.dumps(session_data)
                
                st.download_button(
                    label="⬇️ Download Session File",
                    data=session_bytes,
                    file_name=session_filename,
                    mime="application/octet-stream",
                    help="Save current analysis session for later"
                )
                
                st.success(f"Session prepared for download: {len(dm.files)} files, {len(dm.groups)} groups")
                
            except Exception as e:
                st.error(f"Error saving session: {e}")
        
        # Session load functionality
        st.markdown("### Load Previous Session")
        uploaded_session = st.file_uploader(
            "Upload session file (.pkl)", 
            type=['pkl'],
            help="Load a previously saved analysis session"
        )
        
        if uploaded_session is not None:
            if st.button("📂 Load Session", type="secondary"):
                try:
                    import pickle
                    session_data = pickle.load(uploaded_session)
                    
                    # Validate session data
                    required_keys = ['files', 'groups', 'settings']
                    if all(key in session_data for key in required_keys):
                        # Load session data
                        new_dm = FRAPDataManager()
                        new_dm.files = session_data['files']
                        new_dm.groups = session_data['groups']
                        
                        st.session_state.data_manager = new_dm
                        st.session_state.settings.update(session_data['settings'])
                        
                        st.success(f"Session loaded successfully!")
                        st.info(f"Loaded: {len(new_dm.files)} files, {len(new_dm.groups)} groups")
                        st.info(f"Session from: {session_data.get('timestamp', 'Unknown')}")
                        st.rerun()
                    else:
                        st.error("Invalid session file format")
                        
                except Exception as e:
                    st.error(f"Error loading session: {e}")
    
    with col_session2:
        st.subheader("Data Export")
        
        if dm.groups:
            # Excel export functionality
            if st.button("📊 Export to Excel", type="primary"):
                try:
                    import io
                    
                    # Create comprehensive export data
                    export_data = []
                    
                    # Summary sheet data
                    summary_data = {
                        'Analysis_Summary': {
                            'Generated': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'Total_Groups': len(dm.groups),
                            'Total_Files': len(dm.files)
                        }
                    }
                    
                    # Group data
                    for group_name, group_info in dm.groups.items():
                        if group_info.get('files'):
                            dm.update_group_analysis(group_name)
                            features_df = group_info.get('features_df')
                            if features_df is not None and not features_df.empty:
                                group_export = []
                                
                                for _, row in features_df.iterrows():
                                    file_path = row.get('file_path', '')
                                    file_name = dm.files.get(file_path, {}).get('name', 'Unknown')
                                    
                                    # Get dual interpretation
                                    primary_rate = row.get('rate_constant_fast', row.get('rate_constant', 0))
                                    bleach_radius = st.session_state.settings.get('default_bleach_radius', 1.0)
                                    pixel_size = st.session_state.settings.get('default_pixel_size', 1.0)
                                    effective_radius_um = bleach_radius * pixel_size
                                    
                                    kinetic_interp = interpret_kinetics(
                                        primary_rate,
                                        bleach_radius_um=effective_radius_um,
                                        gfp_d=25.0,
                                        gfp_mw=27.0
                                    )
                                    
                                    group_export.append({
                                        'File_Name': file_name,
                                        'Mobile_Fraction_Percent': row.get('mobile_fraction', np.nan),
                                        'Immobile_Fraction_Percent': row.get('immobile_fraction', np.nan),
                                        'Rate_Constant_k': primary_rate,
                                        'Half_Time_seconds': row.get('half_time_fast', row.get('half_time', np.nan)),
                                        'k_off_per_second': kinetic_interp['k_off'],
                                        'Apparent_D_um2_per_s': kinetic_interp['diffusion_coefficient'],
                                        'Apparent_MW_kDa': kinetic_interp['apparent_mw'],
                                        'Model': row.get('model', 'Unknown'),
                                        'R_squared': row.get('r2', np.nan)
                                    })
                                
                                summary_data[f'Group_{group_name}'] = group_export
                    
                    # Create Excel file using pandas
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                        # Summary sheet
                        summary_df = pd.DataFrame([summary_data['Analysis_Summary']])
                        summary_df.to_excel(writer, sheet_name='Summary', index=False)
                        
                        # Individual group sheets
                        for sheet_name, data in summary_data.items():
                            if sheet_name.startswith('Group_'):
                                group_df = pd.DataFrame(data)
                                group_df.to_excel(writer, sheet_name=sheet_name[:31], index=False)  # Excel sheet name limit
                        
                        # Settings sheet
                        settings_df = pd.DataFrame(list(st.session_state.settings.items()), 
                                                 columns=['Parameter', 'Value'])
                        settings_df.to_excel(writer, sheet_name='Settings', index=False)
                    
                    excel_buffer.seek(0)
                    
                    st.download_button(
                        label="⬇️ Download Excel File",
                        data=excel_buffer.getvalue(),
                        file_name=f"FRAP_Analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        help="Download complete analysis results as Excel workbook"
                    )
                    
                    st.success("Excel export prepared for download!")
                    
                except Exception as e:
                    st.error(f"Error creating Excel export: {e}")
            
            # CSV export for individual groups
            st.markdown("### Export Individual Groups")
            if dm.groups:
                selected_export_group = st.selectbox("Select group to export as CSV:", list(dm.groups.keys()))
                
                if st.button("📄 Export Group as CSV"):
                    group_info = dm.groups[selected_export_group]
                    if group_info.get('files'):
                        features_df = group_info.get('features_df')
                        if features_df is not None and not features_df.empty:
                            # Prepare CSV data with dual interpretation
                            export_data = []
                            for _, row in features_df.iterrows():
                                file_path = row.get('file_path', '')
                                file_name = dm.files.get(file_path, {}).get('name', 'Unknown')
                                
                                primary_rate = row.get('rate_constant_fast', row.get('rate_constant', 0))
                                kinetic_interp = interpret_kinetics(
                                    primary_rate,
                                    bleach_radius_um=st.session_state.settings.get('default_bleach_radius', 1.0) * 
                                                   st.session_state.settings.get('default_pixel_size', 1.0),
                                    gfp_d=25.0,
                                    gfp_mw=27.0
                                )
                                
                                export_data.append({
                                    'File_Name': file_name,
                                    'Mobile_Fraction_Percent': row.get('mobile_fraction', np.nan),
                                    'Immobile_Fraction_Percent': row.get('immobile_fraction', np.nan),
                                    'Rate_Constant_k': primary_rate,
                                    'Half_Time_seconds': row.get('half_time_fast', row.get('half_time', np.nan)),
                                    'k_off_per_second': kinetic_interp['k_off'],
                                    'Apparent_D_um2_per_s': kinetic_interp['diffusion_coefficient'],
                                    'Apparent_MW_kDa': kinetic_interp['apparent_mw'],
                                    'Model': row.get('model', 'Unknown'),
                                    'R_squared': row.get('r2', np.nan)
                                })
                            
                            export_df = pd.DataFrame(export_data)
                            csv_data = export_df.to_csv(index=False)
                            
                            st.download_button(
                                label="⬇️ Download CSV",
                                data=csv_data,
                                file_name=f"FRAP_{selected_export_group}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                help="Download group data as CSV file"
                            )
                            
                            st.success(f"CSV export for {selected_export_group} prepared!")
                        else:
                            st.warning("No data available for export in selected group")
                    else:
                        st.warning("Selected group is empty")
        else:
            st.info("No groups available for export. Create and analyze groups first.")
    
    st.markdown("---")
    st.markdown("### Current Session Status")
    
    col_status1, col_status2, col_status3 = st.columns(3)
    
    with col_status1:
        st.metric("Loaded Files", len(dm.files))
        
    with col_status2:
        st.metric("Created Groups", len(dm.groups))
    
    with col_status3:
        total_processed = sum(1 for group in dm.groups.values() 
                            if group.get('features_df') is not None and not group['features_df'].empty)
        st.metric("Processed Groups", total_processed)
    
    st.markdown("---")
    st.markdown("### Debug Package Generation")
    st.markdown("Create a comprehensive package for external debugging and deployment")
    
    col_debug1, col_debug2 = st.columns([2, 1])
    
    with col_debug1:
        st.markdown("**Debug package includes:**")
        st.markdown("- Complete source code and documentation")
        st.markdown("- Installation scripts for Windows and Unix")
        st.markdown("- Sample data files and test suite")
        st.markdown("- Docker configuration for containerized deployment")
        st.markdown("- Streamlit configuration files")
        
    with col_debug2:
        if st.button("📦 Create Debug Package", type="primary"):
            try:
                with st.spinner("Creating comprehensive debug package..."):
                    # Import and run the debug package creator
                    from create_debug_package import create_debug_package
                    
                    package_file, summary = create_debug_package()
                    
                    # Read the package file
                    with open(package_file, 'rb') as f:
                        package_data = f.read()
                    
                    # Provide download button
                    st.download_button(
                        label="⬇️ Download Debug Package",
                        data=package_data,
                        file_name=package_file,
                        mime="application/zip",
                        help="Download complete debug package with all source code and documentation"
                    )
                    
                    st.success("Debug package created successfully!")
                    st.info(f"Package size: {len(package_data) / 1024 / 1024:.1f} MB")
                    
                    # Clean up temporary file
                    os.remove(package_file)
                    
            except Exception as e:
                st.error(f"Error creating debug package: {e}")
                st.error("Please contact support for assistance")

with tab6:
    st.subheader("Application Settings")
    st.markdown("### General Settings")
    col_gen1,col_gen2=st.columns(2)
    with col_gen1:
        default_criterion=st.selectbox(
            "Default Model Selection Criterion",['aic','r2'],
            index=['aic','r2'].index(st.session_state.settings['default_criterion']),
            format_func=lambda x:{'aic':'Akaike Information Criterion (AIC)','r2':'R-squared'}[x]
        )
        decimal_places=st.number_input("Decimal Places in Results",value=st.session_state.settings['decimal_places'],min_value=0,max_value=6,step=1)
    with col_gen2:
        default_gfp_diffusion=st.number_input("Default GFP Diffusion (μm²/s)",value=st.session_state.settings['default_gfp_diffusion'],min_value=1.0,step=1.0)
        default_gfp_rg=st.number_input("Default GFP Radius of Gyration (nm)",value=st.session_state.settings['default_gfp_rg'],min_value=0.1,step=0.01)
    
    st.markdown("### Experimental Parameters")
    st.markdown("Configure physical parameters for dual-interpretation kinetics analysis:")
    
    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        default_bleach_radius=st.number_input("Bleach Radius (pixels)",value=st.session_state.settings['default_bleach_radius'],min_value=0.1,step=0.1,help="Radius of photobleached region")
        default_pixel_size=st.number_input("Pixel Size (μm/pixel)",value=st.session_state.settings['default_pixel_size'],min_value=0.01,step=0.01,help="Physical size of camera pixel")
        default_target_mw=st.number_input("Target Protein MW (kDa)",value=st.session_state.settings['default_target_mw'],min_value=1.0,step=1.0,help="Expected molecular weight for comparison")
    with col_exp2:
        default_scaling_alpha=st.number_input("Scaling Factor (α)",value=st.session_state.settings['default_scaling_alpha'],min_value=0.1,step=0.1,help="Correction factor for diffusion calculations")
        effective_bleach_size = default_bleach_radius * default_pixel_size
        st.metric("Effective Bleach Size", f"{effective_bleach_size:.2f} μm", help="Physical size of bleach spot")
        
        # Reference protein parameters
        st.markdown("**Reference Protein (GFP):**")
        st.text(f"D = {st.session_state.settings['default_gfp_diffusion']:.1f} μm²/s")
        st.text(f"MW = 27 kDa")
    
    st.markdown("### Advanced Curve Fitting Options")
    col_fit1, col_fit2 = st.columns(2)
    
    with col_fit1:
        fitting_method = st.selectbox(
            "Curve Fitting Method",
            ["least_squares", "robust", "bayesian"],
            index=0,
            format_func=lambda x: {
                "least_squares": "Standard Least Squares",
                "robust": "Robust Fitting (outlier resistant)",
                "bayesian": "Bayesian MCMC (full uncertainty)"
            }[x],
            help="Choose fitting algorithm for kinetic analysis"
        )
        
        max_iterations = st.number_input(
            "Max Fitting Iterations",
            value=2000,
            min_value=100,
            max_value=10000,
            step=100,
            help="Maximum iterations for curve fitting convergence"
        )
        
    with col_fit2:
        parameter_bounds = st.checkbox(
            "Use Parameter Bounds",
            value=True,
            help="Constrain parameters to physically reasonable ranges"
        )
        
        confidence_intervals = st.checkbox(
            "Calculate Confidence Intervals",
            value=False,
            help="Estimate parameter uncertainties (slower fitting)"
        )
        
        bootstrap_samples = st.number_input(
            "Bootstrap Samples",
            value=1000,
            min_value=100,
            max_value=5000,
            step=100,
            disabled=not confidence_intervals,
            help="Number of bootstrap samples for uncertainty estimation"
        )
    
    if st.button("Apply Settings",type="primary"):
        st.session_state.settings.update({
            'default_criterion':default_criterion,'default_gfp_diffusion':default_gfp_diffusion,'default_gfp_rg':default_gfp_rg,
            'default_bleach_radius':default_bleach_radius,'default_pixel_size':default_pixel_size,
            'default_scaling_alpha':default_scaling_alpha,'default_target_mw':default_target_mw,'decimal_places':decimal_places,
            'fitting_method': fitting_method, 'max_iterations': max_iterations,
            'parameter_bounds': parameter_bounds, 'confidence_intervals': confidence_intervals,
            'bootstrap_samples': bootstrap_samples
        })
        st.success("Settings applied successfully.")
        st.rerun()
    
    st.markdown("### Data Management")
    if st.checkbox("I understand that this will DELETE all loaded data and groups."):
        if st.button("Clear All Data",type="secondary"):
            st.session_state.data_manager=FRAPDataManager()
            st.session_state.selected_group_name=None
            st.success("All data cleared successfully.")
            st.rerun()
