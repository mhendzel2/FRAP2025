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
from scipy.optimize import curve_fit
from scipy.ndimage import minimum_position
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, Optional, Tuple, List
import logging
from frap_pdf_reports import generate_pdf_report
from frap_image_analysis import FRAPImageAnalyzer, create_image_analysis_interface
from frap_core import FRAPAnalysisCore as CoreFRAPAnalysis
from frap_reference_database import display_reference_database_ui
from frap_reference_integration import display_reference_comparison_widget
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
    Imports ROI data from an ImageJ .roi file.

    This function now uses the `roifile` library to parse .roi files.

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
        import roifile
        import io

        # Read the ROI data from bytes
        roi = roifile.roiread(io.BytesIO(roi_data))

        # Extract relevant information
        roi_info = {
            'name': roi.name,
            'type': roi.roitype.name,
            'left': roi.left,
            'top': roi.top,
            'right': roi.right,
            'bottom': roi.bottom,
            'width': roi.width,
            'height': roi.height,
            'coordinates': roi.coordinates().tolist() # Convert numpy array to list
        }

        logger.info(f"Successfully imported ROI: {roi.name} ({roi.roitype.name})")
        return roi_info

    except Exception as e:
        logger.error(f"ImageJ ROI import failed: {e}")
        st.error(f"Failed to import ROI file: {e}")
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
    fig.update_layout(
        title="All Individual Recovery Curves",
        xaxis_title="Time (s)",
        yaxis_title="Normalized Intensity",
        yaxis=dict(range=[0, None]),  # Start y-axis from zero
        legend_title="File"
    )
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
    fig.update_layout(
        title="Average FRAP Recovery Curve",
        xaxis_title="Time (s)",
        yaxis_title="Normalized Intensity",
        yaxis=dict(range=[0, None])  # Start y-axis from zero
    )
    return fig

# --- Core Analysis and Data Logic ---

def validate_analysis_results(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensures that the analysis results dictionary contains all required keys.
    Missing keys are added with a value of np.nan.
    """
    if params is None:
        return {}

    required_keys = [
        'mobile_fraction', 'immobile_fraction', 'rate_constant', 'half_time',
        'rate_constant_fast', 'rate_constant_medium', 'rate_constant_slow',
        'half_time_fast', 'half_time_medium', 'half_time_slow',
        'proportion_of_mobile_fast', 'proportion_of_mobile_medium', 'proportion_of_mobile_slow',
        'proportion_of_total_fast', 'proportion_of_total_medium', 'proportion_of_total_slow',
        'model', 'r2'
    ]

    for key in required_keys:
        if key not in params:
            params[key] = np.nan

    return params

class FRAPDataManager:
    def __init__(self):
        self.files,self.groups = {},{}

    def load_file(self,file_path,file_name):
        # Save the original file_path before any modifications
        original_file_path = file_path
        
        try:
            # Extract original extension before the hash suffix
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
                    file_path = temp_path  # Use temp file for loading

            processed_df = CoreFRAPAnalysis.preprocess(CoreFRAPAnalysis.load_data(file_path))
            if 'normalized' in processed_df.columns and not processed_df['normalized'].isnull().all():
                time,intensity = processed_df['time'].values,processed_df['normalized'].values

                # Validate the normalized data
                if np.any(intensity < 0):
                    logger.warning(f"Negative intensities found in normalized data for {file_name}")
                    # Shift to ensure all values are non-negative
                    intensity = intensity - np.min(intensity)

                # Ensure proper normalization (pre-bleach should be ~1.0)
                bleach_idx = np.argmin(intensity)
                if bleach_idx > 0:
                    pre_bleach_mean = np.mean(intensity[:bleach_idx])
                    if not np.isclose(pre_bleach_mean, 1.0, rtol=0.1):
                        logger.warning(f"Pre-bleach intensity not normalized to ~1.0 (got {pre_bleach_mean:.3f}) for {file_name}")
                        # Re-normalize if necessary
                        if pre_bleach_mean > 0:
                            intensity = intensity / pre_bleach_mean

                fits = CoreFRAPAnalysis.fit_all_models(time,intensity)
                best_fit = CoreFRAPAnalysis.select_best_fit(fits,st.session_state.settings['default_criterion'])

                if best_fit:
                    params = CoreFRAPAnalysis.extract_clustering_features(best_fit)
                    # Validate the analysis results
                    params = validate_analysis_results(params)
                else:
                    params = None
                    logger.error(f"No valid fit found for {file_name}")

                # Use the ORIGINAL file_path as the key, not the temp file path
                self.files[original_file_path]={
                    'name':file_name,'data':processed_df,'time':time,'intensity':intensity,
                    'fits':fits,'best_fit':best_fit,'features':params
                }
                logger.info(f"Loaded: {file_name}")
                return True
        except Exception as e:
            st.error(f"Error loading {file_name}: {e}")
            logger.error(f"Detailed error for {file_name}: {e}", exc_info=True)
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
        if group_name not in self.groups:
            st.error(f"DEBUG: Group '{group_name}' not found in groups!")
            return False
        if file_path not in self.files:
            st.error(f"DEBUG: File '{file_path}' not found in files! (load_file probably failed)")
            return False
        if file_path not in self.groups[group_name]['files']:
            self.groups[group_name]['files'].append(file_path)
            return True
        return False  # File already in group

    def fit_group_models(self, group_name, model='single', excluded_files=None):
        """
        Perform global simultaneous fitting for a group with shared kinetic parameters
        but individual amplitudes.

        Parameters:
        -----------
        group_name : str
            Name of the group to fit
        model : str
            Model type ('single', 'double', or 'triple')
        excluded_files : list, optional
            List of file paths to exclude from global fitting

        Returns:
        --------
        dict
            Dictionary containing global fit results
        """
        if group_name not in self.groups:
            raise KeyError(f"Group {group_name} not found.")

        group = self.groups[group_name]
        excluded_files = excluded_files or []

        # Prepare traces for global fitting
        traces = []
        file_names = []

        for file_path in group['files']:
            if file_path not in excluded_files and file_path in self.files:
                file_data = self.files[file_path]
                t, y, _ = CoreFRAPAnalysis.get_post_bleach_data(
                    file_data['time'],
                    file_data['intensity']
                )
                traces.append((t, y))
                file_names.append(file_data['name'])

        if len(traces) < 2:
            raise ValueError("Need at least 2 traces for global fitting")

        try:
            # Perform global fitting using the core analysis function
            global_fit_result = CoreFRAPAnalysis.fit_group_models(traces, model=model)

            # Add file names for reference
            global_fit_result['file_names'] = file_names
            global_fit_result['excluded_files'] = excluded_files

            # Store the result in the group
            if 'global_fit' not in group:
                group['global_fit'] = {}
            group['global_fit'][model] = global_fit_result

            return global_fit_result

        except Exception as e:
            logger.error(f"Error in global fitting for group {group_name}: {e}")
            return {
                'model': model,
                'success': False,
                'error': str(e)
            }

    def load_groups_from_zip_archive(self, zip_file):
        """
        Loads files from a ZIP archive containing subfolders, where each subfolder
        containing data files is treated as a new group. Handles deeply nested structures.
        """
        success_count = 0
        error_count = 0
        error_details = []
        groups_created = []

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                try:
                    with zipfile.ZipFile(io.BytesIO(zip_file.getbuffer())) as z:
                        z.extractall(temp_dir)
                        logger.info(f"Extracted ZIP archive to: {temp_dir}")
                except zipfile.BadZipFile:
                    logger.error("Invalid ZIP file format")
                    st.error("Invalid ZIP file format. Please check that the uploaded file is a valid ZIP archive.")
                    return False
                except Exception as e:
                    logger.error(f"Error extracting ZIP file: {e}")
                    st.error(f"Error extracting ZIP file: {e}")
                    return False

                # Walk through ALL directories to find folders containing data files
                folders_with_data = {}
                
                for root, dirs, files in os.walk(temp_dir):
                    # Check if this folder contains data files
                    data_files = [f for f in files if not f.startswith('.') and 
                                 os.path.splitext(f)[1].lower() in ['.xls', '.xlsx', '.csv', '.tif', '.tiff']]
                    
                    if data_files:
                        # This folder has data files - it should become a group
                        folder_name = os.path.basename(root)
                        
                        # Skip system folders
                        if folder_name.startswith('__') or folder_name.startswith('.'):
                            continue
                        
                        # Store folder info
                        folders_with_data[root] = {
                            'name': folder_name,
                            'files': data_files
                        }

                # DEBUG: Show what we found
                st.info(f"Found {len(folders_with_data)} folders with data files")
                
                # Process each folder with data files
                files_added_to_groups = 0
                files_failed_to_add = 0
                
                for folder_path, folder_info in folders_with_data.items():
                    group_name = folder_info['name']
                    
                    # Create group
                    self.create_group(group_name)
                    groups_created.append(group_name)
                    
                    # DEBUG: Show progress
                    st.write(f"Processing group: **{group_name}** ({len(folder_info['files'])} files)")
                    
                    # Process each data file in this folder
                    for file_in_group in folder_info['files']:
                        file_path_in_temp = os.path.join(folder_path, file_in_group)

                        if os.path.isfile(file_path_in_temp):
                            try:
                                file_ext = os.path.splitext(file_in_group)[1].lower()
                                file_name = os.path.basename(file_in_group)

                                with open(file_path_in_temp, 'rb') as f:
                                    file_content = f.read()
                                if not file_content:
                                    continue

                                content_hash = hash(file_content)

                                if file_ext in ['.tif', '.tiff']:
                                    base_name = os.path.splitext(file_name)[0]
                                    tp = f"data/{base_name}_{content_hash}.csv"
                                else:
                                    tp = f"data/{file_name}_{content_hash}"

                                if tp not in self.files:
                                    os.makedirs(os.path.dirname(tp), exist_ok=True)

                                    if file_ext in ['.tif', '.tiff']:
                                        analyzer = FRAPImageAnalyzer()
                                        if not analyzer.load_image_stack(file_path_in_temp):
                                            raise ValueError("Failed to load image stack.")

                                        settings = st.session_state.settings
                                        analyzer.pixel_size = settings.get('default_pixel_size', 0.3)
                                        analyzer.time_interval = settings.get('default_time_interval', 1.0)

                                        bleach_frame, bleach_coords = analyzer.detect_bleach_event()
                                        if bleach_frame is None or bleach_coords is None:
                                            raise ValueError("Failed to detect bleach event automatically.")

                                        bleach_radius_pixels = int(settings.get('default_bleach_radius', 1.0) / analyzer.pixel_size)
                                        analyzer.define_rois(bleach_coords, bleach_radius=bleach_radius_pixels)
                                        intensity_df = analyzer.extract_intensity_profiles()
                                        intensity_df = intensity_df.rename(columns={'Time': 'time'})
                                        intensity_df.to_csv(tp, index=False)
                                    else:
                                        shutil.copy(file_path_in_temp, tp)

                                    if self.load_file(tp, file_name):
                                        add_result = self.add_file_to_group(group_name, tp)
                                        if add_result:
                                            success_count += 1
                                            files_added_to_groups += 1
                                        else:
                                            files_failed_to_add += 1
                                            logger.error(f"DEBUG: File loaded but add_file_to_group returned False for {file_name}")
                                            logger.error(f"DEBUG: group_name='{group_name}', in self.groups={group_name in self.groups}")
                                            logger.error(f"DEBUG: tp='{tp}', in self.files={tp in self.files}")
                                    else:
                                        raise ValueError("Failed to load data from file.")
                                else:
                                    # File already exists, just add to group
                                    add_result = self.add_file_to_group(group_name, tp)
                                    if add_result:
                                        success_count += 1
                                        files_added_to_groups += 1
                                    else:
                                        files_failed_to_add += 1
                                        logger.error(f"DEBUG: File exists but add_file_to_group returned False for {file_name}")
                                        logger.error(f"DEBUG: group_name='{group_name}', in self.groups={group_name in self.groups}")
                                        logger.error(f"DEBUG: tp='{tp}', in self.files={tp in self.files}")

                            except Exception as e:
                                error_count += 1
                                error_details.append(f"Error processing file {file_in_group} in group {group_name}: {str(e)}")
                                logger.error(f"Error processing file {file_in_group} in group {group_name}: {str(e)}", exc_info=True)
                            
        except Exception as e:
            logger.error(f"Error processing ZIP archive with subfolders: {e}")
            st.error(f"An unexpected error occurred: {e}")
            return False

        if success_count > 0:
            # DEBUG: Show what happened
            st.write(f"**DEBUG Summary:**")
            st.write(f"- Files loaded successfully: {success_count}")
            st.write(f"- Files added to groups: {files_added_to_groups}")
            st.write(f"- Files failed to add: {files_failed_to_add}")
            st.write(f"- Groups created: {len(groups_created)}")
            
            for group_name in groups_created:
                self.update_group_analysis(group_name)
                # DEBUG: Show actual group contents
                if group_name in self.groups:
                    actual_files = len(self.groups[group_name].get('files', []))
                    st.write(f"  - Group '{group_name}': {actual_files} files in memory")
                    
            st.success(f"Successfully loaded {success_count} files into {len(groups_created)} groups.")
            if error_count > 0:
                st.warning(f"{error_count} files were skipped due to errors.")
                with st.expander("View Error Details"):
                    for error in error_details:
                        st.text(error)
            return True
        else:
            st.error("No files could be processed from the ZIP archive.")
            return False

    def load_zip_archive_and_create_group(self, zip_file, group_name):
        """
        Loads files from a ZIP archive, creates a new group, and adds the files to it.
        Gracefully handles unreadable files.
        """
        success_count = 0
        error_count = 0
        error_details = []

        try:
            self.create_group(group_name)
            with zipfile.ZipFile(io.BytesIO(zip_file.read())) as z:
                file_list = z.namelist()
                for file_in_zip in file_list:
                    # Ignore directories and hidden files (like __MACOSX)
                    if not file_in_zip.endswith('/') and '__MACOSX' not in file_in_zip:
                        try:
                            # Check if file has valid extension
                            file_ext = os.path.splitext(file_in_zip)[1].lower()
                            if file_ext not in ['.xls', '.xlsx', '.csv']:
                                logger.warning(f"Skipping unsupported file type: {file_in_zip}")
                                continue

                            file_content = z.read(file_in_zip)

                            # Skip empty files
                            if len(file_content) == 0:
                                logger.warning(f"Skipping empty file: {file_in_zip}")
                                continue

                            file_name = os.path.basename(file_in_zip)

                            # Create a temporary file to use with the existing load_file logic
                            tp = f"data/{file_name}_{hash(file_content)}"
                            if tp not in self.files:
                                with open(tp, "wb") as f:
                                    f.write(file_content)

                                # Attempt to load the file
                                if self.load_file(tp, file_name):
                                    self.add_file_to_group(group_name, tp)
                                    success_count += 1
                                else:
                                    error_count += 1
                                    error_msg = f"Failed to load file: {file_in_zip}"
                                    error_details.append(error_msg)
                                    logger.error(error_msg)
                                    # Clean up failed file
                                    if os.path.exists(tp):
                                        os.remove(tp)
                            else:
                                # File already exists, just add to group
                                self.add_file_to_group(group_name, tp)
                                success_count += 1

                        except Exception as e:
                            error_count += 1
                            error_msg = f"Error processing file {file_in_zip}: {str(e)}"
                            error_details.append(error_msg)
                            logger.error(error_msg)
                            continue

            # Update group analysis only if we have successfully loaded files
            if success_count > 0:
                self.update_group_analysis(group_name)
                st.session_state.selected_group_name = group_name

                # Report results
                logger.info(f"Successfully processed {success_count} files for group {group_name}")
                if error_count > 0:
                    logger.warning(f"{error_count} files could not be processed and were skipped")
                    # Display error details in Streamlit
                    if hasattr(st, 'warning'):
                        st.warning(f"Successfully loaded {success_count} files to group '{group_name}'. "
                                 f"{error_count} files were skipped due to errors.")
                        if error_details:
                            with st.expander("⚠️ View Skipped Files Details"):
                                for error in error_details[:10]:  # Show first 10 errors
                                    st.text(f"• {error}")
                                if len(error_details) > 10:
                                    st.text(f"... and {len(error_details) - 10} more errors")
                return True
            else:
                # No files could be loaded, remove the empty group
                logger.error(f"No files could be processed for group {group_name}")
                if group_name in self.groups:
                    del self.groups[group_name]
                return False

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
    st.markdown("""
    **Expected ZIP Structure:**
    ```
    your_archive.zip
    ├── Group1/
    │   ├── file1.xls
    │   └── file2.xls
    ├── Group2/
    │   ├── file3.xlsx
    │   └── file4.csv
    └── Group3/
        └── file5.xls
    ```
    Each subfolder will become a group, and files within will be added to that group.
    """)
    uploaded_zip = st.file_uploader("Upload a .zip file with group subfolders", type=['zip'], key="zip_uploader")

    if uploaded_zip:
        if st.button(f"Create Groups from '{uploaded_zip.name}'"):
            with st.spinner(f"Processing groups from '{uploaded_zip.name}'..."):
                if 'data_manager' in st.session_state:
                    # Clear any existing progress messages
                    success = dm.load_groups_from_zip_archive(uploaded_zip)
                    if success:
                        # Show successful groups
                        if dm.groups:
                            st.success("Successfully created {} groups from ZIP archive:".format(len(dm.groups)))
                            for group_name, group_data in dm.groups.items():
                                file_count = len(group_data.get('files', []))
                                st.write(f"📁 **{group_name}**: {file_count} files")

                            # Show summary of what was processed
                            total_files = sum(len(group_data.get('files', [])) for group_data in dm.groups.values())
                            st.info(f"Total files processed: {total_files}")

                            # Show detailed breakdown
                            with st.expander("📋 View Detailed Breakdown"):
                                for group_name, group_data in dm.groups.items():
                                    st.write(f"**{group_name}**:")
                                    files_in_group = group_data.get('files', [])
                                    for file_path in files_in_group:
                                        if file_path in dm.files:
                                            file_name = dm.files[file_path]['name']
                                            st.write(f"  • {file_name}")
                                        else:
                                            st.write(f"  • {file_path} (file not found)")
                        else:
                            st.warning("ZIP archive was processed but no groups were created.")
                            st.info("This might happen if:")
                            st.write("- The ZIP file contains no subfolders")
                            st.write("- All files were filtered out due to unsupported formats")
                            st.write("- Files could not be loaded due to format issues")
                        st.rerun()
                    else:
                        st.error("Failed to process ZIP archive with subfolders. Please check the ZIP file structure and file formats.")

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

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📊 Single File Analysis", "📈 Group Analysis", "📊 Multi-Group Comparison", "🖼️ Image Analysis", "💾 Session Management", "⚙️ Settings", "📚 Reference Database"])

with tab1:
    st.header("Single File Analysis")
    if dm.files:
        selected_file_path = st.selectbox("Select file to analyze", list(dm.files.keys()), format_func=lambda p: dm.files[p]['name'])
        if selected_file_path and selected_file_path in dm.files:
            file_data=dm.files[selected_file_path]
            st.subheader(f"Results for: {file_data['name']}")
            if file_data['best_fit']:
                best_fit,features=file_data['best_fit'],file_data['features']
                t_fit,intensity_fit,_=CoreFRAPAnalysis.get_post_bleach_data(file_data['time'],file_data['intensity'])

                # Enhanced metrics display with validation warnings
                st.markdown("### Kinetic Analysis Results")

                # Check for problematic results and show warnings
                mobile_frac = features.get('mobile_fraction', 0)
                if mobile_frac > 100:
                    st.error(f"⚠️ Warning: Mobile fraction ({mobile_frac:.1f}%) exceeds 100% - this indicates a problem with the analysis")
                elif mobile_frac < 0:
                    st.error(f"⚠️ Warning: Mobile fraction ({mobile_frac:.1f}%) is negative - this indicates a problem with the analysis")

                # Check data quality
                r2 = best_fit.get('r2', 0)
                if r2 < 0.8:
                    st.warning(f"⚠️ Warning: Low R² ({r2:.3f}) indicates poor curve fit quality")

                col1,col2,col3,col4=st.columns(4)
                with col1:
                    # Ensure mobile fraction is displayed correctly
                    display_mobile = max(0, min(100, mobile_frac))  # Clamp to 0-100%
                    st.metric("Mobile Fraction",f"{display_mobile:.1f}%")
                with col2:
                    half_time = features.get('half_time_fast',features.get('half_time',0))
                    st.metric("Half-time",f"{half_time:.1f} s" if np.isfinite(half_time) and half_time > 0 else "N/A")
                with col3:
                    st.metric("R²",f"{r2:.3f}")
                with col4:
                    st.metric("Model",best_fit['model'].title())

                # Additional goodness-of-fit metrics
                col5,col6,col7,col8=st.columns(4)
                with col5:
                    aic_val = best_fit.get('aic', np.nan)
                    st.metric("AIC",f"{aic_val:.1f}" if np.isfinite(aic_val) else "N/A")
                with col6:
                    adj_r2_val = best_fit.get('adj_r2', np.nan)
                    st.metric("Adj. R²",f"{adj_r2_val:.3f}" if np.isfinite(adj_r2_val) else "N/A")
                with col7:
                    bic_val = best_fit.get('bic', np.nan)
                    st.metric("BIC",f"{bic_val:.1f}" if np.isfinite(bic_val) else "N/A")
                with col8:
                    red_chi2_val = best_fit.get('red_chi2', np.nan)
                    st.metric("Red. χ²",f"{red_chi2_val:.3f}" if np.isfinite(red_chi2_val) else "N/A")

                # Add data quality assessment
                st.markdown("### Data Quality Assessment")

                # Check normalization
                bleach_idx = np.argmin(file_data['intensity'])
                if bleach_idx > 0:
                    pre_bleach_mean = np.mean(file_data['intensity'][:bleach_idx])
                    post_bleach_min = np.min(file_data['intensity'][bleach_idx:])

                    col_qual1, col_qual2, col_qual3 = st.columns(3)
                    with col_qual1:
                        st.metric("Pre-bleach Level", f"{pre_bleach_mean:.3f}")
                        if not np.isclose(pre_bleach_mean, 1.0, rtol=0.1):
                            st.warning("Pre-bleach should be ~1.0")

                    with col_qual2:
                        st.metric("Bleach Depth", f"{post_bleach_min:.3f}")
                        if post_bleach_min < 0:
                            st.error("Negative intensities detected")

                    with col_qual3:
                        bleach_efficiency = (pre_bleach_mean - post_bleach_min) / pre_bleach_mean * 100
                        st.metric("Bleach Efficiency", f"{bleach_efficiency:.1f}%")
                        if bleach_efficiency < 10:
                            st.warning("Low bleach efficiency (<10%)")
                        elif bleach_efficiency > 90:
                            st.warning("Very high bleach efficiency (>90%)")

                # Main recovery curve plot with proper FRAP visualization
                fig = go.Figure()

                # Get the bleach frame index and interpolated bleach time
                bleach_idx = np.argmin(file_data['intensity'])

                # Calculate the interpolated bleach time (same as in get_post_bleach_data)
                if bleach_idx > 0:
                    interpolated_bleach_time = (file_data['time'][bleach_idx-1] + file_data['time'][bleach_idx]) / 2.0
                else:
                    interpolated_bleach_time = file_data['time'][bleach_idx]

                # Pre-bleach data (shown but not fitted)
                pre_bleach_time = file_data['time'][:bleach_idx]
                pre_bleach_intensity = file_data['intensity'][:bleach_idx]

                # Add pre-bleach data (gray markers)
                if len(pre_bleach_time) > 0:
                    fig.add_trace(go.Scatter(
                        x=pre_bleach_time,
                        y=pre_bleach_intensity,
                        mode='markers',
                        name='Pre-bleach (not fitted)',
                        marker=dict(color='lightgray', size=6, opacity=0.7),
                        showlegend=True
                    ))

                # Convert fitted timepoints back to original time scale for plotting
                # t_fit starts from 0, so we add the interpolated bleach time
                t_fit_original_scale = t_fit + interpolated_bleach_time

                # Add post-bleach data (blue markers) - use the actual fitted data points
                fig.add_trace(go.Scatter(
                    x=t_fit_original_scale,
                    y=intensity_fit,
                    mode='markers',
                    name='Post-bleach (fitted)',
                    marker=dict(color='blue', size=6),
                    showlegend=True
                ))

                # Add the fitted curve starting from the interpolated point
                # Convert fitted curve timepoints to original scale
                fig.add_trace(go.Scatter(
                    x=t_fit_original_scale,
                    y=best_fit['fitted_values'],
                    mode='lines',
                    name=f"Fit: {best_fit['model'].title()}",
                    line=dict(color='red', width=3),
                    showlegend=True
                ))

                # Add a vertical line at the interpolated bleach time
                # Use the interpolated bleach time we calculated above
                bleach_time = interpolated_bleach_time
                min_intensity = 0  # Start y-axis from zero
                max_intensity = max(np.max(file_data['intensity']), np.max(intensity_fit))

                fig.add_shape(
                    type="line",
                    x0=bleach_time, y0=min_intensity,
                    x1=bleach_time, y1=max_intensity,
                    line=dict(color="orange", width=2, dash="dash"),
                )

                # Add annotation for bleach event
                fig.add_annotation(
                    x=bleach_time,
                    y=max_intensity * 0.9,
                    text="Bleach Event",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="orange",
                    font=dict(color="orange")
                )

                # Update layout with proper scaling
                fig.update_layout(
                    title='FRAP Recovery Curve',
                    xaxis_title='Time (s)',
                    yaxis_title='Normalized Intensity',
                    height=450,
                    yaxis=dict(range=[0, max_intensity * 1.05]),  # Start from 0, add 5% headroom
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )

                st.plotly_chart(fig, use_container_width=True)

                # Add explanation text
                st.markdown("""
                **Plot Explanation:**
                - **Gray points**: Pre-bleach data (shown for context, not included in fitting)
                - **Blue points**: Post-bleach data starting from interpolated bleach point (used for curve fitting)
                - **Red curve**: Fitted exponential recovery model aligned with post-bleach timepoints
                - **Orange dashed line**: Interpolated bleach event (midpoint between last pre-bleach and first post-bleach frames)
                - **Y-axis**: Starts at zero for proper scaling
                """)

                # Add comprehensive multi-panel plot option
                st.markdown("---")
                st.markdown("### 📊 Comprehensive Analysis View")
                st.markdown("View recovery curve, residuals, and individual components in a single integrated plot")
                
                if st.button("🔍 Show Comprehensive Analysis Plot", key="comprehensive_plot_btn"):
                    from frap_plots import FRAPPlots
                    
                    comprehensive_fig = FRAPPlots.plot_comprehensive_fit(
                        time=t_fit,
                        intensity=intensity_fit,
                        fit_result=best_fit,
                        file_name=file_data['name'],
                        height=800
                    )
                    
                    if comprehensive_fig:
                        st.plotly_chart(comprehensive_fig, use_container_width=True)
                        st.success("✅ Comprehensive analysis plot generated successfully!")
                        
                        st.markdown("""
                        **Comprehensive Plot Features:**
                        - **Top Panel**: Recovery curve with fitted model and parameter annotations
                        - **Middle Panel**: Residuals plot showing fit quality (should be randomly scattered around zero)
                        - **Bottom Panel**: Individual exponential components (for multi-component fits)
                        
                        This integrated view makes it easy to assess:
                        - Overall fit quality
                        - Presence of systematic errors in residuals
                        - Contribution of each kinetic component
                        """)
                    else:
                        st.error("Failed to generate comprehensive plot")

                # Component-wise recovery analysis for multi-component fits
                if best_fit['model'] in ['double', 'triple']:
                    st.markdown("### Component-wise Recovery Analysis")

                    comp_fig = go.Figure()

                    # Add original data
                    comp_fig.add_trace(go.Scatter(
                        x=t_fit, y=intensity_fit, mode='markers', name='Data',
                        marker=dict(color='blue', size=4)
                    ))

                    # Add total fit
                    comp_fig.add_trace(go.Scatter(
                        x=t_fit, y=best_fit['fitted_values'], mode='lines', name='Total Fit',
                        line=dict(color='red', width=3)
                    ))

                    # Calculate and plot individual components
                    model_func = best_fit['func']
                    params = best_fit['params']

                    if best_fit['model'] == 'double':
                        A1, k1, A2, k2, C = params
                        comp1 = A1 * (1 - np.exp(-k1 * t_fit)) + C
                        comp2 = A2 * (1 - np.exp(-k2 * t_fit)) + C

                        comp_fig.add_trace(go.Scatter(
                            x=t_fit, y=comp1, mode='lines', name=f'Fast (k={k1:.4f})',
                            line=dict(dash='dot', color='green')
                        ))
                        comp_fig.add_trace(go.Scatter(
                            x=t_fit, y=comp2, mode='lines', name=f'Slow (k={k2:.4f})',
                            line=dict(dash='dot', color='purple')
                        ))

                    elif best_fit['model'] == 'triple':
                        A1, k1, A2, k2, A3, k3, C = params
                        comp1 = A1 * (1 - np.exp(-k1 * t_fit)) + C
                        comp2 = A2 * (1 - np.exp(-k2 * t_fit)) + C
                        comp3 = A3 * (1 - np.exp(-k3 * t_fit)) + C

                        comp_fig.add_trace(go.Scatter(
                            x=t_fit, y=comp1, mode='lines', name=f'Fast (k={k1:.4f})',
                            line=dict(dash='dot', color='green')
                        ))
                        comp_fig.add_trace(go.Scatter(
                            x=t_fit, y=comp2, mode='lines', name=f'Medium (k={k2:.4f})',
                            line=dict(dash='dot', color='blue')
                        ))
                        comp_fig.add_trace(go.Scatter(
                            x=t_fit, y=comp3, mode='lines', name=f'Slow (k={k3:.4f})',
                            line=dict(dash='dot', color='purple')
                        ))

                    comp_fig.update_layout(
                        title="Component-wise Recovery Analysis",
                        xaxis_title="Time (s)",
                        yaxis_title="Normalized Intensity",
                        height=400,
                        yaxis=dict(range=[0, np.max(intensity_fit) * 1.05])
                    )
                    st.plotly_chart(comp_fig, use_container_width=True)

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

                # Biophysical parameters
                st.markdown("### Biophysical Interpretation")

                # Calculate diffusion coefficient and molecular weight estimates
                features = file_data.get('features', {})
                primary_rate=features.get('rate_constant_fast',features.get('rate_constant',0))

                # More robust validation of rate constant
                if primary_rate is not None and np.isfinite(primary_rate) and primary_rate > 1e-8:
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

                    # Validate calculated values
                    if diffusion_coeff > 100:
                        st.warning("⚠️ Very high apparent diffusion coefficient - check bleach spot size")
                    if apparent_mw > 10000:
                        st.warning("⚠️ Very high apparent molecular weight - may indicate aggregation")

                    col_bio1,col_bio2,col_bio3,col_bio4=st.columns(4)
                    with col_bio1:
                        st.metric("App. D (μm²/s)",f"{diffusion_coeff:.3f}")
                    with col_bio2:
                        st.metric("k_off (s⁻¹)",f"{k_off:.4f}")
                    with col_bio3:
                        st.metric("App. MW (kDa)",f"{apparent_mw:.1f}")
                    with col_bio4:
                        immobile_frac = features.get('immobile_fraction', 100 - display_mobile)
                        st.metric("Immobile (%)",f"{immobile_frac:.1f}")
                
                # Advanced Kinetic Models Section
                st.markdown("---")
                st.markdown("### 🔬 Advanced Kinetic Models")
                st.markdown("""
                Go beyond standard exponential models with advanced kinetic analysis:
                - **Anomalous Diffusion**: For crowded/heterogeneous environments
                - **Reaction-Diffusion**: Separate diffusion from binding kinetics
                """)
                
                if st.checkbox("Run Advanced Kinetic Analysis", key="run_advanced_models"):
                    from frap_core import ADVANCED_FITTING_AVAILABLE
                    
                    if not ADVANCED_FITTING_AVAILABLE:
                        st.error("⚠️ Advanced fitting requires the `lmfit` library")
                        st.code("pip install lmfit", language="bash")
                        st.markdown("[Install lmfit documentation](https://lmfit.github.io/lmfit-py/)")
                    else:
                        with st.spinner("Fitting advanced models... (this may take 30-60 seconds)"):
                            bleach_radius = st.session_state.settings.get('default_bleach_radius', 1.0)
                            pixel_size = st.session_state.settings.get('default_pixel_size', 1.0)
                            
                            advanced_results = CoreFRAPAnalysis.fit_advanced_models(
                                file_data['time'],
                                file_data['intensity'],
                                bleach_radius,
                                pixel_size
                            )
                        
                        if advanced_results:
                            st.success(f"✅ Successfully fitted {len(advanced_results)} advanced model(s)")
                            
                            # Model selection tabs
                            model_names = [r['model_name'].replace('_', ' ').title() for r in advanced_results]
                            adv_tabs = st.tabs(model_names)
                            
                            for tab, result in zip(adv_tabs, advanced_results):
                                with tab:
                                    # Model quality metrics
                                    col_qual1, col_qual2, col_qual3, col_qual4 = st.columns(4)
                                    
                                    with col_qual1:
                                        st.metric("R²", f"{result['r2']:.4f}")
                                    with col_qual2:
                                        st.metric("AIC", f"{result['aic']:.1f}")
                                    with col_qual3:
                                        st.metric("BIC", f"{result['bic']:.1f}")
                                    with col_qual4:
                                        # Compare with standard model
                                        r2_standard = best_fit.get('r2', 0)
                                        improvement = result['r2'] - r2_standard
                                        st.metric("R² vs Standard", f"{improvement:+.4f}")
                                    
                                    # Biological interpretation
                                    st.markdown("#### 🧬 Biological Interpretation")
                                    
                                    interp = result['interpretation']
                                    
                                    if 'anomalous' in result['model_name']:
                                        col_int1, col_int2, col_int3 = st.columns(3)
                                        
                                        with col_int1:
                                            st.metric("Mobile Fraction", f"{interp['mobile_fraction']:.1f}%")
                                            st.metric("β (Anomalous Exponent)", f"{interp['beta']:.3f}")
                                        
                                        with col_int2:
                                            st.metric("τ (Char. Time)", f"{interp['tau']:.2f} s")
                                            st.metric("Effective D", f"{interp['effective_D']:.3f} μm²/s")
                                        
                                        with col_int3:
                                            st.metric("Diffusion Type", interp['diffusion_type'])
                                            st.metric("Anomaly Strength", f"{interp['anomaly_strength']:.3f}")
                                        
                                        # Explanation
                                        st.markdown("**What does β tell you?**")
                                        beta = interp['beta']
                                        if beta < 0.7:
                                            st.warning(f"**Subdiffusion (β={beta:.2f})**: Protein movement is significantly hindered by obstacles, crowding, or transient interactions. This suggests a complex, heterogeneous environment.")
                                        elif beta < 0.95:
                                            st.info(f"**Mild Subdiffusion (β={beta:.2f})**: Some hindrance to free diffusion, but not severe. May indicate moderate crowding or weak interactions.")
                                        elif beta <= 1.05:
                                            st.success(f"**Normal Diffusion (β≈1)**: Protein diffuses freely without significant obstacles. Classic Brownian motion.")
                                        else:
                                            st.info(f"**Superdiffusion (β={beta:.2f})**: Faster than normal diffusion, possibly due to active transport or directed motion.")
                                    
                                    elif 'reaction_diffusion' in result['model_name']:
                                        col_int1, col_int2 = st.columns(2)
                                        
                                        with col_int1:
                                            st.metric("Mobile Fraction", f"{interp['mobile_fraction']:.1f}%")
                                            st.metric("Immobile Fraction", f"{interp['immobile_fraction']:.1f}%")
                                            
                                            if 'bound_fraction' in interp:
                                                st.metric("Bound (of Mobile)", f"{interp['bound_fraction']:.1f}%")
                                                st.metric("Free (of Mobile)", f"{interp['free_fraction']:.1f}%")
                                        
                                        with col_int2:
                                            if 'k_on' in interp:
                                                st.metric("k_on (Binding Rate)", f"{interp['k_on']:.4f} s⁻¹")
                                                st.metric("k_off (Unbinding Rate)", f"{interp['k_off']:.4f} s⁻¹")
                                                st.metric("K_d (Dissociation)", f"{interp['K_d']:.4f}")
                                            else:
                                                st.metric("k_eff (Effective Rate)", f"{interp['effective_k']:.4f} s⁻¹")
                                                st.metric("Residence Time", f"{interp['residence_time']:.2f} s")
                                            
                                            if 'diffusion_coeff' in interp:
                                                st.metric("D (Free Diffusion)", f"{interp['diffusion_coeff']:.3f} μm²/s")
                                            else:
                                                st.metric("D_app (Apparent)", f"{interp['apparent_D']:.3f} μm²/s")
                                        
                                        # Interpretation
                                        st.markdown("**Binding vs. Diffusion:**")
                                        if 'bound_fraction' in interp:
                                            bound_pct = interp['bound_fraction']
                                            if bound_pct > 70:
                                                st.warning(f"**Binding-Dominated ({bound_pct:.0f}%)**: Protein spends most time bound to chromatin/structures. Slow recovery driven by unbinding kinetics.")
                                            elif bound_pct > 40:
                                                st.info(f"**Mixed Regime ({bound_pct:.0f}%)**: Significant contributions from both binding and diffusion. Protein alternates between bound and free states.")
                                            else:
                                                st.success(f"**Diffusion-Dominated ({100-bound_pct:.0f}% free)**: Protein is mostly freely diffusing. Fast recovery with minimal binding.")
                                    
                                    # Plot the fit
                                    st.markdown("#### 📈 Model Fit Visualization")
                                    
                                    fig_adv = go.Figure()
                                    
                                    # Get post-bleach data for proper alignment
                                    t_post, i_post, bleach_idx = CoreFRAPAnalysis.get_post_bleach_data(
                                        file_data['time'], 
                                        file_data['intensity']
                                    )
                                    
                                    # Plot data
                                    fig_adv.add_trace(go.Scatter(
                                        x=t_post,
                                        y=i_post,
                                        mode='markers',
                                        name='Data',
                                        marker=dict(size=6, color='blue', opacity=0.6)
                                    ))
                                    
                                    # Plot advanced fit
                                    fig_adv.add_trace(go.Scatter(
                                        x=t_post,
                                        y=result['fitted_values'],
                                        mode='lines',
                                        name=f'{result["model_name"].replace("_", " ").title()} Fit',
                                        line=dict(color='green', width=3)
                                    ))
                                    
                                    # Plot standard fit for comparison
                                    if 'fitted_values' in best_fit:
                                        fig_adv.add_trace(go.Scatter(
                                            x=t_post,
                                            y=best_fit['fitted_values'],
                                            mode='lines',
                                            name=f'Standard {best_fit["model"].title()} Fit',
                                            line=dict(color='red', width=2, dash='dash'),
                                            opacity=0.5
                                        ))
                                    
                                    fig_adv.update_layout(
                                        title=f'{result["model_name"].replace("_", " ").title()} Model Fit',
                                        xaxis_title='Time (s)',
                                        yaxis_title='Normalized Intensity',
                                        height=400,
                                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                                    )
                                    
                                    st.plotly_chart(fig_adv, use_container_width=True)
                                    
                                    # Residuals
                                    st.markdown("#### Residuals Analysis")
                                    
                                    fig_res = go.Figure()
                                    fig_res.add_trace(go.Scatter(
                                        x=t_post,
                                        y=result['residuals'],
                                        mode='markers',
                                        marker=dict(size=5, color='purple'),
                                        name='Residuals'
                                    ))
                                    fig_res.add_hline(y=0, line_dash="dash", line_color="gray")
                                    fig_res.update_layout(
                                        title='Residuals (should be randomly scattered)',
                                        xaxis_title='Time (s)',
                                        yaxis_title='Residual',
                                        height=250
                                    )
                                    st.plotly_chart(fig_res, use_container_width=True)
                                    
                                    # Parameter details
                                    with st.expander("📊 View All Parameters & Errors"):
                                        st.markdown("**Fitted Parameters:**")
                                        params_df_data = []
                                        for param_name, param_value in result['params'].items():
                                            error = result['param_errors'].get(param_name, None)
                                            params_df_data.append({
                                                'Parameter': param_name,
                                                'Value': f"{param_value:.6f}",
                                                'Std Error': f"{error:.6f}" if error is not None else "Fixed"
                                            })
                                        st.dataframe(pd.DataFrame(params_df_data), use_container_width=True)
                        
                        else:
                            st.warning("❌ Could not fit advanced models. This may be due to:")
                            st.markdown("- Insufficient data points")
                            st.markdown("- Poor data quality (high noise)")
                            st.markdown("- Inappropriate model for this dataset")
                            st.markdown("Try standard exponential models first to ensure data quality.")

                else:
                    # More informative error message with diagnostic information
                    debug_info = []
                    if primary_rate is None:
                        debug_info.append("Rate constant is None")
                    elif np.isnan(primary_rate):
                        debug_info.append("Rate constant is NaN")
                    elif primary_rate <= 0:
                        debug_info.append(f"Rate constant is non-positive ({primary_rate:.6f})")
                    elif primary_rate <= 1e-8:
                        debug_info.append(f"Rate constant is too small ({primary_rate:.2e})")

                    st.error(f"Cannot calculate biophysical parameters - invalid rate constant: {'; '.join(debug_info)}")

                    # Display available parameters for debugging
                    with st.expander("🔍 Debug Information"):
                        st.write("**Available parameters:**")
                        # Fix: Check if params is a dict before calling .items()
                        if isinstance(params, dict):
                            for key, value in params.items():
                                if 'rate' in key.lower() or 'constant' in key.lower():
                                    st.write(f"- {key}: {value}")
                        else:
                            st.write(f"Parameters is not a dict (type: {type(params).__name__})")
                            st.write(f"Value: {params}")
                        st.write("**Model information:**")
                        if best_fit:
                            st.write(f"- Model: {best_fit.get('model', 'Unknown')}")
                            st.write(f"- R²: {best_fit.get('r2', 'N/A')}")
                            st.write(f"- Parameters: {best_fit.get('params', 'N/A')}")

                        # Show raw fitting parameters for debugging
                        if 'params' in best_fit:
                            st.write("**Raw fitting parameters:**")
                            model = best_fit.get('model', 'unknown')
                            params_raw = best_fit['params']
                            if model == 'single' and len(params_raw) >= 3:
                                A, k, C = params_raw[:3]
                                st.write(f"- Amplitude (A): {A}")
                                st.write(f"- Rate constant (k): {k}")
                                st.write(f"- Offset (C): {C}")
                                st.write(f"- Mobile population calc: (1 - (A + C)) * 100 = {(1 - (A + C))*100 if np.isfinite(A) and np.isfinite(C) else 'undefined'}")
                            elif model == 'double' and len(params_raw) >= 5:
                                A1, k1, A2, k2, C = params_raw[:5]
                                st.write(f"- A1: {A1}, k1: {k1}")
                                st.write(f"- A2: {A2}, k2: {k2}")
                                st.write(f"- Offset (C): {C}")
                                total_A = A1 + A2
                                st.write(f"- Total amplitude: {total_A}")
                                st.write(f"- Mobile population calc: (1 - (ΣA + C)) * 100 = {(1 - (total_A + C))*100 if np.isfinite(total_A) and np.isfinite(C) else 'undefined'}")
                            elif model == 'triple' and len(params_raw) >= 6:
                                A1, k1, A2, k2, A3, k3, C = params_raw[:6]
                                st.write(f"- A1: {A1}, k1: {k1}")
                                st.write(f"- A2: {A2}, k2: {k2}")
                                st.write(f"- A3: {A3}, k3: {k3}")
                                st.write(f"- Offset (C): {C}")
                                total_A = A1 + A2 + A3
                                st.write(f"- Total amplitude: {total_A}")
                                st.write(f"- Mobile population calc: (1 - (ΣA + C)) * 100 = {(1 - (total_A + C))*100 if np.isfinite(total_A) and np.isfinite(C) else 'undefined'}")
            else:
                st.error("Could not determine a best fit for this file.")

                # Show debugging information for failed fits
                if file_data.get('fits'):
                    st.markdown("### Available Fits (Debug)")
                    for i, fit in enumerate(file_data['fits']):
                        st.write(f"**Fit {i+1} ({fit.get('model', 'unknown')}):**")
                        st.write(f"- R²: {fit.get('r2', 'N/A')}")
                        st.write(f"- AIC: {fit.get('aic', 'N/A')}")
                        st.write(f"- Parameters: {fit.get('params', 'N/A')}")
        
        # Add reference database comparison for single file analysis when results are available
        if selected_file_path and selected_file_path in dm.files:
            file_features = dm.files[selected_file_path].get('features', {})
            if file_features and file_features.get('mobile_fraction') is not None:
                # Get diffusion coefficient from kinetic interpretation
                kinetic_interp = dm.files[selected_file_path].get('kinetic_interpretation', {})
                if kinetic_interp and kinetic_interp.get('diffusion_coefficient', 0) > 0:
                    display_reference_comparison_widget(
                        experimental_deff=kinetic_interp.get('diffusion_coefficient'),
                        experimental_mf=file_features.get('mobile_fraction', 0),
                        protein_name=selected_file_path.replace('.csv', '').replace('_', ' ') if selected_file_path else None,
                        key_suffix=f"_single_{selected_file_path}"
                    )
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
                
                # Add reference database comparison for group analysis
                if mean_vals.get('mobile_fraction') and filtered_df['diffusion_coefficient'].mean() > 0:
                    avg_deff = filtered_df['diffusion_coefficient'].mean()
                    avg_mf = mean_vals.get('mobile_fraction', 0)
                    display_reference_comparison_widget(
                        experimental_deff=avg_deff,
                        experimental_mf=avg_mf,
                        protein_name=selected_group_name if selected_group_name else None,
                        key_suffix=f"_group_{selected_group_name}"
                    )

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
                st.markdown("### Step 3: Interactive Individual Curve Analysis")

                # Enhanced plot of all individual curves with outliers highlighted
                st.markdown("#### All Individual Curves - Interactive Selection")
                st.info("💡 **Interactive Feature**: Use the checkboxes below to include/exclude curves from analysis. Changes update the plot and statistics in real-time.")
                
                # Track which files should be excluded (start with current excluded_paths)
                if 'interactive_excluded_files' not in st.session_state:
                    st.session_state.interactive_excluded_files = set(excluded_paths)
                
                # Create interface for curve selection
                st.markdown("#### 📁 File Selection Panel")
                
                # Control buttons in columns
                control_cols = st.columns(4)
                with control_cols[0]:
                    if st.button("✅ Include All", help="Include all curves in the analysis"):
                        st.session_state.interactive_excluded_files = set()
                        st.rerun()
                
                with control_cols[1]:
                    if st.button("❌ Exclude All", help="Exclude all curves from analysis"):
                        st.session_state.interactive_excluded_files = set(group['files'])
                        st.rerun()
                
                with control_cols[2]:
                    if st.button("🔄 Reset to Auto-Outliers", help="Reset to automatically detected outliers"):
                        st.session_state.interactive_excluded_files = set(excluded_paths)
                        st.rerun()
                
                with control_cols[3]:
                    if st.button("💾 Apply Selection", type="primary", help="Update the group analysis with current selection"):
                        # Update the group's excluded files
                        new_excluded_paths = list(st.session_state.interactive_excluded_files)
                        dm.update_group_analysis(selected_group_name, excluded_files=new_excluded_paths)
                        st.success(f"✅ Updated analysis! Now using {len(group['files']) - len(new_excluded_paths)} curves.")
                        st.rerun()
                
                # Automatic slope detection (simpler and more direct)
                try:
                    from frap_slope_detection import create_slope_detection_interface
                    create_slope_detection_interface(group, dm, selected_group_name)
                except Exception as e:
                    st.warning(f"Slope detection not available: {e}")
                
                # Statistical outlier detection (IQR-based filtering for robust averages)
                try:
                    from frap_statistical_outliers import create_statistical_outlier_interface
                    create_statistical_outlier_interface(group, dm, selected_group_name)
                except Exception as e:
                    st.warning(f"Statistical outlier detection not available: {e}")
                
                # Enhanced interactive selection interface
                try:
                    from frap_interactive_selection import handle_curve_click_selection, create_hover_selection_plot
                    
                    # Show the enhanced hover plot for quality assessment
                    st.markdown("---")
                    st.markdown("### 📊 Curve Quality Assessment")
                    st.markdown("**Color-coded visualization to help identify problematic curves**")
                    
                    excluded_files = st.session_state.interactive_excluded_files
                    hover_fig = create_hover_selection_plot(group, dm, excluded_files)
                    st.plotly_chart(hover_fig, use_container_width=True)
                    
                    # Interactive selection interface
                    handle_curve_click_selection(group, dm, selected_group_name)
                    
                except ImportError:
                    st.info("💡 Enhanced interactive selection not available")
                except Exception as e:
                    st.warning(f"Interactive selection not available: {e}")
                
                # Advanced ML outlier detection (optional)
                with st.expander("🤖 Advanced ML Outlier Detection", expanded=False):
                    try:
                        from frap_ml_outliers import create_ml_outlier_interface
                        st.info("💡 Advanced option: Machine learning-based outlier detection using 48 extracted features")
                        create_ml_outlier_interface(group, dm, selected_group_name)
                    except ImportError:
                        st.info("💡 ML-based outlier detection requires scikit-learn: `pip install scikit-learn`")
                    except Exception as e:
                        st.warning(f"ML outlier detection not available: {e}")
                
                # File selection grid with checkboxes (legacy interface)
                with st.expander("📋 Legacy Checkbox Interface", expanded=False):
                    st.markdown("**Select files to include in analysis:**")
                    st.info("💡 This is the original checkbox interface. Use the enhanced interfaces above for easier selection.")
                    
                    files_per_row = 4
                    file_list = list(group['files'])
                    
                    # Track changes to trigger plot update
                    selection_changed = False
                    
                    for i in range(0, len(file_list), files_per_row):
                        cols = st.columns(files_per_row)
                        for j, col in enumerate(cols):
                            if i + j < len(file_list):
                                file_path = file_list[i + j]
                                file_name = dm.files[file_path]['name']
                                
                                # Current inclusion status (inverted from exclusion)
                                is_included = file_path not in st.session_state.interactive_excluded_files
                                
                                # Checkbox to toggle inclusion
                                new_status = col.checkbox(
                                    f"📁 {file_name[:25]}{'...' if len(file_name) > 25 else ''}",
                                    value=is_included,
                                    key=f"file_toggle_{file_path}",
                                    help=f"File: {file_name}\nStatus: {'Included' if is_included else 'Excluded'}"
                                )
                                
                                # Update exclusion set based on checkbox
                                if new_status != is_included:  # Status changed
                                    if new_status:  # Include file (remove from exclusion set)
                                        st.session_state.interactive_excluded_files.discard(file_path)
                                    else:  # Exclude file (add to exclusion set)
                                        st.session_state.interactive_excluded_files.add(file_path)
                                    selection_changed = True
                
                # Update excluded_paths for rest of the analysis
                excluded_paths = list(st.session_state.interactive_excluded_files)
                
                # Status summary
                n_included = len(group['files']) - len(excluded_paths)
                n_excluded = len(excluded_paths)
                
                status_cols = st.columns(3)
                with status_cols[0]:
                    st.metric("✅ Included Files", n_included, help="Files included in group analysis")
                with status_cols[1]:
                    st.metric("❌ Excluded Files", n_excluded, help="Files excluded from group analysis")
                with status_cols[2]:
                    st.metric("📊 Total Files", len(group['files']), help="Total files in group")
                
                # Create the updated plot
                fig_indiv = go.Figure()
                group_files_data = {path: dm.files[path] for path in group['files']}
                
                # Generate distinct colors for each curve
                import colorsys
                n_files = len(group_files_data)
                
                for i, (path, file_data) in enumerate(group_files_data.items()):
                    is_excluded = path in excluded_paths
                    
                    # Generate distinct color for each curve
                    if is_excluded:
                        line_color = "rgba(255, 50, 50, 0.8)"  # Red for excluded
                        line_width = 3
                        line_dash = "dash"
                    else:
                        # Generate distinct colors for included files
                        hue = (i * 360 / n_files) % 360
                        rgb = colorsys.hsv_to_rgb(hue/360, 0.7, 0.8)
                        line_color = f"rgba({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)}, 0.8)"
                        line_width = 2
                        line_dash = "solid"

                    fig_indiv.add_trace(go.Scatter(
                        x=file_data['time'],
                        y=file_data['intensity'],
                        mode='lines',
                        name=f"{'❌ ' if is_excluded else '✅ '}{file_data['name']}",
                        line=dict(color=line_color, width=line_width, dash=line_dash),
                        hovertemplate=f"<b>{file_data['name']}</b><br>" +
                                    f"Status: {'EXCLUDED' if is_excluded else 'INCLUDED'}<br>" +
                                    "Time: %{x:.2f}s<br>" +
                                    "Intensity: %{y:.3f}<extra></extra>",
                        legendgroup="excluded" if is_excluded else "included",
                        showlegend=True
                    ))

                fig_indiv.update_layout(
                    title=f"Individual Recovery Curves - {n_included} Included, {n_excluded} Excluded",
                    xaxis_title="Time (s)",
                    yaxis_title="Normalized Intensity",
                    height=600,
                    hovermode='closest',
                    legend=dict(
                        orientation="v",
                        yanchor="top",
                        y=1,
                        xanchor="left",
                        x=1.02,
                        bgcolor="rgba(255,255,255,0.8)"
                    )
                )
                
                st.plotly_chart(fig_indiv, use_container_width=True)

                # Detailed table of individual kinetics
                st.markdown("#### Kinetic Parameters for Each File")
                st.markdown(f"**Currently Analyzing:** {len(group['files']) - len(excluded_paths)} included files, {len(excluded_paths)} excluded files")

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

                        # Use the interactive exclusion status
                        status = 'Excluded' if path in excluded_paths else 'Included'
                        
                        all_files_data.append({
                            'File Name': file_data['name'],
                            'Status': status,
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

                    # Style the dataframe with excluded files highlighted
                    def highlight_excluded(row):
                        if row['Status'] == 'Excluded':
                            return ['background-color: #ffcccc; color: #cc0000' for _ in row]
                        else:
                            return ['background-color: #ccffcc; color: #006600' for _ in row]

                    with st.expander("📊 Show Detailed Kinetics Table", expanded=True):
                        st.dataframe(
                            detailed_df.style.apply(highlight_excluded, axis=1).format({
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

                        # Summary statistics for included files only
                        included_data = detailed_df[detailed_df['Status'] == 'Included']
                        if len(included_data) > 0:
                            st.markdown("##### Summary Statistics (Included Files Only)")
                            
                            summary_cols = st.columns(4)
                            with summary_cols[0]:
                                st.metric("Mean Mobile Fraction", f"{included_data['Mobile (%)'].mean():.1f}%")
                                st.metric("Std Mobile Fraction", f"{included_data['Mobile (%)'].std():.1f}%")
                            
                            with summary_cols[1]:
                                st.metric("Mean Half-time", f"{included_data['Half-time (s)'].mean():.2f} s")
                                st.metric("Std Half-time", f"{included_data['Half-time (s)'].std():.2f} s")
                            
                            with summary_cols[2]:
                                st.metric("Mean App. D", f"{included_data['App. D (μm²/s)'].mean():.3f} μm²/s")
                                st.metric("Std App. D", f"{included_data['App. D (μm²/s)'].std():.3f} μm²/s")
                            
                            with summary_cols[3]:
                                st.metric("Mean R²", f"{included_data['R²'].mean():.3f}")
                                st.metric("Files Used", f"{len(included_data)}/{len(detailed_df)}")
                        else:
                            st.warning("⚠️ No files are currently included in the analysis. Please include at least one file.")

                        # Add report generation section
                        st.markdown("---")
                        st.markdown("### Step 4: Generate Analysis Report")
                        col_report1, col_report2 = st.columns(2)

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
                st.markdown("### Step 5: Interactive Parameter Dashboard")
                
                # Interactive scatter plot linked to recovery curves
                st.markdown("#### 🔍 Explore Data: Click Points to View Recovery Curves")
                
                numeric_cols = [col for col in filtered_df.select_dtypes(include=[np.number]).columns
                               if col not in ['file_path'] and not filtered_df[col].isna().all()]
                
                if numeric_cols and len(numeric_cols) >= 2:
                    col_dash1, col_dash2 = st.columns([1, 1])
                    
                    with col_dash1:
                        x_param = st.selectbox(
                            "X-axis parameter:",
                            numeric_cols,
                            index=numeric_cols.index('mobile_fraction') if 'mobile_fraction' in numeric_cols else 0,
                            key="dashboard_x"
                        )
                    
                    with col_dash2:
                        y_param = st.selectbox(
                            "Y-axis parameter:",
                            numeric_cols,
                            index=numeric_cols.index('half_time_fast') if 'half_time_fast' in numeric_cols else (1 if len(numeric_cols) > 1 else 0),
                            key="dashboard_y"
                        )
                    
                    # Create scatter plot with file names for hover
                    scatter_data = filtered_df.dropna(subset=[x_param, y_param]).copy()
                    scatter_data['file_name'] = scatter_data['file_path'].apply(lambda x: dm.files[x]['name'] if x in dm.files else 'Unknown')
                    scatter_data['point_index'] = range(len(scatter_data))
                    
                    if len(scatter_data) > 0:
                        # Create interactive scatter plot
                        fig_scatter = go.Figure()
                        
                        fig_scatter.add_trace(go.Scatter(
                            x=scatter_data[x_param],
                            y=scatter_data[y_param],
                            mode='markers',
                            marker=dict(
                                size=10,
                                color=scatter_data[y_param],
                                colorscale='Viridis',
                                showscale=True,
                                line=dict(width=1, color='white')
                            ),
                            text=scatter_data['file_name'],
                            customdata=scatter_data['file_path'],
                            hovertemplate='<b>%{text}</b><br>' +
                                        f'{x_param}: %{{x:.3f}}<br>' +
                                        f'{y_param}: %{{y:.3f}}<br>' +
                                        '<extra></extra>'
                        ))
                        
                        fig_scatter.update_layout(
                            title=f"Interactive Parameter Space: {x_param.replace('_', ' ').title()} vs {y_param.replace('_', ' ').title()}",
                            xaxis_title=x_param.replace('_', ' ').title(),
                            yaxis_title=y_param.replace('_', ' ').title(),
                            height=500,
                            hovermode='closest'
                        )
                        
                        # Display scatter plot
                        selected_points = st.plotly_chart(fig_scatter, use_container_width=True, key="scatter_dashboard", on_select="rerun")
                        
                        # Show recovery curves for selected points
                        if selected_points and 'selection' in selected_points and 'points' in selected_points['selection']:
                            selected_indices = selected_points['selection']['points']
                            
                            if len(selected_indices) > 0:
                                st.markdown("##### 📈 Recovery Curves for Selected Points")
                                
                                # Create subplot for selected curves
                                fig_selected = go.Figure()
                                
                                for idx_info in selected_indices[:10]:  # Limit to 10 curves
                                    point_idx = idx_info.get('point_index', idx_info.get('pointIndex'))
                                    if point_idx is not None and point_idx < len(scatter_data):
                                        file_path = scatter_data.iloc[point_idx]['file_path']
                                        
                                        if file_path in dm.files:
                                            file_data = dm.files[file_path]
                                            file_name = file_data['name']
                                            
                                            time_data = file_data.get('time')
                                            intensity_data = file_data.get('intensity')
                                            
                                            if time_data is not None and intensity_data is not None:
                                                fig_selected.add_trace(go.Scatter(
                                                    x=time_data,
                                                    y=intensity_data,
                                                    mode='lines+markers',
                                                    name=file_name,
                                                    marker=dict(size=4),
                                                    line=dict(width=2),
                                                    hovertemplate=f'<b>{file_name}</b><br>Time: %{{x:.2f}}s<br>Intensity: %{{y:.3f}}<extra></extra>'
                                                ))
                                
                                fig_selected.update_layout(
                                    title="Recovery Curves for Selected Points",
                                    xaxis_title="Time (s)",
                                    yaxis_title="Normalized Intensity",
                                    height=400,
                                    showlegend=True,
                                    legend=dict(
                                        orientation="v",
                                        yanchor="middle",
                                        y=0.5,
                                        xanchor="left",
                                        x=1.02
                                    )
                                )
                                
                                st.plotly_chart(fig_selected, use_container_width=True)
                        else:
                            st.info("💡 Click on points in the scatter plot above to view their recovery curves")
                
                st.markdown("---")
                st.markdown("### Step 6: Parameter Distribution Analysis")

                # Parameter distribution visualization
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
                st.markdown("### Step 6.5: Advanced Population Analysis")
                st.markdown("**Post-fitting analysis to identify cell populations and biologically meaningful outliers**")
                
                with st.expander("🔬 Post-Fitting Population Discovery", expanded=False):
                    st.markdown("""
                    **Why post-fitting analysis?**
                    
                    Traditional outlier removal before fitting can mask biologically important information:
                    - **Cell cycle differences** (G1/S/G2 phases may have different kinetics)
                    - **Differentiation states** (partially differentiated cells)
                    - **Microenvironmental effects** (edge vs. center cells)
                    - **Technical outliers** that only become apparent after fitting
                    
                    **Analysis Methods:**
                    1. **Clustering Analysis**: Identify distinct kinetic populations
                    2. **Outlier Detection**: Find cells with unusual parameter combinations
                    3. **Population Characterization**: Determine what makes each population unique
                    4. **Biological Interpretation**: Suggest potential biological causes
                    """)
                    
                    if filtered_df is not None and len(filtered_df) >= 10:  # Need enough cells for meaningful clustering
                        try:
                            # Population analysis using fitted parameters
                            st.markdown("#### 🎯 Population Clustering Analysis")
                            
                            # Select parameters for clustering
                            cluster_params = st.multiselect(
                                "Select parameters for population analysis:",
                                options=[col for col in filtered_df.select_dtypes(include=[np.number]).columns 
                                        if col not in ['file_path'] and not filtered_df[col].isna().all()],
                                default=['mobile_fraction', 'half_time_fast', 'half_time_slow'] if all(col in filtered_df.columns for col in ['mobile_fraction', 'half_time_fast', 'half_time_slow']) else [],
                                help="Choose kinetic parameters to identify cell populations"
                            )
                            
                            if len(cluster_params) >= 2:
                                # Clustering analysis
                                from sklearn.cluster import KMeans
                                from sklearn.preprocessing import StandardScaler
                                from sklearn.metrics import silhouette_score
                                import numpy as np
                                
                                # Prepare data for clustering
                                cluster_data = filtered_df[cluster_params].dropna()
                                
                                if len(cluster_data) >= 6:  # Minimum for clustering
                                    scaler = StandardScaler()
                                    scaled_data = scaler.fit_transform(cluster_data)
                                    
                                    # Determine optimal number of clusters
                                    max_clusters = min(6, len(cluster_data) // 3)  # Reasonable max
                                    silhouette_scores = []
                                    inertias = []
                                    
                                    for k in range(2, max_clusters + 1):
                                        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                                        cluster_labels = kmeans.fit_predict(scaled_data)
                                        sil_score = silhouette_score(scaled_data, cluster_labels)
                                        silhouette_scores.append(sil_score)
                                        inertias.append(kmeans.inertia_)
                                    
                                    # Find optimal k (highest silhouette score)
                                    optimal_k = int(np.argmax(silhouette_scores)) + 2
                                    
                                    col_cluster1, col_cluster2 = st.columns(2)
                                    
                                    with col_cluster1:
                                        n_clusters = st.selectbox(
                                            "Number of populations:",
                                            range(2, max_clusters + 1),
                                            index=optimal_k - 2,
                                            help=f"Optimal: {optimal_k} clusters (highest silhouette score: {max(silhouette_scores):.3f})"
                                        )
                                    
                                    with col_cluster2:
                                        outlier_method = st.selectbox(
                                            "Outlier detection method:",
                                            ["Isolation Forest", "Local Outlier Factor", "Mahalanobis Distance"],
                                            help="Method to identify cells with unusual parameter combinations"
                                        )
                                    
                                    if st.button("Analyze Cell Populations", type="primary"):
                                        with st.spinner("Analyzing cell populations..."):
                                            # Perform clustering
                                            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                                            cluster_labels = kmeans.fit_predict(scaled_data)
                                            
                                            # Outlier detection
                                            if outlier_method == "Isolation Forest":
                                                from sklearn.ensemble import IsolationForest
                                                outlier_detector = IsolationForest(contamination=0.1, random_state=42)
                                                outlier_labels = outlier_detector.fit_predict(scaled_data)
                                            elif outlier_method == "Local Outlier Factor":
                                                from sklearn.neighbors import LocalOutlierFactor
                                                outlier_detector = LocalOutlierFactor(contamination=0.1)
                                                outlier_labels = outlier_detector.fit_predict(scaled_data)
                                            else:  # Mahalanobis Distance
                                                from scipy.spatial.distance import mahalanobis
                                                mean = np.mean(scaled_data, axis=0)
                                                cov = np.cov(scaled_data.T)
                                                inv_cov = np.linalg.pinv(cov)
                                                
                                                mahal_distances = []
                                                for row in scaled_data:
                                                    mahal_dist = mahalanobis(row, mean, inv_cov)
                                                    mahal_distances.append(mahal_dist)
                                                
                                                # Threshold at 95th percentile
                                                threshold = np.percentile(mahal_distances, 90)
                                                outlier_labels = np.where(np.array(mahal_distances) > threshold, -1, 1)
                                            
                                            # Add results to dataframe
                                            results_df = cluster_data.copy()
                                            results_df['Population'] = cluster_labels
                                            results_df['Is_Outlier'] = (outlier_labels == -1)
                                            results_df['File_Name'] = [dm.files[fp]['name'] for fp in results_df.index]
                                            
                                            # Population summary
                                            st.markdown("#### 📊 Population Summary")
                                            
                                            pop_summary = []
                                            for pop in range(n_clusters):
                                                pop_data = results_df[results_df['Population'] == pop]
                                                pop_outliers = pop_data['Is_Outlier'].sum()
                                                
                                                summary = {
                                                    'Population': f"Pop {pop + 1}",
                                                    'Size': len(pop_data),
                                                    'Percentage': f"{len(pop_data)/len(results_df)*100:.1f}%",
                                                    'Outliers': pop_outliers,
                                                    'Outlier_Rate': f"{pop_outliers/len(pop_data)*100:.1f}%" if len(pop_data) > 0 else "0%"
                                                }
                                                
                                                # Add mean parameter values
                                                for param in cluster_params:
                                                    summary[f'{param}_mean'] = pop_data[param].mean()
                                                
                                                pop_summary.append(summary)
                                            
                                            summary_df = pd.DataFrame(pop_summary)
                                            st.dataframe(summary_df, use_container_width=True)
                                            
                                            # Visualization
                                            st.markdown("#### 📈 Population Visualization")
                                            
                                            if len(cluster_params) >= 2:
                                                # 2D scatter plot
                                                fig_pop = go.Figure()
                                                
                                                colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
                                                
                                                for pop in range(n_clusters):
                                                    pop_data = results_df[results_df['Population'] == pop]
                                                    
                                                    # Normal cells
                                                    normal_cells = pop_data[~pop_data['Is_Outlier']]
                                                    if len(normal_cells) > 0:
                                                        fig_pop.add_trace(go.Scatter(
                                                            x=normal_cells[cluster_params[0]],
                                                            y=normal_cells[cluster_params[1]],
                                                            mode='markers',
                                                            name=f'Population {pop + 1}',
                                                            marker=dict(color=colors[pop % len(colors)], size=8),
                                                            text=normal_cells['File_Name'],
                                                            hovertemplate=f'<b>Population {pop + 1}</b><br>' +
                                                                        f'{cluster_params[0]}: %{{x:.3f}}<br>' +
                                                                        f'{cluster_params[1]}: %{{y:.3f}}<br>' +
                                                                        'File: %{text}<extra></extra>'
                                                        ))
                                                    
                                                    # Outlier cells
                                                    outlier_cells = pop_data[pop_data['Is_Outlier']]
                                                    if len(outlier_cells) > 0:
                                                        fig_pop.add_trace(go.Scatter(
                                                            x=outlier_cells[cluster_params[0]],
                                                            y=outlier_cells[cluster_params[1]],
                                                            mode='markers',
                                                            name=f'Pop {pop + 1} Outliers',
                                                            marker=dict(color=colors[pop % len(colors)], size=12, 
                                                                      symbol='x', line=dict(width=2, color='black')),
                                                            text=outlier_cells['File_Name'],
                                                            hovertemplate=f'<b>Population {pop + 1} - OUTLIER</b><br>' +
                                                                        f'{cluster_params[0]}: %{{x:.3f}}<br>' +
                                                                        f'{cluster_params[1]}: %{{y:.3f}}<br>' +
                                                                        'File: %{text}<extra></extra>'
                                                        ))
                                                
                                                fig_pop.update_layout(
                                                    title=f'Cell Populations in {selected_group_name}',
                                                    xaxis_title=cluster_params[0].replace('_', ' ').title(),
                                                    yaxis_title=cluster_params[1].replace('_', ' ').title(),
                                                    height=600,
                                                    hovermode='closest'
                                                )
                                                
                                                st.plotly_chart(fig_pop, use_container_width=True)
                                            
                                            # Biological interpretation
                                            st.markdown("#### 🧬 Biological Interpretation")
                                            
                                            interpretation_lines = []
                                            interpretation_lines.append(f"**Population analysis of {selected_group_name} identified {n_clusters} distinct cell populations:**\n")
                                            
                                            for pop in range(n_clusters):
                                                pop_data = results_df[results_df['Population'] == pop]
                                                interpretation_lines.append(f"**Population {pop + 1}** ({len(pop_data)} cells, {len(pop_data)/len(results_df)*100:.1f}%):")
                                                
                                                for param in cluster_params:
                                                    mean_val = pop_data[param].mean()
                                                    overall_mean = results_df[param].mean()
                                                    fold_change = mean_val / overall_mean
                                                    
                                                    if fold_change > 1.2:
                                                        interpretation_lines.append(f"- Higher {param.replace('_', ' ')} ({mean_val:.3f} vs {overall_mean:.3f})")
                                                    elif fold_change < 0.8:
                                                        interpretation_lines.append(f"- Lower {param.replace('_', ' ')} ({mean_val:.3f} vs {overall_mean:.3f})")
                                                
                                                outlier_count = pop_data['Is_Outlier'].sum()
                                                if outlier_count > 0:
                                                    interpretation_lines.append(f"- Contains {outlier_count} outlier(s)")
                                                
                                                interpretation_lines.append("")
                                            
                                            # Suggest biological causes
                                            interpretation_lines.append("**Potential biological explanations:**")
                                            interpretation_lines.append("- **Population differences** may reflect:")
                                            interpretation_lines.append("  - Cell cycle phase (G1/S/G2 have different chromatin dynamics)")
                                            interpretation_lines.append("  - Differentiation state (stem vs. committed cells)")
                                            interpretation_lines.append("  - Position in tissue (edge vs. center effects)")
                                            interpretation_lines.append("  - Metabolic state (active vs. quiescent)")
                                            interpretation_lines.append("- **Outlier cells** may indicate:")
                                            interpretation_lines.append("  - Mitotic cells (condensed chromatin)")
                                            interpretation_lines.append("  - Apoptotic cells (fragmented chromatin)")
                                            interpretation_lines.append("  - Technical artifacts (poor bleaching, drift)")
                                            interpretation_lines.append("  - Rare cell types or states")
                                            
                                            st.markdown("\n".join(interpretation_lines))
                                            
                                            # Export results
                                            st.markdown("#### 💾 Export Population Analysis")
                                            
                                            # Prepare export data
                                            export_df = results_df.copy()
                                            export_df['Group'] = selected_group_name
                                            export_df = export_df[['Group', 'File_Name', 'Population', 'Is_Outlier'] + cluster_params]
                                            
                                            st.download_button(
                                                label="📊 Download Population Analysis Results",
                                                data=export_df.to_csv(index=False),
                                                file_name=f"population_analysis_{selected_group_name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                                mime="text/csv",
                                                help="Download detailed population analysis with cluster assignments and outlier flags"
                                            )
                                            
                                            st.success(f"✓ Population analysis complete! Found {n_clusters} populations and {(outlier_labels == -1).sum()} outliers.")
                                
                                else:
                                    st.warning("Need at least 6 cells for meaningful population analysis.")
                            else:
                                st.info("Select at least 2 parameters to perform population clustering.")
                        
                        except ImportError:
                            st.error("Population analysis requires scikit-learn. Install with: `pip install scikit-learn`")
                        except Exception as e:
                            st.error(f"Error in population analysis: {e}")
                            import traceback
                            with st.expander("🔍 Error Details"):
                                st.code(traceback.format_exc())
                    
                    else:
                        st.info("Population analysis requires at least 10 cells. Current group has fewer cells.")

                st.markdown("---")
                st.markdown("### Step 7: Global Simultaneous Fitting")
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

                                    # Perform global fitting directly using the main data manager
                                    global_result = dm.fit_group_models(
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
                st.markdown("### Step 8: Group Recovery Plots")
                plot_data={path:dm.files[path] for path in filtered_df['file_path'].tolist()}
                st.markdown("##### Average Recovery Curve")
                avg_fig = plot_average_curve(plot_data)
                st.plotly_chart(avg_fig, use_container_width=True)
                
                # Step 8: Time-Aligned Curves
                st.markdown("---")
                st.markdown("### Step 9: Visualize Aligned Group Curves")
                st.markdown("""
                This plot shows all included curves from the group, **aligned to the bleach point (t=0)** 
                and **interpolated onto a common time axis** to correct for different sampling rates between experiments.
                
                **Why is this important?**
                - Different experiments may use different frame rates or time intervals
                - Direct comparison requires all curves to be on the same time scale
                - Alignment ensures accurate visualization of recovery kinetics
                """)
                
                if st.button("Generate Aligned Curves Plot", type="secondary", key="align_curves_btn"):
                    with st.spinner("Aligning and interpolating curves..."):
                        # Get the data for included files
                        included_files_df = group.get('features_df')
                        if included_files_df is not None and not included_files_df.empty:
                            
                            curves_to_align = []
                            for file_path in included_files_df['file_path']:
                                if file_path in dm.files:
                                    file_data = dm.files[file_path]
                                    curves_to_align.append({
                                        'name': file_data['name'],
                                        'time': file_data['time'],
                                        'intensity': file_data['intensity']
                                    })
                            
                            if curves_to_align:
                                try:
                                    # Call the new alignment and plotting functions
                                    from frap_core import FRAPAnalysisCore
                                    from frap_plots import FRAPPlots
                                    
                                    # Align and interpolate curves
                                    aligned_results = FRAPAnalysisCore.align_and_interpolate_curves(curves_to_align)
                                    
                                    # Check if alignment was successful
                                    if aligned_results['interpolated_curves']:
                                        # Create and display the plot
                                        aligned_fig = FRAPPlots.plot_aligned_curves(aligned_results)
                                        st.plotly_chart(aligned_fig, use_container_width=True)
                                        
                                        # Show summary statistics
                                        st.success(f"✓ Successfully aligned {len(aligned_results['interpolated_curves'])} curves")
                                        st.info(f"📊 Common time axis: 0 to {aligned_results['common_time'][-1]:.2f} seconds ({len(aligned_results['common_time'])} points)")
                                    else:
                                        st.warning("⚠️ No curves could be aligned. Check that your data has valid bleach events.")
                                        
                                except Exception as e:
                                    st.error(f"❌ Error during curve alignment: {e}")
                                    import traceback
                                    with st.expander("🔍 Error Details"):
                                        st.code(traceback.format_exc())
                            else:
                                st.warning("No valid curves to align.")
                        else:
                            st.warning("No included files in this group to generate the plot.")
                            
            else:
                st.warning("No data to display. All files may have been excluded as outliers.")
    else:
        st.info("Create and/or select a group from the sidebar to begin analysis.")

    # Add holistic group comparison section
    if len(dm.groups) >= 2:
        st.markdown("---")
        st.markdown("## 🔬 Multi-Group Comparison")
        st.markdown("""
        **Compare recovery dynamics across experimental conditions using mean recovery curves.**
        
        This approach provides robust, mechanistic insights by analyzing **averaged recovery profiles** 
        that minimize noise from individual cell variability while revealing biological differences.
        
        💡 **Recommended Workflow:**
        - Start here for **comparing treatment conditions** using mean recovery curve analysis
        - Use **Individual Group Analysis (Tab 1)** for in-depth, cell-level analysis of specific conditions
        """)

        with st.expander("📖 Why Focus on Mean Recovery Curves?", expanded=False):
            st.markdown("""
            **Mean recovery curve analysis offers several advantages:**
            
            ✅ **Higher signal-to-noise ratio** - Averaging reduces measurement noise
            ✅ **Sophisticated biophysical models** - Can fit complex models (anomalous diffusion, reaction-diffusion)
            ✅ **Population-level insights** - Reveals treatment effects on overall recovery dynamics
            ✅ **Robust comparisons** - Less sensitive to outliers and individual cell variability
            
            **Individual cell fitting limitations:**
            - Lower SNR limits model complexity (typically only simple exponential models)
            - High variability requires larger sample sizes
            - Best suited for detailed exploration within a single condition
            
            **Example Use Cases:**
            - **Mean Curves:** "Does Drug X slow recovery compared to control?"
            - **Individual Cells:** "What is the distribution of recovery rates in the Drug X group?"
            """)

        # Import required modules
        try:
            from frap_group_comparison import (
                HolisticGroupComparator,
                compute_average_recovery_profile,
                compare_recovery_profiles
            )

            # Group selection for comparison
            st.markdown("### Select Groups to Compare")
            
            available_groups = list(dm.groups.keys())
            
            selected_groups_holistic = st.multiselect(
                "Choose experimental groups:",
                options=available_groups,
                default=available_groups[:2] if len(available_groups) >= 2 else available_groups,
                help="Select 2+ groups to compare their mean recovery curves"
            )

            if len(selected_groups_holistic) >= 2:
                st.markdown("---")
                
                # Initialize comparator
                bleach_radius = st.session_state.settings.get('default_bleach_radius', 1.0)
                pixel_size = st.session_state.settings.get('default_pixel_size', 1.0)
                comparator = HolisticGroupComparator(
                    bleach_radius_um=bleach_radius * pixel_size,
                    pixel_size=pixel_size
                )

                # Prepare features for each selected group
                group_features = {}
                group_data_raw = {}
                
                for group_name in selected_groups_holistic:
                    features_df = dm.groups[group_name].get('features_df')
                    if features_df is not None and not features_df.empty:
                        group_features[group_name] = features_df
                    
                    # Get raw intensity data for profile comparison
                    group_files = dm.groups[group_name].get('files', [])
                    group_data_raw[group_name] = {
                        fp: dm.files[fp] for fp in group_files if fp in dm.files
                    }

                # ============================================================================
                # SECTION 1: MEAN RECOVERY CURVE COMPARISON (PRIMARY)
                # ============================================================================
                st.markdown("## 📉 Mean Recovery Curve Analysis")
                st.markdown("**Direct comparison of averaged curves - robust, model-independent approach**")
                
                if len(group_data_raw) >= 2:
                    try:
                        import plotly.graph_objects as go
                        fig_profiles = go.Figure()
                        
                        colors = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)', 
                                 'rgb(214, 39, 40)', 'rgb(148, 103, 189)', 'rgb(140, 86, 75)']
                        
                        for idx, group_name in enumerate(selected_groups_holistic):
                            if group_name in group_data_raw and group_data_raw[group_name]:
                                try:
                                    t_avg, i_avg, i_sem = compute_average_recovery_profile(
                                        group_data_raw[group_name]
                                    )
                                    
                                    color = colors[idx % len(colors)]
                                    
                                    # Plot mean with error bars
                                    fig_profiles.add_trace(go.Scatter(
                                        x=t_avg,
                                        y=i_avg,
                                        mode='lines',
                                        name=group_name,
                                        line=dict(color=color, width=3),
                                        showlegend=True
                                    ))
                                    
                                    # Add SEM as shaded area
                                    fig_profiles.add_trace(go.Scatter(
                                        x=np.concatenate([t_avg, t_avg[::-1]]),
                                        y=np.concatenate([i_avg + i_sem, (i_avg - i_sem)[::-1]]),
                                        fill='toself',
                                        fillcolor=color.replace('rgb', 'rgba').replace(')', ', 0.2)'),
                                        line=dict(color='rgba(255,255,255,0)'),
                                        name=f'{group_name} (±SEM)',
                                        showlegend=False,
                                        hoverinfo='skip'
                                    ))
                                    
                                except Exception as e:
                                    st.warning(f"Could not compute average profile for {group_name}: {e}")
                        
                        fig_profiles.update_layout(
                            title='Mean Recovery Profiles Comparison',
                            xaxis_title='Time (s)',
                            yaxis_title='Normalized Intensity',
                            height=500,
                            hovermode='x unified',
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            )
                        )
                        
                        st.plotly_chart(fig_profiles, use_container_width=True)
                        
                        # Quantitative profile comparison for 2 groups
                        if len(selected_groups_holistic) == 2:
                            group1_name, group2_name = selected_groups_holistic[0], selected_groups_holistic[1]
                            
                            if (group1_name in group_data_raw and group2_name in group_data_raw and
                                group_data_raw[group1_name] and group_data_raw[group2_name]):
                                try:
                                    profile_comparison = compare_recovery_profiles(
                                        group_data_raw[group1_name],
                                        group_data_raw[group2_name],
                                        group1_name=group1_name,
                                        group2_name=group2_name
                                    )
                                    
                                    st.markdown("#### 📊 Profile Similarity Metrics")
                                    
                                    col_prof1, col_prof2, col_prof3 = st.columns(3)
                                    
                                    # Access metrics from 'comparison' sub-dictionary
                                    comparison_metrics = profile_comparison['comparison']
                                    
                                    with col_prof1:
                                        st.metric(
                                            "Max Difference",
                                            f"{comparison_metrics['max_difference']:.3f}",
                                            help="Maximum absolute difference between averaged profiles"
                                        )
                                    
                                    with col_prof2:
                                        st.metric(
                                            "Mean Difference",
                                            f"{comparison_metrics['mean_difference']:.3f}",
                                            help="Average absolute difference across all time points"
                                        )
                                    
                                    with col_prof3:
                                        st.metric(
                                            "RMSD",
                                            f"{comparison_metrics['rmsd']:.3f}",
                                            help="Root mean square deviation between profiles"
                                        )
                                    
                                except Exception as e:
                                    st.warning(f"Could not compute profile comparison metrics: {e}")
                    
                    except Exception as e:
                        st.error(f"Error plotting averaged profiles: {e}")
                        import traceback
                        st.code(traceback.format_exc())
                
                # Advanced Curve Fitting Option
                st.markdown("---")
                with st.expander("🔬 Advanced Curve Fitting (Mechanistic Models)", expanded=False):
                    st.markdown("""
                    **Fit sophisticated biophysical models to mean recovery curves** to extract mechanistic parameters.
                    
                    **Available Models:**
                    - **Anomalous Diffusion**: For subdiffusive or superdiffusive recovery (α ≠ 1)
                    - **Simple Reaction-Diffusion**: Fast diffusion + slow binding/unbinding
                    - **Full Reaction-Diffusion**: Complete 3-state model (free, bound, intermediate)
                    
                    **Benefits:**
                    - Higher SNR from averaged data enables complex model fitting
                    - Extract biophysical parameters (diffusion coefficient, binding rates, etc.)
                    - Compare mechanisms between groups (e.g., "lost binding capability")
                    
                    **Note:** Individual cell fitting is limited to simple exponential models due to lower SNR.
                    """)
                    
                    if len(selected_groups_holistic) == 2:
                        try:
                            from frap_advanced_fitting import compare_groups_advanced_fitting
                            
                            st.markdown("### ⚙️ Advanced Fitting Configuration")
                            
                            col_adv1, col_adv2 = st.columns(2)
                            with col_adv1:
                                advanced_model = st.selectbox(
                                    "Select Biophysical Model:",
                                    ["auto", "anomalous_diffusion", "reaction_diffusion_simple", "reaction_diffusion_full"],
                                    help="'auto' selects best model by AIC"
                                )
                            
                            with col_adv2:
                                fit_bleach_radius = st.number_input(
                                    "Bleach Radius (μm):",
                                    min_value=0.1,
                                    max_value=10.0,
                                    value=float(bleach_radius * pixel_size),
                                    step=0.1,
                                    help="Effective bleach spot radius in micrometers"
                                )
                            
                            if st.button("Fit Advanced Models to Mean Curves", type="primary"):
                                with st.spinner("Fitting advanced models to mean recovery profiles..."):
                                    try:
                                        group1_name, group2_name = selected_groups_holistic[0], selected_groups_holistic[1]
                                        
                                        # Compute averaged profiles for both groups
                                        if (group1_name in group_data_raw and group2_name in group_data_raw and
                                            group_data_raw[group1_name] and group_data_raw[group2_name]):
                                            
                                            t1_avg, i1_avg, i1_sem = compute_average_recovery_profile(group_data_raw[group1_name])
                                            t2_avg, i2_avg, i2_sem = compute_average_recovery_profile(group_data_raw[group2_name])
                                            
                                            # Perform advanced fitting comparison
                                            fit_results = compare_groups_advanced_fitting(
                                                group1_time=t1_avg,
                                                group1_intensity=i1_avg,
                                                group1_sem=i1_sem,
                                                group2_time=t2_avg,
                                                group2_intensity=i2_avg,
                                                group2_sem=i2_sem,
                                                group1_name=group1_name,
                                                group2_name=group2_name,
                                                bleach_radius_um=fit_bleach_radius,
                                                model=advanced_model
                                            )
                                            
                                            if fit_results['success']:
                                                st.success("✓ Advanced fitting completed successfully!")
                                                
                                                # Display fitted parameters
                                                st.markdown("#### 📈 Fitted Parameters")
                                                st.markdown(f"**Model Used:** `{fit_results['model_used']}`")
                                                
                                                col_fit1, col_fit2 = st.columns(2)
                                                
                                                with col_fit1:
                                                    st.markdown(f"**{group1_name}**")
                                                    st.markdown(f"- R²: `{fit_results['r2_group1']:.4f}`")
                                                    
                                                    params1 = fit_results['group1_fit']['params']
                                                    for param, value in params1.items():
                                                        st.markdown(f"- {param}: `{value:.4f}`")
                                                
                                                with col_fit2:
                                                    st.markdown(f"**{group2_name}**")
                                                    st.markdown(f"- R²: `{fit_results['r2_group2']:.4f}`")
                                                    
                                                    params2 = fit_results['group2_fit']['params']
                                                    for param, value in params2.items():
                                                        st.markdown(f"- {param}: `{value:.4f}`")
                                                
                                                # Parameter comparison
                                                st.markdown("#### 🔍 Parameter Fold Changes")
                                                param_comparison = fit_results['parameter_comparison']
                                                
                                                if param_comparison:
                                                    comparison_data = []
                                                    for param_name, values in param_comparison.items():
                                                        comparison_data.append({
                                                            'Parameter': param_name,
                                                            group1_name: f"{values[group1_name]:.4f}",
                                                            group2_name: f"{values[group2_name]:.4f}",
                                                            'Fold Change': f"{values['fold_change']:.3f}",
                                                            'Percent Change': f"{values['percent_change']:+.1f}%"
                                                        })
                                                    
                                                    param_df = pd.DataFrame(comparison_data)
                                                    st.dataframe(param_df, use_container_width=True)
                                                else:
                                                    st.info("No parameter comparisons available.")
                                                
                                                # Biological interpretation
                                                st.markdown("#### 🧬 Biological Interpretation")
                                                st.markdown(fit_results['interpretation'])
                                                
                                                # Visualization
                                                st.markdown("#### 📊 Fitted Curves Visualization")
                                                from frap_plots import plot_advanced_group_comparison, plot_parameter_comparison
                                                
                                                try:
                                                    fig_fit = plot_advanced_group_comparison(fit_results)
                                                    st.plotly_chart(fig_fit, use_container_width=True)
                                                except Exception as plot_error:
                                                    st.warning(f"Could not generate fitted curves plot: {plot_error}")
                                                
                                                try:
                                                    fig_params = plot_parameter_comparison(fit_results)
                                                    st.plotly_chart(fig_params, use_container_width=True)
                                                except Exception as plot_error:
                                                    st.warning(f"Could not generate parameter comparison plot: {plot_error}")
                                            else:
                                                st.error(f"❌ Advanced fitting failed: {fit_results.get('error', 'Unknown error')}")
                                        else:
                                            st.error("❌ Could not compute averaged profiles for one or both groups.")
                                    
                                    except Exception as e:
                                        st.error(f"Error during advanced fitting: {e}")
                                        import traceback
                                        with st.expander("🔍 Error Details"):
                                            st.code(traceback.format_exc())
                        
                        except ImportError:
                            st.warning("⚠️ Advanced fitting module not available. Ensure `frap_advanced_fitting.py` is in the same directory.")
                    else:
                        st.info("Select exactly 2 groups to enable pairwise advanced fitting comparison.")

                # ============================================================================
                # SECTION 2: POPULATION-BASED ANALYSIS (HOLISTIC COMPARISON)
                # ============================================================================
                st.markdown("---")
                st.markdown("## 📊 Population-Based Analysis")
                st.markdown("**Analyze distributions of kinetic parameters across cell populations**")
                
                with st.expander("ℹ️ What is Population-Based Analysis?", expanded=False):
                    st.markdown("""
                    **Population analysis** characterizes the **distribution of recovery kinetics** within each group,
                    rather than just comparing mean values.
                    
                    **Key Insights:**
                    - **Kinetic Subpopulations**: Categorizes cells into fast diffusion / intermediate / slow binding regimes
                    - **Population Shifts**: Detects changes in the proportion of cells in each regime
                    - **Weighted Kinetics**: Calculates population-weighted rate constants
                    
                    **Example:**
                    - **Control:** 60% binding (slow), 40% diffusion (fast)
                    - **Treatment:** 18% binding, 82% diffusion
                    
                    → Treatment didn't "speed up recovery" - it **eliminated binding capability**!
                    """)
                
                # Multi-group population distribution table
                if group_features:
                    st.markdown("### 📊 Population Distribution Comparison")
                    st.markdown("**How are kinetic components distributed across diffusion/binding regimes?**")
                    
                    try:
                        comparison_df = comparator.compare_groups(group_features)
                        
                        # Display with enhanced formatting
                        st.dataframe(
                            comparison_df.style.format({
                                'n_cells': '{:.0f}',
                                'mobile_fraction_mean': '{:.1f}%',
                                'mobile_fraction_sem': '{:.1f}%',
                                'weighted_k_fast': '{:.3f}',
                                'weighted_k_slow': '{:.3f}',
                                'population_diffusion': '{:.1f}%',
                                'population_intermediate': '{:.1f}%',
                                'population_binding': '{:.1f}%'
                            }),
                            use_container_width=True
                        )
                        
                        # Visualize population distributions
                        st.markdown("#### Population Distribution Visualization")
                        
                        import plotly.graph_objects as go
                        from plotly.subplots import make_subplots
                        
                        fig_pop = go.Figure()
                        
                        x_groups = comparison_df['group'].tolist()
                        
                        fig_pop.add_trace(go.Bar(
                            name='Diffusion (k > 1.0 s⁻¹)',
                            x=x_groups,
                            y=comparison_df['population_diffusion'].tolist(),
                            marker_color='rgb(55, 83, 109)'
                        ))
                        
                        fig_pop.add_trace(go.Bar(
                            name='Intermediate (0.1 < k < 1.0 s⁻¹)',
                            x=x_groups,
                            y=comparison_df['population_intermediate'].tolist(),
                            marker_color='rgb(26, 118, 255)'
                        ))
                        
                        fig_pop.add_trace(go.Bar(
                            name='Binding (k < 0.1 s⁻¹)',
                            x=x_groups,
                            y=comparison_df['population_binding'].tolist(),
                            marker_color='rgb(50, 171, 96)'
                        ))
                        
                        fig_pop.update_layout(
                            barmode='stack',
                            title='Kinetic Population Distribution by Group',
                            xaxis_title='Group',
                            yaxis_title='Population (%)',
                            height=500,
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            )
                        )
                        
                        st.plotly_chart(fig_pop, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Error computing population comparison: {e}")

                # Pairwise statistical comparison (if exactly 2 groups selected)
                if len(selected_groups_holistic) == 2:
                    st.markdown("---")
                    st.markdown("### 📈 Pairwise Statistical Comparison")
                    
                    group1_name, group2_name = selected_groups_holistic[0], selected_groups_holistic[1]
                    
                    if group1_name in group_features and group2_name in group_features:
                        try:
                            # Perform statistical comparison
                            stats_results = comparator.statistical_comparison(
                                group_features[group1_name],
                                group_features[group2_name],
                                group1_name=group1_name,
                                group2_name=group2_name
                            )
                            
                            # Display results in organized columns
                            col_stat1, col_stat2 = st.columns(2)
                            
                            with col_stat1:
                                st.markdown("#### Mobile Fraction")
                                # Fix: Correct key structure from statistical_comparison()
                                if 'mobile_fraction' in stats_results['tests']:
                                    mf_test = stats_results['tests']['mobile_fraction']
                                    mf1 = mf_test['mean_group1']
                                    mf2 = mf_test['mean_group2']
                                    sem1 = mf_test['sem_group1']
                                    sem2 = mf_test['sem_group2']
                                    p_val = mf_test['p_value']
                                    
                                    st.metric(group1_name, f"{mf1:.1f}% ± {sem1:.1f}%")
                                    st.metric(group2_name, f"{mf2:.1f}% ± {sem2:.1f}%")
                                    
                                    if p_val < 0.05:
                                        st.success(f"✓ Significant difference (p={p_val:.4f})")
                                    else:
                                        st.info(f"✗ No significant difference (p={p_val:.4f})")
                                else:
                                    st.warning("Mobile fraction comparison not available")
                            
                            with col_stat2:
                                st.markdown("#### Population Shifts")
                                pop_comp = stats_results['population_comparison']
                                
                                diff_shift = pop_comp['diffusion_shift']
                                bind_shift = pop_comp['binding_shift']
                                
                                st.metric(
                                    "Diffusion Shift",
                                    f"{diff_shift:+.1f}%",
                                    delta=f"{diff_shift:.1f}%",
                                    delta_color="normal" if diff_shift > 0 else "inverse"
                                )
                                
                                st.metric(
                                    "Binding Shift",
                                    f"{bind_shift:+.1f}%",
                                    delta=f"{bind_shift:.1f}%",
                                    delta_color="normal" if bind_shift > 0 else "inverse"
                                )
                            
                            # Biological interpretation
                            st.markdown("---")
                            st.markdown("### 🧬 Biological Interpretation")
                            
                            interpretation = comparator.interpret_differences(stats_results)
                            st.markdown(interpretation)
                        
                        except Exception as e:
                            st.error(f"Error in statistical comparison: {e}")
                            import traceback
                            st.code(traceback.format_exc())

            elif len(selected_groups_holistic) == 1:
                st.info("Select at least 2 groups to enable multi-group comparison.")
            else:
                st.warning("Please select groups to compare.")

        except ImportError as e:
            st.error(f"Could not import holistic comparison module: {e}")
            st.info("Make sure `frap_group_comparison.py` is in the same directory as this script.")

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
                plot_type = st.selectbox("Visualization Type:", 
                                        ["Estimation Plot", "Box Plot", "Violin Plot", "Bar Plot (Mean ± SEM)"])

            # Create visualization
            if plot_type == "Estimation Plot":
                st.markdown("""
                **Estimation Plot Benefits:**
                - Shows all raw data points (transparency)
                - Displays mean with 95% confidence interval
                - Visualizes effect sizes (right panel)
                - More informative than traditional bar charts
                """)
                
                from frap_plots import FRAPPlots
                fig = FRAPPlots.plot_estimation_plot(
                    data_df=combined_df,
                    group_col='group',
                    value_col=param_to_plot,
                    group_names=list(dm.groups.keys()),
                    title=f'Estimation Plot: {param_to_plot.replace("_", " ").title()}',
                    height=600
                )
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("""
                    **How to Read This Plot:**
                    - **Left Panel**: 
                      - Small circles = individual measurements (jittered for visibility)
                      - Large diamonds = group means
                      - Error bars = 95% confidence intervals
                    - **Right Panel**: 
                      - Shows differences from reference group (first group)
                      - Diamonds = mean difference
                      - Error bars = 95% CI of the difference
                      - Dashed line at zero = no difference
                    """)
                else:
                    st.error("Failed to generate estimation plot")
                    
            elif plot_type == "Box Plot":
                fig = px.box(
                    combined_df, x='group', y=param_to_plot, color='group',
                    title=f'Distribution of {param_to_plot} Across Groups',
                    points="all"
                )
                fig.update_xaxes(title="Experimental Group")
                fig.update_yaxes(title=param_to_plot.replace('_', ' ').title())
                fig.update_layout(showlegend=False, height=500)
                st.plotly_chart(fig, use_container_width=True)
                
            elif plot_type == "Violin Plot":
                fig = px.violin(
                    combined_df, x='group', y=param_to_plot, color='group',
                    title=f'Distribution of {param_to_plot} Across Groups',
                    box=True, points="all"
                )
                fig.update_xaxes(title="Experimental Group")
                fig.update_yaxes(title=param_to_plot.replace('_', ' ').title())
                fig.update_layout(showlegend=False, height=500)
                st.plotly_chart(fig, use_container_width=True)
                
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
                            try:
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
                            except Exception as e:
                                st.error(f"Error performing statistical test: {e}")
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
    create_image_analysis_interface(dm)

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
                                        'R²': row.get('r2', np.nan)
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
    # st.markdown("### Debug Package Generation")
    # st.markdown("Create a comprehensive package for external debugging and deployment")

    # col_debug1, col_debug2 = st.columns([2, 1])

    # with col_debug1:
    #     st.markdown("**Debug package includes:**")
    #     st.markdown("- Complete source code and documentation")
    #     st.markdown("- Installation scripts for Windows and Unix")
    #     st.markdown("- Sample data files and test suite")
    #     st.markdown("- Docker configuration for containerized deployment")
    #     st.markdown("- Streamlit configuration files")

    # with col_debug2:
    #     if st.button("📦 Create Debug Package", type="primary"):
    #         try:
    #             with st.spinner("Creating comprehensive debug package..."):
    #                 # Import and run the debug package creator
    #                 from create_debug_package import create_debug_package
    #                 package_file, summary = create_debug_package()

    #                 # Read the package file
    #                 with open(package_file, 'rb') as f:
    #                     package_data = f.read()

    #                 # Provide download button
    #                 st.download_button(
    #                     label="⬇️ Download Debug Package",
    #                     data=package_data,
    #                     file_name=package_file,
    #                     mime="application/zip",
    #                     help="Download complete debug package with all source code and documentation"
    #                 )

    #                 st.success("Debug package created successfully!")
    #                 st.info(f"Package size: {len(package_data) / 1024 / 1024:.1f} MB")

    #                 # Clean up temporary file
    #                 os.remove(package_file)
    #         except Exception as e:
    #             st.error(f"Error creating debug package: {e}")
    #             st.error("Please contact support for assistance")

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
        st.text("MW = 27 kDa")

    st.markdown("### Advanced Curve Fitting Options")
    st.info("""
    **Advanced biophysical model fitting is now available in Multi-Group Comparison (Tab 2)!**
    
    ✅ Available Models:
    - Anomalous Diffusion (subdiffusive/superdiffusive recovery)
    - Reaction-Diffusion Simple (fast diffusion + slow binding)
    - Reaction-Diffusion Full (3-state model)
    
    📍 Location: Tab 2 → Multi-Group Comparison → Expand "Advanced Curve Fitting" section
    
    💡 Advanced models are applied to **mean recovery curves** (averaged across cells) for:
    - Higher signal-to-noise ratio
    - Mechanistic parameter extraction (D, k_on, k_off, etc.)
    - Automatic model selection by AIC
    - Biological interpretation generation
    """)
    
    col_fit1, col_fit2 = st.columns(2)

    with col_fit1:
        # Individual cell fitting uses standard least squares (simple exponential models)
        # Advanced biophysical models are available for mean curves in Tab 2
        fitting_method = "least_squares" # Standard method for individual cell fitting

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

    # confidence_intervals = st.checkbox(
    #     "Calculate Confidence Intervals",
    #     value=False,
    #     help="Calculate confidence intervals for fitted parameters (computationally intensive)"
    # )

    # bootstrap_samples = st.number_input(
    #     "Bootstrap Samples for CI",
    #     value=1000,
    #     min_value=100,
    #     max_value=10000,
    #     step=100,
    #     help="Number of bootstrap samples for confidence interval calculation",
    #     disabled=not confidence_intervals
    # )
    confidence_intervals = False # Hardcode to false
    bootstrap_samples = 1000

if st.button("Apply Settings", type="primary"):
    st.session_state.settings.update({
        'default_criterion': default_criterion, 'default_gfp_diffusion': default_gfp_diffusion, 'default_gfp_rg': default_gfp_rg,
        'default_bleach_radius': default_bleach_radius, 'default_pixel_size': default_pixel_size,
        'default_scaling_alpha': default_scaling_alpha, 'default_target_mw': default_target_mw, 'decimal_places': decimal_places,
        'fitting_method': fitting_method, 'max_iterations': max_iterations,
        'parameter_bounds': parameter_bounds, 'confidence_intervals': confidence_intervals,
        'bootstrap_samples': bootstrap_samples
    })
    st.success("Settings applied successfully.")
    st.rerun()

st.markdown("### Data Management")
if st.checkbox("I understand that this will DELETE all loaded data and groups."):
    if st.button("Clear All Data", type="secondary"):
        st.session_state.data_manager = FRAPDataManager()
        st.session_state.selected_group_name = None
        st.success("All data cleared successfully.")
        st.rerun()

with tab7:
    display_reference_database_ui()
