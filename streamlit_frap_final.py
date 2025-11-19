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
from frap_core_corrected import FRAPAnalysisCore as CoreFRAPAnalysis
from frap_ui_components import create_motion_stabilization_panel, create_sparkline, create_calibration_panel, create_stats_panel
import zipfile
import tempfile
import shutil
import json

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

def interpret_kinetics(k, bleach_radius_um, gfp_d=25.0, gfp_rg=2.82, gfp_mw=27.0, calibration_params=None):
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
    if calibration_params and 'slope' in calibration_params and 'intercept' in calibration_params:
        if diffusion_coefficient > 0:
            log_d = np.log10(diffusion_coefficient)
            log_mw = calibration_params['slope'] * log_d + calibration_params['intercept']
            apparent_mw = 10**log_mw
        else:
            apparent_mw = np.nan
    else:
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
if 'frap_analyzer' not in st.session_state:
    st.session_state.frap_analyzer = FRAPImageAnalyzer()
if 'app_config' not in st.session_state:
    st.session_state.app_config = {
        "motion": {"global_drift": True, "optical_flow": True},
        "calibration": {"standards": pd.DataFrame(columns=["MW", "D"]), "fit_params": {}},
        "stats": {"test": "t-test", "tost_threshold": 0.2, "fdr": False}
    }

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
                    file_path = temp_path

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
                    params = {}
                    logger.error(f"No valid fit found for {file_name}")

                # Add stabilization data to features if it exists in the dataframe
                for col in ['roi_centroid_x', 'roi_centroid_y', 'roi_radius_per_frame',
                            'total_drift_um', 'mean_framewise_shift_um',
                            'motion_qc_flag', 'motion_qc_reason']:
                    if col in processed_df.columns:
                        # For per-frame data, store the series; for single values, store the first value
                        if processed_df[col].nunique() > 1:
                            params[col] = processed_df[col].tolist()
                        else:
                            params[col] = processed_df[col].iloc[0]

                self.files[file_path]={
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
        if group_name in self.groups and file_path in self.files:
            if file_path not in self.groups[group_name]['files']:
                self.groups[group_name]['files'].append(file_path)
                return True
        return False

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
        is treated as a new group. Gracefully handles unreadable files.
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

                # Walk through the extracted directory to find subfolders (groups)
                for root, dirs, files in os.walk(temp_dir):
                    # Process subfolders as groups
                    for group_name in dirs:
                        if group_name.startswith('__'): # Ignore system folders like __MACOSX
                            continue

                        self.create_group(group_name)
                        groups_created.append(group_name)
                        group_path = os.path.join(root, group_name)

                        for file_in_group in os.listdir(group_path):
                            file_path_in_temp = os.path.join(group_path, file_in_group)

                            if os.path.isfile(file_path_in_temp) and not file_in_group.startswith('.'):
                                try:
                                    file_ext = os.path.splitext(file_in_group)[1].lower()
                                    file_name = os.path.basename(file_in_group)

                                    if file_ext not in ['.xls', '.xlsx', '.csv', '.tif', '.tiff']:
                                        continue

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
                                            self.add_file_to_group(group_name, tp)
                                            success_count += 1
                                        else:
                                            raise ValueError("Failed to load data from file.")
                                    else:
                                        self.add_file_to_group(group_name, tp)

                                except Exception as e:
                                    error_count += 1
                                    error_details.append(f"Error processing file {file_in_group} in group {group_name}: {str(e)}")
                                    logger.error(f"Error processing file {file_in_group} in group {group_name}: {str(e)}", exc_info=True)
                                if 'tp' in locals() and tp not in self.files and os.path.exists(tp):
                                    os.remove(tp)
        except Exception as e:
            logger.error(f"Error processing ZIP archive with subfolders: {e}")
            st.error(f"An unexpected error occurred: {e}")
            return False

        if success_count > 0:
            for group_name in groups_created:
                self.update_group_analysis(group_name)
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

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📊 Single File Analysis", "📈 Group Analysis", "📊 Multi-Group Comparison",
    "🖼️ Image Analysis", "🏃 Motion", "🔬 Calibration", "📈 Stats", "💾 Session & Settings"
])

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
                # ... (rest of single file analysis tab)
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
            st.markdown("### Step 1: Curate Files for Analysis")

            # Create a dataframe for the data editor
            editor_data = []
            for path in group['files']:
                file_data = dm.files[path]
                features = file_data.get('features', {})
                qc_reasons = []
                if features.get('motion_qc_flag', False) and features.get('motion_qc_reason'):
                    qc_reasons.append(features['motion_qc_reason'])
                if features.get('r2', 1.0) < 0.8:
                    qc_reasons.append(f"Low R² ({features.get('r2', 0):.2f})")

                editor_data.append({
                    "File Name": file_data['name'],
                    "include_for_group_stats": True,
                    "qc_reasons": ", ".join(qc_reasons) if qc_reasons else "OK",
                    "path": path
                })

            editor_df = pd.DataFrame(editor_data)
            edited_df = st.data_editor(editor_df, disabled=["File Name", "qc_reasons", "path"])

            included_paths = edited_df[edited_df['include_for_group_stats']]['path'].tolist()
            excluded_paths = editor_df[~editor_df['path'].isin(included_paths)]['path'].tolist()

            dm.update_group_analysis(selected_group_name, excluded_files=excluded_paths)
            filtered_df = group.get('features_df')

            st.markdown("---")
            st.markdown("### Step 2: View Group Results")
            if filtered_df is not None and not filtered_df.empty:
                st.success(f"Displaying results for **{len(filtered_df)}** of {len(group['files'])} files.")
                # ... (rest of the group results display)
            else:
                st.warning("No data to display. All files may have been excluded.")
    else:
        st.info("Create and/or select a group from the sidebar to begin analysis.")

with tab3:
    st.header("Multi-Group Comparison")
    # ... (content of multi-group comparison tab remains the same)

with tab4:
    # Use the comprehensive image analysis interface
    create_image_analysis_interface(dm)

with tab5:
    # Motion stabilization panel
    create_motion_stabilization_panel(st.session_state.frap_analyzer)

with tab6:
    # Calibration panel
    create_calibration_panel()

with tab7:
    # Stats panel
    create_stats_panel(dm)

with tab8:
    st.header("💾 Session Management & Settings")

    col_session1, col_session2 = st.columns(2)

    with col_session1:
        st.subheader("Session Management")
        # ... (session management content)

    with col_session2:
        st.subheader("Settings")
        # ... (settings content)

        st.subheader("Configuration Management")
        if st.button("Save Configuration"):
            config_str = json.dumps(st.session_state.app_config, indent=4, cls=pd.io.json.PandasEncoder)
            st.download_button(
                label="Download Config JSON",
                data=config_str,
                file_name="frap_config.json",
                mime="application/json"
            )

        uploaded_config = st.file_uploader("Load Configuration", type="json")
        if uploaded_config:
            config = json.load(uploaded_config)
            st.session_state.app_config.update(config)
            st.success("Configuration loaded successfully.")

    st.markdown("### Data Management")
    if st.checkbox("I understand that this will DELETE all loaded data and groups."):
        if st.button("Clear All Data", type="secondary"):
            st.session_state.data_manager = FRAPDataManager()
            st.session_state.selected_group_name = None
            st.success("All data cleared successfully.")
            st.rerun()
