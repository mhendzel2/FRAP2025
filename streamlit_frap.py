"""
FRAP Analysis App - Final Verified Version
A comprehensive FRAP analysis application with supervised outlier removal, sequential group plots,
a detailed kinetics table showing proportions relative to both the mobile pool and the total population,
and all original settings functionality restored.
"""

import streamlit as st

from streamlit_compat import patch_streamlit_width

patch_streamlit_width(st)
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
try:
    from frap_html_reports import generate_html_report
    HTML_REPORTS_AVAILABLE = True
except Exception:
    generate_html_report = None
    HTML_REPORTS_AVAILABLE = False
from frap_image_analysis import FRAPImageAnalyzer, create_image_analysis_interface
from frap_core import FRAPAnalysisCore as CoreFRAPAnalysis
from calibration import Calibration

# Optional v2 comparative analysis modules
try:
    from frap_populations_v2 import detect_heterogeneity_v2
    from frap_plots_v2 import plot_publication_violin
    from frap_comparison_v2 import UnifiedGroupComparator
    COMPARATIVE_ANALYSIS_V2_AVAILABLE = True
except Exception:
    COMPARATIVE_ANALYSIS_V2_AVAILABLE = False
    detect_heterogeneity_v2 = None
    plot_publication_violin = None
    UnifiedGroupComparator = None
import zipfile
import tempfile
import shutil
# --- Page and Logging Configuration ---
st.set_page_config(page_title="FRAP Analysis", page_icon="🔬", layout="wide", initial_sidebar_state="expanded")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import advanced fitting methods
try:
    from frap_robust_bayesian import (
        robust_fit_single_exp, robust_fit_double_exp, 
        bayesian_fit_single_exp, compare_fitting_methods,
        AdvancedFitResult
    )
    ADVANCED_FITTING_AVAILABLE = True
except ImportError:
    ADVANCED_FITTING_AVAILABLE = False
    logger.warning("Advanced fitting methods (robust/Bayesian) not available")
os.makedirs('data', exist_ok=True)


def apply_advanced_fitting(t: np.ndarray, intensity: np.ndarray, 
                          fitting_method: str, 
                          bleach_radius_um: float = 1.0) -> Optional[Dict[str, Any]]:
    """
    Apply advanced fitting methods (robust or Bayesian) to FRAP recovery data.
    
    Args:
        t: Time array (post-bleach)
        intensity: Normalized intensity array
        fitting_method: One of 'robust_soft_l1', 'robust_huber', 'robust_tukey', 'bayesian'
        bleach_radius_um: Bleach spot radius in microns
        
    Returns:
        Dictionary with fit results compatible with CoreFRAPAnalysis format, or None if failed
    """
    if not ADVANCED_FITTING_AVAILABLE:
        logger.warning("Advanced fitting requested but not available")
        return None
    
    try:
        # Map fitting method to function call
        if fitting_method == 'robust_soft_l1':
            result = robust_fit_single_exp(t, intensity, loss_type='soft_l1')
        elif fitting_method == 'robust_huber':
            result = robust_fit_single_exp(t, intensity, loss_type='huber')
        elif fitting_method == 'robust_tukey':
            result = robust_fit_single_exp(t, intensity, loss_type='tukey')
        elif fitting_method == 'bayesian':
            result = bayesian_fit_single_exp(t, intensity)
        else:
            return None
        
        if not result.success:
            logger.warning(f"Advanced fitting failed: {result.message}")
            return None
        
        # Convert AdvancedFitResult to format compatible with CoreFRAPAnalysis
        params = result.params
        
        # Calculate derived quantities
        A = params.get('A', 0)
        C = params.get('C', 0)
        k = params.get('k', 0)
        
        # Mobile/immobile fractions
        total = A + C if (A + C) > 0 else 1.0
        mobile_frac = A / total if total > 0 else 0
        immobile_frac = C / total if total > 0 else 0
        
        # Diffusion coefficient: D = (w² × k) / 4
        D = (bleach_radius_um**2 * k) / 4.0 if k > 0 else 0
        
        # Half-time
        t_half = np.log(2) / k if k > 0 else 0
        
        # Format as standard fit result
        fit_result = {
            'model': 'single',
            'params': params,
            'params_std': result.params_std,
            'r2': result.r2,
            'rmse': result.rmse,
            'aic': result.aic,
            'bic': result.bic,
            'fitted_values': result.fitted_values,
            'residuals': result.residuals,
            'mobile_fraction': mobile_frac,
            'immobile_fraction': immobile_frac,
            'koff': k,
            'D': D,
            't_half': t_half,
            'n_outliers': result.n_outliers,
            'fitting_method': fitting_method
        }
        
        # Add Bayesian-specific info if available
        if result.params_median:
            fit_result['params_median'] = result.params_median
            fit_result['params_lower'] = result.params_lower
            fit_result['params_upper'] = result.params_upper
            
        return fit_result
        
    except Exception as e:
        logger.error(f"Advanced fitting error: {e}")
        return None


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
if 'calibration_standards' not in st.session_state:
    st.session_state.calibration_standards = [
        {'name': 'GFP monomer', 'mw_kda': 27, 'd_um2_s': 25.0},
        {'name': 'GFP dimer', 'mw_kda': 54, 'd_um2_s': 17.7},
        {'name': 'GFP trimer', 'mw_kda': 81, 'd_um2_s': 14.4},
    ]
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

def create_sparkline(x_coords, y_coords):
    """Generates a tiny sparkline for centroid displacement."""
    if not isinstance(x_coords, (list, np.ndarray, pd.Series)) or not isinstance(y_coords, (list, np.ndarray, pd.Series)):
        return ""

    x = np.array(x_coords)
    y = np.array(y_coords)

    if len(x) < 2 or len(y) < 2 or pd.isnull(x).all() or pd.isnull(y).all():
        return ""

    # Calculate displacement from the initial point
    displacement = np.sqrt((x - x[0])**2 + (y - y[0])**2)

    fig = go.Figure(go.Scatter(
        x=np.arange(len(displacement)), y=displacement,
        mode='lines', line=dict(color='blue', width=2)
    ))

    fig.update_layout(
        width=100, height=30,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )

    return fig.to_image(format="svg+xml", engine="kaleido")

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
    def __init__(self, calibration=None):
        self.files,self.groups = {},{}
        self.calibration = calibration

    def load_file(self,file_path,file_name):
        try:
            # Extract original extension before the hash suffix
            if '_' in file_path and any(ext in file_path for ext in ['.xls_', '.xlsx_', '.csv_']):
                # Find the original extension and create a temporary file with correct extension
                import tempfile
                import shutil
                if '.xlsx_' in file_path:
                    temp_file = tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False)
                    temp_path = temp_file.name
                    temp_file.close()
                elif '.xls_' in file_path:
                    temp_file = tempfile.NamedTemporaryFile(suffix='.xls', delete=False)
                    temp_path = temp_file.name
                    temp_file.close()
                elif '.csv_' in file_path:
                    temp_file = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
                    temp_path = temp_file.name
                    temp_file.close()
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

                # Initialize variables
                fits = {}
                best_fit = None
                params = {}
                
                # Check if advanced fitting method is selected
                fitting_method = st.session_state.settings.get('fitting_method', 'least_squares')
                
                if fitting_method != 'least_squares' and ADVANCED_FITTING_AVAILABLE:
                    # Use advanced fitting methods
                    # Get post-bleach data
                    t_post, y_post, _ = CoreFRAPAnalysis.get_post_bleach_data(time, intensity)
                    
                    # Get bleach radius for calculations
                    bleach_radius_um = (st.session_state.settings.get('default_bleach_radius', 1.0) * 
                                       st.session_state.settings.get('default_pixel_size', 1.0))
                    
                    # Apply advanced fitting
                    advanced_fit = apply_advanced_fitting(t_post, y_post, fitting_method, bleach_radius_um)
                    
                    if advanced_fit:
                        # Use advanced fit as best fit
                        best_fit = advanced_fit
                        params = CoreFRAPAnalysis.extract_clustering_features(
                            best_fit,
                            bleach_spot_radius=st.session_state.settings.get('default_bleach_radius', 1.0),
                            pixel_size=st.session_state.settings.get('default_pixel_size', 1.0),
                            calibration=self.calibration
                        )
                    else:
                        # Fallback to standard fitting
                        fits = CoreFRAPAnalysis.fit_all_models(time, intensity)
                        best_fit = CoreFRAPAnalysis.select_best_fit(fits, st.session_state.settings['default_criterion'])
                        if best_fit:
                            params = CoreFRAPAnalysis.extract_clustering_features(
                                best_fit,
                                bleach_spot_radius=st.session_state.settings.get('default_bleach_radius', 1.0),
                                pixel_size=st.session_state.settings.get('default_pixel_size', 1.0),
                                calibration=self.calibration
                            )
                        else:
                            params = {}
                else:
                    # Standard fitting
                    fits = CoreFRAPAnalysis.fit_all_models(time,intensity)
                    best_fit = CoreFRAPAnalysis.select_best_fit(fits,st.session_state.settings['default_criterion'])

                    if best_fit:
                        params = CoreFRAPAnalysis.extract_clustering_features(
                            best_fit,
                            bleach_spot_radius=st.session_state.settings.get('default_bleach_radius', 1.0),
                            pixel_size=st.session_state.settings.get('default_pixel_size', 1.0),
                            calibration=self.calibration
                        )
                    else:
                        params = {}
                        logger.error(f"No valid fit found for {file_name}")
                
                # Validate the analysis results
                if params:
                    params = validate_analysis_results(params)
                else:
                    params = {}

                # Store fits for compatibility (may not be available for advanced methods)
                if 'fits' not in locals():
                    fits = {'single': best_fit} if best_fit else {}

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

# --- Calibration Setup ---
calibration = Calibration(st.session_state.calibration_standards)
if calibration.warning:
    st.warning(calibration.warning)

dm = st.session_state.data_manager = FRAPDataManager(calibration=calibration) if st.session_state.data_manager is None else st.session_state.data_manager
dm.calibration = calibration # Ensure calibration object is always up to date

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

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Single File Analysis",
    "📈 Group Analysis",
    "📊 Multi-Group Comparison",
    "⚖️ Comparative Analysis",
    "🖼️ Image Analysis",
    "💾 Session Management",
    "⚙️ Settings",
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

                st.plotly_chart(fig, width="stretch")

                # Add explanation text
                st.markdown("""
                **Plot Explanation:**
                - **Gray points**: Pre-bleach data (shown for context, not included in fitting)
                - **Blue points**: Post-bleach data starting from interpolated bleach point (used for curve fitting)
                - **Red curve**: Fitted exponential recovery model aligned with post-bleach timepoints
                - **Orange dashed line**: Interpolated bleach event (midpoint between last pre-bleach and first post-bleach frames)
                - **Y-axis**: Starts at zero for proper scaling
                """)

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
                    # Note: Using mathematical formulas directly instead of function references
                    # to support session files that may not contain function objects
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
                    st.plotly_chart(comp_fig, width="stretch")

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
                    st.plotly_chart(res_fig, width="stretch")

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

                    kinetics = CoreFRAPAnalysis.interpret_kinetics(
                        primary_rate,
                        bleach_radius_um=effective_radius_um,
                        calibration=dm.calibration
                    )

                    # Validate calculated values
                    if kinetics['diffusion_coefficient'] > 100:
                        st.warning("⚠️ Very high apparent diffusion coefficient - check bleach spot size")
                    if kinetics['apparent_mw'] > 10000:
                        st.warning("⚠️ Very high apparent molecular weight - may indicate aggregation")

                    col_bio1,col_bio2,col_bio3,col_bio4=st.columns(4)
                    with col_bio1:
                        st.metric("App. D (μm²/s)",f"{kinetics['diffusion_coefficient']:.3f}")
                    with col_bio2:
                        st.metric("k_off (s⁻¹)",f"{kinetics['k_off']:.4f}")
                    with col_bio3:
                        mw_str = f"{kinetics['apparent_mw']:.1f}"
                        if np.isfinite(kinetics['apparent_mw_ci_low']):
                            mw_str += f" ({kinetics['apparent_mw_ci_low']:.1f} - {kinetics['apparent_mw_ci_high']:.1f})"
                        st.metric("App. MW (kDa)", mw_str)
                    with col_bio4:
                        immobile_frac = features.get('immobile_fraction', 100 - display_mobile)
                        st.metric("Immobile (%)",f"{immobile_frac:.1f}")

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
                        for key, value in params.items():
                            if 'rate' in key.lower() or 'constant' in key.lower():
                                st.write(f"- {key}: {value}")
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
                        kinetic_interp = CoreFRAPAnalysis.interpret_kinetics(
                            k_val,
                            bleach_radius_um=effective_radius_um,
                            calibration=dm.calibration
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
                st.plotly_chart(fig_indiv, width="stretch")

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
                        kinetic_interp = CoreFRAPAnalysis.interpret_kinetics(
                            primary_rate,
                            bleach_radius_um=effective_radius_um,
                            calibration=dm.calibration
                        )

                        sparkline_svg = ""
                        if features.get('motion_qc_flag', False):
                             sparkline_svg = create_sparkline(
                                 features.get('roi_centroid_x', []),
                                 features.get('roi_centroid_y', [])
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
                            'R²': features.get('r2', np.nan),
                            'Centroid Drift': sparkline_svg
                        })

                    detailed_df = pd.DataFrame(all_files_data)

                    # Style the dataframe with outliers highlighted
                    def highlight_outliers(row):
                        return ['background-color: #ffcccc' if row['Status'] == 'Outlier' else '' for _ in row]

                    with st.expander("📊 Show Detailed Kinetics Table", expanded=True):
                        # Convert dataframe to HTML and embed SVG sparklines
                        html = detailed_df.to_html(escape=False, index=False, formatters={
                            'Mobile (%)': '{:.1f}'.format,
                            'Immobile (%)': '{:.1f}'.format,
                            'Primary Rate (k)': '{:.4f}'.format,
                            'Half-time (s)': '{:.2f}'.format,
                            'k_off (1/s)': '{:.4f}'.format,
                            'App. D (μm²/s)': '{:.3f}'.format,
                            'App. MW (kDa)': '{:.1f}'.format,
                            'R²': '{:.3f}'.format
                        })
                        st.markdown(html, unsafe_allow_html=True)

                        # Summary statistics
                        included_data = detailed_df[detailed_df['Status'] == 'Included']
                        st.markdown("##### Summary Statistics (Included Files Only)")

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

                            st.plotly_chart(fig, width="stretch")

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
                            st.dataframe(stats_df, width="stretch")
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

                    st.markdown("#### Global Fit Analysis / Report Configuration")
                    st.caption(
                        "Select which groups are included in this run, designate control/reference groups for pairwise statistics, "
                        "and set output ordering. These settings are applied before computation."
                    )

                    all_group_names = list(dm.groups.keys())
                    if 'global_fit_report_config' not in st.session_state:
                        default_rows = []
                        for i, gname in enumerate(all_group_names, start=1):
                            g_lower = str(gname).lower()
                            default_role = 'Control' if ('control' in g_lower or g_lower in {'ctrl', 'vehicle'}) else 'Sample'
                            default_rows.append({
                                'Include': True,
                                'Group': gname,
                                'Role': default_role,
                                'Order': i,
                            })
                        st.session_state.global_fit_report_config = pd.DataFrame(default_rows)

                    cfg_df = st.session_state.global_fit_report_config.copy()
                    cfg_df = cfg_df[cfg_df['Group'].isin(all_group_names)].copy()
                    if len(cfg_df) != len(all_group_names):
                        # Sync if groups changed (e.g., new upload)
                        existing = set(cfg_df['Group'].tolist())
                        next_order = int(cfg_df['Order'].max()) + 1 if not cfg_df.empty else 1
                        for gname in all_group_names:
                            if gname not in existing:
                                cfg_df = pd.concat([
                                    cfg_df,
                                    pd.DataFrame([{
                                        'Include': True,
                                        'Group': gname,
                                        'Role': 'Sample',
                                        'Order': next_order,
                                    }])
                                ], ignore_index=True)
                                next_order += 1

                    cfg_df = cfg_df[['Include', 'Group', 'Role', 'Order']].copy()
                    cfg_df['Include'] = cfg_df['Include'].fillna(True).astype(bool)
                    cfg_df['Role'] = cfg_df['Role'].fillna('Sample')
                    cfg_df['Order'] = pd.to_numeric(cfg_df['Order'], errors='coerce').fillna(9999).astype(int)

                    edited_cfg = st.data_editor(
                        cfg_df,
                        hide_index=True,
                        width="stretch",
                        column_config={
                            'Include': st.column_config.CheckboxColumn('Include', help='Include group in this run'),
                            'Group': st.column_config.TextColumn('Group', disabled=True),
                            'Role': st.column_config.SelectboxColumn('Role', options=['Control', 'Sample']),
                            'Order': st.column_config.NumberColumn('Order', min_value=1, step=1),
                        },
                        key='global_fit_report_cfg_editor'
                    )
                    st.session_state.global_fit_report_config = edited_cfg.copy()

                    included_cfg = edited_cfg[edited_cfg['Include']].copy()
                    included_cfg = included_cfg.sort_values(['Order', 'Group'])
                    included_groups = included_cfg['Group'].tolist()
                    control_groups = included_cfg[included_cfg['Role'] == 'Control']['Group'].tolist()

                    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
                    with col_cfg1:
                        compute_stats_vs_control = st.checkbox(
                            "Compute pairwise stats vs control",
                            value=True,
                            help="Runs Welch t-tests comparing each sample group to pooled controls (based on global-fit amplitudes)."
                        )
                    with col_cfg2:
                        render_fit_plots = st.checkbox(
                            "Render global-fit plots",
                            value=False,
                            help="Plots can be slow for large datasets. Enable only when needed."
                        )
                    with col_cfg3:
                        auto_make_reports = st.checkbox(
                            "Generate report after run",
                            value=False,
                            help="Creates reports for the included groups using the same configuration (controls + ordering)."
                        )

                    report_format = "Both" if generate_html_report is not None else "PDF"
                    if auto_make_reports:
                        report_format = st.selectbox(
                            "Report output format",
                            options=["PDF", "HTML", "Both"] if generate_html_report is not None else ["PDF"],
                            index=2 if generate_html_report is not None else 0,
                            help="Choose which report(s) to generate after the configured global fit completes."
                        )

                    if not included_groups:
                        st.warning("No groups selected for analysis. Enable at least one group above.")
                    elif compute_stats_vs_control and not control_groups:
                        st.warning("No control/reference group selected. Mark one or more groups as Role = Control to enable pairwise statistics.")

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
                        if st.button("🚀 Run Global Fit (Configured)", type="primary", disabled=(len(included_groups) == 0)):
                            try:
                                with st.spinner(f"Performing global {global_model}-component fitting..."):
                                    def _excluded_for_group(gname: str) -> list:
                                        if include_outliers_global:
                                            return []
                                        g = dm.groups.get(gname, {})
                                        auto = g.get('auto_outliers', [])
                                        return list(auto) if isinstance(auto, (list, tuple)) else []

                                    def _extract_amplitudes(result: dict) -> dict:
                                        vals = []
                                        for params in result.get('individual_params', []) or []:
                                            if not isinstance(params, dict):
                                                continue
                                            if global_model == 'single':
                                                vals.append(params.get('A', np.nan))
                                            elif global_model == 'double':
                                                a1 = params.get('A1', np.nan)
                                                a2 = params.get('A2', np.nan)
                                                vals.append(a1 + a2)
                                            elif global_model == 'triple':
                                                a1 = params.get('A1', np.nan)
                                                a2 = params.get('A2', np.nan)
                                                a3 = params.get('A3', np.nan)
                                                vals.append(a1 + a2 + a3)
                                        arr = np.array(vals, dtype=float)
                                        arr = arr[np.isfinite(arr)]
                                        return {'total_amplitude': arr}

                                    results_by_group = {}
                                    for gname in included_groups:
                                        res = dm.fit_group_models(
                                            gname,
                                            model=global_model,
                                            excluded_files=_excluded_for_group(gname)
                                        )
                                        results_by_group[gname] = res

                                    failed = [g for g, r in results_by_group.items() if not r or not r.get('success', False)]
                                    if failed:
                                        st.error(f"❌ Global fitting failed for: {', '.join(failed)}")
                                        for g in failed:
                                            st.caption(f"{g}: {results_by_group[g].get('error', 'Unknown error')}")

                                    ok_groups = [g for g, r in results_by_group.items() if r and r.get('success', False)]
                                    if not ok_groups:
                                        st.stop()

                                    st.success(f"✅ Global fitting completed for {len(ok_groups)} group(s)!")

                                    # Summary table
                                    summary_rows = []
                                    for gname in ok_groups:
                                        r = results_by_group[gname]
                                        sp = r.get('shared_params', {}) or {}
                                        row = {
                                            'Group': gname,
                                            'Model': global_model,
                                            'N traces': int(len(r.get('file_names', []) or [])),
                                            'Mean R²': r.get('mean_r2', np.nan),
                                            'AIC': r.get('aic', np.nan),
                                        }
                                        if global_model == 'single':
                                            row['k'] = sp.get('k', np.nan)
                                        elif global_model == 'double':
                                            row['k1'] = sp.get('k1', np.nan)
                                            row['k2'] = sp.get('k2', np.nan)
                                        elif global_model == 'triple':
                                            row['k1'] = sp.get('k1', np.nan)
                                            row['k2'] = sp.get('k2', np.nan)
                                            row['k3'] = sp.get('k3', np.nan)
                                        summary_rows.append(row)
                                    global_fit_summary_df = pd.DataFrame(summary_rows)
                                    st.markdown("#### Global Fit Summary")
                                    st.dataframe(global_fit_summary_df, width="stretch")

                                    # Pairwise stats vs pooled controls (based on total amplitude)
                                    if compute_stats_vs_control and control_groups:
                                        pooled_ctrl = []
                                        for g in control_groups:
                                            r = results_by_group.get(g)
                                            if r and r.get('success', False):
                                                pooled_ctrl.append(_extract_amplitudes(r)['total_amplitude'])
                                        if pooled_ctrl:
                                            pooled_ctrl = np.concatenate(pooled_ctrl)

                                            try:
                                                from scipy import stats as _scipy_stats  # type: ignore
                                            except Exception:
                                                _scipy_stats = None

                                            stats_rows = []
                                            for gname in ok_groups:
                                                if gname in control_groups:
                                                    continue
                                                arr = _extract_amplitudes(results_by_group[gname])['total_amplitude']
                                                if len(arr) < 2 or len(pooled_ctrl) < 2:
                                                    pval = np.nan
                                                else:
                                                    if _scipy_stats is not None:
                                                        _t, pval = _scipy_stats.ttest_ind(
                                                            pooled_ctrl,
                                                            arr,
                                                            equal_var=False,
                                                            nan_policy='omit'
                                                        )
                                                    else:
                                                        pval = np.nan
                                                stats_rows.append({
                                                    'Sample group': gname,
                                                    'Control pooled N': int(len(pooled_ctrl)),
                                                    'Sample N': int(len(arr)),
                                                    'Control mean': float(np.mean(pooled_ctrl)) if len(pooled_ctrl) else np.nan,
                                                    'Sample mean': float(np.mean(arr)) if len(arr) else np.nan,
                                                    'p (Welch t-test)': pval,
                                                })

                                            if stats_rows:
                                                st.markdown("#### Pairwise Statistics vs Pooled Controls")
                                                st.dataframe(pd.DataFrame(stats_rows), width="stretch")

                                    # Optional plotting (per-group)
                                    if render_fit_plots:
                                        st.markdown("#### Global Fit Plots")
                                        for gname in ok_groups:
                                            r = results_by_group[gname]
                                            fig_global = go.Figure()
                                            fitted_curves = r.get('fitted_curves', []) or []
                                            common_time = r.get('common_time', None)
                                            if common_time is None:
                                                continue

                                            # Add traces
                                            group_obj = dm.groups.get(gname, {})
                                            for file_name, fitted_curve in zip(r.get('file_names', []) or [], fitted_curves):
                                                file_path = None
                                                for fp in group_obj.get('files', []) or []:
                                                    if fp in (r.get('excluded_files', []) or []):
                                                        continue
                                                    if fp in dm.files and dm.files[fp].get('name') == file_name:
                                                        file_path = fp
                                                        break
                                                if not file_path:
                                                    continue
                                                file_data = dm.files[file_path]
                                                t_post, i_post, _ = CoreFRAPAnalysis.get_post_bleach_data(
                                                    file_data['time'], file_data['intensity']
                                                )
                                                fig_global.add_trace(go.Scatter(
                                                    x=t_post, y=i_post,
                                                    mode='markers',
                                                    name=f"{file_name} (data)",
                                                    marker=dict(size=4, opacity=0.55),
                                                    showlegend=False
                                                ))
                                                fig_global.add_trace(go.Scatter(
                                                    x=common_time, y=fitted_curve,
                                                    mode='lines',
                                                    name=f"{file_name} (fit)",
                                                    line=dict(width=2),
                                                    showlegend=False
                                                ))

                                            fig_global.update_layout(
                                                title=f"{gname}: Global {global_model.title()}-Component Fit",
                                                xaxis_title="Time (s)",
                                                yaxis_title="Normalized Intensity",
                                                height=450
                                            )
                                            st.plotly_chart(fig_global, width="stretch")

                                    # Optional report generation using same configuration
                                    if auto_make_reports:
                                        report_settings = dict(st.session_state.settings)
                                        report_settings.update({
                                            'report_controls': control_groups,
                                            'report_group_order': included_groups,
                                            'report_subgroup': 'Global Fitting',
                                            'report_global_fit_model': global_model,
                                        })

                                        make_pdf = report_format in {'PDF', 'Both'}
                                        make_html = (report_format in {'HTML', 'Both'}) and (generate_html_report is not None)

                                        outputs = []
                                        if make_pdf:
                                            try:
                                                pdf_path = generate_pdf_report(
                                                    dm,
                                                    included_groups,
                                                    None,
                                                    report_settings
                                                )
                                                outputs.append(('PDF', pdf_path))
                                            except Exception as e:
                                                st.error(f"PDF report failed: {e}")

                                        if make_html:
                                            try:
                                                html_path = generate_html_report(
                                                    dm,
                                                    included_groups,
                                                    None,
                                                    report_settings
                                                )
                                                outputs.append(('HTML', html_path))
                                            except Exception as e:
                                                st.error(f"HTML report failed: {e}")

                                        if outputs:
                                            st.success("Reports generated.")

                                            # Offer downloads
                                            for label, out_path in outputs:
                                                try:
                                                    with open(out_path, 'rb') as f:
                                                        data = f.read()
                                                    mime = 'application/pdf' if label == 'PDF' else 'text/html'
                                                    fname = os.path.basename(out_path)
                                                    st.download_button(
                                                        label=f"⬇️ Download {label} ({fname})",
                                                        data=data,
                                                        file_name=fname,
                                                        mime=mime
                                                    )
                                                except Exception as e:
                                                    st.error(f"Failed to prepare {label} download: {e}")

                            except Exception as e:
                                st.error(f"Error during global fitting: {e}")

                st.markdown("---")
                st.markdown("### Step 7: Group Recovery Plots")
                plot_data={path:dm.files[path] for path in filtered_df['file_path'].tolist()}
                st.markdown("##### Average Recovery Curve")
                avg_fig = plot_average_curve(plot_data)
                st.plotly_chart(avg_fig, width="stretch")
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
            st.plotly_chart(fig, width="stretch")

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
            st.markdown("### Automated Report Generation")

            col_pdf1, col_pdf2 = st.columns([2, 1])

            with col_pdf1:
                st.markdown("Generate a comprehensive statistical analysis report including:")
                st.markdown("- Executive summary with group comparisons")
                st.markdown("- Statistical test results (t-tests, ANOVA, effect sizes)")
                st.markdown("- Publication-ready visualizations and tables")
                st.markdown("- Detailed results for each experimental group")

            with col_pdf2:
                st.markdown("**Report Generator Options**")
                st.caption("Define control(s), explicit group order, and subgroup analysis sets.")

                group_names = list(dm.groups.keys())
                default_cfg = pd.DataFrame({
                    'Include': [True] * len(group_names),
                    'Group': group_names,
                    'Role': ['Sample'] * len(group_names),
                    'Order': list(range(1, len(group_names) + 1)),
                    'Subgroup': ['All'] * len(group_names),
                })

                cfg_df = st.data_editor(
                    default_cfg,
                    hide_index=True,
                    key='report_generator_config_v1',
                    column_config={
                        'Include': st.column_config.CheckboxColumn(help='Include this group in report generation'),
                        'Role': st.column_config.SelectboxColumn(options=['Control', 'Sample'], help='Control groups are pooled for control-based comparisons'),
                        'Order': st.column_config.NumberColumn(min_value=1, step=1, help='Lower numbers appear first in the report'),
                        'Subgroup': st.column_config.TextColumn(help='Use to create subgroup analysis sets (e.g., A, B, C). Use "All" for one combined report.'),
                    },
                    disabled=['Group'],
                    use_container_width=True,
                )

                report_format = st.selectbox(
                    "Report output format",
                    options=["PDF", "HTML", "Both"] if HTML_REPORTS_AVAILABLE else ["PDF"],
                    index=0,
                    key='report_output_format_v1',
                    help="Choose which report(s) to generate for each subgroup set."
                )

                make_pdf = report_format in {'PDF', 'Both'}
                make_html = (report_format in {'HTML', 'Both'}) and HTML_REPORTS_AVAILABLE
                if report_format in {'HTML', 'Both'} and not HTML_REPORTS_AVAILABLE:
                    st.caption("HTML reports unavailable (missing frap_html_reports dependencies).")

                run_disabled = (cfg_df is None) or (cfg_df['Include'].sum() < 2) or (not make_pdf and not make_html)
                if st.button("📄 Generate Report(s)", type="primary", disabled=run_disabled):
                    try:
                        with st.spinner("Generating report(s)..."):
                            cfg = cfg_df.copy()
                            cfg = cfg[cfg['Include'] == True]
                            cfg['Order'] = pd.to_numeric(cfg['Order'], errors='coerce').fillna(9999).astype(int)
                            cfg['Subgroup'] = cfg['Subgroup'].fillna('All').astype(str)
                            cfg = cfg.sort_values(['Subgroup', 'Order', 'Group'])

                            # Split into subgroup analysis sets
                            subgroup_labels = cfg['Subgroup'].unique().tolist() if not cfg.empty else []
                            outputs = []  # (filename, bytes, mime)

                            for subgroup in subgroup_labels:
                                sub_cfg = cfg[cfg['Subgroup'] == subgroup]
                                groups_ordered = sub_cfg['Group'].tolist()
                                control_groups = sub_cfg[sub_cfg['Role'] == 'Control']['Group'].tolist()

                                # Merge report metadata into settings
                                report_settings = dict(st.session_state.settings)
                                report_settings['report_controls'] = control_groups
                                report_settings['report_group_order'] = groups_ordered
                                report_settings['report_subgroup'] = subgroup

                                timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                                safe_sub = ''.join([c if c.isalnum() or c in ('-', '_') else '_' for c in str(subgroup)])

                                if make_pdf:
                                    pdf_filename = f"FRAP_Report_{safe_sub}_{timestamp}.pdf"
                                    pdf_path = generate_pdf_report(
                                        data_manager=dm,
                                        groups_to_compare=groups_ordered,
                                        output_filename=pdf_filename,
                                        settings=report_settings,
                                    )
                                    if pdf_path:
                                        with open(pdf_path, 'rb') as f:
                                            outputs.append((pdf_filename, f.read(), 'application/pdf'))
                                        os.remove(pdf_path)

                                if make_html and generate_html_report is not None:
                                    html_filename = f"FRAP_Report_{safe_sub}_{timestamp}.html"
                                    html_path = generate_html_report(
                                        data_manager=dm,
                                        groups_to_compare=groups_ordered,
                                        output_filename=html_filename,
                                        settings=report_settings,
                                    )
                                    if html_path:
                                        with open(html_path, 'rb') as f:
                                            outputs.append((html_filename, f.read(), 'text/html'))
                                        os.remove(html_path)

                            if not outputs:
                                st.error("No reports were generated. Ensure selected groups have processed data.")
                            elif len(outputs) == 1:
                                fn, data, mime = outputs[0]
                                st.download_button(
                                    label=f"⬇️ Download {fn}",
                                    data=data,
                                    file_name=fn,
                                    mime=mime,
                                )
                                st.success("Report generated successfully!")
                            else:
                                # Multiple subgroups -> ZIP
                                import zipfile
                                zip_buf = io.BytesIO()
                                with zipfile.ZipFile(zip_buf, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
                                    for fn, data, _mime in outputs:
                                        zf.writestr(fn, data)
                                zip_buf.seek(0)
                                zip_name = f"FRAP_Reports_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.zip"
                                st.download_button(
                                    label="⬇️ Download Reports (ZIP)",
                                    data=zip_buf.getvalue(),
                                    file_name=zip_name,
                                    mime="application/zip",
                                )
                                st.success(f"Generated {len(outputs)} report file(s) across {len(subgroup_labels)} subgroup(s).")

                    except Exception as e:
                        st.error(f"Error generating report(s): {e}")
                        st.error("Please ensure all selected groups have processed data")

with tab4:
    st.header("⚖️ Comparative Group Analysis")

    if not COMPARATIVE_ANALYSIS_V2_AVAILABLE:
        st.error(
            "Comparative Analysis modules (v2) are not available. "
            "Please ensure frap_comparison_v2.py, frap_plots_v2.py, and frap_populations_v2.py are present."
        )
    elif not dm.groups:
        st.warning("Please upload data first (create groups).")
    else:
        all_groups = list(dm.groups.keys())
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("1. Configuration")
            controls = st.multiselect(
                "Select Control Group(s) (Pooled)",
                all_groups,
                help="Selected controls are pooled to determine the reference model.",
            )
            samples = st.multiselect(
                "Select Sample Groups",
                [g for g in all_groups if g not in controls],
                help="These groups are fit using the control's best model.",
            )

            st.subheader("Heterogeneity Settings")
            sens_mode = st.select_slider(
                "Sub-population Sensitivity",
                options=["low", "normal", "high"],
                value="normal",
                help="Low = requires strong evidence to split populations.",
            )

            st.subheader("Report Configuration")
            metric_options = [
                ("D", "Diffusion (D)"),
                ("mobile_fraction", "Mobile Fraction"),
                ("k", "Rate Constant (k)"),
                ("F_fast", "Fast Fraction (F_fast)"),
            ]
            selected_metrics = []
            for key, label in metric_options:
                if st.checkbox(label, value=(key in ["D", "mobile_fraction"]), key=f"cmp_metric_{key}"):
                    selected_metrics.append(key)

            run_btn = st.button("Run Comparative Analysis", type="primary")

        with col2:
            if run_btn:
                if not controls or not samples:
                    st.warning("Select at least one control group and one sample group.")
                else:
                    if UnifiedGroupComparator is None:
                        st.error("Comparator unavailable; cannot run comparative analysis.")
                        comparator = None
                    else:
                        comparator = UnifiedGroupComparator(dm)

                    with st.spinner("Pooling controls and determining best model..."):
                        if comparator is None:
                            pooled_ctrl_data = {}
                            best_model = 'single'
                        else:
                            pooled_ctrl_data = comparator.pool_controls(controls)
                            best_model = comparator.determine_best_model(pooled_ctrl_data)
                            st.success(f"Reference Model Determined: **{best_model.capitalize()} Exponential**")

                    with st.spinner(f"Fitting all groups using '{best_model}' model..."):
                        if comparator is None:
                            df_ctrl = pd.DataFrame()
                        else:
                            df_ctrl = comparator.fit_group_with_model("Pooled Control", pooled_ctrl_data, best_model)

                        dfs = [df_ctrl]
                        for s_name in samples:
                            if comparator is None:
                                continue
                            s_data = comparator.get_group_curve_data(s_name)
                            df_s = comparator.fit_group_with_model(s_name, s_data, best_model)
                            dfs.append(df_s)

                        full_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
                        st.session_state['comparison_results_v2'] = {
                            'df': full_df,
                            'control_name': 'Pooled Control',
                            'best_model': best_model,
                            'selected_metrics': selected_metrics,
                            'sens_mode': sens_mode,
                        }

            if 'comparison_results_v2' in st.session_state:
                payload = st.session_state['comparison_results_v2']
                res_df = payload.get('df')
                control_name = payload.get('control_name', 'Pooled Control')
                selected_metrics = payload.get('selected_metrics', [])
                sens_mode = payload.get('sens_mode', 'normal')

                comparator = UnifiedGroupComparator(dm) if UnifiedGroupComparator is not None else None

                if res_df is None or res_df.empty:
                    st.warning("No results to display (no curves were fit).")
                else:
                    res_t1, res_t2, res_t3 = st.tabs(["🎻 Graphics", "🔢 Statistics", "💾 Data"])

                    with res_t1:
                        st.subheader("Kinetic Parameter Distributions")
                        group_options = [g for g in res_df['Group'].dropna().unique().tolist()]
                        group_options.sort()
                        groups_to_plot = st.multiselect(
                            "Groups to plot",
                            options=group_options,
                            default=group_options,
                            help="Choose which fitted groups appear in the violin plots.",
                            key="comparison_v2_groups_to_plot",
                        )

                        plot_df = res_df[res_df['Group'].isin(groups_to_plot)].copy() if groups_to_plot else pd.DataFrame()
                        if plot_df.empty:
                            st.warning("No groups selected for plotting.")
                        for metric in selected_metrics:
                            if metric in plot_df.columns:
                                if plot_publication_violin is None:
                                    st.warning("Violin plot helper unavailable; cannot render plots.")
                                    break
                                fig = plot_publication_violin(
                                    plot_df,
                                    x_col='Group',
                                    y_col=metric,
                                    color_col='Group',
                                    title=f"{metric} by Group",
                                )
                                st.plotly_chart(fig, width="stretch")

                    with res_t2:
                        st.subheader("Pairwise Statistical Comparison (vs Control)")
                        metrics_for_stats = [m for m in selected_metrics if m in res_df.columns]
                        if comparator is None:
                            st.warning("Comparator unavailable; cannot compute statistics.")
                        else:
                            stats_df = comparator.calculate_pairwise_stats(res_df, control_name, metrics=metrics_for_stats)
                            st.dataframe(stats_df)

                        st.subheader("Heterogeneity Analysis (Sensitivity Applied)")
                        for grp in res_df['Group'].unique():
                            grp_data = res_df[res_df['Group'] == grp]

                            feature_cols = []
                            if 'k' in grp_data.columns:
                                feature_cols.append('k')
                            elif 'k_fast' in grp_data.columns:
                                feature_cols.append('k_fast')
                            if 'mobile_fraction' in grp_data.columns:
                                feature_cols.append('mobile_fraction')

                            if len(feature_cols) < 2:
                                st.write(f"**{grp}**: Not enough features for clustering.")
                                continue

                            if detect_heterogeneity_v2 is None:
                                st.write(f"**{grp}**: Heterogeneity detector unavailable.")
                            else:
                                _, k_found, _ = detect_heterogeneity_v2(
                                    grp_data,
                                    feature_cols=feature_cols,
                                    sensitivity=sens_mode,
                                )
                                st.write(f"**{grp}**: Identified **{k_found}** distinct recovery profile(s).")

                    with res_t3:
                        st.subheader("Export Data")
                        csv = res_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "Download Full Comparison Data (CSV)",
                            csv,
                            "frap_comparison_results.csv",
                            "text/csv",
                            key='download-csv-comparison-v2'
                        )
                        st.dataframe(res_df)

with tab5:
    # Use the comprehensive image analysis interface
    create_image_analysis_interface()

with tab6:
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

                def sanitize_for_pickle(data):
                    """Remove unpickleable objects like function references"""
                    if isinstance(data, dict):
                        sanitized = {}
                        for key, value in data.items():
                            if key == 'func':  # Skip function objects
                                continue
                            elif callable(value):  # Skip any callable objects
                                continue
                            else:
                                sanitized[key] = sanitize_for_pickle(value)
                        return sanitized
                    elif isinstance(data, list):
                        return [sanitize_for_pickle(item) for item in data]
                    elif isinstance(data, tuple):
                        return tuple(sanitize_for_pickle(item) for item in data)
                    else:
                        return data

                # Sanitize the data to remove unpickleable objects
                sanitized_files = sanitize_for_pickle(dm.files)
                sanitized_groups = sanitize_for_pickle(dm.groups)

                session_data = {
                    'files': sanitized_files,
                    'groups': sanitized_groups,
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

                                    kinetic_interp = CoreFRAPAnalysis.interpret_kinetics(
                                        primary_rate,
                                        bleach_radius_um=effective_radius_um,
                                        calibration=dm.calibration
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
                                        'Apparent_MW_CI_low': kinetic_interp['apparent_mw_ci_low'],
                                        'Apparent_MW_CI_high': kinetic_interp['apparent_mw_ci_high'],
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
                                kinetic_interp = CoreFRAPAnalysis.interpret_kinetics(
                                    primary_rate,
                                    bleach_radius_um=st.session_state.settings.get('default_bleach_radius', 1.0) *
                                                   st.session_state.settings.get('default_pixel_size', 1.0),
                                    calibration=dm.calibration
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
                                    'Apparent_MW_CI_low': kinetic_interp['apparent_mw_ci_low'],
                                    'Apparent_MW_CI_high': kinetic_interp['apparent_mw_ci_high'],
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

with tab7:
    st.subheader("Application Settings")
    st.markdown("### Molecular Weight Calibration")
    st.markdown("Define standards to calibrate diffusion coefficients to apparent molecular weight.")

    edited_standards = st.data_editor(
        pd.DataFrame(st.session_state.calibration_standards),
        num_rows="dynamic",
        key="standards_editor"
    )

    if st.button("Apply Calibration Standards"):
        st.session_state.calibration_standards = edited_standards.to_dict('records')
        st.success("Calibration standards updated.")
        st.rerun()

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
    col_fit1, col_fit2 = st.columns(2)

    with col_fit1:
        if ADVANCED_FITTING_AVAILABLE:
            fitting_method = st.selectbox(
                "Curve Fitting Method",
                ["least_squares", "robust_soft_l1", "robust_huber", "robust_tukey", "bayesian"],
                index=0,
                format_func=lambda x: {
                    "least_squares": "Standard Least Squares",
                    "robust_soft_l1": "Robust (Soft L1 - recommended)",
                    "robust_huber": "Robust (Huber M-estimator)",
                    "robust_tukey": "Robust (Tukey Biweight)",
                    "bayesian": "Bayesian MCMC (full uncertainty)"
                }[x],
                help="""Choose fitting algorithm:
                - Standard: Fast, assumes no outliers
                - Robust: Outlier-resistant, good for noisy data
                - Bayesian: Full uncertainty quantification via MCMC"""
            )
        else:
            st.warning("Advanced fitting methods not available. Install with: pip install emcee")
            fitting_method = "least_squares"
        
        # Add explanatory information
        with st.expander("ℹ️ About Fitting Methods"):
            st.markdown("""
            **Standard Least Squares**
            - Fast and simple
            - Assumes no outliers
            - Good for clean, high-quality data
            
            **Robust Methods (Outlier-Resistant)**
            - *Soft L1*: Balanced approach, recommended for most cases
            - *Huber*: Moderate outlier resistance
            - *Tukey Biweight*: Strong outlier rejection
            - Best for: Noisy data, photobleaching artifacts
            
            **Bayesian MCMC**
            - Full uncertainty quantification
            - Provides credible intervals (95% CI)
            - Computationally intensive (~30-60 sec per curve)
            - Best for: Critical measurements, publication-quality analysis
            
            📊 **When to use each:**
            - Standard → Clean data, quick analysis
            - Robust → Typical experiments with noise
            - Bayesian → Final analysis, rigorous uncertainty estimates
            """)

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
