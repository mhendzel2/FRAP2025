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
from frap_core import FRAPAnalysisCore as CoreFRAPAnalysis

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
- Diffusion coefficient: D = (w² × k) / (4 × ln(2))
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
class FRAPAnalysisCore_Deprecated:
    @staticmethod
    def get_post_bleach_data(time, intensity):
        i_min = np.argmin(intensity)
        if i_min == 0 or i_min >= len(time): return time - time[0], intensity, i_min
        t_pre, t_bleach = time[i_min - 1], time[i_min]
        t_mid = (t_pre + t_bleach) / 2.0
        slope = (intensity[i_min + 1] - intensity[i_min]) / (time[i_min + 1] - t_bleach) if i_min + 1 < len(time) else 0
        extrapolated_intensity = intensity[i_min] - slope * (t_bleach - t_mid)
        new_time = np.insert(time[i_min:], 0, t_mid)
        new_intensity = np.insert(intensity[i_min:], 0, extrapolated_intensity)
        return new_time - t_mid, new_intensity, i_min

    @staticmethod
    def load_data(file_path):
        """
        Load data from various file formats (CSV, XLS, XLSX) with improved column mapping.
        """
        try:
            # Extract the original filename from the path that may include hash suffixes
            logger.info(f"Determining file type for: {file_path}")
            
            file_type = None
            if '.xls_' in file_path:
                file_type = 'xls'
            elif '.xlsx_' in file_path:
                file_type = 'xlsx'
            elif '.csv_' in file_path:
                file_type = 'csv'
            elif file_path.lower().endswith('.xls'):
                file_type = 'xls'
            elif file_path.lower().endswith('.xlsx'):
                file_type = 'xlsx'
            elif file_path.lower().endswith('.csv'):
                file_type = 'csv'
            else:
                raise ValueError(f"Unsupported file format for {file_path}. Only .xls, .xlsx, and .csv are supported.")
            
            logger.info(f"Detected file type: {file_type} for {file_path}")
            
            # Load the data based on file type
            df = None
            if file_type == 'csv':
                # Handle CSV files with multiple encoding attempts
                for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise ValueError(f"Could not decode CSV file {file_path} with any standard encoding")
            elif file_type in ['xls', 'xlsx']:
                # Handle Excel files with appropriate engines
                engine = 'xlrd' if file_type == 'xls' else 'openpyxl'
                try:
                    df = pd.read_excel(file_path, engine=engine)
                    logger.info(f"Successfully loaded {file_path} with {engine}")
                except Exception as e1:
                    if file_type == 'xls':
                        logger.warning(f"Standard Excel reading failed for {file_path}: {e1}")
                        try:
                            # Fallback: Direct xlrd parsing
                            import xlrd
                            workbook = xlrd.open_workbook(file_path, encoding_override='cp1252')
                            sheet = workbook.sheet_by_index(0)
                            
                            # Extract all data
                            data = []
                            for row_idx in range(sheet.nrows):
                                row_data = []
                                for col_idx in range(sheet.ncols):
                                    cell_value = sheet.cell_value(row_idx, col_idx)
                                    row_data.append(cell_value)
                                data.append(row_data)
                            
                            # Create DataFrame with first row as headers
                            if len(data) > 1:
                                df = pd.DataFrame(data[1:], columns=data[0])
                            else:
                                df = pd.DataFrame(data)
                            logger.info(f"Successfully loaded {file_path} with xlrd fallback")
                        except Exception as e2:
                            logger.error(f"Both Excel reading methods failed for {file_path}: {e2}")
                            raise e2
                    else:
                        raise e1
            
            if df is None:
                raise ValueError(f"Failed to load data from {file_path}")
            
            # Improved robust column mapping strategy
            column_mapping = {
                'time': ['time [s]', 'time', 'time(s)', 't', 'seconds', 'sec'],
                'ROI1': ['intensity region 1', 'roi1', 'bleach', 'frap', 'intensity1', 'signal', 'region 1', 'region1'],
                'ROI2': ['intensity region 2', 'roi2', 'control', 'reference', 'intensity2', 'region 2', 'region2'], 
                'ROI3': ['intensity region 3', 'roi3', 'background', 'bg', 'intensity3', 'region 3', 'region3']
            }

            standardized_df = pd.DataFrame()
            missing_columns = []
            
            for standard_name, potential_names in column_mapping.items():
                found = False
                for col in df.columns:
                    if any(potential_name in str(col).lower() for potential_name in potential_names):
                        standardized_df[standard_name] = df[col]
                        logger.info(f"Mapped '{col}' to '{standard_name}'")
                        found = True
                        break
                if not found:
                    missing_columns.append(standard_name)
            
            # If we have missing critical columns, try positional assignment as fallback
            if missing_columns and len(df.columns) >= 4:
                logger.warning(f"Missing columns {missing_columns}, attempting positional assignment")
                col_names = ['time', 'ROI1', 'ROI2', 'ROI3']
                for i, col_name in enumerate(col_names):
                    if col_name not in standardized_df.columns and i < len(df.columns):
                        standardized_df[col_name] = df.iloc[:, i]
                        logger.info(f"Positionally assigned column {i} to '{col_name}'")
            
            # Validate we have all required columns
            required_cols = ['time', 'ROI1', 'ROI2', 'ROI3']
            final_missing = [col for col in required_cols if col not in standardized_df.columns]
            if final_missing:
                raise ValueError(f"Could not find required columns {final_missing} in {file_path}")
            
            logger.info(f"Successfully mapped columns: {list(standardized_df.columns)}")
            
            # Validate the loaded data
            if not validate_frap_data(standardized_df, file_path):
                raise ValueError(f"Data validation failed for {file_path}")
            
            return standardized_df
                
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            raise

    @staticmethod
    def preprocess(df):
        df['ROI1_corr'] = df['ROI1'] - df['ROI3']
        df['ROI2_corr'] = df['ROI2'] - df['ROI3']
        pre_bleach_roi1 = df[df['time'] < 0]['ROI1_corr'].mean()
        pre_bleach_roi2 = df[df['time'] < 0]['ROI2_corr'].mean()
        if pd.isna(pre_bleach_roi1) or pd.isna(pre_bleach_roi2) or pre_bleach_roi2 == 0:
            pre_bleach_factor = df['ROI1_corr'].iloc[0] if not df.empty else 1.0
            df['normalized'] = df['ROI1_corr'] / (pre_bleach_factor if pre_bleach_factor != 0 else 1.0)
        else:
            df['normalized'] = (df['ROI1_corr'] / pre_bleach_roi1) / (df['ROI2_corr'] / pre_bleach_roi2)
        df['normalized'] = df['normalized'].fillna(method='ffill')
        return df

    @staticmethod
    def single_component(t,A,k,C): return A*(1-np.exp(-k*t))+C
    @staticmethod
    def two_component(t,A1,k1,A2,k2,C): return A1*(1-np.exp(-k1*t))+A2*(1-np.exp(-k2*t))+C
    @staticmethod
    def three_component(t,A1,k1,A2,k2,A3,k3,C): return A1*(1-np.exp(-k1*t))+A2*(1-np.exp(-k2*t))+A3*(1-np.exp(-k3*t))+C
    
    @staticmethod
    def compute_r_squared(y,y_fit): 
        ss_res, ss_tot = np.sum((y-y_fit)**2), np.sum((y-np.mean(y))**2)
        return 1-ss_res/ss_tot if ss_tot!=0 else np.nan
    
    @staticmethod
    def compute_aic(rss,n,n_params): 
        return 2*n_params+n*log(rss/n) if rss>0 else np.nan

    @staticmethod
    def fit_all_models(time,intensity):
        t_fit, intensity_fit, _ = CoreFRAPAnalysis.get_post_bleach_data(time, intensity)
        fits, n = [], len(t_fit)
        if n < 3: return fits
        A0, C0, k0 = max(0.1, np.max(intensity_fit)-intensity_fit[0]), intensity_fit[0], 0.1
        models = {
            'single':(CoreFRAPAnalysis.single_component,[A0,k0,C0],([0,1e-6,-np.inf],[np.inf,np.inf,np.inf])),
            'double':(CoreFRAPAnalysis.two_component,[A0/2,k0*2,A0/2,k0/2,C0],([0,1e-6,0,1e-6,-np.inf],[np.inf,np.inf,np.inf,np.inf,np.inf])),
            'triple':(CoreFRAPAnalysis.three_component,[A0/3,k0*3,A0/3,k0,A0/3,k0/3,C0],([0,1e-6,0,1e-6,0,1e-6,-np.inf],[np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf]))
        }
        for name,(func,p0,bounds) in models.items():
            try:
                popt,_ = curve_fit(func, t_fit, intensity_fit, p0=p0, bounds=bounds, maxfev=5000)
                fitted = func(t_fit, *popt)
                rss = np.sum((intensity_fit-fitted)**2)
                fits.append({
                    'model':name,'params':popt,'fitted_values':fitted,'rss':rss,'n_params':len(popt),
                    'r2':CoreFRAPAnalysis.compute_r_squared(intensity_fit,fitted),
                    'aic':CoreFRAPAnalysis.compute_aic(rss,n,len(popt))
                })
            except Exception as e: 
                logger.error(f"{name}-fit failed: {e}")
        return fits

    @staticmethod
    def select_best_fit(fits,criterion='aic'): 
        return min(fits,key=lambda x:x.get(criterion,float('inf'))) if fits else None

    @staticmethod
    def extract_kinetic_parameters(best_fit):
        if not best_fit: return None
        model, params = best_fit['model'], best_fit['params']
        features = {'model':model,'r2':best_fit['r2'],'aic':best_fit['aic']}
        C, amplitudes, rates = params[-1], params[:-1:2], params[1:-1:2]
        total_amplitude = sum(amplitudes)
        plateau = total_amplitude + C
        features.update({
            'mobile_fraction':total_amplitude*100,
            'immobile_fraction':max(0,1-plateau)*100,
            'offset':C
        })
        sorted_components = sorted(zip(rates,amplitudes),reverse=True)
        comp_names = ['fast','medium','slow']
        safe_total_amplitude = total_amplitude if total_amplitude > 1e-9 else 1.0
        for i, (k, A) in enumerate(sorted_components):
            name = comp_names[i]
            features[f'proportion_of_total_{name}'] = A * 100
            features[f'proportion_of_mobile_{name}'] = (A / safe_total_amplitude) * 100
            features[f'rate_constant_{name}'] = k
            features[f'half_time_{name}'] = np.log(2)/k if k>0 else np.nan
        return features

    @staticmethod
    def identify_outliers(features_df,columns,iqr_multiplier=1.5):
        if features_df is None or features_df.empty: return []
        outlier_indices = set()
        for col in columns:
            if col in features_df.columns and pd.api.types.is_numeric_dtype(features_df[col]):
                Q1,Q3 = features_df[col].quantile(0.25),features_df[col].quantile(0.75)
                IQR=Q3-Q1
                if IQR > 0:
                    lower,upper = Q1-iqr_multiplier*IQR,Q3+iqr_multiplier*IQR
                    outliers = features_df[(features_df[col]<lower)|(features_df[col]>upper)]
                    outlier_indices.update(outliers.index)
        return features_df.loc[list(outlier_indices),'file_path'].tolist() if 'file_path' in features_df.columns else []

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
                params = CoreFRAPAnalysis.compute_kinetic_details(best_fit)
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

# --- Streamlit UI Application ---
st.title("🔬 FRAP Analysis Application")
st.markdown("**Fluorescence Recovery After Photobleaching with Supervised Outlier Removal**")
dm = st.session_state.data_manager = FRAPDataManager() if st.session_state.data_manager is None else st.session_state.data_manager

with st.sidebar:
    st.header("Data Management")
    uploaded_files=st.file_uploader("Upload FRAP files",type=['xls','xlsx','csv'],accept_multiple_files=True)
    if uploaded_files:
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
                        key="ungrouped_files_selector"
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
                        key="group_files_selector"
                    )
                else:
                    st.info("No files in this group yet.")
                    files_to_remove = []
            
            # Action buttons
            button_col1, button_col2 = st.columns(2)
            with button_col1:
                if st.button("Add Selected Files", disabled=not selected_files):
                    group['files'].extend(selected_files)
                    dm.update_group_analysis(selected_group_name)
                    st.success(f"Added {len(selected_files)} files to {selected_group_name}")
                    st.rerun()
            
            with button_col2:
                if st.button("Remove Selected Files", disabled=not files_to_remove):
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
                    
                    # Diffusion interpretation: D = (r² × k) / (4 × ln(2))
                    diffusion_coeff=(effective_radius_um**2*primary_rate)/(4.0*np.log(2))
                    
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
                opts=[c for c in features_df.select_dtypes(include=np.number).columns if 'fraction' in c or 'rate' in c]
                defaults=[c for c in ['mobile_fraction','immobile_fraction'] if c in opts]
                outlier_check_features=st.multiselect("Check for outliers based on:",options=opts,default=defaults)
                iqr_multiplier=st.slider("Outlier Sensitivity",1.0,3.0,1.5,0.1,help="Lower value = more sensitive.")
                identified=CoreFRAPAnalysis.identify_outliers(features_df,outlier_check_features,iqr_multiplier)
                excluded_paths=st.multiselect("Select files to EXCLUDE (outliers are pre-selected):",options=group['files'],default=identified,format_func=lambda p:dm.files[p]['name'])
            
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
                st.markdown("### Step 6: Group Recovery Plots")
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
