"""
FRAP Analysis Core Module - CORRECTED VERSION
Contains core computational routines for FRAP analysis with verified mathematical formulas
"""
import numpy as np
import pandas as pd
import os
import logging
from scipy.optimize import curve_fit
from scipy.stats import linregress
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from calibration import Calibration
# ---------------------------------------------------------------------- #
#  PUBLIC HELPER – post‑bleach extraction with recovery extrapolation    #
# ---------------------------------------------------------------------- #
def get_post_bleach_data(time: np.ndarray,
                         intensity: np.ndarray,
                         *,
                         extrapolation_points: int = 3
                         ) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Return **t_post**, **i_post** and the index of the first bleach frame (i_min).

    • The recovery curve is extrapolated back to the photobleach point using the
      initial recovery trajectory, ensuring proper fitting from the true bleach event.
    • Uses linear extrapolation of early recovery points back to bleach time.

    Parameters
    ----------
    time, intensity : 1‑D arrays (already normalised)
    extrapolation_points : int – number of early recovery points to use for extrapolation

    """
    # locate bleach – minimum intensity
    i_min = int(np.argmin(intensity))
    if i_min == 0:                           # guard pathological files
        raise ValueError("Bleach frame at index 0 – cannot segment pre/post.")
    
    if i_min >= len(time) - 2:  # guard end-of-data bleach
        raise ValueError("Bleach frame too close to end of data – insufficient recovery data.")

    # Get the bleach time point (time of minimum intensity)
    t_bleach = time[i_min]
    
    # Use early recovery points for extrapolation (skip the bleach point itself)
    start_idx = i_min + 1
    end_idx = min(i_min + 1 + extrapolation_points, len(time))
    
    if end_idx - start_idx < 2:
        raise ValueError("Insufficient recovery points for extrapolation.")
    
    # Extrapolate recovery trajectory back to bleach time
    t_recovery = time[start_idx:end_idx]
    i_recovery = intensity[start_idx:end_idx]
    
    # Linear regression to find initial recovery trajectory
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(t_recovery, i_recovery)
    
    # Calculate extrapolated intensity at bleach time
    i_bleach_extrapolated = slope * t_bleach + intercept
    
    # Ensure extrapolated value is reasonable (should be close to measured bleach intensity)
    i_bleach_measured = intensity[i_min]
    
    # Use the lower of measured and extrapolated values (more conservative)
    i_bleach_final = min(i_bleach_extrapolated, i_bleach_measured)
    
    # Build corrected vectors starting from extrapolated bleach point
    # The time vector should not include the original bleach timepoint again, only subsequent ones.
    t_post = np.concatenate([[t_bleach], time[i_min+1:]])
    i_post = np.concatenate([[i_bleach_final], intensity[i_min+1:]])
    
    # Reset time to start from zero at bleach event
    t_post = t_post - t_bleach
    
    return t_post, i_post, i_min

# ----------------------------- DIFFUSION ------------------------------- #
def diffusion_coefficient(bleach_radius_um: float, k: float) -> float:
    """
    Corrected 2‑D diffusion coefficient:

        D = (w² × k) / 4

    where **w** is the bleached‑spot radius in µm.  No ln(2) factor.
    """
    return (bleach_radius_um**2 * k) / 4.0

def interpret_kinetics(k: float,
                       bleach_radius_um: float,
                       calibration: Calibration = None,
                       gfp_d: float = 25.0,
                       gfp_mw: float = 27.0) -> dict[str, float]:
    """
    Dual interpretation (diffusion / binding) using the *correct* D formula.
    """
    if k <= 0 or bleach_radius_um <= 0:
        return {key: np.nan for key in
                ("k_off", "diffusion_coefficient", "apparent_mw", "apparent_mw_ci_low", "apparent_mw_ci_high",
                 "half_time_diffusion", "half_time_binding")}

    D = diffusion_coefficient(bleach_radius_um, k)

    apparent_mw, ci_low, ci_high = np.nan, np.nan, np.nan
    if calibration:
        apparent_mw, ci_low, ci_high = calibration.estimate_apparent_mw(D)
    else:
        # Fallback to old method if no calibration object is provided
        if D > 0:
            apparent_mw = gfp_mw * (gfp_d / D)**3

    return {
        "k_off": k,
        "diffusion_coefficient": D,
        "apparent_mw": apparent_mw,
        "apparent_mw_ci_low": ci_low,
        "apparent_mw_ci_high": ci_high,
        "half_time_diffusion": np.log(2) / k,
        "half_time_binding": np.log(2) / k,
    }

# Handle optional imports with fallbacks
try:
    import xlrd
    XLRD_AVAILABLE = True
except ImportError:
    XLRD_AVAILABLE = False
    logging.warning("xlrd not available - .xls file support may be limited")

try:
    import openpyxl
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False
    logging.warning("openpyxl not available - .xlsx file support may be limited")

# Try to import advanced fitting module
try:
    from frap_advanced_fitting import (
        fit_all_advanced_models, 
        select_best_advanced_model,
        LMFIT_AVAILABLE
    )
    ADVANCED_FITTING_AVAILABLE = LMFIT_AVAILABLE
except ImportError:
    ADVANCED_FITTING_AVAILABLE = False
    logging.warning("Advanced fitting module not available - install lmfit for advanced models")

class FRAPAnalysisCore:
    @staticmethod
    def get_post_bleach_data(time, intensity):
        """
        Return post-bleach trace with recovery curve extrapolation back to bleach point.
        
        This method implements proper extrapolation of the early recovery trajectory
        back to the photobleach event to ensure accurate fitting from the true
        starting point of recovery.
        
        Parameters:
        -----------
        time : numpy.ndarray
            Time points
        intensity : numpy.ndarray
            Intensity values
            
        Returns:
        --------
        tuple
            (post_bleach_time, post_bleach_intensity, bleach_index)
        """
        if len(time) != len(intensity):
            raise ValueError("Time and intensity arrays must have the same length")
            
        # Find the bleach point (minimum intensity)
        i_min = np.argmin(intensity)
        
        if i_min == 0:
            raise ValueError("Bleach frame found at first index – cannot extrapolate.")
        
        if i_min >= len(time) - 3:
            raise ValueError("Bleach frame too close to end – insufficient recovery data.")
        
        # Get bleach time
        t_bleach = time[i_min]
        i_bleach_measured = intensity[i_min]
        
        # --- Recovery trajectory extrapolation ---
        # Use the next 3-5 recovery points to extrapolate back to bleach time
        recovery_start = i_min + 1
        recovery_end = min(int(i_min) + 5, len(time))  # Use up to 4 recovery points
        
        if recovery_end - recovery_start < 2:
            raise ValueError("Insufficient recovery points for trajectory extrapolation.")
        
        # Extract early recovery data
        t_recovery = time[recovery_start:recovery_end]
        i_recovery = intensity[recovery_start:recovery_end]
        
        # Linear extrapolation of initial recovery slope
        if len(t_recovery) >= 2:
            # Use linear regression for robust slope estimation
            coeffs = np.polyfit(t_recovery, i_recovery, 1)
            slope, intercept = coeffs[0], coeffs[1]
            
            # Extrapolate intensity at bleach time
            i_bleach_extrapolated = slope * t_bleach + intercept
            
            # Use the minimum of measured and extrapolated (more conservative)
            # This accounts for potential noise in the measured bleach point
            i_bleach_final = min(i_bleach_extrapolated, i_bleach_measured)
            
            # Ensure the extrapolated value is physically reasonable
            if i_bleach_final < 0:
                i_bleach_final = max(0, i_bleach_measured * 0.8)  # Fallback to 80% of measured
                
        else:
            # Fallback: use measured bleach intensity
            i_bleach_final = i_bleach_measured
        
        # --- Build corrected post-bleach vectors ---
        # Start from the extrapolated bleach point
        t_post = np.concatenate([[t_bleach], time[i_min+1:]])
        i_post = np.concatenate([[i_bleach_final], intensity[i_min+1:]])
        
        # Reset time scale to start from zero at bleach event
        t_post = t_post - t_bleach
        
        return t_post, i_post, i_min

    @staticmethod
    def align_and_interpolate_curves(list_of_curves: list, num_points: int = 200) -> dict:
        """
        Aligns multiple FRAP curves to a common time axis (t=0 at bleach) and
        interpolates them for direct comparison.
        
        This function properly handles different sampling rates between experiments
        by time-shifting each curve to start at the bleach event and interpolating
        all curves onto a uniform time grid.

        Parameters
        ----------
        list_of_curves : list[dict]
            A list of dictionaries, where each dict represents a curve and has
            'time', 'intensity', and 'name' keys.
        num_points : int
            The number of points for the common interpolated time axis. Default: 200

        Returns
        -------
        dict
            A dictionary containing:
            - 'common_time': np.ndarray - The common time axis starting at t=0
            - 'interpolated_curves': list[dict] - List of aligned curves with 'name' and 'intensity'
        """
        aligned_data = []
        max_time = 0

        # First pass: align each curve to t=0 and find the max recovery time
        for curve in list_of_curves:
            try:
                # Use the existing function to get time-shifted post-bleach data
                t_post, i_post, _ = FRAPAnalysisCore.get_post_bleach_data(
                    curve['time'], curve['intensity']
                )
                aligned_data.append({
                    'name': curve['name'], 
                    'time': t_post, 
                    'intensity': i_post
                })
                if len(t_post) > 0 and t_post[-1] > max_time:
                    max_time = t_post[-1]
            except (ValueError, IndexError) as e:
                # Skip curves that fail pre-processing (e.g., bleach at frame 0)
                logging.warning(f"Skipping curve '{curve.get('name', 'unknown')}' during alignment: {e}")
                continue

        if not aligned_data:
            logging.warning("No valid curves to align - all curves failed preprocessing")
            return {'common_time': np.array([]), 'interpolated_curves': []}

        # Create a high-resolution common time axis
        common_time_axis = np.linspace(0, max_time, num_points)
        interpolated_curves = []

        # Second pass: interpolate each aligned curve onto the common time axis
        for data in aligned_data:
            # Use numpy's interpolation function
            # Handle edge cases by extending the last value for times beyond the data
            interp_intensity = np.interp(
                common_time_axis, 
                data['time'], 
                data['intensity'], 
                right=data['intensity'][-1]
            )
            interpolated_curves.append({
                'name': data['name'],
                'intensity': interp_intensity
            })

        return {
            'common_time': common_time_axis,
            'interpolated_curves': interpolated_curves
        }

    @staticmethod
    def motion_compensate_stack(stack: np.ndarray,
                                init_center: tuple[float, float],
                                radius: float,
                                *,
                                pixel_size_um: float | None = None,
                                use_optical_flow: bool = True,
                                do_global: bool = True,
                                window_radius: int = 24,
                                flow_window: int = 32,
                                kalman: bool = True) -> dict:
        """
        Run motion compensation on a grayscale image stack.

        Parameters
        ----------
        stack : np.ndarray
            Input image stack shaped (T, H, W), single channel.
        init_center : tuple[float, float]
            Initial bleach-spot center (x, y) in pixels.
        radius : float
            ROI radius in pixels to apply around the tracked center.
        pixel_size_um : float | None
            Pixel size in micrometers. If provided, returns drift in micrometers.
        use_optical_flow : bool
            Use Lucas–Kanade optical flow to help local prediction.
        do_global : bool
            Apply global translation registration before local recentering.
        window_radius : int
            Half-size for Gaussian fitting window.
        flow_window : int
            Half-size for optical flow window.
        kalman : bool
            Apply constant-velocity Kalman smoothing to trajectory.

        Returns
        -------
        dict
            {
              'stabilized_stack': np.ndarray,
              'roi_trace': list[{'frame', 'centroid': {'x','y'}, 'applied_radius', 'displacement_px'}],
              'drift_um': float | None,
              'warnings': list[str],
              'details': {'global': {...} | None, 'tracking': {...}}
            }
        """
        # Local import to keep optional heavy deps isolated
        from image_motion import stabilize_roi

        return stabilize_roi(
            stack=stack,
            init_center=init_center,
            radius=radius,
            pixel_size_um=pixel_size_um,
            use_optical_flow=use_optical_flow,
            window_radius=window_radius,
            flow_window=flow_window,
            kalman=kalman,
            do_global=do_global,
        )

    @staticmethod
    def load_data(file_path):
        """
        Load data from various file formats (CSV, XLS, XLSX)
        
        Parameters:
        -----------
        file_path : str
            Path to the data file
            
        Returns:
        --------
        pandas.DataFrame
            Loaded data with standardized column names
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_extension == '.csv':
                df = pd.read_csv(file_path)
            elif file_extension == '.xls':
                if not XLRD_AVAILABLE:
                    raise ImportError("The 'xlrd' package is required to read .xls files. Please install it using 'pip install xlrd'.")
                df = pd.read_excel(file_path, engine='xlrd')
            elif file_extension == '.xlsx':
                if not OPENPYXL_AVAILABLE:
                    raise ImportError("The 'openpyxl' package is required to read .xlsx files. Please install it using 'pip install openpyxl'.")
                df = pd.read_excel(file_path, engine='openpyxl')
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
        except Exception as e:
            logging.error(f"Error loading file {file_path}: {e}")
            raise

        # Improved robust column mapping strategy
        column_mapping = {
            'time': ['time [s]', 'time', 'time(s)', 't', 'seconds', 'sec'],
            'roi1': ['intensity region 1', 'roi1', 'bleach', 'frap', 'intensity1', 'signal', 'region 1', 'region1'],
            'roi2': ['intensity region 2', 'roi2', 'control', 'reference', 'intensity2', 'region 2', 'region2'], 
            'roi3': ['intensity region 3', 'roi3', 'background', 'bg', 'intensity3', 'region 3', 'region3']
        }

        # Add optional stabilization columns to the mapping
        optional_columns = {
            'roi_centroid_x': ['roi_centroid_x'],
            'roi_centroid_y': ['roi_centroid_y'],
            'roi_radius_per_frame': ['roi_radius_per_frame'],
            'total_drift_um': ['total_drift_um'],
            'mean_framewise_shift_um': ['mean_framewise_shift_um'],
            'motion_qc_flag': ['motion_qc_flag'],
            'motion_qc_reason': ['motion_qc_reason']
        }

        standardized_df = pd.DataFrame()
        missing_columns = []
        
        # Process required columns first
        for standard_name, potential_names in column_mapping.items():
            found = False
            for col in df.columns:
                if any(potential_name in str(col).lower() for potential_name in potential_names):
                    standardized_df[standard_name] = df[col]
                    logging.info(f"Mapped '{col}' to '{standard_name}'")
                    found = True
                    break
            if not found:
                missing_columns.append(standard_name)
        
        # Process optional stabilization columns
        for standard_name, potential_names in optional_columns.items():
            for col in df.columns:
                if any(potential_name in str(col).lower() for potential_name in potential_names):
                    standardized_df[standard_name] = df[col]
                    logging.info(f"Mapped optional column '{col}' to '{standard_name}'")
                    break

        # If we have missing critical columns, try positional assignment as fallback
        if missing_columns and len(df.columns) >= 4:
            logging.warning(f"Missing columns {missing_columns}, attempting positional assignment")
            col_names = ['time', 'roi1', 'roi2', 'roi3']
            for i, col_name in enumerate(col_names):
                if col_name not in standardized_df.columns and i < len(df.columns):
                    standardized_df[col_name] = df.iloc[:, i]
                    logging.info(f"Positionally assigned column {i} to '{col_name}'")
        
        # Validate we have all required columns
        required_cols = ['time', 'roi1', 'roi2', 'roi3']
        final_missing = [col for col in required_cols if col not in standardized_df.columns]
        if final_missing:
            raise ValueError(f"Could not find required columns {final_missing} in {file_path}")
        
        logging.info(f"Successfully mapped columns: {list(standardized_df.columns)}")
        return standardized_df

    @staticmethod
    def preprocess(df):
        """
        Preprocess the data by applying background correction and normalization
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Raw data with time, roi1, roi2, roi3 columns
            
        Returns:
        --------
        pandas.DataFrame
            Processed dataframe with additional columns
        """
        try:
            # Make a copy to avoid modifying the original
            result_df = df.copy()
            
            # Validate input data
            required_cols = ['time', 'roi1', 'roi2', 'roi3']
            missing_cols = [col for col in required_cols if col not in result_df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Check for sufficient data points
            if len(result_df) < 3:
                raise ValueError("Insufficient data points for preprocessing (need at least 3)")
            
            # Background correction with error handling
            result_df['roi1_bg_corrected'] = result_df['roi1'] - result_df['roi3']
            result_df['roi2_bg_corrected'] = result_df['roi2'] - result_df['roi3']
            
            # Find pre-bleach values (average of first few points)
            pre_bleach_points = min(5, max(1, len(df) // 4))  # Ensure at least 1 point
            pre_bleach_roi1 = result_df['roi1_bg_corrected'][:pre_bleach_points].mean()
            pre_bleach_roi2 = result_df['roi2_bg_corrected'][:pre_bleach_points].mean()
            
            # Handle edge cases for normalization
            if pd.isna(pre_bleach_roi1) or pre_bleach_roi1 == 0:
                logging.warning("Invalid pre-bleach ROI1 value, using fallback normalization")
                pre_bleach_roi1 = result_df['roi1_bg_corrected'].iloc[0] if len(result_df) > 0 else 1.0
                if pre_bleach_roi1 == 0:
                    pre_bleach_roi1 = 1.0
            
            if pd.isna(pre_bleach_roi2) or pre_bleach_roi2 == 0:
                logging.warning("Invalid pre-bleach ROI2 value, using fallback normalization")
                pre_bleach_roi2 = result_df['roi2_bg_corrected'].iloc[0] if len(result_df) > 0 else 1.0
                if pre_bleach_roi2 == 0:
                    pre_bleach_roi2 = 1.0
            
            # Calculate reference-corrected values with error handling
            roi2_normalized = result_df['roi2_bg_corrected'] / pre_bleach_roi2
            
            # Avoid division by zero in double normalization
            safe_roi2_normalized = roi2_normalized.replace(0, np.nan)
            result_df['double_normalized'] = result_df['roi1_bg_corrected'] / safe_roi2_normalized / pre_bleach_roi1
            
            # Simple normalization (without reference correction)
            result_df['normalized'] = result_df['roi1_bg_corrected'] / pre_bleach_roi1
            
            # Fill NaN values using forward fill (modern pandas syntax)
            result_df['normalized'] = result_df['normalized'].ffill()
            result_df['double_normalized'] = result_df['double_normalized'].ffill()
            
            # Final validation
            if result_df['normalized'].isna().all():
                raise ValueError("Normalization failed - all values are NaN")
            
            logging.info("Data preprocessing completed successfully")
            return result_df
            
        except Exception as e:
            logging.error(f"Error in preprocessing: {e}")
            raise

    @staticmethod
    def single_component(t, A, k, C):
        """
        Single component exponential recovery model
        
        Parameters:
        -----------
        t : numpy.ndarray
            Time points
        A : float
            Amplitude
        k : float
            Rate constant
        C : float
            Offset
            
        Returns:
        --------
        numpy.ndarray
            Model values at each time point
        """
        return A * (1 - np.exp(-k * t)) + C

    @staticmethod
    def two_component(t, A1, k1, A2, k2, C):
        """
        Two component exponential recovery model
        
        Parameters:
        -----------
        t : numpy.ndarray
            Time points
        A1, A2 : float
            Amplitudes for each component
        k1, k2 : float
            Rate constants for each component
        C : float
            Offset
            
        Returns:
        --------
        numpy.ndarray
            Model values at each time point
        """
        return A1 * (1 - np.exp(-k1 * t)) + A2 * (1 - np.exp(-k2 * t)) + C

    @staticmethod
    def three_component(t, A1, k1, A2, k2, A3, k3, C):
        """
        Three component exponential recovery model
        
        Parameters:
        -----------
        t : numpy.ndarray
            Time points
        A1, A2, A3 : float
            Amplitudes for each component
        k1, k2, k3 : float
            Rate constants for each component
        C : float
            Offset
            
        Returns:
        --------
        numpy.ndarray
            Model values at each time point
        """
        return A1 * (1 - np.exp(-k1 * t)) + A2 * (1 - np.exp(-k2 * t)) + A3 * (1 - np.exp(-k3 * t)) + C

    @staticmethod
    def compute_r_squared(y, y_fit):
        """
        Calculate R-squared value
        
        Parameters:
        -----------
        y : numpy.ndarray
            Observed values
        y_fit : numpy.ndarray
            Predicted values
            
        Returns:
        --------
        float
            R-squared value
        """
        ss_total = np.sum((y - np.mean(y))**2)
        ss_residual = np.sum((y - y_fit)**2)
        
        if ss_total == 0:
            return 0  # Avoid division by zero
            
        return 1 - (ss_residual / ss_total)

    @staticmethod
    def compute_adjusted_r_squared(y, y_fit, n_params):
        """
        Calculate adjusted R-squared value
        
        Parameters:
        -----------
        y : numpy.ndarray
            Observed values
        y_fit : numpy.ndarray
            Predicted values
        n_params : int
            Number of parameters in the model
            
        Returns:
        --------
        float
            Adjusted R-squared value
        """
        n = len(y)
        if n <= n_params + 1:
            return np.nan  # Not enough data points
            
        r2 = FRAPAnalysisCore.compute_r_squared(y, y_fit)
        return 1 - (1 - r2) * (n - 1) / (n - n_params - 1)

    @staticmethod
    def compute_aic(rss, n, n_params):
        """
        Calculate Akaike Information Criterion (AIC)
        
        Parameters:
        -----------
        rss : float
            Residual sum of squares
        n : int
            Number of data points
        n_params : int
            Number of parameters in the model
            
        Returns:
        --------
        float
            AIC value
        """
        if n <= n_params + 2:
            return np.inf  # Not enough data points
            
        return n * np.log(rss / n) + 2 * n_params

    @staticmethod
    def compute_bic(rss, n, n_params):
        """
        Calculate Bayesian Information Criterion (BIC)
        
        Parameters:
        -----------
        rss : float
            Residual sum of squares
        n : int
            Number of data points
        n_params : int
            Number of parameters in the model
            
        Returns:
        --------
        float
            BIC value
        """
        if n <= n_params + 2:
            return np.inf  # Not enough data points
            
        return n * np.log(rss / n) + n_params * np.log(n)

    @staticmethod
    def compute_reduced_chi_squared(y, y_fit, n_params, error_variance=1):
        """
        Calculate reduced chi-squared statistic
        
        Parameters:
        -----------
        y : numpy.ndarray
            Observed values
        y_fit : numpy.ndarray
            Predicted values
        n_params : int
            Number of parameters in the model
        error_variance : float
            Variance of the error
            
        Returns:
        --------
        float
            Reduced chi-squared value
        """
        n = len(y)
        if n <= n_params:
            return np.nan  # Not enough data points
            
        residuals = y - y_fit
        chi_squared = np.sum((residuals)**2 / error_variance)
        
        return chi_squared / (n - n_params)

    @staticmethod
    def fit_all_models(time, intensity):
        """
        Fit single, two, and three component models to the FRAP data with robust error handling
        
        Parameters:
        -----------
        time : numpy.ndarray
            Time points
        intensity : numpy.ndarray
            Intensity values
            
        Returns:
        --------
        list
            List of dictionaries containing model fitting results
        """
        try:
            # Input validation
            if len(time) != len(intensity):
                raise ValueError("Time and intensity arrays must have the same length")
            
            if len(time) < 5:
                raise ValueError("Insufficient data points for curve fitting (need at least 5)")
            
            # Check for NaN or infinite values
            if np.any(np.isnan(time)) or np.any(np.isnan(intensity)):
                raise ValueError("Input data contains NaN values")
            
            if np.any(np.isinf(time)) or np.any(np.isinf(intensity)):
                raise ValueError("Input data contains infinite values")
            
            t_fit, intensity_fit, _ = FRAPAnalysisCore.get_post_bleach_data(time, intensity)
            n = len(t_fit)
            fits = []
            
            # Validate post-bleach data
            if n < 3:
                raise ValueError("Insufficient post-bleach data points for fitting")
            
            # Initial guesses with robust calculation
            C0 = np.min(intensity_fit)
            A0 = np.max(intensity_fit) - np.min(intensity_fit)
            
            # Ensure positive amplitude
            if A0 <= 0:
                logging.warning("Non-positive amplitude detected, using fallback value")
                A0 = np.std(intensity_fit)
                if A0 <= 0:
                    A0 = 0.1
            
            # Calculate rate constant guess with error handling
            time_span = t_fit[-1] - t_fit[0]
            if time_span <= 0:
                raise ValueError("Invalid time span for rate constant calculation")
            
            k0 = -np.log(0.4) / (time_span / 3) if time_span > 0 else 0.1
            
            # Ensure reasonable bounds for k0
            k0 = max(1e-6, min(k0, 100.0))
            
            logging.info(f"Initial parameter estimates: A0={A0:.4f}, k0={k0:.4f}, C0={C0:.4f}")
            
            # Single Component Fit.
            try:
                p0 = [A0, k0, C0]
                bounds = ([0, 1e-6, -np.inf], [np.inf, np.inf, np.inf])
                popt, _ = curve_fit(FRAPAnalysisCore.single_component, t_fit, intensity_fit, p0=p0, bounds=bounds)
                fitted = FRAPAnalysisCore.single_component(t_fit, *popt)
                rss = np.sum((intensity_fit - fitted)**2)
                r2 = FRAPAnalysisCore.compute_r_squared(intensity_fit, fitted)
                adj_r2 = FRAPAnalysisCore.compute_adjusted_r_squared(intensity_fit, fitted, len(p0))
                aic = FRAPAnalysisCore.compute_aic(rss, n, len(p0))
                bic = FRAPAnalysisCore.compute_bic(rss, n, len(p0))
                red_chi2 = FRAPAnalysisCore.compute_reduced_chi_squared(intensity_fit, fitted, len(p0))
                fits.append({'model': 'single', 'func': FRAPAnalysisCore.single_component, 'params': popt,
                             'rss': rss, 'r2': r2, 'adj_r2': adj_r2, 'aic': aic, 'bic': bic,
                             'red_chi2': red_chi2, 'fitted_values': fitted})
            except Exception as e:
                logging.error(f"Single-component fit failed: {e}")
                
            # Two Component Fit.
            try:
                A1_0 = A0 / 2
                k1_0 = k0 * 2
                A2_0 = A0 / 2
                k2_0 = k0 / 2
                p0_double = [A1_0, k1_0, A2_0, k2_0, C0]
                bounds_double = ([0, 1e-6, 0, 1e-6, -np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf])
                popt, _ = curve_fit(FRAPAnalysisCore.two_component, t_fit, intensity_fit, p0=p0_double, bounds=bounds_double)
                fitted = FRAPAnalysisCore.two_component(t_fit, *popt)
                rss = np.sum((intensity_fit - fitted)**2)
                r2 = FRAPAnalysisCore.compute_r_squared(intensity_fit, fitted)
                adj_r2 = FRAPAnalysisCore.compute_adjusted_r_squared(intensity_fit, fitted, len(p0_double))
                aic = FRAPAnalysisCore.compute_aic(rss, n, len(p0_double))
                bic = FRAPAnalysisCore.compute_bic(rss, n, len(p0_double))
                red_chi2 = FRAPAnalysisCore.compute_reduced_chi_squared(intensity_fit, fitted, len(p0_double))
                fits.append({'model': 'double', 'func': FRAPAnalysisCore.two_component, 'params': popt,
                             'rss': rss, 'r2': r2, 'adj_r2': adj_r2, 'aic': aic, 'bic': bic,
                             'red_chi2': red_chi2, 'fitted_values': fitted})
            except Exception as e:
                logging.error(f"Two-component fit failed: {e}")
                
            # Three Component Fit.
            try:
                A1_0 = A0 / 3
                k1_0 = k0 * 3
                A2_0 = A0 / 3
                k2_0 = k0
                A3_0 = A0 / 3
                k3_0 = k0 / 3
                p0_triple = [A1_0, k1_0, A2_0, k2_0, A3_0, k3_0, C0]
                bounds_triple = ([0, 1e-6, 0, 1e-6, 0, 1e-6, -np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
                popt, _ = curve_fit(FRAPAnalysisCore.three_component, t_fit, intensity_fit, p0=p0_triple, bounds=bounds_triple)
                fitted = FRAPAnalysisCore.three_component(t_fit, *popt)
                rss = np.sum((intensity_fit - fitted)**2)
                r2 = FRAPAnalysisCore.compute_r_squared(intensity_fit, fitted)
                adj_r2 = FRAPAnalysisCore.compute_adjusted_r_squared(intensity_fit, fitted, len(p0_triple))
                aic = FRAPAnalysisCore.compute_aic(rss, n, len(p0_triple))
                bic = FRAPAnalysisCore.compute_bic(rss, n, len(p0_triple))
                red_chi2 = FRAPAnalysisCore.compute_reduced_chi_squared(intensity_fit, fitted, len(p0_triple))
                fits.append({'model': 'triple', 'func': FRAPAnalysisCore.three_component, 'params': popt,
                             'rss': rss, 'r2': r2, 'adj_r2': adj_r2, 'aic': aic, 'bic': bic,
                             'red_chi2': red_chi2, 'fitted_values': fitted})
            except Exception as e:
                logging.error(f"Three-component fit failed: {e}")
                
            return fits
            
        except Exception as e:
            logging.error(f"Error in fit_all_models: {e}")
            return []

    @staticmethod
    def select_best_fit(fits, criterion='aic'):
        """
        Select the best fit model based on the specified criterion
        
        Parameters:
        -----------
        fits : list
            List of dictionaries containing model fitting results
        criterion : str
            Criterion for model selection ('aic', 'bic', 'adj_r2', 'r2', or 'red_chi2')
            
        Returns:
        --------
        dict or None
            Dictionary containing the best fit model information
        """
        if not fits:
            return None
            
        if criterion == 'aic':
            best = min(fits, key=lambda f: f['aic'] if not np.isnan(f['aic']) else np.inf)
        elif criterion == 'bic':
            best = min(fits, key=lambda f: f['bic'] if not np.isnan(f['bic']) else np.inf)
        elif criterion == 'adj_r2':
            best = max(fits, key=lambda f: f['adj_r2'] if not np.isnan(f['adj_r2']) else -np.inf)
        elif criterion == 'r2':
            best = max(fits, key=lambda f: f['r2'] if not np.isnan(f['r2']) else -np.inf)
        elif criterion == 'red_chi2':
            best = min(fits, key=lambda f: f['red_chi2'] if not np.isnan(f['red_chi2']) else np.inf)
        else:
            best = min(fits, key=lambda f: f['aic'] if not np.isnan(f['aic']) else np.inf)
            
        return best

    @staticmethod
    def compute_diffusion_coefficient(rate_constant, bleach_spot_radius=1.0):
        """
        Calculate diffusion coefficient from rate constant using CORRECTED formula
        
        Parameters:
        -----------
        rate_constant : float
            Rate constant from FRAP recovery
        bleach_spot_radius : float
            Radius of the bleach spot in μm
            
        Returns:
        --------
        float
            Diffusion coefficient in μm²/s
        """
        # CORRECTED FORMULA: D = (w² × k) / 4
        # This is the mathematically correct formula for 2D diffusion in FRAP
        return (bleach_spot_radius**2 * rate_constant) / 4.0
    
    @staticmethod
    def interpret_kinetics(k, bleach_radius_um, calibration=None, gfp_d=25.0, gfp_mw=27.0):
        """
        Centralized kinetics interpretation function with CORRECTED mathematics
        
        Parameters:
        -----------
        k : float
            The fitted kinetic rate constant (1/s)
        bleach_radius_um : float
            The radius of the photobleach spot in micrometers
        calibration : Calibration, optional
            A Calibration object for MW estimation.
        gfp_d : float
            Reference diffusion coefficient for GFP (um^2/s)
        gfp_mw : float
            Reference molecular weight for GFP (kDa)
            
        Returns:
        --------
        dict
            Dictionary containing both diffusion and binding interpretations
        """
        if k <= 0:
            return {
                'k_off': k,
                'diffusion_coefficient': np.nan,
                'apparent_mw': np.nan,
                'apparent_mw_ci_low': np.nan,
                'apparent_mw_ci_high': np.nan,
                'half_time_diffusion': np.nan,
                'half_time_binding': np.nan
            }

        # 1. Interpretation as a binding/unbinding process
        k_off = k  # The rate is the dissociation constant
        half_time_binding = np.log(2) / k  # Time for 50% recovery via binding

        # 2. Interpretation as a diffusion process
        # CORRECTED FORMULA: For 2D diffusion: D = (w^2 * k) / 4 
        # where w is bleach radius and k is rate constant
        diffusion_coefficient = (bleach_radius_um**2 * k) / 4.0
        half_time_diffusion = np.log(2) / k  # Half-time from rate constant

        # Estimate apparent molecular weight from diffusion coefficient
        apparent_mw, ci_low, ci_high = np.nan, np.nan, np.nan
        if calibration:
            apparent_mw, ci_low, ci_high = calibration.estimate_apparent_mw(diffusion_coefficient)
        else:
            # Fallback to old method if no calibration object is provided
            if diffusion_coefficient > 0:
                apparent_mw = gfp_mw * (gfp_d / diffusion_coefficient)**3

        return {
            'k_off': k_off,
            'diffusion_coefficient': diffusion_coefficient,
            'apparent_mw': apparent_mw,
            'apparent_mw_ci_low': ci_low,
            'apparent_mw_ci_high': ci_high,
            'half_time_diffusion': half_time_diffusion,
            'half_time_binding': half_time_binding
        }

    @staticmethod
    def extract_clustering_features(best_fit, bleach_spot_radius=1.0, pixel_size=1.0, calibration=None):
        """
        Extract features for clustering from the best fit model.
        """
        if best_fit is None:
            return None

        model = best_fit.get('model')
        params = best_fit.get('params')

        if not model or not isinstance(params, (list, tuple, np.ndarray)):
            return None

        features = {}
        effective_radius_um = bleach_spot_radius * pixel_size
        
        # --- Common feature extraction ---
        fitted_vals = best_fit.get('fitted_values')
        plateau_reached = True
        if isinstance(fitted_vals, (list, np.ndarray)) and len(fitted_vals) >= 3:
            if (fitted_vals[-1] - fitted_vals[-3]) > 0.01:
                plateau_reached = False
        features['plateau_reached'] = plateau_reached

        # --- Model-specific feature extraction ---
        if model == 'single' and len(params) >= 3:
            A, k, C = params
            endpoint = A + C
            # Mobile fraction is the endpoint (final plateau) in a normalized curve
            # Clamp to 100% maximum to prevent nonsensical values from fitting instabilities
            mobile_fraction = min(endpoint * 100, 100.0) if np.isfinite(endpoint) else np.nan
            features.update({
                'amplitude': A, 'rate_constant': k, 'offset': C,
                'mobile_fraction': mobile_fraction,
                'rate_constant_fast': k,
                'proportion_of_mobile_fast': 100.0
            })
            features.update(FRAPAnalysisCore.interpret_kinetics(k, effective_radius_um, calibration))
            features['half_time_fast'] = features.get('half_time_binding')
            features['proportion_of_total_fast'] = features['mobile_fraction']

        elif model == 'double' and len(params) >= 5:
            A1, k1, A2, k2, C = params
            total_amp = A1 + A2
            endpoint = total_amp + C
            components = sorted([(k1, A1), (k2, A2)], key=lambda x: x[0], reverse=True)
            k_fast, A_fast = components[0]
            k_slow, A_slow = components[1]

            # Mobile fraction is the endpoint (final plateau) in a normalized curve
            # Clamp to 100% maximum to prevent nonsensical values from fitting instabilities
            mobile_fraction = min(endpoint * 100, 100.0) if np.isfinite(endpoint) else np.nan
            
            features.update({
                'mobile_fraction': mobile_fraction,
                'rate_constant_fast': k_fast,
                'rate_constant_slow': k_slow,
                'proportion_of_mobile_fast': (A_fast / total_amp) * 100 if total_amp > 0 else 0,
                'proportion_of_mobile_slow': (A_slow / total_amp) * 100 if total_amp > 0 else 0,
            })
            features['proportion_of_total_fast'] = (A_fast / endpoint) * 100 if endpoint > 0 else 0
            features['proportion_of_total_slow'] = (A_slow / endpoint) * 100 if endpoint > 0 else 0
            features.update(FRAPAnalysisCore.interpret_kinetics(k_fast, effective_radius_um, calibration))
            features['half_time_fast'] = features.get('half_time_binding')

        elif model == 'triple' and len(params) >= 7:
            A1, k1, A2, k2, A3, k3, C = params
            total_amp = A1 + A2 + A3
            endpoint = total_amp + C
            components = sorted([(k1, A1), (k2, A2), (k3, A3)], key=lambda x: x[0], reverse=True)
            k_fast, A_fast = components[0]
            k_med, A_med = components[1]
            k_slow, A_slow = components[2]

            # Mobile fraction is the endpoint (final plateau) in a normalized curve
            # Clamp to 100% maximum to prevent nonsensical values from fitting instabilities
            mobile_fraction = min(endpoint * 100, 100.0) if np.isfinite(endpoint) else np.nan
            
            features.update({
                'mobile_fraction': mobile_fraction,
                'rate_constant_fast': k_fast,
                'rate_constant_medium': k_med,
                'rate_constant_slow': k_slow,
                'proportion_of_mobile_fast': (A_fast / total_amp) * 100 if total_amp > 0 else 0,
                'proportion_of_mobile_medium': (A_med / total_amp) * 100 if total_amp > 0 else 0,
                'proportion_of_mobile_slow': (A_slow / total_amp) * 100 if total_amp > 0 else 0,
            })
            features['proportion_of_total_fast'] = (A_fast / endpoint) * 100 if endpoint > 0 else 0
            features['proportion_of_total_medium'] = (A_med / endpoint) * 100 if endpoint > 0 else 0
            features['proportion_of_total_slow'] = (A_slow / endpoint) * 100 if endpoint > 0 else 0
            features.update(FRAPAnalysisCore.interpret_kinetics(k_fast, effective_radius_um, calibration))
            features['half_time_fast'] = features.get('half_time_binding')

        features['immobile_fraction'] = 100 - features.get('mobile_fraction', 0)
        
        # Add model information and quality metrics
        features.update({
            'model': model,
            'r2': best_fit.get('r2', np.nan),
            'aic': best_fit.get('aic', np.nan),
            'bic': best_fit.get('bic', np.nan)
        })
        
        return features

    @staticmethod
    def perform_clustering(features_df, n_clusters=2, method='kmeans'):
        """
        Perform clustering on features extracted from FRAP fits
        
        Parameters:
        -----------
        features_df : pandas.DataFrame
            DataFrame containing features for clustering
        n_clusters : int
            Number of clusters to form
        method : str
            Clustering method ('kmeans', 'hierarchical', or 'dbscan')
            
        Returns:
        --------
        tuple
            (labels, model, silhouette)
        """
        if features_df.empty:
            return None, None, None
            
        # Select numeric features for clustering
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        X = features_df[numeric_cols].values
        
        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0)
        
        # Standardize data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform clustering based on method
        if method == 'kmeans':
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = model.fit_predict(X_scaled)
        elif method == 'hierarchical':
            model = AgglomerativeClustering(n_clusters=n_clusters)
            labels = model.fit_predict(X_scaled)
        elif method == 'dbscan':
            model = DBSCAN(eps=0.5, min_samples=5)
            labels = model.fit_predict(X_scaled)
        else:
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = model.fit_predict(X_scaled)
            
        # Calculate silhouette score if there are enough samples and more than one cluster
        if len(np.unique(labels)) > 1 and len(X) > n_clusters:
            silhouette = silhouette_score(X_scaled, labels)
        else:
            silhouette = np.nan
            
        return labels, model, silhouette

    @staticmethod
    def identify_outliers(features_df, columns, iqr_multiplier=1.5):
        """
        Identify outliers in the dataset based on the IQR method.
        
        Parameters:
        -----------
        features_df : pandas.DataFrame
            DataFrame containing features. Must contain a 'file_path' column.
        columns : list
            List of columns to check for outliers.
        iqr_multiplier : float
            The multiplier for the IQR range.
            
        Returns:
        --------
        list
            List of file paths identified as outliers.
        """
        if features_df.empty or not columns or 'file_path' not in features_df.columns:
            return []
        outlier_indices = set()
        for col in columns:
            if col in features_df.columns and pd.api.types.is_numeric_dtype(features_df[col]):
                Q1, Q3 = features_df[col].quantile(0.25), features_df[col].quantile(0.75)
                IQR = Q3 - Q1
                if IQR == 0: continue
                lower_bound, upper_bound = Q1 - iqr_multiplier * IQR, Q3 + iqr_multiplier * IQR
                current_outliers = features_df.index[(features_df[col] < lower_bound) | (features_df[col] > upper_bound)].tolist()
                outlier_indices.update(current_outliers)
        return features_df.loc[list(outlier_indices), 'file_path'].tolist() if outlier_indices else []

    @staticmethod
    def analyze_frap_data(df, intensity_col='normalized', time_col='time'):
        """Lightweight analysis pipeline returning best fit and feature dict.

        Parameters
        ----------
        df : pandas.DataFrame
            Must contain columns for time and normalized intensity or raw intensity.
        intensity_col : str
            Column name containing normalized intensities (fallback to 'intensity').
        time_col : str
            Column name containing time values.

        Returns
        -------
        dict with keys: 'fits', 'best_fit', 'features'
        """
        import pandas as pd
        if df is None or len(df) == 0:
            return {'fits': [], 'best_fit': None, 'features': None}
        if time_col not in df.columns:
            raise ValueError(f"Missing time column '{time_col}'")
        if intensity_col not in df.columns:
            # Try to create normalized column from raw intensity / pre-bleach mean
            raw_col = 'intensity' if 'intensity' in df.columns else None
            if raw_col is None:
                raise ValueError("No intensity column available for normalization")
            pre_bleach_mean = df[raw_col].iloc[:max(2, len(df)//10)].mean()
            df[intensity_col] = df[raw_col] / pre_bleach_mean if pre_bleach_mean != 0 else df[raw_col]
        time = df[time_col].to_numpy(dtype=float)
        intensity = df[intensity_col].to_numpy(dtype=float)
        fits = FRAPAnalysisCore.fit_all_models(time, intensity)
        best = FRAPAnalysisCore.select_best_fit(fits) if fits else None
        features = FRAPAnalysisCore.extract_clustering_features(best) if best else None
        return {'fits': fits, 'best_fit': best, 'features': features}

    @staticmethod
    def fit_group_models(traces, model='single'):
        """
        Perform global simultaneous fitting across multiple traces with shared kinetic parameters
        but individual amplitudes and offsets.
        
        Parameters:
        -----------
        traces : list
            List of (time, intensity) tuples for each trace
        model : str
            Model type ('single', 'double', or 'triple')
            
        Returns:
        --------
        dict
            Dictionary containing global fit results
        """
        from scipy import optimize
        
        if len(traces) < 2:
            raise ValueError("Need at least 2 traces for global fitting")
        
        # Use the shortest timeline as reference and interpolate others
        min_time_end = min(t[-1] for t, _ in traces)
        t_common = np.linspace(0, min_time_end, 200)
        
        # Interpolate all traces to common time grid
        Y = []
        for t, y in traces:
            Y.append(np.interp(t_common, t, y))
        Y = np.vstack(Y)  # shape: n_files × n_time
        
        n_traces = Y.shape[0]
        
        if model == 'single':
            # Parameters: [k_shared, A1, A2, ..., An, C_shared]
            def residual(p):
                k = p[0]  # shared rate constant
                As = p[1:1+n_traces]  # per-trace amplitudes
                C = p[-1]  # shared offset
                
                # Model: y = A_i * (1 - exp(-k * t)) + C
                fit = As[:, None] * (1 - np.exp(-k * t_common)) + C
                return (Y - fit).ravel()
            
            # Initial parameters: k, A_i..., C
            A0_estimates = [np.max(y) - np.min(y) for y in Y]
            k0 = 0.05  # reasonable initial guess
            C0 = np.mean([np.min(y) for y in Y])
            
            p0 = np.hstack([k0, A0_estimates, C0])
            bounds_lower = np.hstack([1e-6, np.zeros(n_traces), -np.inf])
            bounds_upper = np.hstack([np.inf, np.full(n_traces, np.inf), np.inf])
            bounds = (bounds_lower, bounds_upper)
            
        elif model == 'double':
            # Parameters: [k1_shared, k2_shared, A1_1, A2_1, A1_2, A2_2, ..., C_shared]
            def residual(p):
                k1, k2 = p[0], p[1]  # shared rate constants
                As = p[2:2+2*n_traces].reshape(n_traces, 2)  # per-trace amplitudes (A1, A2 for each trace)
                C = p[-1]  # shared offset
                
                # Model: y = A1_i * (1 - exp(-k1 * t)) + A2_i * (1 - exp(-k2 * t)) + C
                fit = (As[:, 0:1] * (1 - np.exp(-k1 * t_common)) + 
                       As[:, 1:2] * (1 - np.exp(-k2 * t_common)) + C)
                return (Y - fit).ravel()
            
            # Initial parameters
            A0_estimates = [(np.max(y) - np.min(y))/2 for y in Y]
            k1_0, k2_0 = 0.1, 0.02  # fast and slow components
            C0 = np.mean([np.min(y) for y in Y])
            
            p0 = np.hstack([k1_0, k2_0, np.tile(A0_estimates, 2), C0])
            bounds_lower = np.hstack([1e-6, 1e-6, np.zeros(2*n_traces), -np.inf])
            bounds_upper = np.hstack([np.inf, np.inf, np.full(2*n_traces, np.inf), np.inf])
            bounds = (bounds_lower, bounds_upper)
            
        elif model == 'triple':
            # Parameters: [k1_shared, k2_shared, k3_shared, A1_1, A2_1, A3_1, A1_2, A2_2, A3_2, ..., C_shared]
            def residual(p):
                k1, k2, k3 = p[0], p[1], p[2]  # shared rate constants
                As = p[3:3+3*n_traces].reshape(n_traces, 3)  # per-trace amplitudes
                C = p[-1]  # shared offset
                
                # Model: y = A1_i * (1 - exp(-k1 * t)) + A2_i * (1 - exp(-k2 * t)) + A3_i * (1 - exp(-k3 * t)) + C
                fit = (As[:, 0:1] * (1 - np.exp(-k1 * t_common)) + 
                       As[:, 1:2] * (1 - np.exp(-k2 * t_common)) + 
                       As[:, 2:3] * (1 - np.exp(-k3 * t_common)) + C)
                return (Y - fit).ravel()
            
            # Initial parameters
            A0_estimates = [(np.max(y) - np.min(y))/3 for y in Y]
            k1_0, k2_0, k3_0 = 0.2, 0.05, 0.01  # fast, medium, slow components
            C0 = np.mean([np.min(y) for y in Y])
            
            p0 = np.hstack([k1_0, k2_0, k3_0, np.tile(A0_estimates, 3), C0])
            bounds_lower = np.hstack([1e-6, 1e-6, 1e-6, np.zeros(3*n_traces), -np.inf])
            bounds_upper = np.hstack([np.inf, np.inf, np.inf, np.full(3*n_traces, np.inf), np.inf])
            bounds = (bounds_lower, bounds_upper)
        else:
            raise ValueError(f"Unsupported model type: {model}")
        
        # Initialize outputs to safe defaults to avoid unbound-variable warnings in static analysis
        shared_params = {}
        individual_params = []
        shared_offset = None
        fitted_curves = None
        r2_values = []

        try:
            # Perform global optimization
            sol = optimize.least_squares(residual, p0, bounds=bounds, max_nfev=5000)
            
            # Calculate fit statistics
            fitted_residuals = residual(sol.x)
            rss = np.sum(fitted_residuals**2)
            n_data = len(fitted_residuals)
            n_params = len(sol.x)
            
            # Calculate AIC and BIC
            aic = FRAPAnalysisCore.compute_aic(rss, n_data, n_params)
            bic = FRAPAnalysisCore.compute_bic(rss, n_data, n_params)
            
            # Calculate R-squared for each trace
            # residual() returns (Y - fit), so reconstruct fitted values as fit = Y - residuals
            fitted_curves = Y - fitted_residuals.reshape(Y.shape)
            r2_values = []
            for i in range(n_traces):
                r2 = FRAPAnalysisCore.compute_r_squared(Y[i], fitted_curves[i])
                r2_values.append(r2)
            
            # Extract parameters based on model
            if model == 'single':
                shared_params = {'k': sol.x[0]}
                individual_params = [{'A': sol.x[1+i]} for i in range(n_traces)]
                shared_offset = sol.x[-1]
                
            elif model == 'double':
                shared_params = {'k1': sol.x[0], 'k2': sol.x[1]}
                individual_params = []
                for i in range(n_traces):
                    individual_params.append({
                        'A1': sol.x[2 + i],
                        'A2': sol.x[2 + n_traces + i]
                    })
                shared_offset = sol.x[-1]
                
            elif model == 'triple':
                shared_params = {'k1': sol.x[0], 'k2': sol.x[1], 'k3': sol.x[2]}
                individual_params = []
                for i in range(n_traces):
                    individual_params.append({
                        'A1': sol.x[3 + i],
                        'A2': sol.x[3 + n_traces + i],
                        'A3': sol.x[3 + 2*n_traces + i]
                    })
                shared_offset = sol.x[-1]
            
            return {
                'model': model,
                'success': sol.success,
                'shared_params': shared_params,
                'individual_params': individual_params,
                'shared_offset': shared_offset,
                'rss': rss,
                'aic': aic,
                'bic': bic,
                'r2_values': r2_values,
                'mean_r2': np.mean(r2_values) if r2_values else np.nan,
                'n_traces': n_traces,
                'n_params': n_params,
                'fitted_curves': fitted_curves,
                'common_time': t_common,
                'optimization_result': sol
            }
            
        except Exception as e:
            logging.error(f"Global fitting failed for {model} model: {e}")
            return {
                'model': model,
                'success': False,
                'error': str(e)
            }
    
    @staticmethod
    def fit_advanced_models(time, intensity, bleach_radius_um, pixel_size=1.0):
        """
        Fit advanced kinetic models to FRAP data using lmfit.
        
        This method provides sophisticated models for complex biological phenomena:
        - Anomalous diffusion (subdiffusion in crowded environments)
        - Reaction-diffusion kinetics (binding + diffusion)
        
        Parameters:
        -----------
        time : np.ndarray
            Time points
        intensity : np.ndarray
            Normalized intensity values
        bleach_radius_um : float
            Bleach spot radius in microns
        pixel_size : float
            Pixel size in microns (default: 1.0)
            
        Returns:
        --------
        list of dict
            Results for each successfully fitted advanced model, or empty list if unavailable
            
        Notes:
        ------
        Requires lmfit library: pip install lmfit
        """
        if not ADVANCED_FITTING_AVAILABLE:
            logging.warning("Advanced fitting not available - install lmfit")
            return []
        
        try:
            # Get post-bleach data
            t_fit, intensity_fit, _ = FRAPAnalysisCore.get_post_bleach_data(time, intensity)
            
            # Convert bleach radius to physical units
            effective_radius_um = bleach_radius_um * pixel_size
            
            # Fit all advanced models
            results = fit_all_advanced_models(t_fit, intensity_fit, effective_radius_um)
            
            return results
            
        except Exception as e:
            logging.error(f"Advanced model fitting failed: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return []
