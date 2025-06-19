"""
FRAP Analysis Core Module
Contains core computational routines for FRAP analysis
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

class FRAPAnalysisCore:
    @staticmethod
    def get_post_bleach_data(time, intensity):
        """
        Identify the minimum ROI1 intensity (bleach point) and return only the post-bleach data.
        The returned time is re-zeroed at the bleach event.
        
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
        bleach_index = np.argmin(intensity)
        
        # Get the post-bleach data
        post_bleach_time = time[bleach_index:]
        post_bleach_intensity = intensity[bleach_index:]
        
        # Re-zero the time at the bleach point
        post_bleach_time = post_bleach_time - post_bleach_time[0]
        
        return post_bleach_time, post_bleach_intensity, bleach_index

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
                df = pd.read_excel(file_path, engine='xlrd')
            elif file_extension == '.xlsx':
                df = pd.read_excel(file_path, engine='openpyxl')
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
        except Exception as e:
            logging.error(f"Error loading file {file_path}: {e}")
            raise

        # Standardize column names
        # First, let's determine which columns are which
        def score_column(col_name):
            """Score each column name to determine its likely content"""
            col_lower = str(col_name).lower()
            
            # Time column scores
            time_keywords = ['time', 'sec', 'seconds', 't']
            time_score = sum(keyword in col_lower for keyword in time_keywords)
            
            # ROI1 column scores (primary bleach region)
            roi1_keywords = ['roi1', 'roi 1', 'bleach', 'frap', 'intensity1', 'intensity 1']
            roi1_score = sum(keyword in col_lower for keyword in roi1_keywords)
            
            # ROI2 column scores (reference/control region)
            roi2_keywords = ['roi2', 'roi 2', 'control', 'reference', 'intensity2', 'intensity 2']
            roi2_score = sum(keyword in col_lower for keyword in roi2_keywords)
            
            # ROI3 column scores (background region)
            roi3_keywords = ['roi3', 'roi 3', 'background', 'bg', 'intensity3', 'intensity 3']
            roi3_score = sum(keyword in col_lower for keyword in roi3_keywords)
            
            return {'time': time_score, 'roi1': roi1_score, 'roi2': roi2_score, 'roi3': roi3_score}
        
        # Calculate scores for each column
        scores = {col: score_column(col) for col in df.columns}
        
        # Identify columns based on highest scores
        col_assignments = {role: max(scores.keys(), key=lambda col: scores[col][role]) 
                          for role in ['time', 'roi1', 'roi2', 'roi3']}
        
        # Check if we have multiple columns with the same highest score
        # If so, use column position as a tiebreaker
        if len(set(col_assignments.values())) < 4:
            # Assume columns are in order: time, roi1, roi2, roi3
            numeric_cols = [col for col in df.columns if np.issubdtype(df[col].dtype, np.number)]
            if len(numeric_cols) >= 4:
                col_assignments = {
                    'time': numeric_cols[0],
                    'roi1': numeric_cols[1],
                    'roi2': numeric_cols[2],
                    'roi3': numeric_cols[3]
                }
        
        # Create a new dataframe with standardized column names
        standardized_df = pd.DataFrame({
            'time': df[col_assignments['time']],
            'roi1': df[col_assignments['roi1']],
            'roi2': df[col_assignments['roi2']],
            'roi3': df[col_assignments['roi3']]
        })
        
        return standardized_df

    @staticmethod
    def preprocess(df):
        """
        Preprocess the data by applying background correction and normalization
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Raw data with time, ROI1, ROI2, ROI3 columns
            
        Returns:
        --------
        pandas.DataFrame
            Processed dataframe with additional columns
        """
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        # Background correction
        result_df['roi1_bg_corrected'] = result_df['roi1'] - result_df['roi3']
        result_df['roi2_bg_corrected'] = result_df['roi2'] - result_df['roi3']
        
        # Find pre-bleach values (average of first few points)
        pre_bleach_points = min(5, len(df) // 4)  # Use first 5 points or 1/4 of data, whichever is smaller
        pre_bleach_roi1 = result_df['roi1_bg_corrected'][:pre_bleach_points].mean()
        pre_bleach_roi2 = result_df['roi2_bg_corrected'][:pre_bleach_points].mean()
        
        # Calculate reference-corrected values
        roi2_normalized = result_df['roi2_bg_corrected'] / pre_bleach_roi2
        result_df['double_normalized'] = result_df['roi1_bg_corrected'] / roi2_normalized / pre_bleach_roi1
        
        # Simple normalization (without reference correction)
        result_df['normalized'] = result_df['roi1_bg_corrected'] / pre_bleach_roi1
        
        return result_df

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
        Fit single, two, and three component models to the FRAP data
        
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
        t_fit, intensity_fit, _ = FRAPAnalysisCore.get_post_bleach_data(time, intensity)
        n = len(t_fit)
        fits = []
        
        # Initial guesses - starting from simpler model
        # For offset C, use the minimum intensity as a guess
        C0 = np.min(intensity_fit)
        
        # For amplitude A, use the difference between max and min as a guess
        A0 = np.max(intensity_fit) - np.min(intensity_fit)
        
        # For rate constant k, use a reasonable guess based on the time scale
        # Assuming ~60% recovery at 1/3 of the time span
        time_span = t_fit[-1] - t_fit[0]
        k0 = -np.log(0.4) / (time_span / 3)
        
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
    def extract_clustering_features(best_fit):
        """
        Extract features for clustering from the best fit model
        
        Parameters:
        -----------
        best_fit : dict
            Dictionary containing the best fit model information
            
        Returns:
        --------
        dict or None
            Dictionary containing features for clustering
        """
        if best_fit is None:
            return None
            
        model = best_fit['model']
        params = best_fit['params']
        features = {}
        
        # Default bleach spot radius in μm (will be replaced with user input in real application)
        default_spot_radius = 1.0  # μm
        
        # GFP reference values
        D_GFP = 25.0  # μm²/s (default reference value)
        Rg_GFP = 2.82  # nm
        MW_GFP = 27.0  # kDa
        
        if model == 'single':
            A, k, C = params
            features['amplitude'] = A
            features['rate_constant'] = k
            features['mobile_fraction'] = A / (1.0 - C) if C < 1.0 else np.nan
            features['half_time'] = np.log(2) / k if k > 0 else np.nan
            
            # Calculate diffusion coefficient - D = w²k/4 where w is the radius of the bleach spot
            diffusion_coef = (default_spot_radius**2 * k) / 4.0  # μm²/s
            features['diffusion_coefficient'] = diffusion_coef
            
            # Calculate radius of gyration using GFP as reference
            features['radius_of_gyration'] = Rg_GFP * (D_GFP / diffusion_coef) if diffusion_coef > 0 else np.nan
            
            # Estimate molecular weight (scales with Rg^3 for globular proteins)
            features['molecular_weight_estimate'] = MW_GFP * (features['radius_of_gyration'] / Rg_GFP)**3 if not np.isnan(features['radius_of_gyration']) else np.nan
            
        elif model == 'double':
            A1, k1, A2, k2, C = params
            total_amp = A1 + A2
            
            # Sort components by rate constant (fast to slow)
            components = sorted([(k1, A1), (k2, A2)], reverse=True)
            sorted_rates = [comp[0] for comp in components]
            sorted_amps = [comp[1] for comp in components]
            
            features['amplitude_1'] = sorted_amps[0]
            features['rate_constant_1'] = sorted_rates[0]
            features['amplitude_2'] = sorted_amps[1]
            features['rate_constant_2'] = sorted_rates[1]
            
            # Calculate proportions
            features['proportion_1'] = sorted_amps[0] / total_amp if total_amp > 0 else np.nan
            features['proportion_2'] = sorted_amps[1] / total_amp if total_amp > 0 else np.nan
            
            # Mobile fraction
            features['mobile_fraction'] = total_amp / (1.0 - C) if C < 1.0 else np.nan
            
            # Weighted average half-time
            if total_amp > 0:
                features['half_time'] = (sorted_amps[0] * np.log(2) / sorted_rates[0] + 
                                         sorted_amps[1] * np.log(2) / sorted_rates[1]) / total_amp if sorted_rates[0] > 0 and sorted_rates[1] > 0 else np.nan
            else:
                features['half_time'] = np.nan
                
            # Calculate diffusion coefficients for both components
            features['diffusion_coefficient_1'] = (default_spot_radius**2 * sorted_rates[0]) / 4.0
            features['diffusion_coefficient_2'] = (default_spot_radius**2 * sorted_rates[1]) / 4.0
            
            # Calculate radii of gyration
            features['radius_of_gyration_1'] = Rg_GFP * (D_GFP / features['diffusion_coefficient_1']) if features['diffusion_coefficient_1'] > 0 else np.nan
            features['radius_of_gyration_2'] = Rg_GFP * (D_GFP / features['diffusion_coefficient_2']) if features['diffusion_coefficient_2'] > 0 else np.nan
            
            # Estimate molecular weights
            features['molecular_weight_estimate_1'] = MW_GFP * (features['radius_of_gyration_1'] / Rg_GFP)**3 if not np.isnan(features['radius_of_gyration_1']) else np.nan
            features['molecular_weight_estimate_2'] = MW_GFP * (features['radius_of_gyration_2'] / Rg_GFP)**3 if not np.isnan(features['radius_of_gyration_2']) else np.nan
            
        elif model == 'triple':
            A1, k1, A2, k2, A3, k3, C = params
            total_amp = A1 + A2 + A3
            
            # Sort components by rate constant (fast to slow)
            components = sorted([(k1, A1), (k2, A2), (k3, A3)], reverse=True)
            sorted_rates = [comp[0] for comp in components]
            sorted_amps = [comp[1] for comp in components]
            
            features['amplitude_1'] = sorted_amps[0]
            features['rate_constant_1'] = sorted_rates[0]
            features['amplitude_2'] = sorted_amps[1]
            features['rate_constant_2'] = sorted_rates[1]
            features['amplitude_3'] = sorted_amps[2]
            features['rate_constant_3'] = sorted_rates[2]
            
            # Calculate proportions
            features['proportion_1'] = sorted_amps[0] / total_amp if total_amp > 0 else np.nan
            features['proportion_2'] = sorted_amps[1] / total_amp if total_amp > 0 else np.nan
            features['proportion_3'] = sorted_amps[2] / total_amp if total_amp > 0 else np.nan
            
            # Mobile fraction
            features['mobile_fraction'] = total_amp / (1.0 - C) if C < 1.0 else np.nan
            
            # Weighted average half-time
            if total_amp > 0:
                features['half_time'] = (sorted_amps[0] * np.log(2) / sorted_rates[0] + 
                                         sorted_amps[1] * np.log(2) / sorted_rates[1] + 
                                         sorted_amps[2] * np.log(2) / sorted_rates[2]) / total_amp if sorted_rates[0] > 0 and sorted_rates[1] > 0 and sorted_rates[2] > 0 else np.nan
            else:
                features['half_time'] = np.nan
                
            # Calculate diffusion coefficients for all components
            features['diffusion_coefficient_1'] = (default_spot_radius**2 * sorted_rates[0]) / 4.0
            features['diffusion_coefficient_2'] = (default_spot_radius**2 * sorted_rates[1]) / 4.0
            features['diffusion_coefficient_3'] = (default_spot_radius**2 * sorted_rates[2]) / 4.0
            
            # Calculate radii of gyration
            features['radius_of_gyration_1'] = Rg_GFP * (D_GFP / features['diffusion_coefficient_1']) if features['diffusion_coefficient_1'] > 0 else np.nan
            features['radius_of_gyration_2'] = Rg_GFP * (D_GFP / features['diffusion_coefficient_2']) if features['diffusion_coefficient_2'] > 0 else np.nan
            features['radius_of_gyration_3'] = Rg_GFP * (D_GFP / features['diffusion_coefficient_3']) if features['diffusion_coefficient_3'] > 0 else np.nan
            
            # Estimate molecular weights
            features['molecular_weight_estimate_1'] = MW_GFP * (features['radius_of_gyration_1'] / Rg_GFP)**3 if not np.isnan(features['radius_of_gyration_1']) else np.nan
            features['molecular_weight_estimate_2'] = MW_GFP * (features['radius_of_gyration_2'] / Rg_GFP)**3 if not np.isnan(features['radius_of_gyration_2']) else np.nan
            features['molecular_weight_estimate_3'] = MW_GFP * (features['radius_of_gyration_3'] / Rg_GFP)**3 if not np.isnan(features['radius_of_gyration_3']) else np.nan
        else:
            return None
            
        return features
        
    @staticmethod
    def compute_diffusion_coefficient(rate_constant, bleach_spot_radius=1.0):
        """
        Calculate diffusion coefficient from rate constant
        
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
        return (bleach_spot_radius**2 * rate_constant) / 4.0
    
    @staticmethod
    def interpret_kinetics(k, bleach_radius_um, gfp_d=25.0, gfp_rg=2.82, gfp_mw=27.0):
        """
        Centralized kinetics interpretation function with corrected mathematics
        
        Parameters:
        -----------
        k : float
            The fitted kinetic rate constant (1/s)
        bleach_radius_um : float
            The radius of the photobleach spot in micrometers
        gfp_d : float
            Reference diffusion coefficient for GFP (um^2/s)
        gfp_rg : float
            Reference radius of gyration for GFP (nm)
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
                'half_time_diffusion': np.nan,
                'half_time_binding': np.nan
            }

        # 1. Interpretation as a binding/unbinding process
        k_off = k  # The rate is the dissociation constant
        half_time_binding = np.log(2) / k  # Time for 50% recovery via binding

        # 2. Interpretation as a diffusion process
        # For 2D diffusion: D = (w^2 * k) / 4 where w is bleach radius and k is rate constant
        # This is the correct formula without the erroneous np.log(2) factor
        diffusion_coefficient = (bleach_radius_um**2 * k) / 4.0
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
        
    @staticmethod
    def compute_radius_of_gyration(diffusion_coefficient, reference_D=25.0, reference_Rg=2.82):
        """
        Calculate radius of gyration using the Stokes-Einstein relation with GFP as reference
        
        Parameters:
        -----------
        diffusion_coefficient : float
            Diffusion coefficient in μm²/s
        reference_D : float
            Reference diffusion coefficient (default: GFP = 25 μm²/s)
        reference_Rg : float
            Reference radius of gyration (default: GFP = 2.82 nm)
            
        Returns:
        --------
        float
            Radius of gyration in nm
        """
        if diffusion_coefficient <= 0:
            return np.nan
        return reference_Rg * (reference_D / diffusion_coefficient)
        
    @staticmethod
    def compute_molecular_weight_estimate(radius_of_gyration, reference_Rg=2.82, reference_MW=27.0):
        """
        Estimate molecular weight from radius of gyration
        Assumption: MW scales with Rg^3 for globular proteins
        
        Parameters:
        -----------
        radius_of_gyration : float
            Radius of gyration in nm
        reference_Rg : float
            Reference radius of gyration (default: GFP = 2.82 nm)
        reference_MW : float
            Reference molecular weight (default: GFP = 27 kDa)
            
        Returns:
        --------
        float
            Estimated molecular weight in kDa
        """
        if np.isnan(radius_of_gyration) or radius_of_gyration <= 0:
            return np.nan
        return reference_MW * (radius_of_gyration / reference_Rg)**3
    
    @staticmethod
    def compute_kinetic_details(fit_data, bleach_spot_radius=1.0, pixel_size=1.0, 
                               reference_D=25.0, reference_Rg=2.82, reference_MW=27.0, 
                               target_MW=27.0, scaling_alpha=1.0):
        """
        Compute detailed kinetic parameters from fit data, including both 
        diffusion and binding interpretations.
        
        Parameters:
        -----------
        fit_data : dict
            Dictionary containing fit information
        bleach_spot_radius : float
            Radius of the bleach spot in μm
        pixel_size : float
            Size of a pixel in μm
        reference_D : float
            Reference diffusion coefficient (GFP) in μm²/s
        reference_Rg : float
            Reference radius of gyration (GFP) in nm
        reference_MW : float
            Reference molecular weight (GFP) in kDa
        target_MW : float
            Target molecular weight to compare in kDa
        scaling_alpha : float
            Scaling factor for diffusion calculation
            
        Returns:
        --------
        dict
            Dictionary of detailed kinetic parameters
        """
        if fit_data is None:
            return {}
            
        model = fit_data['model']
        params = fit_data['params']
        actual_spot_radius = bleach_spot_radius * pixel_size
        details = {}
        
        if model == 'single':
            A, k, C = params
            # Mobile fraction
            mobile_fraction = A / (1.0 - C) if C < 1.0 else np.nan
            details['mobile_fraction'] = mobile_fraction
            
            # Interpret as diffusion
            diffusion_coef = (actual_spot_radius**2 * k) / 4.0
            radius_gyration = reference_Rg * (reference_D / diffusion_coef) if diffusion_coef > 0 else np.nan
            mw_estimate = reference_MW * (radius_gyration / reference_Rg)**3 if not np.isnan(radius_gyration) else np.nan
            
            # Calculate expected diffusion if it were pure diffusion
            expected_rg = reference_Rg * (reference_MW / target_MW)**(1/3) * scaling_alpha
            expected_D = reference_D * (reference_Rg / expected_rg)
            
            details['single_component'] = {
                'rate_constant': k,
                'half_time': np.log(2) / k if k > 0 else np.nan,
                'diffusion_coef': diffusion_coef,
                'radius_gyration': radius_gyration,
                'mw_estimate': mw_estimate,
                'expected_D': expected_D,
                'expected_rg': expected_rg,
                'is_pure_diffusion': np.isclose(diffusion_coef, expected_D, rtol=0.2)
            }
            
        elif model == 'double':
            A1, k1, A2, k2, C = params
            total_amp = A1 + A2
            mobile_fraction = total_amp / (1.0 - C) if C < 1.0 else np.nan
            details['mobile_fraction'] = mobile_fraction
            
            # Sort components (fast to slow)
            components = sorted([(k1, A1), (k2, A2)], reverse=True)
            rates = [comp[0] for comp in components]
            amps = [comp[1] for comp in components]
            
            details['components'] = []
            
            # Process each component
            for i, (k, A) in enumerate(zip(rates, amps)):
                # Proportion of this component
                prop = A / total_amp if total_amp > 0 else np.nan
                
                # Interpret as diffusion
                diffusion_coef = (actual_spot_radius**2 * k) / 4.0
                radius_gyration = reference_Rg * (reference_D / diffusion_coef) if diffusion_coef > 0 else np.nan
                mw_estimate = reference_MW * (radius_gyration / reference_Rg)**3 if not np.isnan(radius_gyration) else np.nan
                
                # Calculate expected diffusion if it were pure diffusion
                expected_rg = reference_Rg * (reference_MW / target_MW)**(1/3) * scaling_alpha
                expected_D = reference_D * (reference_Rg / expected_rg)
                
                details['components'].append({
                    'component': i+1,
                    'proportion': prop,
                    'rate_constant': k,
                    'half_time': np.log(2) / k if k > 0 else np.nan,
                    'diffusion_coef': diffusion_coef,
                    'radius_gyration': radius_gyration,
                    'mw_estimate': mw_estimate,
                    'expected_D': expected_D,
                    'expected_rg': expected_rg,
                    'is_pure_diffusion': np.isclose(diffusion_coef, expected_D, rtol=0.2)
                })
                
        elif model == 'triple':
            A1, k1, A2, k2, A3, k3, C = params
            total_amp = A1 + A2 + A3
            mobile_fraction = total_amp / (1.0 - C) if C < 1.0 else np.nan
            details['mobile_fraction'] = mobile_fraction
            
            # Sort components (fast to slow)
            components = sorted([(k1, A1), (k2, A2), (k3, A3)], reverse=True)
            rates = [comp[0] for comp in components]
            amps = [comp[1] for comp in components]
            
            details['components'] = []
            
            # Process each component
            for i, (k, A) in enumerate(zip(rates, amps)):
                # Proportion of this component
                prop = A / total_amp if total_amp > 0 else np.nan
                
                # Interpret as diffusion
                diffusion_coef = (actual_spot_radius**2 * k) / 4.0
                radius_gyration = reference_Rg * (reference_D / diffusion_coef) if diffusion_coef > 0 else np.nan
                mw_estimate = reference_MW * (radius_gyration / reference_Rg)**3 if not np.isnan(radius_gyration) else np.nan
                
                # Calculate expected diffusion if it were pure diffusion
                expected_rg = reference_Rg * (reference_MW / target_MW)**(1/3) * scaling_alpha
                expected_D = reference_D * (reference_Rg / expected_rg)
                
                details['components'].append({
                    'component': i+1,
                    'proportion': prop,
                    'rate_constant': k,
                    'half_time': np.log(2) / k if k > 0 else np.nan,
                    'diffusion_coef': diffusion_coef,
                    'radius_gyration': radius_gyration,
                    'mw_estimate': mw_estimate,
                    'expected_D': expected_D,
                    'expected_rg': expected_rg,
                    'is_pure_diffusion': np.isclose(diffusion_coef, expected_D, rtol=0.2)
                })
                
        return details

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