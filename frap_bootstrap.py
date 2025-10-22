"""
Bootstrap confidence intervals for FRAP parameters
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, cpu_count
import logging
from scipy.optimize import curve_fit

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_model_function(model_name):
    """Get the appropriate model function by name."""
    from frap_core import FRAPAnalysisCore
    
    model_functions = {
        'single': FRAPAnalysisCore.single_component,
        'single_component': FRAPAnalysisCore.single_component,
        'double': FRAPAnalysisCore.two_component,
        'two_component': FRAPAnalysisCore.two_component,
        'triple': FRAPAnalysisCore.three_component,
        'three_component': FRAPAnalysisCore.three_component
    }
    
    return model_functions.get(model_name)


def _extract_params_from_popt(popt, model_name, model_func):
    """Extract key parameters from fitted model parameters."""
    if model_name == 'single':
        A, k, C = popt
        mobile_fraction = (A + C)
        primary_rate = k
    elif model_name == 'double':
        A1, k1, A2, k2, C = popt
        mobile_fraction = (A1 + A2 + C)
        primary_rate = max(k1, k2)
    elif model_name == 'triple':
        A1, k1, A2, k2, A3, k3, C = popt
        mobile_fraction = (A1 + A2 + A3 + C)
        primary_rate = max(k1, k2, k3)
    elif model_name.startswith('anomalous'):
        A, tau, beta, C = popt
        mobile_fraction = (A + C)
        # Convert tau to effective rate for comparison
        primary_rate = 1.0 / tau if tau > 0 else np.nan
    else:
        return None
    return {'mobile_fraction': mobile_fraction, 'primary_rate': primary_rate}


def _run_single_bootstrap_iteration(t_fit, fitted_y, residuals, model_func, initial_params, bounds, 
                                    bleach_radius_um, gfp_mw, gfp_d, model_name):
    """
    Runs a single iteration of the bootstrap process.
    
    Args:
        t_fit (np.ndarray): Time data for fitting.
        fitted_y (np.ndarray): The fitted curve from the original data.
        residuals (np.ndarray): Residuals from the original fit.
        model_func (callable): The function for the curve fitting model.
        initial_params (list): Initial parameters for the fit.
        bounds (tuple): Bounds for the fitting parameters.
        bleach_radius_um (float): Bleach radius in micrometers.
        gfp_mw (float): Molecular weight of GFP.
        gfp_d (float): Diffusion coefficient of GFP.
        model_name (str): Name of the model being fit.
        
    Returns:
        dict: A dictionary of the calculated parameters for this iteration, or None if fitting fails.
    """
    try:
        # Create bootstrap sample by adding resampled residuals
        bootstrap_residuals = np.random.choice(residuals, size=len(residuals), replace=True)
        bootstrap_y = fitted_y + bootstrap_residuals
        
        # Refit the model to the bootstrap sample
        popt, _ = curve_fit(model_func, t_fit, bootstrap_y, p0=initial_params, bounds=bounds)
        
        # Extract parameters from the new fit
        params = _extract_params_from_popt(popt, model_name, model_func)
        
        if not params:
            return None
        
        # Calculate derived quantities
        k = params['primary_rate']
        D = (bleach_radius_um**2 * k) / 4.0 if k > 0 else np.nan
        app_mw = gfp_mw * (gfp_d / D)**3 if D > 0 else np.nan
        
        return {
            'koff': k,
            'D': D,
            'mobile_fraction': params.get('mobile_fraction', np.nan),
            'app_mw': app_mw
        }
    except Exception as e:
        logging.debug(f"Bootstrap iteration failed: {e}")
        return None


def run_bootstrap(best_fit, t_fit, intensity_fit, bleach_radius_um, n_bootstrap=1000, gfp_mw=27.0, gfp_d=25.0):
    """
    Performs parametric bootstrap to estimate confidence intervals for fitted parameters.
    
    Args:
        best_fit (dict): The best fit dictionary from FRAPAnalysisCore.
        t_fit (np.ndarray): The time data used for fitting.
        intensity_fit (np.ndarray): The intensity data used for fitting.
        bleach_radius_um (float): The radius of the bleach spot in micrometers.
        n_bootstrap (int): The number of bootstrap iterations.
        gfp_mw (float): Molecular weight of GFP.
        gfp_d (float): Diffusion coefficient of GFP.
        
    Returns:
        dict: A dictionary containing the median and 95% CIs for the parameters.
    """
    if not best_fit:
        return None
    
    # Import model functions instead of relying on 'func' key which may not exist in session files
    from frap_core import FRAPAnalysisCore
    
    model_name = best_fit['model']
    initial_params = best_fit['params']
    fitted_y = best_fit['fitted_values']
    residuals = intensity_fit - fitted_y
    
    # Map model names to functions
    model_functions = {
        'single': FRAPAnalysisCore.single_component,
        'double': FRAPAnalysisCore.two_component,
        'triple': FRAPAnalysisCore.three_component
    }
    
    model_func = model_functions.get(model_name)
    if model_func is None:
        print(f"Warning: Unknown model type '{model_name}' for bootstrap analysis")
        return None
    
    # Define bounds for the model
    if model_name == 'single':
        bounds = ([0, 1e-6, -np.inf], [np.inf, np.inf, np.inf])
    elif model_name == 'double':
        bounds = ([0, 1e-6, 0, 1e-6, -np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf])
    elif model_name == 'triple':
        bounds = ([0, 1e-6, 0, 1e-6, 0, 1e-6, -np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
    elif model_name.startswith('anomalous'):
        bounds = ([0, 1e-6, 1e-3, -np.inf], [np.inf, np.inf, 1.0, np.inf])
    else:
        return None
    
    # Adjust n_bootstrap based on CPU availability
    n_cpus = cpu_count()
    if n_cpus <= 2 and n_bootstrap > 200:
        n_bootstrap = 200
        logging.info("CPU limited, reducing bootstrap iterations to 200.")
    
    # Parallelize the bootstrap iterations
    results = Parallel(n_jobs=-1)(
        delayed(_run_single_bootstrap_iteration)(
            t_fit, fitted_y, residuals, model_func, initial_params, bounds, 
            bleach_radius_um, gfp_mw, gfp_d, model_name
        ) for _ in range(n_bootstrap)
    )
    
    # Filter out failed iterations and convert to DataFrame
    valid_results = [r for r in results if r is not None]
    if not valid_results:
        logging.warning("All bootstrap iterations failed.")
        return None
    
    results_df = pd.DataFrame(valid_results)
    
    # Calculate median and 95% confidence intervals
    ci_results = {}
    for param in ['koff', 'D', 'mobile_fraction', 'app_mw']:
        if param in results_df.columns:
            # Sort the results to calculate percentiles
            param_dist = results_df[param].dropna().sort_values()
            if len(param_dist) > 0:
                ci_results[f'{param}_median'] = param_dist.quantile(0.5)
                ci_results[f'{param}_ci_low'] = param_dist.quantile(0.025)
                ci_results[f'{param}_ci_high'] = param_dist.quantile(0.975)
            else:
                ci_results[f'{param}_median'] = np.nan
                ci_results[f'{param}_ci_low'] = np.nan
                ci_results[f'{param}_ci_high'] = np.nan
    
    return ci_results
