import numpy as np
import pandas as pd
from joblib import Parallel, delayed, cpu_count
import logging
from frap_core import FRAPAnalysisCore
from scipy.optimize import curve_fit

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _extract_params_from_popt(popt, model_name):
    if model_name == 'single':
        A, k, C = popt
        mobile_fraction = (A + C) * 100.0
        primary_rate = k
    elif model_name == 'double':
        A1, k1, A2, k2, C = popt
        mobile_fraction = (A1 + A2 + C) * 100.0
        primary_rate = max(k1, k2)
    elif model_name == 'triple':
        A1, k1, A2, k2, A3, k3, C = popt
        mobile_fraction = (A1 + A2 + A3 + C) * 100.0
        primary_rate = max(k1, k2, k3)
    else:
        return None
    return {'mobile_fraction': mobile_fraction, 'primary_rate': primary_rate}

def _run_single_bootstrap_iteration(t_fit, fitted_y, residuals, model_func, initial_params, bounds, bleach_radius_um, gfp_mw, gfp_d):
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
        model_name = 'single' if model_func == FRAPAnalysisCore.single_component else 'double' if model_func == FRAPAnalysisCore.two_component else 'triple'
        params = _extract_params_from_popt(popt, model_name)

        if not params:
            return None

        # Interpret kinetics to get D and app_mw
        kinetic_interp = FRAPAnalysisCore.interpret_kinetics(params['primary_rate'], bleach_radius_um, gfp_d=gfp_d, gfp_mw=gfp_mw)

        return {
            'koff': kinetic_interp['k_off'],
            'D': kinetic_interp['diffusion_coefficient'],
            'mobile_fraction': params.get('mobile_fraction', np.nan),
            'app_mw': kinetic_interp['apparent_mw']
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

    model_func = best_fit['func']
    initial_params = best_fit['params']
    fitted_y = best_fit['fitted_values']
    residuals = intensity_fit - fitted_y

    # Define bounds for the model
    if best_fit['model'] == 'single':
        bounds = ([0, 1e-6, -np.inf], [np.inf, np.inf, np.inf])
    elif best_fit['model'] == 'double':
        bounds = ([0, 1e-6, 0, 1e-6, -np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf])
    elif best_fit['model'] == 'triple':
        bounds = ([0, 1e-6, 0, 1e-6, 0, 1e-6, -np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
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
            t_fit, fitted_y, residuals, model_func, initial_params, bounds, bleach_radius_um, gfp_mw, gfp_d
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
