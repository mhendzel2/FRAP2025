"""
Advanced FRAP Fitting Models using lmfit
Provides sophisticated models for complex biological phenomena:
- Anomalous diffusion (stretched exponential)
- Reaction-diffusion kinetics
- Effective diffusion with binding
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# Check if lmfit is available
try:
    from lmfit import Model, Parameters
    LMFIT_AVAILABLE = True
except ImportError:
    LMFIT_AVAILABLE = False
    logger.warning("lmfit not available - advanced fitting models are disabled")
    logger.warning("Install with: pip install lmfit")

# =============================================================================
# ADVANCED MODEL DEFINITIONS
# =============================================================================

def anomalous_diffusion(t, A, C, tau, beta):
    """
    Stretched exponential model for anomalous diffusion.
    
    This model describes diffusion in crowded or heterogeneous environments
    where normal Fickian diffusion doesn't apply.
    
    Model: I(t) = A * (1 - exp(-(t/tau)^beta)) + C
    
    Parameters:
    -----------
    t : array-like
        Time points
    A : float
        Amplitude of recovery (mobile fraction - immobile fraction)
    C : float
        Baseline intensity (immobile fraction)
    tau : float
        Characteristic diffusion time
    beta : float
        Anomalous exponent (0 < beta <= 1)
        - beta = 1: Normal diffusion (Brownian motion)
        - beta < 1: Subdiffusion (hindered by obstacles/crowding)
        - beta > 1: Superdiffusion (rare, directed transport)
        
    Returns:
    --------
    array-like
        Predicted intensity values
    """
    return A * (1 - np.exp(-(t / tau)**beta)) + C


def reaction_diffusion_simple(t, F_imm, F_b, F_f, k_eff, D_app, w):
    """
    Simplified reaction-diffusion model for proteins that both diffuse and bind.
    
    This model captures the interplay between free diffusion and reversible
    binding to immobile structures (e.g., chromatin, nuclear matrix).
    
    Model combines:
    - Free diffusion component: exp(-4*D/w² * t)
    - Effective binding: exp(-k_eff * t)
    
    Parameters:
    -----------
    t : array-like
        Time points
    F_imm : float
        Immobile fraction (never recovers)
    F_b : float
        Bound fraction (recovers slowly via binding kinetics)
    F_f : float
        Free fraction (recovers quickly via diffusion)
    k_eff : float
        Effective binding rate (combination of k_on and k_off)
    D_app : float
        Apparent diffusion coefficient (μm²/s)
    w : float
        Bleach spot radius (μm) - fixed parameter
        
    Returns:
    --------
    array-like
        Predicted intensity values
        
    Notes:
    ------
    Total mobile fraction = F_f + F_b
    F_imm + F_b + F_f should sum to ~1.0 (normalized)
    """
    # Diffusion component (fast recovery)
    k_diff = 4 * D_app / (w**2)
    diffusion_term = F_f * (1 - np.exp(-k_diff * t))
    
    # Binding component (slow recovery)
    binding_term = F_b * (1 - np.exp(-k_eff * t))
    
    return F_imm + diffusion_term + binding_term


def reaction_diffusion_full(t, F_imm, F_mobile, k_on, k_off, D, w):
    """
    Full reaction-diffusion model with explicit on/off rates.
    
    This model explicitly accounts for binding and unbinding kinetics,
    allowing estimation of:
    - k_on: Binding rate to chromatin
    - k_off: Unbinding rate from chromatin
    - D: Free diffusion coefficient
    
    Parameters:
    -----------
    t : array-like
        Time points
    F_imm : float
        Truly immobile fraction
    F_mobile : float
        Mobile fraction (free + transiently bound)
    k_on : float
        Binding rate (s⁻¹)
    k_off : float
        Unbinding rate (s⁻¹)
    D : float
        Free diffusion coefficient (μm²/s)
    w : float
        Bleach spot radius (μm)
        
    Returns:
    --------
    array-like
        Predicted intensity values
    """
    # Effective rate constant
    k_eff = k_on + k_off
    
    # Diffusion component
    k_diff = 4 * D / (w**2)
    
    # Fraction of time bound
    f_bound = k_on / k_eff
    
    # Combined recovery
    recovery = F_mobile * (
        (1 - f_bound) * (1 - np.exp(-k_diff * t)) +
        f_bound * (1 - np.exp(-k_eff * t))
    )
    
    return F_imm + recovery


def double_anomalous(t, A1, tau1, beta1, A2, tau2, beta2, C):
    """
    Two-component anomalous diffusion model.
    
    Useful for systems with two distinct anomalous diffusion populations
    (e.g., different chromatin environments).
    
    Model: I(t) = A1*(1-exp(-(t/tau1)^beta1)) + A2*(1-exp(-(t/tau2)^beta2)) + C
    
    Parameters:
    -----------
    t : array-like
        Time points
    A1, A2 : float
        Amplitudes of two populations
    tau1, tau2 : float
        Characteristic times
    beta1, beta2 : float
        Anomalous exponents for each population
    C : float
        Baseline (immobile fraction)
        
    Returns:
    --------
    array-like
        Predicted intensity values
    """
    comp1 = A1 * (1 - np.exp(-(t / tau1)**beta1))
    comp2 = A2 * (1 - np.exp(-(t / tau2)**beta2))
    return comp1 + comp2 + C


def power_law_diffusion(t, A, C, D0, alpha):
    """
    Power-law diffusion model for highly heterogeneous environments.
    
    Describes systems where mean squared displacement grows as:
    <r²(t)> ~ t^alpha
    
    Parameters:
    -----------
    t : array-like
        Time points
    A : float
        Amplitude
    C : float
        Baseline
    D0 : float
        Generalized diffusion coefficient
    alpha : float
        Anomalous exponent (0 < alpha < 2)
        - alpha = 1: Normal diffusion
        - alpha < 1: Subdiffusion
        - alpha > 1: Superdiffusion
        
    Returns:
    --------
    array-like
        Predicted intensity values
    """
    return A * (1 - np.exp(-D0 * t**alpha)) + C


# =============================================================================
# FITTING FUNCTIONS
# =============================================================================

def fit_anomalous_diffusion(t: np.ndarray, y: np.ndarray, 
                           bleach_radius_um: float) -> Dict[str, Any]:
    """
    Fit anomalous diffusion (stretched exponential) model.
    
    Parameters:
    -----------
    t : np.ndarray
        Time points (post-bleach, starting from 0)
    y : np.ndarray
        Normalized intensity values
    bleach_radius_um : float
        Bleach spot radius in microns
        
    Returns:
    --------
    dict
        Fitting results with keys:
        - success: bool
        - model_name: str
        - params: dict of fitted parameters
        - fit_result: lmfit.ModelResult object
        - fitted_values: np.ndarray
        - r2: float
        - aic: float
        - bic: float
        - interpretation: dict
    """
    if not LMFIT_AVAILABLE:
        return {'success': False, 'error': 'lmfit not installed'}
    
    try:
        model = Model(anomalous_diffusion)
        
        # Initial parameter estimates
        y_min = np.min(y)
        y_max = np.max(y)
        y_range = y_max - y_min
        
        params = model.make_params(
            A=dict(value=y_range, min=0, max=2.0),
            C=dict(value=y_min, min=-0.1, max=1.0),
            tau=dict(value=np.median(t[t > 0]), min=1e-3, max=np.max(t)*2),
            beta=dict(value=0.7, min=0.1, max=1.5)
        )
        
        # Perform fit
        result = model.fit(y, params, t=t, method='least_squares')
        
        # Calculate goodness of fit metrics
        residuals = y - result.best_fit
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        n = len(y)
        k = len(result.params)
        aic = n * np.log(ss_res / n) + 2 * k
        bic = n * np.log(ss_res / n) + k * np.log(n)
        
        # Extract parameters
        A = result.params['A'].value
        C = result.params['C'].value
        tau = result.params['tau'].value
        beta = result.params['beta'].value
        
        # Physical interpretation
        mobile_fraction = (A / (A + C)) * 100 if (A + C) > 0 else np.nan
        
        # Classify diffusion type
        if beta < 0.5:
            diffusion_type = "Highly Subdiffusive (Strongly Hindered)"
        elif beta < 0.9:
            diffusion_type = "Subdiffusive (Hindered)"
        elif beta <= 1.1:
            diffusion_type = "Normal Diffusion (Brownian)"
        else:
            diffusion_type = "Superdiffusive (Directed Transport)"
        
        # Estimate effective diffusion coefficient
        # Using approximation: tau ≈ w²/(4*D_eff)
        D_eff = (bleach_radius_um**2) / (4 * tau)
        
        interpretation = {
            'mobile_fraction': mobile_fraction,
            'immobile_fraction': 100 - mobile_fraction,
            'tau': tau,
            'beta': beta,
            'diffusion_type': diffusion_type,
            'effective_D': D_eff,
            'anomaly_strength': abs(1 - beta)
        }
        
        return {
            'success': True,
            'model_name': 'anomalous_diffusion',
            'params': result.best_values,
            'param_errors': {name: result.params[name].stderr for name in result.params},
            'fit_result': result,
            'fitted_values': result.best_fit,
            'residuals': residuals,
            'r2': r2,
            'aic': aic,
            'bic': bic,
            'interpretation': interpretation
        }
        
    except Exception as e:
        logger.error(f"Anomalous diffusion fitting failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {'success': False, 'error': str(e)}


def fit_reaction_diffusion(t: np.ndarray, y: np.ndarray, 
                          bleach_radius_um: float,
                          model_type: str = 'simple') -> Dict[str, Any]:
    """
    Fit reaction-diffusion model.
    
    Parameters:
    -----------
    t : np.ndarray
        Time points
    y : np.ndarray
        Normalized intensity
    bleach_radius_um : float
        Bleach spot radius
    model_type : str
        'simple' or 'full' reaction-diffusion model
        
    Returns:
    --------
    dict
        Fitting results
    """
    if not LMFIT_AVAILABLE:
        return {'success': False, 'error': 'lmfit not installed'}
    
    try:
        if model_type == 'simple':
            model = Model(reaction_diffusion_simple)
            
            y_min = np.min(y)
            y_max = np.max(y)
            
            params = Parameters()
            params.add('F_imm', value=y_min, min=0, max=0.5)
            params.add('F_b', value=0.3, min=0, max=1)
            params.add('F_f', value=0.5, min=0, max=1)
            params.add('k_eff', value=0.1, min=1e-4, max=10)
            params.add('D_app', value=1.0, min=0.01, max=100)
            params.add('w', value=bleach_radius_um, vary=False)
            
            # Add constraint: F_imm + F_b + F_f ≈ 1
            params.add('total', expr='F_imm + F_b + F_f')
            
        else:  # full model
            model = Model(reaction_diffusion_full)
            
            y_min = np.min(y)
            
            params = Parameters()
            params.add('F_imm', value=y_min, min=0, max=0.5)
            params.add('F_mobile', value=1-y_min, min=0.5, max=1)
            params.add('k_on', value=0.1, min=1e-4, max=10)
            params.add('k_off', value=0.1, min=1e-4, max=10)
            params.add('D', value=1.0, min=0.01, max=100)
            params.add('w', value=bleach_radius_um, vary=False)
        
        # Perform fit
        result = model.fit(y, params, t=t, method='least_squares')
        
        # Calculate metrics
        residuals = y - result.best_fit
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        n = len(y)
        k = len([p for p in result.params.values() if p.vary])
        aic = n * np.log(ss_res / n) + 2 * k
        bic = n * np.log(ss_res / n) + k * np.log(n)
        
        # Interpretation
        if model_type == 'simple':
            F_imm = result.params['F_imm'].value
            F_b = result.params['F_b'].value
            F_f = result.params['F_f'].value
            k_eff = result.params['k_eff'].value
            D_app = result.params['D_app'].value
            
            mobile_fraction = (F_b + F_f) * 100
            bound_fraction = (F_b / (F_b + F_f)) * 100 if (F_b + F_f) > 0 else 0
            
            # Residence time for bound state
            t_res = 1 / k_eff if k_eff > 0 else np.inf
            
            interpretation = {
                'mobile_fraction': mobile_fraction,
                'immobile_fraction': F_imm * 100,
                'bound_fraction': bound_fraction,
                'free_fraction': 100 - bound_fraction,
                'effective_k': k_eff,
                'residence_time': t_res,
                'apparent_D': D_app
            }
        else:
            F_imm = result.params['F_imm'].value
            F_mobile = result.params['F_mobile'].value
            k_on = result.params['k_on'].value
            k_off = result.params['k_off'].value
            D = result.params['D'].value
            
            # Equilibrium binding fraction
            K_d = k_off / k_on if k_on > 0 else np.inf
            f_bound_eq = 1 / (1 + K_d) if np.isfinite(K_d) else 0
            
            interpretation = {
                'mobile_fraction': F_mobile * 100,
                'immobile_fraction': F_imm * 100,
                'k_on': k_on,
                'k_off': k_off,
                'K_d': K_d,
                'equilibrium_bound': f_bound_eq * 100,
                'diffusion_coeff': D,
                'on_rate_lifetime': 1/k_on if k_on > 0 else np.inf,
                'off_rate_lifetime': 1/k_off if k_off > 0 else np.inf
            }
        
        return {
            'success': True,
            'model_name': f'reaction_diffusion_{model_type}',
            'params': result.best_values,
            'param_errors': {name: result.params[name].stderr for name in result.params if result.params[name].vary},
            'fit_result': result,
            'fitted_values': result.best_fit,
            'residuals': residuals,
            'r2': r2,
            'aic': aic,
            'bic': bic,
            'interpretation': interpretation
        }
        
    except Exception as e:
        logger.error(f"Reaction-diffusion fitting failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {'success': False, 'error': str(e)}


def fit_all_advanced_models(t: np.ndarray, y: np.ndarray,
                           bleach_radius_um: float) -> list:
    """
    Fit all available advanced models and return results.
    
    Parameters:
    -----------
    t : np.ndarray
        Time points
    y : np.ndarray
        Intensity values
    bleach_radius_um : float
        Bleach spot radius
        
    Returns:
    --------
    list of dict
        Results for each successfully fitted model
    """
    results = []
    
    # Anomalous diffusion
    anom_result = fit_anomalous_diffusion(t, y, bleach_radius_um)
    if anom_result['success']:
        results.append(anom_result)
    
    # Reaction-diffusion (simple)
    rd_simple = fit_reaction_diffusion(t, y, bleach_radius_um, model_type='simple')
    if rd_simple['success']:
        results.append(rd_simple)
    
    # Reaction-diffusion (full)
    rd_full = fit_reaction_diffusion(t, y, bleach_radius_um, model_type='full')
    if rd_full['success']:
        results.append(rd_full)
    
    return results


def select_best_advanced_model(results: list) -> Optional[Dict[str, Any]]:
    """
    Select the best model based on AIC (lower is better).
    
    Parameters:
    -----------
    results : list
        List of fitting results
        
    Returns:
    --------
    dict or None
        Best model result
    """
    if not results:
        return None
    
    valid_results = [r for r in results if r['success'] and 'aic' in r]
    
    if not valid_results:
        return None
    
    # Sort by AIC (lower is better)
    best = min(valid_results, key=lambda x: x['aic'])
    
    return best
