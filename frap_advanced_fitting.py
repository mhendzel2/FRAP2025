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


# =============================================================================
# GROUP-LEVEL FITTING FUNCTIONS
# =============================================================================

def fit_mean_recovery_profile(time: np.ndarray, 
                              intensity_mean: np.ndarray,
                              intensity_sem: Optional[np.ndarray] = None,
                              bleach_radius_um: float = 1.0,
                              model: str = 'all') -> Dict[str, Any]:
    """
    Fit advanced models to mean recovery profile of a group.
    
    This allows comparison of recovery kinetics at the group level using
    sophisticated models that can capture anomalous diffusion, reaction-diffusion,
    and other complex behaviors.
    
    Parameters:
    -----------
    time : np.ndarray
        Time points (post-bleach)
    intensity_mean : np.ndarray
        Mean normalized intensity for the group
    intensity_sem : np.ndarray, optional
        Standard error of the mean (used for weighted fitting if provided)
    bleach_radius_um : float
        Bleach spot radius in microns
    model : str
        Which model(s) to fit:
        - 'all': Try all models and return best
        - 'anomalous': Anomalous diffusion only
        - 'reaction_diffusion_simple': Simplified reaction-diffusion
        - 'reaction_diffusion_full': Full reaction-diffusion
        
    Returns:
    --------
    dict
        Fitting results with model parameters and interpretation
    """
    if len(time) == 0 or len(intensity_mean) == 0:
        return {'success': False, 'error': 'Empty data arrays'}
    
    # Use weights if SEM provided
    weights = None
    if intensity_sem is not None and len(intensity_sem) == len(intensity_mean):
        # Avoid division by zero; use inverse variance weighting
        weights = 1.0 / (intensity_sem + 1e-10)**2
        weights = weights / np.sum(weights)  # Normalize
    
    results = []
    
    if model == 'all' or model == 'anomalous':
        result = fit_anomalous_diffusion(time, intensity_mean, bleach_radius_um)
        if result['success']:
            result['weights_used'] = weights is not None
            results.append(result)
    
    if model == 'all' or model == 'reaction_diffusion_simple':
        result = fit_reaction_diffusion(time, intensity_mean, bleach_radius_um, model_type='simple')
        if result['success']:
            result['weights_used'] = weights is not None
            results.append(result)
    
    if model == 'all' or model == 'reaction_diffusion_full':
        result = fit_reaction_diffusion(time, intensity_mean, bleach_radius_um, model_type='full')
        if result['success']:
            result['weights_used'] = weights is not None
            results.append(result)
    
    if model == 'all':
        # Return best model by AIC
        best_result = select_best_advanced_model(results)
        if best_result:
            best_result['all_results'] = results
            best_result['n_models_tested'] = len(results)
            return best_result
        else:
            return {'success': False, 'error': 'No models converged'}
    else:
        # Return single model result
        return results[0] if results else {'success': False, 'error': 'Model fitting failed'}


def compare_groups_advanced_fitting(group1_time: np.ndarray,
                                    group1_intensity: np.ndarray,
                                    group1_sem: Optional[np.ndarray],
                                    group2_time: np.ndarray,
                                    group2_intensity: np.ndarray,
                                    group2_sem: Optional[np.ndarray],
                                    group1_name: str = "Group 1",
                                    group2_name: str = "Group 2",
                                    bleach_radius_um: float = 1.0,
                                    model: str = 'all') -> Dict[str, Any]:
    """
    Compare two groups by fitting advanced models to their mean recovery profiles.
    
    This provides mechanistic insight into differences between groups by fitting
    sophisticated biophysical models (anomalous diffusion, reaction-diffusion) to
    the averaged recovery curves.
    
    Parameters:
    -----------
    group1_time, group1_intensity, group1_sem : arrays
        Mean recovery profile for group 1
    group2_time, group2_intensity, group2_sem : arrays
        Mean recovery profile for group 2
    group1_name, group2_name : str
        Names of the groups
    bleach_radius_um : float
        Bleach spot radius
    model : str
        Which model(s) to fit
        
    Returns:
    --------
    dict
        Comparison results including fitted parameters, interpretations, and
        parameter fold changes between groups
    """
    logger.info(f"Fitting advanced models for group comparison: {group1_name} vs {group2_name}")
    
    # Fit group 1
    fit1 = fit_mean_recovery_profile(
        group1_time, group1_intensity, group1_sem,
        bleach_radius_um, model
    )
    
    # Fit group 2
    fit2 = fit_mean_recovery_profile(
        group2_time, group2_intensity, group2_sem,
        bleach_radius_um, model
    )
    
    if not fit1['success'] or not fit2['success']:
        return {
            'success': False,
            'error': f"Fitting failed for one or both groups. Group1: {fit1.get('error')}, Group2: {fit2.get('error')}"
        }
    
    # Extract parameters for comparison
    params1 = fit1['params']
    params2 = fit2['params']
    
    # Calculate fold changes
    param_fold_changes = {}
    for param_name in params1.keys():
        if param_name in params2:
            val1 = params1[param_name]
            val2 = params2[param_name]
            if val1 != 0:
                fold_change = val2 / val1
                param_fold_changes[param_name] = {
                    f'{group1_name}': val1,
                    f'{group2_name}': val2,
                    'fold_change': fold_change,
                    'percent_change': (fold_change - 1) * 100
                }
    
    # Extract interpretations
    interp1 = fit1.get('interpretation', {})
    interp2 = fit2.get('interpretation', {})
    
    # Compare key metrics
    metric_comparison = {}
    for key in interp1.keys():
        if key in interp2:
            val1 = interp1[key]
            val2 = interp2[key]
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                metric_comparison[key] = {
                    f'{group1_name}': val1,
                    f'{group2_name}': val2,
                    'difference': val2 - val1,
                    'fold_change': val2 / val1 if val1 != 0 else np.inf
                }
    
    # Generate interpretation
    interpretation = _generate_group_comparison_interpretation(
        fit1, fit2, group1_name, group2_name, metric_comparison
    )
    
    return {
        'success': True,
        'group1_name': group1_name,
        'group2_name': group2_name,
        'model_used': fit1['model_name'],
        'group1_fit': fit1,
        'group2_fit': fit2,
        'parameter_comparison': param_fold_changes,
        'metric_comparison': metric_comparison,
        'interpretation': interpretation,
        'r2_group1': fit1.get('r2', np.nan),
        'r2_group2': fit2.get('r2', np.nan)
    }


def _generate_group_comparison_interpretation(fit1: Dict, fit2: Dict,
                                              name1: str, name2: str,
                                              metrics: Dict) -> str:
    """Generate biological interpretation of group comparison."""
    lines = []
    
    model_name = fit1['model_name']
    lines.append(f"### Advanced Model Comparison: {name1} vs {name2}\n")
    lines.append(f"**Model:** {model_name}\n")
    lines.append(f"**R² values:** {name1} = {fit1.get('r2', 0):.3f}, {name2} = {fit2.get('r2', 0):.3f}\n")
    
    # Model-specific interpretations
    if 'anomalous' in model_name:
        lines.append("\n#### Anomalous Diffusion Analysis")
        
        if 'mobile_fraction' in metrics:
            mf = metrics['mobile_fraction']
            lines.append(f"- **Mobile Fraction:** {name1} = {mf[name1]:.1f}%, {name2} = {mf[name2]:.1f}% "
                        f"(Δ = {mf['difference']:.1f}%)")
        
        if 'beta' in metrics:
            beta = metrics['beta']
            lines.append(f"- **Anomalous Exponent (β):** {name1} = {beta[name1]:.3f}, {name2} = {beta[name2]:.3f}")
            if beta['difference'] < -0.1:
                lines.append(f"  → {name2} shows more hindered diffusion (increased subdiffusion)")
            elif beta['difference'] > 0.1:
                lines.append(f"  → {name2} shows less hindered diffusion (closer to Brownian)")
        
        if 'tau' in metrics:
            tau = metrics['tau']
            lines.append(f"- **Characteristic Time (τ):** {name1} = {tau[name1]:.2f}s, {name2} = {tau[name2]:.2f}s "
                        f"({tau['fold_change']:.2f}x)")
        
        if 'effective_D' in metrics:
            D = metrics['effective_D']
            lines.append(f"- **Effective Diffusion Coefficient:** {name1} = {D[name1]:.3f} µm²/s, "
                        f"{name2} = {D[name2]:.3f} µm²/s ({D['fold_change']:.2f}x)")
    
    elif 'reaction_diffusion' in model_name:
        lines.append("\n#### Reaction-Diffusion Analysis")
        
        if 'mobile_fraction' in metrics:
            mf = metrics['mobile_fraction']
            lines.append(f"- **Mobile Fraction:** {name1} = {mf[name1]:.1f}%, {name2} = {mf[name2]:.1f}% "
                        f"(Δ = {mf['difference']:.1f}%)")
        
        if 'bound_fraction' in metrics:
            bf = metrics['bound_fraction']
            lines.append(f"- **Bound Fraction:** {name1} = {bf[name1]:.1f}%, {name2} = {bf[name2]:.1f}% "
                        f"(Δ = {bf['difference']:.1f}%)")
            if bf['difference'] < -10:
                lines.append(f"  → {name2} shows reduced binding to chromatin")
            elif bf['difference'] > 10:
                lines.append(f"  → {name2} shows enhanced binding to chromatin")
        
        if 'effective_k' in metrics:
            k = metrics['effective_k']
            lines.append(f"- **Effective Rate (k_eff):** {name1} = {k[name1]:.4f} s⁻¹, "
                        f"{name2} = {k[name2]:.4f} s⁻¹ ({k['fold_change']:.2f}x)")
        
        if 'residence_time' in metrics:
            rt = metrics['residence_time']
            lines.append(f"- **Residence Time:** {name1} = {rt[name1]:.2f}s, {name2} = {rt[name2]:.2f}s")
        
        if 'k_on' in metrics and 'k_off' in metrics:
            k_on = metrics['k_on']
            k_off = metrics['k_off']
            lines.append(f"- **Binding Rate (k_on):** {name1} = {k_on[name1]:.4f} s⁻¹, "
                        f"{name2} = {k_on[name2]:.4f} s⁻¹ ({k_on['fold_change']:.2f}x)")
            lines.append(f"- **Unbinding Rate (k_off):** {name1} = {k_off[name1]:.4f} s⁻¹, "
                        f"{name2} = {k_off[name2]:.4f} s⁻¹ ({k_off['fold_change']:.2f}x)")
            
            if k_on['fold_change'] < 0.5:
                lines.append(f"  → {name2} shows reduced binding rate (slower association)")
            elif k_on['fold_change'] > 2.0:
                lines.append(f"  → {name2} shows increased binding rate (faster association)")
            
            if k_off['fold_change'] < 0.5:
                lines.append(f"  → {name2} shows reduced unbinding rate (more stable binding)")
            elif k_off['fold_change'] > 2.0:
                lines.append(f"  → {name2} shows increased unbinding rate (less stable binding)")
    
    lines.append("\n#### Biological Implications")
    
    # Overall assessment based on mobile fraction and model-specific parameters
    if 'mobile_fraction' in metrics:
        mf_diff = metrics['mobile_fraction']['difference']
        if abs(mf_diff) > 10:
            if mf_diff > 0:
                lines.append(f"- {name2} shows **increased overall mobility** compared to {name1}")
            else:
                lines.append(f"- {name2} shows **decreased overall mobility** compared to {name1}")
    
    # Model-specific biological interpretation
    if 'bound_fraction' in metrics:
        bf_diff = metrics['bound_fraction']['difference']
        if bf_diff < -15:
            lines.append(f"- **Loss of chromatin binding:** {name2} appears to have lost binding capability")
            lines.append(f"  This could indicate disrupted protein-DNA interactions")
        elif bf_diff > 15:
            lines.append(f"- **Enhanced chromatin binding:** {name2} shows stronger association with chromatin")
            lines.append(f"  This could indicate stabilized protein-DNA interactions")
    
    if 'beta' in metrics:
        beta_diff = metrics['beta']['difference']
        if beta_diff < -0.15:
            lines.append(f"- **Increased subdiffusion:** {name2} experiences more hindered motion")
            lines.append(f"  This suggests increased molecular crowding or obstacles")
        elif beta_diff > 0.15:
            lines.append(f"- **Decreased subdiffusion:** {name2} experiences less hindered motion")
            lines.append(f"  This suggests reduced molecular crowding or obstacles")
    
    return "\n".join(lines)
