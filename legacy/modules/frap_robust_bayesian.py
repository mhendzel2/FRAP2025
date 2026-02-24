"""
Advanced FRAP Fitting Methods
Includes robust fitting (outlier-resistant) and Bayesian fitting (uncertainty quantification)
"""

import numpy as np
import logging
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from scipy.optimize import least_squares, differential_evolution
from scipy.stats import norm, t as t_dist
import warnings

logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    import emcee
    EMCEE_AVAILABLE = True
except ImportError:
    EMCEE_AVAILABLE = False
    logger.warning("emcee not available - Bayesian fitting will be limited")

try:
    from scipy.optimize import curve_fit
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


@dataclass
class AdvancedFitResult:
    """Results from advanced fitting methods"""
    # Best-fit parameters
    params: Dict[str, float]
    params_std: Dict[str, float]  # Standard errors
    
    # Uncertainty quantification (for Bayesian)
    params_median: Optional[Dict[str, float]] = None
    params_lower: Optional[Dict[str, float]] = None  # 95% CI lower
    params_upper: Optional[Dict[str, float]] = None  # 95% CI upper
    
    # Quality metrics
    r2: float = 0.0
    rmse: float = np.inf
    aic: float = np.inf
    bic: float = np.inf
    
    # Fit information
    fitted_values: np.ndarray = None
    residuals: np.ndarray = None
    n_outliers: int = 0
    outlier_mask: np.ndarray = None
    
    # Method info
    method: str = "unknown"
    success: bool = False
    message: str = ""
    
    # MCMC specific (for Bayesian)
    samples: Optional[np.ndarray] = None
    acceptance_fraction: Optional[float] = None
    autocorr_time: Optional[float] = None


# ============================================================================
# Model Functions
# ============================================================================

def single_exp_model(t: np.ndarray, A: float, C: float, k: float) -> np.ndarray:
    """
    Single exponential FRAP model: I(t) = A*(1 - exp(-k*t)) + C
    
    Parameters:
    -----------
    t : Time points
    A : Recovery amplitude (mobile fraction)
    C : Immobile fraction (plateau offset)
    k : Recovery rate constant
    """
    return A * (1 - np.exp(-k * t)) + C


def double_exp_model(t: np.ndarray, A1: float, k1: float, A2: float, k2: float, C: float) -> np.ndarray:
    """
    Double exponential FRAP model: I(t) = A1*(1-exp(-k1*t)) + A2*(1-exp(-k2*t)) + C
    
    Parameters:
    -----------
    t : Time points
    A1, A2 : Recovery amplitudes for fast/slow components
    k1, k2 : Recovery rate constants (k1 > k2 for fast/slow)
    C : Immobile fraction
    """
    return A1 * (1 - np.exp(-k1 * t)) + A2 * (1 - np.exp(-k2 * t)) + C


# ============================================================================
# Robust Fitting (M-estimators with iterative reweighting)
# ============================================================================

def huber_loss(residuals: np.ndarray, delta: float = 1.0) -> np.ndarray:
    """
    Huber loss function - quadratic for small residuals, linear for large
    
    Parameters:
    -----------
    residuals : Residual values
    delta : Threshold for switching from quadratic to linear
    """
    abs_res = np.abs(residuals)
    return np.where(abs_res <= delta, 
                   0.5 * residuals**2, 
                   delta * (abs_res - 0.5 * delta))


def tukey_biweight(residuals: np.ndarray, c: float = 4.685) -> np.ndarray:
    """
    Tukey's biweight loss function - completely ignores large outliers
    
    Parameters:
    -----------
    residuals : Residual values
    c : Tuning constant (4.685 gives 95% efficiency for normal data)
    """
    u = residuals / c
    abs_u = np.abs(u)
    return np.where(abs_u <= 1,
                   (c**2 / 6) * (1 - (1 - u**2)**3),
                   c**2 / 6)


def compute_weights_huber(residuals: np.ndarray, delta: float = 1.0) -> np.ndarray:
    """Compute Huber weights for iteratively reweighted least squares"""
    abs_res = np.abs(residuals)
    return np.where(abs_res <= delta, 1.0, delta / abs_res)


def compute_weights_tukey(residuals: np.ndarray, c: float = 4.685) -> np.ndarray:
    """Compute Tukey biweight weights for IRLS"""
    u = residuals / c
    abs_u = np.abs(u)
    return np.where(abs_u <= 1, (1 - u**2)**2, 0)


def robust_fit_single_exp(t: np.ndarray, intensity: np.ndarray, 
                          loss_type: str = 'huber',
                          max_iter: int = 50,
                          tol: float = 1e-6) -> AdvancedFitResult:
    """
    Robust single exponential fitting using M-estimators
    
    Parameters:
    -----------
    t : Time points (starting from 0)
    intensity : Normalized intensity values
    loss_type : 'huber' or 'tukey' or 'soft_l1'
    max_iter : Maximum iterations for IRLS
    tol : Convergence tolerance
    
    Returns:
    --------
    AdvancedFitResult with fitted parameters and outlier detection
    """
    if len(t) < 4:
        return AdvancedFitResult(
            params={}, params_std={},
            method=f"robust_{loss_type}",
            success=False,
            message="Insufficient data points"
        )
    
    # Initial parameter estimates
    y_min = np.min(intensity)
    y_max = np.max(intensity)
    y_end = np.median(intensity[-5:]) if len(intensity) >= 5 else intensity[-1]
    
    A_init = y_end - y_min  # Mobile fraction amplitude
    C_init = y_min  # Immobile fraction
    k_init = 0.2  # Initial rate guess
    
    # Ensure initial guesses are within reasonable ranges
    A_init = np.clip(A_init, 0.01, 1.2)
    C_init = np.clip(C_init, 0, 0.95)
    k_init = np.clip(k_init, 0.01, 5.0)
    
    # Bounds - make them wider to accommodate various data
    bounds = ([0, 0, 0.001], [2.0, 1.5, 10.0])  # (A, C, k)
    
    try:
        if loss_type == 'soft_l1':
            # Use scipy's built-in soft_l1 loss
            result = least_squares(
                lambda p: single_exp_model(t, *p) - intensity,
                [A_init, C_init, k_init],
                bounds=bounds,
                loss='soft_l1',
                f_scale=0.1,
                max_nfev=2000
            )
            
            params_opt = result.x
            residuals = result.fun
            success = result.success
            
        else:
            # Iteratively Reweighted Least Squares (IRLS)
            params = np.array([A_init, C_init, k_init])
            weights = np.ones(len(t))
            
            for iteration in range(max_iter):
                # Weighted least squares step
                result = least_squares(
                    lambda p: np.sqrt(weights) * (single_exp_model(t, *p) - intensity),
                    params,
                    bounds=bounds,
                    max_nfev=1000
                )
                
                params_new = result.x
                
                # Compute residuals
                fitted = single_exp_model(t, *params_new)
                residuals = intensity - fitted
                
                # Robust scale estimate (MAD)
                mad = np.median(np.abs(residuals - np.median(residuals)))
                scale = 1.4826 * mad  # Scale factor for consistency with std
                
                if scale < 1e-10:
                    scale = np.std(residuals)
                
                # Update weights based on scaled residuals
                scaled_res = residuals / scale
                
                if loss_type == 'huber':
                    weights = compute_weights_huber(scaled_res, delta=1.345)
                elif loss_type == 'tukey':
                    weights = compute_weights_tukey(scaled_res, c=4.685)
                else:
                    raise ValueError(f"Unknown loss type: {loss_type}")
                
                # Check convergence
                if np.linalg.norm(params_new - params) < tol:
                    logger.debug(f"IRLS converged in {iteration+1} iterations")
                    break
                
                params = params_new
            
            params_opt = params
            success = True
        
        # Final fit evaluation
        fitted_values = single_exp_model(t, *params_opt)
        residuals = intensity - fitted_values
        
        # Identify outliers (residuals > 2.5 sigma)
        mad = np.median(np.abs(residuals - np.median(residuals)))
        scale = 1.4826 * mad
        outlier_threshold = 2.5
        outlier_mask = np.abs(residuals) > outlier_threshold * scale
        n_outliers = np.sum(outlier_mask)
        
        # Compute quality metrics
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((intensity - np.mean(intensity))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        rmse = np.sqrt(np.mean(residuals**2))
        
        n_params = 3
        n = len(t)
        aic = n * np.log(ss_res / n) + 2 * n_params
        bic = n * np.log(ss_res / n) + n_params * np.log(n)
        
        # Estimate parameter uncertainties (approximation)
        # Use robust covariance estimate
        J = _compute_jacobian_single_exp(t, params_opt)
        robust_cov = _robust_covariance(J, residuals, scale)
        params_std = np.sqrt(np.diag(robust_cov))
        
        return AdvancedFitResult(
            params={'A': params_opt[0], 'C': params_opt[1], 'k': params_opt[2]},
            params_std={'A': params_std[0], 'C': params_std[1], 'k': params_std[2]},
            r2=r2,
            rmse=rmse,
            aic=aic,
            bic=bic,
            fitted_values=fitted_values,
            residuals=residuals,
            n_outliers=n_outliers,
            outlier_mask=outlier_mask,
            method=f"robust_{loss_type}",
            success=success,
            message=f"Converged with {n_outliers} outliers detected"
        )
        
    except Exception as e:
        logger.error(f"Robust fitting failed: {e}")
        return AdvancedFitResult(
            params={}, params_std={},
            method=f"robust_{loss_type}",
            success=False,
            message=str(e)
        )


def robust_fit_double_exp(t: np.ndarray, intensity: np.ndarray,
                          loss_type: str = 'soft_l1') -> AdvancedFitResult:
    """
    Robust double exponential fitting
    
    Parameters:
    -----------
    t : Time points
    intensity : Normalized intensity values
    loss_type : 'soft_l1', 'huber', or 'tukey'
    
    Returns:
    --------
    AdvancedFitResult with fitted parameters
    """
    if len(t) < 6:
        return AdvancedFitResult(
            params={}, params_std={},
            method=f"robust_{loss_type}_double",
            success=False,
            message="Insufficient data points for double exponential"
        )
    
    # Initial parameter estimates
    y_min = np.min(intensity)
    y_max = np.max(intensity)
    y_end = np.median(intensity[-5:]) if len(intensity) >= 5 else intensity[-1]
    
    # Start with assumption: 60% fast, 40% slow recovery
    total_amplitude = y_end - y_min
    A1_init = 0.6 * total_amplitude  # Fast component
    A2_init = 0.4 * total_amplitude  # Slow component
    C_init = y_min
    k1_init = 0.5  # Fast rate
    k2_init = 0.05  # Slow rate
    
    # Ensure initial guesses are within bounds
    A1_init = np.clip(A1_init, 0.01, 1.0)
    A2_init = np.clip(A2_init, 0.01, 1.0)
    C_init = np.clip(C_init, 0, 0.95)
    k1_init = np.clip(k1_init, 0.02, 8.0)
    k2_init = np.clip(k2_init, 0.002, 4.0)
    
    # Bounds: (A1, k1, A2, k2, C) - wider bounds for flexibility
    bounds = ([0, 0.01, 0, 0.001, 0], 
              [2.0, 10.0, 2.0, 5.0, 1.5])
    
    try:
        result = least_squares(
            lambda p: double_exp_model(t, *p) - intensity,
            [A1_init, k1_init, A2_init, k2_init, C_init],
            bounds=bounds,
            loss=loss_type,
            f_scale=0.1,
            max_nfev=3000
        )
        
        params_opt = result.x
        fitted_values = double_exp_model(t, *params_opt)
        residuals = result.fun
        
        # Ensure k1 > k2 (fast > slow)
        if params_opt[1] < params_opt[3]:
            # Swap fast/slow components
            params_opt = [params_opt[2], params_opt[3], params_opt[0], params_opt[1], params_opt[4]]
        
        # Quality metrics
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((intensity - np.mean(intensity))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        rmse = np.sqrt(np.mean(residuals**2))
        
        n_params = 5
        n = len(t)
        aic = n * np.log(ss_res / n) + 2 * n_params
        bic = n * np.log(ss_res / n) + n_params * np.log(n)
        
        # Outlier detection
        mad = np.median(np.abs(residuals - np.median(residuals)))
        scale = 1.4826 * mad
        outlier_mask = np.abs(residuals) > 2.5 * scale
        n_outliers = np.sum(outlier_mask)
        
        # Parameter uncertainties
        J = _compute_jacobian_double_exp(t, params_opt)
        robust_cov = _robust_covariance(J, residuals, scale)
        params_std = np.sqrt(np.diag(robust_cov))
        
        return AdvancedFitResult(
            params={'A1': params_opt[0], 'k1': params_opt[1], 
                   'A2': params_opt[2], 'k2': params_opt[3], 'C': params_opt[4]},
            params_std={'A1': params_std[0], 'k1': params_std[1],
                       'A2': params_std[2], 'k2': params_std[3], 'C': params_std[4]},
            r2=r2,
            rmse=rmse,
            aic=aic,
            bic=bic,
            fitted_values=fitted_values,
            residuals=residuals,
            n_outliers=n_outliers,
            outlier_mask=outlier_mask,
            method=f"robust_{loss_type}_double",
            success=result.success,
            message=f"Double exponential fit with {n_outliers} outliers"
        )
        
    except Exception as e:
        logger.error(f"Robust double exponential fitting failed: {e}")
        return AdvancedFitResult(
            params={}, params_std={},
            method=f"robust_{loss_type}_double",
            success=False,
            message=str(e)
        )


# ============================================================================
# Bayesian Fitting (MCMC with emcee)
# ============================================================================

def log_prior_single_exp(params: np.ndarray) -> float:
    """
    Log prior for single exponential parameters
    Uniform priors with physical constraints
    """
    A, C, k = params
    
    # Physical constraints
    if A < 0 or A > 1.5:  # Mobile fraction amplitude
        return -np.inf
    if C < 0 or C > 1.0:  # Immobile fraction
        return -np.inf
    if k < 0.001 or k > 10.0:  # Rate constant
        return -np.inf
    if A + C > 1.5:  # Total recovery shouldn't exceed normalized max
        return -np.inf
    
    # Uniform prior (log = 0)
    return 0.0


def log_likelihood_single_exp(params: np.ndarray, t: np.ndarray, 
                               intensity: np.ndarray, sigma: float) -> float:
    """
    Log likelihood for single exponential model
    Assumes Gaussian errors with known variance
    """
    model = single_exp_model(t, *params)
    residuals = intensity - model
    
    # Gaussian log-likelihood
    log_l = -0.5 * np.sum((residuals / sigma)**2 + np.log(2 * np.pi * sigma**2))
    
    return log_l


def log_probability_single_exp(params: np.ndarray, t: np.ndarray,
                                intensity: np.ndarray, sigma: float) -> float:
    """Log posterior = log prior + log likelihood"""
    lp = log_prior_single_exp(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_single_exp(params, t, intensity, sigma)


def bayesian_fit_single_exp(t: np.ndarray, intensity: np.ndarray,
                             n_walkers: int = 32,
                             n_steps: int = 2000,
                             n_burn: int = 500) -> AdvancedFitResult:
    """
    Bayesian single exponential fitting using MCMC (emcee)
    
    Parameters:
    -----------
    t : Time points
    intensity : Normalized intensity values
    n_walkers : Number of MCMC walkers
    n_steps : Number of MCMC steps
    n_burn : Burn-in steps to discard
    
    Returns:
    --------
    AdvancedFitResult with parameter distributions and credible intervals
    """
    if not EMCEE_AVAILABLE:
        logger.warning("emcee not installed - falling back to maximum likelihood")
        # Fallback to robust fitting
        return robust_fit_single_exp(t, intensity, loss_type='soft_l1')
    
    if len(t) < 4:
        return AdvancedFitResult(
            params={}, params_std={},
            method="bayesian_mcmc",
            success=False,
            message="Insufficient data points"
        )
    
    try:
        # Estimate measurement uncertainty from data
        # Use robust estimate from residuals of quick fit
        quick_result = robust_fit_single_exp(t, intensity, loss_type='soft_l1')
        if quick_result.success:
            sigma = quick_result.rmse
            initial_guess = [quick_result.params['A'], quick_result.params['C'], quick_result.params['k']]
        else:
            # Fallback estimates
            y_min = np.min(intensity)
            y_end = np.median(intensity[-5:]) if len(intensity) >= 5 else intensity[-1]
            sigma = 0.05  # Default noise level
            initial_guess = [y_end - y_min, y_min, 0.2]
        
        # Initialize walkers around initial guess
        n_dim = 3
        pos = initial_guess + 1e-3 * np.random.randn(n_walkers, n_dim)
        
        # Ensure initial positions satisfy constraints
        pos[:, 0] = np.clip(pos[:, 0], 0.01, 1.0)  # A
        pos[:, 1] = np.clip(pos[:, 1], 0, 0.5)      # C
        pos[:, 2] = np.clip(pos[:, 2], 0.01, 5.0)   # k
        
        # Run MCMC
        sampler = emcee.EnsembleSampler(
            n_walkers, n_dim, log_probability_single_exp,
            args=(t, intensity, sigma)
        )
        
        logger.info(f"Running MCMC: {n_walkers} walkers, {n_steps} steps...")
        sampler.run_mcmc(pos, n_steps, progress=False)
        
        # Extract samples (discard burn-in)
        samples = sampler.get_chain(discard=n_burn, flat=True)
        
        # Compute statistics
        params_median = np.median(samples, axis=0)
        params_mean = np.mean(samples, axis=0)
        params_std = np.std(samples, axis=0)
        
        # 95% credible intervals
        params_lower = np.percentile(samples, 2.5, axis=0)
        params_upper = np.percentile(samples, 97.5, axis=0)
        
        # Use median as best estimate
        fitted_values = single_exp_model(t, *params_median)
        residuals = intensity - fitted_values
        
        # Quality metrics
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((intensity - np.mean(intensity))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        rmse = np.sqrt(np.mean(residuals**2))
        
        n_params = 3
        n = len(t)
        aic = n * np.log(ss_res / n) + 2 * n_params
        bic = n * np.log(ss_res / n) + n_params * np.log(n)
        
        # MCMC diagnostics
        try:
            acceptance_fraction = np.mean(sampler.acceptance_fraction)
            autocorr_time = np.mean(sampler.get_autocorr_time(quiet=True))
        except:
            acceptance_fraction = None
            autocorr_time = None
        
        return AdvancedFitResult(
            params={'A': params_mean[0], 'C': params_mean[1], 'k': params_mean[2]},
            params_std={'A': params_std[0], 'C': params_std[1], 'k': params_std[2]},
            params_median={'A': params_median[0], 'C': params_median[1], 'k': params_median[2]},
            params_lower={'A': params_lower[0], 'C': params_lower[1], 'k': params_lower[2]},
            params_upper={'A': params_upper[0], 'C': params_upper[1], 'k': params_upper[2]},
            r2=r2,
            rmse=rmse,
            aic=aic,
            bic=bic,
            fitted_values=fitted_values,
            residuals=residuals,
            method="bayesian_mcmc",
            success=True,
            message=f"MCMC completed: {len(samples)} samples",
            samples=samples,
            acceptance_fraction=acceptance_fraction,
            autocorr_time=autocorr_time
        )
        
    except Exception as e:
        logger.error(f"Bayesian fitting failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return AdvancedFitResult(
            params={}, params_std={},
            method="bayesian_mcmc",
            success=False,
            message=str(e)
        )


# ============================================================================
# Helper Functions
# ============================================================================

def _compute_jacobian_single_exp(t: np.ndarray, params: np.ndarray, 
                                 eps: float = 1e-8) -> np.ndarray:
    """Compute numerical Jacobian for single exponential model"""
    n = len(t)
    J = np.zeros((n, 3))
    
    for i in range(3):
        params_plus = params.copy()
        params_plus[i] += eps
        f_plus = single_exp_model(t, *params_plus)
        
        params_minus = params.copy()
        params_minus[i] -= eps
        f_minus = single_exp_model(t, *params_minus)
        
        J[:, i] = (f_plus - f_minus) / (2 * eps)
    
    return J


def _compute_jacobian_double_exp(t: np.ndarray, params: np.ndarray,
                                  eps: float = 1e-8) -> np.ndarray:
    """Compute numerical Jacobian for double exponential model"""
    n = len(t)
    J = np.zeros((n, 5))
    
    for i in range(5):
        params_plus = params.copy()
        params_plus[i] += eps
        f_plus = double_exp_model(t, *params_plus)
        
        params_minus = params.copy()
        params_minus[i] -= eps
        f_minus = double_exp_model(t, *params_minus)
        
        J[:, i] = (f_plus - f_minus) / (2 * eps)
    
    return J


def _robust_covariance(J: np.ndarray, residuals: np.ndarray, 
                       scale: float) -> np.ndarray:
    """
    Compute robust covariance matrix using sandwich estimator
    
    Parameters:
    -----------
    J : Jacobian matrix
    residuals : Fit residuals
    scale : Robust scale estimate
    """
    try:
        # Sandwich estimator: (J'J)^-1 * J'WJ * (J'J)^-1
        # where W is diagonal weight matrix
        weights = compute_weights_huber(residuals / scale)
        W = np.diag(weights)
        
        JtJ = J.T @ J
        JtWJ = J.T @ W @ J
        
        JtJ_inv = np.linalg.inv(JtJ + 1e-10 * np.eye(JtJ.shape[0]))
        
        cov = JtJ_inv @ JtWJ @ JtJ_inv * scale**2
        
        return cov
    except:
        # Fallback to standard covariance
        try:
            cov = np.linalg.inv(J.T @ J) * scale**2
            return cov
        except:
            # Return identity if all else fails
            return np.eye(J.shape[1]) * scale**2


def compare_fitting_methods(t: np.ndarray, intensity: np.ndarray) -> Dict[str, AdvancedFitResult]:
    """
    Compare multiple advanced fitting methods on the same data
    
    Parameters:
    -----------
    t : Time points
    intensity : Normalized intensity values
    
    Returns:
    --------
    Dictionary of fitting results from different methods
    """
    results = {}
    
    # Robust methods
    logger.info("Fitting with robust methods...")
    results['robust_soft_l1'] = robust_fit_single_exp(t, intensity, loss_type='soft_l1')
    results['robust_huber'] = robust_fit_single_exp(t, intensity, loss_type='huber')
    results['robust_tukey'] = robust_fit_single_exp(t, intensity, loss_type='tukey')
    
    # Double exponential
    results['robust_double'] = robust_fit_double_exp(t, intensity, loss_type='soft_l1')
    
    # Bayesian (if available and requested)
    if EMCEE_AVAILABLE:
        logger.info("Fitting with Bayesian MCMC...")
        results['bayesian'] = bayesian_fit_single_exp(t, intensity, n_walkers=32, n_steps=1500, n_burn=500)
    
    return results


def get_best_method(results: Dict[str, AdvancedFitResult]) -> Tuple[str, AdvancedFitResult]:
    """
    Select best fitting method based on AIC
    
    Parameters:
    -----------
    results : Dictionary of fitting results
    
    Returns:
    --------
    (method_name, best_result)
    """
    best_method = None
    best_aic = np.inf
    
    for method, result in results.items():
        if result.success and result.aic < best_aic:
            best_aic = result.aic
            best_method = method
    
    if best_method is None:
        # Return first successful method
        for method, result in results.items():
            if result.success:
                return method, result
        # No successful fits
        return list(results.keys())[0], list(results.values())[0]
    
    return best_method, results[best_method]


# ============================================================================
# Utility Functions for Interpretation
# ============================================================================

def interpret_robust_fit(result: AdvancedFitResult, bleach_radius_um: float = None) -> Dict[str, any]:
    """
    Interpret robust fitting results in biological terms
    
    Parameters:
    -----------
    result : AdvancedFitResult object
    bleach_radius_um : Bleach spot radius in microns (optional)
    
    Returns:
    --------
    Dictionary with biological interpretations
    """
    if not result.success:
        return {'error': result.message}
    
    interp = {}
    
    # Mobile and immobile fractions
    if 'A' in result.params and 'C' in result.params:
        mobile_frac = result.params['A'] / (result.params['A'] + result.params['C']) if (result.params['A'] + result.params['C']) > 0 else 0
        immobile_frac = result.params['C'] / (result.params['A'] + result.params['C']) if (result.params['A'] + result.params['C']) > 0 else 0
        
        interp['mobile_fraction'] = mobile_frac
        interp['immobile_fraction'] = immobile_frac
        
        # Interpret mobility
        if mobile_frac > 0.9:
            interp['mobility_type'] = "Highly mobile"
        elif mobile_frac > 0.7:
            interp['mobility_type'] = "Mobile"
        elif mobile_frac > 0.5:
            interp['mobility_type'] = "Partially mobile"
        else:
            interp['mobility_type'] = "Mostly immobile"
    
    # Recovery kinetics
    if 'k' in result.params:
        k = result.params['k']
        t_half = np.log(2) / k
        
        interp['rate_constant'] = k
        interp['half_time'] = t_half
        
        # Diffusion coefficient (if radius known)
        if bleach_radius_um is not None:
            # k is the exponential recovery rate in exp(-k*t): k = 1/tau (not 1/t_half).
            D = (bleach_radius_um**2 * k) / 4.0
            interp['diffusion_coefficient'] = D
            
            # Interpret diffusion
            if D > 2.0:
                interp['diffusion_type'] = "Fast diffusion"
            elif D > 0.5:
                interp['diffusion_type'] = "Normal diffusion"
            else:
                interp['diffusion_type'] = "Slow diffusion"
    
    # For double exponential
    if 'k1' in result.params and 'k2' in result.params:
        interp['fast_rate'] = result.params['k1']
        interp['slow_rate'] = result.params['k2']
        interp['fast_amplitude'] = result.params['A1']
        interp['slow_amplitude'] = result.params['A2']
        
        # Ratio indicates heterogeneity
        rate_ratio = result.params['k1'] / result.params['k2'] if result.params['k2'] > 0 else np.inf
        interp['rate_ratio'] = rate_ratio
        
        if rate_ratio > 10:
            interp['heterogeneity'] = "Distinct populations"
        elif rate_ratio > 3:
            interp['heterogeneity'] = "Moderate heterogeneity"
        else:
            interp['heterogeneity'] = "Similar kinetics"
    
    # Quality assessment
    interp['fit_quality'] = result.r2
    interp['outliers_detected'] = result.n_outliers
    
    if result.n_outliers > len(result.residuals) * 0.1:
        interp['data_quality'] = "Many outliers detected - check data"
    elif result.n_outliers > 0:
        interp['data_quality'] = "Good (few outliers removed)"
    else:
        interp['data_quality'] = "Excellent (no outliers)"
    
    return interp
