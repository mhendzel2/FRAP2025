"""
FRAP Curve Fitting Module
Robust single and double exponential fitting with model selection
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy.optimize import least_squares, curve_fit
from joblib import Parallel, delayed
import logging

logger = logging.getLogger(__name__)


@dataclass
class FitResult:
    """Results from exponential curve fitting"""
    # Primary parameters
    A: float  # Plateau intensity
    B: float  # Amplitude
    k: float  # Recovery rate
    
    # Additional parameters (for 2-exp)
    k2: Optional[float] = None
    B2: Optional[float] = None
    
    # Covariance and quality metrics
    cov: Optional[np.ndarray] = None
    r2: float = 0.0
    sse: float = np.inf
    
    # Model info
    n_params: int = 3
    n_points: int = 0
    
    # Derived parameters
    I0: float = np.nan
    I_inf: float = np.nan
    t_half: float = np.nan
    mobile_frac: float = np.nan
    
    # Model selection
    aic: float = np.inf
    bic: float = np.inf
    fit_method: str = "1exp"
    
    # QC
    success: bool = False
    message: str = ""


def model_1exp(t: np.ndarray, A: float, B: float, k: float) -> np.ndarray:
    """Single exponential: I(t) = A - B*exp(-k*t)"""
    return A - B * np.exp(-k * t)


def model_2exp(
    t: np.ndarray, 
    A: float, 
    B1: float, 
    k1: float, 
    B2: float, 
    k2: float
) -> np.ndarray:
    """Double exponential: I(t) = A - B1*exp(-k1*t) - B2*exp(-k2*t)"""
    return A - B1 * np.exp(-k1 * t) - B2 * np.exp(-k2 * t)


def residuals_1exp(params: np.ndarray, t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Residuals for 1-exp model"""
    A, B, k = params
    return model_1exp(t, A, B, k) - y


def residuals_2exp(params: np.ndarray, t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Residuals for 2-exp model"""
    A, B1, k1, B2, k2 = params
    return model_2exp(t, A, B1, k1, B2, k2) - y


def fit_recovery(
    t: np.ndarray, 
    y: np.ndarray,
    robust: bool = True,
    bounds_k: tuple[float, float] = (1e-6, 10.0)
) -> FitResult:
    """
    Fit single exponential recovery curve with robust loss
    
    Parameters
    ----------
    t : np.ndarray
        Time points (should start from 0 at bleach)
    y : np.ndarray
        Intensity values
    robust : bool
        Use soft_l1 loss for robustness
    bounds_k : tuple[float, float]
        Bounds for rate constant k
        
    Returns
    -------
    FitResult
        Fit results and derived parameters
    """
    if len(t) < 4 or len(y) < 4:
        return FitResult(
            A=np.nan, B=np.nan, k=np.nan,
            n_points=len(t),
            success=False,
            message="Insufficient data points"
        )
    
    # Initial parameter guess
    y_min = np.min(y)
    y_max = np.max(y)
    y_end = np.mean(y[-3:])  # Last 3 points
    
    A_init = y_end
    B_init = y_end - y_min
    k_init = 0.3  # Reasonable default
    
    p0 = [A_init, B_init, k_init]
    
    # Parameter bounds
    bounds_lower = [y_min * 0.8, 0, bounds_k[0]]
    bounds_upper = [y_max * 1.2, y_max * 2, bounds_k[1]]
    
    try:
        if robust:
            # Robust fitting with soft_l1 loss
            result = least_squares(
                residuals_1exp,
                p0,
                args=(t, y),
                bounds=(bounds_lower, bounds_upper),
                loss='soft_l1',
                f_scale=0.1,
                max_nfev=1000
            )
            
            popt = result.x
            success = result.success
            message = result.message
            
            # Compute covariance (approximate)
            try:
                # Jacobian from least_squares
                J = result.jac
                cov = np.linalg.inv(J.T @ J) * (result.fun ** 2).mean()
            except:
                cov = None
                
        else:
            # Standard curve_fit
            popt, pcov = curve_fit(
                model_1exp,
                t,
                y,
                p0=p0,
                bounds=(bounds_lower, bounds_upper),
                maxfev=1000
            )
            cov = pcov
            success = True
            message = "Standard fit"
        
        A, B, k = popt
        
        # Compute residuals
        y_pred = model_1exp(t, A, B, k)
        residuals = y - y_pred
        sse = np.sum(residuals ** 2)
        
        # R-squared
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (sse / ss_tot) if ss_tot > 0 else 0.0
        
        # AIC and BIC
        n = len(t)
        n_params = 3
        aic = n * np.log(sse / n) + 2 * n_params
        bic = n * np.log(sse / n) + n_params * np.log(n)
        
        # Derived parameters
        I0 = A - B  # Intensity at t=0
        I_inf = A
        t_half = np.log(2) / k if k > 0 else np.inf
        
        return FitResult(
            A=A, B=B, k=k,
            cov=cov,
            r2=r2,
            sse=sse,
            n_params=n_params,
            n_points=n,
            I0=I0,
            I_inf=I_inf,
            t_half=t_half,
            aic=aic,
            bic=bic,
            fit_method="1exp",
            success=success,
            message=message
        )
        
    except Exception as e:
        logger.debug(f"1-exp fit failed: {e}")
        return FitResult(
            A=np.nan, B=np.nan, k=np.nan,
            n_points=len(t),
            success=False,
            message=str(e)
        )


def fit_recovery_2exp(
    t: np.ndarray, 
    y: np.ndarray,
    robust: bool = True
) -> FitResult:
    """
    Fit double exponential recovery curve
    
    Parameters
    ----------
    t : np.ndarray
        Time points
    y : np.ndarray
        Intensity values
    robust : bool
        Use soft_l1 loss
        
    Returns
    -------
    FitResult
        Fit results for 2-exponential model
    """
    if len(t) < 6:
        return FitResult(
            A=np.nan, B=np.nan, k=np.nan,
            n_points=len(t),
            success=False,
            message="Insufficient points for 2-exp"
        )
    
    # Initial guess - start from 1-exp result
    fit_1exp = fit_recovery(t, y, robust=False)
    
    if not fit_1exp.success:
        return FitResult(
            A=np.nan, B=np.nan, k=np.nan,
            n_points=len(t),
            success=False,
            message="1-exp failed, cannot initialize 2-exp"
        )
    
    # Split amplitude between two components
    A_init = fit_1exp.A
    B1_init = fit_1exp.B * 0.6
    B2_init = fit_1exp.B * 0.4
    k1_init = fit_1exp.k * 0.5  # Slower
    k2_init = fit_1exp.k * 2.0  # Faster
    
    p0 = [A_init, B1_init, k1_init, B2_init, k2_init]
    
    # Bounds
    y_min, y_max = np.min(y), np.max(y)
    bounds_lower = [y_min * 0.8, 0, 1e-6, 0, 1e-6]
    bounds_upper = [y_max * 1.2, y_max * 2, 10, y_max * 2, 10]
    
    try:
        if robust:
            result = least_squares(
                residuals_2exp,
                p0,
                args=(t, y),
                bounds=(bounds_lower, bounds_upper),
                loss='soft_l1',
                f_scale=0.1,
                max_nfev=2000
            )
            popt = result.x
            success = result.success
            message = result.message
            
            try:
                J = result.jac
                cov = np.linalg.inv(J.T @ J) * (result.fun ** 2).mean()
            except:
                cov = None
        else:
            popt, pcov = curve_fit(
                model_2exp,
                t,
                y,
                p0=p0,
                bounds=(bounds_lower, bounds_upper),
                maxfev=2000
            )
            cov = pcov
            success = True
            message = "Standard 2-exp fit"
        
        A, B1, k1, B2, k2 = popt
        
        # Compute metrics
        y_pred = model_2exp(t, A, B1, k1, B2, k2)
        residuals = y - y_pred
        sse = np.sum(residuals ** 2)
        
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (sse / ss_tot) if ss_tot > 0 else 0.0
        
        # AIC and BIC
        n = len(t)
        n_params = 5
        aic = n * np.log(sse / n) + 2 * n_params
        bic = n * np.log(sse / n) + n_params * np.log(n)
        
        # Derived parameters (use slower component)
        I0 = A - B1 - B2
        I_inf = A
        t_half = np.log(2) / k1 if k1 > 0 else np.inf
        
        return FitResult(
            A=A, B=B1, k=k1,
            k2=k2, B2=B2,
            cov=cov,
            r2=r2,
            sse=sse,
            n_params=n_params,
            n_points=n,
            I0=I0,
            I_inf=I_inf,
            t_half=t_half,
            aic=aic,
            bic=bic,
            fit_method="2exp",
            success=success,
            message=message
        )
        
    except Exception as e:
        logger.debug(f"2-exp fit failed: {e}")
        return FitResult(
            A=np.nan, B=np.nan, k=np.nan,
            n_points=len(t),
            success=False,
            message=str(e)
        )


def select_best_model(
    fit_1exp: FitResult, 
    fit_2exp: FitResult,
    bic_threshold: float = 10.0
) -> FitResult:
    """
    Select best model using BIC criterion
    
    Parameters
    ----------
    fit_1exp : FitResult
        1-exponential fit
    fit_2exp : FitResult
        2-exponential fit
    bic_threshold : float
        Minimum BIC improvement to prefer 2-exp
        
    Returns
    -------
    FitResult
        Best fit
    """
    # Prefer simpler model unless 2-exp is significantly better
    if not fit_1exp.success:
        return fit_2exp if fit_2exp.success else fit_1exp
    
    if not fit_2exp.success:
        return fit_1exp
    
    delta_bic = fit_1exp.bic - fit_2exp.bic
    
    if delta_bic > bic_threshold:
        logger.debug(f"Selected 2-exp (ΔBIC = {delta_bic:.2f})")
        return fit_2exp
    else:
        logger.debug(f"Selected 1-exp (ΔBIC = {delta_bic:.2f})")
        return fit_1exp


def fit_with_model_selection(
    t: np.ndarray, 
    y: np.ndarray,
    try_2exp: bool = True,
    bic_threshold: float = 10.0,
    robust: bool = True
) -> FitResult:
    """
    Fit with automatic model selection
    
    Parameters
    ----------
    t : np.ndarray
        Time points
    y : np.ndarray
        Intensity values
    try_2exp : bool
        Whether to try 2-exponential model
    bic_threshold : float
        BIC threshold for model selection
    robust : bool
        Use robust fitting
        
    Returns
    -------
    FitResult
        Best fit result
    """
    # Fit 1-exp
    fit_1 = fit_recovery(t, y, robust=robust)
    
    if not try_2exp or len(t) < 6:
        return fit_1
    
    # Fit 2-exp
    fit_2 = fit_recovery_2exp(t, y, robust=robust)
    
    # Select best
    return select_best_model(fit_1, fit_2, bic_threshold)


def compute_mobile_fraction(
    I0: float,
    I_inf: float,
    pre_bleach: float
) -> float:
    """
    Compute mobile fraction
    
    Parameters
    ----------
    I0 : float
        Intensity immediately after bleach
    I_inf : float
        Plateau intensity
    pre_bleach : float
        Pre-bleach intensity
        
    Returns
    -------
    float
        Mobile fraction
    """
    denominator = pre_bleach - I0
    
    if abs(denominator) < 1e-10:
        logger.warning("Near-zero denominator in mobile fraction")
        return np.nan
    
    mobile_frac = (I_inf - I0) / denominator
    
    # Clamp to [0, 1]
    mobile_frac = np.clip(mobile_frac, 0.0, 1.0)
    
    return mobile_frac


def fit_cell_parallel(
    cell_data: tuple[np.ndarray, np.ndarray, float, str],
    robust: bool = True,
    try_2exp: bool = True
) -> tuple[str, FitResult]:
    """
    Fit single cell (for parallel processing)
    
    Parameters
    ----------
    cell_data : tuple
        (t, y, pre_bleach, cell_id)
    robust : bool
        Use robust fitting
    try_2exp : bool
        Try 2-exponential model
        
    Returns
    -------
    tuple[str, FitResult]
        (cell_id, fit_result)
    """
    t, y, pre_bleach, cell_id = cell_data
    
    # Fit curve
    fit = fit_with_model_selection(t, y, try_2exp=try_2exp, robust=robust)
    
    # Compute mobile fraction
    if fit.success:
        fit.mobile_frac = compute_mobile_fraction(fit.I0, fit.I_inf, pre_bleach)
    else:
        fit.mobile_frac = np.nan
    
    return cell_id, fit


def fit_cells_parallel(
    cell_data_list: list[tuple[np.ndarray, np.ndarray, float, str]],
    n_jobs: int = -1,
    robust: bool = True,
    try_2exp: bool = True
) -> dict[str, FitResult]:
    """
    Fit multiple cells in parallel
    
    Parameters
    ----------
    cell_data_list : list
        List of (t, y, pre_bleach, cell_id) tuples
    n_jobs : int
        Number of parallel jobs (-1 for all cores)
    robust : bool
        Use robust fitting
    try_2exp : bool
        Try 2-exponential model
        
    Returns
    -------
    dict[str, FitResult]
        Dictionary mapping cell_id to FitResult
    """
    logger.info(f"Fitting {len(cell_data_list)} cells in parallel with {n_jobs} jobs")
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(fit_cell_parallel)(cell_data, robust, try_2exp)
        for cell_data in cell_data_list
    )
    
    return dict(results)


def compute_fit_diagnostics(fit: FitResult, t: np.ndarray, y: np.ndarray) -> dict:
    """
    Compute diagnostic statistics for a fit
    
    Parameters
    ----------
    fit : FitResult
        Fit result
    t : np.ndarray
        Time points
    y : np.ndarray
        Observed values
        
    Returns
    -------
    dict
        Diagnostic statistics
    """
    if not fit.success:
        return {'valid': False}
    
    # Predicted values
    if fit.fit_method == "1exp":
        y_pred = model_1exp(t, fit.A, fit.B, fit.k)
    elif fit.fit_method == "2exp":
        y_pred = model_2exp(t, fit.A, fit.B, fit.k, fit.B2, fit.k2)
    else:
        return {'valid': False}
    
    # Residuals
    residuals = y - y_pred
    
    # Normalized residuals
    rmse = np.sqrt(np.mean(residuals ** 2))
    norm_residuals = residuals / rmse if rmse > 0 else residuals
    
    # Durbin-Watson statistic (autocorrelation test)
    dw = np.sum(np.diff(residuals) ** 2) / np.sum(residuals ** 2)
    
    # Mean absolute percentage error
    mape = np.mean(np.abs(residuals / (y + 1e-10))) * 100
    
    return {
        'valid': True,
        'rmse': rmse,
        'max_abs_residual': np.max(np.abs(residuals)),
        'durbin_watson': dw,
        'mape': mape,
        'mean_residual': np.mean(residuals),
        'std_residual': np.std(residuals)
    }
