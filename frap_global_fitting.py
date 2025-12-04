"""
FRAP Global Fitting and Unified Model Selection Module

This module implements a sophisticated approach to FRAP analysis using:
- Global fitting: Fit all replicates within a group simultaneously with shared kinetic parameters
- Unified model selection: Apply the same model across all groups for "apples-to-apples" comparison
- Statistical model comparison: F-test, AIC, BIC for rigorous model selection

Based on lmfit for robust non-linear least-squares minimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
import logging
from datetime import datetime

try:
    from lmfit import Parameters, Minimizer, minimize, report_fit
    from lmfit.minimizer import MinimizerResult
    LMFIT_AVAILABLE = True
except ImportError:
    LMFIT_AVAILABLE = False
    Parameters = None
    Minimizer = None
    MinimizerResult = None

import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class CurveData:
    """Single FRAP recovery curve data"""
    curve_id: str
    group_id: str
    time: np.ndarray
    intensity: np.ndarray
    
    def __post_init__(self):
        self.time = np.asarray(self.time, dtype=np.float64)
        self.intensity = np.asarray(self.intensity, dtype=np.float64)


@dataclass
class GlobalFitResult:
    """Results from a global fitting procedure"""
    group_id: str
    model_name: str
    success: bool
    message: str
    
    # Fit quality metrics
    chisqr: float = np.nan
    redchi: float = np.nan
    aic: float = np.nan
    bic: float = np.nan
    ndata: int = 0
    nvarys: int = 0
    nfree: int = 0
    
    # Parameters: {param_name: {'value': v, 'stderr': e, 'is_global': bool}}
    parameters: Dict[str, Dict] = field(default_factory=dict)
    
    # Per-curve results
    curve_fits: Dict[str, Dict] = field(default_factory=dict)  # curve_id -> {fitted, residuals, r2}
    
    # Confidence intervals (if computed)
    confidence_intervals: Dict[str, Dict] = field(default_factory=dict)
    
    # Raw lmfit result for advanced access
    lmfit_result: Any = None


@dataclass
class ModelComparisonResult:
    """Results from comparing two models"""
    simple_model: str
    complex_model: str
    f_statistic: float
    p_value: float
    delta_aic: float  # AIC_simple - AIC_complex (positive = complex better)
    delta_bic: float  # BIC_simple - BIC_complex
    preferred_model: str
    significance_level: float = 0.05


@dataclass 
class UnifiedAnalysisResult:
    """Complete results from unified model selection workflow"""
    # Phase 1: Per-group model exploration
    group_explorations: Dict[str, Dict[str, GlobalFitResult]] = field(default_factory=dict)
    group_preferred_models: Dict[str, str] = field(default_factory=dict)
    model_selection_table: Optional[pd.DataFrame] = None
    
    # Phase 2: Unified model decision
    unified_model: str = ""
    unification_rationale: str = ""
    
    # Phase 3: Final comparative fits
    final_fits: Dict[str, GlobalFitResult] = field(default_factory=dict)
    comparison_table: Optional[pd.DataFrame] = None
    
    # Metadata
    timestamp: str = ""
    groups_analyzed: List[str] = field(default_factory=list)


# =============================================================================
# FRAP MODEL LIBRARY
# =============================================================================

def model_single_exponential(t: np.ndarray, Mf: float, k: float) -> np.ndarray:
    """
    Single exponential recovery model.
    
    F(t) = Mf * (1 - exp(-k * t))
    
    Parameters
    ----------
    t : array
        Time points (starting from 0 at bleach)
    Mf : float
        Mobile fraction (plateau amplitude, 0-1 for normalized data)
    k : float
        Recovery rate constant (1/s)
        
    Returns
    -------
    array
        Predicted normalized intensity
    """
    return Mf * (1.0 - np.exp(-k * t))


def model_double_exponential(t: np.ndarray, Mf: float, F_fast: float, 
                              k_fast: float, k_slow: float) -> np.ndarray:
    """
    Double exponential recovery model for two kinetic subpopulations.
    
    F(t) = Mf * [F_fast * (1 - exp(-k_fast * t)) + (1 - F_fast) * (1 - exp(-k_slow * t))]
    
    Parameters
    ----------
    t : array
        Time points
    Mf : float
        Total mobile fraction
    F_fast : float
        Fraction of mobile pool in fast component (0-1)
    k_fast : float
        Fast rate constant (must be > k_slow)
    k_slow : float
        Slow rate constant
        
    Returns
    -------
    array
        Predicted normalized intensity
    """
    recovery_fast = F_fast * (1.0 - np.exp(-k_fast * t))
    recovery_slow = (1.0 - F_fast) * (1.0 - np.exp(-k_slow * t))
    return Mf * (recovery_fast + recovery_slow)


def model_triple_exponential(t: np.ndarray, Mf: float, F1: float, F2: float,
                              k1: float, k2: float, k3: float) -> np.ndarray:
    """
    Triple exponential recovery model for three kinetic subpopulations.
    
    F(t) = Mf * [F1*(1-exp(-k1*t)) + F2*(1-exp(-k2*t)) + (1-F1-F2)*(1-exp(-k3*t))]
    
    Parameters
    ----------
    t : array
        Time points
    Mf : float
        Total mobile fraction
    F1, F2 : float
        Fractions of mobile pool (F3 = 1 - F1 - F2)
    k1, k2, k3 : float
        Rate constants (k1 > k2 > k3 by convention)
    """
    F3 = 1.0 - F1 - F2
    return Mf * (F1 * (1.0 - np.exp(-k1 * t)) + 
                 F2 * (1.0 - np.exp(-k2 * t)) + 
                 F3 * (1.0 - np.exp(-k3 * t)))


# Model registry with metadata
MODEL_REGISTRY = {
    'single': {
        'function': model_single_exponential,
        'n_kinetic_params': 1,  # k
        'n_amplitude_params': 1,  # Mf
        'global_params': ['k'],  # Kinetics are shared
        'local_params': ['Mf'],  # Amplitude can vary per curve
        'param_names': ['Mf', 'k'],
        'description': 'Single exponential recovery'
    },
    'double': {
        'function': model_double_exponential,
        'n_kinetic_params': 3,  # k_fast, k_slow, F_fast
        'n_amplitude_params': 1,  # Mf
        'global_params': ['k_fast', 'k_slow', 'F_fast'],
        'local_params': ['Mf'],
        'param_names': ['Mf', 'F_fast', 'k_fast', 'k_slow'],
        'description': 'Double exponential (two populations)'
    },
    'triple': {
        'function': model_triple_exponential,
        'n_kinetic_params': 5,  # k1, k2, k3, F1, F2
        'n_amplitude_params': 1,
        'global_params': ['k1', 'k2', 'k3', 'F1', 'F2'],
        'local_params': ['Mf'],
        'param_names': ['Mf', 'F1', 'F2', 'k1', 'k2', 'k3'],
        'description': 'Triple exponential (three populations)'
    }
}


# =============================================================================
# GLOBAL FITTING ENGINE
# =============================================================================

class GlobalFitter:
    """
    Engine for global fitting of FRAP data.
    
    Global fitting simultaneously fits all replicates in a group while
    sharing kinetic parameters. This provides more robust parameter
    estimates and allows for rigorous statistical comparisons.
    """
    
    def __init__(self, mf_is_local: bool = True):
        """
        Initialize the global fitter.
        
        Parameters
        ----------
        mf_is_local : bool
            If True, mobile fraction (Mf) is fitted independently for each curve.
            If False, Mf is shared across all curves in the group.
        """
        if not LMFIT_AVAILABLE:
            raise ImportError("lmfit is required for global fitting. Install with: pip install lmfit")
        
        self.mf_is_local = mf_is_local
        self.models = MODEL_REGISTRY
    
    def _create_parameters(self, model_name: str, curve_ids: List[str],
                           initial_guesses: Optional[Dict] = None) -> Parameters:
        """
        Create lmfit Parameters object for global fitting.
        
        Parameters
        ----------
        model_name : str
            Name of the model ('single', 'double', 'triple')
        curve_ids : list
            List of curve identifiers
        initial_guesses : dict, optional
            Initial parameter values
            
        Returns
        -------
        Parameters
            lmfit Parameters object with appropriate constraints
        """
        params = Parameters()
        model_info = self.models[model_name]
        
        # Default initial guesses
        defaults = {
            'Mf': 0.8,
            'k': 0.1,
            'k_fast': 0.5,
            'k_slow': 0.05,
            'F_fast': 0.5,
            'k1': 1.0,
            'k2': 0.1,
            'k3': 0.01,
            'F1': 0.33,
            'F2': 0.33
        }
        
        if initial_guesses:
            defaults.update(initial_guesses)
        
        # Add global kinetic parameters
        if model_name == 'single':
            params.add('k_global', value=defaults['k'], min=1e-6, max=50)
            
        elif model_name == 'double':
            # Use delta parameterization to ensure k_fast > k_slow
            params.add('k_slow_global', value=defaults['k_slow'], min=1e-6, max=10)
            params.add('delta_k', value=defaults['k_fast'] - defaults['k_slow'], min=1e-6)
            params.add('k_fast_global', expr='k_slow_global + delta_k')
            params.add('F_fast_global', value=defaults['F_fast'], min=0.01, max=0.99)
            
        elif model_name == 'triple':
            # Enforce k1 > k2 > k3 using delta parameterization
            params.add('k3_global', value=defaults['k3'], min=1e-6, max=5)
            params.add('delta_k23', value=defaults['k2'] - defaults['k3'], min=1e-6)
            params.add('k2_global', expr='k3_global + delta_k23')
            params.add('delta_k12', value=defaults['k1'] - defaults['k2'], min=1e-6)
            params.add('k1_global', expr='k2_global + delta_k12')
            # Fractions must sum to < 1
            params.add('F1_global', value=defaults['F1'], min=0.01, max=0.98)
            params.add('F2_global', value=defaults['F2'], min=0.01, 
                      expr=f'min(0.98 - F1_global, {defaults["F2"]})')
        
        # Add Mf parameters (local or global)
        if self.mf_is_local:
            for cid in curve_ids:
                safe_cid = self._sanitize_id(cid)
                params.add(f'Mf_{safe_cid}', value=defaults['Mf'], min=0.01, max=1.5)
        else:
            params.add('Mf_global', value=defaults['Mf'], min=0.01, max=1.5)
        
        return params
    
    def _sanitize_id(self, curve_id: str) -> str:
        """Sanitize curve ID for use as parameter name"""
        return str(curve_id).replace('-', '_').replace('.', '_').replace(' ', '_')
    
    def _global_objective(self, params: Parameters, datasets: Dict[str, Tuple[np.ndarray, np.ndarray]],
                          model_name: str) -> np.ndarray:
        """
        Global objective function for minimization.
        
        Calculates concatenated residuals across all curves.
        
        Parameters
        ----------
        params : Parameters
            lmfit Parameters object
        datasets : dict
            {curve_id: (time_array, intensity_array)}
        model_name : str
            Model name
            
        Returns
        -------
        array
            Concatenated residuals from all curves
        """
        all_residuals = []
        pvals = params.valuesdict()
        model_func = self.models[model_name]['function']
        
        for curve_id, (t_data, y_data) in datasets.items():
            safe_cid = self._sanitize_id(curve_id)
            
            # Get Mf (local or global)
            if self.mf_is_local:
                Mf = pvals.get(f'Mf_{safe_cid}', 0.8)
            else:
                Mf = pvals.get('Mf_global', 0.8)
            
            # Calculate prediction based on model
            if model_name == 'single':
                k = pvals['k_global']
                y_pred = model_func(t_data, Mf, k)
                
            elif model_name == 'double':
                k_fast = pvals['k_fast_global']
                k_slow = pvals['k_slow_global']
                F_fast = pvals['F_fast_global']
                y_pred = model_func(t_data, Mf, F_fast, k_fast, k_slow)
                
            elif model_name == 'triple':
                k1 = pvals['k1_global']
                k2 = pvals['k2_global']
                k3 = pvals['k3_global']
                F1 = pvals['F1_global']
                F2 = pvals['F2_global']
                y_pred = model_func(t_data, Mf, F1, F2, k1, k2, k3)
            else:
                raise ValueError(f"Unknown model: {model_name}")
            
            residuals = y_data - y_pred
            all_residuals.append(residuals)
        
        return np.concatenate(all_residuals)
    
    def fit_group(self, curves: List[CurveData], model_name: str,
                  initial_guesses: Optional[Dict] = None,
                  calc_confidence: bool = False) -> GlobalFitResult:
        """
        Perform global fitting on a group of curves.
        
        Parameters
        ----------
        curves : list of CurveData
            Curves to fit (should all be from same group)
        model_name : str
            Model to use ('single', 'double', 'triple')
        initial_guesses : dict, optional
            Initial parameter values
        calc_confidence : bool
            Whether to calculate confidence intervals (slower)
            
        Returns
        -------
        GlobalFitResult
            Comprehensive fit results
        """
        if model_name not in self.models:
            raise ValueError(f"Unknown model '{model_name}'. Available: {list(self.models.keys())}")
        
        if len(curves) == 0:
            return GlobalFitResult(
                group_id="",
                model_name=model_name,
                success=False,
                message="No curves provided"
            )
        
        group_id = curves[0].group_id
        curve_ids = [c.curve_id for c in curves]
        
        # Prepare datasets
        datasets = {c.curve_id: (c.time, c.intensity) for c in curves}
        
        # Create parameters
        params = self._create_parameters(model_name, curve_ids, initial_guesses)
        
        # Run minimization
        try:
            minimizer = Minimizer(
                self._global_objective,
                params,
                fcn_args=(datasets, model_name)
            )
            result = minimizer.minimize(method='leastsq')
            
            # Calculate additional metrics
            ndata = sum(len(c.intensity) for c in curves)
            nvarys = result.nvarys
            nfree = ndata - nvarys
            
            # Extract parameters
            parameters = {}
            pvals = result.params.valuesdict()
            
            for pname, param in result.params.items():
                is_global = '_global' in pname or pname.startswith('delta_')
                parameters[pname] = {
                    'value': param.value,
                    'stderr': param.stderr if param.stderr is not None else np.nan,
                    'is_global': is_global,
                    'vary': param.vary,
                    'expr': param.expr
                }
            
            # Calculate per-curve fit quality
            curve_fits = {}
            model_func = self.models[model_name]['function']
            
            for curve in curves:
                safe_cid = self._sanitize_id(curve.curve_id)
                
                # Get Mf
                if self.mf_is_local:
                    Mf = pvals.get(f'Mf_{safe_cid}', 0.8)
                else:
                    Mf = pvals.get('Mf_global', 0.8)
                
                # Calculate fitted curve
                if model_name == 'single':
                    fitted = model_func(curve.time, Mf, pvals['k_global'])
                elif model_name == 'double':
                    fitted = model_func(curve.time, Mf, pvals['F_fast_global'],
                                       pvals['k_fast_global'], pvals['k_slow_global'])
                elif model_name == 'triple':
                    fitted = model_func(curve.time, Mf, pvals['F1_global'], pvals['F2_global'],
                                       pvals['k1_global'], pvals['k2_global'], pvals['k3_global'])
                
                residuals = curve.intensity - fitted
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((curve.intensity - np.mean(curve.intensity))**2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                
                curve_fits[curve.curve_id] = {
                    'fitted': fitted,
                    'residuals': residuals,
                    'r2': r2,
                    'Mf': Mf
                }
            
            # Confidence intervals
            ci = {}
            if calc_confidence and result.success:
                try:
                    ci_result = minimizer.conf_interval()
                    for pname in ci_result:
                        ci[pname] = {
                            '95%_low': ci_result[pname][1][1],
                            '95%_high': ci_result[pname][5][1]
                        }
                except Exception as e:
                    logger.warning(f"Could not calculate confidence intervals: {e}")
            
            return GlobalFitResult(
                group_id=group_id,
                model_name=model_name,
                success=result.success,
                message=result.message if hasattr(result, 'message') else "Optimization complete",
                chisqr=result.chisqr,
                redchi=result.redchi,
                aic=result.aic,
                bic=result.bic,
                ndata=ndata,
                nvarys=nvarys,
                nfree=nfree,
                parameters=parameters,
                curve_fits=curve_fits,
                confidence_intervals=ci,
                lmfit_result=result
            )
            
        except Exception as e:
            logger.error(f"Global fitting failed: {e}")
            return GlobalFitResult(
                group_id=group_id,
                model_name=model_name,
                success=False,
                message=str(e)
            )


# =============================================================================
# MODEL SELECTION
# =============================================================================

class ModelSelector:
    """
    Statistical methods for comparing and selecting FRAP models.
    """
    
    @staticmethod
    def f_test_nested_models(result_simple: GlobalFitResult, 
                              result_complex: GlobalFitResult,
                              alpha: float = 0.05) -> ModelComparisonResult:
        """
        F-test for comparing nested models (e.g., single vs double exponential).
        
        The extra sum-of-squares F-test determines if the additional parameters
        in the complex model provide a statistically significant improvement.
        
        Parameters
        ----------
        result_simple : GlobalFitResult
            Results from simpler model (fewer parameters)
        result_complex : GlobalFitResult
            Results from complex model (more parameters)
        alpha : float
            Significance level for the test
            
        Returns
        -------
        ModelComparisonResult
            Statistical comparison results
        """
        ssr_simple = result_simple.chisqr  # Sum of squared residuals
        ssr_complex = result_complex.chisqr
        
        dof_simple = result_simple.nfree
        dof_complex = result_complex.nfree
        
        delta_dof = dof_simple - dof_complex
        
        if delta_dof <= 0:
            raise ValueError("Complex model must have more parameters (fewer DOF)")
        
        if ssr_complex <= 0 or dof_complex <= 0:
            raise ValueError("Invalid chi-squared or DOF values")
        
        # F-statistic
        F = ((ssr_simple - ssr_complex) / delta_dof) / (ssr_complex / dof_complex)
        
        # P-value (survival function of F-distribution)
        p_value = float(stats.f.sf(F, delta_dof, dof_complex))
        
        # AIC/BIC comparison
        delta_aic = result_simple.aic - result_complex.aic
        delta_bic = result_simple.bic - result_complex.bic
        
        # Determine preferred model
        # Complex model preferred if p < alpha AND (AIC or BIC supports it)
        if p_value < alpha and (delta_aic > 2 or delta_bic > 2):
            preferred = result_complex.model_name
        elif delta_aic > 10:  # Strong AIC preference for complex
            preferred = result_complex.model_name
        else:
            preferred = result_simple.model_name
        
        return ModelComparisonResult(
            simple_model=result_simple.model_name,
            complex_model=result_complex.model_name,
            f_statistic=F,
            p_value=p_value,
            delta_aic=delta_aic,
            delta_bic=delta_bic,
            preferred_model=preferred,
            significance_level=alpha
        )
    
    @staticmethod
    def compare_models_aic(results: Dict[str, GlobalFitResult]) -> Tuple[Optional[str], pd.DataFrame]:
        """
        Compare multiple models using AIC weights.
        
        Parameters
        ----------
        results : dict
            {model_name: GlobalFitResult}
            
        Returns
        -------
        tuple
            (best_model_name, comparison_dataframe)
        """
        data = []
        
        for model_name, result in results.items():
            data.append({
                'model': model_name,
                'n_params': result.nvarys,
                'chisqr': result.chisqr,
                'aic': result.aic,
                'bic': result.bic,
                'success': result.success
            })
        
        df = pd.DataFrame(data)
        
        # Filter successful fits
        df_success = df[df['success']].copy()
        
        if df_success.empty:
            return None, df
        
        # Calculate delta AIC and AIC weights
        min_aic = df_success['aic'].min()
        df_success['delta_aic'] = df_success['aic'] - min_aic
        df_success['aic_weight'] = np.exp(-0.5 * df_success['delta_aic'])
        df_success['aic_weight'] = df_success['aic_weight'] / df_success['aic_weight'].sum()
        
        # Best model has lowest AIC
        best_model = df_success.loc[df_success['aic'].idxmin(), 'model']
        
        return best_model, df_success
    
    @staticmethod
    def select_preferred_model(results: Dict[str, GlobalFitResult],
                                alpha: float = 0.05) -> Tuple[Optional[str], str]:
        """
        Select the preferred model using hierarchical testing.
        
        Uses F-tests for nested model comparisons combined with AIC/BIC.
        
        Parameters
        ----------
        results : dict
            {model_name: GlobalFitResult}
        alpha : float
            Significance level
            
        Returns
        -------
        tuple
            (preferred_model_name, rationale_string)
        """
        available = [m for m, r in results.items() if r.success]
        
        if not available:
            return None, "No successful fits"
        
        if len(available) == 1:
            return available[0], "Only one model converged"
        
        rationale_parts = []
        
        # Start with simplest model as baseline
        model_complexity = {'single': 1, 'double': 2, 'triple': 3}
        available_sorted = sorted(available, key=lambda x: model_complexity.get(x, 99))
        
        current_best = available_sorted[0]
        
        # Test against progressively more complex models
        for complex_model in available_sorted[1:]:
            simple_result = results[current_best]
            complex_result = results[complex_model]
            
            try:
                comparison = ModelSelector.f_test_nested_models(
                    simple_result, complex_result, alpha
                )
                
                rationale_parts.append(
                    f"{current_best} vs {complex_model}: F={comparison.f_statistic:.2f}, "
                    f"p={comparison.p_value:.4f}, ΔAIC={comparison.delta_aic:.1f}"
                )
                
                if comparison.preferred_model == complex_model:
                    current_best = complex_model
                    rationale_parts.append(f"  → {complex_model} significantly better")
                else:
                    rationale_parts.append(f"  → {current_best} sufficient (parsimony)")
                    
            except Exception as e:
                rationale_parts.append(f"Could not compare {current_best} vs {complex_model}: {e}")
        
        return current_best, "\n".join(rationale_parts)


# =============================================================================
# WORKFLOW ORCHESTRATOR
# =============================================================================

class UnifiedModelWorkflow:
    """
    Orchestrates the complete unified model selection workflow.
    
    Three phases:
    1. Group-level exploration: Fit all candidate models to each group
    2. Unified model selection: Identify the model required for fair comparison
    3. Final comparative analysis: Refit all groups with the unified model
    """
    
    def __init__(self, mf_is_local: bool = True, alpha: float = 0.05):
        """
        Initialize workflow.
        
        Parameters
        ----------
        mf_is_local : bool
            Whether mobile fraction is fitted per-curve
        alpha : float
            Significance level for model selection
        """
        self.fitter = GlobalFitter(mf_is_local=mf_is_local)
        self.selector = ModelSelector()
        self.alpha = alpha
        self.candidate_models = ['single', 'double']  # Can add 'triple'
    
    def prepare_data(self, df: pd.DataFrame,
                     time_col: str = 'Time',
                     intensity_col: str = 'Intensity', 
                     curve_col: str = 'CurveID',
                     group_col: str = 'GroupID') -> Dict[str, List[CurveData]]:
        """
        Prepare data from DataFrame into CurveData objects.
        
        Parameters
        ----------
        df : DataFrame
            Must contain columns for time, intensity, curve ID, and group ID
        time_col, intensity_col, curve_col, group_col : str
            Column names
            
        Returns
        -------
        dict
            {group_id: [CurveData, ...]}
        """
        grouped_data = {}
        
        for group_id in df[group_col].unique():
            group_df = df[df[group_col] == group_id]
            curves = []
            
            for curve_id in group_df[curve_col].unique():
                curve_df = group_df[group_df[curve_col] == curve_id].sort_values(time_col)
                
                curves.append(CurveData(
                    curve_id=str(curve_id),
                    group_id=str(group_id),
                    time=curve_df[time_col].values,
                    intensity=curve_df[intensity_col].values
                ))
            
            grouped_data[str(group_id)] = curves
        
        return grouped_data
    
    def normalize_curves(self, curves: List[CurveData],
                         prebleach_fraction: float = 0.1) -> List[CurveData]:
        """
        Apply double normalization to FRAP curves.
        
        Normalizes so that:
        - Average pre-bleach intensity = 1.0
        - Intensity at t=0 (post-bleach) = 0.0
        
        Parameters
        ----------
        curves : list of CurveData
            Raw curves to normalize
        prebleach_fraction : float
            Fraction of initial timepoints to consider as pre-bleach
            
        Returns
        -------
        list of CurveData
            Normalized curves
        """
        normalized = []
        
        for curve in curves:
            t = curve.time.copy()
            y = curve.intensity.copy()
            
            # Find bleach point (minimum intensity)
            bleach_idx = np.argmin(y)
            
            # Pre-bleach average
            if bleach_idx > 0:
                prebleach = np.mean(y[:bleach_idx])
            else:
                prebleach = y[0]
            
            # Post-bleach intensity at t=0
            I_0 = y[bleach_idx]
            
            # Double normalization
            if prebleach != I_0:
                y_norm = (y - I_0) / (prebleach - I_0)
            else:
                y_norm = y / prebleach if prebleach > 0 else y
            
            # Shift time so bleach = 0
            t_norm = t - t[bleach_idx]
            
            # Keep only post-bleach
            mask = t_norm >= 0
            
            normalized.append(CurveData(
                curve_id=curve.curve_id,
                group_id=curve.group_id,
                time=t_norm[mask],
                intensity=y_norm[mask]
            ))
        
        return normalized
    
    def phase1_explore_groups(self, grouped_data: Dict[str, List[CurveData]],
                               progress_callback: Optional[Callable] = None) -> Dict[str, Dict[str, GlobalFitResult]]:
        """
        Phase 1: Fit all candidate models to each group.
        
        Parameters
        ----------
        grouped_data : dict
            {group_id: [CurveData, ...]}
        progress_callback : callable, optional
            Called with (current_step, total_steps, message)
            
        Returns
        -------
        dict
            {group_id: {model_name: GlobalFitResult}}
        """
        results = {}
        total_fits = len(grouped_data) * len(self.candidate_models)
        current = 0
        
        for group_id, curves in grouped_data.items():
            results[group_id] = {}
            
            for model_name in self.candidate_models:
                if progress_callback:
                    progress_callback(current, total_fits, 
                                     f"Fitting {model_name} to {group_id}")
                
                fit_result = self.fitter.fit_group(curves, model_name)
                results[group_id][model_name] = fit_result
                current += 1
        
        return results
    
    def phase2_select_unified_model(self, 
                                     exploration_results: Dict[str, Dict[str, GlobalFitResult]]) -> Tuple[str, Dict[str, str], str]:
        """
        Phase 2: Determine the unified model for comparative analysis.
        
        The unified model is the most complex model that is statistically
        required by ANY group. This ensures "apples-to-apples" comparison.
        
        Parameters
        ----------
        exploration_results : dict
            Results from phase 1
            
        Returns
        -------
        tuple
            (unified_model_name, {group: preferred_model}, rationale)
        """
        group_preferences = {}
        rationale_parts = []
        
        model_complexity = {'single': 1, 'double': 2, 'triple': 3}
        max_required_complexity = 0
        unified_model = 'single'
        
        for group_id, model_results in exploration_results.items():
            preferred, group_rationale = self.selector.select_preferred_model(
                model_results, self.alpha
            )
            
            group_preferences[group_id] = preferred
            rationale_parts.append(f"\n{group_id}: {preferred}")
            rationale_parts.append(f"  {group_rationale}")
            
            if preferred and model_complexity.get(preferred, 0) > max_required_complexity:
                max_required_complexity = model_complexity[preferred]
                unified_model = preferred
        
        rationale_parts.append(f"\n→ UNIFIED MODEL: {unified_model}")
        rationale_parts.append(f"  (Most complex model required by any group)")
        
        return unified_model, group_preferences, "\n".join(rationale_parts)
    
    def phase3_final_analysis(self, grouped_data: Dict[str, List[CurveData]],
                               unified_model: str,
                               calc_confidence: bool = True) -> Dict[str, GlobalFitResult]:
        """
        Phase 3: Refit all groups with the unified model.
        
        Parameters
        ----------
        grouped_data : dict
            {group_id: [CurveData, ...]}
        unified_model : str
            Model to use for all groups
        calc_confidence : bool
            Whether to compute confidence intervals
            
        Returns
        -------
        dict
            {group_id: GlobalFitResult}
        """
        final_results = {}
        
        for group_id, curves in grouped_data.items():
            final_results[group_id] = self.fitter.fit_group(
                curves, unified_model, calc_confidence=calc_confidence
            )
        
        return final_results
    
    def run_full_analysis(self, df: pd.DataFrame,
                          time_col: str = 'Time',
                          intensity_col: str = 'Intensity',
                          curve_col: str = 'CurveID',
                          group_col: str = 'GroupID',
                          normalize: bool = True,
                          progress_callback: Optional[Callable] = None) -> UnifiedAnalysisResult:
        """
        Run the complete unified model selection workflow.
        
        Parameters
        ----------
        df : DataFrame
            Input data
        time_col, intensity_col, curve_col, group_col : str
            Column names
        normalize : bool
            Whether to apply double normalization
        progress_callback : callable, optional
            Progress reporting function
            
        Returns
        -------
        UnifiedAnalysisResult
            Complete analysis results
        """
        result = UnifiedAnalysisResult(timestamp=datetime.now().isoformat())
        
        # Prepare data
        grouped_data = self.prepare_data(df, time_col, intensity_col, curve_col, group_col)
        
        # Optionally normalize
        if normalize:
            grouped_data = {gid: self.normalize_curves(curves) 
                          for gid, curves in grouped_data.items()}
        
        result.groups_analyzed = list(grouped_data.keys())
        
        # Phase 1: Exploration
        if progress_callback:
            progress_callback(0, 3, "Phase 1: Exploring models for each group...")
        
        result.group_explorations = self.phase1_explore_groups(grouped_data, progress_callback)
        
        # Build model selection table
        selection_data = []
        for group_id, model_results in result.group_explorations.items():
            for model_name, fit_result in model_results.items():
                selection_data.append({
                    'Group': group_id,
                    'Model': model_name,
                    'Success': fit_result.success,
                    'Chi-squared': fit_result.chisqr,
                    'Reduced Chi-sq': fit_result.redchi,
                    'AIC': fit_result.aic,
                    'BIC': fit_result.bic,
                    'N_params': fit_result.nvarys
                })
        
        result.model_selection_table = pd.DataFrame(selection_data)
        
        # Phase 2: Unified model selection
        if progress_callback:
            progress_callback(1, 3, "Phase 2: Selecting unified model...")
        
        result.unified_model, result.group_preferred_models, result.unification_rationale = \
            self.phase2_select_unified_model(result.group_explorations)
        
        # Phase 3: Final comparative analysis
        if progress_callback:
            progress_callback(2, 3, f"Phase 3: Final fitting with {result.unified_model} model...")
        
        result.final_fits = self.phase3_final_analysis(
            grouped_data, result.unified_model, calc_confidence=True
        )
        
        # Build comparison table
        comparison_data = []
        for group_id, fit_result in result.final_fits.items():
            row = {
                'Group': group_id,
                'Model': fit_result.model_name,
                'Chi-squared': fit_result.chisqr,
                'R² (mean)': np.mean([cf['r2'] for cf in fit_result.curve_fits.values()]),
                'N_curves': len(fit_result.curve_fits)
            }
            
            # Add global parameters
            for pname, pdata in fit_result.parameters.items():
                if pdata.get('is_global') and pdata.get('vary', True):
                    row[pname] = pdata['value']
                    if pdata['stderr'] is not None and not np.isnan(pdata['stderr']):
                        row[f'{pname}_err'] = pdata['stderr']
            
            comparison_data.append(row)
        
        result.comparison_table = pd.DataFrame(comparison_data)
        
        return result


# =============================================================================
# REPORTING AND VISUALIZATION
# =============================================================================

class GlobalFitReporter:
    """Generate reports and visualizations for global fitting results."""
    
    @staticmethod
    def plot_group_fits(grouped_data: Dict[str, List[CurveData]],
                        fit_results: Dict[str, GlobalFitResult],
                        figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot raw data with global fit overlays for each group.
        
        Parameters
        ----------
        grouped_data : dict
            {group_id: [CurveData, ...]}
        fit_results : dict
            {group_id: GlobalFitResult}
        figsize : tuple
            Figure size
            
        Returns
        -------
        Figure
            Matplotlib figure
        """
        n_groups = len(grouped_data)
        n_cols = min(3, n_groups)
        n_rows = (n_groups + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
        axes = axes.flatten()
        
        colors = plt.cm.tab10.colors
        
        for idx, (group_id, curves) in enumerate(grouped_data.items()):
            ax = axes[idx]
            fit_result = fit_results.get(group_id)
            
            # Plot raw data points
            for i, curve in enumerate(curves):
                alpha = 0.5 if len(curves) > 5 else 0.7
                ax.scatter(curve.time, curve.intensity, s=10, alpha=alpha,
                          color=colors[i % len(colors)], label=f'{curve.curve_id}' if i < 5 else None)
                
                # Plot fit if available
                if fit_result and fit_result.success:
                    curve_fit = fit_result.curve_fits.get(curve.curve_id)
                    if curve_fit:
                        ax.plot(curve.time, curve_fit['fitted'], '-', 
                               color=colors[i % len(colors)], linewidth=1.5)
            
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Normalized Intensity')
            ax.set_title(f'{group_id}\n({fit_result.model_name if fit_result else "No fit"})')
            
            if len(curves) <= 5:
                ax.legend(fontsize=8)
        
        # Hide unused axes
        for idx in range(len(grouped_data), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_residuals(grouped_data: Dict[str, List[CurveData]],
                       fit_results: Dict[str, GlobalFitResult],
                       figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Plot residuals for each group to assess fit quality.
        """
        n_groups = len(grouped_data)
        
        fig, axes = plt.subplots(2, n_groups, figsize=figsize, squeeze=False)
        
        for idx, (group_id, curves) in enumerate(grouped_data.items()):
            fit_result = fit_results.get(group_id)
            
            if not fit_result or not fit_result.success:
                continue
            
            # Collect all residuals
            all_residuals = []
            all_times = []
            
            for curve in curves:
                curve_fit = fit_result.curve_fits.get(curve.curve_id)
                if curve_fit:
                    all_residuals.extend(curve_fit['residuals'])
                    all_times.extend(curve.time)
            
            # Residuals vs time
            axes[0, idx].scatter(all_times, all_residuals, s=5, alpha=0.5)
            axes[0, idx].axhline(y=0, color='r', linestyle='--', linewidth=1)
            axes[0, idx].set_xlabel('Time (s)')
            axes[0, idx].set_ylabel('Residual')
            axes[0, idx].set_title(f'{group_id} - Residuals vs Time')
            
            # Residual histogram
            axes[1, idx].hist(all_residuals, bins=30, density=True, alpha=0.7)
            axes[1, idx].set_xlabel('Residual')
            axes[1, idx].set_ylabel('Density')
            axes[1, idx].set_title(f'{group_id} - Residual Distribution')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def generate_comparison_table(analysis_result: UnifiedAnalysisResult) -> pd.DataFrame:
        """
        Generate a formatted comparison table for the final results.
        
        Returns
        -------
        DataFrame
            Formatted comparison table with parameters and confidence intervals
        """
        if analysis_result.comparison_table is None:
            return pd.DataFrame()
        
        # Make a copy and format
        df = analysis_result.comparison_table.copy()
        
        # Round numeric columns
        for col in df.columns:
            if df[col].dtype in [np.float64, np.float32]:
                df[col] = df[col].round(4)
        
        return df
    
    @staticmethod
    def generate_model_selection_report(analysis_result: UnifiedAnalysisResult) -> str:
        """
        Generate a text report summarizing model selection.
        
        Returns
        -------
        str
            Formatted report text
        """
        lines = [
            "=" * 60,
            "GLOBAL FITTING MODEL SELECTION REPORT",
            "=" * 60,
            f"Generated: {analysis_result.timestamp}",
            f"Groups analyzed: {', '.join(analysis_result.groups_analyzed)}",
            "",
            "-" * 60,
            "PHASE 1: GROUP-LEVEL MODEL EXPLORATION",
            "-" * 60,
        ]
        
        if analysis_result.model_selection_table is not None:
            lines.append(analysis_result.model_selection_table.to_string(index=False))
        
        lines.extend([
            "",
            "-" * 60,
            "PHASE 2: UNIFIED MODEL SELECTION",
            "-" * 60,
            "",
            "Per-group preferred models:",
        ])
        
        for group, model in analysis_result.group_preferred_models.items():
            lines.append(f"  {group}: {model}")
        
        lines.extend([
            "",
            f"UNIFIED MODEL: {analysis_result.unified_model}",
            "",
            "Selection rationale:",
            analysis_result.unification_rationale,
            "",
            "-" * 60,
            "PHASE 3: FINAL COMPARATIVE RESULTS",
            "-" * 60,
        ])
        
        if analysis_result.comparison_table is not None:
            lines.append(analysis_result.comparison_table.to_string(index=False))
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    @staticmethod
    def plot_parameter_comparison(analysis_result: UnifiedAnalysisResult,
                                   param_name: str,
                                   figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
        """
        Create bar plot comparing a parameter across groups with error bars.
        
        Parameters
        ----------
        analysis_result : UnifiedAnalysisResult
            Analysis results
        param_name : str
            Parameter to plot (e.g., 'k_global', 'k_fast_global')
        figsize : tuple
            Figure size
            
        Returns
        -------
        Figure
            Matplotlib figure
        """
        groups = []
        values = []
        errors = []
        
        for group_id, fit_result in analysis_result.final_fits.items():
            if param_name in fit_result.parameters:
                groups.append(group_id)
                values.append(fit_result.parameters[param_name]['value'])
                err = fit_result.parameters[param_name].get('stderr', 0)
                errors.append(err if err is not None and not np.isnan(err) else 0)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        x = np.arange(len(groups))
        bars = ax.bar(x, values, yerr=errors, capsize=5, color='steelblue', alpha=0.8)
        
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=45, ha='right')
        ax.set_ylabel(param_name)
        ax.set_title(f'{param_name} Comparison Across Groups')
        
        # Add value labels on bars
        for bar, val, err in zip(bars, values, errors):
            height = bar.get_height()
            ax.annotate(f'{val:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        return fig


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def run_global_frap_analysis(df: pd.DataFrame,
                              time_col: str = 'Time',
                              intensity_col: str = 'Intensity',
                              curve_col: str = 'CurveID',
                              group_col: str = 'GroupID',
                              mf_is_local: bool = True,
                              alpha: float = 0.05,
                              normalize: bool = True) -> UnifiedAnalysisResult:
    """
    Convenience function to run complete global FRAP analysis.
    
    Parameters
    ----------
    df : DataFrame
        Input data with columns for time, intensity, curve ID, and group ID
    time_col, intensity_col, curve_col, group_col : str
        Column names in the DataFrame
    mf_is_local : bool
        If True, mobile fraction is fitted per-curve
    alpha : float
        Significance level for model selection
    normalize : bool
        Whether to apply double normalization
        
    Returns
    -------
    UnifiedAnalysisResult
        Complete analysis results including model selection and final fits
        
    Example
    -------
    >>> import pandas as pd
    >>> from frap_global_fitting import run_global_frap_analysis
    >>> 
    >>> # Load your data
    >>> df = pd.read_csv('frap_data.csv')
    >>> 
    >>> # Run analysis
    >>> result = run_global_frap_analysis(
    ...     df, 
    ...     time_col='Time',
    ...     intensity_col='Intensity',
    ...     curve_col='CellID',
    ...     group_col='Condition'
    ... )
    >>> 
    >>> # Access results
    >>> print(f"Unified model: {result.unified_model}")
    >>> print(result.comparison_table)
    """
    workflow = UnifiedModelWorkflow(mf_is_local=mf_is_local, alpha=alpha)
    return workflow.run_full_analysis(
        df, time_col, intensity_col, curve_col, group_col, normalize
    )


def convert_analyzer_to_dataframe(analyzer_dict: Dict[str, Any],
                                   group_names: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Convert existing FRAPGroupAnalyzer results to DataFrame format for global fitting.
    
    This bridges the existing analysis infrastructure with the global fitting module.
    
    Parameters
    ----------
    analyzer_dict : dict
        Dictionary of {group_name: analyzer_object} from existing workflow
    group_names : list, optional
        Specific groups to include
        
    Returns
    -------
    DataFrame
        Data formatted for global fitting
    """
    rows = []
    
    groups_to_process = group_names if group_names else list(analyzer_dict.keys())
    
    for group_name in groups_to_process:
        if group_name not in analyzer_dict:
            continue
            
        analyzer = analyzer_dict[group_name]
        
        # Handle both object and dict access patterns
        if hasattr(analyzer, 'curves'):
            curves = analyzer.curves
        elif isinstance(analyzer, dict) and 'curves' in analyzer:
            curves = analyzer['curves']
        else:
            continue
        
        for curve_idx, curve in enumerate(curves):
            # Get time and intensity arrays
            if hasattr(curve, 'time_post_bleach'):
                time = curve.time_post_bleach
                intensity = curve.intensity_post_bleach
                curve_id = getattr(curve, 'filename', f'curve_{curve_idx}')
            elif isinstance(curve, dict):
                time = curve.get('time_post_bleach', curve.get('time', []))
                intensity = curve.get('intensity_post_bleach', curve.get('intensity', []))
                curve_id = curve.get('filename', f'curve_{curve_idx}')
            else:
                continue
            
            for t_val, i_val in zip(time, intensity):
                rows.append({
                    'Time': t_val,
                    'Intensity': i_val,
                    'CurveID': curve_id,
                    'GroupID': group_name
                })
    
    return pd.DataFrame(rows)


# Module exports
__all__ = [
    'CurveData',
    'GlobalFitResult', 
    'ModelComparisonResult',
    'UnifiedAnalysisResult',
    'model_single_exponential',
    'model_double_exponential',
    'model_triple_exponential',
    'MODEL_REGISTRY',
    'GlobalFitter',
    'ModelSelector',
    'UnifiedModelWorkflow',
    'GlobalFitReporter',
    'run_global_frap_analysis',
    'convert_analyzer_to_dataframe',
    'LMFIT_AVAILABLE'
]
