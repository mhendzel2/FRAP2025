import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass
import lmfit
from scipy.special import i0, i1
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
import scipy.stats as stats

from frap_input_handler import FRAPCurveData

logger = logging.getLogger(__name__)

# --- Model Definitions ---

def soumpasis_diffusion(t, F_inf, F_0, tau_D):
    """
    Soumpasis diffusion model for circular ROI.
    F(t) = F_inf - (F_inf - F_0) * (1 / (1 + t/tau_D))
    Note: This is a simplified form. The full Soumpasis involves Bessel functions.
    Full Soumpasis: f(t) = exp(-2*tau/t) * (I0(2*tau/t) + I1(2*tau/t))
    Let's use the full form if possible, or the standard approximation.
    Standard approximation: F(t) = F_inf - (F_inf - F_0) * (1 / (1 + (t/tau_D))) is often used but Soumpasis is specific.
    
    Soumpasis (1983):
    f(t) = exp(-2*tau_D/t) * (I0(2*tau_D/t) + I1(2*tau_D/t))
    F(t) = F_0 + (F_inf - F_0) * f(t)
    """
    # Avoid division by zero
    t_safe = np.maximum(t, 1e-9)
    arg = 2 * tau_D / t_safe
    f_t = np.exp(-arg) * (i0(arg) + i1(arg))
    return F_0 + (F_inf - F_0) * f_t

def single_exponential(t, F_inf, F_0, k_off):
    """
    Single exponential recovery (Reaction dominant).
    F(t) = F_inf - (F_inf - F_0) * exp(-k_off * t)
    """
    return F_inf - (F_inf - F_0) * np.exp(-k_off * t)

def double_exponential(t, F_inf, F_0, k_fast, f_fast, k_slow):
    """
    Double exponential recovery.
    F(t) = F_inf - (F_inf - F_0) * (f_fast * exp(-k_fast * t) + (1 - f_fast) * exp(-k_slow * t))
    """
    return F_inf - (F_inf - F_0) * (f_fast * np.exp(-k_fast * t) + (1 - f_fast) * np.exp(-k_slow * t))

# --- Analysis Classes ---

@dataclass
class FitResult:
    model_name: str
    params: dict
    metrics: dict  # r2, aic, bic
    fitted_curve: np.ndarray
    residuals: np.ndarray
    success: bool
    message: str

class FRAPFitter:
    """
    Handles fitting of FRAP models to data.
    """
    
    def __init__(self):
        self.models = {
            'soumpasis': soumpasis_diffusion,
            'single_exp': single_exponential,
            'double_exp': double_exponential
        }

    def fit_model(self, data: FRAPCurveData, model_name: str, p0: dict = None) -> FitResult:
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        func = self.models[model_name]
        model = lmfit.Model(func)
        
        # Set up parameters
        params = model.make_params()
        
        # Initial guesses
        if data.f_zero_comp is not None:
            # Fix F_0 to the compensated value as requested
            params.add('F_0', value=data.f_zero_comp, vary=False)
        else:
            params.add('F_0', value=np.min(data.intensity_post_bleach), min=0)
            
        params.add('F_inf', value=np.max(data.intensity_post_bleach), min=0, max=1.5) # Normalized usually <= 1 but allow some overshoot
        
        if model_name == 'soumpasis':
            params.add('tau_D', value=1.0, min=1e-6)
        elif model_name == 'single_exp':
            params.add('k_off', value=0.1, min=1e-6)
        elif model_name == 'double_exp':
            params.add('k_fast', value=1.0, min=1e-6)
            params.add('k_slow', value=0.1, min=1e-6)
            params.add('f_fast', value=0.5, min=0, max=1)
            
        # Override with user provided p0
        if p0:
            for k, v in p0.items():
                if k in params:
                    params[k].set(value=v)

        try:
            result = model.fit(data.intensity_post_bleach, params, t=data.time_post_bleach, method='leastsq') # Levenberg-Marquardt
            
            # Calculate metrics
            r2 = 1 - result.redchi / np.var(data.intensity_post_bleach) if np.var(data.intensity_post_bleach) > 0 else 0
            
            return FitResult(
                model_name=model_name,
                params=result.best_values,
                metrics={
                    'chisqr': result.chisqr,
                    'redchi': result.redchi,
                    'aic': result.aic,
                    'bic': result.bic,
                    'r2': r2
                },
                fitted_curve=result.best_fit,
                residuals=result.residual,
                success=result.success,
                message=result.message
            )
        except Exception as e:
            logger.error(f"Fitting failed for {model_name}: {e}")
            return FitResult(
                model_name=model_name,
                params={},
                metrics={},
                fitted_curve=None,
                residuals=None,
                success=False,
                message=str(e)
            )

    def calculate_derived_parameters(self, fit_result: FitResult, bleach_radius: float) -> dict:
        """
        Calculates D, MW_eff, Mf, If based on fit results.
        """
        params = fit_result.params
        derived = {}
        
        # Mobile Fraction
        # Mf = (F_inf - F_0) / (1 - F_0)  <-- Assuming normalized to pre-bleach=1
        # But user formula: Mf = (F_inf - F(0)_comp) / (F_pre - F(0)_comp)
        # Since we normalized F_pre to 1 (in double normalization usually), 
        # Mf = (F_inf - F_0) / (1 - F_0)
        
        f_inf = params.get('F_inf', 0)
        f_0 = params.get('F_0', 0)
        
        mf = (f_inf - f_0) / (1.0 - f_0) if f_0 != 1.0 else 0
        derived['Mf'] = mf
        derived['If'] = 1 - mf
        
        if fit_result.model_name == 'soumpasis':
            tau_d = params.get('tau_D')
            if tau_d:
                # D = w^2 / (4 * tau_D) ? No, for Soumpasis tau_D is characteristic time.
                # Soumpasis formula: D = 0.224 * w^2 / t_half
                # But in the formula used above: f(t) depends on tau_D/t.
                # Usually tau_D in the formula corresponds to characteristic diffusion time.
                # Let's assume tau_D is the diffusion time constant.
                # D = w^2 / (4 * tau_D) is for 2D Gaussian.
                # For Soumpasis, t_half = 0.224 * w^2 / D -> D = 0.224 * w^2 / t_half
                # We need to relate tau_D parameter to t_half or D directly.
                # In the function `soumpasis_diffusion`: arg = 2 * tau_D / t
                # This implies tau_D is a time constant.
                # Let's stick to the standard relation if we can find it.
                # Often tau_D is defined such that D = w^2 / tau_D or similar.
                # Let's use the t_half from the fitted curve to be safe.
                
                # Find t_half from the fitted curve
                # F(t_half) = F_0 + (F_inf - F_0) / 2
                target_y = f_0 + (f_inf - f_0) / 2
                # Find t where fitted_curve is closest to target_y
                # This is approximate
                # Better: D = 0.224 * r^2 / t_1/2
                pass

        return derived

class FRAPGroupAnalyzer:
    """
    Analyzes a group of FRAP curves.
    """
    def __init__(self):
        self.curves: List[FRAPCurveData] = []
        self.fit_results: List[FitResult] = []
        self.features: pd.DataFrame = None
        
    def add_curve(self, curve: FRAPCurveData):
        self.curves.append(curve)
        
    def analyze_group(self, model_name: str = 'soumpasis'):
        fitter = FRAPFitter()
        results = []
        feature_rows = []
        
        for i, curve in enumerate(self.curves):
            res = fitter.fit_model(curve, model_name)
            results.append(res)
            
            if res.success:
                row = res.params.copy()
                row['r2'] = res.metrics['r2']
                row['aic'] = res.metrics['aic']
                row['bic'] = res.metrics['bic']
                feature_rows.append(row)
            else:
                feature_rows.append({}) # Empty or NaN
                
        self.fit_results = results
        self.features = pd.DataFrame(feature_rows)
        
    def detect_subpopulations(self, n_components_range=range(1, 4)):
        """
        Uses GMM to detect subpopulations.
        """
        if self.features is None or self.features.empty:
            return
            
        # Select features for clustering
        # e.g., diffusion time, mobile fraction
        # Need to handle NaNs
        data_clean = self.features.dropna()
        
        if data_clean.empty:
            return

        best_gmm = None
        best_bic = np.inf
        
        # Try different number of components
        max_components = min(len(data_clean), max(n_components_range))
        if max_components < 1:
            return

        for n in range(1, max_components + 1):
            gmm = GaussianMixture(n_components=n, random_state=42)
            gmm.fit(data_clean)
            bic = gmm.bic(data_clean)
            
            if bic < best_bic:
                best_bic = bic
                best_gmm = gmm
                
        # Assign labels
        if best_gmm is not None:
            labels = best_gmm.predict(data_clean)
            self.features.loc[data_clean.index, 'subpopulation'] = labels
        
    def detect_outliers(self):
        """
        Uses Isolation Forest to detect outliers.
        """
        if self.features is None or self.features.empty:
            return
            
        data_clean = self.features.dropna()
        if data_clean.empty:
            return
            
        iso = IsolationForest(contamination=0.1, random_state=42)
        outliers = iso.fit_predict(data_clean)
        # -1 is outlier, 1 is inlier
        self.features.loc[data_clean.index, 'is_outlier'] = (outliers == -1)

class FRAPStatisticalComparator:
    """
    Handles statistical comparisons between groups.
    """
    
    @staticmethod
    def compare_groups(group1_features: pd.DataFrame, group2_features: pd.DataFrame, param: str) -> dict:
        """
        Compares a parameter between two groups.
        """
        data1 = group1_features[param].dropna()
        data2 = group2_features[param].dropna()
        
        # Check normality
        _, p_norm1 = stats.shapiro(data1)
        _, p_norm2 = stats.shapiro(data2)
        
        # Check variance homogeneity
        _, p_levene = stats.levene(data1, data2)
        
        normal = (p_norm1 > 0.05) and (p_norm2 > 0.05)
        equal_var = p_levene > 0.05
        
        if normal:
            if equal_var:
                res = stats.ttest_ind(data1, data2)
                test_name = "Student's t-test"
            else:
                res = stats.ttest_ind(data1, data2, equal_var=False)
                test_name = "Welch's t-test"
            stat, p_val = res[0], res[1]
        else:
            res = stats.mannwhitneyu(data1, data2)
            test_name = "Mann-Whitney U test"
            stat, p_val = res[0], res[1]
            
        return {
            'parameter': param,
            'test': test_name,
            'statistic': float(stat),
            'p_value': float(p_val),
            'significant': float(p_val) < 0.05,
            'mean1': float(np.mean(data1)),
            'mean2': float(np.mean(data2)),
            'std1': float(np.std(data1)),
            'std2': float(np.std(data2))
        }

