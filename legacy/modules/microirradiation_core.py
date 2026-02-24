"""
Microirradiation Analysis Core Module
Core computational routines for laser microirradiation experiments including:
- Protein recruitment kinetics to DSB sites
- Chromatin decondensation (ROI expansion) analysis
- Combined microirradiation + photobleaching analysis
"""

import numpy as np
import pandas as pd
import logging
from scipy.optimize import curve_fit
from scipy.stats import linregress
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class MicroirradiationResult:
    """Container for microirradiation analysis results"""
    # Recruitment kinetics
    recruitment_rate: float = np.nan
    recruitment_amplitude: float = np.nan
    recruitment_plateau: float = np.nan
    recruitment_half_time: float = np.nan
    
    # ROI expansion/decondensation
    initial_area: float = np.nan
    final_area: float = np.nan
    expansion_rate: float = np.nan
    expansion_half_time: float = np.nan
    max_expansion: float = np.nan
    
    # Combined analysis (when photobleaching is also present)
    is_combined_analysis: bool = False
    bleach_frame: Optional[int] = None
    pre_bleach_recruitment: float = np.nan
    post_bleach_recovery: float = np.nan
    
    # Fit quality metrics
    recruitment_r_squared: float = np.nan
    expansion_r_squared: float = np.nan
    recruitment_aic: float = np.nan
    expansion_aic: float = np.nan


# ===== RECRUITMENT KINETICS MODELS =====

def single_exponential_recruitment(t: np.ndarray, amplitude: float, rate: float, baseline: float) -> np.ndarray:
    """
    Single exponential recruitment model: I(t) = baseline + amplitude * (1 - exp(-rate * t))
    
    Parameters:
    -----------
    t : array-like
        Time points
    amplitude : float
        Maximum recruitment amplitude
    rate : float
        Recruitment rate constant (1/time)
    baseline : float
        Initial intensity before recruitment
        
    Returns:
    --------
    np.ndarray
        Predicted intensities
    """
    return baseline + amplitude * (1 - np.exp(-rate * t))


def double_exponential_recruitment(t: np.ndarray, 
                                 amp1: float, rate1: float,
                                 amp2: float, rate2: float, 
                                 baseline: float) -> np.ndarray:
    """
    Double exponential recruitment model for complex recruitment kinetics
    I(t) = baseline + amp1 * (1 - exp(-rate1 * t)) + amp2 * (1 - exp(-rate2 * t))
    
    Parameters:
    -----------
    t : array-like
        Time points
    amp1, amp2 : float
        Recruitment amplitudes for fast and slow components
    rate1, rate2 : float
        Recruitment rate constants (1/time)
    baseline : float
        Initial intensity before recruitment
        
    Returns:
    --------
    np.ndarray
        Predicted intensities
    """
    return baseline + amp1 * (1 - np.exp(-rate1 * t)) + amp2 * (1 - np.exp(-rate2 * t))


def sigmoidal_recruitment(t: np.ndarray, amplitude: float, rate: float, 
                         lag_time: float, baseline: float) -> np.ndarray:
    """
    Sigmoidal recruitment model with lag time
    I(t) = baseline + amplitude / (1 + exp(-rate * (t - lag_time)))
    
    Parameters:
    -----------
    t : array-like
        Time points
    amplitude : float
        Maximum recruitment amplitude
    rate : float
        Recruitment rate constant
    lag_time : float
        Time delay before recruitment begins
    baseline : float
        Initial intensity
        
    Returns:
    --------
    np.ndarray
        Predicted intensities
    """
    return baseline + amplitude / (1 + np.exp(-rate * (t - lag_time)))


# ===== ROI EXPANSION MODELS =====

def exponential_expansion(t: np.ndarray, initial_size: float, 
                         max_expansion: float, rate: float) -> np.ndarray:
    """
    Exponential ROI expansion model: A(t) = initial_size + max_expansion * (1 - exp(-rate * t))
    
    Parameters:
    -----------
    t : array-like
        Time points
    initial_size : float
        Initial ROI area/size
    max_expansion : float
        Maximum expansion amount
    rate : float
        Expansion rate constant
        
    Returns:
    --------
    np.ndarray
        Predicted ROI sizes
    """
    return initial_size + max_expansion * (1 - np.exp(-rate * t))


def linear_expansion(t: np.ndarray, initial_size: float, rate: float) -> np.ndarray:
    """
    Linear ROI expansion model: A(t) = initial_size + rate * t
    
    Parameters:
    -----------
    t : array-like
        Time points
    initial_size : float
        Initial ROI area/size
    rate : float
        Linear expansion rate
        
    Returns:
    --------
    np.ndarray
        Predicted ROI sizes
    """
    return initial_size + rate * t


def power_law_expansion(t: np.ndarray, initial_size: float, 
                       coefficient: float, exponent: float) -> np.ndarray:
    """
    Power law ROI expansion model: A(t) = initial_size + coefficient * t^exponent
    
    Parameters:
    -----------
    t : array-like
        Time points
    initial_size : float
        Initial ROI area/size
    coefficient : float
        Power law coefficient
    exponent : float
        Power law exponent
        
    Returns:
    --------
    np.ndarray
        Predicted ROI sizes
    """
    return initial_size + coefficient * np.power(t, exponent)


# ===== ANALYSIS FUNCTIONS =====

def analyze_recruitment_kinetics(time: np.ndarray, 
                                intensity: np.ndarray,
                                damage_frame: int = 0,
                                models: List[str] = ['single_exp', 'double_exp', 'sigmoidal']) -> Dict:
    """
    Analyze protein recruitment kinetics at damage site
    
    Parameters:
    -----------
    time : np.ndarray
        Time points
    intensity : np.ndarray  
        Intensity values at damage ROI
    damage_frame : int
        Frame when microirradiation occurred
    models : list
        List of models to fit ['single_exp', 'double_exp', 'sigmoidal']
        
    Returns:
    --------
    dict
        Analysis results with best fit parameters and model selection
    """
    # Extract post-damage data
    post_damage_time = time[damage_frame:]
    post_damage_intensity = intensity[damage_frame:]
    
    # Normalize time to start from 0 at damage
    post_damage_time = post_damage_time - post_damage_time[0]
    
    results = {}
    fits = {}
    
    # Initial parameter estimates
    baseline = np.mean(post_damage_intensity[:3]) if len(post_damage_intensity) > 3 else post_damage_intensity[0]
    max_intensity = np.max(post_damage_intensity)
    amplitude_guess = max_intensity - baseline
    
    # Single exponential recruitment
    if 'single_exp' in models:
        try:
            p0 = [amplitude_guess, 0.1, baseline]
            bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])
            popt, pcov = curve_fit(single_exponential_recruitment, 
                                 post_damage_time, post_damage_intensity,
                                 p0=p0, bounds=bounds, maxfev=2000)
            
            y_pred = single_exponential_recruitment(post_damage_time, *popt)
            r_squared = 1 - np.sum((post_damage_intensity - y_pred)**2) / np.sum((post_damage_intensity - np.mean(post_damage_intensity))**2)
            aic = len(post_damage_intensity) * np.log(np.sum((post_damage_intensity - y_pred)**2) / len(post_damage_intensity)) + 2 * len(popt)
            
            fits['single_exp'] = {
                'params': popt,
                'r_squared': r_squared,
                'aic': aic,
                'amplitude': popt[0],
                'rate': popt[1],
                'baseline': popt[2],
                'half_time': np.log(2) / popt[1] if popt[1] > 0 else np.inf
            }
        except Exception as e:
            logger.warning(f"Single exponential fit failed: {e}")
            fits['single_exp'] = None
    
    # Double exponential recruitment
    if 'double_exp' in models:
        try:
            p0 = [amplitude_guess/2, 0.2, amplitude_guess/2, 0.05, baseline]
            bounds = ([0, 0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf, np.inf])
            popt, pcov = curve_fit(double_exponential_recruitment, 
                                 post_damage_time, post_damage_intensity,
                                 p0=p0, bounds=bounds, maxfev=2000)
            
            y_pred = double_exponential_recruitment(post_damage_time, *popt)
            r_squared = 1 - np.sum((post_damage_intensity - y_pred)**2) / np.sum((post_damage_intensity - np.mean(post_damage_intensity))**2)
            aic = len(post_damage_intensity) * np.log(np.sum((post_damage_intensity - y_pred)**2) / len(post_damage_intensity)) + 2 * len(popt)
            
            fits['double_exp'] = {
                'params': popt,
                'r_squared': r_squared,
                'aic': aic,
                'amp1': popt[0],
                'rate1': popt[1], 
                'amp2': popt[2],
                'rate2': popt[3],
                'baseline': popt[4],
                'total_amplitude': popt[0] + popt[2],
                'fast_half_time': np.log(2) / popt[1] if popt[1] > 0 else np.inf,
                'slow_half_time': np.log(2) / popt[3] if popt[3] > 0 else np.inf
            }
        except Exception as e:
            logger.warning(f"Double exponential fit failed: {e}")
            fits['double_exp'] = None
    
    # Sigmoidal recruitment
    if 'sigmoidal' in models:
        try:
            p0 = [amplitude_guess, 0.1, np.mean(post_damage_time), baseline]
            bounds = ([0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf])
            popt, pcov = curve_fit(sigmoidal_recruitment, 
                                 post_damage_time, post_damage_intensity,
                                 p0=p0, bounds=bounds, maxfev=2000)
            
            y_pred = sigmoidal_recruitment(post_damage_time, *popt)
            r_squared = 1 - np.sum((post_damage_intensity - y_pred)**2) / np.sum((post_damage_intensity - np.mean(post_damage_intensity))**2)
            aic = len(post_damage_intensity) * np.log(np.sum((post_damage_intensity - y_pred)**2) / len(post_damage_intensity)) + 2 * len(popt)
            
            fits['sigmoidal'] = {
                'params': popt,
                'r_squared': r_squared,
                'aic': aic,
                'amplitude': popt[0],
                'rate': popt[1],
                'lag_time': popt[2],
                'baseline': popt[3],
                'half_time': popt[2]  # Approximate half-time at lag time
            }
        except Exception as e:
            logger.warning(f"Sigmoidal fit failed: {e}")
            fits['sigmoidal'] = None
    
    # Model selection based on AIC
    valid_fits = {k: v for k, v in fits.items() if v is not None}
    if valid_fits:
        best_model = min(valid_fits.keys(), key=lambda k: valid_fits[k]['aic'])
        results['best_model'] = best_model
        results['best_fit'] = valid_fits[best_model]
        results['all_fits'] = fits
    else:
        results['best_model'] = None
        results['best_fit'] = None
        results['all_fits'] = fits
    
    return results


def analyze_roi_expansion(time: np.ndarray, 
                         roi_areas: np.ndarray,
                         damage_frame: int = 0,
                         models: List[str] = ['exponential', 'linear', 'power_law']) -> Dict:
    """
    Analyze chromatin decondensation through ROI expansion
    
    Parameters:
    -----------
    time : np.ndarray
        Time points
    roi_areas : np.ndarray
        ROI areas over time
    damage_frame : int
        Frame when microirradiation occurred
    models : list
        List of models to fit ['exponential', 'linear', 'power_law']
        
    Returns:
    --------
    dict
        Analysis results with best fit parameters and model selection
    """
    # Extract post-damage data
    post_damage_time = time[damage_frame:]
    post_damage_areas = roi_areas[damage_frame:]
    
    # Normalize time to start from 0 at damage
    post_damage_time = post_damage_time - post_damage_time[0]
    
    results = {}
    fits = {}
    
    initial_area = post_damage_areas[0]
    max_area = np.max(post_damage_areas)
    
    # Exponential expansion
    if 'exponential' in models:
        try:
            p0 = [initial_area, max_area - initial_area, 0.1]
            bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])
            popt, pcov = curve_fit(exponential_expansion, 
                                 post_damage_time, post_damage_areas,
                                 p0=p0, bounds=bounds, maxfev=2000)
            
            y_pred = exponential_expansion(post_damage_time, *popt)
            r_squared = 1 - np.sum((post_damage_areas - y_pred)**2) / np.sum((post_damage_areas - np.mean(post_damage_areas))**2)
            aic = len(post_damage_areas) * np.log(np.sum((post_damage_areas - y_pred)**2) / len(post_damage_areas)) + 2 * len(popt)
            
            fits['exponential'] = {
                'params': popt,
                'r_squared': r_squared,
                'aic': aic,
                'initial_size': popt[0],
                'max_expansion': popt[1],
                'rate': popt[2],
                'half_time': np.log(2) / popt[2] if popt[2] > 0 else np.inf
            }
        except Exception as e:
            logger.warning(f"Exponential expansion fit failed: {e}")
            fits['exponential'] = None
    
    # Linear expansion  
    if 'linear' in models:
        try:
            p0 = [initial_area, (max_area - initial_area) / np.max(post_damage_time)]
            popt, pcov = curve_fit(linear_expansion, 
                                 post_damage_time, post_damage_areas,
                                 p0=p0, maxfev=2000)
            
            y_pred = linear_expansion(post_damage_time, *popt)
            r_squared = 1 - np.sum((post_damage_areas - y_pred)**2) / np.sum((post_damage_areas - np.mean(post_damage_areas))**2)
            aic = len(post_damage_areas) * np.log(np.sum((post_damage_areas - y_pred)**2) / len(post_damage_areas)) + 2 * len(popt)
            
            fits['linear'] = {
                'params': popt,
                'r_squared': r_squared,
                'aic': aic,
                'initial_size': popt[0],
                'rate': popt[1]
            }
        except Exception as e:
            logger.warning(f"Linear expansion fit failed: {e}")
            fits['linear'] = None
    
    # Power law expansion
    if 'power_law' in models:
        try:
            p0 = [initial_area, 1.0, 0.5]
            bounds = ([0, 0, 0], [np.inf, np.inf, 2])  # Constrain exponent to be reasonable
            popt, pcov = curve_fit(power_law_expansion, 
                                 post_damage_time, post_damage_areas,
                                 p0=p0, bounds=bounds, maxfev=2000)
            
            y_pred = power_law_expansion(post_damage_time, *popt)
            r_squared = 1 - np.sum((post_damage_areas - y_pred)**2) / np.sum((post_damage_areas - np.mean(post_damage_areas))**2)
            aic = len(post_damage_areas) * np.log(np.sum((post_damage_areas - y_pred)**2) / len(post_damage_areas)) + 2 * len(popt)
            
            fits['power_law'] = {
                'params': popt,
                'r_squared': r_squared,
                'aic': aic,
                'initial_size': popt[0],
                'coefficient': popt[1],
                'exponent': popt[2]
            }
        except Exception as e:
            logger.warning(f"Power law expansion fit failed: {e}")
            fits['power_law'] = None
    
    # Model selection based on AIC
    valid_fits = {k: v for k, v in fits.items() if v is not None}
    if valid_fits:
        best_model = min(valid_fits.keys(), key=lambda k: valid_fits[k]['aic'])
        results['best_model'] = best_model
        results['best_fit'] = valid_fits[best_model]
        results['all_fits'] = fits
    else:
        results['best_model'] = None
        results['best_fit'] = None
        results['all_fits'] = fits
    
    return results


def analyze_combined_experiment(time: np.ndarray,
                               intensity: np.ndarray,
                               damage_frame: int,
                               bleach_frame: int,
                               roi_areas: Optional[np.ndarray] = None) -> MicroirradiationResult:
    """
    Analyze combined microirradiation + photobleaching experiment
    
    Parameters:
    -----------
    time : np.ndarray
        Time points
    intensity : np.ndarray
        Intensity values
    damage_frame : int
        Frame when microirradiation occurred
    bleach_frame : int
        Frame when photobleaching occurred  
    roi_areas : np.ndarray, optional
        ROI areas for expansion analysis
        
    Returns:
    --------
    MicroirradiationResult
        Comprehensive analysis results
    """
    result = MicroirradiationResult()
    result.is_combined_analysis = True
    result.bleach_frame = bleach_frame
    
    # Determine experiment sequence
    if damage_frame < bleach_frame:
        # Damage first, then bleach - analyze recruitment then recovery
        
        # 1. Analyze recruitment phase (damage to bleach)
        recruitment_time = time[damage_frame:bleach_frame+1]
        recruitment_intensity = intensity[damage_frame:bleach_frame+1]
        
        if len(recruitment_time) > 3:  # Need enough points for fitting
            recruitment_results = analyze_recruitment_kinetics(
                recruitment_time, recruitment_intensity, 
                damage_frame=0  # Already extracted from damage frame
            )
            
            if recruitment_results['best_fit']:
                result.recruitment_rate = recruitment_results['best_fit'].get('rate', np.nan)
                result.recruitment_amplitude = recruitment_results['best_fit'].get('amplitude', np.nan)
                result.recruitment_half_time = recruitment_results['best_fit'].get('half_time', np.nan)
                result.recruitment_r_squared = recruitment_results['best_fit'].get('r_squared', np.nan)
                result.recruitment_aic = recruitment_results['best_fit'].get('aic', np.nan)
                result.pre_bleach_recruitment = recruitment_results['best_fit'].get('amplitude', np.nan)
        
        # 2. Analyze recovery phase (post-bleach)
        # This would use standard FRAP analysis from the existing codebase
        # Import and use existing FRAP analysis functions
        
    elif bleach_frame < damage_frame:
        # Bleach first, then damage - analyze recovery then recruitment
        
        # 1. Analyze FRAP recovery phase (bleach to damage)
        # Use existing FRAP analysis
        
        # 2. Analyze post-damage recruitment
        recruitment_results = analyze_recruitment_kinetics(
            time, intensity, damage_frame=damage_frame
        )
        
        if recruitment_results['best_fit']:
            result.recruitment_rate = recruitment_results['best_fit'].get('rate', np.nan)
            result.recruitment_amplitude = recruitment_results['best_fit'].get('amplitude', np.nan)
            result.recruitment_half_time = recruitment_results['best_fit'].get('half_time', np.nan)
            result.recruitment_r_squared = recruitment_results['best_fit'].get('r_squared', np.nan)
            result.recruitment_aic = recruitment_results['best_fit'].get('aic', np.nan)
    
    # Analyze ROI expansion if data provided
    if roi_areas is not None:
        expansion_results = analyze_roi_expansion(
            time, roi_areas, damage_frame=damage_frame
        )
        
        if expansion_results['best_fit']:
            result.initial_area = expansion_results['best_fit'].get('initial_size', np.nan)
            result.expansion_rate = expansion_results['best_fit'].get('rate', np.nan)
            result.max_expansion = expansion_results['best_fit'].get('max_expansion', np.nan)
            result.expansion_r_squared = expansion_results['best_fit'].get('r_squared', np.nan)
            result.expansion_aic = expansion_results['best_fit'].get('aic', np.nan)
            if 'half_time' in expansion_results['best_fit']:
                result.expansion_half_time = expansion_results['best_fit']['half_time']
    
    return result


# ===== UTILITY FUNCTIONS =====

def calculate_recruitment_metrics(intensity: np.ndarray, 
                                 baseline: float,
                                 time_points: np.ndarray) -> Dict[str, float]:
    """
    Calculate basic recruitment metrics
    
    Parameters:
    -----------
    intensity : np.ndarray
        Intensity time series
    baseline : float
        Pre-damage baseline intensity
    time_points : np.ndarray
        Time points
        
    Returns:
    --------
    dict
        Dictionary of recruitment metrics
    """
    max_intensity = np.max(intensity)
    fold_increase = max_intensity / baseline if baseline > 0 else np.inf
    
    # Time to half-max recruitment
    half_max = baseline + (max_intensity - baseline) / 2
    time_to_half_max = np.nan
    for i, val in enumerate(intensity):
        if val >= half_max:
            time_to_half_max = time_points[i]
            break
    
    # Area under curve (total recruitment)
    total_recruitment = np.trapz(intensity - baseline, time_points)
    
    return {
        'max_intensity': max_intensity,
        'fold_increase': fold_increase,
        'time_to_half_max': time_to_half_max,
        'total_recruitment': total_recruitment,
        'recruitment_amplitude': max_intensity - baseline
    }


def generate_adaptive_mask(roi_center: Tuple[int, int], 
                          expansion_factor: float,
                          original_radius: float,
                          image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Generate adaptive mask based on ROI expansion
    
    Parameters:
    -----------
    roi_center : tuple
        (x, y) center coordinates
    expansion_factor : float
        Factor by which ROI has expanded
    original_radius : float
        Original ROI radius
    image_shape : tuple
        (height, width) of image
        
    Returns:
    --------
    np.ndarray
        Boolean mask array
    """
    h, w = image_shape
    y, x = np.ogrid[:h, :w]
    
    # Calculate expanded radius
    expanded_radius = original_radius * expansion_factor
    
    # Create circular mask
    mask = (x - roi_center[0])**2 + (y - roi_center[1])**2 <= expanded_radius**2
    
    return mask