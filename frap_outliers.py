"""
FRAP Outlier Detection Module
Functions for identifying and removing outliers in FRAP data
"""
import numpy as np
import pandas as pd
from scipy import stats
import logging

def identify_biological_outliers(curves):
    """
    Identify curves with biologically impossible values that should be automatically rejected.
    
    This catches severe data quality issues like:
    - Mobile fraction > 100% (over-recovery, normalization failure)
    - Mobile fraction < 0% (negative values)
    - Extremely high values (e.g., >200%) indicating calculation errors
    
    Parameters:
    -----------
    curves : list
        List of curve dictionaries with clustering_features
        
    Returns:
    --------
    tuple
        (outlier_indices, reasons) where reasons is a dict mapping index to reason string
    """
    outlier_indices = []
    reasons = {}
    
    for i, curve in enumerate(curves):
        features = curve.get('clustering_features', {})
        mobile_frac = features.get('mobile_fraction', np.nan)
        
        if not np.isfinite(mobile_frac):
            continue
            
        # Check for biologically impossible values
        if mobile_frac > 200:
            outlier_indices.append(i)
            reasons[i] = f"Extreme over-recovery ({mobile_frac:.1f}%) - severe normalization failure"
            logging.warning(f"Auto-rejecting {curve.get('filename', 'unknown')}: extreme mobile fraction {mobile_frac:.1f}%")
        elif mobile_frac > 100:
            outlier_indices.append(i)
            reasons[i] = f"Over-recovery ({mobile_frac:.1f}%) - normalization issue"
            logging.warning(f"Auto-rejecting {curve.get('filename', 'unknown')}: over-recovery {mobile_frac:.1f}%")
        elif mobile_frac < 0:
            outlier_indices.append(i)
            reasons[i] = f"Negative mobile fraction ({mobile_frac:.1f}%)"
            logging.warning(f"Auto-rejecting {curve.get('filename', 'unknown')}: negative mobile fraction {mobile_frac:.1f}%")
    
    return outlier_indices, reasons

def identify_outliers_iqr(values, multiplier=1.5):
    """
    Identify outliers using the IQR method
    
    Parameters:
    -----------
    values : array-like
        Values to check for outliers
    multiplier : float
        Multiplier for IQR (default 1.5)
        
    Returns:
    --------
    tuple
        (is_outlier, lower_bound, upper_bound)
    """
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    
    # Handle case where IQR = 0 (all values identical)
    if iqr == 0:
        return np.zeros_like(values, dtype=bool), q1, q3
    
    lower_bound = q1 - (multiplier * iqr)
    upper_bound = q3 + (multiplier * iqr)
    
    is_outlier = (values < lower_bound) | (values > upper_bound)
    
    return is_outlier, lower_bound, upper_bound

def identify_outliers_zscore(values, threshold=3.0):
    """
    Identify outliers using Z-score method
    
    Parameters:
    -----------
    values : array-like
        Values to check for outliers
    threshold : float
        Z-score threshold for outliers (default 3.0)
        
    Returns:
    --------
    array
        Boolean array indicating which values are outliers
    """
    z_scores = np.abs(stats.zscore(values))
    return z_scores > threshold

def identify_outliers_mad(values, threshold=3.5):
    """
    Identify outliers using Median Absolute Deviation (MAD) method
    
    Parameters:
    -----------
    values : array-like
        Values to check for outliers
    threshold : float
        MAD threshold for outliers (default 3.5)
        
    Returns:
    --------
    array
        Boolean array indicating which values are outliers
    """
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    
    # Handle case where MAD = 0 (all values identical)
    if mad == 0:
        return np.zeros_like(values, dtype=bool)
    
    # Convert MAD to standard deviation estimate (for normal distribution)
    mad_to_std = 1.4826
    z_scores = mad_to_std * np.abs(values - median) / mad
    
    return z_scores > threshold

def identify_curve_outliers(curves, method='iqr', feature='half_time', threshold=1.5, auto_reject_biological=True):
    """
    Identify outlier curves based on a specific feature
    
    Parameters:
    -----------
    curves : list
        List of curve dictionaries with extracted features
    method : str
        Method for outlier detection ('iqr', 'zscore', or 'mad')
    feature : str
        Feature to use for outlier detection (e.g., 'half_time', 'mobile_fraction')
    threshold : float
        Threshold for outlier detection
    auto_reject_biological : bool
        If True, automatically reject biologically impossible values (e.g., mobile fraction >100%)
        
    Returns:
    --------
    tuple
        (outlier_indices, feature_values, biological_reasons) where biological_reasons is a dict
    """
    # First, check for biological outliers if requested
    biological_outliers = []
    biological_reasons = {}
    if auto_reject_biological:
        biological_outliers, biological_reasons = identify_biological_outliers(curves)
        if biological_outliers:
            logging.info(f"Auto-rejected {len(biological_outliers)} curves with biologically impossible values")
    
    # Extract feature values from curves
    feature_values = []
    for curve in curves:
        if curve.get('clustering_features') and feature in curve['clustering_features']:
            # Handle NaN values
            value = curve['clustering_features'][feature]
            if not np.isnan(value):
                feature_values.append(value)
            else:
                feature_values.append(None)
        else:
            feature_values.append(None)
            
    # Remove None values and biological outliers for statistical outlier detection
    valid_indices = [i for i, v in enumerate(feature_values) 
                    if v is not None and i not in biological_outliers]
    valid_values = [feature_values[i] for i in valid_indices]
    
    if len(valid_values) < 3:
        # Not enough data for statistical outlier detection, but still return biological outliers
        return biological_outliers, feature_values, biological_reasons
        
    # Detect statistical outliers based on selected method
    if method == 'iqr':
        is_outlier, _, _ = identify_outliers_iqr(valid_values, multiplier=threshold)
    elif method == 'zscore':
        is_outlier = identify_outliers_zscore(valid_values, threshold=threshold)
    elif method == 'mad':
        is_outlier = identify_outliers_mad(valid_values, threshold=threshold)
    else:
        # Default to IQR
        is_outlier, _, _ = identify_outliers_iqr(valid_values, multiplier=threshold)
        
    # Map statistical outliers back to original indices
    statistical_outliers = [valid_indices[i] for i, is_out in enumerate(is_outlier) if is_out]
    
    # Combine biological and statistical outliers
    all_outliers = list(set(biological_outliers + statistical_outliers))
    
    return all_outliers, feature_values, biological_reasons

def remove_outliers(curves, outlier_indices):
    """
    Remove outlier curves from a list
    
    Parameters:
    -----------
    curves : list
        List of curve dictionaries
    outlier_indices : list
        Indices of outlier curves to remove
        
    Returns:
    --------
    list
        List of curves with outliers removed
    """
    return [curve for i, curve in enumerate(curves) if i not in outlier_indices]

def get_outlier_statistics(curves, outlier_indices, feature_values):
    """
    Get statistics about outliers
    
    Parameters:
    -----------
    curves : list
        List of curve dictionaries
    outlier_indices : list
        Indices of outlier curves
    feature_values : list
        Feature values used for outlier detection
        
    Returns:
    --------
    dict
        Dictionary of outlier statistics
    """
    stats_dict = {
        'total_curves': len(curves),
        'valid_curves': sum(1 for v in feature_values if v is not None),
        'outlier_count': len(outlier_indices),
        'outlier_percent': len(outlier_indices) / len(curves) * 100 if len(curves) > 0 else 0,
        'outlier_files': [curves[i]['filename'] for i in outlier_indices]
    }
    
    return stats_dict
