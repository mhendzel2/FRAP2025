"""
FRAP Utilities Module
Helper functions for the FRAP Analysis application
"""

import pandas as pd
import numpy as np
import base64
import io
from scipy import stats
import logging

logger = logging.getLogger(__name__)

def get_parameter_description(param_name):
    """
    Get description for model parameters
    
    Parameters:
    -----------
    param_name : str
        Parameter name
        
    Returns:
    --------
    str
        Description of the parameter
    """
    descriptions = {
        'amplitude': 'The total fluorescence recovery amplitude',
        'rate_constant': 'The rate of fluorescence recovery',
        'amplitude_1': 'The amplitude of the first (fast) component',
        'rate_constant_1': 'The rate constant of the first (fast) component',
        'amplitude_2': 'The amplitude of the second (slower) component',
        'rate_constant_2': 'The rate constant of the second (slower) component',
        'amplitude_3': 'The amplitude of the third (slowest) component',
        'rate_constant_3': 'The rate constant of the third (slowest) component',
    'mobile_fraction': 'Mobile population (%) = (1 - (ΣA + C))*100; NaN if curve not at plateau',
        'half_time': 'The time required to reach half of the final recovery level'
    }
    
    return descriptions.get(param_name, 'No description available')

def format_parameter_value(name, value):
    """
    Format parameter value with appropriate units
    
    Parameters:
    -----------
    name : str
        Parameter name
    value : float
        Parameter value
        
    Returns:
    --------
    str or float
        Formatted value (as string with units or as float for DataFrame operations)
    """
    if np.isnan(value):
        return "N/A"
    
    # For use in DataFrames (to avoid dtype issues), return numeric values
    if isinstance(name, str) and 'mobile_fraction' in name:
        return float(value*100) if not np.isnan(value) else np.nan
    
    # For display purposes, return formatted strings with units
    if isinstance(name, str):
        if 'rate_constant' in name:
            return f"{value:.4f} s⁻¹"
        elif 'half_time' in name:
            return f"{value:.2f} s"
        elif 'mobile_fraction' in name:
            return f"{value*100:.1f}%"
        elif 'proportion' in name:
            return f"{value*100:.1f}%"
        elif 'diffusion' in name:
            return f"{value:.4f} µm²/s"
        elif 'radius' in name or 'gyration' in name:
            return f"{value:.2f} nm"
        elif 'molecular' in name or 'weight' in name:
            return f"{value:.1f} kDa"
        elif 'amplitude' in name:
            return f"{value:.4f}"
    
    # Default for numeric tables
    return float(f"{value:.4f}") if not np.isnan(value) else np.nan

def format_display_value(name, value):
    """
    Format parameter value with appropriate units for display only
    
    Parameters:
    -----------
    name : str
        Parameter name
    value : float
        Parameter value
        
    Returns:
    --------
    str
        Formatted value with units
    """
    if np.isnan(value):
        return "N/A"
        
    if 'rate_constant' in name:
        return f"{value:.4f} s⁻¹"
    elif 'half_time' in name:
        return f"{value:.2f} s"
    elif 'mobile_fraction' in name:
        return f"{value*100:.1f}%"
    elif 'proportion' in name:
        return f"{value*100:.1f}%"
    elif 'diffusion' in name:
        return f"{value:.4f} µm²/s"
    elif 'radius' in name or 'gyration' in name:
        return f"{value:.2f} nm"
    elif 'molecular' in name or 'weight' in name:
        return f"{value:.1f} kDa"
    elif 'amplitude' in name:
        return f"{value:.4f}"
    else:
        return f"{value:.4f}"

def generate_model_equation(model_type):
    """
    Generate LaTeX formula for model equation
    
    Parameters:
    -----------
    model_type : str
        Model type ('single', 'double', 'triple')
        
    Returns:
    --------
    str
        LaTeX formula for the model
    """
    if model_type == 'single':
        return r"F(t) = A \cdot (1 - e^{-k \cdot t}) + C"
    elif model_type == 'double':
        return r"F(t) = A_1 \cdot (1 - e^{-k_1 \cdot t}) + A_2 \cdot (1 - e^{-k_2 \cdot t}) + C"
    elif model_type == 'triple':
        return r"F(t) = A_1 \cdot (1 - e^{-k_1 \cdot t}) + A_2 \cdot (1 - e^{-k_2 \cdot t}) + A_3 \cdot (1 - e^{-k_3 \cdot t}) + C"
    else:
        return "Unknown model type"

def perform_statistical_tests(group1_data, group2_data, feature_name):
    """
    Perform statistical tests to compare two groups
    
    Parameters:
    -----------
    group1_data : pandas.Series
        Data from first group
    group2_data : pandas.Series
        Data from second group
    feature_name : str
        Name of the feature being compared
        
    Returns:
    --------
    dict
        Dictionary with test results
    """
    results = {}
    
    try:
        # Check normality of data (Shapiro-Wilk test)
        if len(group1_data) >= 3:
            _, p_norm1 = stats.shapiro(group1_data)
            results['shapiro_group1'] = {'p_value': p_norm1, 'normal': p_norm1 > 0.05}
        else:
            results['shapiro_group1'] = {'p_value': None, 'normal': None}
            
        if len(group2_data) >= 3:
            _, p_norm2 = stats.shapiro(group2_data)
            results['shapiro_group2'] = {'p_value': p_norm2, 'normal': p_norm2 > 0.05}
        else:
            results['shapiro_group2'] = {'p_value': None, 'normal': None}
        
        # Determine whether to use parametric or non-parametric test
        normal_dist = (results['shapiro_group1']['normal'] is True and 
                       results['shapiro_group2']['normal'] is True)
        
        # Perform appropriate test
        if normal_dist:
            # Check equal variance (Levene's test)
            _, p_var = stats.levene(group1_data, group2_data)
            results['levene'] = {'p_value': p_var, 'equal_var': p_var > 0.05}
            
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(
                group1_data, 
                group2_data, 
                equal_var=results['levene']['equal_var']
            )
            test_name = "t-test (equal variance)" if results['levene']['equal_var'] else "Welch's t-test"
            results['parametric_test'] = {
                'name': test_name,
                'statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        else:
            # Perform Mann-Whitney U test
            u_stat, p_value = stats.mannwhitneyu(group1_data, group2_data)
            results['nonparametric_test'] = {
                'name': "Mann-Whitney U test",
                'statistic': u_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        
        # Calculate effect size (Cohen's d for parametric, rank-biserial correlation for non-parametric)
        if normal_dist:
            # Cohen's d
            mean1, mean2 = group1_data.mean(), group2_data.mean()
            sd1, sd2 = group1_data.std(), group2_data.std()
            n1, n2 = len(group1_data), len(group2_data)
            
            # Pooled standard deviation
            sd_pooled = np.sqrt(((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / (n1 + n2 - 2))
            
            # Cohen's d
            d = (mean2 - mean1) / sd_pooled
            results['effect_size'] = {
                'name': "Cohen's d",
                'value': d,
                'interpretation': interpret_cohens_d(d)
            }
        else:
            # For non-parametric, we calculate rank-biserial correlation
            r = 1 - (2 * u_stat) / (len(group1_data) * len(group2_data))
            results['effect_size'] = {
                'name': "Rank-biserial correlation",
                'value': r,
                'interpretation': interpret_rank_biserial(r)
            }
        
        results['summary'] = {
            'feature': feature_name,
            'test_used': results['parametric_test']['name'] if normal_dist else results['nonparametric_test']['name'],
            'p_value': results['parametric_test']['p_value'] if normal_dist else results['nonparametric_test']['p_value'],
            'significant': results['parametric_test']['significant'] if normal_dist else results['nonparametric_test']['significant'],
            'effect_size': results['effect_size']['name'] + ": " + str(round(results['effect_size']['value'], 3)),
            'effect_interpretation': results['effect_size']['interpretation']
        }
        
        return results
        
    except Exception as e:
        logger.error(f"Error performing statistical tests: {e}")
        return {'error': str(e)}

def interpret_cohens_d(d):
    """
    Interpret Cohen's d effect size
    
    Parameters:
    -----------
    d : float
        Cohen's d value
        
    Returns:
    --------
    str
        Interpretation of effect size
    """
    d = abs(d)
    if d < 0.2:
        return "Negligible effect"
    elif d < 0.5:
        return "Small effect"
    elif d < 0.8:
        return "Medium effect"
    else:
        return "Large effect"

def interpret_rank_biserial(r):
    """
    Interpret rank-biserial correlation effect size
    
    Parameters:
    -----------
    r : float
        Rank-biserial correlation value
        
    Returns:
    --------
    str
        Interpretation of effect size
    """
    r = abs(r)
    if r < 0.2:
        return "Negligible effect"
    elif r < 0.4:
        return "Small effect"
    elif r < 0.6:
        return "Medium effect"
    else:
        return "Large effect"

def to_csv_download_link(df, filename="data.csv"):
    """
    Generate a download link for a DataFrame as CSV
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to convert
    filename : str
        Name for the downloaded file
        
    Returns:
    --------
    str
        HTML link for downloading the CSV
    """
    csv = df.to_csv(index=True)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV</a>'
    return href

def to_excel_download_link(df, filename="data.xlsx"):
    """
    Generate a download link for a DataFrame as Excel
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to convert
    filename : str
        Name for the downloaded file
        
    Returns:
    --------
    str
        HTML link for downloading the Excel file
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Sheet1')
    excel_data = output.getvalue()
    b64 = base64.b64encode(excel_data).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">Download Excel</a>'
    return href
