"""
FRAP Statistics Module
Linear mixed models, bootstrap confidence intervals, and effect sizes
"""
import numpy as np
import pandas as pd
from typing import Optional, Union
import logging
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)


def lmm_param(
    df: pd.DataFrame,
    param: str,
    group_col: str = "condition",
    batch_col: str = "exp_id"
) -> dict:
    """
    Fit linear mixed-effects model for a parameter
    
    Model: param ~ 1 + group_col with random intercept by batch_col
    
    Parameters
    ----------
    df : pd.DataFrame
        Cell features with group and batch columns
    param : str
        Parameter to analyze
    group_col : str
        Column for group/condition
    batch_col : str
        Column for batch/experiment ID
        
    Returns
    -------
    dict
        Beta, SE, 95% CI, p-value, effect size, omega_sq
    """
    # Prepare data
    data = df[[param, group_col, batch_col]].dropna().copy()
    
    if len(data) < 10:
        logger.warning(f"Insufficient data for LMM on {param}")
        return {
            'param': param,
            'success': False,
            'message': 'Insufficient data'
        }
    
    # Ensure group is categorical
    data[group_col] = data[group_col].astype(str)
    groups = sorted(data[group_col].unique())
    
    if len(groups) < 2:
        logger.warning(f"Need at least 2 groups for {param}")
        return {
            'param': param,
            'success': False,
            'message': 'Need at least 2 groups'
        }
    
    try:
        # Fit mixed model
        formula = f"{param} ~ C({group_col})"
        model = smf.mixedlm(
            formula,
            data=data,
            groups=data[batch_col],
            re_formula="1"
        )
        result = model.fit(method='lbfgs', maxiter=100)
        
        # Extract results
        params = result.params
        pvalues = result.pvalues
        conf_int = result.conf_int()
        
        # Build results
        results = {
            'param': param,
            'success': True,
            'groups': groups,
            'n_total': len(data),
            'n_groups': len(groups)
        }
        
        # Intercept (reference group)
        results['intercept'] = params['Intercept']
        results['intercept_se'] = result.bse['Intercept']
        results['intercept_ci'] = [conf_int.loc['Intercept', 0], conf_int.loc['Intercept', 1]]
        
        # Group comparisons (vs reference)
        for i, group in enumerate(groups[1:], 1):
            key = f"C({group_col}, Treatment('{groups[0]}'))[T.{group}]"
            
            if key in params:
                beta = params[key]
                se = result.bse[key]
                p = pvalues[key]
                ci = [conf_int.loc[key, 0], conf_int.loc[key, 1]]
                
                # Effect size (Cohen's d approximation from t-statistic)
                t_stat = beta / se if se > 0 else 0
                n1 = len(data[data[group_col] == groups[0]])
                n2 = len(data[data[group_col] == group])
                cohens_d = t_stat * np.sqrt((n1 + n2) / (n1 * n2))
                
                # Hedges' g correction
                correction = 1 - (3 / (4 * (n1 + n2) - 9))
                hedges_g = cohens_d * correction
                
                results[f'beta_{group}'] = beta
                results[f'se_{group}'] = se
                results[f'p_{group}'] = p
                results[f'ci_{group}'] = ci
                results[f'hedges_g_{group}'] = hedges_g
                results[f'n_{group}'] = n2
        
        results['n_' + groups[0]] = len(data[data[group_col] == groups[0]])
        
        # Compute omega-squared (effect size for overall model)
        # For multi-group, use variance explained
        residual_var = result.scale
        random_var = float(result.cov_re.iloc[0, 0]) if hasattr(result, 'cov_re') else 0
        total_var = residual_var + random_var
        
        # Omega-squared approximation
        if len(groups) > 2:
            # Compute variance explained by fixed effects
            y_mean = data[param].mean()
            y_pred_fe = result.fittedvalues
            ss_model = np.sum((y_pred_fe - y_mean) ** 2)
            ss_total = np.sum((data[param] - y_mean) ** 2)
            omega_sq = 1 - (result.scale / (ss_total / len(data)))
            omega_sq = max(0, omega_sq)  # Can't be negative
            results['omega_squared'] = omega_sq
        
        # AIC/BIC
        results['aic'] = result.aic
        results['bic'] = result.bic
        
        return results
        
    except Exception as e:
        logger.error(f"LMM failed for {param}: {e}")
        return {
            'param': param,
            'success': False,
            'message': str(e)
        }


def bootstrap_bca_ci(
    data: np.ndarray,
    statistic_func: callable,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: int = 0
) -> tuple[float, tuple[float, float]]:
    """
    Compute bias-corrected and accelerated (BCa) bootstrap confidence interval
    
    Parameters
    ----------
    data : np.ndarray
        Data array
    statistic_func : callable
        Function to compute statistic
    n_bootstrap : int
        Number of bootstrap samples
    confidence : float
        Confidence level
    random_state : int
        Random seed
        
    Returns
    -------
    tuple[float, tuple[float, float]]
        (point_estimate, (lower, upper))
    """
    rng = np.random.RandomState(random_state)
    n = len(data)
    
    # Point estimate
    theta_hat = statistic_func(data)
    
    # Bootstrap replicates
    theta_boot = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        indices = rng.randint(0, n, size=n)
        theta_boot[i] = statistic_func(data[indices])
    
    # Bias correction
    z0 = stats.norm.ppf(np.mean(theta_boot < theta_hat))
    
    # Acceleration
    theta_jack = np.zeros(n)
    for i in range(n):
        indices = np.concatenate([np.arange(i), np.arange(i+1, n)])
        theta_jack[i] = statistic_func(data[indices])
    
    theta_jack_mean = np.mean(theta_jack)
    num = np.sum((theta_jack_mean - theta_jack) ** 3)
    den = 6 * (np.sum((theta_jack_mean - theta_jack) ** 2) ** 1.5)
    a = num / den if den != 0 else 0
    
    # BCa percentiles
    alpha = 1 - confidence
    z_alpha_lower = stats.norm.ppf(alpha / 2)
    z_alpha_upper = stats.norm.ppf(1 - alpha / 2)
    
    p_lower = stats.norm.cdf(z0 + (z0 + z_alpha_lower) / (1 - a * (z0 + z_alpha_lower)))
    p_upper = stats.norm.cdf(z0 + (z0 + z_alpha_upper) / (1 - a * (z0 + z_alpha_upper)))
    
    # Compute CI
    lower = np.percentile(theta_boot, p_lower * 100)
    upper = np.percentile(theta_boot, p_upper * 100)
    
    return theta_hat, (lower, upper)


def bootstrap_group_comparison(
    data1: np.ndarray,
    data2: np.ndarray,
    statistic_func: callable = np.mean,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: int = 0
) -> dict:
    """
    Bootstrap comparison of two groups
    
    Parameters
    ----------
    data1, data2 : np.ndarray
        Data for two groups
    statistic_func : callable
        Function to compute statistic
    n_bootstrap : int
        Number of bootstrap samples
    confidence : float
        Confidence level
    random_state : int
        Random seed
        
    Returns
    -------
    dict
        Statistics and confidence intervals
    """
    # Point estimates
    stat1, ci1 = bootstrap_bca_ci(data1, statistic_func, n_bootstrap, confidence, random_state)
    stat2, ci2 = bootstrap_bca_ci(data2, statistic_func, n_bootstrap, confidence, random_state)
    
    # Difference
    def diff_func(combined):
        n1 = len(data1)
        return statistic_func(combined[:n1]) - statistic_func(combined[n1:])
    
    combined = np.concatenate([data1, data2])
    diff, diff_ci = bootstrap_bca_ci(combined, diff_func, n_bootstrap, confidence, random_state)
    
    # Effect size
    pooled_std = np.sqrt((np.var(data1) * (len(data1) - 1) + np.var(data2) * (len(data2) - 1)) / 
                         (len(data1) + len(data2) - 2))
    cohens_d = (stat1 - stat2) / pooled_std if pooled_std > 0 else 0
    
    # Hedges' g correction
    correction = 1 - (3 / (4 * (len(data1) + len(data2)) - 9))
    hedges_g = cohens_d * correction
    
    return {
        'group1_mean': stat1,
        'group1_ci': ci1,
        'group1_n': len(data1),
        'group2_mean': stat2,
        'group2_ci': ci2,
        'group2_n': len(data2),
        'difference': diff,
        'difference_ci': diff_ci,
        'cohens_d': cohens_d,
        'hedges_g': hedges_g
    }


def analyze_parameter_across_groups(
    df: pd.DataFrame,
    param: str,
    group_col: str = "condition",
    batch_col: str = "exp_id",
    n_bootstrap: int = 1000,
    random_state: int = 0
) -> dict:
    """
    Complete analysis of one parameter across groups
    
    Parameters
    ----------
    df : pd.DataFrame
        Cell features
    param : str
        Parameter to analyze
    group_col : str
        Group column
    batch_col : str
        Batch column
    n_bootstrap : int
        Bootstrap iterations
    random_state : int
        Random seed
        
    Returns
    -------
    dict
        Complete analysis results
    """
    # LMM analysis
    lmm_results = lmm_param(df, param, group_col, batch_col)
    
    # Bootstrap for each group
    groups = sorted(df[group_col].unique())
    bootstrap_results = {}
    
    for group in groups:
        group_data = df[df[group_col] == group][param].dropna().values
        if len(group_data) > 0:
            mean, ci = bootstrap_bca_ci(
                group_data,
                np.mean,
                n_bootstrap,
                0.95,
                random_state
            )
            bootstrap_results[group] = {
                'mean': mean,
                'ci': ci,
                'n': len(group_data),
                'std': np.std(group_data),
                'median': np.median(group_data)
            }
    
    return {
        'param': param,
        'lmm': lmm_results,
        'bootstrap': bootstrap_results,
        'groups': groups
    }


def multi_parameter_analysis(
    df: pd.DataFrame,
    params: list[str],
    group_col: str = "condition",
    batch_col: str = "exp_id",
    fdr_method: str = "fdr_bh",
    n_bootstrap: int = 1000,
    random_state: int = 0
) -> pd.DataFrame:
    """
    Analyze multiple parameters with FDR correction
    
    Parameters
    ----------
    df : pd.DataFrame
        Cell features
    params : list[str]
        Parameters to analyze
    group_col : str
        Group column
    batch_col : str
        Batch column
    fdr_method : str
        FDR correction method
    n_bootstrap : int
        Bootstrap iterations
    random_state : int
        Random seed
        
    Returns
    -------
    pd.DataFrame
        Results table with FDR-adjusted p-values
    """
    results = []
    
    for param in params:
        if param not in df.columns:
            logger.warning(f"Parameter {param} not in dataframe")
            continue
        
        result = analyze_parameter_across_groups(
            df, param, group_col, batch_col, n_bootstrap, random_state
        )
        
        if result['lmm']['success']:
            lmm = result['lmm']
            
            # Extract pairwise comparisons
            groups = lmm['groups']
            for group in groups[1:]:
                p_key = f'p_{group}'
                if p_key in lmm:
                    row = {
                        'param': param,
                        'comparison': f"{groups[0]}_vs_{group}",
                        'beta': lmm[f'beta_{group}'],
                        'se': lmm[f'se_{group}'],
                        'p': lmm[p_key],
                        'ci_lower': lmm[f'ci_{group}'][0],
                        'ci_upper': lmm[f'ci_{group}'][1],
                        'hedges_g': lmm[f'hedges_g_{group}'],
                        'n_ref': lmm[f'n_{groups[0]}'],
                        'n_comp': lmm[f'n_{group}']
                    }
                    results.append(row)
    
    if not results:
        logger.warning("No successful analyses")
        return pd.DataFrame()
    
    results_df = pd.DataFrame(results)
    
    # FDR correction
    if len(results_df) > 1:
        reject, pvals_corrected, _, _ = multipletests(
            results_df['p'].values,
            alpha=0.05,
            method=fdr_method
        )
        results_df['q'] = pvals_corrected
        results_df['significant'] = reject
    else:
        results_df['q'] = results_df['p']
        results_df['significant'] = results_df['p'] < 0.05
    
    logger.info(f"Completed analysis of {len(params)} parameters with FDR correction")
    
    return results_df


def compute_effect_size_ci(
    data1: np.ndarray,
    data2: np.ndarray,
    n_bootstrap: int = 1000,
    random_state: int = 0
) -> dict:
    """
    Compute effect size with bootstrap CI
    
    Parameters
    ----------
    data1, data2 : np.ndarray
        Data for two groups
    n_bootstrap : int
        Bootstrap iterations
    random_state : int
        Random seed
        
    Returns
    -------
    dict
        Effect size with CI
    """
    # Point estimate
    mean1, mean2 = np.mean(data1), np.mean(data2)
    pooled_std = np.sqrt((np.var(data1) * (len(data1) - 1) + np.var(data2) * (len(data2) - 1)) / 
                         (len(data1) + len(data2) - 2))
    cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
    
    # Hedges' g
    correction = 1 - (3 / (4 * (len(data1) + len(data2)) - 9))
    hedges_g = cohens_d * correction
    
    # Bootstrap CI for effect size
    rng = np.random.RandomState(random_state)
    effect_sizes = np.zeros(n_bootstrap)
    
    for i in range(n_bootstrap):
        idx1 = rng.randint(0, len(data1), len(data1))
        idx2 = rng.randint(0, len(data2), len(data2))
        
        boot_data1 = data1[idx1]
        boot_data2 = data2[idx2]
        
        boot_mean1 = np.mean(boot_data1)
        boot_mean2 = np.mean(boot_data2)
        boot_pooled_std = np.sqrt(
            (np.var(boot_data1) * (len(boot_data1) - 1) + 
             np.var(boot_data2) * (len(boot_data2) - 1)) / 
            (len(boot_data1) + len(boot_data2) - 2)
        )
        
        if boot_pooled_std > 0:
            boot_cohens = (boot_mean1 - boot_mean2) / boot_pooled_std
            effect_sizes[i] = boot_cohens * correction
        else:
            effect_sizes[i] = 0
    
    ci_lower = np.percentile(effect_sizes, 2.5)
    ci_upper = np.percentile(effect_sizes, 97.5)
    
    return {
        'cohens_d': cohens_d,
        'hedges_g': hedges_g,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }


def permutation_test(
    data1: np.ndarray,
    data2: np.ndarray,
    n_permutations: int = 10000,
    random_state: int = 0
) -> float:
    """
    Permutation test for difference in means
    
    Parameters
    ----------
    data1, data2 : np.ndarray
        Data for two groups
    n_permutations : int
        Number of permutations
    random_state : int
        Random seed
        
    Returns
    -------
    float
        P-value
    """
    observed_diff = np.mean(data1) - np.mean(data2)
    
    combined = np.concatenate([data1, data2])
    n1 = len(data1)
    
    rng = np.random.RandomState(random_state)
    perm_diffs = np.zeros(n_permutations)
    
    for i in range(n_permutations):
        rng.shuffle(combined)
        perm_diffs[i] = np.mean(combined[:n1]) - np.mean(combined[n1:])
    
    p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
    
    return p_value
