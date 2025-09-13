import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm
from statsmodels.stats.multitest import multipletests
import pingouin as pg
from itertools import combinations

def _welch_t_test(data, metric, group1, group2):
    """Perform Welch's t-test."""
    group1_data = data[data['group'] == group1][metric].dropna()
    group2_data = data[data['group'] == group2][metric].dropna()
    if len(group1_data) < 2 or len(group2_data) < 2:
        return np.nan, np.nan
    t_stat, p_val = stats.ttest_ind(group1_data, group2_data, equal_var=False)
    return t_stat, p_val

def _one_way_anova(data, metric, groups):
    """Perform one-way ANOVA."""
    groups_data = [data[data['group'] == g][metric].dropna() for g in groups]
    groups_data_valid = [g for g in groups_data if len(g) > 1]
    if len(groups_data_valid) < 2:
        return np.nan, np.nan
    f_stat, p_val = stats.f_oneway(*groups_data_valid)
    return f_stat, p_val

def _permutation_test(data, metric, group1, group2, n_permutations=10000):
    """Perform a permutation test."""
    group1_data = data[data['group'] == group1][metric].dropna()
    group2_data = data[data['group'] == group2][metric].dropna()
    if len(group1_data) < 1 or len(group2_data) < 1:
        return np.nan

    observed_diff = np.mean(group1_data) - np.mean(group2_data)
    combined = np.concatenate([group1_data, group2_data])
    count = 0
    for _ in range(n_permutations):
        perm = np.random.permutation(combined)
        perm_group1 = perm[:len(group1_data)]
        perm_group2 = perm[len(group1_data):]
        perm_diff = np.mean(perm_group1) - np.mean(perm_group2)
        if abs(perm_diff) >= abs(observed_diff):
            count += 1
    return count / n_permutations

def _calculate_effect_sizes(data, metric, group1, group2):
    """Calculate Cohen's d and Cliff's delta."""
    group1_data = data[data['group'] == group1][metric].dropna()
    group2_data = data[data['group'] == group2][metric].dropna()
    if len(group1_data) < 2 or len(group2_data) < 2:
        return np.nan, np.nan

    cohen_d = pg.compute_effsize(group1_data, group2_data, eftype='cohen')
    cliffs_delta = pg.compute_effsize(group1_data, group2_data, eftype='cles')
    return cohen_d, cliffs_delta

def _tost_equivalence(data, metric, group1, group2, low_eq_bound, high_eq_bound):
    """Perform TOST equivalence test."""
    group1_data = data[data['group'] == group1][metric].dropna()
    group2_data = data[data['group'] == group2][metric].dropna()
    if len(group1_data) < 2 or len(group2_data) < 2:
        return np.nan, "N/A"

    tost_results = pg.tost(group1_data, group2_data, bound=high_eq_bound, paired=False)
    p_val = tost_results['pval'].iloc[0]
    result = "Equivalent" if p_val < 0.05 else "Not Equivalent"
    return p_val, result

def _mixed_effects_model(data, metric, group_col='group', random_effect_col='experiment_id'):
    """Fit a mixed-effects model."""
    if random_effect_col not in data.columns:
        return "Random effect column not found in data."

    formula = f"{metric} ~ {group_col}"
    model = mixedlm(formula, data, groups=data[random_effect_col])
    try:
        result = model.fit()
        return str(result.summary())
    except Exception as e:
        return f"Error fitting mixed-effects model: {e}"

def calculate_group_stats(
    data: pd.DataFrame,
    metrics: list,
    group_order: list,
    tost_thresholds: dict = None,
    use_mixed_effects: bool = False,
    random_effect_col: str = 'experiment_id'
):
    """
    Perform robust group statistics.
    """
    results = []

    for metric in metrics:
        if len(group_order) == 2:
            group1, group2 = group_order[0], group_order[1]

            # Welch's t-test
            t_stat, p_welch = _welch_t_test(data, metric, group1, group2)

            # Permutation test
            p_perm = _permutation_test(data, metric, group1, group2)

            # Effect sizes
            cohen_d, cliffs_delta = _calculate_effect_sizes(data, metric, group1, group2)

            # TOST
            if tost_thresholds and metric in tost_thresholds:
                low_eq_bound, high_eq_bound = tost_thresholds[metric]
                p_tost, tost_outcome = _tost_equivalence(data, metric, group1, group2, low_eq_bound, high_eq_bound)
            else:
                p_tost, tost_outcome = np.nan, "N/A"

            results.append({
                'metric': metric,
                'test': 'Welch t-test',
                'p_value': p_welch,
                'p_perm': p_perm,
                'cohen_d': cohen_d,
                'cliffs_delta': cliffs_delta,
                'p_tost': p_tost,
                'tost_outcome': tost_outcome
            })

        elif len(group_order) > 2:
            # One-way ANOVA
            f_stat, p_anova = _one_way_anova(data, metric, group_order)
            results.append({
                'metric': metric,
                'test': 'ANOVA',
                'p_value': p_anova,
            })

            # Pairwise comparisons
            for g1, g2 in combinations(group_order, 2):
                t_stat, p_welch = _welch_t_test(data, metric, g1, g2)
                p_perm = _permutation_test(data, metric, g1, g2)
                cohen_d, cliffs_delta = _calculate_effect_sizes(data, metric, g1, g2)
                if tost_thresholds and metric in tost_thresholds:
                    low_eq_bound, high_eq_bound = tost_thresholds[metric]
                    p_tost, tost_outcome = _tost_equivalence(data, metric, g1, g2, low_eq_bound, high_eq_bound)
                else:
                    p_tost, tost_outcome = np.nan, "N/A"

                results.append({
                    'metric': f"{metric} ({g1} vs {g2})",
                    'test': 'Pairwise Welch t-test',
                    'p_value': p_welch,
                    'p_perm': p_perm,
                    'cohen_d': cohen_d,
                    'cliffs_delta': cliffs_delta,
                    'p_tost': p_tost,
                    'tost_outcome': tost_outcome
                })

    if not results:
        return pd.DataFrame()

    results_df = pd.DataFrame(results)

    # FDR correction
    if not results_df.empty and 'p_value' in results_df.columns:
        results_df['q_value'] = multipletests(results_df['p_value'].dropna(), method='fdr_bh')[1]

    # Mixed-effects model
    if use_mixed_effects:
        mixed_effects_summaries = {}
        for metric in metrics:
            summary = _mixed_effects_model(data, metric, random_effect_col=random_effect_col)
            mixed_effects_summaries[metric] = summary
        results_df['mixed_effects_summary'] = results_df['metric'].map(mixed_effects_summaries)

    return results_df
