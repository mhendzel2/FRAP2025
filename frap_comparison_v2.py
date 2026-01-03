import pandas as pd
import numpy as np
from scipy import stats
import logging
from frap_global_fitting import GlobalFitter, MODEL_REGISTRY

logger = logging.getLogger(__name__)

class UnifiedGroupComparator:
    """
    Manages the comparison workflow:
    1. Pool Controls -> Fit -> Determine Best Model.
    2. Fit Samples using THAT specific model.
    3. Generate stats and formatted data for plotting.
    """
    
    def __init__(self, data_manager):
        self.dm = data_manager
        self.fitter = GlobalFitter()

    def get_group_curve_data(self, group_name: str) -> dict:
        """Return curve data for a group in a flexible way.

        Supports two shapes:
        - data_manager.groups[group]['data'] -> {curve_id: {'time':..., 'intensity':...}}
        - data_manager.groups[group]['files'] + data_manager.files[...] (Streamlit FRAPDataManager)
        """
        group = getattr(self.dm, 'groups', {}).get(group_name, {})
        if isinstance(group, dict) and isinstance(group.get('data'), dict):
            return group.get('data', {})

        curve_data: dict = {}
        files = group.get('files', []) if isinstance(group, dict) else []
        dm_files = getattr(self.dm, 'files', {})
        for file_path in files:
            file_entry = dm_files.get(file_path)
            if not isinstance(file_entry, dict):
                continue
            if 'time' in file_entry and 'intensity' in file_entry:
                curve_data[file_path] = {
                    'time': file_entry['time'],
                    'intensity': file_entry['intensity'],
                }
        return curve_data
        
    def pool_controls(self, control_group_names):
        """
        Creates a virtual 'Pooled Control' group from multiple selected groups.
        Returns the data dictionary for the pooled group.
        """
        pooled_data = {}
        for grp_name in control_group_names:
            grp_data = self.get_group_curve_data(grp_name)
            pooled_data.update(grp_data)

        return pooled_data

    def determine_best_model(self, data_dict):
        """
        Fits Single, Double, and Triple exponential models to the data.
        Returns the name of the best model based on AIC.
        """
        results = {}
        best_model = 'single'
        min_aic = np.inf
        
        # Prepare data for fitter
        # GlobalFitter expects: {curve_id: (time, intensity)}
        fit_data = {
            k: (v['time'], v['intensity']) 
            for k, v in data_dict.items() 
            if 'time' in v and 'intensity' in v
        }
        
        if not fit_data:
            return 'single'

        for model in ['single', 'double']: # Triple often overfits, sticking to standard 2
            try:
                # Create params
                params = self.fitter._create_parameters(model, list(fit_data.keys()))
                # Minimize
                from lmfit import minimize
                res = minimize(self.fitter._global_objective, params, 
                             args=(fit_data, model), method='leastsq')
                
                if res.aic < min_aic:
                    min_aic = res.aic
                    best_model = model
            except Exception as e:
                logger.error(f"Model selection failed for {model}: {e}")
                
        return best_model

    def fit_group_with_model(self, group_name, data_dict, model_name):
        """
        Fits a specific group using a specific ENFORCED model.
        Returns a DataFrame of parameters for every curve in the group.
        """
        fit_data = {
            k: (v['time'], v['intensity']) 
            for k, v in data_dict.items() 
            if 'time' in v and 'intensity' in v
        }
        
        params = self.fitter._create_parameters(model_name, list(fit_data.keys()))
        
        from lmfit import minimize
        result = minimize(self.fitter._global_objective, params, 
                        args=(fit_data, model_name), method='leastsq')
        
        # Extract per-curve parameters
        rows = []
        pvals = result.params.valuesdict()
        
        for curve_id in fit_data.keys():
            safe_cid = self.fitter._sanitize_id(curve_id)
            
            # Common params
            row = {'Filename': curve_id, 'Group': group_name, 'Model': model_name}
            
            # Extract Mobile Fraction
            if f'Mf_{safe_cid}' in pvals:
                row['mobile_fraction'] = pvals[f'Mf_{safe_cid}']
            else:
                row['mobile_fraction'] = pvals.get('Mf_global', 0)
                
            # Extract Kinetics based on model
            if model_name == 'single':
                row['k'] = pvals['k_global']
                row['D'] = (1.0**2 * row['k']) / 4.0 # Assuming r=1um standard, scaling handled elsewhere
                
            elif model_name == 'double':
                row['k_fast'] = pvals['k_fast_global']
                row['k_slow'] = pvals['k_slow_global']
                row['F_fast'] = pvals['F_fast_global']
                row['F_slow'] = 1.0 - row['F_fast']
                # Diffusion usually dominated by fast
                row['D'] = (1.0**2 * row['k_fast']) / 4.0 
                
            rows.append(row)
            
        return pd.DataFrame(rows)

    def calculate_pairwise_stats(self, comparison_df, control_group_name, metrics=None, alpha: float = 0.05):
        """Calculate pairwise Welch t-tests for all groups vs a control group.

        Also reports an overall one-way ANOVA p-value per metric when 3+ groups
        have sufficient observations.
        """
        stats_results = []

        default_metrics = ['mobile_fraction', 'D', 'k', 'k_fast', 'k_slow', 'F_fast', 'F_slow']
        metrics = metrics or default_metrics

        if 'Group' not in comparison_df.columns:
            return pd.DataFrame()

        control_data = comparison_df[comparison_df['Group'] == control_group_name]
        if control_data.empty:
            return pd.DataFrame()

        all_groups = [g for g in comparison_df['Group'].unique() if pd.notna(g)]

        for metric in metrics:
            if metric not in comparison_df.columns:
                continue

            # Overall ANOVA across all groups (if possible)
            anova_p = np.nan
            group_vectors = []
            for g in all_groups:
                vals = comparison_df.loc[comparison_df['Group'] == g, metric].dropna()
                if len(vals) > 1:
                    group_vectors.append(vals.values)
            if len(group_vectors) >= 3:
                try:
                    _, anova_p = stats.f_oneway(*group_vectors)
                except Exception:
                    anova_p = np.nan

            # Pairwise tests vs control
            c_vals = control_data[metric].dropna()
            if len(c_vals) <= 1:
                continue

            for group in all_groups:
                if group == control_group_name:
                    continue
                g_vals = comparison_df.loc[comparison_df['Group'] == group, metric].dropna()
                if len(g_vals) <= 1:
                    continue
                try:
                    _, p_val = stats.ttest_ind(c_vals, g_vals, equal_var=False)
                except Exception:
                    continue

                stats_results.append({
                    'Metric': metric,
                    'Reference': control_group_name,
                    'Comparison': group,
                    'Mean Ref': c_vals.mean(),
                    'Mean Comp': g_vals.mean(),
                    't-test p-value': p_val,
                    'Overall ANOVA p-value': anova_p,
                    'Significant': 'Yes' if (pd.notna(p_val) and p_val < alpha) else 'No'
                })

        return pd.DataFrame(stats_results)