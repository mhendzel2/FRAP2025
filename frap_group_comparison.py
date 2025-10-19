"""
FRAP Holistic Group Comparison Module

Provides comprehensive comparison of FRAP kinetics between groups by:
1. Comparing averaged recovery profiles
2. Analyzing population distributions across kinetic components  
3. Computing weighted kinetic metrics that account for component abundance
4. Identifying biological differences (e.g., loss of binding, shift to diffusion)

This addresses the limitation of comparing individual fitted components which can
miss the biological story. Instead, we look at the whole picture.

Author: FRAP Analysis Team
Date: October 2025
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class HolisticGroupComparator:
    """
    Analyzes FRAP kinetics by comparing population distributions and
    weighted kinetic metrics between experimental groups.
    
    This class provides a more biologically meaningful comparison than
    simply looking at individual kinetic components.
    """
    
    # Define typical kinetic regimes (adjust based on bleach spot size)
    DIFFUSION_THRESHOLD = 1.0   # k > 1.0 s⁻¹ typically diffusion-dominated
    BINDING_THRESHOLD = 0.1     # k < 0.1 s⁻¹ typically binding-dominated
    
    def __init__(self, bleach_radius_um=1.0, pixel_size=0.3):
        """
        Initialize holistic comparator.
        
        Parameters
        ----------
        bleach_radius_um : float
            Bleach spot radius in micrometers
        pixel_size : float
            Pixel size in micrometers
        """
        self.bleach_radius = bleach_radius_um
        self.pixel_size = pixel_size
        self.effective_radius = bleach_radius_um * pixel_size
        
    def categorize_kinetics(self, k: float) -> str:
        """
        Categorize a rate constant into kinetic regime.
        
        Parameters
        ----------
        k : float
            Rate constant (s⁻¹)
            
        Returns
        -------
        str
            'diffusion', 'binding', or 'intermediate'
        """
        if k > self.DIFFUSION_THRESHOLD:
            return 'diffusion'
        elif k < self.BINDING_THRESHOLD:
            return 'binding'
        else:
            return 'intermediate'
    
    def compute_weighted_kinetics(self, features_df: pd.DataFrame) -> Dict:
        """
        Compute abundance-weighted kinetic metrics for a group.
        
        This provides a holistic view of the kinetics by weighting each
        component by its relative abundance, rather than treating all
        components equally.
        
        Parameters
        ----------
        features_df : pd.DataFrame
            Features dataframe for a single group
            
        Returns
        -------
        dict
            Weighted kinetic metrics and population distributions
        """
        results = {
            'n_cells': len(features_df),
            'mobile_fraction_mean': np.nan,
            'mobile_fraction_sem': np.nan,
            'weighted_k_fast': np.nan,
            'weighted_k_fast_sem': np.nan,
            'weighted_k_slow': np.nan,
            'weighted_k_slow_sem': np.nan,
            'population_diffusion': 0.0,
            'population_binding': 0.0,
            'population_intermediate': 0.0,
        }
        
        if len(features_df) == 0:
            return results
        
        # Mobile fraction statistics
        if 'mobile_fraction' in features_df.columns:
            mobile = features_df['mobile_fraction'].dropna()
            if len(mobile) > 0:
                results['mobile_fraction_mean'] = mobile.mean()
                results['mobile_fraction_sem'] = mobile.sem()
        
        # Analyze kinetic populations
        populations = {'diffusion': [], 'binding': [], 'intermediate': []}
        k_fast_values = []  # (k, weight) tuples
        k_slow_values = []  # (k, weight) tuples
        
        for _, row in features_df.iterrows():
            model = row.get('model', 'single')
            
            if model == 'single':
                # Single component - classify the single rate
                k = row.get('rate_constant', np.nan)
                if np.isfinite(k):
                    k_fast_values.append((k, 1.0))  # 100% in this component
                    regime = self.categorize_kinetics(k)
                    populations[regime].append(1.0)
                    
            elif model == 'double':
                # Double component - get both rates and their abundances
                k_fast = row.get('rate_constant_fast', np.nan)
                k_slow = row.get('rate_constant_slow', np.nan)
                prop_fast = row.get('proportion_of_mobile_fast', np.nan) / 100.0
                prop_slow = row.get('proportion_of_mobile_slow', np.nan) / 100.0
                
                if np.isfinite(k_fast) and np.isfinite(prop_fast):
                    k_fast_values.append((k_fast, prop_fast))
                    regime = self.categorize_kinetics(k_fast)
                    populations[regime].append(prop_fast)
                
                if np.isfinite(k_slow) and np.isfinite(prop_slow):
                    k_slow_values.append((k_slow, prop_slow))
                    regime = self.categorize_kinetics(k_slow)
                    populations[regime].append(prop_slow)
                    
            elif model == 'triple':
                # Triple component - get all rates and abundances
                k_fast = row.get('rate_constant_fast', np.nan)
                k_med = row.get('rate_constant_medium', np.nan)
                k_slow = row.get('rate_constant_slow', np.nan)
                prop_fast = row.get('proportion_of_mobile_fast', np.nan) / 100.0
                prop_med = row.get('proportion_of_mobile_medium', np.nan) / 100.0
                prop_slow = row.get('proportion_of_mobile_slow', np.nan) / 100.0
                
                if np.isfinite(k_fast) and np.isfinite(prop_fast):
                    k_fast_values.append((k_fast, prop_fast))
                    regime = self.categorize_kinetics(k_fast)
                    populations[regime].append(prop_fast)
                
                if np.isfinite(k_med) and np.isfinite(prop_med):
                    # Medium component could be fast or slow depending on value
                    regime = self.categorize_kinetics(k_med)
                    populations[regime].append(prop_med)
                    
                    if k_med > 0.3:  # Treat as "fast" component
                        k_fast_values.append((k_med, prop_med))
                    else:  # Treat as "slow" component
                        k_slow_values.append((k_med, prop_med))
                
                if np.isfinite(k_slow) and np.isfinite(prop_slow):
                    k_slow_values.append((k_slow, prop_slow))
                    regime = self.categorize_kinetics(k_slow)
                    populations[regime].append(prop_slow)
        
        # Compute population percentages
        total_pop = sum(sum(v) for v in populations.values())
        if total_pop > 0:
            results['population_diffusion'] = (sum(populations['diffusion']) / total_pop) * 100
            results['population_binding'] = (sum(populations['binding']) / total_pop) * 100
            results['population_intermediate'] = (sum(populations['intermediate']) / total_pop) * 100
        
        # Compute abundance-weighted rate constants
        if k_fast_values:
            weighted_k_fast = sum(k * weight for k, weight in k_fast_values) / sum(w for _, w in k_fast_values)
            results['weighted_k_fast'] = weighted_k_fast
            results['weighted_k_fast_sem'] = np.std([k for k, _ in k_fast_values]) / np.sqrt(len(k_fast_values))
        
        if k_slow_values:
            weighted_k_slow = sum(k * weight for k, weight in k_slow_values) / sum(w for _, w in k_slow_values)
            results['weighted_k_slow'] = weighted_k_slow
            results['weighted_k_slow_sem'] = np.std([k for k, _ in k_slow_values]) / np.sqrt(len(k_slow_values))
        
        return results
    
    def compare_groups(self, group_features: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Compare kinetic populations across multiple groups.
        
        Parameters
        ----------
        group_features : dict
            Dictionary mapping group names to their features dataframes
            
        Returns
        -------
        pd.DataFrame
            Comparison table with population metrics for each group
        """
        comparison_data = []
        
        for group_name, features_df in group_features.items():
            metrics = self.compute_weighted_kinetics(features_df)
            metrics['group'] = group_name
            comparison_data.append(metrics)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Reorder columns for readability
        col_order = ['group', 'n_cells', 'mobile_fraction_mean', 'mobile_fraction_sem',
                     'population_diffusion', 'population_binding', 'population_intermediate',
                     'weighted_k_fast', 'weighted_k_fast_sem', 
                     'weighted_k_slow', 'weighted_k_slow_sem']
        
        comparison_df = comparison_df[[c for c in col_order if c in comparison_df.columns]]
        
        return comparison_df
    
    def statistical_comparison(self, 
                               group1_features: pd.DataFrame, 
                               group2_features: pd.DataFrame,
                               group1_name: str = "Group 1",
                               group2_name: str = "Group 2") -> Dict:
        """
        Perform statistical comparison between two groups.
        
        Parameters
        ----------
        group1_features : pd.DataFrame
            Features for group 1
        group2_features : pd.DataFrame
            Features for group 2
        group1_name : str
            Name of group 1
        group2_name : str
            Name of group 2
            
        Returns
        -------
        dict
            Statistical test results including p-values and effect sizes
        """
        results = {
            'group1_name': group1_name,
            'group2_name': group2_name,
            'tests': {}
        }
        
        # Compare mobile fractions
        mf1 = group1_features['mobile_fraction'].dropna()
        mf2 = group2_features['mobile_fraction'].dropna()
        
        if len(mf1) > 0 and len(mf2) > 0:
            t_stat, p_val = stats.ttest_ind(mf1, mf2)
            cohen_d = (mf1.mean() - mf2.mean()) / np.sqrt((mf1.std()**2 + mf2.std()**2) / 2)
            
            results['tests']['mobile_fraction'] = {
                'mean_group1': mf1.mean(),
                'mean_group2': mf2.mean(),
                'sem_group1': mf1.sem(),
                'sem_group2': mf2.sem(),
                't_statistic': t_stat,
                'p_value': p_val,
                'cohen_d': cohen_d,
                'significant': p_val < 0.05
            }
        
        # Compare population distributions
        metrics1 = self.compute_weighted_kinetics(group1_features)
        metrics2 = self.compute_weighted_kinetics(group2_features)
        
        results['population_comparison'] = {
            'diffusion_shift': metrics2['population_diffusion'] - metrics1['population_diffusion'],
            'binding_shift': metrics2['population_binding'] - metrics1['population_binding'],
            'group1_populations': {
                'diffusion': metrics1['population_diffusion'],
                'binding': metrics1['population_binding'],
                'intermediate': metrics1['population_intermediate']
            },
            'group2_populations': {
                'diffusion': metrics2['population_diffusion'],
                'binding': metrics2['population_binding'],
                'intermediate': metrics2['population_intermediate']
            }
        }
        
        # Compare weighted rate constants
        if np.isfinite(metrics1['weighted_k_fast']) and np.isfinite(metrics2['weighted_k_fast']):
            results['kinetics_comparison'] = {
                'k_fast_group1': metrics1['weighted_k_fast'],
                'k_fast_group2': metrics2['weighted_k_fast'],
                'k_fast_fold_change': metrics2['weighted_k_fast'] / metrics1['weighted_k_fast'],
            }
        
        if np.isfinite(metrics1['weighted_k_slow']) and np.isfinite(metrics2['weighted_k_slow']):
            if 'kinetics_comparison' not in results:
                results['kinetics_comparison'] = {}
            results['kinetics_comparison'].update({
                'k_slow_group1': metrics1['weighted_k_slow'],
                'k_slow_group2': metrics2['weighted_k_slow'],
                'k_slow_fold_change': metrics2['weighted_k_slow'] / metrics1['weighted_k_slow'],
            })
        
        return results
    
    def interpret_differences(self, comparison_results: Dict) -> str:
        """
        Generate biological interpretation of differences between groups.
        
        Parameters
        ----------
        comparison_results : dict
            Results from statistical_comparison()
            
        Returns
        -------
        str
            Narrative interpretation of the differences
        """
        interpretation = []
        
        group1 = comparison_results['group1_name']
        group2 = comparison_results['group2_name']
        
        # Mobile fraction interpretation
        if 'mobile_fraction' in comparison_results['tests']:
            mf_test = comparison_results['tests']['mobile_fraction']
            mf_diff = mf_test['mean_group2'] - mf_test['mean_group1']
            
            interpretation.append(f"\n📊 **Mobile Fraction Comparison:**")
            interpretation.append(f"   {group1}: {mf_test['mean_group1']:.1f}% ± {mf_test['sem_group1']:.1f}%")
            interpretation.append(f"   {group2}: {mf_test['mean_group2']:.1f}% ± {mf_test['sem_group2']:.1f}%")
            
            if mf_test['significant']:
                direction = "higher" if mf_diff > 0 else "lower"
                interpretation.append(f"   ✓ {group2} shows significantly {direction} mobility (p={mf_test['p_value']:.4f}, Cohen's d={mf_test['cohen_d']:.2f})")
            else:
                interpretation.append(f"   ✗ No significant difference in overall mobility (p={mf_test['p_value']:.4f})")
        
        # Population distribution interpretation
        if 'population_comparison' in comparison_results:
            pop_comp = comparison_results['population_comparison']
            
            interpretation.append(f"\n🔬 **Population Distribution Analysis:**")
            
            diffusion_shift = pop_comp['diffusion_shift']
            binding_shift = pop_comp['binding_shift']
            
            interpretation.append(f"   {group1}:")
            interpretation.append(f"      Diffusion: {pop_comp['group1_populations']['diffusion']:.1f}%")
            interpretation.append(f"      Binding: {pop_comp['group1_populations']['binding']:.1f}%")
            interpretation.append(f"      Intermediate: {pop_comp['group1_populations']['intermediate']:.1f}%")
            
            interpretation.append(f"   {group2}:")
            interpretation.append(f"      Diffusion: {pop_comp['group2_populations']['diffusion']:.1f}%")
            interpretation.append(f"      Binding: {pop_comp['group2_populations']['binding']:.1f}%")
            interpretation.append(f"      Intermediate: {pop_comp['group2_populations']['intermediate']:.1f}%")
            
            # Interpret major shifts
            if abs(diffusion_shift) > 10:
                direction = "increased" if diffusion_shift > 0 else "decreased"
                interpretation.append(f"\n   💡 **Key Finding:** {group2} shows {direction} diffusing population ({abs(diffusion_shift):.1f}% shift)")
            
            if abs(binding_shift) > 10:
                direction = "increased" if binding_shift > 0 else "decreased"
                interpretation.append(f"   💡 **Key Finding:** {group2} shows {direction} binding population ({abs(binding_shift):.1f}% shift)")
        
        # Kinetics comparison
        if 'kinetics_comparison' in comparison_results:
            kin_comp = comparison_results['kinetics_comparison']
            
            interpretation.append(f"\n⚡ **Kinetic Rate Comparison:**")
            
            if 'k_fast_fold_change' in kin_comp:
                interpretation.append(f"   Fast component (weighted by abundance):")
                interpretation.append(f"      {group1}: k = {kin_comp['k_fast_group1']:.3f} s⁻¹")
                interpretation.append(f"      {group2}: k = {kin_comp['k_fast_group2']:.3f} s⁻¹")
                interpretation.append(f"      Fold change: {kin_comp['k_fast_fold_change']:.2f}x")
            
            if 'k_slow_fold_change' in kin_comp:
                interpretation.append(f"   Slow component (weighted by abundance):")
                interpretation.append(f"      {group1}: k = {kin_comp['k_slow_group1']:.3f} s⁻¹")
                interpretation.append(f"      {group2}: k = {kin_comp['k_slow_group2']:.3f} s⁻¹")
                interpretation.append(f"      Fold change: {kin_comp['k_slow_fold_change']:.2f}x")
        
        # Biological interpretation
        interpretation.append(f"\n🧬 **Biological Interpretation:**")
        
        if 'population_comparison' in comparison_results:
            pop_comp = comparison_results['population_comparison']
            
            # Scenario 1: Loss of binding
            if binding_shift < -15:  # >15% decrease in binding population
                interpretation.append(f"   → {group2} appears to have LOST BINDING CAPABILITY")
                interpretation.append(f"      Population shifted from binding to diffusion/intermediate states")
                interpretation.append(f"      This suggests mutation disrupts chromatin association")
            
            # Scenario 2: Gain of binding
            elif binding_shift > 15:  # >15% increase in binding population
                interpretation.append(f"   → {group2} shows ENHANCED BINDING")
                interpretation.append(f"      Population shifted toward slow-recovering binding states")
                interpretation.append(f"      This suggests mutation strengthens chromatin association")
            
            # Scenario 3: Shift to pure diffusion
            elif diffusion_shift > 20:  # >20% increase in diffusion
                interpretation.append(f"   → {group2} shows predominantly DIFFUSIVE behavior")
                interpretation.append(f"      Minimal binding interactions detected")
                interpretation.append(f"      This suggests loss of specific interactions with chromatin")
            
            # Scenario 4: No major population shift but different kinetics
            elif 'kinetics_comparison' in comparison_results:
                kin_comp = comparison_results['kinetics_comparison']
                if 'k_fast_fold_change' in kin_comp:
                    if kin_comp['k_fast_fold_change'] > 1.5:
                        interpretation.append(f"   → {group2} shows FASTER kinetics within same population")
                        interpretation.append(f"      Binding/unbinding rates are accelerated")
                    elif kin_comp['k_fast_fold_change'] < 0.67:
                        interpretation.append(f"   → {group2} shows SLOWER kinetics within same population")
                        interpretation.append(f"      Binding/unbinding rates are decelerated")
        
        return "\n".join(interpretation)


def compute_average_recovery_profile(group_data: Dict[str, Dict], 
                                     time_points: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute average recovery profile for a group by averaging individual curves.
    
    This is the most direct way to compare groups - by looking at the actual
    averaged recovery curves rather than individual fitted parameters.
    
    Parameters
    ----------
    group_data : dict
        Dictionary mapping file paths to their data dictionaries
        Each data dict should have 'time' and 'intensity' arrays
    time_points : np.ndarray, optional
        Common time grid for interpolation. If None, uses union of all time points
        
    Returns
    -------
    tuple
        (time_grid, mean_intensity, sem_intensity)
    """
    if not group_data:
        return np.array([]), np.array([]), np.array([])
    
    # Extract all time series
    time_series = []
    intensity_series = []
    
    for file_path, data in group_data.items():
        if 'time' in data and 'intensity' in data:
            time_series.append(data['time'])
            intensity_series.append(data['intensity'])
    
    if not time_series:
        return np.array([]), np.array([]), np.array([])
    
    # Create common time grid if not provided
    if time_points is None:
        # Use the union of all unique time points
        all_times = np.concatenate(time_series)
        time_points = np.unique(all_times)
        time_points.sort()
    
    # Interpolate all curves to common time grid
    interpolated_intensities = []
    
    for t, i in zip(time_series, intensity_series):
        # Linear interpolation
        i_interp = np.interp(time_points, t, i)
        interpolated_intensities.append(i_interp)
    
    # Compute mean and SEM
    intensity_array = np.array(interpolated_intensities)
    mean_intensity = np.mean(intensity_array, axis=0)
    sem_intensity = stats.sem(intensity_array, axis=0)
    
    return time_points, mean_intensity, sem_intensity


def compare_recovery_profiles(group1_data: Dict, group2_data: Dict,
                              group1_name: str = "Group 1", 
                              group2_name: str = "Group 2") -> Dict:
    """
    Compare averaged recovery profiles between two groups.
    
    This provides a direct visual comparison of the recovery kinetics
    without relying on fitted parameters.
    
    Parameters
    ----------
    group1_data : dict
        Data for group 1 (file_path -> data dict)
    group2_data : dict
        Data for group 2 (file_path -> data dict)
    group1_name : str
        Name of group 1
    group2_name : str
        Name of group 2
        
    Returns
    -------
    dict
        Comparison results including averaged profiles and statistics
    """
    # Compute average profiles
    t1, i1_mean, i1_sem = compute_average_recovery_profile(group1_data)
    t2, i2_mean, i2_sem = compute_average_recovery_profile(group2_data)
    
    # Create common time grid
    t_common = np.union1d(t1, t2)
    
    # Interpolate both to common grid
    i1_common = np.interp(t_common, t1, i1_mean) if len(t1) > 0 else np.array([])
    i2_common = np.interp(t_common, t2, i2_mean) if len(t2) > 0 else np.array([])
    
    # Compute differences
    if len(i1_common) > 0 and len(i2_common) > 0:
        intensity_diff = i2_common - i1_common
        max_diff = np.max(np.abs(intensity_diff))
        max_diff_time = t_common[np.argmax(np.abs(intensity_diff))]
    else:
        intensity_diff = np.array([])
        max_diff = np.nan
        max_diff_time = np.nan
    
    results = {
        'group1_name': group1_name,
        'group2_name': group2_name,
        'group1_profile': {
            'time': t1,
            'intensity_mean': i1_mean,
            'intensity_sem': i1_sem,
            'n_cells': len(group1_data)
        },
        'group2_profile': {
            'time': t2,
            'intensity_mean': i2_mean,
            'intensity_sem': i2_sem,
            'n_cells': len(group2_data)
        },
        'comparison': {
            'time_common': t_common,
            'intensity_diff': intensity_diff,
            'max_difference': max_diff,
            'max_difference_time': max_diff_time
        }
    }
    
    return results
