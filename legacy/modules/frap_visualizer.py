import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class FRAPVisualizer:
    """
    Generates visualizations for FRAP analysis.
    """
    
    @staticmethod
    def plot_recovery_curves(
        time: np.ndarray,
        intensities: List[np.ndarray],
        fitted_curves: Optional[List[np.ndarray]] = None,
        title: str = "Recovery Curves",
        show_error: bool = True
    ) -> plt.Figure:
        """
        Plots average recovery curve with error bands.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Convert list to array for stats
        # Assuming all intensities have same length and time points
        # If not, we need interpolation or binning. 
        # For now assume standardized time axis.
        
        data_matrix = np.array(intensities)
        mean_curve = np.mean(data_matrix, axis=0)
        std_curve = np.std(data_matrix, axis=0)
        sem_curve = std_curve / np.sqrt(len(intensities))
        
        ax.plot(time, mean_curve, label='Mean Data', color='blue')
        
        if show_error:
            ax.fill_between(time, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2, color='blue', label='SD')
            
        if fitted_curves:
            fit_matrix = np.array(fitted_curves)
            mean_fit = np.mean(fit_matrix, axis=0)
            ax.plot(time, mean_fit, label='Mean Fit', color='red', linestyle='--')
            
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Normalized Intensity')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig

    @staticmethod
    def plot_residuals(
        time: np.ndarray,
        residuals: List[np.ndarray],
        title: str = "Residuals"
    ) -> plt.Figure:
        """
        Plots residuals.
        """
        fig, ax = plt.subplots(figsize=(10, 4))
        
        res_matrix = np.array(residuals)
        mean_res = np.mean(res_matrix, axis=0)
        
        ax.plot(time, mean_res, color='black')
        ax.axhline(0, color='red', linestyle='--')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Residuals')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        return fig

    @staticmethod
    def plot_parameter_distribution(
        features: pd.DataFrame,
        param_name: str,
        group_col: str = None,
        plot_type: str = 'violin'
    ) -> plt.Figure:
        """
        Plots distribution of a parameter.
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if plot_type == 'violin':
            sns.violinplot(data=features, x=group_col, y=param_name, ax=ax)
        elif plot_type == 'box':
            sns.boxplot(data=features, x=group_col, y=param_name, ax=ax)
        elif plot_type == 'cdf':
            sns.ecdfplot(data=features, x=param_name, hue=group_col, ax=ax)
            
        ax.set_title(f'Distribution of {param_name}')
        return fig

    @staticmethod
    def plot_subpopulations(
        features: pd.DataFrame,
        x_param: str,
        y_param: str,
        cluster_col: str = 'subpopulation'
    ) -> plt.Figure:
        """
        Scatter plot of subpopulations.
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if cluster_col in features.columns:
            sns.scatterplot(data=features, x=x_param, y=y_param, hue=cluster_col, palette='viridis', ax=ax)
        else:
            sns.scatterplot(data=features, x=x_param, y=y_param, ax=ax)
            
        ax.set_title(f'Subpopulations: {x_param} vs {y_param}')
        return fig
