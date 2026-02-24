"""
FRAP Visualization Module
Plotting functions for single-cell FRAP analysis
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import zscore
from typing import Optional, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Try to import dabest
try:
    import dabest
    DABEST_AVAILABLE = True
except ImportError:
    DABEST_AVAILABLE = False
    logger.warning("dabest not available, using custom estimation plots")


def plot_spaghetti(
    traces_df: pd.DataFrame,
    cell_features: pd.DataFrame,
    condition: str,
    param: str = 'signal_norm',
    bootstrap_ci: bool = True,
    n_bootstrap: int = 200,
    alpha_individual: float = 0.1,
    figsize: tuple[float, float] = (10, 6)
) -> plt.Figure:
    """
    Plot per-cell recovery curves with mean ± bootstrap CI
    
    Parameters
    ----------
    traces_df : pd.DataFrame
        ROI traces
    cell_features : pd.DataFrame
        Cell features (for filtering)
    condition : str
        Condition to plot
    param : str
        Parameter to plot ('signal_norm', 'signal_corr')
    bootstrap_ci : bool
        Show bootstrap confidence interval
    n_bootstrap : int
        Bootstrap iterations
    alpha_individual : float
        Alpha for individual cell lines
    figsize : tuple
        Figure size
        
    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Filter by condition
    cell_ids = cell_features[cell_features.get('condition') == condition]['cell_id'].values
    data = traces_df[traces_df['cell_id'].isin(cell_ids)].copy()
    
    if len(data) == 0:
        ax.text(0.5, 0.5, f"No data for {condition}", ha='center', va='center')
        return fig
    
    # Plot individual cells
    for cell_id in cell_ids:
        cell_data = data[data['cell_id'] == cell_id]
        ax.plot(cell_data['t'], cell_data[param], 
               color='gray', alpha=alpha_individual, linewidth=0.5)
    
    # Compute mean
    time_points = np.sort(data['t'].unique())
    means = []
    
    for t in time_points:
        values = data[data['t'] == t][param].values
        if len(values) > 0:
            means.append(np.nanmean(values))
        else:
            means.append(np.nan)
    
    means = np.array(means)
    
    # Plot mean
    ax.plot(time_points, means, 'b-', linewidth=2, label='Mean')
    
    # Bootstrap CI
    if bootstrap_ci and len(cell_ids) > 3:
        lower = np.zeros_like(means)
        upper = np.zeros_like(means)
        
        rng = np.random.RandomState(0)
        
        for i, t in enumerate(time_points):
            values = data[data['t'] == t][param].dropna().values
            if len(values) > 1:
                boot_means = []
                for _ in range(n_bootstrap):
                    boot_sample = rng.choice(values, size=len(values), replace=True)
                    boot_means.append(np.mean(boot_sample))
                lower[i] = np.percentile(boot_means, 2.5)
                upper[i] = np.percentile(boot_means, 97.5)
            else:
                lower[i] = upper[i] = means[i]
        
        ax.fill_between(time_points, lower, upper, alpha=0.3, color='blue', label='95% CI')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(param.replace('_', ' ').title())
    ax.set_title(f'Recovery Curves - {condition} (n={len(cell_ids)} cells)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_heatmap(
    traces_df: pd.DataFrame,
    condition: str,
    param: str = 'signal_norm',
    cluster_rows: bool = True,
    figsize: tuple[float, float] = (12, 8)
) -> plt.Figure:
    """
    Plot time × cell heatmap with optional hierarchical clustering
    
    Parameters
    ----------
    traces_df : pd.DataFrame
        ROI traces
    condition : str
        Condition to plot
    param : str
        Parameter to plot
    cluster_rows : bool
        Apply hierarchical clustering to rows (cells)
    figsize : tuple
        Figure size
        
    Returns
    -------
    plt.Figure
    """
    # Pivot to matrix
    data = traces_df.copy()
    
    # Create matrix: rows = cells, columns = time
    matrix = data.pivot_table(
        index='cell_id',
        columns='t',
        values=param,
        aggfunc='mean'
    )
    
    if matrix.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data", ha='center', va='center')
        return fig
    
    # Z-score normalize per cell
    matrix_z = matrix.apply(zscore, axis=1)
    matrix_z = matrix_z.fillna(0)
    
    # Cluster rows
    if cluster_rows and matrix_z.shape[0] > 1:
        row_linkage = linkage(matrix_z.values, method='ward')
        row_order = dendrogram(row_linkage, no_plot=True)['leaves']
        matrix_z = matrix_z.iloc[row_order]
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(matrix_z.values, aspect='auto', cmap='RdBu_r', 
                   vmin=-2, vmax=2, interpolation='nearest')
    
    # Labels
    ax.set_xlabel('Time Point')
    ax.set_ylabel('Cell ID')
    ax.set_title(f'Recovery Heatmap - {condition} (Z-scored per cell)')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Z-score')
    
    plt.tight_layout()
    return fig


def plot_estimation(
    df: pd.DataFrame,
    param: str,
    group_col: str = "condition",
    figsize: tuple[float, float] = (10, 6)
) -> plt.Figure:
    """
    Gardner-Altman estimation plot (mean difference with CI)
    
    Parameters
    ----------
    df : pd.DataFrame
        Cell features
    param : str
        Parameter to plot
    group_col : str
        Grouping column
    figsize : tuple
        Figure size
        
    Returns
    -------
    plt.Figure
    """
    if DABEST_AVAILABLE:
        # Use dabest library
        try:
            groups = sorted(df[group_col].unique())
            
            if len(groups) < 2:
                fig, ax = plt.subplots(figsize=figsize)
                ax.text(0.5, 0.5, "Need at least 2 groups", ha='center', va='center')
                return fig
            
            # Create dabest object
            dabest_obj = dabest.load(
                df, idx=groups, x=group_col, y=param
            )
            
            # Plot
            fig = dabest_obj.mean_diff.plot(
                fig_size=figsize,
                show_pairs=True
            )
            
            return fig.get_figure() if hasattr(fig, 'get_figure') else fig
            
        except Exception as e:
            logger.warning(f"dabest plot failed: {e}, using custom plot")
    
    # Custom implementation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, 
                                    gridspec_kw={'width_ratios': [2, 1]})
    
    groups = sorted(df[group_col].unique())
    
    # Left panel: Swarm plot
    positions = []
    for i, group in enumerate(groups):
        group_data = df[df[group_col] == group][param].dropna().values
        
        # Add jitter for visualization
        x = np.random.normal(i, 0.04, size=len(group_data))
        ax1.scatter(x, group_data, alpha=0.5, s=30)
        
        # Mean with CI
        mean = np.mean(group_data)
        ci_lower = np.percentile(group_data, 2.5)
        ci_upper = np.percentile(group_data, 97.5)
        
        ax1.plot([i-0.2, i+0.2], [mean, mean], 'r-', linewidth=2)
        ax1.plot([i, i], [ci_lower, ci_upper], 'r-', linewidth=1)
        
        positions.append(mean)
    
    ax1.set_xticks(range(len(groups)))
    ax1.set_xticklabels(groups, rotation=45, ha='right')
    ax1.set_ylabel(param.replace('_', ' ').title())
    ax1.set_title('Raw Data')
    ax1.grid(alpha=0.3, axis='y')
    
    # Right panel: Mean difference
    if len(groups) == 2:
        data1 = df[df[group_col] == groups[0]][param].dropna().values
        data2 = df[df[group_col] == groups[1]][param].dropna().values
        
        diff = np.mean(data2) - np.mean(data1)
        
        # Bootstrap CI for difference
        n_boot = 1000
        boot_diffs = []
        rng = np.random.RandomState(0)
        
        for _ in range(n_boot):
            boot1 = rng.choice(data1, size=len(data1), replace=True)
            boot2 = rng.choice(data2, size=len(data2), replace=True)
            boot_diffs.append(np.mean(boot2) - np.mean(boot1))
        
        ci_lower = np.percentile(boot_diffs, 2.5)
        ci_upper = np.percentile(boot_diffs, 97.5)
        
        ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax2.plot([0], [diff], 'ro', markersize=10)
        ax2.plot([0, 0], [ci_lower, ci_upper], 'r-', linewidth=2)
        ax2.set_xlim(-0.5, 0.5)
        ax2.set_xticks([])
        ax2.set_ylabel('Mean Difference')
        ax2.set_title(f'Δ = {diff:.3f}\n[{ci_lower:.3f}, {ci_upper:.3f}]')
        ax2.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def plot_pairplot(
    cell_features: pd.DataFrame,
    params: Optional[list[str]] = None,
    color_by: str = 'cluster',
    figsize: tuple[float, float] = (12, 12)
) -> plt.Figure:
    """
    Pair plot colored by cluster
    
    Parameters
    ----------
    cell_features : pd.DataFrame
        Cell features
    params : list[str], optional
        Parameters to include
    color_by : str
        Column to color by
    figsize : tuple
        Figure size
        
    Returns
    -------
    plt.Figure
    """
    if params is None:
        params = ['mobile_frac', 'k', 't_half', 'pre_bleach']
    
    # Filter valid data
    df = cell_features[params + [color_by]].dropna()
    
    if len(df) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No valid data", ha='center', va='center')
        return fig
    
    # Use seaborn pairplot
    g = sns.pairplot(
        df,
        vars=params,
        hue=color_by,
        diag_kind='kde',
        plot_kws={'alpha': 0.6, 's': 30},
        height=3
    )
    
    g.fig.suptitle('Feature Pair Plot', y=1.01)
    
    return g.fig


def plot_qc_dashboard(
    roi_traces: pd.DataFrame,
    cell_features: pd.DataFrame,
    figsize: tuple[float, float] = (15, 10)
) -> plt.Figure:
    """
    QC dashboard with multiple diagnostic plots
    
    Parameters
    ----------
    roi_traces : pd.DataFrame
        ROI traces
    cell_features : pd.DataFrame
        Cell features
    figsize : tuple
        Figure size
        
    Returns
    -------
    plt.Figure
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Drift distribution
    ax1 = fig.add_subplot(gs[0, 0])
    if 'drift_px' in cell_features.columns:
        ax1.hist(cell_features['drift_px'].dropna(), bins=30, edgecolor='black')
        ax1.axvline(cell_features['drift_px'].median(), color='r', linestyle='--', 
                   label=f"Median: {cell_features['drift_px'].median():.2f} px")
        ax1.set_xlabel('Drift (pixels)')
        ax1.set_ylabel('Count')
        ax1.set_title('ROI Drift Distribution')
        ax1.legend()
    
    # 2. R² distribution
    ax2 = fig.add_subplot(gs[0, 1])
    if 'r2' in cell_features.columns:
        ax2.hist(cell_features['r2'].dropna(), bins=30, edgecolor='black')
        ax2.axvline(0.5, color='r', linestyle='--', label='QC threshold')
        ax2.set_xlabel('R²')
        ax2.set_ylabel('Count')
        ax2.set_title('Fit Quality (R²)')
        ax2.legend()
    
    # 3. Motion artifacts per cell
    ax3 = fig.add_subplot(gs[0, 2])
    if 'qc_motion' in roi_traces.columns:
        motion_counts = roi_traces.groupby('cell_id')['qc_motion'].sum()
        ax3.hist(motion_counts, bins=20, edgecolor='black')
        ax3.set_xlabel('Frames flagged')
        ax3.set_ylabel('Count')
        ax3.set_title('Motion Artifacts per Cell')
    
    # 4. Mobile fraction vs k
    ax4 = fig.add_subplot(gs[1, 0])
    if 'mobile_frac' in cell_features.columns and 'k' in cell_features.columns:
        scatter = ax4.scatter(
            cell_features['mobile_frac'],
            cell_features['k'],
            c=cell_features.get('cluster', 0),
            cmap='tab10',
            alpha=0.6,
            s=50
        )
        ax4.set_xlabel('Mobile Fraction')
        ax4.set_ylabel('Recovery Rate k (s⁻¹)')
        ax4.set_title('Mobile Fraction vs Recovery Rate')
        if 'cluster' in cell_features.columns:
            plt.colorbar(scatter, ax=ax4, label='Cluster')
    
    # 5. Outlier distribution
    ax5 = fig.add_subplot(gs[1, 1])
    if 'outlier' in cell_features.columns:
        outlier_counts = cell_features['outlier'].value_counts()
        ax5.bar(['Normal', 'Outlier'], 
               [outlier_counts.get(False, 0), outlier_counts.get(True, 0)],
               color=['green', 'red'], alpha=0.6, edgecolor='black')
        ax5.set_ylabel('Count')
        ax5.set_title('Outlier Detection')
    
    # 6. Cluster sizes
    ax6 = fig.add_subplot(gs[1, 2])
    if 'cluster' in cell_features.columns:
        cluster_counts = cell_features[cell_features['cluster'] != -1]['cluster'].value_counts().sort_index()
        ax6.bar(cluster_counts.index, cluster_counts.values, 
               edgecolor='black', alpha=0.6)
        ax6.set_xlabel('Cluster')
        ax6.set_ylabel('Count')
        ax6.set_title('Cluster Sizes')
    
    # 7. Residuals example
    ax7 = fig.add_subplot(gs[2, :])
    if len(cell_features) > 0:
        # Show first few cells' residuals
        sample_cells = cell_features.head(5)['cell_id'].values
        
        for i, cell_id in enumerate(sample_cells):
            cell_traces = roi_traces[roi_traces['cell_id'] == cell_id]
            if len(cell_traces) > 0:
                ax7.plot(cell_traces['t'], 
                        cell_traces.get('signal_corr', cell_traces.get('signal_raw', [])),
                        alpha=0.5, label=f'Cell {cell_id}')
        
        ax7.set_xlabel('Time (s)')
        ax7.set_ylabel('Signal')
        ax7.set_title('Sample Recovery Curves')
        ax7.legend(ncol=5, loc='best')
        ax7.grid(alpha=0.3)
    
    fig.suptitle('Quality Control Dashboard', fontsize=16, y=0.995)
    
    return fig


def save_all_figures(
    output_dir: Union[str, Path],
    traces_df: pd.DataFrame,
    cell_features: pd.DataFrame,
    format: str = 'png',
    dpi: int = 300
) -> list[Path]:
    """
    Generate and save all standard figures
    
    Parameters
    ----------
    output_dir : str or Path
        Output directory
    traces_df : pd.DataFrame
        ROI traces
    cell_features : pd.DataFrame
        Cell features
    format : str
        Image format
    dpi : int
        Resolution
        
    Returns
    -------
    list[Path]
        Saved file paths
    """
    from pathlib import Path
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    
    # Spaghetti plots per condition
    if 'condition' in cell_features.columns:
        for condition in cell_features['condition'].unique():
            fig = plot_spaghetti(traces_df, cell_features, condition)
            path = output_dir / f'spaghetti_{condition}.{format}'
            fig.savefig(path, dpi=dpi, bbox_inches='tight')
            saved_files.append(path)
            plt.close(fig)
    
    # QC dashboard
    fig = plot_qc_dashboard(traces_df, cell_features)
    path = output_dir / f'qc_dashboard.{format}'
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    saved_files.append(path)
    plt.close(fig)
    
    logger.info(f"Saved {len(saved_files)} figures to {output_dir}")
    
    return saved_files
