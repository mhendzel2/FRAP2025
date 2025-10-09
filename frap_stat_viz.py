"""
FRAP Statistical Visualizations
Volcano plots, forest plots, and effect size visualizations
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)


def plot_volcano(
    results_df: pd.DataFrame,
    alpha: float = 0.05,
    fc_threshold: float = 0.5,
    use_fdr: bool = True,
    title: str = "Volcano Plot"
) -> go.Figure:
    """
    Create volcano plot for multiple comparison results
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results from multi_parameter_analysis with columns:
        - param: parameter name
        - log2_fold_change: log2 fold change
        - neg_log10_p: -log10(p-value)
        - neg_log10_q: -log10(q-value) for FDR
        - significant: boolean significance flag
    alpha : float
        Significance threshold
    fc_threshold : float
        Fold-change threshold (log2 scale)
    use_fdr : bool
        Use FDR-adjusted q-values instead of p-values
    title : str
        Plot title
        
    Returns
    -------
    go.Figure
        Plotly figure
    """
    if results_df.empty:
        logger.warning("Empty results dataframe")
        return go.Figure()
    
    df = results_df.copy()
    
    # Select p-value column
    if use_fdr and 'neg_log10_q' in df.columns:
        neg_log10_pval = df['neg_log10_q']
        pval_label = "q-value (FDR)"
        threshold = -np.log10(alpha)
    else:
        neg_log10_pval = df['neg_log10_p']
        pval_label = "p-value"
        threshold = -np.log10(alpha)
    
    # Classify points
    df['category'] = 'Not Significant'
    df.loc[(np.abs(df['log2_fold_change']) >= fc_threshold) & 
           (neg_log10_pval >= threshold), 'category'] = 'Significant'
    df.loc[(np.abs(df['log2_fold_change']) < fc_threshold) & 
           (neg_log10_pval >= threshold), 'category'] = 'Significant (small effect)'
    df.loc[(np.abs(df['log2_fold_change']) >= fc_threshold) & 
           (neg_log10_pval < threshold), 'category'] = 'Large effect (not sig.)'
    
    # Color map
    color_map = {
        'Significant': '#e74c3c',
        'Significant (small effect)': '#f39c12',
        'Large effect (not sig.)': '#3498db',
        'Not Significant': '#95a5a6'
    }
    
    fig = go.Figure()
    
    for category in color_map.keys():
        mask = df['category'] == category
        if mask.sum() == 0:
            continue
        
        fig.add_trace(go.Scatter(
            x=df.loc[mask, 'log2_fold_change'],
            y=neg_log10_pval[mask],
            mode='markers',
            name=category,
            marker=dict(
                size=8,
                color=color_map[category],
                line=dict(width=1, color='white')
            ),
            text=df.loc[mask, 'param'] + '<br>' + df.loc[mask, 'comparison'],
            hovertemplate='<b>%{text}</b><br>' +
                         f'log2(FC): %{{x:.2f}}<br>' +
                         f'-log10({pval_label}): %{{y:.2f}}<br>' +
                         '<extra></extra>'
        ))
    
    # Add threshold lines
    fig.add_hline(y=threshold, line_dash="dash", line_color="gray", 
                  annotation_text=f"α = {alpha}")
    fig.add_vline(x=fc_threshold, line_dash="dash", line_color="gray")
    fig.add_vline(x=-fc_threshold, line_dash="dash", line_color="gray")
    
    fig.update_layout(
        title=title,
        xaxis_title="log₂(Fold Change)",
        yaxis_title=f"-log₁₀({pval_label})",
        template="plotly_white",
        hovermode='closest',
        width=800,
        height=600,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    
    return fig


def plot_forest(
    results_df: pd.DataFrame,
    param: Optional[str] = None,
    sort_by: str = "hedges_g",
    title: str = "Forest Plot - Effect Sizes"
) -> go.Figure:
    """
    Create forest plot for effect sizes with confidence intervals
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results from multi_parameter_analysis
    param : str, optional
        Filter to specific parameter
    sort_by : str
        Column to sort by ('hedges_g', 'p', 'q')
    title : str
        Plot title
        
    Returns
    -------
    go.Figure
        Plotly figure
    """
    if results_df.empty:
        logger.warning("Empty results dataframe")
        return go.Figure()
    
    df = results_df.copy()
    
    # Filter to parameter if specified
    if param is not None:
        df = df[df['param'] == param]
    
    if df.empty:
        logger.warning(f"No results for parameter {param}")
        return go.Figure()
    
    # Sort
    df = df.sort_values(sort_by, ascending=False).reset_index(drop=True)
    
    # Create labels
    df['label'] = df['param'] + '<br>' + df['comparison']
    
    # Compute CI from beta and se
    df['ci_lower_es'] = df['hedges_g'] - 1.96 * df['se'] / (df['beta'] / df['hedges_g'] + 1e-10)
    df['ci_upper_es'] = df['hedges_g'] + 1.96 * df['se'] / (df['beta'] / df['hedges_g'] + 1e-10)
    
    # Color by significance
    df['color'] = df['significant'].map({True: '#e74c3c', False: '#95a5a6'})
    
    fig = go.Figure()
    
    # Add error bars
    for idx, row in df.iterrows():
        fig.add_trace(go.Scatter(
            x=[row['ci_lower_es'], row['hedges_g'], row['ci_upper_es']],
            y=[idx, idx, idx],
            mode='lines+markers',
            marker=dict(
                size=[0, 10, 0],
                color=row['color'],
                symbol=['line-ns', 'diamond', 'line-ns']
            ),
            line=dict(color=row['color'], width=2),
            name=row['label'],
            showlegend=False,
            hovertemplate=f"<b>{row['label']}</b><br>" +
                         f"Hedges' g: {row['hedges_g']:.3f}<br>" +
                         f"95% CI: [{row['ci_lower_es']:.3f}, {row['ci_upper_es']:.3f}]<br>" +
                         f"p: {row['p']:.4f}<br>" +
                         f"q: {row['q']:.4f}<br>" +
                         "<extra></extra>"
        ))
    
    # Add zero line
    fig.add_vline(x=0, line_dash="solid", line_color="black", line_width=1)
    
    # Add reference lines for small/medium/large effects
    fig.add_vline(x=0.2, line_dash="dot", line_color="gray", opacity=0.5)
    fig.add_vline(x=0.5, line_dash="dot", line_color="gray", opacity=0.5)
    fig.add_vline(x=0.8, line_dash="dot", line_color="gray", opacity=0.5)
    fig.add_vline(x=-0.2, line_dash="dot", line_color="gray", opacity=0.5)
    fig.add_vline(x=-0.5, line_dash="dot", line_color="gray", opacity=0.5)
    fig.add_vline(x=-0.8, line_dash="dot", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title=title,
        xaxis_title="Hedges' g (Effect Size)",
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(df))),
            ticktext=df['label'].tolist(),
            autorange="reversed"
        ),
        template="plotly_white",
        width=1000,
        height=max(400, len(df) * 40),
        hovermode='closest'
    )
    
    return fig


def plot_effect_size_heatmap(
    results_df: pd.DataFrame,
    title: str = "Effect Size Heatmap"
) -> go.Figure:
    """
    Create heatmap of effect sizes across parameters and comparisons
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results from multi_parameter_analysis
    title : str
        Plot title
        
    Returns
    -------
    go.Figure
        Plotly figure
    """
    if results_df.empty:
        logger.warning("Empty results dataframe")
        return go.Figure()
    
    # Pivot table
    pivot = results_df.pivot_table(
        values='hedges_g',
        index='param',
        columns='comparison',
        aggfunc='first'
    )
    
    # Create significance mask
    pivot_sig = results_df.pivot_table(
        values='significant',
        index='param',
        columns='comparison',
        aggfunc='first'
    ).fillna(False)
    
    # Create hover text
    hover_text = []
    for i, param in enumerate(pivot.index):
        row_text = []
        for j, comp in enumerate(pivot.columns):
            effect = pivot.iloc[i, j]
            is_sig = pivot_sig.iloc[i, j]
            
            # Find corresponding row in results_df
            mask = (results_df['param'] == param) & (results_df['comparison'] == comp)
            if mask.sum() > 0:
                row_data = results_df[mask].iloc[0]
                text = (f"<b>{param}</b><br>"
                       f"{comp}<br>"
                       f"Hedges' g: {effect:.3f}<br>"
                       f"p: {row_data['p']:.4f}<br>"
                       f"q: {row_data['q']:.4f}<br>"
                       f"Significant: {'Yes' if is_sig else 'No'}")
            else:
                text = f"{param}<br>{comp}<br>N/A"
            row_text.append(text)
        hover_text.append(row_text)
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        text=hover_text,
        hovertemplate='%{text}<extra></extra>',
        colorscale='RdBu_r',
        zmid=0,
        colorbar=dict(title="Hedges' g")
    ))
    
    # Add asterisks for significant results
    for i, param in enumerate(pivot.index):
        for j, comp in enumerate(pivot.columns):
            if pivot_sig.iloc[i, j]:
                fig.add_annotation(
                    x=j, y=i,
                    text="*",
                    showarrow=False,
                    font=dict(size=20, color="black")
                )
    
    fig.update_layout(
        title=title,
        xaxis_title="Comparison",
        yaxis_title="Parameter",
        template="plotly_white",
        width=max(600, len(pivot.columns) * 100),
        height=max(400, len(pivot.index) * 40)
    )
    
    return fig


def plot_pvalue_histogram(
    results_df: pd.DataFrame,
    use_fdr: bool = True,
    title: str = "P-value Distribution"
) -> go.Figure:
    """
    Create histogram of p-values to assess multiple testing
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results from multi_parameter_analysis
    use_fdr : bool
        Show both p and q values
    title : str
        Plot title
        
    Returns
    -------
    go.Figure
        Plotly figure
    """
    if results_df.empty:
        logger.warning("Empty results dataframe")
        return go.Figure()
    
    fig = go.Figure()
    
    # P-values
    fig.add_trace(go.Histogram(
        x=results_df['p'],
        nbinsx=20,
        name='p-values',
        marker_color='#3498db',
        opacity=0.7
    ))
    
    if use_fdr and 'q' in results_df.columns:
        fig.add_trace(go.Histogram(
            x=results_df['q'],
            nbinsx=20,
            name='q-values (FDR)',
            marker_color='#e74c3c',
            opacity=0.7
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Value",
        yaxis_title="Count",
        barmode='overlay',
        template="plotly_white",
        width=800,
        height=500
    )
    
    return fig


def plot_qq(
    results_df: pd.DataFrame,
    title: str = "Q-Q Plot of P-values"
) -> go.Figure:
    """
    Create Q-Q plot to assess p-value calibration
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results from multi_parameter_analysis
    title : str
        Plot title
        
    Returns
    -------
    go.Figure
        Plotly figure
    """
    if results_df.empty:
        logger.warning("Empty results dataframe")
        return go.Figure()
    
    # Sort p-values
    pvals = np.sort(results_df['p'].values)
    n = len(pvals)
    expected = np.arange(1, n + 1) / (n + 1)
    
    fig = go.Figure()
    
    # Observed vs expected
    fig.add_trace(go.Scatter(
        x=-np.log10(expected),
        y=-np.log10(pvals),
        mode='markers',
        name='Observed',
        marker=dict(size=8, color='#3498db')
    ))
    
    # Reference line
    max_val = max(-np.log10(expected).max(), -np.log10(pvals).max())
    fig.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode='lines',
        name='Expected',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="-log₁₀(Expected p-value)",
        yaxis_title="-log₁₀(Observed p-value)",
        template="plotly_white",
        width=600,
        height=600
    )
    
    return fig


def plot_comparison_summary(
    results_df: pd.DataFrame,
    group_by: str = "param",
    title: str = "Comparison Summary"
) -> go.Figure:
    """
    Create summary bar chart of significant tests
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results from multi_parameter_analysis
    group_by : str
        Column to group by ('param' or 'comparison')
    title : str
        Plot title
        
    Returns
    -------
    go.Figure
        Plotly figure
    """
    if results_df.empty:
        logger.warning("Empty results dataframe")
        return go.Figure()
    
    # Count significant tests
    summary = results_df.groupby(group_by).agg({
        'significant': ['sum', 'count'],
        'hedges_g': 'mean'
    }).reset_index()
    
    summary.columns = [group_by, 'n_sig', 'n_total', 'mean_effect']
    summary['pct_sig'] = 100 * summary['n_sig'] / summary['n_total']
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Significant Tests", "Mean Effect Size")
    )
    
    # Significant tests
    fig.add_trace(
        go.Bar(
            x=summary[group_by],
            y=summary['pct_sig'],
            name='% Significant',
            marker_color='#e74c3c',
            text=summary['n_sig'].astype(str) + '/' + summary['n_total'].astype(str),
            textposition='auto'
        ),
        row=1, col=1
    )
    
    # Mean effect size
    fig.add_trace(
        go.Bar(
            x=summary[group_by],
            y=summary['mean_effect'],
            name="Mean Hedges' g",
            marker_color='#3498db',
            text=summary['mean_effect'].round(3).astype(str),
            textposition='auto'
        ),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text=group_by.capitalize(), row=1, col=1)
    fig.update_xaxes(title_text=group_by.capitalize(), row=1, col=2)
    fig.update_yaxes(title_text="% Significant", row=1, col=1)
    fig.update_yaxes(title_text="Mean Effect Size", row=1, col=2)
    
    fig.update_layout(
        title_text=title,
        template="plotly_white",
        width=1200,
        height=500,
        showlegend=False
    )
    
    return fig
