import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def plot_publication_violin(df, x_col, y_col, color_col=None, title=None, y_label=None, x_label=None):
    """
    Generates a publication-quality violin plot with box overlay and data points.
    """
    if color_col is None:
        color_col = x_col
        
    fig = px.violin(
        df, 
        x=x_col, 
        y=y_col, 
        color=color_col,
        box=True,           # Draw box plot inside violin
        points='all',       # Show all data points
        hover_data=df.columns,
        title=title
    )
    
    # Styling for publication
    fig.update_layout(
        template="simple_white", 
        yaxis_title=y_label if y_label else y_col,
        xaxis_title=x_label if x_label else "",
        font=dict(family="Arial", size=14, color="black"),
        legend_title_text="Group",
        showlegend=False, # Often cleaner without redundant legend if x-axis is labeled
        height=500
    )
    
    # Improve visual aesthetics
    fig.update_traces(
        meanline_visible=True,
        marker=dict(size=4, opacity=0.5, line=dict(width=0.5, color='DarkSlateGrey'))
    )
    
    return fig

def plot_kinetic_comparison(comparison_df, metric='D', title="Kinetic Parameter Comparison"):
    """
    Wrapper to plot specific kinetic metrics from the comparison dataframe.
    """
    labels = {
        'D': 'Diffusion Coefficient (µm²/s)',
        'k': 'Rate Constant (1/s)',
        'mobile_fraction': 'Mobile Fraction',
        'binding_fraction': 'Binding Population Fraction'
    }
    
    y_lbl = labels.get(metric, metric)
    
    return plot_publication_violin(
        comparison_df, 
        x_col='Group', 
        y_col=metric, 
        color_col='Group',
        title=title,
        y_label=y_lbl
    )