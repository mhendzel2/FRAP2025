import os
import io
import base64
import pandas as pd
from datetime import datetime
from typing import List, Optional, Dict, Any

try:  # Plotly is already used in the Streamlit app; keep optional for headless usage
    import plotly.express as px
except Exception:  # pragma: no cover - if plotly missing we degrade gracefully
    px = None

try:
    from scipy import stats
except Exception:  # pragma: no cover - statistical tests optional
    stats = None

def _encode_plot(fig) -> str:
    """Return an HTML div for a plotly figure or a placeholder string."""
    if fig is None:
        return "<p><i>Plot unavailable.</i></p>"
    try:
        return fig.to_html(full_html=False, include_plotlyjs='cdn')
    except Exception:
        return "<p><i>Plot rendering failed.</i></p>"


def _stat_tests_table(combined_df: pd.DataFrame, group_order: List[str]) -> str:
    """Generate HTML for statistical tests across groups (t-test or ANOVA)."""
    if stats is None or len(group_order) < 2:
        return "<p><i>Statistical tests unavailable (scipy not installed or insufficient groups).</i></p>"

    metrics = [m for m in ['mobile_fraction', 'rate_constant', 'half_time'] if m in combined_df.columns]
    if not metrics:
        return "<p><i>No comparable metrics found for statistical testing.</i></p>"

    rows = []
    for metric in metrics:
        groups_data = [combined_df[combined_df['group'] == g][metric].dropna() for g in group_order]
        groups_data_valid = [g for g in groups_data if len(g) > 1]
        if len(groups_data_valid) < 2:
            rows.append((metric, 'N/A', 'n/a', 'Insufficient data'))
            continue
        if len(group_order) == 2:
            g1, g2 = groups_data_valid[0], groups_data_valid[1]
            try:
                t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=False)
                rows.append((metric, "Welch t-test", f"{p_val:.4g}", "Yes" if p_val < 0.05 else "No"))
            except Exception:
                rows.append((metric, 'Welch t-test', 'err', 'Error'))
        else:
            try:
                f_stat, p_val = stats.f_oneway(*groups_data_valid)
                rows.append((metric, "ANOVA", f"{p_val:.4g}", "Yes" if p_val < 0.05 else "No"))
            except Exception:
                rows.append((metric, 'ANOVA', 'err', 'Error'))

    html = ["<table><thead><tr><th>Metric</th><th>Test</th><th>p-value</th><th>Significant (p<0.05)</th></tr></thead><tbody>"]
    for metric, test, pval, sig in rows:
        sig_color = '#d4edda' if sig == 'Yes' else '#f8d7da' if sig == 'No' and pval not in ('n/a', 'err') else 'white'
        html.append(f"<tr style='background:{sig_color}'><td>{metric.replace('_',' ')}</td><td>{test}</td><td>{pval}</td><td>{sig}</td></tr>")
    html.append("</tbody></table>")
    return ''.join(html)


def generate_html_report(data_manager, groups_to_compare, output_filename: Optional[str] = None, settings: Optional[Dict[str, Any]] = None):
    """Generate a comprehensive HTML report.

    Enhancements:
    - Supports single file (wrapped as a temp group) or multiple groups.
    - Adds statistical comparison (t-test / ANOVA) when >1 group.
    - Embeds interactive Plotly boxplots (if plotly present) for key metrics.
    """
    if not groups_to_compare:
        return None

    # Collect and combine per-group feature data
    all_group_data: List[pd.DataFrame] = []
    for group_name in groups_to_compare:
        grp = data_manager.groups.get(group_name)
        if not grp:
            continue
        df = grp.get('features_df')
        if df is not None and not df.empty:
            tmp = df.copy()
            tmp['group'] = group_name
            all_group_data.append(tmp)

    if not all_group_data:
        return None

    combined_df = pd.concat(all_group_data, ignore_index=True)

    # Header
    html_parts: List[str] = [
        "<h1>FRAP Analysis Report</h1>",
        f"<p><b>Report Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
        f"<p><b>Groups:</b> {', '.join(groups_to_compare)}</p>",
    ]

    # Settings
    if settings:
        settings_df = pd.DataFrame(list(settings.items()), columns=['Parameter', 'Value'])
        html_parts.append("<h2>Analysis Settings</h2>")
        html_parts.append(settings_df.to_html(index=False))

    # Summary Statistics
    html_parts.append("<h2>Summary Statistics</h2>")
    summary_metrics = [m for m in ['mobile_fraction', 'rate_constant', 'half_time'] if m in combined_df.columns]
    if summary_metrics:
        summary_table = combined_df.groupby('group')[summary_metrics].agg(['mean', 'std']).round(3)
        if not summary_table.empty:
            summary_table.columns = [' '.join(col).strip() for col in summary_table.columns.values]
            html_parts.append(summary_table.to_html())
    else:
        html_parts.append("<p><i>No summary metrics available.</i></p>")

    # Statistical Comparison
    if len(groups_to_compare) > 1:
        html_parts.append("<h2>Statistical Comparison</h2>")
        html_parts.append(_stat_tests_table(combined_df, groups_to_compare))

    # Plots (only if plotly available and multiple groups)
    if px is not None and len(groups_to_compare) >= 1 and summary_metrics:
        try:
            long_df = combined_df.melt(id_vars='group', value_vars=summary_metrics, var_name='Metric', value_name='Value')
            fig_box = px.box(long_df, x='Metric', y='Value', color='group', notched=True,
                             title='Comparison of Key Metrics Across Groups')
            html_parts.append("<h2>Comparison Plots</h2>")
            html_parts.append(_encode_plot(fig_box))
        except Exception:
            html_parts.append("<p><i>Plot generation failed.</i></p>")

    # Detailed results
    html_parts.append("<h2>Detailed Results</h2>")
    html_parts.append(combined_df.to_html(index=False))

    # Assemble full HTML
    body_html = '\n'.join(html_parts)
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset='utf-8'/>
        <title>FRAP Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 2em; }}
            h1, h2, h3 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 95%; margin-bottom: 1.5em; }}
            th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: left; font-size: 13px; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #fafafa; }}
        </style>
    </head>
    <body>
        {body_html}
    </body>
    </html>
    """

    if output_filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f"FRAP_Report_{timestamp}.html"

    output_path = os.path.abspath(output_filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_template)
    return output_path
