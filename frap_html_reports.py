import os
import io
import base64
import pandas as pd
from datetime import datetime
from typing import List, Optional, Dict, Any

try:
    import plotly.express as px
except ImportError:
    px = None

try:
    from group_stats import calculate_group_stats
except ImportError:
    calculate_group_stats = None


def _encode_plot(fig) -> str:
    """Return an HTML div for a plotly figure or a placeholder string."""
    if fig is None:
        return "<p><i>Plot unavailable.</i></p>"
    try:
        return fig.to_html(full_html=False, include_plotlyjs='cdn')
    except Exception:
        return "<p><i>Plot rendering failed.</i></p>"


def _create_stats_table_html(stats_df: pd.DataFrame) -> str:
    """Generate HTML table from the statistics DataFrame."""
    if stats_df.empty:
        return "<p><i>No statistical results to display.</i></p>"

    # Define headers
    headers = [
        "Metric", "Test", "p-value", "q-value", "Permutation p-value",
        "Cohen's d", "Cliff's Delta", "TOST p-value", "TOST Outcome"
    ]

    html = ["<table><thead><tr>"]
    for header in headers:
        html.append(f"<th>{header}</th>")
    html.append("</tr></thead><tbody>")

    for _, row in stats_df.iterrows():
        p_val = row.get('p_value', 'N/A')
        q_val = row.get('q_value', 'N/A')

        # Color rows based on significance
        sig_color = 'white'
        if pd.notna(q_val) and q_val < 0.05:
            sig_color = '#d4edda'  # Green for significant
        elif pd.notna(p_val) and p_val < 0.05:
             sig_color = '#f8d7da' # Red for ns after correction

        html.append(f"<tr style='background:{sig_color}'>")
        html.append(f"<td>{row.get('metric', 'N/A')}</td>")
        html.append(f"<td>{row.get('test', 'N/A')}</td>")
        html.append(f"<td>{p_val:.4g}</td>" if pd.notna(p_val) else "<td>N/A</td>")
        html.append(f"<td>{q_val:.4g}</td>" if pd.notna(q_val) else "<td>N/A</td>")
        p_perm_val = row.get('p_perm')
        html.append(f"<td>{p_perm_val:.4g}</td>" if pd.notna(p_perm_val) else "<td>N/A</td>")
        html.append(f"<td>{row.get('cohen_d', 'N/A'):.3f}</td>" if pd.notna(row.get('cohen_d')) else "<td>N/A</td>")
        html.append(f"<td>{row.get('cliffs_delta', 'N/A'):.3f}</td>" if pd.notna(row.get('cliffs_delta')) else "<td>N/A</td>")
        html.append(f"<td>{row.get('p_tost', 'N/A'):.4g}</td>" if pd.notna(row.get('p_tost')) else "<td>N/A</td>")
        html.append(f"<td>{row.get('tost_outcome', 'N/A')}</td>")
        html.append("</tr>")

    html.append("</tbody></table>")

    if 'mixed_effects_summary' in stats_df.columns:
        html.append("<h3>Mixed-Effects Model Summaries</h3>")
        for metric, summary in stats_df.groupby('metric')['mixed_effects_summary'].first().items():
            if pd.notna(summary):
                html.append(f"<h4>{metric}</h4>")
                html.append(f"<pre>{summary}</pre>")

    return ''.join(html)


def generate_html_report(
    data_manager,
    groups_to_compare: List[str],
    output_filename: Optional[str] = None,
    settings: Optional[Dict[str, Any]] = None,
    use_mixed_effects: bool = False
):
    """Generate a comprehensive HTML report with robust statistics."""
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
            # Add a placeholder for experiment_id if it's missing
            if 'experiment_id' not in tmp.columns:
                tmp['experiment_id'] = group_name
            all_group_data.append(tmp)

    if not all_group_data:
        return None

    combined_df = pd.concat(all_group_data, ignore_index=True)

    # Rename columns for consistency
    combined_df.rename(columns={
        'diffusion_coefficient': 'D_um2_s',
        'molecular_weight_estimate': 'app_mw_kDa',
        'rate_constant': 'koff'
    }, inplace=True)


    html_parts: List[str] = [
        "<h1>FRAP Analysis Report</h1>",
        f"<p><b>Report Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
        f"<p><b>Groups:</b> {', '.join(groups_to_compare)}</p>",
    ]

    if settings:
        settings_df = pd.DataFrame(list(settings.items()), columns=['Parameter', 'Value'])
        html_parts.append("<h2>Analysis Settings</h2>")
        html_parts.append(settings_df.to_html(index=False))

    summary_metrics = [m for m in ['mobile_fraction', 'koff', 'D_um2_s', 'app_mw_kDa'] if m in combined_df.columns]
    if summary_metrics:
        html_parts.append("<h2>Summary Statistics</h2>")
        summary_table = combined_df.groupby('group')[summary_metrics].agg(['mean', 'std']).round(3)
        summary_table.columns = [' '.join(col).strip() for col in summary_table.columns.values]
        html_parts.append(summary_table.to_html())
    else:
        html_parts.append("<p><i>No summary metrics available.</i></p>")

    # Statistical Comparison
    if len(groups_to_compare) > 1 and calculate_group_stats is not None:
        html_parts.append("<h2>Statistical Comparison</h2>")

        tost_thresholds = {
            'D_um2_s': (-0.2, 0.2),
            'mobile_fraction': (-0.1, 0.1)
        }

        stats_df = calculate_group_stats(
            data=combined_df,
            metrics=summary_metrics,
            group_order=groups_to_compare,
            tost_thresholds=tost_thresholds,
            use_mixed_effects=use_mixed_effects
        )
        html_parts.append(_create_stats_table_html(stats_df))
    elif calculate_group_stats is None:
        html_parts.append("<p><i>Statistical tests unavailable (group_stats module not found).</i></p>")


    if px is not None and len(groups_to_compare) >= 1 and summary_metrics:
        try:
            long_df = combined_df.melt(id_vars='group', value_vars=summary_metrics, var_name='Metric', value_name='Value')
            fig_box = px.box(long_df, x='Metric', y='Value', color='group', notched=True,
                             title='Comparison of Key Metrics Across Groups')
            html_parts.append("<h2>Comparison Plots</h2>")
            html_parts.append(_encode_plot(fig_box))
        except Exception:
            html_parts.append("<p><i>Plot generation failed.</i></p>")

    html_parts.append("<h2>Detailed Results</h2>")
    html_parts.append(combined_df.to_html(index=False))

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
