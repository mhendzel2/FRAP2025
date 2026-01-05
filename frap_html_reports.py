import os
import io
import base64
import html as _html
import pandas as pd
from datetime import datetime
from typing import List, Optional, Dict, Any

try:  # Plotly is already used in the Streamlit app; keep optional for headless usage
    import plotly.express as px
except Exception:  # pragma: no cover - if plotly missing we degrade gracefully
    px = None

try:
    import plotly.graph_objects as go
except Exception:  # pragma: no cover
    go = None

try:
    from scipy import stats
except Exception:  # pragma: no cover - statistical tests optional
    stats = None

from frap_core import get_post_bleach_data

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


def _control_tests_table(combined_df: pd.DataFrame, group_order: List[str], control_groups: List[str]) -> str:
    """Generate HTML for pooled-control vs sample Welch t-tests."""
    if stats is None or len(group_order) < 2:
        return "<p><i>Control-based tests unavailable (scipy not installed or insufficient groups).</i></p>"

    control_groups = [g for g in control_groups if g in group_order]
    if not control_groups:
        return "<p><i>No valid control groups selected.</i></p>"

    pooled = combined_df[combined_df['group'].isin(control_groups)]
    metrics = [m for m in ['mobile_fraction', 'rate_constant', 'half_time'] if m in combined_df.columns]
    if not metrics:
        return "<p><i>No comparable metrics found for control-based testing.</i></p>"

    rows = []
    for metric in metrics:
        c_vals = pooled[metric].dropna()
        if len(c_vals) <= 1:
            continue
        for g in group_order:
            if g in control_groups:
                continue
            g_vals = combined_df[combined_df['group'] == g][metric].dropna()
            if len(g_vals) <= 1:
                continue
            try:
                _t, p_val = stats.ttest_ind(c_vals, g_vals, equal_var=False)
            except Exception:
                continue
            rows.append((metric, ", ".join(control_groups), g, len(c_vals), len(g_vals), p_val, "Yes" if p_val < 0.05 else "No"))

    if not rows:
        return "<p><i>Insufficient data for control-based comparisons.</i></p>"

    html = ["<table><thead><tr><th>Metric</th><th>Control(s)</th><th>Comparison</th><th>N Ctrl</th><th>N Comp</th><th>p-value</th><th>Significant</th></tr></thead><tbody>"]
    for metric, ctrls, comp, n_ctrl, n_comp, pval, sig in rows:
        sig_color = '#d4edda' if sig == 'Yes' else '#f8d7da'
        html.append(
            f"<tr style='background:{sig_color}'><td>{metric.replace('_',' ')}</td><td>{_html.escape(ctrls)}</td><td>{_html.escape(str(comp))}</td><td>{n_ctrl}</td><td>{n_comp}</td><td>{pval:.4g}</td><td>{sig}</td></tr>"
        )
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

    report_controls: List[str] = []
    report_order: List[str] = list(groups_to_compare)
    report_global_model: Optional[str] = None
    requested_sections: set[str] = set()
    if settings:
        report_controls = list(settings.get('report_controls') or [])
        report_order = list(settings.get('report_group_order') or report_order)
        report_global_model = settings.get('report_global_fit_model')
        requested_sections = set(settings.get('report_output_sections') or [])

    def _section_enabled(section: str) -> bool:
        return (not requested_sections) or (section in requested_sections)

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

    # Global fitting section (if present)
    def _extract_total_amplitudes(global_fit_result: dict) -> list[float]:
        vals: list[float] = []
        for params in (global_fit_result.get('individual_params') or []):
            if not isinstance(params, dict):
                continue
            if report_global_model == 'single':
                v = params.get('A')
                if v is not None:
                    vals.append(float(v))
            elif report_global_model == 'double':
                a1 = params.get('A1')
                a2 = params.get('A2')
                if a1 is not None and a2 is not None:
                    vals.append(float(a1) + float(a2))
            elif report_global_model == 'triple':
                a1 = params.get('A1')
                a2 = params.get('A2')
                a3 = params.get('A3')
                if a1 is not None and a2 is not None and a3 is not None:
                    vals.append(float(a1) + float(a2) + float(a3))
        return vals

    any_global = False
    if report_global_model in {'single', 'double', 'triple'}:
        for gname in report_order:
            if gname not in groups_to_compare:
                continue
            g = data_manager.groups.get(gname)
            if g and isinstance(g.get('global_fit'), dict) and report_global_model in g['global_fit']:
                if g['global_fit'][report_global_model].get('success', False):
                    any_global = True
                    break

    if any_global and _section_enabled('global_fits'):
        html_parts.append("<h2>Global Fitting Results</h2>")
        if report_controls:
            html_parts.append(f"<p><b>Control group(s):</b> {_html.escape(', '.join(report_controls))}</p>")
        html_parts.append(f"<p><b>Model:</b> {_html.escape(str(report_global_model))}</p>")

        pooled_ctrl = []
        for gname in report_controls:
            g = data_manager.groups.get(gname) or {}
            r = (g.get('global_fit') or {}).get(report_global_model)
            if r and r.get('success', False):
                pooled_ctrl.extend(_extract_total_amplitudes(r))

        for gname in report_order:
            if gname not in groups_to_compare:
                continue
            g = data_manager.groups.get(gname) or {}
            r = (g.get('global_fit') or {}).get(report_global_model)
            if not r or not r.get('success', False):
                continue

            html_parts.append(f"<h3>{_html.escape(str(gname))}</h3>")
            sp = r.get('shared_params') or {}
            meta_rows = [
                ("N traces", str(len(r.get('file_names') or []))),
                ("Mean R²", f"{float(r.get('mean_r2')):.4f}" if isinstance(r.get('mean_r2'), (int, float)) else "n/a"),
                ("AIC", f"{float(r.get('aic')):.2f}" if isinstance(r.get('aic'), (int, float)) else "n/a"),
            ]
            if report_global_model == 'single':
                meta_rows.append(("k", f"{float(sp.get('k', float('nan'))):.6g}"))
            elif report_global_model == 'double':
                meta_rows.append(("k1", f"{float(sp.get('k1', float('nan'))):.6g}"))
                meta_rows.append(("k2", f"{float(sp.get('k2', float('nan'))):.6g}"))
            elif report_global_model == 'triple':
                meta_rows.append(("k1", f"{float(sp.get('k1', float('nan'))):.6g}"))
                meta_rows.append(("k2", f"{float(sp.get('k2', float('nan'))):.6g}"))
                meta_rows.append(("k3", f"{float(sp.get('k3', float('nan'))):.6g}"))

            html_parts.append("<table><thead><tr><th>Field</th><th>Value</th></tr></thead><tbody>")
            for k, v in meta_rows:
                html_parts.append(f"<tr><td>{_html.escape(str(k))}</td><td>{_html.escape(str(v))}</td></tr>")
            html_parts.append("</tbody></table>")

            # Per-file amplitude table
            html_parts.append("<p><b>Per-file Amplitudes (Global Fit)</b></p>")
            html_parts.append("<table><thead><tr><th>File</th><th>R²</th><th>Total Amplitude</th></tr></thead><tbody>")
            for fname, params, r2v in zip(r.get('file_names') or [], r.get('individual_params') or [], r.get('r2_values') or []):
                total_amp = None
                if isinstance(params, dict):
                    if report_global_model == 'single':
                        total_amp = params.get('A')
                    elif report_global_model == 'double':
                        a1 = params.get('A1')
                        a2 = params.get('A2')
                        if a1 is not None and a2 is not None:
                            total_amp = float(a1) + float(a2)
                    elif report_global_model == 'triple':
                        a1 = params.get('A1')
                        a2 = params.get('A2')
                        a3 = params.get('A3')
                        if a1 is not None and a2 is not None and a3 is not None:
                            total_amp = float(a1) + float(a2) + float(a3)

                html_parts.append(
                    f"<tr><td>{_html.escape(str(fname))}</td><td>{_html.escape(f'{float(r2v):.4f}' if isinstance(r2v,(int,float)) else 'n/a')}</td><td>{_html.escape(f'{float(total_amp):.6g}' if isinstance(total_amp,(int,float)) else 'n/a')}</td></tr>"
                )
            html_parts.append("</tbody></table>")

            # Plotly figure
            if go is not None:
                try:
                    common_time = r.get('common_time')
                    fitted_curves = r.get('fitted_curves') or []
                    fig = go.Figure()
                    if common_time is not None and fitted_curves:
                        group_obj = data_manager.groups.get(gname) or {}
                        excluded = set(r.get('excluded_files') or [])
                        for fname, fitted in zip(r.get('file_names') or [], fitted_curves):
                            file_path = None
                            for fp in (group_obj.get('files') or []):
                                if fp in excluded:
                                    continue
                                if fp in data_manager.files and data_manager.files[fp].get('name') == fname:
                                    file_path = fp
                                    break
                            if file_path is None:
                                continue
                            fd = data_manager.files[file_path]
                            t_post, y_post, _ = get_post_bleach_data(fd['time'], fd['intensity'])
                            fig.add_trace(go.Scatter(x=t_post, y=y_post, mode='markers', marker=dict(size=4, opacity=0.55), showlegend=False))
                            fig.add_trace(go.Scatter(x=common_time, y=fitted, mode='lines', line=dict(width=2), showlegend=False))

                        fig.update_layout(
                            title=f"{gname}: Global {str(report_global_model).title()} Fit",
                            xaxis_title="Time (s)",
                            yaxis_title="Normalized Intensity",
                            height=450,
                            showlegend=False,
                        )
                        html_parts.append(_encode_plot(fig))
                except Exception:
                    html_parts.append("<p><i>Global fit plot unavailable.</i></p>")

            # Control-based amplitude stat
            if stats is not None and pooled_ctrl and (gname not in report_controls):
                sample_vals = _extract_total_amplitudes(r)
                if len(sample_vals) > 1 and len(pooled_ctrl) > 1:
                    try:
                        _t, p_val = stats.ttest_ind(pooled_ctrl, sample_vals, equal_var=False)
                        html_parts.append(
                            f"<p><b>Pooled-control comparison (Total Amplitude):</b> p = {_html.escape(f'{float(p_val):.4g}')}</p>"
                        )
                    except Exception:
                        pass

    # Report-generator metadata
    if settings and _section_enabled('settings'):
        ctrls = settings.get('report_controls')
        subgroup = settings.get('report_subgroup')
        if ctrls:
            html_parts.append(f"<p><b>Control group(s):</b> {_html.escape(', '.join(list(ctrls)))}</p>")
        if subgroup is not None:
            html_parts.append(f"<p><b>Subgroup set:</b> {_html.escape(str(subgroup))}</p>")

    # Settings
    if settings and _section_enabled('settings'):
        settings_df = pd.DataFrame(list(settings.items()), columns=['Parameter', 'Value'])
        html_parts.append("<h2>Analysis Settings</h2>")
        html_parts.append(settings_df.to_html(index=False))

    # Summary Statistics
    if _section_enabled('summary_stats'):
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
    if len(groups_to_compare) > 1 and _section_enabled('stat_tests'):
        html_parts.append("<h2>Statistical Comparison</h2>")
        html_parts.append(_stat_tests_table(combined_df, groups_to_compare))

        # Control-based comparison table (pooled controls)
        if settings and settings.get('report_controls'):
            html_parts.append("<h3>Control-Based Comparisons (Pooled Controls)</h3>")
            html_parts.append(_control_tests_table(combined_df, groups_to_compare, list(settings.get('report_controls'))))

    # Plots (only if plotly available and multiple groups)
    if px is not None and len(groups_to_compare) >= 1 and 'summary_metrics' in locals() and summary_metrics and _section_enabled('plots'):
        try:
            long_df = combined_df.melt(id_vars='group', value_vars=summary_metrics, var_name='Metric', value_name='Value')
            fig_box = px.box(long_df, x='Metric', y='Value', color='group', notched=True,
                             title='Comparison of Key Metrics Across Groups')
            html_parts.append("<h2>Comparison Plots</h2>")
            html_parts.append(_encode_plot(fig_box))
        except Exception:
            html_parts.append("<p><i>Plot generation failed.</i></p>")

    # Detailed results
    if _section_enabled('detailed_results'):
        html_parts.append("<h2>Detailed Results</h2>")
        html_parts.append(combined_df.to_html(index=False))

    # Global multi-spot comparison (special application; included only if present on group)
    multi_spot_any = False
    if _section_enabled('multi_spot'):
        for group_name in groups_to_compare:
            grp = data_manager.groups.get(group_name) if hasattr(data_manager, 'groups') else None
            if not grp:
                continue
            if grp.get('global_multispot_compare') or grp.get('global_multispot_report_md'):
                multi_spot_any = True
                break

    if multi_spot_any and _section_enabled('multi_spot'):
        html_parts.append("<h2>Global Multi-Spot Model Comparison</h2>")
        html_parts.append("<p><i>Included only when run manually; not part of batch processing.</i></p>")
        for group_name in groups_to_compare:
            grp = data_manager.groups.get(group_name)
            if not grp:
                continue
            compare = grp.get('global_multispot_compare')
            report_md = grp.get('global_multispot_report_md')

            if not (compare or report_md):
                continue

            html_parts.append(f"<h3>Group: {_html.escape(str(group_name))}</h3>")

            # Preferred: structured dict
            if isinstance(compare, dict):
                best_model = compare.get('best_model')
                results = compare.get('results') or {}
                html_parts.append(f"<p><b>Best model (AIC):</b> {_html.escape(str(best_model))}</p>")

                # Summary table
                rows = []
                for m, r in results.items():
                    if not isinstance(r, dict):
                        continue
                    rows.append({
                        'model': m,
                        'success': r.get('success'),
                        'aic': r.get('aic'),
                        'delta_aic': r.get('delta_aic'),
                        'rss': r.get('rss'),
                        'params': r.get('params'),
                    })
                if rows:
                    df_ms = pd.DataFrame(rows)
                    html_parts.append(df_ms.to_html(index=False))

                # Detailed per-model sections
                order = [
                    'diffusion_only',
                    'reaction_diffusion',
                    'reaction_diffusion_immobile',
                    'fast_exchange_plus_specific',
                    'two_binding',
                ]
                for m in order:
                    r = results.get(m)
                    if not isinstance(r, dict):
                        continue
                    html_parts.append(f"<h4>Model: {_html.escape(str(m))}</h4>")
                    html_parts.append("<ul>")
                    html_parts.append(f"<li><b>Success:</b> {_html.escape(str(r.get('success')))} ({_html.escape(str(r.get('message')))} )</li>")
                    if r.get('rss') is not None:
                        html_parts.append(f"<li><b>RSS:</b> {_html.escape(str(r.get('rss')))}</li>")
                    if r.get('aic') is not None:
                        html_parts.append(f"<li><b>AIC:</b> {_html.escape(str(r.get('aic')))}</li>")
                    if r.get('delta_aic') is not None:
                        html_parts.append(f"<li><b>ΔAIC:</b> {_html.escape(str(r.get('delta_aic')))}</li>")
                    html_parts.append(f"<li><b>Params:</b> {_html.escape(str(r.get('params')))}</li>")
                    html_parts.append("</ul>")
            else:
                # Fallback: render saved markdown as preformatted text
                html_parts.append("<pre style='white-space:pre-wrap;border:1px solid #ddd;padding:8px'>")
                html_parts.append(_html.escape(str(report_md)))
                html_parts.append("</pre>")

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
