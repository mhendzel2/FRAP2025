"""FRAP PDF Reports Module.

Generates PDF summaries of FRAP analysis results with optional statistical
comparisons. This clean rewrite replaces a previously corrupted version.
"""

from __future__ import annotations

import os
import io
import logging
from datetime import datetime
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER

try:  # SciPy optional
    from scipy import stats  # type: ignore
except Exception:  # pragma: no cover
    stats = None
    logging.warning("scipy not available; statistical tests skipped in PDF report.")

logger = logging.getLogger(__name__)


def _styles():
    s = getSampleStyleSheet()
    if 'FRAPTitle' not in s:
        s.add(ParagraphStyle(name='FRAPTitle', parent=s['Heading1'], fontSize=16, alignment=TA_CENTER))
    if 'FRAPSubtitle' not in s:
        s.add(ParagraphStyle(name='FRAPSubtitle', parent=s['Heading2'], fontSize=14))
    if 'FRAPSection' not in s:
        s.add(ParagraphStyle(name='FRAPSection', parent=s['Heading3'], fontSize=12))
    if 'FRAPBody' not in s:
        s.add(ParagraphStyle(name='FRAPBody', parent=s['Normal'], fontSize=10))
    return s


def generate_pdf_report(data_manager, groups_to_compare: List[str], output_filename: Optional[str] = None, settings: Optional[dict] = None) -> Optional[str]:
    """Create PDF report, returning path or None on failure."""
    try:
        if not groups_to_compare:
            raise ValueError("No groups specified")
        if output_filename is None:
            output_filename = f"FRAP_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        out_path = os.path.abspath(output_filename)

        styles = _styles()
        elements: List = []

        # Title / metadata
        elements.append(Paragraph("FRAP Analysis Report", styles['FRAPTitle']))
        elements.append(Spacer(1, 0.25 * inch))
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        elements.append(Paragraph(f"Generated: {ts}", styles['FRAPBody']))
        elements.append(Paragraph(f"Groups included: {', '.join(groups_to_compare)}", styles['FRAPBody']))
        if settings:
            controls = settings.get('report_controls')
            subgroup = settings.get('report_subgroup')
            if controls:
                elements.append(Paragraph(f"Control group(s): {', '.join(list(controls))}", styles['FRAPBody']))
            if subgroup is not None:
                elements.append(Paragraph(f"Subgroup set: {subgroup}", styles['FRAPBody']))
        elements.append(Spacer(1, 0.25 * inch))

        # Settings table
        if settings:
            rows = [["Parameter", "Value"]]
            for key, label in [
                ('default_criterion', 'Model Selection Criterion'),
                ('default_bleach_radius', 'Bleach Radius (pixels)'),
                ('default_pixel_size', 'Pixel Size (µm/pixel)'),
                ('default_gfp_diffusion', 'Reference GFP Diffusion (µm²/s)')
            ]:
                if key in settings:
                    val = settings[key]
                    rows.append([label, f"{val:.3f}" if isinstance(val, float) else str(val)])
            br = settings.get('default_bleach_radius', 1.0)
            px = settings.get('default_pixel_size', 0.3)
            rows.append(["Effective Bleach Radius (µm)", f"{br * px:.3f}"])
            tbl = Table(rows, colWidths=[3 * inch, 2 * inch])
            tbl.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey)
            ]))
            elements.append(Paragraph("Analysis Settings", styles['FRAPSubtitle']))
            elements.append(tbl)
            elements.append(Spacer(1, 0.25 * inch))

        # Group summaries
        elements.append(Paragraph("Group Summaries", styles['FRAPSubtitle']))
        elements.append(Spacer(1, 0.15 * inch))
        key_metrics = [
            'mobile_fraction', 'immobile_fraction', 'rate_constant', 'k_off', 'half_time',
            'diffusion_coefficient', 'radius_of_gyration', 'molecular_weight_estimate',
            # PDE coupled binding comparison (when available)
            'pde_one_state_D', 'pde_one_state_k_on', 'pde_one_state_k_off', 'pde_one_state_aic',
            'pde_two_state_D', 'pde_two_state_k_on1', 'pde_two_state_k_off1', 'pde_two_state_k_on2', 'pde_two_state_k_off2', 'pde_two_state_aic',
            'pde_delta_aic', 'pde_prefers_two_state'
        ]
        component_metrics = ['fast', 'medium', 'slow']

        # Global multi-spot comparison (special application; included only if present)
        multi_spot_any = False
        for gname in groups_to_compare:
            g = data_manager.groups.get(gname)
            if g and g.get('global_multispot_compare'):
                multi_spot_any = True
                break

        if multi_spot_any:
            elements.append(Paragraph("Global Multi-Spot Model Comparison", styles['FRAPSubtitle']))
            elements.append(Paragraph("(Included only when run manually; not part of batch processing.)", styles['FRAPBody']))
            elements.append(Spacer(1, 0.15 * inch))

            order = [
                'diffusion_only',
                'reaction_diffusion',
                'reaction_diffusion_immobile',
                'fast_exchange_plus_specific',
                'two_binding',
            ]

            for gname in groups_to_compare:
                g = data_manager.groups.get(gname)
                if not g or not isinstance(g.get('global_multispot_compare'), dict):
                    continue
                compare = g['global_multispot_compare']
                best_model = compare.get('best_model')
                results = compare.get('results') or {}

                elements.append(Paragraph(f"Group: {gname}", styles['FRAPSection']))
                elements.append(Paragraph(f"Best model (AIC): {best_model}", styles['FRAPBody']))

                rows = [["Model", "AIC", "ΔAIC", "RSS", "Params"]]
                for m in order:
                    r = results.get(m)
                    if not isinstance(r, dict):
                        continue
                    rows.append([
                        str(m),
                        f"{r.get('aic'):.3f}" if isinstance(r.get('aic'), (int, float)) else str(r.get('aic')),
                        f"{r.get('delta_aic'):.3f}" if isinstance(r.get('delta_aic'), (int, float)) else str(r.get('delta_aic')),
                        f"{r.get('rss'):.6g}" if isinstance(r.get('rss'), (int, float)) else str(r.get('rss')),
                        str(r.get('params')),
                    ])

                tbl = Table(rows, colWidths=[1.3*inch, 0.7*inch, 0.7*inch, 0.9*inch, 2.4*inch])
                tbl.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey)
                ]))
                elements.append(tbl)
                elements.append(Spacer(1, 0.2 * inch))

        for gname in groups_to_compare:
            g = data_manager.groups.get(gname)
            if not g or g.get('features_df') is None or g['features_df'].empty:
                continue
            df = g['features_df']
            elements.append(Paragraph(f"Group: {gname}", styles['FRAPSection']))
            elements.append(Paragraph(f"Total Files: {len(g.get('files', []))}, Analyzed: {len(df)}", styles['FRAPBody']))
            avail = [m for m in key_metrics if m in df.columns]
            if avail:
                rows = [["Metric", "Mean", "Std", "Median", "Min", "Max"]]
                for m in avail:
                    series = df[m].dropna()
                    if not len(series):
                        continue
                    rows.append([
                        m.replace('_', ' ').title(),
                        f"{series.mean():.3f}", f"{series.std():.3f}", f"{series.median():.3f}", f"{series.min():.3f}", f"{series.max():.3f}"
                    ])
                mtbl = Table(rows, colWidths=[1.6*inch, 0.7*inch, 0.7*inch, 0.7*inch, 0.7*inch, 0.7*inch])
                mtbl.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                    ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey)
                ]))
                elements.append(mtbl)
            # component summary
            comp_rows = [["Component", "% Mobile", "Rate (k)", "Half-time (s)"]]
            has_comp = False
            for comp in component_metrics:
                pcol = f'proportion_of_mobile_{comp}'
                rcol = f'rate_constant_{comp}'
                hcol = f'half_time_{comp}'
                if pcol in df.columns and rcol in df.columns:
                    pvals = df[pcol].dropna(); rvals = df[rcol].dropna(); hvals = df[hcol].dropna() if hcol in df.columns else pd.Series(dtype=float)
                    if len(pvals) and len(rvals):
                        comp_rows.append([
                            comp.capitalize(),
                            f"{pvals.mean():.1f}%",
                            f"{rvals.mean():.4f}",
                            f"{hvals.mean():.2f}" if len(hvals) else 'N/A'
                        ])
                        has_comp = True
            if has_comp:
                elements.append(Spacer(1, 0.05 * inch))
                elements.append(Paragraph("Component Analysis", styles['FRAPBody']))
                ctbl = Table(comp_rows, colWidths=[1.3*inch]*4)
                ctbl.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                    ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey)
                ]))
                elements.append(ctbl)
            elements.append(Spacer(1, 0.2 * inch))

        # Statistical comparison
        if len(groups_to_compare) > 1:
            elements.append(Paragraph("Statistical Comparison", styles['FRAPSubtitle']))
            elements.append(Spacer(1, 0.15 * inch))
            combined_parts = []
            for gname in groups_to_compare:
                g = data_manager.groups.get(gname)
                if g and g.get('features_df') is not None and not g['features_df'].empty:
                    tmp = g['features_df'].copy(); tmp['group'] = gname; combined_parts.append(tmp)
            if combined_parts:
                combined = pd.concat(combined_parts, ignore_index=True)
                for metric in ['mobile_fraction', 'rate_constant', 'half_time']:
                    if metric not in combined.columns:
                        continue
                    elements.append(Paragraph(metric.replace('_', ' ').title() + " Comparison", styles['FRAPSection']))
                    stat_rows = [["Group", "N", "Mean", "Std Dev"]]
                    gstats = combined.groupby('group')[metric].agg(['count', 'mean', 'std']).reset_index()
                    for _, r in gstats.iterrows():
                        stat_rows.append([r['group'], int(r['count']), f"{r['mean']:.3f}", f"{r['std']:.3f}"])
                    stbl = Table(stat_rows, colWidths=[1.6*inch, 0.7*inch, 1.0*inch, 1.0*inch])
                    stbl.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                        ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey)
                    ]))
                    elements.append(stbl)
                    if stats is None:
                        elements.append(Paragraph("(SciPy not installed – tests skipped)", styles['FRAPBody']))
                    else:
                        if len(groups_to_compare) == 2:
                            g1 = combined[combined.group == groups_to_compare[0]][metric].dropna()
                            g2 = combined[combined.group == groups_to_compare[1]][metric].dropna()
                            if len(g1) > 1 and len(g2) > 1:
                                t, p = stats.ttest_ind(g1, g2, equal_var=False)
                                elements.append(Paragraph(f"Welch t-test: t={t:.3f}, p={p:.4f}{' (significant)' if p < 0.05 else ''}", styles['FRAPBody']))
                        else:
                            series = [combined[combined.group == g][metric].dropna() for g in groups_to_compare]
                            series = [s for s in series if len(s) > 1]
                            if len(series) >= 2:
                                F, p = stats.f_oneway(*series)
                                elements.append(Paragraph(f"ANOVA: F={F:.3f}, p={p:.4f}{' (significant)' if p < 0.05 else ''}", styles['FRAPBody']))
                    elements.append(Spacer(1, 0.15 * inch))

            # Control-based comparisons (pooled controls vs each sample)
            if settings and settings.get('report_controls'):
                control_groups = [g for g in settings.get('report_controls') if g in groups_to_compare]
                if control_groups:
                    elements.append(Paragraph("Control-Based Comparisons (Pooled Controls)", styles['FRAPSubtitle']))
                    elements.append(Spacer(1, 0.10 * inch))

                    if stats is None:
                        elements.append(Paragraph("(SciPy not installed – control-based tests skipped)", styles['FRAPBody']))
                    else:
                        pooled = combined[combined['group'].isin(control_groups)]
                        for metric in ['mobile_fraction', 'rate_constant', 'half_time']:
                            if metric not in combined.columns:
                                continue
                            c_vals = pooled[metric].dropna()
                            if len(c_vals) <= 1:
                                continue

                            rows = [["Metric", "Control(s)", "Comparison", "N Control", "N Comp", "p-value", "Significant"]]
                            for g in groups_to_compare:
                                if g in control_groups:
                                    continue
                                g_vals = combined[combined['group'] == g][metric].dropna()
                                if len(g_vals) <= 1:
                                    continue
                                try:
                                    _, p = stats.ttest_ind(c_vals, g_vals, equal_var=False)
                                except Exception:
                                    continue
                                rows.append([
                                    metric.replace('_', ' ').title(),
                                    ", ".join(control_groups),
                                    g,
                                    int(len(c_vals)),
                                    int(len(g_vals)),
                                    f"{p:.4g}",
                                    "Yes" if p < 0.05 else "No",
                                ])

                            if len(rows) > 1:
                                tbl = Table(rows, colWidths=[1.3*inch, 1.2*inch, 1.2*inch, 0.7*inch, 0.7*inch, 0.8*inch, 0.8*inch])
                                tbl.setStyle(TableStyle([
                                    ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                    ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                                    ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey)
                                ]))
                                elements.append(Paragraph(metric.replace('_', ' ').title(), styles['FRAPSection']))
                                elements.append(tbl)
                                elements.append(Spacer(1, 0.12 * inch))

        # Simple plot (mobile fraction boxplot)
        if len(groups_to_compare) > 1:
            try:
                plt.figure(figsize=(6.0, 4.0))
                labels, data = [], []
                for gname in groups_to_compare:
                    g = data_manager.groups.get(gname)
                    if g and g.get('features_df') is not None and 'mobile_fraction' in g['features_df']:
                        vals = g['features_df']['mobile_fraction'].dropna()
                        if len(vals):
                            labels.append(gname); data.append(vals)
                if data:
                    plt.boxplot(data, labels=labels)
                    plt.title('Mobile Fraction Comparison')
                    plt.ylabel('Mobile Fraction (%)')
                    plt.grid(alpha=0.4, linestyle='--')
                    buf = io.BytesIO(); plt.savefig(buf, format='png', dpi=140); buf.seek(0)
                    elements.append(Paragraph("Mobile Fraction Comparison", styles['FRAPSection']))
                    elements.append(Image(buf, width=6 * inch, height=4 * inch))
            except Exception as pe:  # pragma: no cover
                elements.append(Paragraph(f"Plot error: {pe}", styles['FRAPBody']))
            finally:
                plt.close()

        # Build PDF
        doc = SimpleDocTemplate(out_path, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
        doc.build(elements)
        return out_path
    except Exception as e:  # pragma: no cover
        logger.error(f"Error generating PDF report: {e}")
        import traceback; logger.error(traceback.format_exc())
        return None


def add_matplotlib_plot_to_pdf(elements, plot_function, *args, **kwargs) -> bool:  # legacy helper
    try:
        fig = plot_function(*args, **kwargs)
        buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=150); buf.seek(0)
        elements.append(Image(buf, width=6 * inch, height=4 * inch))
        plt.close(fig)
        return True
    except Exception as e:  # pragma: no cover
        logger.error(f"Error adding plot to PDF: {e}")
        return False


if __name__ == '__main__':  # pragma: no cover
    print("PDF report module ready.")
