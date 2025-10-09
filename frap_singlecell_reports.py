"""
Report Generation for FRAP Single-Cell Analysis
Creates PDF and HTML reports with tables, figures, and methods sections
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any
import logging
import json
import base64
from io import BytesIO

# PDF generation
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.units import inch
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, Image as RLImage, KeepTogether
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    logging.warning("reportlab not available, PDF generation disabled")

# HTML generation
try:
    from jinja2 import Template
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False
    logging.warning("jinja2 not available, HTML generation may be limited")

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# PDF REPORT GENERATION
# ============================================================================

def build_pdf_report(
    cell_features: pd.DataFrame,
    stats_results: Optional[Dict] = None,
    figures: Optional[Dict[str, Any]] = None,
    output_path: str = 'frap_report.pdf',
    title: str = 'FRAP Single-Cell Analysis Report',
    recipe: Optional[Dict] = None,
    **kwargs
) -> bool:
    """
    Generate a comprehensive PDF report
    
    Parameters
    ----------
    cell_features : pd.DataFrame
        Cell features table with fitted parameters
    stats_results : dict, optional
        Results from analyze() with 'comparisons' and 'cluster_stats'
    figures : dict, optional
        Dictionary of matplotlib figures {'name': fig}
    output_path : str
        Output PDF path
    title : str
        Report title
    recipe : dict, optional
        Analysis recipe with parameters and versions
    **kwargs
        Additional metadata
    
    Returns
    -------
    bool
        True if successful
    """
    if not REPORTLAB_AVAILABLE:
        logger.error("reportlab not installed. Install with: pip install reportlab")
        return False
    
    try:
        logger.info(f"Generating PDF report: {output_path}")
        
        # Create document
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=1*inch,
            bottomMargin=0.75*inch
        )
        
        # Build content
        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            fontSize=24,
            textColor=colors.HexColor('#0066CC'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading1'],
            fontSize=16,
            textColor=colors.HexColor('#333333'),
            spaceAfter=12,
            spaceBefore=12
        )
        
        # Title
        story.append(Paragraph(title, title_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Summary section
        story.extend(_build_summary_section(cell_features, styles))
        story.append(Spacer(1, 0.3*inch))
        
        # Statistics section
        if stats_results:
            story.extend(_build_statistics_section(stats_results, styles, heading_style))
            story.append(Spacer(1, 0.3*inch))
        
        # Figures section
        if figures:
            story.extend(_build_figures_section(figures, styles, heading_style))
            story.append(PageBreak())
        
        # Methods section
        story.extend(_build_methods_section(cell_features, recipe, styles, heading_style))
        story.append(Spacer(1, 0.3*inch))
        
        # Footer with recipe
        if recipe:
            story.extend(_build_footer_section(recipe, styles))
        
        # Build PDF
        doc.build(story)
        
        logger.info(f"✓ PDF report saved: {output_path}")
        return True
        
    except Exception as e:
        logger.exception(f"Failed to generate PDF report: {e}")
        return False


def _build_summary_section(df: pd.DataFrame, styles) -> List:
    """Build summary section for PDF"""
    elements = []
    
    heading_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading1'],
        fontSize=16,
        textColor=colors.HexColor('#0066CC'),
        spaceAfter=12
    )
    
    elements.append(Paragraph("Summary", heading_style))
    
    # Compute summary stats
    n_cells = len(df)
    n_experiments = df['exp_id'].nunique()
    conditions = df['condition'].unique() if 'condition' in df.columns else ['unknown']
    n_conditions = len(conditions)
    n_clusters = len(df['cluster'].unique()) - (1 if -1 in df['cluster'].values else 0) if 'cluster' in df.columns else 0
    n_outliers = df['outlier'].sum() if 'outlier' in df.columns else 0
    qc_pass_rate = df['bleach_qc'].mean() * 100 if 'bleach_qc' in df.columns else 0
    
    # Summary table
    summary_data = [
        ['Metric', 'Value'],
        ['Total Cells', f'{n_cells:,}'],
        ['Experiments', f'{n_experiments}'],
        ['Conditions', ', '.join(map(str, conditions[:3])) + ('...' if n_conditions > 3 else '')],
        ['Clusters Identified', f'{n_clusters}'],
        ['Outliers', f'{n_outliers} ({n_outliers/n_cells*100:.1f}%)'],
        ['QC Pass Rate', f'{qc_pass_rate:.1f}%'],
        ['Analysis Date', datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
    ]
    
    t = Table(summary_data, colWidths=[2.5*inch, 3*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0066CC')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
    ]))
    
    elements.append(t)
    
    return elements


def _build_statistics_section(stats_results: Dict, styles, heading_style) -> List:
    """Build statistics section for PDF"""
    elements = []
    
    elements.append(Paragraph("Statistical Analysis", heading_style))
    
    if 'comparisons' in stats_results:
        comparisons_df = stats_results['comparisons']
        
        # Filter significant results
        sig_results = comparisons_df[comparisons_df['significant'] == True]
        
        elements.append(Paragraph(
            f"<b>Significant Comparisons:</b> {len(sig_results)} of {len(comparisons_df)} "
            f"({len(sig_results)/len(comparisons_df)*100:.1f}% at FDR < 0.05)",
            styles['Normal']
        ))
        elements.append(Spacer(1, 0.1*inch))
        
        # Build table
        if not sig_results.empty:
            table_data = [['Parameter', 'Comparison', 'Effect Size (g)', 'p-value', 'q-value (FDR)']]
            
            for _, row in sig_results.iterrows():
                table_data.append([
                    row['param'],
                    row['comparison'],
                    f"{row['hedges_g']:.3f}",
                    f"{row['p']:.4f}",
                    f"{row['q']:.4f}"
                ])
            
            t = Table(table_data, colWidths=[1.2*inch, 1.5*inch, 1.2*inch, 1*inch, 1*inch])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0066CC')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
            ]))
            
            elements.append(t)
        else:
            elements.append(Paragraph(
                "<i>No significant differences detected at FDR < 0.05</i>",
                styles['Normal']
            ))
    
    # Cluster statistics
    if 'cluster_stats' in stats_results:
        elements.append(Spacer(1, 0.2*inch))
        elements.append(Paragraph("<b>Cluster Statistics:</b>", styles['Normal']))
        elements.append(Spacer(1, 0.1*inch))
        
        cluster_stats = stats_results['cluster_stats']
        
        for param, param_stats in cluster_stats.items():
            table_data = [['Cluster', 'N', 'Mean ± SD', 'Median', 'Q25-Q75']]
            
            for cluster_id in sorted(param_stats['mean'].keys()):
                if cluster_id == -1:
                    continue
                
                mean = param_stats['mean'][cluster_id]
                std = param_stats['std'][cluster_id]
                median = param_stats['median'][cluster_id]
                q25 = param_stats['q25'][cluster_id]
                q75 = param_stats['q75'][cluster_id]
                n = param_stats['n'][cluster_id]
                
                table_data.append([
                    f'Cluster {cluster_id}',
                    f'{n}',
                    f'{mean:.3f} ± {std:.3f}',
                    f'{median:.3f}',
                    f'{q25:.3f}-{q75:.3f}'
                ])
            
            elements.append(Paragraph(f"<b>{param}</b>", styles['Normal']))
            
            t = Table(table_data, colWidths=[1*inch, 0.8*inch, 1.5*inch, 1*inch, 1.5*inch])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#FF8C00')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
            ]))
            
            elements.append(t)
            elements.append(Spacer(1, 0.1*inch))
    
    return elements


def _build_figures_section(figures: Dict, styles, heading_style) -> List:
    """Build figures section for PDF"""
    elements = []
    
    elements.append(Paragraph("Figures", heading_style))
    
    for fig_name, fig in figures.items():
        # Save figure to buffer
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        
        # Add to PDF
        img = RLImage(buf, width=6*inch, height=4*inch)
        
        caption = Paragraph(f"<b>Figure:</b> {fig_name}", styles['Normal'])
        
        elements.append(KeepTogether([img, Spacer(1, 0.1*inch), caption]))
        elements.append(Spacer(1, 0.3*inch))
        
        buf.close()
    
    return elements


def _build_methods_section(df: pd.DataFrame, recipe: Optional[Dict], styles, heading_style) -> List:
    """Build methods section for PDF"""
    elements = []
    
    elements.append(Paragraph("Methods", heading_style))
    
    methods_text = f"""
    <b>ROI Tracking:</b> Cells were tracked using a multi-method approach combining 
    Gaussian centroid fitting (2D Gaussian with scipy.optimize.curve_fit), 
    Kalman filtering (constant velocity model with process noise), and 
    Lucas-Kanade optical flow (OpenCV). Adaptive radius adjustment used 
    Sobel gradient analysis. Quality control flags were applied for drift 
    >10 pixels and tracking innovation >3 standard deviations.
    <br/><br/>
    
    <b>Signal Extraction:</b> Intensity signals were extracted from circular ROIs 
    with background correction using an annular ring (5-10 pixel offset). 
    Background values used outlier-resistant median calculation (IQR filtering). 
    Signals were normalized as (I - I0) / (I∞ - I0) where I0 is the bleach 
    minimum and I∞ is the plateau intensity.
    <br/><br/>
    
    <b>Curve Fitting:</b> Recovery curves were fitted to single-exponential 
    I(t) = A - B·exp(-k·t) using robust least-squares (soft_l1 loss) with 
    scipy.optimize.least_squares. Two-exponential models were tested when 
    appropriate, with selection via BIC (threshold ΔBIC > 10). Mobile fraction 
    was computed as (I∞ - I0) / (pre_bleach - I0). Parallel processing used 
    joblib with n_jobs=-1.
    <br/><br/>
    
    <b>Outlier Detection:</b> Statistical outliers were identified using an 
    ensemble of Isolation Forest (n_estimators=300, contamination={recipe.get('parameters', {}).get('clustering', {}).get('contamination', 0.07) if recipe else 0.07}) 
    and Elliptic Envelope (contamination matching). Features were scaled with 
    RobustScaler before detection.
    <br/><br/>
    
    <b>Clustering:</b> Subpopulations were identified using Gaussian Mixture Models 
    with BIC-based selection (k=1 to {recipe.get('parameters', {}).get('clustering', {}).get('max_k', 6) if recipe else 6}). 
    DBSCAN was used as fallback for noisy data. Cluster probabilities provided 
    confidence scores.
    <br/><br/>
    
    <b>Statistical Analysis:</b> Group comparisons used linear mixed-effects models 
    (LMM) with random intercepts by experiment batch (statsmodels.mixedlm). 
    Effect sizes computed as Hedges' g with small-sample correction. Bootstrap 
    confidence intervals used bias-corrected and accelerated (BCa) method 
    (n_bootstrap={recipe.get('parameters', {}).get('statistics', {}).get('n_bootstrap', 1000) if recipe else 1000}). 
    Multiple comparisons corrected using Benjamini-Hochberg FDR (α=0.05).
    <br/><br/>
    
    <b>Software:</b> Analysis performed using Python {recipe.get('software_versions', {}).get('python', '3.10+') if recipe else '3.10+'} 
    with NumPy {recipe.get('software_versions', {}).get('numpy', '1.24+') if recipe else '1.24+'}, 
    SciPy {recipe.get('software_versions', {}).get('scipy', '1.10+') if recipe else '1.10+'}, 
    Pandas {recipe.get('software_versions', {}).get('pandas', '2.0+') if recipe else '2.0+'}, 
    scikit-learn {recipe.get('software_versions', {}).get('sklearn', '1.3+') if recipe else '1.3+'}, 
    and statsmodels {recipe.get('software_versions', {}).get('statsmodels', '0.14+') if recipe else '0.14+'}.
    """
    
    elements.append(Paragraph(methods_text, ParagraphStyle(
        'Methods',
        parent=styles['Normal'],
        alignment=TA_JUSTIFY,
        fontSize=9,
        leading=12
    )))
    
    return elements


def _build_footer_section(recipe: Dict, styles) -> List:
    """Build footer section with recipe"""
    elements = []
    
    if recipe:
        import hashlib
        recipe_str = json.dumps(recipe, sort_keys=True)
        recipe_hash = hashlib.md5(recipe_str.encode()).hexdigest()[:8]
        
        footer_text = f"""
        <br/><br/>
        <font size="8" color="grey">
        <b>Analysis Recipe:</b> #{recipe_hash}<br/>
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
        Reproducibility: Full recipe available in report metadata
        </font>
        """
        
        elements.append(Paragraph(footer_text, styles['Normal']))
    
    return elements


# ============================================================================
# HTML REPORT GENERATION
# ============================================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        h1 {
            color: #0066CC;
            font-size: 2.5rem;
            margin-bottom: 10px;
            text-align: center;
        }
        
        h2 {
            color: #0066CC;
            font-size: 1.8rem;
            margin-top: 30px;
            margin-bottom: 15px;
            border-bottom: 2px solid #0066CC;
            padding-bottom: 5px;
        }
        
        h3 {
            color: #FF8C00;
            font-size: 1.3rem;
            margin-top: 20px;
            margin-bottom: 10px;
        }
        
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }
        
        .metric-card h3 {
            color: white;
            font-size: 0.9rem;
            margin: 0 0 5px 0;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .metric-card .value {
            font-size: 2rem;
            font-weight: bold;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        
        table th {
            background: #0066CC;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }
        
        table td {
            padding: 10px 12px;
            border-bottom: 1px solid #ddd;
        }
        
        table tr:nth-child(even) {
            background: #f9f9f9;
        }
        
        table tr:hover {
            background: #f0f0f0;
        }
        
        .figure {
            margin: 30px 0;
            text-align: center;
        }
        
        .figure img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        .figure-caption {
            margin-top: 10px;
            font-style: italic;
            color: #666;
        }
        
        .methods {
            background: #f9f9f9;
            padding: 20px;
            border-left: 4px solid #0066CC;
            margin: 20px 0;
            line-height: 1.8;
        }
        
        .footer {
            margin-top: 40px;
            padding-top: 20px;
            border-top: 2px solid #ddd;
            font-size: 0.85rem;
            color: #666;
            text-align: center;
        }
        
        .badge {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 0.85rem;
            font-weight: 600;
        }
        
        .badge-success {
            background: #28a745;
            color: white;
        }
        
        .badge-warning {
            background: #ffc107;
            color: #333;
        }
        
        @media print {
            body { background: white; padding: 0; }
            .container { box-shadow: none; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>{{ title }}</h1>
        <p style="text-align: center; color: #666; margin-bottom: 30px;">
            Generated: {{ timestamp }}
        </p>
        
        <h2>Summary</h2>
        <div class="summary-grid">
            <div class="metric-card">
                <h3>Cells</h3>
                <div class="value">{{ n_cells }}</div>
            </div>
            <div class="metric-card">
                <h3>Experiments</h3>
                <div class="value">{{ n_experiments }}</div>
            </div>
            <div class="metric-card">
                <h3>Clusters</h3>
                <div class="value">{{ n_clusters }}</div>
            </div>
            <div class="metric-card">
                <h3>Outliers</h3>
                <div class="value">{{ n_outliers }}</div>
            </div>
        </div>
        
        {% if stats_table %}
        <h2>Statistical Analysis</h2>
        <p><strong>Significant Results:</strong> {{ n_significant }} of {{ n_total }} comparisons (FDR < 0.05)</p>
        {{ stats_table }}
        {% endif %}
        
        {% if cluster_tables %}
        <h2>Cluster Statistics</h2>
        {% for param, table in cluster_tables.items() %}
        <h3>{{ param }}</h3>
        {{ table }}
        {% endfor %}
        {% endif %}
        
        {% if figures %}
        <h2>Figures</h2>
        {% for name, img_data in figures.items() %}
        <div class="figure">
            <img src="data:image/png;base64,{{ img_data }}" alt="{{ name }}">
            <div class="figure-caption">Figure: {{ name }}</div>
        </div>
        {% endfor %}
        {% endif %}
        
        <h2>Methods</h2>
        <div class="methods">
            {{ methods_html|safe }}
        </div>
        
        <div class="footer">
            {% if recipe_hash %}
            <strong>Analysis Recipe:</strong> #{{ recipe_hash }}<br>
            {% endif %}
            Full recipe available in report metadata<br>
            Generated with FRAP Single-Cell Analysis v1.0
        </div>
    </div>
</body>
</html>
"""


def build_html_report(
    cell_features: pd.DataFrame,
    stats_results: Optional[Dict] = None,
    figures: Optional[Dict[str, Any]] = None,
    output_path: str = 'frap_report.html',
    title: str = 'FRAP Single-Cell Analysis Report',
    recipe: Optional[Dict] = None,
    **kwargs
) -> bool:
    """
    Generate HTML report with interactive features
    
    Parameters
    ----------
    cell_features : pd.DataFrame
        Cell features table
    stats_results : dict, optional
        Statistical analysis results
    figures : dict, optional
        Dictionary of matplotlib figures
    output_path : str
        Output HTML path
    title : str
        Report title
    recipe : dict, optional
        Analysis recipe
    
    Returns
    -------
    bool
        True if successful
    """
    try:
        logger.info(f"Generating HTML report: {output_path}")
        
        # Prepare data
        context = {
            'title': title,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'n_cells': len(cell_features),
            'n_experiments': cell_features['exp_id'].nunique(),
            'n_clusters': len(cell_features['cluster'].unique()) - (1 if -1 in cell_features['cluster'].values else 0) if 'cluster' in cell_features.columns else 0,
            'n_outliers': cell_features['outlier'].sum() if 'outlier' in cell_features.columns else 0
        }
        
        # Statistics table
        if stats_results and 'comparisons' in stats_results:
            comparisons_df = stats_results['comparisons']
            sig_results = comparisons_df[comparisons_df['significant'] == True]
            
            context['n_significant'] = len(sig_results)
            context['n_total'] = len(comparisons_df)
            context['stats_table'] = sig_results[['param', 'comparison', 'hedges_g', 'p', 'q']].to_html(
                index=False,
                float_format='%.4f',
                classes='table'
            )
        
        # Cluster statistics
        if stats_results and 'cluster_stats' in stats_results:
            cluster_tables = {}
            for param, param_stats in stats_results['cluster_stats'].items():
                # Build DataFrame
                rows = []
                for cluster_id in sorted(param_stats['mean'].keys()):
                    if cluster_id == -1:
                        continue
                    rows.append({
                        'Cluster': f'Cluster {cluster_id}',
                        'N': param_stats['n'][cluster_id],
                        'Mean': f"{param_stats['mean'][cluster_id]:.3f}",
                        'SD': f"{param_stats['std'][cluster_id]:.3f}",
                        'Median': f"{param_stats['median'][cluster_id]:.3f}"
                    })
                
                cluster_df = pd.DataFrame(rows)
                cluster_tables[param] = cluster_df.to_html(index=False, classes='table')
            
            context['cluster_tables'] = cluster_tables
        
        # Figures
        if figures:
            fig_data = {}
            for name, fig in figures.items():
                buf = BytesIO()
                fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                buf.seek(0)
                img_base64 = base64.b64encode(buf.read()).decode('utf-8')
                fig_data[name] = img_base64
                buf.close()
            
            context['figures'] = fig_data
        
        # Methods
        context['methods_html'] = _build_methods_html(recipe)
        
        # Recipe hash
        if recipe:
            import hashlib
            recipe_str = json.dumps(recipe, sort_keys=True)
            context['recipe_hash'] = hashlib.md5(recipe_str.encode()).hexdigest()[:8]
        
        # Render template
        if JINJA2_AVAILABLE:
            template = Template(HTML_TEMPLATE)
            html = template.render(**context)
        else:
            # Simple string replacement
            html = HTML_TEMPLATE
            for key, value in context.items():
                if isinstance(value, str):
                    html = html.replace(f'{{{{ {key} }}}}', value)
        
        # Write file
        Path(output_path).write_text(html, encoding='utf-8')
        
        logger.info(f"✓ HTML report saved: {output_path}")
        return True
        
    except Exception as e:
        logger.exception(f"Failed to generate HTML report: {e}")
        return False


def _build_methods_html(recipe: Optional[Dict]) -> str:
    """Build methods section for HTML"""
    contamination = recipe.get('parameters', {}).get('clustering', {}).get('contamination', 0.07) if recipe else 0.07
    max_k = recipe.get('parameters', {}).get('clustering', {}).get('max_k', 6) if recipe else 6
    n_bootstrap = recipe.get('parameters', {}).get('statistics', {}).get('n_bootstrap', 1000) if recipe else 1000
    
    versions = recipe.get('software_versions', {}) if recipe else {}
    python_ver = versions.get('python', '3.10+')
    numpy_ver = versions.get('numpy', '1.24+')
    scipy_ver = versions.get('scipy', '1.10+')
    pandas_ver = versions.get('pandas', '2.0+')
    sklearn_ver = versions.get('sklearn', '1.3+')
    statsmodels_ver = versions.get('statsmodels', '0.14+')
    
    return f"""
    <p><strong>ROI Tracking:</strong> Cells were tracked using a multi-method approach combining 
    Gaussian centroid fitting (2D Gaussian with scipy.optimize.curve_fit), Kalman filtering 
    (constant velocity model), and Lucas-Kanade optical flow (OpenCV). Adaptive radius adjustment 
    used Sobel gradient analysis. Quality control flags were applied for drift >10 pixels.</p>
    
    <p><strong>Signal Extraction:</strong> Intensity signals were extracted from circular ROIs 
    with background correction using an annular ring (5-10 pixel offset). Background values used 
    outlier-resistant median calculation. Signals were normalized as (I - I0) / (I∞ - I0).</p>
    
    <p><strong>Curve Fitting:</strong> Recovery curves were fitted to single-exponential 
    I(t) = A - B·exp(-k·t) using robust least-squares (soft_l1 loss). Two-exponential models 
    were tested with BIC-based selection (ΔBIC > 10). Mobile fraction computed as 
    (I∞ - I0) / (pre_bleach - I0).</p>
    
    <p><strong>Outlier Detection:</strong> Statistical outliers identified using Isolation Forest 
    (n_estimators=300, contamination={contamination}) and Elliptic Envelope ensemble. Features 
    scaled with RobustScaler.</p>
    
    <p><strong>Clustering:</strong> Subpopulations identified using Gaussian Mixture Models with 
    BIC-based selection (k=1 to {max_k}). DBSCAN used as fallback for noisy data.</p>
    
    <p><strong>Statistical Analysis:</strong> Group comparisons used linear mixed-effects models 
    with random intercepts. Effect sizes computed as Hedges' g with small-sample correction. 
    Bootstrap confidence intervals used BCa method (n_bootstrap={n_bootstrap}). Multiple 
    comparisons corrected using Benjamini-Hochberg FDR (α=0.05).</p>
    
    <p><strong>Software:</strong> Python {python_ver}, NumPy {numpy_ver}, SciPy {scipy_ver}, 
    Pandas {pandas_ver}, scikit-learn {sklearn_ver}, statsmodels {statsmodels_ver}.</p>
    """


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def build_report(
    cell_features: pd.DataFrame,
    stats_results: Optional[Dict] = None,
    figures: Optional[Dict[str, Any]] = None,
    output_path: str = 'frap_report.pdf',
    format: str = 'pdf',
    **kwargs
) -> bool:
    """
    Build report in specified format
    
    Parameters
    ----------
    cell_features : pd.DataFrame
        Cell features table
    stats_results : dict, optional
        Statistical results
    figures : dict, optional
        Matplotlib figures
    output_path : str
        Output path
    format : str
        'pdf' or 'html'
    **kwargs
        Additional arguments
    
    Returns
    -------
    bool
        True if successful
    """
    if format.lower() == 'pdf':
        return build_pdf_report(cell_features, stats_results, figures, output_path, **kwargs)
    elif format.lower() == 'html':
        return build_html_report(cell_features, stats_results, figures, output_path, **kwargs)
    else:
        raise ValueError(f"Unknown format: {format}. Use 'pdf' or 'html'")


if __name__ == '__main__':
    # Test report generation
    print("Testing report generation...")
    
    # Generate synthetic data
    from test_synthetic import synth_multi_movie_dataset
    from frap_singlecell_api import analyze
    
    traces, features = synth_multi_movie_dataset(
        n_movies=4,
        n_cells_per_movie=20,
        conditions=['control', 'treated']
    )
    
    # Run analysis
    stats = analyze(features, params=['mobile_frac', 'k', 't_half'])
    
    # Generate reports
    print("\nGenerating PDF report...")
    success = build_report(
        features,
        stats_results=stats,
        output_path='test_report.pdf',
        format='pdf',
        title='Test FRAP Analysis Report'
    )
    print(f"PDF: {'✓' if success else '✗'}")
    
    print("\nGenerating HTML report...")
    success = build_report(
        features,
        stats_results=stats,
        output_path='test_report.html',
        format='html',
        title='Test FRAP Analysis Report'
    )
    print(f"HTML: {'✓' if success else '✗'}")
    
    print("\nDone!")
