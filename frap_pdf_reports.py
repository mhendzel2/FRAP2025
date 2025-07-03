"""
FRAP PDF Report Generator
Automated report generation with statistical analysis and visualization
"""

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from scipy import stats
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

def create_statistical_plots(combined_df, param='mobile_fraction'):
    """Create statistical visualization plots for PDF report"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Box plot
    sns.boxplot(data=combined_df, x='group', y=param, ax=ax1)
    ax1.set_title(f'{param.replace("_", " ").title()} Distribution by Group')
    ax1.tick_params(axis='x', rotation=45)
    
    # Violin plot
    sns.violinplot(data=combined_df, x='group', y=param, ax=ax2)
    ax2.set_title(f'{param.replace("_", " ").title()} Density by Group')
    ax2.tick_params(axis='x', rotation=45)
    
    # Bar plot with error bars
    group_stats = combined_df.groupby('group')[param].agg(['mean', 'sem']).reset_index()
    ax3.bar(group_stats['group'], group_stats['mean'], yerr=group_stats['sem'], capsize=5)
    ax3.set_title(f'Mean {param.replace("_", " ").title()} (±SEM)')
    ax3.tick_params(axis='x', rotation=45)
    
    # Individual data points with means
    for i, group in enumerate(combined_df['group'].unique()):
        group_data = combined_df[combined_df['group'] == group][param]
        x_vals = np.random.normal(i, 0.04, size=len(group_data))
        ax4.scatter(x_vals, group_data, alpha=0.6, s=30)
        ax4.scatter(i, group_data.mean(), color='red', s=100, marker='_', linewidth=3)
    
    ax4.set_xticks(range(len(combined_df['group'].unique())))
    ax4.set_xticklabels(combined_df['group'].unique(), rotation=45)
    ax4.set_title(f'{param.replace("_", " ").title()} - Individual Points & Means')
    
    plt.tight_layout()
    
    # Save to buffer
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
    img_buffer.seek(0)
    plt.close()
    
    return img_buffer

def perform_comprehensive_statistics(combined_df, param='mobile_fraction'):
    """Perform comprehensive statistical analysis between groups"""
    results = {
        'parameter': param,
        'groups': list(combined_df['group'].unique()),
        'group_stats': {},
        'statistical_tests': {},
        'effect_sizes': {}
    }
    
    # Group statistics
    for group in results['groups']:
        group_data = combined_df[combined_df['group'] == group][param].dropna()
        results['group_stats'][group] = {
            'n': len(group_data),
            'mean': group_data.mean(),
            'std': group_data.std(),
            'sem': group_data.std() / np.sqrt(len(group_data)),
            'median': group_data.median(),
            'q25': group_data.quantile(0.25),
            'q75': group_data.quantile(0.75)
        }
    
    # Statistical tests
    if len(results['groups']) == 2:
        # Two-group comparison
        group1_data = combined_df[combined_df['group'] == results['groups'][0]][param].dropna()
        group2_data = combined_df[combined_df['group'] == results['groups'][1]][param].dropna()
        
        # Normality tests
        _, p_norm1 = stats.shapiro(group1_data) if len(group1_data) <= 5000 else (None, 0.05)
        _, p_norm2 = stats.shapiro(group2_data) if len(group2_data) <= 5000 else (None, 0.05)
        
        # Choose appropriate test
        if p_norm1 > 0.05 and p_norm2 > 0.05:
            t_stat, p_value = stats.ttest_ind(group1_data, group2_data)
            test_name = "Student's t-test"
        else:
            t_stat, p_value = stats.mannwhitneyu(group1_data, group2_data, alternative='two-sided')
            test_name = "Mann-Whitney U test"
        
        results['statistical_tests']['two_group'] = {
            'test': test_name,
            'statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
        
        # Cohen's d
        pooled_std = np.sqrt(((len(group1_data)-1)*group1_data.var() + (len(group2_data)-1)*group2_data.var()) / (len(group1_data)+len(group2_data)-2))
        cohens_d = (group1_data.mean() - group2_data.mean()) / pooled_std
        
        results['effect_sizes']['cohens_d'] = {
            'value': cohens_d,
            'magnitude': 'Small' if abs(cohens_d) < 0.2 else 'Medium' if abs(cohens_d) < 0.8 else 'Large'
        }
    
    elif len(results['groups']) > 2:
        # Multi-group comparison
        groups_data = []
        for group in results['groups']:
            group_data = combined_df[combined_df['group'] == group][param].dropna()
            if len(group_data) > 1:
                groups_data.append(group_data)
        
        if len(groups_data) >= 2:
            # ANOVA
            f_stat, p_anova = stats.f_oneway(*groups_data)
            
            results['statistical_tests']['anova'] = {
                'test': 'One-way ANOVA',
                'f_statistic': f_stat,
                'p_value': p_anova,
                'significant': p_anova < 0.05
            }
            
            # Post-hoc pairwise comparisons
            if p_anova < 0.05:
                pairwise_results = []
                for i in range(len(groups_data)):
                    for j in range(i+1, len(groups_data)):
                        _, p_pair = stats.ttest_ind(groups_data[i], groups_data[j])
                        # Bonferroni correction
                        n_comparisons = len(groups_data) * (len(groups_data) - 1) // 2
                        p_corrected = min(p_pair * n_comparisons, 1.0)
                        
                        pairwise_results.append({
                            'group1': results['groups'][i],
                            'group2': results['groups'][j],
                            'p_value': p_pair,
                            'p_corrected': p_corrected,
                            'significant': p_corrected < 0.05
                        })
                
                results['statistical_tests']['pairwise'] = pairwise_results
    
    return results

def generate_pdf_report(data_manager, groups_to_compare=None, output_filename=None, settings=None):
    """Generate comprehensive PDF report with automated group analysis"""
    
    if not output_filename:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f'FRAP_Comprehensive_Report_{timestamp}.pdf'
    
    # Prepare data
    if groups_to_compare is None:
        groups_to_compare = list(data_manager.groups.keys())
    
    if len(groups_to_compare) < 2:
        raise ValueError("Need at least 2 groups for comparison analysis")
    
    # Combine data from selected groups
    all_group_data = []
    for group_name in groups_to_compare:
        if group_name in data_manager.groups:
            group_info = data_manager.groups[group_name]
            if group_info.get('files'):
                data_manager.update_group_analysis(group_name)
                features_df = group_info.get('features_df')
                if features_df is not None and not features_df.empty:
                    temp_df = features_df.copy()
                    temp_df['group'] = group_name
                    all_group_data.append(temp_df)
    
    if not all_group_data:
        raise ValueError("No processed data available for the selected groups")
    
    combined_df = pd.concat(all_group_data, ignore_index=True)
    
    # Create PDF document
    doc = SimpleDocTemplate(output_filename, pagesize=A4, 
                           rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    
    # Build story
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=12,
        spaceBefore=20
    )
    
    # Title page
    story.append(Paragraph("FRAP Analysis Comprehensive Report", title_style))
    story.append(Spacer(1, 20))
    
    # Report metadata
    metadata_data = [
        ['Report Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        ['Groups Analyzed:', ', '.join(groups_to_compare)],
        ['Total Files:', str(len(combined_df))],
        ['Analysis Software:', 'FRAP Analysis Platform v1.0']
    ]
    
    metadata_table = Table(metadata_data, colWidths=[2*inch, 3*inch])
    metadata_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('BACKGROUND', (0, 0), (0, -1), colors.grey),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.whitesmoke),
    ]))
    
    story.append(metadata_table)
    story.append(Spacer(1, 30))
    
    # Executive summary
    story.append(Paragraph("Executive Summary", heading_style))
    
    summary_text = f"""
    This report presents a comprehensive analysis of FRAP (Fluorescence Recovery After Photobleaching) 
    data comparing {len(groups_to_compare)} experimental groups with a total of {len(combined_df)} 
    individual measurements. The analysis includes dual-interpretation kinetics (diffusion and binding), 
    statistical comparisons, and quality assessment metrics.
    """
    
    story.append(Paragraph(summary_text, styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Group overview table
    story.append(Paragraph("Group Overview", heading_style))
    
    group_overview_data = [['Group', 'N', 'Mobile (%)', 'Rate (k)', 'Half-time (s)', 'R² (avg)']]
    
    for group in groups_to_compare:
        group_data = combined_df[combined_df['group'] == group]
        mobile_mean = group_data['mobile_fraction'].mean()
        rate_mean = group_data.get('rate_constant_fast', group_data.get('rate_constant', pd.Series([np.nan]))).mean()
        half_time_mean = group_data.get('half_time_fast', group_data.get('half_time', pd.Series([np.nan]))).mean()
        r2_mean = group_data.get('r2', pd.Series([np.nan])).mean()
        
        group_overview_data.append([
            group,
            str(len(group_data)),
            f"{mobile_mean:.1f}" if not np.isnan(mobile_mean) else "N/A",
            f"{rate_mean:.4f}" if not np.isnan(rate_mean) else "N/A",
            f"{half_time_mean:.1f}" if not np.isnan(half_time_mean) else "N/A",
            f"{r2_mean:.3f}" if not np.isnan(r2_mean) else "N/A"
        ])
    
    group_table = Table(group_overview_data, colWidths=[1.2*inch, 0.6*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch])
    group_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(group_table)
    story.append(PageBreak())
    
    # Statistical analysis for key parameters
    key_parameters = ['mobile_fraction', 'rate_constant_fast', 'half_time_fast']
    available_params = [p for p in key_parameters if p in combined_df.columns]
    
    for param in available_params:
        story.append(Paragraph(f"Statistical Analysis: {param.replace('_', ' ').title()}", heading_style))
        
        # Perform statistical analysis
        stats_results = perform_comprehensive_statistics(combined_df, param)
        
        # Create visualization
        plot_buffer = create_statistical_plots(combined_df, param)
        img = Image(plot_buffer, width=6*inch, height=5*inch)
        story.append(img)
        story.append(Spacer(1, 10))
        
        # Statistical results text
        if 'two_group' in stats_results['statistical_tests']:
            test_result = stats_results['statistical_tests']['two_group']
            effect_size = stats_results['effect_sizes']['cohens_d']
            
            stats_text = f"""
            Two-group comparison using {test_result['test']}:
            • Test statistic: {test_result['statistic']:.4f}
            • P-value: {test_result['p_value']:.6f}
            • Significance: {'Significant' if test_result['significant'] else 'Not significant'} (α = 0.05)
            • Effect size (Cohen's d): {effect_size['value']:.3f} ({effect_size['magnitude']})
            """
            
        elif 'anova' in stats_results['statistical_tests']:
            anova_result = stats_results['statistical_tests']['anova']
            
            stats_text = f"""
            Multi-group comparison using {anova_result['test']}:
            • F-statistic: {anova_result['f_statistic']:.4f}
            • P-value: {anova_result['p_value']:.6f}
            • Significance: {'Significant' if anova_result['significant'] else 'Not significant'} (α = 0.05)
            """
            
            if 'pairwise' in stats_results['statistical_tests']:
                stats_text += "\n\nPost-hoc pairwise comparisons (Bonferroni corrected):"
                for pair in stats_results['statistical_tests']['pairwise']:
                    sig_text = "***" if pair['p_corrected'] < 0.001 else "**" if pair['p_corrected'] < 0.01 else "*" if pair['p_corrected'] < 0.05 else "ns"
                    stats_text += f"\n• {pair['group1']} vs {pair['group2']}: p = {pair['p_corrected']:.4f} {sig_text}"
        
        story.append(Paragraph(stats_text, styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Group statistics table
        group_stats_data = [['Group', 'N', 'Mean ± SEM', 'Median', 'Q25-Q75']]
        
        for group in stats_results['groups']:
            stats = stats_results['group_stats'][group]
            group_stats_data.append([
                group,
                str(stats['n']),
                f"{stats['mean']:.3f} ± {stats['sem']:.3f}",
                f"{stats['median']:.3f}",
                f"{stats['q25']:.3f} - {stats['q75']:.3f}"
            ])
        
        stats_table = Table(group_stats_data, colWidths=[1.5*inch, 0.6*inch, 1.2*inch, 1*inch, 1.2*inch])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(stats_table)
        story.append(PageBreak())
    
    # Detailed results section
    story.append(Paragraph("Detailed Results by Group", heading_style))
    
    for group_name in groups_to_compare:
        group_data = combined_df[combined_df['group'] == group_name]
        
        story.append(Paragraph(f"Group: {group_name}", styles['Heading3']))
        
        # Individual file results
        detailed_data = [['File', 'Mobile (%)', 'Rate (k)', 'Half-time (s)', 'Model', 'R²']]
        
        for _, row in group_data.iterrows():
            file_path = row.get('file_path', '')
            file_name = data_manager.files.get(file_path, {}).get('name', 'Unknown')[:20]  # Truncate long names
            
            detailed_data.append([
                file_name,
                f"{row.get('mobile_fraction', np.nan):.1f}" if not pd.isna(row.get('mobile_fraction')) else "N/A",
                f"{row.get('rate_constant_fast', row.get('rate_constant', np.nan)):.4f}" if not pd.isna(row.get('rate_constant_fast', row.get('rate_constant'))) else "N/A",
                f"{row.get('half_time_fast', row.get('half_time', np.nan)):.1f}" if not pd.isna(row.get('half_time_fast', row.get('half_time'))) else "N/A",
                row.get('model', 'Unknown'),
                f"{row.get('r2', np.nan):.3f}" if not pd.isna(row.get('r2')) else "N/A"
            ])
        
        detailed_table = Table(detailed_data, colWidths=[1.5*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.6*inch])
        detailed_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
        ]))
        
        story.append(detailed_table)
        story.append(Spacer(1, 15))
    
    # Build PDF
    doc.build(story)
    
    return output_filename

# Import centralized kinetics interpretation from core module
from frap_core_corrected.py import FRAPAnalysisCore

def interpret_kinetics(rate_constant, bleach_radius_um=1.0, gfp_d=25.0, gfp_mw=27.0):
    """Use centralized kinetics interpretation function for consistency"""
    result = FRAPAnalysisCore.interpret_kinetics(rate_constant, bleach_radius_um, gfp_d, 2.82, gfp_mw)
    return {
        'diffusion_coefficient': result['diffusion_coefficient'],
        'k_off': result['k_off'],
        'apparent_mw': result['apparent_mw']
    }