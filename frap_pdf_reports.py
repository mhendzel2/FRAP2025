"""
FRAP PDF Reports Module
Generate comprehensive PDF reports for FRAP analysis results
"""

import os
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import io
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
import logging

logger = logging.getLogger(__name__)

def generate_pdf_report(data_manager, groups_to_compare, output_filename=None, settings=None):
    """
    Generates a comprehensive PDF report for FRAP analysis results.
    
    Parameters:
    -----------
    data_manager : FRAPDataManager
        The data manager containing all FRAP data
    groups_to_compare : list
        List of group names to include in the report
    output_filename : str, optional
        Filename for the output PDF, defaults to a timestamped name
    settings : dict, optional
        Analysis settings to include in the report
        
    Returns:
    --------
    str
        Path to the generated PDF file
    """
    try:
        # Create a temporary file if no output filename is provided
        if output_filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_filename = f"FRAP_Report_{timestamp}.pdf"
            
        # Get the absolute path to the output file
        output_path = os.path.abspath(output_filename)
        
        # Create a PDF document
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Define styles
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(
            name='Title',
            parent=styles['Heading1'],
            fontSize=16,
            alignment=TA_CENTER
        ))
        styles.add(ParagraphStyle(
            name='Subtitle',
            parent=styles['Heading2'],
            fontSize=14
        ))
        styles.add(ParagraphStyle(
            name='Section',
            parent=styles['Heading3'],
            fontSize=12
        ))
        styles.add(ParagraphStyle(
            name='Normal',
            parent=styles['Normal'],
            fontSize=10
        ))
        
        # Initialize the elements list
        elements = []
        
        # Title
        elements.append(Paragraph("FRAP Analysis Report", styles['Title']))
        elements.append(Spacer(1, 0.25*inch))
        
        # Report metadata
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        elements.append(Paragraph(f"Generated: {timestamp}", styles['Normal']))
        elements.append(Paragraph(f"Groups included: {', '.join(groups_to_compare)}", styles['Normal']))
        elements.append(Spacer(1, 0.25*inch))
        
        # Settings information if provided
        if settings:
            elements.append(Paragraph("Analysis Settings", styles['Subtitle']))
            
            # Create a table for settings
            settings_data = [["Parameter", "Value"]]
            important_settings = [
                ('default_criterion', 'Model Selection Criterion'), 
                ('default_bleach_radius', 'Bleach Radius (pixels)'),
                ('default_pixel_size', 'Pixel Size (µm/pixel)'),
                ('default_gfp_diffusion', 'Reference GFP Diffusion (µm²/s)')
            ]
            
            for key, label in important_settings:
                if key in settings:
                    value = settings[key]
                    if isinstance(value, float):
                        value_str = f"{value:.3f}"
                    else:
                        value_str = str(value)
                    settings_data.append([label, value_str])
            
            # Add effective bleach radius (calculated)
            bleach_radius = settings.get('default_bleach_radius', 1.0)
            pixel_size = settings.get('default_pixel_size', 0.3)
            effective_radius = bleach_radius * pixel_size
            settings_data.append(["Effective Bleach Radius (µm)", f"{effective_radius:.3f}"])
            
            # Create the table
            settings_table = Table(settings_data, colWidths=[3*inch, 2*inch])
            settings_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (1, 0), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (1, 0), colors.black),
                ('ALIGN', (0, 0), (1, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey)
            ]))
            
            elements.append(settings_table)
            elements.append(Spacer(1, 0.25*inch))
        
        # Group Summary Section
        elements.append(Paragraph("Group Summaries", styles['Subtitle']))
        elements.append(Spacer(1, 0.15*inch))
        
        # Group comparison tables
        for group_name in groups_to_compare:
            if group_name not in data_manager.groups:
                continue
                
            group = data_manager.groups[group_name]
            
            # Skip empty groups
            if not group.get('files') or not group.get('features_df') is not None:
                continue
            
            # Group header
            elements.append(Paragraph(f"Group: {group_name}", styles['Section']))
            
            # Files count
            total_files = len(group.get('files', []))
            analyzed_files = len(group.get('features_df', pd.DataFrame())) if group.get('features_df') is not None else 0
            elements.append(Paragraph(f"Total Files: {total_files}, Analyzed: {analyzed_files}", styles['Normal']))
            elements.append(Spacer(1, 0.15*inch))
            
            # Group statistics
            if group.get('features_df') is not None and not group['features_df'].empty:
                df = group['features_df']
                
                # Basic statistics table
                key_metrics = ['mobile_fraction', 'immobile_fraction', 'rate_constant', 'half_time']
                available_metrics = [m for m in key_metrics if m in df.columns]
                
                if available_metrics:
                    # Calculate statistics
                    stats_data = [["Metric", "Mean", "Std. Dev.", "Median", "Min", "Max"]]
                    
                    for metric in available_metrics:
                        values = df[metric].dropna()
                        if not values.empty:
                            mean_val = values.mean()
                            std_val = values.std()
                            median_val = values.median()
                            min_val = values.min()
                            max_val = values.max()
                            
                            metric_name = metric.replace('_', ' ').title()
                            stats_data.append([
                                metric_name, 
                                f"{mean_val:.3f}", 
                                f"{std_val:.3f}", 
                                f"{median_val:.3f}", 
                                f"{min_val:.3f}", 
                                f"{max_val:.3f}"
                            ])
                    
                    # Create the table
                    stats_table = Table(stats_data, colWidths=[1.2*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch])
                    stats_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey)
                    ]))
                    
                    elements.append(stats_table)
                    elements.append(Spacer(1, 0.2*inch))
                
                # Check if we have component-specific data
                component_cols = [col for col in df.columns if 'proportion' in col and ('fast' in col or 'medium' in col or 'slow' in col)]
                
                if component_cols:
                    # Create component statistics
                    component_data = [["Component", "Proportion (%)", "Rate (k)", "Half-time (s)"]]
                    
                    for comp in ['fast', 'medium', 'slow']:
                        prop_col = f'proportion_of_mobile_{comp}'
                        rate_col = f'rate_constant_{comp}'
                        half_col = f'half_time_{comp}'
                        
                        if prop_col in df.columns and rate_col in df.columns:
                            prop_vals = df[prop_col].dropna()
                            rate_vals = df[rate_col].dropna()
                            half_vals = df[half_col].dropna() if half_col in df.columns else pd.Series()
                            
                            if not prop_vals.empty and not rate_vals.empty:
                                mean_prop = prop_vals.mean()
                                mean_rate = rate_vals.mean()
                                mean_half = half_vals.mean() if not half_vals.empty else np.nan
                                
                                component_data.append([
                                    comp.capitalize(),
                                    f"{mean_prop:.1f}%",
                                    f"{mean_rate:.4f}",
                                    f"{mean_half:.2f}" if not np.isnan(mean_half) else "N/A"
                                ])
                    
                    if len(component_data) > 1:  # Only add if we have component data
                        elements.append(Paragraph("Component Analysis", styles['Normal']))
                        
                        # Create the table
                        comp_table = Table(component_data, colWidths=[1.2*inch, 1.2*inch, 1.2*inch, 1.2*inch])
                        comp_table.setStyle(TableStyle([
                            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                            ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey)
                        ]))
                        
                        elements.append(comp_table)
            
            elements.append(Spacer(1, 0.25*inch))
        
        # Statistical comparison section (when multiple groups)
        if len(groups_to_compare) > 1:
            elements.append(Paragraph("Statistical Comparison", styles['Subtitle']))
            elements.append(Spacer(1, 0.15*inch))
            
            # Collect data from all groups
            all_group_data = []
            for group_name in groups_to_compare:
                if group_name in data_manager.groups:
                    group = data_manager.groups[group_name]
                    if group.get('features_df') is not None and not group['features_df'].empty:
                        temp_df = group['features_df'].copy()
                        temp_df['group'] = group_name
                        all_group_data.append(temp_df)
            
            # Perform statistical comparison if we have data
            if all_group_data:
                combined_df = pd.concat(all_group_data, ignore_index=True)
                
                # Compare key metrics
                key_metrics = ['mobile_fraction', 'rate_constant', 'half_time']
                available_metrics = [m for m in key_metrics if m in combined_df.columns]
                
                for metric in available_metrics:
                    # Get group summaries
                    group_stats = combined_df.groupby('group')[metric].agg(['count', 'mean', 'std']).reset_index()
                    
                    # Format the data for table
                    stat_data = [["Group", "N", "Mean", "Std. Dev."]]
                    for _, row in group_stats.iterrows():
                        stat_data.append([
                            row['group'],
                            str(int(row['count'])),
                            f"{row['mean']:.3f}",
                            f"{row['std']:.3f}"
                        ])
                    
                    elements.append(Paragraph(f"{metric.replace('_', ' ').title()} Comparison", styles['Normal']))
                    
                    # Create the table
                    stat_table = Table(stat_data, colWidths=[1.5*inch, 0.8*inch, 1.2*inch, 1.2*inch])
                    stat_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey)
                    ]))
                    
                    elements.append(stat_table)
                    elements.append(Spacer(1, 0.2*inch))

                    # Calculate ANOVA or t-test results
                    # Add statistical test results here
                    if len(groups_to_compare) == 2:
                        # For two groups, use t-test
                        from scipy import stats
                        group1_data = combined_df[combined_df['group'] == groups_to_compare[0]][metric].dropna()
                        group2_data = combined_df[combined_df['group'] == groups_to_compare[1]][metric].dropna()
                        
                        if len(group1_data) > 1 and len(group2_data) > 1:
                            # Perform t-test
                            t_stat, p_value = stats.ttest_ind(group1_data, group2_data)
                            
                            elements.append(Paragraph(
                                f"t-test result: t={t_stat:.3f}, p={p_value:.4f} " + 
                                ("(significant)" if p_value < 0.05 else "(not significant)"),
                                styles['Normal']
                            ))
                    else:
                        # For more than two groups, use ANOVA
                        from scipy import stats
                        
                        groups_data = []
                        for group_name in groups_to_compare:
                            group_data = combined_df[combined_df['group'] == group_name][metric].dropna()
                            if len(group_data) > 1:
                                groups_data.append(group_data)
                        
                        if len(groups_data) >= 2:
                            # Perform ANOVA
                            f_stat, p_value = stats.f_oneway(*groups_data)
                            
                            elements.append(Paragraph(
                                f"ANOVA result: F={f_stat:.3f}, p={p_value:.4f} " + 
                                ("(significant)" if p_value < 0.05 else "(not significant)"),
                                styles['Normal']
                            ))
                    
                    elements.append(Spacer(1, 0.25*inch))
        
        # Generate plots for visual representation
        # This would typically involve generating plots using matplotlib or plotly,
        # saving them to a BytesIO object, and then adding them to the PDF
        
        # For example, let's generate a basic plot comparing mobile fractions
        if len(groups_to_compare) > 1 and all_group_data:
            try:
                # Create a mobile population comparison plot
                plt.figure(figsize=(7, 5))
                
                # Use boxplot for comparison
                group_data = []
                group_labels = []
                
                for group_name in groups_to_compare:
                    if group_name in data_manager.groups:
                        group = data_manager.groups[group_name]
                        if group.get('features_df') is not None and not group['features_df'].empty:
                            if 'mobile_fraction' in group['features_df'].columns:
                                mobile_values = group['features_df']['mobile_fraction'].dropna()
                                if not mobile_values.empty:
                                    group_data.append(mobile_values)
                                    group_labels.append(group_name)
                
                if group_data:
                    plt.boxplot(group_data, labels=group_labels)
                    plt.title('Mobile Population Comparison')
                    plt.ylabel('Mobile Fraction (%)')
                    plt.grid(True, linestyle='--', alpha=0.7)
                    
                    # Save the plot to a BytesIO object
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', dpi=150)
                    buf.seek(0)
                    
                    # Add the image to the PDF
                    elements.append(Paragraph("Mobile Population Comparison", styles['Section']))
                    elements.append(Image(buf, width=6*inch, height=4*inch))
                    elements.append(Spacer(1, 0.2*inch))
                    
                plt.close()
                
                # Create a half-time comparison plot if available
                if 'half_time' in combined_df.columns or 'half_time_fast' in combined_df.columns:
                    plt.figure(figsize=(7, 5))
                    
                    half_time_col = 'half_time_fast' if 'half_time_fast' in combined_df.columns else 'half_time'
                    
                    group_data = []
                    group_labels = []
                    
                    for group_name in groups_to_compare:
                        if group_name in data_manager.groups:
                            group = data_manager.groups[group_name]
                            if group.get('features_df') is not None and not group['features_df'].empty:
                                if half_time_col in group['features_df'].columns:
                                    half_time_values = group['features_df'][half_time_col].dropna()
                                    if not half_time_values.empty:
                                        group_data.append(half_time_values)
                                        group_labels.append(group_name)
                    
                    if group_data:
                        plt.boxplot(group_data, labels=group_labels)
                        plt.title('Half-time Comparison')
                        plt.ylabel('Half-time (s)')
                        plt.grid(True, linestyle='--', alpha=0.7)
                        
                        # Save the plot to a BytesIO object
                        buf = io.BytesIO()
                        plt.savefig(buf, format='png', dpi=150)
                        buf.seek(0)
                        
                        # Add the image to the PDF
                        elements.append(Paragraph("Half-time Comparison", styles['Section']))
                        elements.append(Image(buf, width=6*inch, height=4*inch))
                        elements.append(Spacer(1, 0.2*inch))
                        
                    plt.close()
            except Exception as e:
                logger.error(f"Error generating plots: {e}")
                elements.append(Paragraph(f"Error generating plots: {str(e)}", styles['Normal']))
        
        # Build the PDF
        doc.build(elements)
        
        logger.info(f"PDF report successfully generated: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error generating PDF report: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def add_matplotlib_plot_to_pdf(elements, plot_function, *args, **kwargs):
    """Helper function to add matplotlib plots to PDF"""
    try:
        # Generate the plot
        fig = plot_function(*args, **kwargs)
        
        # Save the plot to a BytesIO object
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        
        # Add the image to the PDF
        img = Image(buf, width=6*inch, height=4*inch)
        elements.append(img)
        
        # Close the figure to free memory
        plt.close(fig)
        
        return True
    except Exception as e:
        logger.error(f"Error adding plot to PDF: {e}")
        return False

if __name__ == "__main__":
    # Test function if run directly
    print("This module is not intended to be run directly.")
    print("Import and use the generate_pdf_report function in your application.")
