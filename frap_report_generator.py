import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from io import BytesIO
from datetime import datetime
from typing import Dict, List, Optional, Any

# Import local analysis modules
from frap_group_comparison import HolisticGroupComparator, compare_recovery_profiles
from frap_populations import detect_outliers_and_clusters, compute_cluster_statistics
from frap_reference_database import FRAPReferenceDatabase
from calibration import Calibration

class EnhancedFRAPReportGenerator:
    """
    Generates an interpretive HTML report for FRAP analysis.
    Integrates holistic comparison, population analysis, reference database, and calibration.
    """

    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Initialize helper classes
        self.holistic_comparator = HolisticGroupComparator()
        self.reference_db = FRAPReferenceDatabase()

    @staticmethod
    def _figure_to_html(fig) -> str:
        """Converts Plotly or Matplotlib figure to HTML string."""
        # Check for Plotly figure (go.Figure)
        if isinstance(fig, go.Figure): 
            return fig.to_html(full_html=False, include_plotlyjs='cdn')
        else: # Matplotlib
            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
            buf.seek(0)
            data = base64.b64encode(buf.read()).decode('utf-8')
            return f'<img src="data:image/png;base64,{data}" style="max-width:100%;">'

    def generate_report(self, 
                        data_groups: Dict[str, Any], 
                        selected_groups: List[str], 
                        calibration_data: Optional[Dict] = None,
                        filename: str = "FRAP_Interpretive_Report.html"):
        """
        Main entry point to generate the comprehensive report.
        """
        
        # 1. Collect Data
        group_features = {}
        combined_df = pd.DataFrame()
        
        for group in selected_groups:
            if group in data_groups:
                analyzer = data_groups[group]
                # Handle both dictionary and object access for flexibility
                if isinstance(analyzer, dict):
                    df = analyzer.get('features_df', pd.DataFrame()).copy()
                elif hasattr(analyzer, 'features'):
                    df = analyzer.features.copy() if analyzer.features is not None else pd.DataFrame()
                else:
                    continue
                
                if not df.empty:
                    df['group'] = group
                    group_features[group] = df
                    combined_df = pd.concat([combined_df, df], ignore_index=True)

        if combined_df.empty:
            print("No data found for selected groups.")
            return None

        # 2. Start HTML Construction
        html_content = [
            self._generate_header(),
            f"<h1>FRAP Comparative Analysis Report</h1>",
            f"<p class='timestamp'>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>",
            f"<p><b>Comparing Groups:</b> {', '.join(selected_groups)}</p>"
        ]

        # 3. Executive Biological Interpretation (The "Story")
        html_content.append("<h2>1. Biological Interpretation</h2>")
        if len(selected_groups) >= 2:
            # Compare the first two groups as the primary comparison (usually Control vs Treatment)
            primary = selected_groups[0]
            secondary = selected_groups[1]
            
            if primary in group_features and secondary in group_features:
                stats_res = self.holistic_comparator.statistical_comparison(
                    group_features[primary], group_features[secondary],
                    group1_name=primary, group2_name=secondary
                )
                
                interpretation_text = self.holistic_comparator.interpret_differences(stats_res)
                # Format markdown to HTML
                interpretation_html = interpretation_text.replace('\n', '<br>').replace('**', '<b>').replace('__', '</b>')
                
                html_content.append(f"<div class='interpretation-box'>{interpretation_html}</div>")
            else:
                html_content.append("<p><i>Insufficient data for comparison.</i></p>")
        else:
            html_content.append("<p><i>Select at least two groups for automatic biological comparison.</i></p>")

        # 4. Population & Heterogeneity Analysis
        html_content.append("<h2>2. Population Heterogeneity Analysis</h2>")
        html_content.append("<p>Are the cells in each group behaving uniformly, or are there subpopulations?</p>")
        
        # Run clustering on combined data to see if groups map to specific clusters
        # Ensure required columns exist
        required_cols = ['mobile_fraction', 't_half']
        available_cols = [c for c in required_cols if c in combined_df.columns]
        
        if len(available_cols) == len(required_cols):
            try:
                clustered_df = detect_outliers_and_clusters(combined_df, feature_cols=required_cols)
                
                # Plot clusters
                fig_clusters = px.scatter(
                    clustered_df, x='t_half', y='mobile_fraction', color='group', symbol='cluster',
                    title="Population Distribution: Kinetics vs Mobility",
                    hover_data=['filename'] if 'filename' in clustered_df.columns else None,
                    labels={'t_half': 'Half Time (s)', 'mobile_fraction': 'Mobile Fraction'},
                    log_x=True
                )
                html_content.append(self._figure_to_html(fig_clusters))
                
                # Check for heterogeneity
                if 'cluster' in clustered_df.columns:
                    n_clusters = clustered_df['cluster'].nunique()
                    if n_clusters > 1:
                        html_content.append(f"<p class='warning'><b>Note:</b> {n_clusters} distinct kinetic subpopulations were detected. " 
                                            "Treatment effects may be specific to one subpopulation.</p>")
            except Exception as e:
                html_content.append(f"<p class='warning'>Could not perform clustering analysis: {e}</p>")
        else:
            html_content.append(f"<p>Missing columns for clustering: {set(required_cols) - set(available_cols)}</p>")

        # 5. Reference Database Context
        html_content.append("<h2>3. Reference Database Benchmarking</h2>")
        avg_deff = combined_df['diffusion_coeff'].mean() if 'diffusion_coeff' in combined_df.columns else None
        avg_mf = combined_df['mobile_fraction'].mean() if 'mobile_fraction' in combined_df.columns else None
        
        if avg_deff is not None or avg_mf is not None:
            ref_analysis = self.reference_db.compare_experimental_to_reference(
                experimental_deff=avg_deff,
                experimental_mf=avg_mf
            )
            
            html_content.append(f"<div class='info-box'><b>Database Context:</b> {ref_analysis['interpretation']}</div>")
            
            if ref_analysis['closest_matches'] is not None and not ref_analysis['closest_matches'].empty:
                html_content.append("<h3>Proteins with Similar Kinetics</h3>")
                # Select specific columns and convert to HTML
                cols_to_show = ['protein_probe', 'class_type', 'deff_um2_s', 'mobile_fraction_pct', 'key_finding']
                available_cols = [c for c in cols_to_show if c in ref_analysis['closest_matches'].columns]
                matches_table = ref_analysis['closest_matches'][available_cols].to_html(classes='table table-striped', index=False)
                html_content.append(matches_table)
        else:
            html_content.append("<p>Insufficient data for reference database comparison.</p>")

        # 6. Internal Standard Calibration (Optional)
        if calibration_data:
            html_content.append("<h2>4. Molecular Size Calibration</h2>")
            try:
                calib = Calibration(calibration_data.get('standards'))
                
                # Estimate MW for each group
                calib_results = []
                for group in selected_groups:
                    if group in group_features and 'diffusion_coeff' in group_features[group].columns:
                        g_deff = group_features[group]['diffusion_coeff'].median()
                        mw, low, high = calib.estimate_apparent_mw(g_deff)
                        calib_results.append({
                            'Group': group,
                            'Median D_eff': f"{g_deff:.2f}",
                            'Est. MW (kDa)': f"{mw:.1f}",
                            '95% CI (kDa)': f"{low:.1f} - {high:.1f}"
                        })
                
                if calib_results:
                    html_content.append(pd.DataFrame(calib_results).to_html(classes='table', index=False))
                    html_content.append("<p><i>Estimates based on Stokes-Einstein relation derived from internal standards.</i></p>")
            except Exception as e:
                html_content.append(f"<p class='warning'>Calibration error: {e}</p>")

        # 7. Detailed Statistics Table
        html_content.append("<h2>5. Detailed Kinetic Statistics</h2>")
        try:
            comp_df = self.holistic_comparator.compare_groups(group_features)
            html_content.append(comp_df.to_html(classes='table', float_format=lambda x: '{:.3f}'.format(x) if pd.notnull(x) else 'N/A'))
        except Exception as e:
            html_content.append(f"<p>Could not generate statistics table: {e}</p>")

        # Close HTML
        html_content.append(self._generate_footer())
        
        # Write File
        full_path = os.path.join(self.output_dir, filename)
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(html_content))
        
        return full_path

    def _generate_header(self):
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>FRAP Interpretive Report</title>
            <style>
                body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; color: #333; background-color: #f9f9f9; }
                h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
                h2 { color: #2980b9; margin-top: 30px; }
                table { border-collapse: collapse; width: 100%; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.2); }
                th { background-color: #34495e; color: white; padding: 12px; text-align: left; }
                td { border: 1px solid #ddd; padding: 10px; }
                tr:nth-child(even) { background-color: #f2f2f2; }
                .interpretation-box { background-color: #e8f6f3; border-left: 5px solid #1abc9c; padding: 15px; margin: 20px 0; }
                .info-box { background-color: #eaf2f8; border-left: 5px solid #3498db; padding: 15px; margin: 20px 0; }
                .warning { background-color: #fdf2e9; border-left: 5px solid #e67e22; padding: 15px; color: #d35400; }
                .timestamp { color: #7f8c8d; font-size: 0.9em; }
            </style>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
        """

    def _generate_footer(self):
        return """
        <div style="margin-top: 50px; border-top: 1px solid #ccc; padding-top: 10px; text-align: center; color: #777;">
            <p>Generated by FRAP Analysis Platform 2025</p>
        </div>
        </body>
        </html>
        """
