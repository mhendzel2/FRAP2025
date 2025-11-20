import os
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO

class FRAPReportGenerator:
    """
    Generates HTML reports for FRAP analysis.
    """
    
    @staticmethod
    def figure_to_base64(fig: plt.Figure) -> str:
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        data = base64.b64encode(buf.read()).decode('utf-8')
        return f"data:image/png;base64,{data}"

    @staticmethod
    def generate_html_report(
        features: pd.DataFrame,
        figures: dict,
        output_path: str,
        title: str = "FRAP Analysis Report"
    ):
        """
        Generates a comprehensive HTML report.
        
        Args:
            features: DataFrame containing analysis results.
            figures: Dictionary of {name: plt.Figure}.
            output_path: Path to save the HTML file.
        """
        
        html_content = [
            f"<html><head><title>{title}</title>",
            "<style>",
            "body { font-family: sans-serif; margin: 20px; }",
            "h1, h2 { color: #333; }",
            "table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }",
            "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "th { background-color: #f2f2f2; }",
            ".figure { margin-bottom: 30px; text-align: center; }",
            ".figure img { max-width: 80%; border: 1px solid #ccc; }",
            "</style>",
            "</head><body>",
            f"<h1>{title}</h1>",
            "<h2>Analysis Summary</h2>",
            features.describe().to_html(classes='table'),
            "<h2>Detailed Results</h2>",
            features.head(10).to_html(classes='table'), # Show first 10 rows
            "<p><em>(First 10 rows shown)</em></p>",
            "<h2>Visualizations</h2>"
        ]
        
        for name, fig in figures.items():
            img_src = FRAPReportGenerator.figure_to_base64(fig)
            html_content.append(f"<div class='figure'><h3>{name}</h3><img src='{img_src}'></div>")
            
        html_content.append("</body></html>")
        
        with open(output_path, 'w') as f:
            f.write("\n".join(html_content))
            
        print(f"Report saved to {output_path}")
