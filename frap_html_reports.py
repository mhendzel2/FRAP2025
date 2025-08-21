import os
import pandas as pd
from datetime import datetime

def generate_html_report(data_manager, groups_to_compare, output_filename=None, settings=None):
    """
    Generates a comprehensive HTML report for FRAP analysis results.
    """
    if not groups_to_compare:
        return None

    # Combine data from selected groups
    all_group_data = []
    for group_name in groups_to_compare:
        if group_name in data_manager.groups:
            group = data_manager.groups[group_name]
            if group.get('features_df') is not None and not group['features_df'].empty:
                temp_df = group['features_df'].copy()
                temp_df['group'] = group_name
                all_group_data.append(temp_df)

    if not all_group_data:
        return None

    combined_df = pd.concat(all_group_data, ignore_index=True)

    # --- HTML Content Generation ---
    html = "<h1>FRAP Analysis Report</h1>"
    html += f"<p><b>Report Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>"
    html += f"<p><b>Groups:</b> {', '.join(groups_to_compare)}</p>"

    if settings:
        html += "<h2>Analysis Settings</h2>"
        settings_df = pd.DataFrame(list(settings.items()), columns=['Parameter', 'Value'])
        html += settings_df.to_html(index=False)

    html += "<h2>Summary Statistics</h2>"
    summary_table = combined_df.groupby('group').agg({
        'mobile_fraction': ['mean', 'std'],
        'rate_constant': ['mean', 'std'],
        'half_time': ['mean', 'std'],
    }).round(3)
    if not summary_table.empty:
        summary_table.columns = [' '.join(col).strip() for col in summary_table.columns.values]
        html += summary_table.to_html()

    html += "<h2>Detailed Results</h2>"
    html += combined_df.to_html(index=False)

    # --- HTML Template ---
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>FRAP Analysis Report</title>
        <style>
            body {{ font-family: sans-serif; margin: 2em; }}
            h1, h2, h3 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 80%; margin-bottom: 1em; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        {html}
    </body>
    </html>
    """

    if output_filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f"FRAP_Report_{timestamp}.html"

    output_path = os.path.abspath(output_filename)

    with open(output_path, "w") as f:
        f.write(html_template)

    return output_path
