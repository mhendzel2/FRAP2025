import re

def verify_ui_elements():
    """
    Reads streamlit_frap_final.py and checks for the presence of key UI elements.
    """
    try:
        with open('streamlit_frap_final.py', 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print("Error: streamlit_frap_final.py not found.")
        return

    print("Verifying UI elements in streamlit_frap_final.py...")

    # Check for Report Generation section
    report_generation_header = re.search(r'st\.header\("📄 Report Generation"\)', content)
    report_format_radio = re.search(r'st\.radio\("Select report format", \("PDF", "HTML"\)', content)

    if report_generation_header and report_format_radio:
        print("✅ Report Generation UI (PDF/HTML) found in the sidebar.")
    else:
        print("❌ Report Generation UI (PDF/HTML) NOT found.")

    # Check for the main analysis tabs
    tabs_creation = re.search(r'st\.tabs\(\["🔬 Image Analysis", "📊 Single File Analysis", "📈 Group Analysis"\]\)', content)

    if tabs_creation:
        print("✅ Main analysis tabs ('Image Analysis', 'Single File Analysis', 'Group Analysis') found.")
    else:
        print("❌ Main analysis tabs NOT found.")

    # Check for the content of the tabs
    image_analysis_tab_content = re.search(r'with tab1:\s+create_image_analysis_interface\(dm\)', content)
    if image_analysis_tab_content:
        print("✅ 'Image Analysis' tab content is correctly implemented.")
    else:
        print("❌ 'Image Analysis' tab content is missing.")

if __name__ == "__main__":
    verify_ui_elements()
