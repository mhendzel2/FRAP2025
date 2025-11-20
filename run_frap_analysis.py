import numpy as np
import pandas as pd
import os
import logging
from frap_input_handler import FRAPInputHandler
from frap_analysis_enhanced import FRAPGroupAnalyzer, FRAPStatisticalComparator
from frap_visualizer import FRAPVisualizer
from frap_report_generator import FRAPReportGenerator
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_dummy_data(filename="dummy_frap_data.csv"):
    """Creates a dummy CSV file for testing."""
    t = np.linspace(0, 100, 100)
    # Simulate recovery: F(t) = F_inf - (F_inf - F_0) * exp(-k*t)
    # Bleach at t=10 (index 10)
    bleach_idx = 10
    t_bleach = t[bleach_idx]
    
    # Pre-bleach
    roi = np.ones(len(t)) * 1000
    ref = np.ones(len(t)) * 2000
    bg = np.ones(len(t)) * 100
    
    # Bleach event
    roi[bleach_idx] = 200 # Deep bleach
    
    # Recovery
    k = 0.1
    recovery = 800 - (800 - 200) * np.exp(-k * (t[bleach_idx+1:] - t_bleach))
    roi[bleach_idx+1:] = recovery
    
    # Add noise
    roi += np.random.normal(0, 10, len(t))
    ref += np.random.normal(0, 20, len(t)) # Some bleaching in ref too maybe?
    # Let's simulate ref bleaching
    ref = ref * np.exp(-0.001 * t)
    
    df = pd.DataFrame({
        'Time': t,
        'ROI': roi,
        'Reference': ref,
        'Background': bg
    })
    
    df.to_csv(filename, index=False)
    logger.info(f"Created dummy data: {filename}")
    return filename

def main():
    # 1. Setup
    data_file = "dummy_frap_data.csv"
    if not os.path.exists(data_file):
        create_dummy_data(data_file)
        
    # 2. Load Data
    logger.info("Loading data...")
    curve_data = FRAPInputHandler.load_csv(data_file)
    
    # 3. Preprocessing
    logger.info("Preprocessing...")
    # Assume bleach is at index 10 (known from creation, but in real app we detect it)
    # Simple detection: min intensity
    bleach_idx = int(np.argmin(curve_data.roi_intensity))
    
    curve_data = FRAPInputHandler.double_normalization(curve_data, bleach_idx)
    curve_data = FRAPInputHandler.time_zero_correction(curve_data, bleach_idx)
    
    # 4. Analysis
    logger.info("Analyzing...")
    analyzer = FRAPGroupAnalyzer()
    analyzer.add_curve(curve_data)
    
    # Add a second curve (copy with noise) to test group analysis
    curve_data2 = FRAPInputHandler.load_csv(data_file)
    curve_data2.roi_intensity += np.random.normal(0, 50, len(curve_data2.roi_intensity))
    curve_data2 = FRAPInputHandler.double_normalization(curve_data2, bleach_idx)
    curve_data2 = FRAPInputHandler.time_zero_correction(curve_data2, bleach_idx)
    analyzer.add_curve(curve_data2)
    
    analyzer.analyze_group(model_name='single_exp')
    analyzer.detect_subpopulations()
    analyzer.detect_outliers()
    
    print("Analysis Results:")
    print(analyzer.features)
    
    # 5. Visualization
    logger.info("Generating plots...")
    
    figures = {}
    
    # Recovery Curve
    # Ensure data is available
    if curve_data.time_post_bleach is not None:
        times = curve_data.time_post_bleach
        # Note: intensities here should be the DATA, not the fit.
        # Filter out None values
        data_intensities = [c.intensity_post_bleach for c in analyzer.curves if c.intensity_post_bleach is not None]
        fitted_curves = [res.fitted_curve for res in analyzer.fit_results if res.success and res.fitted_curve is not None]
        
        if data_intensities:
            fig1 = FRAPVisualizer.plot_recovery_curves(
                times, 
                data_intensities, 
                fitted_curves=fitted_curves,
                title="Group Recovery"
            )
            figures['Recovery Curves'] = fig1
    
    # Parameter Distribution
    if not analyzer.features.empty:
        fig2 = FRAPVisualizer.plot_parameter_distribution(analyzer.features, 'k_off', plot_type='violin')
        figures['k_off Distribution'] = fig2
        
        fig3 = FRAPVisualizer.plot_subpopulations(analyzer.features, 'k_off', 'F_inf')
        figures['Subpopulations'] = fig3
        
    # 6. Report Generation
    logger.info("Generating report...")
    FRAPReportGenerator.generate_html_report(
        analyzer.features,
        figures,
        "FRAP_Analysis_Report.html"
    )
        
    logger.info("Done. Check FRAP_Analysis_Report.html")

if __name__ == "__main__":
    main()
