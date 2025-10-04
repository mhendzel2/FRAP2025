#!/usr/bin/env python3
"""
Quick Start Script for FRAP Single-Cell UI
Generates test data and launches the Streamlit application
"""

import sys
import subprocess
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if required packages are installed"""
    required = {
        'streamlit': 'Streamlit web framework',
        'plotly': 'Interactive plotting',
        'pandas': 'Data manipulation',
        'numpy': 'Numerical computing'
    }
    
    missing = []
    
    for package, description in required.items():
        try:
            __import__(package)
            logger.info(f"✓ {package} - {description}")
        except ImportError:
            logger.error(f"✗ {package} - MISSING")
            missing.append(package)
    
    if missing:
        logger.error(f"\nMissing packages: {', '.join(missing)}")
        logger.error("Install with: pip install streamlit plotly pandas numpy")
        return False
    
    return True


def check_analysis_modules():
    """Check if FRAP analysis modules are available"""
    required = [
        'frap_data_model',
        'frap_singlecell_api',
        'frap_visualizations',
        'frap_data_loader',
        'test_synthetic'
    ]
    
    missing = []
    
    for module in required:
        try:
            __import__(module)
            logger.info(f"✓ {module}")
        except ImportError:
            logger.error(f"✗ {module} - MISSING")
            missing.append(module)
    
    if missing:
        logger.error(f"\nMissing analysis modules: {', '.join(missing)}")
        logger.error("Ensure you're in the correct directory:")
        logger.error(f"  cd {Path(__file__).parent}")
        return False
    
    return True


def generate_test_data():
    """Generate example datasets if they don't exist"""
    output_dir = Path('./output')
    
    # Check if examples already exist
    example_dirs = [
        output_dir / 'example1',
        output_dir / 'example3'
    ]
    
    all_exist = all(
        (d / 'roi_traces.parquet').exists() and 
        (d / 'cell_features.parquet').exists()
        for d in example_dirs
    )
    
    if all_exist:
        logger.info("✓ Test data already exists")
        return True
    
    logger.info("Generating test data...")
    logger.info("This may take 1-2 minutes...")
    
    try:
        # Run quick start examples
        import quick_start_singlecell
        
        logger.info("\nRunning Example 1: Single movie analysis")
        quick_start_singlecell.example_1_single_movie()
        
        logger.info("\nRunning Example 3: Multi-condition analysis")
        quick_start_singlecell.example_3_multi_condition()
        
        logger.info("\n✓ Test data generated successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to generate test data: {e}")
        logger.exception("Details:")
        return False


def launch_ui():
    """Launch the Streamlit UI"""
    ui_file = Path(__file__).parent / 'streamlit_singlecell.py'
    
    if not ui_file.exists():
        logger.error(f"UI file not found: {ui_file}")
        return False
    
    logger.info("\n" + "="*60)
    logger.info("Launching FRAP Single-Cell UI...")
    logger.info("="*60)
    logger.info("\nThe app will open in your default browser at:")
    logger.info("  http://localhost:8501")
    logger.info("\nTo stop the app:")
    logger.info("  Press Ctrl+C in this terminal")
    logger.info("\nQuick start:")
    logger.info("  1. Click '📂 Load Data' in the left sidebar")
    logger.info("  2. Go to '📊 Example Data' tab")
    logger.info("  3. Select 'example3' and click 'Load Example'")
    logger.info("  4. Explore the tabs: Single-cell, Group, Multi-group, QC")
    logger.info("\n" + "="*60 + "\n")
    
    try:
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run',
            str(ui_file),
            '--server.headless', 'false'
        ], check=True)
        
    except KeyboardInterrupt:
        logger.info("\n\nUI stopped by user")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to launch UI: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.exception("Details:")
        return False


def print_info():
    """Print information about the UI"""
    logger.info("\n" + "="*60)
    logger.info("FRAP Single-Cell Analysis UI")
    logger.info("="*60)
    logger.info("\nFeatures:")
    logger.info("  • Interactive cohort builder with saved presets")
    logger.info("  • Single-cell inspector with recovery curves")
    logger.info("  • Group analysis with spaghetti plots")
    logger.info("  • Multi-group statistical comparisons")
    logger.info("  • QC dashboard with live filtering")
    logger.info("  • Export panel with reproducible recipes")
    logger.info("\nDocumentation:")
    logger.info("  • README_UI.md - Quick start guide")
    logger.info("  • UI_GUIDE.md - Comprehensive user guide")
    logger.info("  • UI_IMPLEMENTATION_SUMMARY.md - Technical details")
    logger.info("\n" + "="*60 + "\n")


def main():
    """Main entry point"""
    print_info()
    
    logger.info("Step 1/4: Checking dependencies...")
    if not check_dependencies():
        return 1
    
    logger.info("\nStep 2/4: Checking analysis modules...")
    if not check_analysis_modules():
        return 1
    
    logger.info("\nStep 3/4: Checking test data...")
    if not generate_test_data():
        logger.warning("Test data generation failed, but you can still use the UI")
        logger.warning("You'll need to provide your own data files")
    
    logger.info("\nStep 4/4: Launching UI...")
    if not launch_ui():
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
