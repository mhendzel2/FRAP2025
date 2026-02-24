#!/usr/bin/env python3
"""
Test script to verify that the comprehensive plot function works correctly.
This tests the fix for the "Failed to generate comprehensive plot" error.
"""

import numpy as np
from frap_plots import FRAPPlots

def test_comprehensive_plot():
    """Test that comprehensive plots can be generated correctly"""
    
    print("Testing comprehensive plot generation...")
    
    # Create sample data
    time = np.linspace(0, 100, 50)
    
    # Test cases for different models using the actual fit_result structure
    test_cases = [
        {
            'name': 'Single Exponential Model',
            'fit_result': {
                'model': 'single',
                'fitted_values': None,  # Will be generated
                'r2': 0.95,
                'aic': 45.2,
                'bic': 48.7
            }
        },
        {
            'name': 'Double Exponential Model',  
            'fit_result': {
                'model': 'double',
                'fitted_values': None,  # Will be generated
                'r2': 0.98,
                'aic': 42.1,
                'bic': 47.3
            }
        },
        {
            'name': 'Unknown Model Type',
            'fit_result': {
                'model': 'custom',
                'fitted_values': None,  # Will be generated
                'r2': 0.92,
                'aic': 50.5,
                'bic': 54.2
            }
        }
    ]
    
    print(f"{'Test Case':<30} {'Status':<15} {'Details'}")
    print("-" * 70)
    
    all_passed = True
    
    for test_case in test_cases:
        name = test_case['name']
        fit_result = test_case['fit_result']
        
        # Generate synthetic intensity data (mock recovery curve)
        intensity = 0.2 + 0.6 * (1 - np.exp(-0.05 * time))  # Simple recovery curve
        intensity += np.random.normal(0, 0.01, len(intensity))  # Add noise
        
        # Generate fitted values (slightly smoother version)
        fitted_values = 0.2 + 0.6 * (1 - np.exp(-0.05 * time))
        fit_result['fitted_values'] = fitted_values
        
        try:
            # Test the comprehensive plot function
            fig = FRAPPlots.plot_comprehensive_fit(
                time=time,
                intensity=intensity,
                fit_result=fit_result,
                file_name=f"test_{fit_result['model']}_model",
                height=800
            )
            
            if fig is not None:
                # Check that the figure has the expected structure
                if hasattr(fig, 'data') and len(fig.data) > 0:
                    status = "✅ PASS"
                    details = f"Generated {len(fig.data)} traces"
                else:
                    status = "❌ FAIL"
                    details = "Empty figure generated"
                    all_passed = False
            else:
                status = "❌ FAIL"
                details = "Function returned None"
                all_passed = False
                
        except Exception as e:
            status = "❌ ERROR"
            details = f"Exception: {str(e)[:30]}..."
            all_passed = False
            
        print(f"{name:<30} {status:<15} {details}")
    
    print("-" * 70)
    if all_passed:
        print("✅ ALL TESTS PASSED! Comprehensive plot generation is working correctly.")
        print("The 'Failed to generate comprehensive plot' error should now be resolved.")
    else:
        print("❌ SOME TESTS FAILED! There may still be issues with comprehensive plot generation.")
    
    return all_passed

if __name__ == "__main__":
    # Test the comprehensive plot directly without worrying about model functions
    print("Testing comprehensive plot with simplified approach...")
    
    try:
        # Run the test with mock data
        success = test_comprehensive_plot()
        
        if success:
            print("\n🎉 SUCCESS! The comprehensive plot function should now work correctly.")
            print("The 'Failed to generate comprehensive plot' error should be resolved.")
        else:
            print("\n⚠️ Some issues remain. Check the error details above.")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()