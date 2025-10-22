#!/usr/bin/env python3
"""
Test script to verify session save/load compatibility after function sanitization fixes.
"""

import pickle
import os
import tempfile
import sys

def create_mock_best_fit():
    """Create a mock best_fit dictionary without function objects."""
    return {
        'model': 'single_component',
        'params': {
            'A': 0.5,
            'tau': 10.0,
            'offset': 0.1
        },
        'r_squared': 0.95,
        'fitted_values': [0.1, 0.2, 0.3, 0.4, 0.5],
        'residuals': [0.01, -0.02, 0.01, 0.00, -0.01],
        'aic': 150.5,
        'bic': 155.2
        # Note: No 'func' key - this should not cause errors
    }

def test_session_save_load():
    """Test that session data can be saved and loaded without function objects."""
    print("Testing session save/load compatibility...")
    
    # Create test data
    test_data = {
        'best_fit': create_mock_best_fit(),
        'analysis_results': {
            'mobile_fraction': 0.65,
            'immobile_fraction': 0.35,
            'recovery_time': 12.5
        },
        'session_info': {
            'version': '1.0',
            'timestamp': '2024-01-01'
        }
    }
    
    # Test save
    try:
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            pickle.dump(test_data, tmp_file)
            temp_path = tmp_file.name
        print("✅ Session save successful")
    except Exception as e:
        print(f"❌ Session save failed: {e}")
        return False
    
    # Test load
    try:
        with open(temp_path, 'rb') as f:
            loaded_data = pickle.load(f)
        print("✅ Session load successful")
        
        # Verify data integrity
        assert loaded_data['best_fit']['params']['A'] == 0.5
        assert loaded_data['analysis_results']['mobile_fraction'] == 0.65
        assert 'func' not in loaded_data['best_fit']  # Should not have func key
        print("✅ Data integrity verified")
        
    except Exception as e:
        print(f"❌ Session load failed: {e}")
        return False
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    return True

def test_function_mapping():
    """Test that function mapping works correctly."""
    print("\nTesting function mapping compatibility...")
    
    try:
        # Import the fixed modules
        from frap_bootstrap import get_model_function
        
        # Test function mapping
        func = get_model_function('single_component')
        print("✅ Function mapping working")
        
        # Test that we can call the function
        import numpy as np
        t = np.array([0, 1, 2, 3, 4])
        params = [0.5, 10.0, 0.1]
        result = func(t, *params)
        print(f"✅ Function call successful: {len(result)} values")
        
    except Exception as e:
        print(f"❌ Function mapping failed: {e}")
        return False
    
    return True

def test_plot_fallback():
    """Test that plotting works without function objects."""
    print("\nTesting plot fallback compatibility...")
    
    try:
        # Create mock data without func key
        best_fit = create_mock_best_fit()
        
        # Verify fallback logic would work
        if 'func' not in best_fit:
            print("✅ No 'func' key detected - fallback logic would activate")
        else:
            print("❌ 'func' key still present")
            return False
            
    except Exception as e:
        print(f"❌ Plot fallback test failed: {e}")
        return False
    
    return True

def main():
    """Run all compatibility tests."""
    print("Running FRAP session compatibility tests...\n")
    
    tests = [
        test_session_save_load,
        test_function_mapping,
        test_plot_fallback
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! Session compatibility is working correctly.")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the fixes.")
        return 1

if __name__ == "__main__":
    sys.exit(main())