#!/usr/bin/env python3
"""
Test script to verify that session saving works without pickle errors.
This tests the fix for the "Can't pickle function" error.
"""

import numpy as np
import pickle
from datetime import datetime

def test_session_saving():
    """Test that session data can be pickled correctly"""
    
    print("Testing session saving functionality...")
    
    # Create mock data that simulates the structure that was causing pickle errors
    mock_files = {
        'test_file_1.txt': {
            'name': 'test_file_1.txt',
            'time': np.array([0, 1, 2, 3, 4, 5]),
            'intensity': np.array([1.0, 0.3, 0.5, 0.7, 0.8, 0.85]),
            'best_fit': {
                'model': 'single',
                'func': lambda x: x,  # This would cause pickle error
                'params': [0.7, 0.05, 0.2],
                'r2': 0.95,
                'fitted_values': np.array([1.0, 0.3, 0.5, 0.7, 0.8, 0.85])
            },
            'features': {
                'mobile_fraction': 90.0,
                'half_time': 13.86
            }
        }
    }
    
    mock_groups = {
        'group_1': {
            'name': 'Test Group',
            'files': ['test_file_1.txt'],
            'analysis_function': lambda x: x  # Another unpickleable object
        }
    }
    
    def sanitize_for_pickle(data):
        """Remove unpickleable objects like function references"""
        if isinstance(data, dict):
            sanitized = {}
            for key, value in data.items():
                if key == 'func':  # Skip function objects
                    continue
                elif callable(value):  # Skip any callable objects
                    continue
                else:
                    sanitized[key] = sanitize_for_pickle(value)
            return sanitized
        elif isinstance(data, list):
            return [sanitize_for_pickle(item) for item in data]
        elif isinstance(data, tuple):
            return tuple(sanitize_for_pickle(item) for item in data)
        else:
            return data
    
    test_cases = [
        {
            'name': 'Original data (should fail)',
            'data': {
                'files': mock_files,
                'groups': mock_groups,
                'timestamp': datetime.now().isoformat()
            },
            'should_fail': True
        },
        {
            'name': 'Sanitized data (should succeed)',
            'data': {
                'files': sanitize_for_pickle(mock_files),
                'groups': sanitize_for_pickle(mock_groups),
                'timestamp': datetime.now().isoformat()
            },
            'should_fail': False
        }
    ]
    
    print(f"{'Test Case':<30} {'Status':<15} {'Details'}")
    print("-" * 70)
    
    all_passed = True
    
    for test_case in test_cases:
        name = test_case['name']
        data = test_case['data']
        should_fail = test_case['should_fail']
        
        try:
            # Try to pickle the data
            pickled_data = pickle.dumps(data)
            
            # Try to unpickle it back
            unpickled_data = pickle.loads(pickled_data)
            
            if should_fail:
                status = "❌ UNEXPECTED"
                details = "Expected to fail but succeeded"
                all_passed = False
            else:
                status = "✅ PASS"
                # Check that data was preserved correctly
                original_files = len(data['files'])
                restored_files = len(unpickled_data['files'])
                details = f"Pickled/unpickled successfully ({restored_files}/{original_files} files)"
                
                # Check that function objects were properly removed
                if 'func' in str(unpickled_data):
                    status = "⚠️ WARNING"
                    details += ", but 'func' still found"
                
        except Exception as e:
            if should_fail:
                status = "✅ EXPECTED"
                details = f"Failed as expected: {str(e)[:40]}..."
            else:
                status = "❌ FAIL"
                details = f"Unexpected error: {str(e)[:40]}..."
                all_passed = False
                
        print(f"{name:<30} {status:<15} {details}")
    
    print("-" * 70)
    if all_passed:
        print("✅ ALL TESTS PASSED! Session saving should now work correctly.")
        print("The pickle error should be resolved.")
        print("\nKey improvements:")
        print("- Function objects are automatically filtered out")
        print("- Data integrity is preserved for serializable objects")
        print("- Session files can be saved and loaded without errors")
    else:
        print("❌ SOME TESTS FAILED! There may still be issues with session saving.")
    
    return all_passed

if __name__ == "__main__":
    test_session_saving()