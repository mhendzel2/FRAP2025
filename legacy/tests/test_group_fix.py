#!/usr/bin/env python3
"""
Test script to verify the group population fix.
"""
import sys

# Mock streamlit for testing
class MockSt:
    def __init__(self): 
        class MockSession: 
            def __init__(self): self.settings = {}
        self.session_state = MockSession()
    def error(self, msg): print(f'ERROR: {msg}')
    def progress(self, val): 
        class MockProgress:
            def progress(self, v): pass
            def empty(self): pass
        return MockProgress()
    def empty(self):
        class MockEmpty:
            def text(self, t): print(f'STATUS: {t}')
            def empty(self): pass
        return MockEmpty()

sys.modules['streamlit'] = MockSt()
from streamlit_frap_final_restored import FRAPDataManager

def test_group_population():
    """Test the group population functionality"""
    print("🧪 Testing group population fix...")
    
    # Initialize data manager
    dm = FRAPDataManager()
    print(f"✓ Initialized FRAPDataManager")
    
    # Test group creation
    dm.create_group('TestGroup')
    print(f"✓ Created group: {list(dm.groups.keys())}")
    
    # Simulate file data
    dm.files['test_file.csv'] = {'name': 'test_file.csv', 'fitted': False}
    print("✓ Added mock file to dm.files")
    
    # Test adding file to group using proper method
    result = dm.add_file_to_group('TestGroup', 'test_file.csv')
    print(f"✓ add_file_to_group result: {result}")
    
    # Verify file was added
    files_in_group = dm.groups['TestGroup']['files']
    print(f"✓ Files in TestGroup: {files_in_group}")
    
    # Test adding same file again (should not duplicate)
    result2 = dm.add_file_to_group('TestGroup', 'test_file.csv')
    print(f"✓ Second add_file_to_group result: {result2}")
    files_after_second_add = dm.groups['TestGroup']['files']
    print(f"✓ Files after second add: {files_after_second_add}")
    
    # Validate results
    assert result == True, "First add should succeed"
    assert result2 == False, "Second add should fail (already exists)"
    assert len(files_in_group) == 1, "Should have exactly 1 file"
    assert len(files_after_second_add) == 1, "Should still have exactly 1 file"
    assert 'test_file.csv' in files_in_group, "File should be in group"
    
    print("🎉 All tests passed! Group population is working correctly.")
    return True

if __name__ == "__main__":
    try:
        test_group_population()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)