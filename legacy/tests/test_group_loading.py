"""
Test script to diagnose group loading issues.
This creates a minimal FRAPDataManager and tests the file loading and group creation flow.
"""
import os
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Mock the streamlit dependencies for testing
class MockSessionState:
    def __init__(self):
        self.settings = {
            'default_criterion': 'aic',
            'default_pixel_size': 0.3,
            'default_time_interval': 1.0,
            'default_bleach_radius': 1.0
        }

class MockSt:
    def __init__(self):
        self.session_state = MockSessionState()
    
    def error(self, msg):
        print(f"ERROR: {msg}")
    
    def progress(self, val):
        class MockProgress:
            def progress(self, v):
                pass
            def empty(self):
                pass
        return MockProgress()
    
    def empty(self):
        class MockEmpty:
            def text(self, t):
                print(f"STATUS: {t}")
            def empty(self):
                pass
        return MockEmpty()

sys.modules['streamlit'] = MockSt()
import streamlit as st

# Now test the actual loading
print("=" * 80)
print("Testing FRAPDataManager group loading")
print("=" * 80)

# Create a test data manager
from streamlit_frap_final_restored import FRAPDataManager

dm = FRAPDataManager()
print(f"\nInitialized FRAPDataManager")
print(f"Groups: {dm.groups}")
print(f"Files: {dm.files}")

# Test creating a group
group_name = "TestGroup"
dm.create_group(group_name)
print(f"\nCreated group '{group_name}'")
print(f"Groups: {list(dm.groups.keys())}")
print(f"Group data: {dm.groups[group_name]}")

# Test adding a mock file path
test_file_path = "data/test_file.csv"
test_file_name = "test_file.csv"

# Manually add to files dict to simulate load_file
dm.files[test_file_path] = {
    'name': test_file_name,
    'fitted': False
}
print(f"\nManually added file to dm.files")
print(f"Files: {list(dm.files.keys())}")

# Test adding file to group
result = dm.add_file_to_group(group_name, test_file_path)
print(f"\nadd_file_to_group returned: {result}")
print(f"Group '{group_name}' files: {dm.groups[group_name]['files']}")

print("\n" + "=" * 80)
print("Test complete!")
print("=" * 80)
