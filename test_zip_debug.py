#!/usr/bin/env python3
"""
Test script to debug ZIP loading functionality
"""

import os
import sys
import tempfile
import zipfile
import io
from frap_manager import FRAPDataManager

def test_zip_loading():
    """Test the ZIP loading functionality with debug output"""
    
    # Create a simple test structure
    dm = FRAPDataManager()
    
    print("Initial state:")
    print(f"Groups: {len(dm.groups)}")
    print(f"Files: {len(dm.files)}")
    
    # Test with a sample data ZIP if it exists
    sample_zip_path = "sample_data.zip"  # Adjust path as needed
    
    if os.path.exists(sample_zip_path):
        print(f"\nTesting with ZIP file: {sample_zip_path}")
        
        # Simulate streamlit file upload object
        class MockFileUpload:
            def __init__(self, file_path):
                self.name = os.path.basename(file_path)
                with open(file_path, 'rb') as f:
                    self._content = f.read()
            
            def getbuffer(self):
                return io.BytesIO(self._content).getvalue()
        
        mock_zip = MockFileUpload(sample_zip_path)
        
        try:
            success = dm.load_groups_from_zip_archive(mock_zip)
            print(f"\nResult: {'Success' if success else 'Failed'}")
            print(f"Groups created: {len(dm.groups)}")
            print(f"Files loaded: {len(dm.files)}")
            
            # Show details
            for group_name, group_data in dm.groups.items():
                file_count = len(group_data.get('files', []))
                print(f"  📁 {group_name}: {file_count} files")
                
                # Show files in each group
                for file_path in group_data.get('files', []):
                    if file_path in dm.files:
                        file_info = dm.files[file_path]
                        print(f"    • {file_info.get('name', 'Unknown')}")
                        print(f"      - Original path: {file_info.get('original_path', 'N/A')}")
                        print(f"      - Group name: {file_info.get('group_name', 'N/A')}")
                    else:
                        print(f"    • {file_path} (NOT FOUND IN FILES DICT)")
                        
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Test ZIP file {sample_zip_path} not found")
        print("Available files in current directory:")
        for f in os.listdir("."):
            if f.endswith(".zip"):
                print(f"  - {f}")

if __name__ == "__main__":
    test_zip_loading()
