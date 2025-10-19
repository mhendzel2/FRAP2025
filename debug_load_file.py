"""
Debug script to test file loading from ZIP archive
"""
import zipfile
import tempfile
import os
import sys

# Test loading a file from the ZIP
zip_path = "data/265 PARP2.zip"

print(f"Testing ZIP file: {zip_path}")
print("=" * 80)

with tempfile.TemporaryDirectory() as temp_dir:
    # Extract ZIP
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(temp_dir)
        print(f"✅ Extracted to: {temp_dir}\n")
    
    # Walk through and find data files
    folders_with_data = {}
    
    for root, dirs, files in os.walk(temp_dir):
        data_files = [f for f in files if not f.startswith('.') and 
                     os.path.splitext(f)[1].lower() in ['.xls', '.xlsx', '.csv', '.tif', '.tiff']]
        
        if data_files:
            folder_name = os.path.basename(root)
            if not folder_name.startswith('__') and not folder_name.startswith('.'):
                folders_with_data[root] = {
                    'name': folder_name,
                    'files': data_files[:3]  # Just first 3 files for testing
                }
    
    print(f"Found {len(folders_with_data)} folders with data files\n")
    
    # Test loading first file from first folder
    if folders_with_data:
        first_folder_path = list(folders_with_data.keys())[0]
        first_folder_info = folders_with_data[first_folder_path]
        
        print(f"Testing folder: {first_folder_info['name']}")
        print(f"Files in folder: {len(first_folder_info['files'])}")
        print()
        
        if first_folder_info['files']:
            test_file = first_folder_info['files'][0]
            test_file_path = os.path.join(first_folder_path, test_file)
            
            print(f"Testing file: {test_file}")
            print(f"Full path: {test_file_path}")
            print(f"File exists: {os.path.isfile(test_file_path)}")
            print(f"File size: {os.path.getsize(test_file_path)} bytes")
            print()
            
            # Try to load with CoreFRAPAnalysis
            try:
                from frap_core import CoreFRAPAnalysis
                
                print("Attempting to load with CoreFRAPAnalysis...")
                data = CoreFRAPAnalysis.load_data(test_file_path)
                print(f"✅ Loaded data shape: {data.shape}")
                print(f"✅ Columns: {list(data.columns)}")
                print()
                
                print("Preprocessing...")
                processed = CoreFRAPAnalysis.preprocess(data)
                print(f"✅ Processed shape: {processed.shape}")
                print(f"✅ Processed columns: {list(processed.columns)}")
                print()
                
                if 'normalized' in processed.columns:
                    print(f"✅ Has 'normalized' column")
                    print(f"   Normalized range: {processed['normalized'].min():.3f} to {processed['normalized'].max():.3f}")
                    print(f"   Has nulls: {processed['normalized'].isnull().any()}")
                else:
                    print(f"❌ Missing 'normalized' column!")
                    print(f"   Available columns: {list(processed.columns)}")
                
            except Exception as e:
                print(f"❌ Error loading file: {e}")
                import traceback
                traceback.print_exc()
