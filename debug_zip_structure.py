"""
Debug script to analyze ZIP structure and test group loading
"""
import zipfile
import os
import tempfile

zip_path = 'data/265 PARP2.zip'

print("=" * 70)
print("ZIP STRUCTURE ANALYSIS")
print("=" * 70)

with zipfile.ZipFile(zip_path) as z:
    all_files = z.namelist()
    print(f"\nTotal entries in ZIP: {len(all_files)}")
    
    # Find all directories
    dirs = [f for f in all_files if f.endswith('/')]
    files = [f for f in all_files if not f.endswith('/')]
    
    print(f"Total directories: {len(dirs)}")
    print(f"Total files: {len(files)}")
    
    print("\n" + "-" * 70)
    print("DIRECTORY STRUCTURE:")
    print("-" * 70)
    for d in sorted(dirs)[:20]:  # Show first 20 dirs
        level = d.count('/')
        indent = "  " * (level - 1)
        folder_name = d.rstrip('/').split('/')[-1]
        print(f"{indent}{folder_name}/")
    
    if len(dirs) > 20:
        print(f"  ... and {len(dirs) - 20} more directories")
    
    # Find folders that contain .xls or .xlsx files
    print("\n" + "-" * 70)
    print("FOLDERS CONTAINING DATA FILES:")
    print("-" * 70)
    
    folders_with_files = {}
    for f in files:
        if f.endswith(('.xls', '.xlsx', '.csv')):
            folder = os.path.dirname(f)
            if folder not in folders_with_files:
                folders_with_files[folder] = []
            folders_with_files[folder].append(os.path.basename(f))
    
    for folder in sorted(folders_with_files.keys())[:10]:
        file_count = len(folders_with_files[folder])
        print(f"\n📁 {folder}")
        print(f"   Files: {file_count}")
        print(f"   Sample files: {', '.join(folders_with_files[folder][:3])}")
        if file_count > 3:
            print(f"   ... and {file_count - 3} more")
    
    if len(folders_with_files) > 10:
        print(f"\n... and {len(folders_with_files) - 10} more folders with data files")
    
    print("\n" + "=" * 70)
    print("PROPOSED GROUP STRUCTURE:")
    print("=" * 70)
    
    # Show what groups should be created
    for folder in sorted(folders_with_files.keys())[:10]:
        group_name = os.path.basename(folder)
        file_count = len(folders_with_files[folder])
        print(f"Group: '{group_name}' ({file_count} files)")
    
    print("\n" + "=" * 70)
    print("TESTING os.walk() BEHAVIOR:")
    print("=" * 70)
    
    # Extract and test os.walk
    with tempfile.TemporaryDirectory() as temp_dir:
        z.extractall(temp_dir)
        print(f"\nExtracted to: {temp_dir}")
        
        print("\nos.walk() - First iteration only (current fix):")
        for root, dirs, files in os.walk(temp_dir):
            print(f"\nRoot: {root}")
            print(f"Subdirs: {dirs}")
            print(f"Files: {len(files)} files")
            break  # This is what the current code does
        
        print("\n" + "-" * 70)
        print("os.walk() - All iterations (recursive):")
        iteration = 0
        for root, dirs, files in os.walk(temp_dir):
            iteration += 1
            rel_path = os.path.relpath(root, temp_dir)
            data_files = [f for f in files if f.endswith(('.xls', '.xlsx', '.csv'))]
            if data_files:
                print(f"\n[Iteration {iteration}] {rel_path}")
                print(f"  └─ {len(data_files)} data files found")
                print(f"  └─ Subdirs here: {dirs}")
            if iteration > 15:  # Limit output
                print(f"\n... (showing first 15 iterations)")
                break

print("\n" + "=" * 70)
print("END OF ANALYSIS")
print("=" * 70)
