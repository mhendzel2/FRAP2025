import os
import shutil

def flatten_directory_structure(target_parent_directory):
    """
    Scans the target directory for all nested subdirectories.
    Moves every nested subdirectory to be a direct child of the target_parent_directory.
    Renames folders if a folder with the same name already exists at the top level.
    """
    
    target_parent_directory = os.path.abspath(target_parent_directory)
    
    if not os.path.exists(target_parent_directory):
        print(f"Error: The directory '{target_parent_directory}' does not exist.")
        return

    print(f"Processing parent folder: {target_parent_directory}")
    
    # We need to walk bottom-up (topdown=False) so we move deepest folders first.
    # This prevents issues where moving a parent folder breaks the path to its children.
    # However, since we are moving the children OUT of the parent, we actually want
    # to find the deep folders, move them, and then proceed.
    
    # Actually, a safer way is to just get a list of all directories first, 
    # filter out the top-level ones (which are already where we want them),
    # and then move the rest.
    
    all_dirs_to_move = []
    
    for root, dirs, files in os.walk(target_parent_directory):
        # We modify 'dirs' in place to prevent os.walk from walking into directories we just moved?
        # No, os.walk is a generator. Moving directories while walking is dangerous.
        # Standard approach: Collect all paths first, then move.
        
        for d in dirs:
            full_path = os.path.join(root, d)
            
            # Check if this directory is ALREADY a direct child of the target
            # os.path.dirname(full_path) gives the parent folder.
            if os.path.abspath(os.path.dirname(full_path)) == target_parent_directory:
                continue # It's already at the top level
            
            all_dirs_to_move.append(full_path)

    if not all_dirs_to_move:
        print("No nested subdirectories found to flatten.")
        return

    print(f"Found {len(all_dirs_to_move)} nested folders to promote.")
    
    # We must sort by length of path (longest first) to move deep folders before their parents.
    # Actually, order matters less if we are moving TO the top, but deep-first is safer logic.
    all_dirs_to_move.sort(key=len, reverse=True)

    moved_count = 0
    
    for source_path in all_dirs_to_move:
        folder_name = os.path.basename(source_path)
        destination_path = os.path.join(target_parent_directory, folder_name)
        
        # Check if the folder is still there (it might have been inside a folder we just moved?)
        # Since we are moving nested folders OUT, moving a deep child first is good. 
        # But wait, if we have A/B/C. 
        # If we move C to Root/C. 
        # Then we move B to Root/B. 
        # A is now empty (or has files). 
        # This works perfectly.
        
        if not os.path.exists(source_path):
            print(f"Skipping {source_path} (not found - maybe moved with parent?)")
            continue

        # Handle Duplicate Folder Names at Destination
        if os.path.exists(destination_path):
            counter = 1
            while os.path.exists(destination_path):
                new_name = f"{folder_name}_{counter}"
                destination_path = os.path.join(target_parent_directory, new_name)
                counter += 1
            print(f"Duplicate name. Renaming '{folder_name}' -> '{os.path.basename(destination_path)}'")
        
        try:
            shutil.move(source_path, destination_path)
            # print(f"Moved: {source_path} -> {destination_path}")
            moved_count += 1
        except Exception as e:
            print(f"Error moving {source_path}: {e}")

    print("-" * 30)
    print(f"Process Complete. Promoted {moved_count} nested folders to the top level.")

if __name__ == "__main__":
    # --- CONFIGURATION ---
    target_folder = r"C:\Users\mjhen\Downloads\PARP_FRAP_aLL"
    
    print(f"WARNING: This script will move ALL nested folders inside:")
    print(f"'{target_folder}'")
    print("to become direct subfolders of that parent folder.")
    print("Example: 'Main/A/B/C' becomes 'Main/A', 'Main/B', 'Main/C'")
    
    confirm = input("Type 'yes' to proceed: ")
    
    if confirm.lower() == 'yes':
        flatten_directory_structure(target_folder)
    else:
        print("Operation cancelled.")