import os
import numpy as np

def load_csv_data(root_path):
    """
    Recursively searches for .csv files in subfolders of root_path.
    Loads data into NumPy arrays and organizes them into a nested dictionary
    mirroring the folder structure.
    
    Args:
        root_path (str): The directory to start searching from.
        
    Returns:
        dict: A nested dictionary where keys are folder names or filenames (without extension),
              and values are either sub-dictionaries or NumPy arrays containing the data.
    """
    data_structure = {}

    for dirpath, dirnames, filenames in os.walk(root_path):
        # Determine the relative path to compute dictionary keys
        rel_path = os.path.relpath(dirpath, root_path)
        
        # Split path into parts to navigate the dictionary
        if rel_path == '.':
            path_parts = []
        else:
            path_parts = rel_path.split(os.sep)

        for filename in filenames:
            if filename.endswith('.csv'):
                file_path = os.path.join(dirpath, filename)
                
                # Load the CSV file
                try:
                    # distinct header handling might be needed if header is complex, 
                    # but typically skiprows=1 works for single line headers.
                    csv_data = np.loadtxt(file_path, delimiter=',', skiprows=1)
                except Exception as e:
                    print(f"Warning: Could not load {file_path}. Error: {e}")
                    continue

                # Remove extension for the key
                key_name = os.path.splitext(filename)[0]
                
                # Navigate/Create nested dictionaries
                current_level = data_structure
                for part in path_parts:
                    if part not in current_level:
                        current_level[part] = {}
                    current_level = current_level[part]
                    # Ensure we are not trying to index into an array if a directory 
                    # and file have conflicting names (unlikely but good to consider)
                    if not isinstance(current_level, dict):
                         print(f"Warning: Name collision or structure issue at {part} in {rel_path}")
                         break
                
                # Assign data
                current_level[key_name] = csv_data

    return data_structure

def print_structure(d, indent=0):
    """Recursively prints the keys of the dictionary to show structure."""
    for key, value in d.items():
        print("  " * indent + str(key))
        if isinstance(value, dict):
            print_structure(value, indent + 1)
        elif isinstance(value, np.ndarray):
            print("  " * (indent + 1) + f"<Array shape={value.shape}>")

if __name__ == "__main__":
    current_dir = "."
    print(f"Scanning {os.path.abspath(current_dir)}...")
    data = load_csv_data(current_dir)
    print("\nLoaded Data Structure:")
    print_structure(data)
