import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askdirectory
import BiasAndSegmentation
from Interpolation import interpolate_and_visualize

# Function to select folder
def select_folder():
    Tk().withdraw()  # Close the root window
    folder_path = askdirectory(title="Select Folder")
    return folder_path

# Function to load and process files (modified for new structure)
def load_and_process_files(folder_path):
    import pandas as pd
    import os
    import re

    # Function to find header and frequency row
    def find_header_freq_row(file_path):
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if "ACC X" in line:
                    header_row = i
                    next_line = f.readline()
                    freq_row = i + 1 if "Hz" in next_line else None
                    return header_row, freq_row
        return None, None

    # Function to extract trial number
    def extract_trial_num(filename):
        match = re.search(r'Trial_(\d+)', filename)
        return match.group(1) if match else "0"

    trial_files = [f for f in os.listdir(folder_path) 
                  if f.startswith('Trial_') and f.endswith('.csv')]
    trial_files.sort()

    acc_data = []
    gyro_data = []
    or_data = []
    acc_files = []
    gyro_files = []
    or_files = []

    for trial_file in trial_files:
        file_path = os.path.join(folder_path, trial_file)
        header_row, freq_row = find_header_freq_row(file_path)
        
        if None in (header_row, freq_row):
            print(f"Skipping {trial_file} - invalid format")
            continue

        # Read data with proper numeric conversion
        df = pd.read_csv(file_path, skiprows=freq_row+1)
        df = df.apply(pd.to_numeric, errors='coerce').dropna()

        # Extract trial number for filename reconstruction
        trial_num = extract_trial_num(trial_file)

        # Original column structure
        acc_df = df.iloc[:, :3].copy()
        acc_df.columns = ['ACC X', 'ACC Y', 'ACC Z']
        
        gyro_df = df.iloc[:, 3:6].copy()
        gyro_df.columns = ['GYRO X', 'GYRO Y', 'GYRO Z']
        
        or_df = df.iloc[:, 7:11].copy()
        or_df.columns = ['ORIENTATION W', 'ORIENTATION X', 
                        'ORIENTATION Y', 'ORIENTATION Z']

        # Maintain original data structure
        acc_data.append(acc_df)
        gyro_data.append(gyro_df)
        or_data.append(or_df)

        # Reconstruct original filenames
        acc_files.append(f"ACC_Profile_Data_Trial_{trial_num}.csv")
        gyro_files.append(f"GYRO_Profile_Data_Trial_{trial_num}.csv")
        or_files.append(f"OR_Profile_Data_Trial_{trial_num}.csv")

    return acc_data, gyro_data, or_data, acc_files, gyro_files, or_files


# Function to extract timestamp from filename (modified)
def extract_timestamp(filename):
    # Extract trial number as "timestamp"
    parts = filename.split("_")
    for part in parts:
        if part.replace(".csv", "").isdigit():
            return part.replace(".csv", "")
    return None

# Function to group files by timestamp (modified)
def group_files_by_timestamp(acc_files, gyro_files, or_files):
    grouped_indices = {}
    
    # Process ACC files
    for i, file_name in enumerate(acc_files):
        timestamp = extract_timestamp(file_name)
        if timestamp not in grouped_indices:
            grouped_indices[timestamp] = {"acc": None, "gyro": None, "or": None}
        grouped_indices[timestamp]["acc"] = i
    
    # Process GYRO files
    for i, file_name in enumerate(gyro_files):
        timestamp = extract_timestamp(file_name)
        if timestamp not in grouped_indices:
            grouped_indices[timestamp] = {"acc": None, "gyro": None, "or": None}
        grouped_indices[timestamp]["gyro"] = i
    
    # Process OR files
    for i, file_name in enumerate(or_files):
        timestamp = extract_timestamp(file_name)
        if timestamp not in grouped_indices:
            grouped_indices[timestamp] = {"acc": None, "gyro": None, "or": None}
        grouped_indices[timestamp]["or"] = i
    
    # Filter out incomplete groups
    complete_groups = {ts: indices for ts, indices in grouped_indices.items() 
                      if indices["acc"] is not None and indices["gyro"] is not None and indices["or"] is not None}
    
    return complete_groups
