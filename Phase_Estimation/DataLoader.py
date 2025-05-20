import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askdirectory


# Function to select folder
def select_folder():
    Tk().withdraw()  # Close the root window
    folder_path = askdirectory(title="Select Folder")
    return folder_path

# Function to load and process files
def load_and_process_files(folder_path):
    acc_files = []
    gyro_files = []
    or_files = []

    # Iterate through files in the folder
    for file_name in os.listdir(folder_path):
        if "ACC_Profile" in file_name:
            acc_files.append(file_name)
        elif "GYRO_Profile" in file_name:
            gyro_files.append(file_name)
        elif "OR_Profile" in file_name and "OR_Debug_Profile" not in file_name:
            or_files.append(file_name)
    
    # Sort files
    acc_files.sort()
    gyro_files.sort()
    or_files.sort()
    
    # Load and process ACC files (and remove last 3 columns)
    acc_data = []
    for acc_file in acc_files:
        df = pd.read_csv(os.path.join(folder_path, acc_file))
        df = df.iloc[:, :-3]  # Remove last 3 columns
        acc_data.append(df)
    
    # Load GYRO files
    gyro_data = []
    for gyro_file in gyro_files:
        df = pd.read_csv(os.path.join(folder_path, gyro_file))
        gyro_data.append(df)
    
    # Load OR files
    or_data = []
    for or_file in or_files:
        df = pd.read_csv(os.path.join(folder_path, or_file))
        or_data.append(df)
    
    return acc_data, gyro_data, or_data, acc_files, gyro_files, or_files

# Function to extract timestamp from filename
def extract_timestamp(filename):
    parts = filename.split("_")
    for part in parts:
        if (len(part) == 2 and part.isdigit()) or (len(part) == 1 and part.isdigit()):  # YYYYMMDDHHMMSS format
            return part
    return None

# Function to group files by timestamp
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

