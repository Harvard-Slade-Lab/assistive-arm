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
    Tk().withdraw()
    folder_path = askdirectory(title="Select Folder")
    return folder_path

# Function to load and process files (for new IMU structure)
def load_and_process_files(folder_path):
    import re

    def extract_trial_num(filename):
        match = re.search(r'Trial_(\d+)', filename)
        return match.group(1) if match else "0"

    # Find all IMU_Profile_*.csv files
    trial_files = [f for f in os.listdir(folder_path)
                   if f.startswith('IMU_Profile_') and f.endswith('.csv')]
    trial_files.sort()

    acc_data = []
    gyro_data = []
    or_data = []
    acc_files = []
    gyro_files = []
    or_files = []

    for trial_file in trial_files:
        file_path = os.path.join(folder_path, trial_file)
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Skipping {trial_file} - cannot read CSV: {e}")
            continue

        # Validate columns
        expected_cols = ['Roll', 'Pitch', 'Yaw', 'AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ']
        if not all(col in df.columns for col in expected_cols):
            print(f"Skipping {trial_file} - missing expected columns")
            continue

        trial_num = extract_trial_num(trial_file)

        # Orientation (Roll, Pitch, Yaw)
        or_df = df[['Roll', 'Pitch', 'Yaw']].copy()
        or_df.columns = ['ORIENTATION ROLL', 'ORIENTATION PITCH', 'ORIENTATION YAW']

        # Accelerometer (AccX, AccY, AccZ)
        acc_df = df[['AccX', 'AccY', 'AccZ']].copy()
        acc_df.columns = ['ACC X', 'ACC Y', 'ACC Z']

        # Gyroscope (GyroX, GyroY, GyroZ)
        gyro_df = df[['GyroX', 'GyroY', 'GyroZ']].copy()
        gyro_df.columns = ['GYRO X', 'GYRO Y', 'GYRO Z']

        acc_data.append(acc_df)
        gyro_data.append(gyro_df)
        or_data.append(or_df)

        acc_files.append(f"ACC_Profile_Data_Trial_{trial_num}.csv")
        gyro_files.append(f"GYRO_Profile_Data_Trial_{trial_num}.csv")
        or_files.append(f"OR_Profile_Data_Trial_{trial_num}.csv")

    return acc_data, gyro_data, or_data, acc_files, gyro_files, or_files

# Function to extract timestamp from filename (unchanged)
def extract_timestamp(filename):
    parts = filename.split("_")
    for part in parts:
        if part.replace(".csv", "").isdigit():
            return part.replace(".csv", "")
    return None

# Function to group files by timestamp (unchanged)
def group_files_by_timestamp(acc_files, gyro_files, or_files):
    grouped_indices = {}

    for i, file_name in enumerate(acc_files):
        timestamp = extract_timestamp(file_name)
        if timestamp not in grouped_indices:
            grouped_indices[timestamp] = {"acc": None, "gyro": None, "or": None}
        grouped_indices[timestamp]["acc"] = i

    for i, file_name in enumerate(gyro_files):
        timestamp = extract_timestamp(file_name)
        if timestamp not in grouped_indices:
            grouped_indices[timestamp] = {"acc": None, "gyro": None, "or": None}
        grouped_indices[timestamp]["gyro"] = i

    for i, file_name in enumerate(or_files):
        timestamp = extract_timestamp(file_name)
        if timestamp not in grouped_indices:
            grouped_indices[timestamp] = {"acc": None, "gyro": None, "or": None}
        grouped_indices[timestamp]["or"] = i

    complete_groups = {ts: indices for ts, indices in grouped_indices.items()
                      if indices["acc"] is not None and indices["gyro"] is not None and indices["or"] is not None}

    return complete_groups

# Example usage
if __name__ == "__main__":
    folder = select_folder()
    acc_data, gyro_data, or_data, acc_files, gyro_files, or_files = load_and_process_files(folder)
    groups = group_files_by_timestamp(acc_files, gyro_files, or_files)
    print(f"Found {len(groups)} complete trial(s).")
