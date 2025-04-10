import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tkinter.filedialog import askdirectory
from Interpolation import interpolate_and_visualize
import DataLoader
import MatrixCreator

# ----------- HYPERPARAMETERS -----------------
# Hyperparameters for bias removal
bias_average_window = 1000 # Number of samples to average for bias removal
frequency = 519

# Select folder
folder_path = DataLoader.select_folder()

if not folder_path:
    print("No folder selected. Exiting...")
    
print(f"Selected folder: {folder_path}")


# Load and process files
acc_data, gyro_data, or_data, acc_files, gyro_files, or_files = DataLoader.load_and_process_files(folder_path)
print(f"Loaded {len(acc_files)} ACC files, {len(gyro_files)} GYRO files, and {len(or_files)} OR files")
# Group files by timestamp
grouped_indices = DataLoader.group_files_by_timestamp(acc_files, gyro_files, or_files)
    
# Sort timestamps to ensure chronological order
sorted_timestamps = sorted(grouped_indices.keys())
for timestamp in sorted_timestamps:
    indices = grouped_indices[timestamp]
    
    # Get the data for this timestamp
    gyro = gyro_data[indices["gyro"]]
    
    print(f"Processing data set from timestamp: {timestamp}")

        # ------------------------- BIAS REMOVAL ----------------------------
    print("Removing bias...")
    non_zero_index = (gyro != 0).any(axis=1).idxmax()
    sample_size = bias_average_window
    
    if non_zero_index + sample_size <= len(gyro):
        means = gyro.iloc[non_zero_index:non_zero_index + sample_size].mean()
        print("Initial mean values:", means)
    else:
        print("Not enough data after first non-zero value")

    gyro_data_centered = gyro - means
    gyro_data_trimmed = gyro_data_centered.iloc[non_zero_index:].reset_index(drop=True)
    raw_magnitude = np.sqrt(gyro_data_trimmed.iloc[:,0]**2 + 
                    gyro_data_trimmed.iloc[:,1]**2 + 
                    gyro_data_trimmed.iloc[:,2]**2)
   
    # Filtering of the Magntiude
    from scipy.signal import butter
    from scipy.signal import filtfilt
    sampling_rate = frequency  # Hz
    nyquist = 0.5 * sampling_rate
    cutoff = 5  # Hz
    normal_cutoff = cutoff / nyquist
    b, a = butter(4, normal_cutoff, btype='low', analog=False)
    magnitude = filtfilt(b, a, raw_magnitude)

    plt.figure(figsize=(15, 5))
    plt.plot(magnitude, label='Filtered Magnitude', color='blue')
    plt.title(f'Filtered Magnitude for {timestamp}')
    plt.xlabel('Samples')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.grid()
    # Add this inside your timestamp loop after creating the figure
fig = plt.figure(figsize=(15, 5))
plt.plot(magnitude, label='Filtered Magnitude', color='blue')
plt.title(f'Filtered Magnitude for {timestamp}')
plt.xlabel('Samples')
plt.ylabel('Magnitude')
plt.legend()
plt.grid()

# Initialize interactive elements
vertical_line = plt.axvline(0, color='r', linestyle='--', alpha=0.5)
vertical_line.set_visible(False)
selected_index = None

# Create annotations dictionary to store plot elements
annotations = {
    'vertical_line': vertical_line,
    'selected_point': None,
    'timestamp': timestamp
}

def on_motion(event):
    if event.inaxes:
        x = int(round(event.xdata))
        annotations['vertical_line'].set_xdata([x, x])
        annotations['vertical_line'].set_visible(True)
        fig.canvas.draw_idle()

def on_click(event):
    global selected_index
    if event.inaxes:
        x = int(round(event.xdata))
        selected_index = x
        
        # Update visualization
        if annotations['selected_point']:
            annotations['selected_point'].remove()
        annotations['selected_point'] = plt.scatter(x, magnitude[x], 
                                                   color='red', zorder=5,
                                                   label='Selected Index')
        plt.legend()
        fig.canvas.draw_idle()
        print(f"Selected index {x} for {annotations['timestamp']}")

# Connect event listeners
fig.canvas.mpl_connect('motion_notify_event', on_motion)
fig.canvas.mpl_connect('button_press_event', on_click)

plt.show(block=True)

# After closing plot, selected_index contains the chosen value
print(f"\nFinal selection for {timestamp}: Index {selected_index}")


    