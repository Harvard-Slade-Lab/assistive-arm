import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tkinter.filedialog import askdirectory
from scipy import signal
from Interpolation import interpolate_and_visualize
import DataLoader
import MatrixCreator

# ----------- HYPERPARAMETERS -----------------
# Hyperparameters for bias removal
bias_average_window = 1000  # Number of samples to average for bias removal
frequency = 519

# Hyperparameters for SHOE detector - DRAMATICALLY INCREASED THRESHOLD
shoe_window_size = 20
min_segment_duration = 0.2  # Minimum duration for a valid segment (seconds)
GAMMA_THRESHOLD = 100000  # MUCH HIGHER threshold based on your data scale

# Function to remove gravity from acceleration data
def remove_gravity(acc_data, window_size=100):
    """Remove gravity component using a low-pass filter."""
    # Convert DataFrame to numpy array for filtering
    acc_array = acc_data.values
    
    # Apply low-pass filter to extract gravity
    b, a = signal.butter(4, 0.1, 'low')
    gravity = signal.filtfilt(b, a, acc_array, axis=0)
    
    # Subtract gravity from acceleration
    linear_acc = pd.DataFrame(acc_array - gravity, columns=acc_data.columns)
    
    return linear_acc

# SHOE detector implementation - MODIFIED to handle extreme values
def shoe_detector(linear_acc, angular_vel, window_size=20, gamma=100000):
    """
    SHOE detector with direct threshold comparison (no variance normalization).
    
    Parameters:
    - linear_acc: DataFrame of 3D linear acceleration data (gravity removed)
    - angular_vel: DataFrame of 3D angular velocity data
    - window_size: Size of the sliding window (N)
    - gamma: Threshold parameter - DRAMATICALLY INCREASED
    
    Returns:
    - zero_vel: Array of binary values (1: stationary, 0: moving)
    """
    # Convert DataFrames to numpy arrays
    linear_acc_array = linear_acc.values
    angular_vel_array = angular_vel.values
    
    data_length = len(linear_acc)
    
    # Calculate squared magnitudes WITHOUT variance normalization
    acc_sq_norm = np.sum(linear_acc_array**2, axis=1)
    gyro_sq_norm = np.sum(angular_vel_array**2, axis=1)
    
    # Combined term - balance contributions by scaling
    combined_term = acc_sq_norm + gyro_sq_norm
    
    # Moving average using convolution
    window = np.ones(window_size) / window_size
    windowed_avg = np.convolve(combined_term, window, mode='same')
    
    # Apply threshold (1 when below threshold - stationary)
    zero_vel = np.where(windowed_avg < gamma, 1, 0)
    
    # Debugging visualization
    plt.figure(figsize=(15, 10))
    plt.subplot(4, 1, 1)
    plt.plot(acc_sq_norm, label='Acc Square Norm')
    plt.title('Accelerometer Component')
    plt.legend()
    
    plt.subplot(4, 1, 2)
    plt.plot(gyro_sq_norm, label='Gyro Square Norm')
    plt.title('Gyroscope Component')
    plt.legend()
    
    plt.subplot(4, 1, 3)
    plt.plot(windowed_avg, label='Windowed Average')
    plt.axhline(y=gamma, color='r', linestyle='--', label=f'Threshold (γ={gamma})')
    plt.title(f'Combined Term with Window Size={window_size}')
    plt.legend()
    
    plt.subplot(4, 1, 4)
    plt.plot(zero_vel, label='Zero-Velocity Detection')
    plt.title('Detection Result (1: Stationary, 0: Moving)')
    plt.ylim(-0.1, 1.1)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return zero_vel

# Function to segment motion based on zero-velocity detection
def segment_motion(zero_vel, min_duration=0.2, frequency=519):
    """Segment motion using zero-velocity detection results."""
    segments = []
    current_state = zero_vel[0]
    start_idx = 0
    
    for i in range(1, len(zero_vel)):
        if zero_vel[i] != current_state:
            # Calculate duration in seconds
            segment_duration = (i - start_idx) / frequency
            
            if segment_duration >= min_duration:
                segments.append((start_idx, i-1, bool(current_state)))
            
            start_idx = i
            current_state = zero_vel[i]
    
    # Add the final segment if it meets duration requirement
    if (len(zero_vel) - 1 - start_idx) / frequency >= min_duration:
        segments.append((start_idx, len(zero_vel)-1, bool(current_state)))
    
    return segments

# Function to visualize segmentation results
def visualize_segments(acc_data, gyro_data, zero_vel, segments, raw_magnitude=None, title="Motion Segmentation"):
    """Visualize motion segments and sensor data."""
    # Calculate magnitudes if not provided
    acc_magnitude = np.sqrt(acc_data.iloc[:,0]**2 + acc_data.iloc[:,1]**2 + acc_data.iloc[:,2]**2)
    if raw_magnitude is None:
        gyro_magnitude = np.sqrt(gyro_data.iloc[:,0]**2 + gyro_data.iloc[:,1]**2 + gyro_data.iloc[:,2]**2)
    else:
        gyro_magnitude = raw_magnitude
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    
    # Plot accelerometer data
    axes[0].plot(acc_magnitude, 'b-', label='Acceleration Magnitude')
    axes[0].set_ylabel('Acceleration Magnitude')
    axes[0].legend()
    
    # Plot gyroscope data
    axes[1].plot(gyro_magnitude, 'g-', label='Angular Velocity Magnitude')
    axes[1].set_ylabel('Angular Velocity Magnitude')
    axes[1].legend()
    
    # Plot zero-velocity detection
    axes[2].plot(zero_vel, 'r-', label='Zero-Velocity Detection')
    axes[2].set_ylabel('State (1: Static, 0: Moving)')
    axes[2].set_ylim(-0.1, 1.1)
    axes[2].legend()
    
    # Highlight segments
    for start_idx, end_idx, is_stationary in segments:
        color = 'green' if is_stationary else 'red'
        for ax in axes:
            ax.axvspan(start_idx, end_idx, alpha=0.2, color=color)
    
    axes[2].set_xlabel('Samples')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# Function to find optimal threshold automatically
def find_optimal_threshold(linear_acc, angular_vel, raw_magnitude):
    """Find optimal threshold by analyzing gyro magnitude pattern."""
    # Convert DataFrames to numpy arrays
    linear_acc_array = linear_acc.values
    angular_vel_array = angular_vel.values
    
    # Calculate squared magnitudes
    acc_sq_norm = np.sum(linear_acc_array**2, axis=1)
    gyro_sq_norm = np.sum(angular_vel_array**2, axis=1)
    
    # Combined term
    combined_term = acc_sq_norm + gyro_sq_norm
    
    # Find maximum value and suggest 5-10% of that for threshold
    max_value = np.max(combined_term)
    suggested_thresholds = [max_value * 0.01, max_value * 0.05, max_value * 0.1]
    
    print(f"Maximum combined value: {max_value}")
    print(f"Suggested thresholds: {suggested_thresholds}")
    
    # Test different thresholds
    window = np.ones(shoe_window_size) / shoe_window_size
    windowed_avg = np.convolve(combined_term, window, mode='same')
    
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    plt.plot(raw_magnitude, label='Gyro Magnitude')
    plt.title('Gyro Magnitude for Reference')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(windowed_avg, label='Combined Term')
    for i, thresh in enumerate(suggested_thresholds):
        plt.axhline(y=thresh, color=f'C{i+1}', linestyle='--', 
                   label=f'Threshold {i+1}: {thresh:.1f} ({thresh/max_value*100:.1f}% of max)')
    
    plt.title('Combined Term and Potential Thresholds')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return suggested_thresholds

# Select folder
folder_path = DataLoader.select_folder()

if not folder_path:
    print("No folder selected. Exiting...")

print(f"Selected folder: {folder_path}")

# Load and process files
acceleration_data, gyro_data, or_data, acc_files, gyro_files, or_files = DataLoader.load_and_process_files(folder_path)
print(f"Loaded {len(acc_files)} ACC files, {len(gyro_files)} GYRO files, and {len(or_files)} OR files")

# Group files by timestamp
grouped_indices = DataLoader.group_files_by_timestamp(acc_files, gyro_files, or_files)

# Sort timestamps to ensure chronological order
sorted_timestamps = sorted(grouped_indices.keys())
for timestamp in sorted_timestamps:
    indices = grouped_indices[timestamp]

    # Get the data for this timestamp
    gyro = gyro_data[indices["gyro"]]
    acc_data = acceleration_data[indices["acc"]]
    orientation_data = or_data[indices["or"]]

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

    # ------------------------- SHOE DETECTOR IMPLEMENTATION ----------------------------
    print("Applying SHOE detector for motion segmentation...")
    
    # Trim acceleration data to match gyro data
    acc_data_trimmed = acc_data.iloc[non_zero_index:].reset_index(drop=True)
    
    # Remove gravity from acceleration data
    linear_acc = remove_gravity(acc_data_trimmed)
    
    # Find optimal threshold (auto-suggestion)
    suggested_thresholds = find_optimal_threshold(linear_acc, gyro_data_trimmed, raw_magnitude)
    
    # Choose a threshold (use the middle value from suggestions by default)
    chosen_threshold = suggested_thresholds[1]  # Adjust this index if needed
    print(f"Using threshold: {chosen_threshold}")
    
    # Apply SHOE detector with the chosen threshold
    zero_vel = shoe_detector(linear_acc, gyro_data_trimmed, 
                            window_size=shoe_window_size, 
                            gamma=chosen_threshold)
    
    # Segment motion
    segments = segment_motion(zero_vel, min_duration=min_segment_duration, frequency=frequency)
    
    print(f"Detected {len(segments)} motion segments")
    for i, (start_idx, end_idx, is_stationary) in enumerate(segments):
        state = "Stationary" if is_stationary else "Moving"
        duration = (end_idx - start_idx) / frequency
        print(f"Segment {i+1}: {state}, Duration: {duration:.2f}s, From {start_idx} to {end_idx}")
    
    # Visualize original gyro magnitude
    plt.figure(figsize=(15, 5))
    plt.plot(raw_magnitude, label='Filtered Magnitude', color='blue')
    plt.title(f'Filtered Magnitude for {timestamp}')
    plt.xlabel('Samples')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.grid()
    plt.show(block=True)
    
    # Visualize segmentation results
    visualize_segments(acc_data_trimmed, gyro_data_trimmed, zero_vel, segments, 
                      raw_magnitude=raw_magnitude, title=f"Motion Segmentation for {timestamp}")
