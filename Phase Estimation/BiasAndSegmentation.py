import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog, Tk
from scipy import interpolate


# ----------- HYPERPARAMETERS -----------------
# Hyperparameters for bias removal
bias_average_window = 1000 # Number of samples to average for bias removal

# Hyperparameters for segmentation
offset = 7.0 # Offset for threshold calculation
before_count = 200 # Number of samples to check before a potential transition
after_count = 200 # Number of samples to check after a potential transition
min_below_ratio = 0.8 # Minimum ratio of samples that must be below threshold in the before region
min_above_ratio = 0.8 # Minimum ratio of samples that must be above threshold in the after region

plt.ion()

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from matplotlib.widgets import Cursor

def robust_find_peak_start(signal, threshold_idx, window_size=5, slope_threshold=0.5):
    """Find the true start of peak using robust slope analysis"""
    # Calculate smoothed derivatives using a sliding window
    derivatives = []
    for i in range(window_size, threshold_idx):
        # Compute average slope in window
        avg_slope = (signal[i] - signal[i-window_size]) / window_size
        derivatives.append((i, avg_slope))
    
    # Work backwards from threshold crossing
    for i in range(threshold_idx-window_size, window_size, -1):
        cur_slope = (signal[i] - signal[i-window_size]) / window_size
        # Detect significant slope change
        if cur_slope < slope_threshold:
            return i
    
    return max(0, threshold_idx - 100)  # Fallback if no suitable point found

def comprehensive_peak_start(signal, threshold_idx, window_size=5, noise_level=None):
    """Robust peak start detection using multiple features
    
    Args:
        signal: array-like, input signal
        threshold_idx: int, index where signal crosses threshold
        window_size: int, window size for slope calculation
        noise_level: float, estimated noise level (auto-calculated if None)
        
    Returns:
        int: index of detected peak start
    """
    if noise_level is None:
        # Estimate noise level from signal baseline
        baseline = signal[:min(100, threshold_idx//2)]
        noise_level = np.std(baseline) * 2
    
    # Methods array to hold candidate points
    candidates = []
    
    # Method 1: Slope-based detection
    # Start from threshold crossing and move backwards
    for i in range(threshold_idx-window_size, window_size, -1):
        # Calculate slope over window
        avg_slope = (signal[i] - signal[i-window_size]) / window_size
        if abs(avg_slope) < 0.5:  # Detect significant slope change
            candidates.append(i)
            break
    
    # Method 2: Noise-threshold based detection
    for i in range(threshold_idx-1, 0, -1):
        if signal[i] < noise_level:
            # Confirm by checking previous few points
            if np.mean(signal[max(0, i-5):i+1]) < noise_level:
                candidates.append(i)
                break
    
    # Method 3: Curvature analysis (detect inflection point)
    if threshold_idx > 10:
        derivatives = np.diff(signal[:threshold_idx])
        if len(derivatives) > 2:
            second_derivatives = np.diff(derivatives)
            if len(second_derivatives) > 2:
                # Find where second derivative changes sign (inflection points)
                inflection_indices = np.where(np.diff(np.sign(second_derivatives)) != 0)[0]
                if len(inflection_indices) > 0:
                    # Get the last inflection point before threshold (closest to peak)
                    for idx in reversed(inflection_indices):
                        if idx < threshold_idx - 5:  # Ensure it's not too close to threshold
                            candidates.append(idx)
                            break
    
    # Failsafe: If no candidates found, use a fixed offset
    if not candidates:
        candidates.append(max(0, threshold_idx - 20))
    
    # Return median of candidate points for robustness
    return int(np.median(candidates))
    
    # Return median of candidate points for robustness
    return int(np.median(candidates))

def comprehensive_peak_end(signal, threshold_idx, window_size=20, noise_level=None):
    """Robust peak end detection using multiple features
    
    Args:
        signal: array-like, input signal
        threshold_idx: int, index where signal crosses threshold
        window_size: int, window size for slope calculation
        noise_level: float, estimated noise level (auto-calculated if None)
        
    Returns:
        int: index of detected peak end
    """
    if noise_level is None:
        # Estimate noise level from signal baseline (using data after the peak)
        end_idx = min(len(signal), threshold_idx + 100)
        baseline = signal[end_idx-50:end_idx] if end_idx > 50 else signal[-50:]
        noise_level = np.std(baseline) * 2
    
    # Methods array to hold candidate points
    candidates = []
    
    # Method 1: Slope-based detection
    for i in range(threshold_idx, len(signal)-window_size):
        # Calculate slope over window
        avg_slope = (signal[i+window_size] - signal[i]) / window_size
        if abs(avg_slope) < 0.5:  # Detect when slope becomes flat
            candidates.append(i)
            break
    
    # Method 2: Noise-threshold based detection
    for i in range(threshold_idx, len(signal)):
        if signal[i] < noise_level:
            # Confirm by checking next few points to avoid false detection
            end_check = min(i+5, len(signal))
            if np.mean(signal[i:end_check]) < noise_level:
                candidates.append(i)
                break
    
    # Method 3: Curvature analysis (detect inflection point)
    if len(signal) - threshold_idx > 10:
        derivatives = np.diff(signal[threshold_idx:])
        if len(derivatives) > 2:
            second_derivatives = np.diff(derivatives)
            if len(second_derivatives) > 2:
                # Find where second derivative changes sign (inflection points)
                raw_indices = np.where(np.diff(np.sign(second_derivatives)) != 0)[0]
                if len(raw_indices) > 0:
                    # Convert back to original signal indices
                    inflection_indices = raw_indices + threshold_idx + 2
                    # Get the first inflection point after threshold (closest to peak)
                    for idx in inflection_indices:
                        if idx > threshold_idx + 5:  # Ensure it's not too close to threshold
                            candidates.append(idx)
                            break
    
    # Failsafe: If no candidates found, use a fixed offset
    if not candidates:
        candidates.append(min(len(signal)-1, threshold_idx + 20))
    
    # Return median of candidate points for robustness
    return int(np.median(candidates))

def detect_peak_boundaries(signal, s_idx, e_idx, threshold, visualize=True):

    
    initial_start = s_idx
    initial_end = e_idx
    
    # Apply robust start detection
    refined_start = comprehensive_peak_start(signal, initial_start)
    
    # Apply robust end detection (similar to start, but forward direction)
    refined_end = comprehensive_peak_end(signal, initial_end)
    
    
    # Generate visualization if requested
    if visualize:
        plt.figure(figsize=(12, 7))
        time = np.arange(len(signal))
        
        # Plot original signal
        plt.plot(time, signal, 'purple', label='Magnitude')
        
        # Plot threshold and crossings
        plt.axhline(y=threshold, color='green', linestyle='--', 
                    label=f'Threshold ({threshold:.2f})')
        plt.axvline(x=initial_start, color='orange', linestyle='-', 
                    label='Initial Motion Start')
        plt.axvline(x=initial_end, color='orange', linestyle='-', 
                    label='Initial Motion End')
        
        # Plot refined boundaries
        plt.axvline(x=refined_start, color='blue', linestyle='--', 
                    label='Refined Start')
        plt.axvline(x=refined_end, color='red', linestyle='--', 
                    label='Refined End')
        
        
        plt.title('Signal Magnitude Analysis with Advanced Peak Detection')
        plt.xlabel('Time (s)')
        plt.ylabel('Composite Magnitude')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    return refined_start, refined_end


def check_real_motion(magnitude, threshold_indices, threshold=None, 
                     before_count=before_count, after_count=after_count,
                     min_below_ratio=min_below_ratio, min_above_ratio=min_above_ratio):
    """
    Check for real motion by identifying clean transitions in the magnitude signal.
    
    Args:
        magnitude: The magnitude signal array
        threshold_indices: Indices where magnitude exceeds threshold (used as fallback)
        threshold: The threshold value for magnitude comparison (if None, will be estimated)
        before_count: Number of samples to check before a potential transition
        after_count: Number of samples to check after a potential transition
        min_below_ratio: Minimum ratio of samples that must be below threshold in the before region
        min_above_ratio: Minimum ratio of samples that must be above threshold in the after region
    
    Returns:
        start_idx, end_idx: The start and end indices of the real motion
    """
    # Convert threshold_indices to a list to avoid boolean evaluation errors
    threshold_indices_list = list(threshold_indices)
    
    if len(threshold_indices_list) == 0:
        return None, None
    
    # If threshold not provided, estimate it from the threshold crossings
    if threshold is None:
        threshold = min(magnitude[i] for i in threshold_indices_list)
    
    # Parameter validation
    before_count = max(1, min(before_count, len(magnitude) - 1))
    after_count = max(1, min(after_count, len(magnitude) - 1))
    
    # Calculate minimum counts to satisfy ratio requirements
    min_below_count = int(before_count * min_below_ratio)
    min_above_count = int(after_count * min_above_ratio)
    
    # Find start index - looking for transition from below threshold to above threshold
    start_idx = None
    # Scan through the signal, not just the threshold indices
    for idx in range(before_count, len(magnitude) - after_count):
        # Skip if not near a threshold crossing to speed up processing
        if idx not in threshold_indices_list and idx-1 not in threshold_indices_list and idx+1 not in threshold_indices_list:
            continue
            
        # Count samples below threshold before this point
        below_count_before = sum(1 for i in range(idx - before_count, idx) if magnitude[i] <= threshold)
        
        # Count samples above threshold after this point (including current)
        above_count_after = sum(1 for i in range(idx, idx + after_count) if magnitude[i] > threshold)
        
        # Check if this point represents a clean transition
        if below_count_before >= min_below_count and above_count_after >= min_above_count:
            start_idx = idx
            break
    
    # Find end index - looking for transition from above threshold to below threshold
    end_idx = None
    # Scan backward through the signal
    for idx in range(len(magnitude) - after_count - 1, before_count - 1, -1):
        # Skip if not near a threshold crossing
        if idx not in threshold_indices_list and idx-1 not in threshold_indices_list and idx+1 not in threshold_indices_list:
            continue
            
        # Count samples above threshold before this point (including current)
        above_count_before = sum(1 for i in range(idx - before_count + 1, idx + 1) if magnitude[i] > threshold)
        
        # Count samples below threshold after this point
        below_count_after = sum(1 for i in range(idx + 1, idx + after_count + 1) if magnitude[i] <= threshold)
        
        # Check if this point represents a clean transition
        if above_count_before >= min_below_count and below_count_after >= min_above_count:
            end_idx = idx
            break
    
    # If no suitable points found, use first/last threshold indices as fallback
    if start_idx is None and len(threshold_indices_list) > 0:
        start_idx = threshold_indices_list[0]
    if end_idx is None and len(threshold_indices_list) > 0:
        end_idx = threshold_indices_list[-1]
    
    return start_idx, end_idx



import numpy as np
import matplotlib.pyplot as plt

def segmentation_and_bias(gyro_data, acc_data, orientation_data, frequencies=None, selectManually=False, plot_flag=True):

    print(gyro_data.head())
    print(acc_data.head())
    print(orientation_data.head())

    if gyro_data.empty or acc_data.empty or orientation_data.empty:
        print("Error: The data is empty.")
        return

    if frequencies is None:
        print("\nFrequencies not found, please input them:")
        frequencies = sensors_frequencies()

    # ------------------------- COMPUTE TIME VECTORS -------------------
    time_gyro = np.arange(len(gyro_data)) / frequencies[0]
    time_acc = np.arange(len(acc_data)) / frequencies[1]
    time_orientation = np.arange(len(orientation_data)) / frequencies[2]
    
    if plot_flag:
        # FIGURE 1: Initial Analysis (2x2 grid)
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # Raw Data Plot
        for column in gyro_data.columns:
            axs[0, 0].plot(time_gyro, gyro_data[column], label=column)
        axs[0, 0].set_title("Raw Gyro Data")
        axs[0, 0].set(xlabel="Time (s)", ylabel="Angular Velocity (rad/s)")
        axs[0, 0].legend().set_visible(True)
        axs[0, 0].grid(True)

    # ------------------------- BIAS REMOVAL ----------------------------
    print("Removing bias...")
    non_zero_index = (gyro_data != 0).any(axis=1).idxmax()
    sample_size = bias_average_window
    
    if non_zero_index + sample_size <= len(gyro_data):
        means = gyro_data.iloc[non_zero_index:non_zero_index + sample_size].mean()
        print("Initial mean values:", means)
    else:
        print("Not enough data after first non-zero value")

    gyro_data_centered = gyro_data - means
    
    if plot_flag:
        # Bias-Removed Plot
        for column in gyro_data_centered.columns:
            axs[0, 1].plot(time_gyro, gyro_data_centered[column], label=column)
        axs[0, 1].set_title("Bias-Removed Data")
        axs[0, 1].set(xlabel="Time (s)", ylabel="Angular Velocity (rad/s)")
        axs[0, 1].legend().set_visible(True)
        axs[0, 1].grid(True)

    # Trim and Process Data
    gyro_data_trimmed = gyro_data_centered.iloc[non_zero_index:].reset_index(drop=True)
    
    if plot_flag:
        # Trimmed Data Plot
        for column in gyro_data_trimmed.columns:
            axs[1, 0].plot(time_gyro[non_zero_index:], gyro_data_trimmed[column], label=column)
        axs[1, 0].set_title("Trimmed Sensor Data")
        axs[1, 0].set(xlabel="Time (s)", ylabel="Angular Velocity (rad/s)")
        axs[1, 0].legend().set_visible(True)
        axs[1, 0].grid(True)

    # ------------------------- MAGNITUDE ANALYSIS ---------------------------
    print("Calculating magnitude...")
    raw_magnitude = np.sqrt(gyro_data_trimmed.iloc[:,0]**2 + 
                       gyro_data_trimmed.iloc[:,1]**2 + 
                       gyro_data_trimmed.iloc[:,2]**2)
   
    # Filtering of the Magntiude
    from scipy.signal import butter
    from scipy.signal import filtfilt
    sampling_rate = frequencies[0]  # Hz
    nyquist = 0.5 * sampling_rate
    cutoff = 5  # Hz
    normal_cutoff = cutoff / nyquist
    b, a = butter(4, normal_cutoff, btype='low', analog=False)
    magnitude = filtfilt(b, a, raw_magnitude)

    gyro_data_trimmed['magnitude'] = magnitude
    
    mean_magnitude = gyro_data_trimmed['magnitude'].iloc[:sample_size].mean()
    threshold = mean_magnitude + offset
    threshold_indices = np.where(magnitude > threshold)[0]
    
   
    # Select start and end indices based on threshold crossings or manually
    if selectManually == True:
            # Call the function to select indices
        start_idx, end_idx = select_indices(magnitude, time_gyro[non_zero_index:], threshold, frequencies)
        start_idx = start_idx - non_zero_index
        end_idx = end_idx - non_zero_index
    else:
        s_idx, e_idx = check_real_motion(magnitude, threshold_indices, threshold=threshold)
        start_idx, end_idx = detect_peak_boundaries(magnitude, s_idx, e_idx, threshold, visualize=plot_flag)
                                        
    if plot_flag:
        # Magnitude Analysis Plot
        axs[1,1].plot(time_gyro[non_zero_index:], magnitude, color='purple', label='Magnitude')
        axs[1,1].axhline(threshold, color='green', linestyle='--', label=f'Threshold ({threshold:.2f})')
        if start_idx is not None:
            axs[1,1].axvline(time_gyro[start_idx + non_zero_index], color='orange', label='Motion Start')
            axs[1,1].axvline(time_gyro[end_idx + non_zero_index], color='orange', label='Motion End')
        axs[1,1].set_title("Signal Magnitude Analysis")
        axs[1,1].set(xlabel="Time (s)", ylabel="Composite Magnitude")
        axs[1,1].legend()
        axs[1,1].grid(True)

        plt.tight_layout()
        plt.draw()

    # gyro_data_trimmed.iloc[:, :3] = gyro_data_trimmed.iloc[:, :3] + means
    # ------------------------- SEGMENTATION ---------------------------
    # Filter the data before training
    nyquist = 0.5 * sampling_rate
    cutoff = 50  # Hz
    normal_cutoff = cutoff / nyquist
    b, a = butter(4, normal_cutoff, btype='low', analog=False)
    print("Segmenting data...")
    gyro_data_segmented = gyro_data_trimmed.iloc[start_idx:end_idx].reset_index(drop=True)
    # for column in gyro_data_segmented.columns[:3]:  # Exclude magnitude column
    #     gyro_data_segmented[column] = filtfilt(b, a, gyro_data_segmented[column])
    time_gyro_segmented = time_gyro[non_zero_index + start_idx:non_zero_index + end_idx]

    # Process acceleration data
    start_idx_acc = int(start_idx * frequencies[1] / frequencies[0])
    end_idx_acc = int(end_idx * frequencies[1] / frequencies[0])
    non_zero_index_acc = int(non_zero_index * frequencies[1] / frequencies[0])
    acc_data_trimmed = acc_data.iloc[non_zero_index_acc:].reset_index(drop=True)
    acc_data_segmented = acc_data_trimmed.iloc[start_idx_acc:end_idx_acc].reset_index(drop=True)
    # for column in acc_data_segmented.columns:
    #     acc_data_segmented[column] = filtfilt(b, a, acc_data_segmented[column])
    time_acc_segmented = time_acc[non_zero_index_acc + start_idx_acc:non_zero_index_acc + end_idx_acc]

    # Process orientation data
    start_idx_or = int(start_idx * frequencies[2] / frequencies[0])
    end_idx_or = int(end_idx * frequencies[2] / frequencies[0])
    non_zero_index_or = int(non_zero_index * frequencies[2] / frequencies[0])
    or_data_trimmed = orientation_data.iloc[non_zero_index_or:].reset_index(drop=True)
    or_data_segmented = or_data_trimmed.iloc[start_idx_or:end_idx_or].reset_index(drop=True)
    time_orientation_segmented = time_orientation[non_zero_index_or + start_idx_or:non_zero_index_or + end_idx_or]

    if plot_flag:
        # FIGURE 2: Motion Region Highlights (3x1 grid)
        fig2, axs2 = plt.subplots(3, 1, figsize=(15, 12))
        
        # Gyro Highlight
        for column in gyro_data_trimmed.columns[:3]:  # Exclude magnitude column
            axs2[0].plot(time_gyro[non_zero_index:], gyro_data_trimmed[column], label=column)
        if start_idx is not None:
            axs2[0].axvline(time_gyro[start_idx + non_zero_index], color='lime', linewidth=2, label='Motion Start')
            axs2[0].axvline(time_gyro[end_idx + non_zero_index], color='red', linewidth=2, label='Motion End')
        axs2[0].set_title("Gyro Data with Motion Region")
        axs2[0].set(xlabel="Time (s)", ylabel="Angular Velocity (rad/s)")
        axs2[0].legend()
        axs2[0].grid(True)

        # Acceleration Highlight
        for column in acc_data_trimmed.columns:
            axs2[1].plot(time_acc[non_zero_index_acc:], acc_data_trimmed[column], label=column)
        if start_idx is not None:
            axs2[1].axvline(time_acc[non_zero_index_acc + start_idx_acc], color='lime', linewidth=2)
            axs2[1].axvline(time_acc[non_zero_index_acc + end_idx_acc], color='red', linewidth=2)
        axs2[1].set_title("Acceleration Data with Motion Region")
        axs2[1].set(xlabel="Time (s)", ylabel="Acceleration (m/s²)")
        axs2[1].legend()
        axs2[1].grid(True)

        # Orientation Highlight
        for column in or_data_trimmed.columns:
            axs2[2].plot(time_orientation[non_zero_index_or:], or_data_trimmed[column], label=column)
        if start_idx is not None:
            axs2[2].axvline(time_orientation[non_zero_index_or + start_idx_or], color='lime', linewidth=2)
            axs2[2].axvline(time_orientation[non_zero_index_or + end_idx_or], color='red', linewidth=2)
        axs2[2].set_title("Orientation Data with Motion Region")
        axs2[2].set(xlabel="Time (s)", ylabel="Orientation (rad)")
        axs2[2].legend()
        axs2[2].grid(True)

        plt.tight_layout()
        plt.show(block=True)

    return (gyro_data_segmented, acc_data_segmented, or_data_segmented,
            time_gyro_segmented, time_acc_segmented, time_orientation_segmented)


def sensors_frequencies():
    # Function to get a valid frequency input from the user
    def get_frequency_input(sensor_name):
        while True:
            try:
                frequency = float(input(f"Enter the frequency for {sensor_name} (in Hz): "))
                if frequency > 0:
                    return frequency
                else:
                    print("Please enter a positive number.")
            except ValueError:
                print("Invalid input. Please enter a numeric value.")

    # Get frequencies for each sensor
    gyro_freq = get_frequency_input("GYRO")
    acc_freq = get_frequency_input("ACC")
    orientation_freq = get_frequency_input("ORIENTATION")

    # Create a vector (list) with the three frequencies
    frequency_vector = [gyro_freq, acc_freq, orientation_freq]

    # Print the resulting vector
    print("\nFrequency vector:", frequency_vector)

    # Return the vector
    return frequency_vector




# # Guard to prevent execution when imported
# if __name__ == "__main__":
#     segmentation_and_bias()

# Interactive selection of start_idx and end_idx
# Interactive selection of start_idx and end_idx
def select_indices(signal, time_vector, threshold, frequencies):
    indices = []

    def onclick(event):
        if event.inaxes:
            idx = int(event.xdata * frequencies[0])  # Convert time to index
            indices.append(idx)
            print(f"Selected index: {idx}")
            if len(indices) == 2:
                plt.close()

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(time_vector, signal, label='Magnitude')
    ax.axhline(threshold, color='green', linestyle='--', label=f'Threshold ({threshold:.2f})')
    ax.set_title("Select Start and End Indices")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Composite Magnitude")
    ax.legend()
    ax.grid(True)

    # Add a cursor that moves with the mouse
    cursor = Cursor(ax, useblit=True, color='red', linewidth=1)

    # Connect the click event
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show(block=True)

    if len(indices) < 2:
        print("Error: Both start and end indices must be selected.")
        return None, None

    return indices[0], indices[1]



