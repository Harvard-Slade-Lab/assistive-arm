import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog, Tk

# ----------- HYPERPARAMETERS -----------------
# Hyperparameters for bias removal
bias_average_window = 1000 # Number of samples to average for bias removal

# Hyperparameters for segmentation
offset = 7.0 # Offset for threshold calculation
before_count = 200 # Number of samples to check before a potential transition
after_count = 200 # Number of samples to check after a potential transition
min_below_ratio = 0.8 # Minimum ratio of samples that must be below threshold in the before region
min_above_ratio = 0.8 # Minimum ratio of samples that must be above threshold in the after region


def select_file():
    """Open a file dialog to select a CSV file"""
    root = Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(
        title="Select CSV File",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    root.destroy()
    return file_path

def load_csv(file_path):
    """Load and return CSV data as DataFrame"""
    try:
        data = pd.read_csv(file_path)
        print("CSV file loaded successfully.")
        return data
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
    except pd.errors.EmptyDataError:
        print("Error: The file is empty.")
    except Exception as e:
        print(f"An error occurred: {e}")

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



def segmentation_and_bias(file_path=None):
    # If no file path is provided, prompt the user to select one
    if file_path is None:
        file_path = select_file()
    
    data = load_csv(file_path)
    
    if data is None:
        return
    
    print(data.head())
    
    # Create figure with 4 subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Raw data plot
    data.plot(ax=axs[0,0])
    axs[0,0].set_title("Raw Sensor Data")
    axs[0,0].set_ylabel("Values")
    axs[0,0].grid(True)
 #---------------------------- BIAS REMOVAL ----------------------------
    # Bias calculation
    non_zero_index = (data != 0).any(axis=1).idxmax()
    sample_size = bias_average_window
    
    if non_zero_index + sample_size <= len(data):
        means = data.iloc[non_zero_index:non_zero_index + sample_size].mean()
        print("Initial mean values:")
        print(means)
    else:
        print("Not enough data after first non-zero value")

    # Centered data plot
    data_centered = data - means
    data_centered.plot(ax=axs[0,1])
    axs[0,1].set_title("Bias-Removed Data")
    axs[0,1].grid(True)

    # Trimmed data plot
    data_trimmed = data_centered.iloc[non_zero_index:].reset_index(drop=True)
    data_trimmed.plot(ax=axs[1,0])
    axs[1,0].set_title("Trimmed Sensor Data")
    axs[1,0].set_xlabel("Index")
    axs[1,0].set_ylabel("Values")
    axs[1,0].grid(True)
 #----------------------------- SEGMENTATION ---------------------------
    # Magnitude calculation
    magnitude = np.sqrt(data_trimmed.iloc[:,0]**2 + 
                       data_trimmed.iloc[:,1]**2 + 
                       data_trimmed.iloc[:,2]**2)
    data_trimmed['magnitude'] = magnitude
    
    # Threshold analysis
    mean_magnitude = data_trimmed['magnitude'].iloc[:sample_size].mean()
    threshold = mean_magnitude + offset
    threshold_indices = np.where(magnitude > threshold)[0]
    
    # Find real motion start and end
    start_idx, end_idx = check_real_motion(magnitude, threshold_indices, threshold=threshold)
                                        
    
    # Magnitude plot with threshold
    axs[1,1].plot(magnitude, color='purple', label='Magnitude')
    axs[1,1].axhline(threshold, color='green', linestyle='--', label=f'Threshold ({threshold:.2f})')
    if start_idx is not None:
        axs[1,1].axvline(x=start_idx, color='orange', linestyle='-', label='Motion Start')
        axs[1,1].axvline(x=end_idx, color='orange', linestyle='-', label='Motion End')
    axs[1,1].set_title("Signal Magnitude with Threshold")
    axs[1,1].set_xlabel("Index")
    axs[1,1].set_ylabel("Magnitude")
    axs[1,1].legend()
    axs[1,1].grid(True)

    plt.tight_layout()
    plt.show()

    # Combined vertical lines plot
    plt.figure(figsize=(12, 8))
    
    # Plot all three sensor signals
    for i in range(3):
        plt.plot(data_trimmed.iloc[:,i], label=data_trimmed.columns[i])
    
    
    # Plot threshold crossings
    if start_idx is not None:
        plt.axvline(x=start_idx, color='lime', linestyle='-', label='Motion Start')
        plt.axvline(x=end_idx, color='lime', linestyle='-', label='Motion End')
    
    plt.title("Movement Analysis: Zero Points (Red) and Active Regions (Green)")
    plt.xlabel("Index")
    plt.ylabel("Values")
    plt.legend(loc="upper right")
    plt.grid(True)
    
    # Create unified annotation
    annotation_text = []
    if start_idx is not None:
        annotation_text.append(f"Motion duration: {end_idx - start_idx} samples")
    
    if annotation_text:
        plt.text(0.72, 0.95, "\n".join(annotation_text),
                transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

    # Statistics output
    print("\nAdvanced Magnitude Analysis:")
    print(f"Baseline mean (first {sample_size} samples): {mean_magnitude:.4f}")
    print(f"Threshold value: {threshold:.4f}")
    print(f"Threshold crossings: {len(threshold_indices)}")
    print(f"Peak magnitude: {magnitude.max():.4f}")

    isolated_movement = data_trimmed.iloc[start_idx:end_idx]
    # Plot isolated movement regions
    if start_idx is not None:
        plt.figure(figsize=(12, 6))
        
        for i in range(3):
            plt.plot(data_trimmed.iloc[start_idx:end_idx,i], 
                    label=data_trimmed.columns[i])
        
        plt.title("Isolated Movement Region")
        plt.xlabel("Index (Relative)")
        plt.ylabel("Values")
        plt.legend()
        plt.grid(True)
        plt.show()
    return isolated_movement

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