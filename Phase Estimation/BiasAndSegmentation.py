import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog, Tk
from scipy import interpolate
import GyroMagnitudeSegmentation
import AREDSegmentation
import SHOE


# ----------- HYPERPARAMETERS -----------------
# Hyperparameters for bias removal
bias_average_window = 1000 # Number of samples to average for bias removal
plot_flag_gyro = False  # Flag to plot gyro data

plt.ion()  # Enable interactive mode for plotting
def segmentation_and_bias(gyro_data, acc_data, orientation_data, segment_choice, timestamp, frequencies=None, plot_flag=True):

    print(gyro_data.head())
    print(acc_data.head())
    print(orientation_data.head())

    if gyro_data.empty or acc_data.empty or orientation_data.empty:
        print("Error: The data is empty.")
        return

    if frequencies is None:
        print("\nFrequencies not found, please input them:")
        frequencies = sensors_frequencies()

    # Time vectors for each sensor:
    time_gyro = np.arange(len(gyro_data)) / frequencies[0]
    time_acc = np.arange(len(acc_data)) / frequencies[1]
    time_orientation = np.arange(len(orientation_data)) / frequencies[2]
    
    # Removing BIAS from gyro data:
    print("Removing bias...")
    non_zero_index = (gyro_data != 0).any(axis=1).idxmax()
    sample_size = bias_average_window
    if non_zero_index + sample_size <= len(gyro_data):
        means = gyro_data.iloc[non_zero_index:non_zero_index + sample_size].mean()
        print("Initial mean values:", means)
    else:
        print("Not enough data after first non-zero value")
    gyro_data_centered = gyro_data - means

    # Removing the zero values from the gyro data:
    gyro_data_trimmed = gyro_data_centered.iloc[non_zero_index:].reset_index(drop=True)
    non_zero_index_acc = int(non_zero_index * frequencies[1] / frequencies[0])
    acc_data_trimmed = acc_data.iloc[non_zero_index_acc:].reset_index(drop=True)
    non_zero_index_or = int(non_zero_index * frequencies[2] / frequencies[0])
    or_data_trimmed = orientation_data.iloc[non_zero_index_or:].reset_index(drop=True)

    # Computing Magnitude:
    print("Calculating magnitude...")
    raw_magnitude = np.sqrt(gyro_data_trimmed.iloc[:,0]**2 + 
                       gyro_data_trimmed.iloc[:,1]**2 + 
                       gyro_data_trimmed.iloc[:,2]**2)

    if plot_flag_gyro:
        # FIGURE 1: Gyro processed data
        fig, axs = plt.subplots(3, 1, figsize=(15, 10))      
        # Raw Data Plot
        for column in gyro_data.columns:
            axs[0].plot(time_gyro, gyro_data[column], label=column)
        axs[0].set_title("Raw Gyro Data")
        axs[0].set(xlabel="Time (s)", ylabel="Angular Velocity (rad/s)")
        axs[0].legend().set_visible(True)
        axs[0].grid(True)
        # Bias-Removed Plot
        for column in gyro_data_centered.columns:
            axs[1].plot(time_gyro, gyro_data_centered[column], label=column)
        axs[1].set_title("Bias-Removed Data")
        axs[1].set(xlabel="Time (s)", ylabel="Angular Velocity (rad/s)")
        axs[1].legend().set_visible(True)
        axs[1].grid(True)
        # Trimmed Data Plot
        for column in gyro_data_trimmed.columns:
            axs[2].plot(time_gyro[non_zero_index:], gyro_data_trimmed[column], label=column)
        axs[2].set_title("Trimmed Sensor Data")
        axs[2].set(xlabel="Time (s)", ylabel="Angular Velocity (rad/s)")
        axs[2].legend().set_visible(True)
        axs[2].grid(True)

    # Apply the chosen segmetation method:
    if segment_choice == '1':
        start_idx, end_idx = GyroMagnitudeSegmentation.GyroMagnitudeSegmentation(frequencies, raw_magnitude, gyro_data_trimmed, time_gyro, non_zero_index, threshold=5, plot_flag=plot_flag)
    elif segment_choice == '2':
        start_idx, end_idx = AREDSegmentation.AREDSegmentation(raw_magnitude, timestamp, plot_flag=plot_flag)
    elif segment_choice == '3':
        start_idx, end_idx = SHOE.motion_segmenter(acc_data_trimmed, gyro_data_trimmed, frequency=519, visualize=True)
   

    gyro_data_segmented = gyro_data_trimmed.iloc[start_idx:end_idx].reset_index(drop=True)
    time_gyro_segmented = time_gyro[non_zero_index + start_idx:non_zero_index + end_idx]

    # Process acceleration data
    start_idx_acc = int(start_idx * frequencies[1] / frequencies[0])
    end_idx_acc = int(end_idx * frequencies[1] / frequencies[0])
    acc_data_segmented = acc_data_trimmed.iloc[start_idx_acc:end_idx_acc].reset_index(drop=True)
    time_acc_segmented = time_acc[non_zero_index_acc + start_idx_acc:non_zero_index_acc + end_idx_acc]

    # Process orientation data
    start_idx_or = int(start_idx * frequencies[2] / frequencies[0])
    end_idx_or = int(end_idx * frequencies[2] / frequencies[0])
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