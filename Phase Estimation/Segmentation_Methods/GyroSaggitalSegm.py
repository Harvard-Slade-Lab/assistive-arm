import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

def GyroSaggitalSegm(gyro_data, gyro_freq):
    # Extract Sagittal plane gyro (GyroZ)
    gyro_saggital = gyro_data.iloc[:, 2].values

    # Downsample to 200Hz
    downsample_factor = int(gyro_freq / 200)
    downsampled_gyro = gyro_saggital[::downsample_factor]
    
    # Filter parameters
    cutoff_freq = 3  # Hz
    nyquist = 0.5 * 200
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(4, normal_cutoff, btype='low')
    filtered_gyro = filtfilt(b, a, downsampled_gyro)

    # Absolute value of the filtered signal
    abs_filtered_gyro = np.abs(filtered_gyro)   
    # Derivative of the absolute value
    abs_filtered_gyro_derivative = np.gradient(abs_filtered_gyro)
  


    # Improved zero-crossing detection with hysteresis
    threshold = 0.1  # Noise threshold (adjust based on signal)
    positive_deriv_crossings = []
    
    # Calculate first derivative
    gradient = np.gradient(filtered_gyro)
    
    # Detect zero-crossings with positive derivative
    for i in range(1, len(filtered_gyro)):

        prev = filtered_gyro[i-1]
        current = filtered_gyro[i]
        
        # Zero crossing with hysteresis
        if (prev < -threshold and current > threshold) or \
           (abs(prev) < threshold and current > threshold and gradient[i] > 0):
            
            # Verify positive derivative
            if gradient[i] > 0:
                positive_deriv_crossings.append(i)

    # Create segments between consecutive positive crossings
    segments = []
    for i in range(len(positive_deriv_crossings)-1):
        start = positive_deriv_crossings[i]
        end = positive_deriv_crossings[i+1]
        segments.append((start, end))

    # Visualization as subplots
    fig, axs = plt.subplots(2, 1, figsize=(12, 12))
    # Plot filtered signal with segments
    axs[0].plot(filtered_gyro, label='Filtered Signal')
    colors = plt.cm.tab10.colors
    for idx, (start, end) in enumerate(zip(positive_deriv_crossings[:-1], 
                                           positive_deriv_crossings[1:])):
        axs[0].axvspan(start, end, color=colors[idx % 10], alpha=0.3)
        axs[0].axvline(start, color='k', linestyle='--', alpha=0.5)
    axs[0].set_title('Signal Segmentation at Positive Zero-Crossings')
    axs[0].set_xlabel('Samples')
    axs[0].set_ylabel('Angular Velocity (rad/s)')
    axs[0].legend()
    # Plot derivative of the absolute value with segments
    axs[1].plot(abs_filtered_gyro_derivative, label='Derivative of Absolute Value')
    for idx, (start, end) in enumerate(zip(positive_deriv_crossings[:-1], 
                                           positive_deriv_crossings[1:])):
        axs[1].axvspan(start, end, color=colors[idx % 10], alpha=0.3)
        axs[1].axvline(start, color='k', linestyle='--', alpha=0.5)
    axs[1].set_title('Derivative of Absolute Value with Segments')
    axs[1].set_xlabel('Samples')
    axs[1].set_ylabel('Derivative Value')
    axs[1].legend()
    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()


    return segments, abs_filtered_gyro_derivative
