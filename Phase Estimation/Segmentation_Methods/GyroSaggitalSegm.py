import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

def GyroSaggitalSegm(gyro_data, gyro_freq):
    # Extract Sagittal plane gyro (GyroZ)
    gyro_saggital = gyro_data.iloc[:, 0].values

    
    
    # Filter parameters
    cutoff_freq = 0.5  # Hz
    nyquist = 0.5 * 200
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(4, normal_cutoff, btype='low')
    filtered_gyro = filtfilt(b, a, gyro_saggital)

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
    for i in range(0, len(positive_deriv_crossings) - 1):
        start = positive_deriv_crossings[i]
        # middle = positive_deriv_crossings[i + 1]
        end = positive_deriv_crossings[i + 1]
        segments.append((start, end))

    # Visualization as subplots
    fig, axs = plt.subplots(3, 1, figsize=(12, 18))
    
    # Plot filtered signal with segments
    axs[0].plot(filtered_gyro, label='Filtered Signal')
    colors = plt.cm.tab10.colors
    for idx, (start, end) in enumerate(segments):
        axs[0].axvspan(start, end, color=colors[idx % 10], alpha=0.3)
        axs[0].axvline(start, color='k', linestyle='--', alpha=0.5)
    axs[0].set_title('Signal Segmentation at Positive Zero-Crossings')
    axs[0].set_xlabel('Samples')
    axs[0].set_ylabel('Angular Velocity (rad/s)')
    axs[0].legend()
    
    # Plot derivative of the absolute value with segments
    axs[1].plot(abs_filtered_gyro_derivative, label='Derivative of Absolute Value')
    for idx, (start, end) in enumerate(segments):
        axs[1].axvspan(start, end, color=colors[idx % 10], alpha=0.3)
        axs[1].axvline(start, color='k', linestyle='--', alpha=0.5)
    axs[1].set_title('Derivative of Absolute Value with Segments')
    axs[1].set_xlabel('Samples')
    axs[1].set_ylabel('Derivative Value')
    axs[1].legend()
    
    # Plot original gyro_data (all three columns) with segments
    axs[2].plot(gyro_data.iloc[:, 0], label='GyroX')
    axs[2].plot(gyro_data.iloc[:, 1], label='GyroY')
    axs[2].plot(gyro_data.iloc[:, 2], label='GyroZ')
    for idx, (start, end) in enumerate(segments):
        axs[2].axvspan(start, end, color=colors[idx % 10], alpha=0.3)
        axs[2].axvline(start, color='k', linestyle='--', alpha=0.5)
    axs[2].set_title('Original Gyro Data with Segments')
    axs[2].set_xlabel('Samples')
    axs[2].set_ylabel('Angular Velocity (rad/s)')
    axs[2].legend()
    
    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()


    return segments, abs_filtered_gyro_derivative
