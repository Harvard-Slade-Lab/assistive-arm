import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.signal import butter
from scipy.signal import filtfilt
from scipy.signal import find_peaks
import Interpolation

plot_flag_gyro = False  # Flag to plot gyro data

def motion_segmenter(gyro_data, acc_data, orientation_data, timestamp, test_flag = False, frequencies=None, plot_flag=True):

    X1 = []
    Y1 = []
    segment_lengths1 = []
    

    print(gyro_data.head())
    print(acc_data.head())
    print(orientation_data.head())

    if gyro_data.empty or acc_data.empty or orientation_data.empty:
        print("Error: The data is empty.")
        return

    # Time vectors for each sensor:
    time_gyro = np.arange(len(gyro_data)) / frequencies[0]
    time_acc = np.arange(len(acc_data)) / frequencies[1]
    time_orientation = np.arange(len(orientation_data)) / frequencies[2]


    # Removing BIAS from gyro data:
    print("Removing bias...")
    non_zero_index = 0

    # Removing the zero values from the gyro data:
    gyro_data_trimmed = gyro_data.iloc[non_zero_index:].reset_index(drop=True)
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
        for column in gyro_data.columns:
            axs[1].plot(time_gyro, gyro_data[column], label=column)
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

    # Magitude Filtering:
    sampling_rate = frequencies[0]  # Hz
    nyquist = 0.5 * sampling_rate
    cutoff = 5  # Hz
    normal_cutoff = cutoff / nyquist
    b, a = butter(4, normal_cutoff, btype='low', analog=False)
    magnitude = filtfilt(b, a, raw_magnitude)

    # Storing the magnitude in the gyro_data dataframe:
    gyro_data_trimmed['magnitude'] = magnitude

    # After calculating raw_magnitude in your function:
    time_trimmed = np.arange(len(gyro_data_trimmed)) / frequencies[0]

    # Segment the gait cycles
    segments, peaks = segment_gait_cycles(magnitude, time_trimmed)

    # Store individual step data
    step_data = []
    for i, (start, end) in enumerate(segments):
        step_data.append({
            'step_number': i+1,
            'gyro': gyro_data_trimmed.iloc[start:end],
            'acc': acc_data_trimmed.iloc[int(start*frequencies[1]/frequencies[0]):int(end*frequencies[1]/frequencies[0])],
            'orientation': or_data_trimmed.iloc[int(start*frequencies[2]/frequencies[0]):int(end*frequencies[2]/frequencies[0])],
            'magnitude': magnitude[start:end],
            'duration': (end-start)/frequencies[0]
        })
    
    timestamp_matrices = {}
    
    # Print head of step data for debugging
    for step in step_data:
        # Apply interpolation
        gyro_processed = step['gyro']
        acc_processed = step['acc']
        or_processed = step['orientation']

        gyro_interp, acc_interp, or_interp = Interpolation.interpolate_and_visualize(
            gyro_processed, acc_processed, or_processed, 
            frequencies, plot_flag=False
        )
        
        # Concatenate features for X matrix
        features = np.concatenate([acc_interp.values, gyro_interp.values, or_interp.values], axis=1)

        if test_flag == True:
            timestamp_matrices[step['step_number']] = features
            
        else:
            X1.append(features)

            dataset_length = len(features)
            segment_lengths1.append(dataset_length)

            # Create Y matrix segment
            dataset_length = len(features)
            y = np.linspace(0, 1, dataset_length)
            Y1.append(y)
            
            print(f"Step {step['step_number']}:\n", step['gyro'].head())
            print(f"Step {step['step_number']} Acc:\n", step['acc'].head())
            print(f"Step {step['step_number']} Orientation:\n", step['orientation'].head())
    
    if test_flag == True:
        print(f"Detected {len(step_data)} steps")
        acc_cols = [f"ACC_{col}" for col in acc_interp.columns]
        gyro_cols = [f"GYRO_{col}" for col in gyro_interp.columns]
        or_cols = [f"OR_{col}" for col in or_interp.columns]
        feature_names = acc_cols + gyro_cols + or_cols
        return timestamp_matrices, feature_names # In case of testing, returns the segments singularly
    else:
        print(f"Detected {len(step_data)} steps")
        acc_cols = [f"ACC_{col}" for col in acc_interp.columns]
        gyro_cols = [f"GYRO_{col}" for col in gyro_interp.columns]
        or_cols = [f"OR_{col}" for col in or_interp.columns]
        feature_names = acc_cols + gyro_cols + or_cols
        return X1, Y1, segment_lengths1, feature_names  # In case of training, returns the segments all together in a matrix format

def segment_gait_cycles(magnitude_signal, time_vector, plot_results=True):
    # Find peaks with constraints
    peaks, _ = find_peaks(magnitude_signal, 
                          height=150,          # Minimum peak height (adjust based on your data)
                          distance=300,         # Minimum samples between peaks
                          prominence=50)       # Minimum peak prominence
    # Find valleys (local minima) preceding the peaks
    # valleys, _ = find_peaks(-magnitude_signal, 
    #                         distance=300)       # Minimum samples between valleys

    # # Filter valleys to ensure they precede the detected peaks
    # filtered_valleys = []
    # for peak in peaks:
    #     preceding_valleys = [valley for valley in valleys if valley < peak]
    #     if preceding_valleys:
    #         filtered_valleys.append(preceding_valleys[-1])  # Take the closest preceding valley

    # valleys = np.array(filtered_valleys)
    
    # Create segments based on peaks
    segments = []
    for i in range(len(peaks)-1):
        start_idx = peaks[i]
        end_idx = peaks[i+1]
        segments.append((start_idx, end_idx))
    
    # Plot results if requested
    if plot_results:
        plt.figure(figsize=(12, 6))
        plt.plot(time_vector, magnitude_signal, 'b-', label='Magnitude')
        plt.plot(time_vector[peaks], magnitude_signal[peaks], 'ro', label='Detected Peaks')
        
        # Highlight segments
        for i, (start, end) in enumerate(segments):
            plt.axvspan(time_vector[start], time_vector[end], 
                      alpha=0.2, color='g', label='Segment' if i==0 else None)
        
        plt.legend()
        plt.title('Gait Cycle Segmentation')
        plt.xlabel('Time (s)')
        plt.ylabel('Magnitude')
        plt.grid(True)
        plt.show()
    
    return segments, peaks
