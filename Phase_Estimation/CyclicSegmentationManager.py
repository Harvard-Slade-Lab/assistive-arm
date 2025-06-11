import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.signal import butter
from scipy.signal import filtfilt
from scipy.signal import find_peaks
from Phase_Estimation.Segmentation_Methods import GyroSaggitalSegm

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

    # Computing Magnitude:
    print("Calculating magnitude...")
    raw_magnitude = np.sqrt(gyro_data.iloc[:,0]**2 + 
                       gyro_data.iloc[:,1]**2 + 
                       gyro_data.iloc[:,2]**2)

    # Magnitude Filtering:
    sampling_rate = frequencies[0]  # Hz
    nyquist = 0.5 * sampling_rate
    cutoff = 5  # Hz
    normal_cutoff = cutoff / nyquist
    b, a = butter(4, normal_cutoff, btype='low', analog=False)
    magnitude = filtfilt(b, a, raw_magnitude)

    
    segments, abs_filtered_gyro_derivative = GyroSaggitalSegm.GyroSaggitalSegm(gyro_data, frequencies[0])

    abs_filtered_gyro_derivative = pd.DataFrame(abs_filtered_gyro_derivative, columns=['x'])  # Replace 'col1' with the actual column name

    # # Segment the gait cycles
    # segments = Cyclic_PeaksSegmentation.segment_gait_cycles(magnitude, time_gyro)

    # Store individual step data
    step_data = []
    for i, (start, end) in enumerate(segments):
        step_data.append({
            'step_number': i+1,
            'gyro': gyro_data.iloc[start:end],
            'acc': acc_data.iloc[int(start*frequencies[1]/frequencies[0]):int(end*frequencies[1]/frequencies[0])],
            'orientation': orientation_data.iloc[int(start*frequencies[2]/frequencies[0]):int(end*frequencies[2]/frequencies[0])],
            'magnitude': magnitude[start:end],
            'duration': (end-start)/frequencies[0],
            'absgyro': abs_filtered_gyro_derivative.iloc[start:end]
        })
        print(f"Step {i+1}: Start = {start}, End = {end}, ")

    # Downsample all gyro data but use only gyroz for plotting
    min_length = min(len(step['gyro']) for step in step_data)

    for step in step_data:
        # Downsample all gyro components to the smallest length
        downsampled_gyro = np.array([
            np.interp(
                np.linspace(0, len(step['gyro'].iloc[:, i]) - 1, min_length),
                np.arange(len(step['gyro'].iloc[:, i])),
                step['gyro'].iloc[:, i]
            ) for i in range(step['gyro'].shape[1])
        ]).T
        step['downsampled_gyroz'] = downsampled_gyro[:, 0]  # Store only the z component

#################################################### PLOT #######################################################

    plt.figure(figsize=(10, 6))
    for step in step_data:
        # Plot all downsampled gyroz overlapped
        plt.plot(step['downsampled_gyroz'], alpha=0.7)
    plt.title("Segmented Saggital Gyro")
    plt.xlabel("Normalized Time")
    plt.ylabel("Gyro")
    plt.grid(True)
    plt.show()

    # Calculate the mean and standard deviation of downsampled gyroz across all steps
    all_downsampled_gyroz = np.array([step['downsampled_gyroz'] for step in step_data])
    mean_gyroz = np.mean(all_downsampled_gyroz, axis=0)
    std_gyroz = np.std(all_downsampled_gyroz, axis=0)

    # Plot the mean and standard deviation
    plt.figure(figsize=(10, 6))
    plt.plot(mean_gyroz, label='Mean GyroZ', color='blue')
    plt.fill_between(
        np.arange(len(mean_gyroz)),
        mean_gyroz - std_gyroz,
        mean_gyroz + std_gyroz,
        color='blue',
        alpha=0.2,
        label='Mean ± Std Dev'
    )
    plt.title("Mean and Standard Deviation of Step saggital Gyro")
    plt.xlabel("Normalized Time")
    plt.ylabel("Gyro")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Calculate and print the overall mse and standard deviation of downsampled gyroz
    mse = np.mean((all_downsampled_gyroz - mean_gyroz) ** 2)
    overall_std_gyroz = np.mean(std_gyroz)
    print("Mean Squared Error (MSE) of downsampled gyroz:", mse)
    print("Overall Standard Deviation of downsampled gyroz:", overall_std_gyroz)

    
    return step_data

