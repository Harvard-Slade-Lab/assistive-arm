import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.signal import butter
from scipy.signal import filtfilt
from scipy.signal import find_peaks
import Interpolation
from Segmentation_Methods import AREDCyclicSegm
from Segmentation_Methods import Cyclic_PeaksSegmentation
from Segmentation_Methods import GyroSaggitalSegm

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

    # Magitude Filtering:
    sampling_rate = frequencies[0]  # Hz
    nyquist = 0.5 * sampling_rate
    cutoff = 5  # Hz
    normal_cutoff = cutoff / nyquist
    b, a = butter(4, normal_cutoff, btype='low', analog=False)
    magnitude = filtfilt(b, a, raw_magnitude)

    
    segments, abs_filtered_gyro_derivative = GyroSaggitalSegm.GyroSaggitalSegm(gyro_data, frequencies[0])

    abs_filtered_gyro_derivative = pd.DataFrame(
    abs_filtered_gyro_derivative, 
    columns=['x']
    )
    abs_filtered_gyro_derivative = pd.concat(
        [abs_filtered_gyro_derivative]*3, 
        axis=1
    )
    abs_filtered_gyro_derivative.columns = ['x', 'y', 'z']


    # # Segment the gait cycles
    # segments, peaks = Cyclic_PeaksSegmentation.segment_gait_cycles(magnitude, time_gyro)

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
            'absgyro': abs_filtered_gyro_derivative.iloc[start:end]  # Changed to .iloc
        })


    
    return step_data

