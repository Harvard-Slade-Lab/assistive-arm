import sys
from PyQt5 import QtWidgets
from EMGDataCollector import EMGDataCollector
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tkinter.filedialog import askdirectory
from TrainingManager import Training_Manager
import path_setup
import TestManager

def main():
    # HYPERPARAMETERS
    window_duration = 5  # Duration in seconds

    data_directory = "C:/Users/patty/Desktop/new_data_acquisition/assistive-arm/Data_Place/"
    # Flag for real time plots
    plot = True
    # Flag for Socket connection (can be changed with reconnect to raspi)
    socket = False

    # Flag for EMG control (default is wired IMU, also needs to be changed in control script)
    emg_control = False

    # Flag for real time processing (if it is off, the data will be segmented by the start and stop buttons)
    # This is a backup if the segemntation fails due to lag or other issues
    # Default is OR (start detection through OR, stop detection through OR), if real_time_processing is True
    real_time_processing = True
    if real_time_processing:
        # Flag for imu score calcualtion (start detection through IMU, stop detection through IMU)
        imu_processing = True
        # Flag for mixed processing (start detection through IMU, stop detection through OR)
        mixed_processing = False
    else:
        imu_processing = False
        mixed_processing = False

    # IMU porcessing has potential to be used as continuous processing (only press start, motors stop due to imu angle)
    # Peak is detected by function "detect_peak_and_calculate", which can be used by uncommenting in stream_data function
    # Needs more testing though

    current_model = None 
    appQt = QtWidgets.QApplication(sys.argv)
    collector = EMGDataCollector(current_model, plot, socket, imu_processing, mixed_processing, emg_control, real_time_processing, window_duration=window_duration, data_directory=data_directory)
    collector.connect_base()
    collector.scan_and_pair_sensors()
    appQt.exec_()
    print("Qt application closed, continuing execution...")

if __name__ == "__main__":
    main()
