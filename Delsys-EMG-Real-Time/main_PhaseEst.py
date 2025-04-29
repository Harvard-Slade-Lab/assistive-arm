# import sys
# from PyQt5 import QtWidgets
# from EMGDataCollector import EMGDataCollector
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

    # Frequencies:
    frequencies = [519, 519, 222]  # Gyro, Acc, OR
    # frequencies = [370.3704, 370.3704, 74.0741]  # Gyro, Acc, OR

    data_directory = "C:/Users/patty/Desktop/new_data_acquisition/assistive-arm/Data_Place/"
    # Flag for real time plots
    plot = False

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

    # Prompt user to decide whether to train the model or use existing training data
    user_choice = input("Do you want to train the model or use existing training data?\n1: Train\n2: Use").strip().lower()

    if user_choice == "1":
        print("You chose to train the model.")
        print("Starting EMG Data Collector for training...")

        # appQt = QtWidgets.QApplication(sys.argv)
        # collector = EMGDataCollector(plot, socket, imu_processing, mixed_processing, emg_control, real_time_processing, window_duration=window_duration, data_directory=data_directory)
        # collector.connect_base()
        # collector.scan_and_pair_sensors()
        # appQt.exec_()
        # print("Qt application closed, continuing execution...")
    
    elif user_choice == "2":
        print("You chose to use existing training data.")
        # Add any additional logic for loading existing training data if needed

    ######################################################## PROCESSING DATA FOR TRAINING ########################################################
    print("Processing data for training...")
    # PLOT Flags:
    training_segmentation_flag = False
    training_interpolation_flag = False
    tests_segment_flag = False
    tests_interp_flag = False

    current_model, segment_choice = Training_Manager(frequencies, training_segmentation_flag, training_interpolation_flag, tests_segment_flag, tests_interp_flag)

    ######################################################### TEST ################################################################################

    # choice = 4
    # TestManager.handle_test_decision(choice, current_model, frequencies, segment_choice, plot_flag_segment=tests_segment_flag, plot_flag_interp=tests_interp_flag)
    
if __name__ == "__main__":
    main()
