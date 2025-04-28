import sys
from PyQt5 import QtWidgets
from EMGDataCollector import EMGDataCollector
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tkinter.filedialog import askdirectory
from Phase_Estimation.Interpolation import interpolate_and_visualize
from Phase_Estimation import DataLoader
from Phase_Estimation import DataLoaderYinkai
from Phase_Estimation import MatrixCreator
from Phase_Estimation import TestManager
from Phase_Estimation.Regression_Methods import RidgeRegressionCV
from Phase_Estimation.Regression_Methods import LassoRegressionCV        
from Phase_Estimation.Regression_Methods import Linear_Reg
from Phase_Estimation.Regression_Methods import SVR_Reg
from Phase_Estimation.Regression_Methods import RandomForest

def main():
    # HYPERPARAMETERS
    window_duration = 5  # Duration in seconds
    # Frequencies:
    frequencies = [519, 519, 222]  # Gyro, Acc, OR
    # frequencies = [370.3704, 370.3704, 74.0741]  # Gyro, Acc, OR

    data_directory = "C:/Users/patty/Desktop/Nate_3rd_arm/code/assistive-arm/Data/"
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
    user_choice = input("Do you want to train the model or use existing training data? (train/use): ").strip().lower()

    if user_choice == "train":
        print("You chose to train the model.")
        # Add any additional logic for training if needed

        print("Starting EMG Data Collector for training...")
        appQt = QtWidgets.QApplication(sys.argv)
        collector = EMGDataCollector(plot, socket, imu_processing, mixed_processing, emg_control, real_time_processing, window_duration=window_duration, data_directory=data_directory)
        collector.connect_base()
        collector.scan_and_pair_sensors()
        appQt.exec_()
        print("Qt application closed, continuing execution...")
    
    elif user_choice == "use":
        print("You chose to use existing training data.")
        # Add any additional logic for loading existing training data if needed

    ######################################################## PROCESSING DATA FOR TRAINING ########################################################
    # PLOT Flags:
    training_segmentation_flag = False
    training_interpolation_flag = False
    tests_segment_flag = False
    tests_interp_flag = False

    # Select folder
    folder_path = DataLoader.select_folder()

    if not folder_path:
        print("No folder selected. Exiting...")
        
    print(f"Selected folder: {folder_path}")

    # Segmentation Selection:
    segment_choice = input("Select segmentation method:\n1. GyroMagnitude Segmentation\n2. ARED Segmentation\n3. SHOE Segmentation\n4. ARED_VARSegmentation\n5. Cyclic Segmentation").strip()
    # Decide to load Yinkai's choice or not
    load_choice = 1


    # Load and process files
    acc_data, gyro_data, or_data, acc_files, gyro_files, or_files = DataLoader.load_and_process_files(folder_path)
    print(f"Loaded {len(acc_files)} ACC files, {len(gyro_files)} GYRO files, and {len(or_files)} OR files")
    
    # Group files by timestamp
    grouped_indices = DataLoader.group_files_by_timestamp(acc_files, gyro_files, or_files)
    print(f"Found {len(grouped_indices)} complete data sets")
    if not grouped_indices:
        print("No complete data sets found. Exiting...")

    # Create X and Y matrices
    X, Y, timestamps, segment_lengths, feature_names = MatrixCreator.create_matrices(acc_data, gyro_data, or_data, grouped_indices, segment_choice, frequencies, biasPlot_flag=training_segmentation_flag, interpPlot_flag=training_interpolation_flag)
    print(f"Created X matrix with shape {X.shape} and Y matrix with length {len(Y)}")
    
    # Print column information
    print("\nColumn information:")
    for i, name in enumerate(feature_names):
        print(f"Column {i}: {name}")

    # Visualize matrices
    MatrixCreator.visualize_matrices(X, Y, timestamps, segment_choice, segment_lengths, feature_names)

    # SVR Regression
    # parameter grid for SVR, reduce the number of parameters for faster computation
    # Note: The grid search will take longer with more parameters
    fast_comput = input("Do you want to reduce the number of parameters for faster SVR computation? (yes/no): ").strip().lower()
    if fast_comput == 'yes':
            # # Reduced grid (3x3x3 = 27 combinations)
        # param_grid = {
        #     'svr__C': np.logspace(1, 3, 3),       # [10, 100, 1000]
        #     'svr__epsilon': np.logspace(-3, -1, 3), # [0.001, 0.01, 0.1]
        #     'svr__gamma': np.logspace(-3, -1, 3)   # [0.001, 0.01, 0.1]
        # }   
        param_grid = {
            'svr__C': [100],      # [10, 100, 1000]
            'svr__epsilon': [0.01], # [0.001, 0.01, 0.1]
            'svr__gamma': [0.01]   # [0.001, 0.01, 0.1]
        }   
    else:
        # Enhanced grid (7x4x6 = 168 combinations)
        param_grid = {
            'svr__C': np.logspace(-3, 3, 7),
            'svr__epsilon': np.logspace(-3, 0, 4),
            'svr__gamma': np.logspace(-4, 1, 6)
            }
    svr_model, y_svr = SVR_Reg.enhanced_svr_regression(X,Y, kernel='rbf', param_grid=param_grid, plot=True, frequencies=frequencies)


    ######################################################## TESTING DATA ########################################################

    current_model = svr_model['model']
    choice = 4
    TestManager.handle_test_decision(choice, current_model, frequencies, segment_choice, load_choice, plot_flag_segment=tests_segment_flag, plot_flag_interp=tests_interp_flag)


plt.show(block= True)

if __name__ == "__main__":
    main()
