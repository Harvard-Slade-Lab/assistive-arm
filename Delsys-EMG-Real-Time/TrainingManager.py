import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from tkinter.filedialog import askdirectory
import joblib
import path_setup
from Interpolation import interpolate_and_visualize
import DataLoader
import MatrixCreator
from Regression_Methods import SVR_Reg

def Training_Manager(training_segmentation_flag = False, training_interpolation_flag = False, tests_segment_flag = False, tests_interp_flag = False):
    # Frequencies:
    frequencies = [519, 519, 222]  # Gyro, Acc, OR
    # frequencies = [370.3704, 370.3704, 74.0741]  # Gyro, Acc, OR

    # Select folder
    folder_path = DataLoader.select_folder()

    if not folder_path:
        print("No folder selected. Exiting...")
        
    print(f"Selected folder: {folder_path}")

    while True:
        # Segmentation Selection:
        segment_choice = input("Select segmentation method:\n1. ARED Segmentation\n2. Cyclic Segmentation").strip()
        break


    # Load and process files
    acc_data, gyro_data, or_data, acc_files, gyro_files, or_files = DataLoader.load_and_process_files(folder_path)
    print(f"Loaded {len(acc_files)} ACC files, {len(gyro_files)} GYRO files, and {len(or_files)} OR files")

    # Group files by timestamp
    grouped_indices = DataLoader.group_files_by_timestamp(acc_files, gyro_files, or_files)
    print(f"Found {len(grouped_indices)} complete data sets")

    # Create X and Y matrices
    X, Y, timestamps, segment_lengths, feature_names = MatrixCreator.create_matrices(acc_data, gyro_data, or_data, grouped_indices, segment_choice, frequencies, biasPlot_flag=training_segmentation_flag, interpPlot_flag=training_interpolation_flag)
    print(f"Created X matrix with shape {X.shape} and Y matrix with length {len(Y)}")
    
    # Print column information
    print("\nColumn information:")
    for i, name in enumerate(feature_names):
        print(f"Column {i}: {name}")

    # Visualize matrices
    MatrixCreator.visualize_matrices(X, Y, timestamps, segment_lengths, feature_names)

    # SVR Regression
    # parameter grid for SVR, reduce the number of parameters for faster computation
    # Note: The grid search will take longer with more parameters
    fast_comput = input("Do you want to reduce the number of parameters for faster SVR computation? (yes/no): ").strip().lower()
    if fast_comput == 'yes':
        param_grid = {
            'svr__C': [100],
            'svr__epsilon': [0.01],
            'svr__gamma': [0.01]
        }   
    else:
        # Enhanced grid (7x4x6 = 168 combinations)
        param_grid = {
            'svr__C': np.logspace(-3, 3, 7),
            'svr__epsilon': np.logspace(-3, 0, 4),
            'svr__gamma': np.logspace(-4, 1, 6)
            }
    svr_model, y_svr = SVR_Reg.enhanced_svr_regression(X,Y, kernel='rbf', param_grid=param_grid, plot=True, frequencies=frequencies)
    # Save the model
    joblib.dump(svr_model, 'svr_phase_model.joblib')

    current_model = svr_model['model']

    plt.show(block= False)

    return current_model, segment_choice

from PyQt5.QtWidgets import QFileDialog, QInputDialog, QMessageBox

def Training_Manager_GUI(parent=None, training_segmentation_flag=False, training_interpolation_flag=False,
                          tests_segment_flag=False, tests_interp_flag=False):
    # Frequencies:
    frequencies = [519, 519, 222]  # Gyro, Acc, OR

    # Select folder using QFileDialog
    folder_path = QFileDialog.getExistingDirectory(parent, "Select Folder for Training Data")
    if not folder_path:
        QMessageBox.information(parent, "No Folder Selected", "Training cancelled. No folder was selected.")
        return None, None

    print(f"Selected folder: {folder_path}")

    # Ask user for segmentation method via GUI
    segment_choice, ok = QInputDialog.getItem(
        parent, "Segmentation Method", "Select segmentation method:\n1. One Shot Segmentation\n2. Cyclic Segmentation",
        ["1", "2"], 0, False
    )
    if not ok:
        QMessageBox.information(parent, "Cancelled", "Training cancelled. No segmentation method selected.")
        return None, None
    
    print(f"Selected segmentation method: {segment_choice}")

    # Load and process files
    acc_data, gyro_data, or_data, acc_files, gyro_files, or_files = DataLoader.load_and_process_files(folder_path)
    print(f"Loaded {len(acc_files)} ACC files, {len(gyro_files)} GYRO files, and {len(or_files)} OR files")

    grouped_indices = DataLoader.group_files_by_timestamp(acc_files, gyro_files, or_files)
    print(f"Found {len(grouped_indices)} complete data sets")

    X, Y, timestamps, segment_lengths, feature_names = MatrixCreator.create_matrices(
        acc_data, gyro_data, or_data, grouped_indices,
        segment_choice, frequencies,
        biasPlot_flag=training_segmentation_flag,
        interpPlot_flag=training_interpolation_flag
    )

    print(f"Created X matrix with shape {X.shape} and Y matrix with length {len(Y)}")

    print("\nColumn information:")
    for i, name in enumerate(feature_names):
        print(f"Column {i}: {name}")

    # MatrixCreator.visualize_matrices(X, Y, timestamps, segment_lengths, feature_names)

    # Ask user if they want faster computation
    fast_comput, ok = QInputDialog.getItem(
        parent, "Fast Computation?", "Reduce SVR grid search size for faster training?",
        ["yes", "no"], 0, False
    )
    if not ok:
        QMessageBox.information(parent, "Cancelled", "Training cancelled during parameter selection.")
        return None, None

    if fast_comput == 'yes':
        param_grid = {
            'svr__C': [100],
            'svr__epsilon': [0.01],
            'svr__gamma': [0.01]
        }
    else:
        param_grid = {
            'svr__C': np.logspace(-3, 3, 7),
            'svr__epsilon': np.logspace(-3, 0, 4),
            'svr__gamma': np.logspace(-4, 1, 6)
        }

    svr_model, y_svr = SVR_Reg.enhanced_svr_regression(
        X, Y, kernel='rbf',
        param_grid=param_grid,
        plot=True,
        frequencies=frequencies
    )

    joblib.dump(svr_model, 'svr_phase_model.joblib')
    current_model = svr_model['model']

    plt.show(block=False)  # Non-blocking for GUI compatibility

    return current_model, segment_choice
    