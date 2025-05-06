import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from tkinter.filedialog import askdirectory
import joblib
import path_setup
import DataLoader
import MatrixCreator
from Regression_Methods import SVR_Reg
from PyQt5.QtWidgets import QFileDialog, QInputDialog, QMessageBox

def Training_Manager_GUI(parent=None, training_segmentation_flag=False):
    # Frequencies:
    frequencies = parent.frequency_vector
    print(f"Frequencies: {frequencies}")

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

    # If you want to add fictitious trials, there is a section in create_matrices that can be modified
    X, Y, timestamps, segment_lengths, feature_names = MatrixCreator.create_matrices(
        acc_data, gyro_data, or_data, grouped_indices,
        segment_choice, frequencies,
        biasPlot_flag=training_segmentation_flag
    )

    print(f"Created X matrix with shape {X.shape} and Y matrix with length {len(Y)}")

    print("\nColumn information:")
    for i, name in enumerate(feature_names):
        print(f"Column {i}: {name}")

    MatrixCreator.visualize_matrices(X, Y, timestamps, segment_lengths, feature_names)

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

    return current_model
    