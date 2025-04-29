import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tkinter.filedialog import askdirectory
from Interpolation import interpolate_and_visualize
import DataLoader
import DataLoaderYinkai
import MatrixCreator
import TestManager
from Regression_Methods import RidgeRegressionCV
from Regression_Methods import LassoRegressionCV        
from Regression_Methods import Linear_Reg
from Regression_Methods import SVR_Reg
from Regression_Methods import RandomForest

def Training_Manager(frequencies, training_segmentation_flag = False, training_interpolation_flag = False, tests_segment_flag = False, tests_interp_flag = False):
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

    current_model = svr_model['model']

    return current_model, segment_choice, load_choice
    
    plt.show(block= True)