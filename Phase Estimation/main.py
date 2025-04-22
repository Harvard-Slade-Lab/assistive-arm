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

try:
    # Segmentation Selection:
    segment_choice = input("Select segmentation method:\n1. GyroMagnitude Segmentation\n2. ARED Segmentation\n3. SHOE Segmentation\n4. HMMSegmentation\n5. Cyclic Segmentation").strip()
     
    if segment_choice == '5':
        # Load and process files
        acc_data, gyro_data, or_data, acc_files, gyro_files, or_files = DataLoaderYinkai.load_and_process_files(folder_path)
        print(f"Loaded {len(acc_files)} ACC files, {len(gyro_files)} GYRO files, and {len(or_files)} OR files")
        
        # Group files by timestamp
        grouped_indices = DataLoaderYinkai.group_files_by_timestamp(acc_files, gyro_files, or_files)
        print(f"Found {len(grouped_indices)} complete data sets")
        if not grouped_indices:
            print("No complete data sets found. Exiting...")
    else:
        # Load and process files
        acc_data, gyro_data, or_data, acc_files, gyro_files, or_files = DataLoader.load_and_process_files(folder_path)
        print(f"Loaded {len(acc_files)} ACC files, {len(gyro_files)} GYRO files, and {len(or_files)} OR files")
        
        # Group files by timestamp
        grouped_indices = DataLoader.group_files_by_timestamp(acc_files, gyro_files, or_files)
        print(f"Found {len(grouped_indices)} complete data sets")
        if not grouped_indices:
            print("No complete data sets found. Exiting...")

    # Create X and Y matrices
    X, Y, timestamps, segment_lengths, feature_names, frequencies = MatrixCreator.create_matrices(acc_data, gyro_data, or_data, grouped_indices, segment_choice, biasPlot_flag=training_segmentation_flag, interpPlot_flag=training_interpolation_flag)
    print(f"Created X matrix with shape {X.shape} and Y matrix with length {len(Y)}")
    
    # Print column information
    print("\nColumn information:")
    for i, name in enumerate(feature_names):
        print(f"Column {i}: {name}")

    # Visualize matrices
    MatrixCreator.visualize_matrices(X, Y, timestamps, segment_choice, segment_lengths, feature_names)
    
    # User interaction
    print("\nRegression Options:")
    print("1. Ridge Regression\n2. Lasso Regression\n3. Linear Regression\n4. SVR Regression\n5. All regression\n6. Random Forest Regression")
    choice = input("Enter your choice (1-5): ")
    
    # Initialize variables
    models = {'ridge': None, 'lasso': None, 'linear': None}
    results = {'ridge': None, 'lasso': None, 'linear': None}
    
    
    # Perform selected regression(s)
    if choice == '1':
        ridge_result, y_ridge = RidgeRegressionCV.enhanced_ridge_regression(X,Y,feature_names,alpha_range=(-7, 7, 40), cv=None, plot=True, frequencies=frequencies)
    elif choice == '2':
        lasso_result, y_lasso = LassoRegressionCV.enhanced_lasso_regression(X,Y,feature_names,alpha_range=(-7, 7, 40), cv=None, plot=True, frequencies=frequencies)
    elif choice == '3':
        linear_model, y_linear, _, mse_linear = Linear_Reg.linear_regression(X,Y, frequencies, feature_names=feature_names,  plot=True)
    elif choice == '6':
        randomforest_result, y_randomforest = RandomForest.enhanced_random_forest_regression(X,Y,feature_names,param_grid=None, cv=5, plot=True, frequencies=frequencies)
    elif choice == '5':
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
        ridge_result, y_ridge = RidgeRegressionCV.enhanced_ridge_regression(X,Y,feature_names,alpha_range=(-7, 7, 40), cv=None, plot=True, frequencies=frequencies)
        lasso_result, y_lasso = LassoRegressionCV.enhanced_lasso_regression(X,Y,feature_names,alpha_range=(-7, 7, 40), cv=None, plot=True, frequencies=frequencies)
        linear_model, y_linear, _, mse_linear = Linear_Reg.linear_regression(X,Y, frequencies, feature_names=feature_names, plot=True)
        svr_model, y_svr = SVR_Reg.enhanced_svr_regression(X,Y, kernel='rbf', param_grid=param_grid, plot=True, frequencies=frequencies)
        randomforest_result, y_randomforest = RandomForest.enhanced_random_forest_regression(X,Y,feature_names,param_grid=None, cv=5, plot=True, frequencies=frequencies)
    elif choice == '4':
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
    else:
        print("Invalid choice")
    
        # Handle testing
    if choice == '5':
        mse_ridge = ridge_result['mse']
        mse_lasso = lasso_result['mse']
        mse_linear = mse_linear
        mse_svr = svr_model['mse']
        mse_randomforest = randomforest_result['mse']

        print("\nPlotting comparison...")
        plt.figure(figsize=(12, 6))
        plt.plot(y_linear, label='Linear', color='blue')
        plt.plot(y_ridge, label='Ridge', color='orange')
        plt.plot(y_lasso, label='Lasso', color='green')
        plt.plot(y_svr, label='SVR', color='purple')
        plt.plot(y_randomforest, label='Random Forest', color='brown')
        plt.plot(Y, label='Target', color='red', linestyle='--')
        plt.xlabel('Time (s)')
        plt.ylabel('Percentage (%)')
        plt.title('Regression Comparison')
        plt.legend(loc='upper left')
        plt.grid(True)
        
        # Format MSE values as x*10^-y
        def format_mse(mse):
            exponent = int(np.floor(np.log10(mse)))
            base = mse / (10**exponent)
            return f"{base:.2f}×10⁻{abs(exponent)}"
        
        mse_text = (f"Linear MSE: {format_mse(mse_linear)}\n"
                f"Ridge MSE: {format_mse(mse_ridge)}\n"
                f"Lasso MSE: {format_mse(mse_lasso)}\n"
                f"SVR MSE: {format_mse(mse_svr)}\n"
                f"Random Forest MSE: {format_mse(mse_randomforest)}")
        
        # Add MSE values as text in the plot, keeping it within the grid
        plt.text(0.95, 0.05, mse_text, fontsize=10, ha='right', va='bottom', 
             transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
        
        plt.show()
        # Store all models for testing
        current_model = {
            'ridge': ridge_result['model'] if 'ridge_result' in locals() else None,
            'lasso': lasso_result['model'] if 'lasso_result' in locals() else None,
            'linear': linear_model if 'linear_model' in locals() else None,
            'svr': svr_model['model'] if 'svr_model' in locals() else None,
            'randomforest': randomforest_result['model'] if 'randomforest_result' in locals() else None
        }

        # Testing:
        TestManager.handle_test_decision(choice, current_model, frequencies, segment_choice, plot_flag_segment=tests_segment_flag, plot_flag_interp=tests_interp_flag)
    else:
        current_model = ridge_result['model'] if choice == '1' else lasso_result['model'] if choice == '2' else svr_model['model'] if choice == '4' else randomforest_result['model'] if choice == '6' else linear_model
        TestManager.handle_test_decision(choice, current_model, frequencies, segment_choice, plot_flag_segment=tests_segment_flag, plot_flag_interp=tests_interp_flag)
   
   
    plt.show(block=True)
    
except Exception as e:
    print(f"An error occurred: {e}")
