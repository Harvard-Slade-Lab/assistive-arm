import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tkinter.filedialog import askdirectory
from Interpolation import interpolate_and_visualize
import DataLoader
import MatrixCreator
import RidgeRegressionCV
import LassoRegressionCV
import Linear_Reg
import TestManager
import SVR_Reg
import BiasAndSegmentation
from tqdm import tqdm
# Add at the top with other imports
import tkinter as tk
from tkinter import ttk

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
    # Load and process files
    acc_data, gyro_data, or_data, acc_files, gyro_files, or_files = DataLoader.load_and_process_files(folder_path)
    print(f"Loaded {len(acc_files)} ACC files, {len(gyro_files)} GYRO files, and {len(or_files)} OR files")
    
    # Group files by timestamp
    grouped_indices = DataLoader.group_files_by_timestamp(acc_files, gyro_files, or_files)
    print(f"Found {len(grouped_indices)} complete data sets")
    if not grouped_indices:
        print("No complete data sets found. Exiting...")


    # Grid search:
    grid = input("Do you want to perform the grid search? (yes/no): ").strip().lower()
    if grid == 'yes':
        # Perform grid search for fictitious trials parameter tuning:
        # Parameters:
        acc_noise_percent = np.linspace(1, 2, 2)
        gyro_noise_percent = np.linspace(1, 2, 2)
        angles = np.linspace(10, 40, 4, dtype=int)
        warp_min = np.linspace(0.8, 1, 2)
        warp_max = np.linspace(1.2, 1.4, 2)
        number_of_fict = np.linspace(10, 110, 10, dtype=int)  # Added parameter as integer

        
        # User interaction
        print("\nRegression Options:")
        print("1. Ridge Regression\n2. Lasso Regression\n3. Linear Regression")
        choice = input("Enter your choice (1-3): ")
        
        # Initialize variables
        best_params = {
            'acc_noise': None,
            'gyro_noise': None,
            'angle': None,
            'warp_min': None,
            'warp_max': None,
            'number_of_fict': None  # Added to tracking
        }
        best_model = None
        lowest_mse = float('inf')  # Initialize with infinity
        best_y_pred_list = []  # Store the y_pred_list for the best parameters
        
        print("Load data for the testing")
        # Select folder
        folder_path = TestManager.select_folder()
        if not folder_path:
            print("No folder selected. Exiting.")
            exit()
        
        # Load and process files
        acc_data_test, gyro_data_test, or_data_test, acc_files_test, gyro_files_test, or_files_test = TestManager.load_and_process_files(folder_path)
        
        # Group files by timestamp
        grouped_indices_test = TestManager.group_files_by_timestamp(acc_files_test, gyro_files_test, or_files_test)
        if not grouped_indices_test:
            print("No complete groups of files found. Exiting.")
            exit()
        
        # Get frequencies (only need to do this once)
        frequencies = BiasAndSegmentation.sensors_frequencies()
        print("Creating test matrices")
        # Create matrices for each timestamp
        timestamp_matrices, feature_names, frequencies = TestManager.create_timestamp_matrices(
            acc_data_test, gyro_data_test, or_data_test, grouped_indices_test, frequencies,
            biasPlot_flag=False, interpPlot_flag=False
        )
        
        # Initialize counter for progress tracking
        total_combinations = (len(acc_noise_percent) * len(gyro_noise_percent) *
                                len(angles) * len(warp_min) * len(warp_max) *
                                len(number_of_fict))  # Updated total combinations
        current_combination = 0
        window = tk.Tk()
        window.title("Grid Search Progress")
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(window, variable=progress_var, maximum=total_combinations)
        progress_bar.pack(padx=20, pady=10)
        status_label = tk.Label(window, text="Starting grid search...")
        status_label.pack(pady=5)
        window.update()

        # Create a results tracking list to store all results
        all_results = []
        
        X_train, Y_train, sorted_timestamps_train, segment_lengths_train, feature_names_train, frequencies = MatrixCreator.create_matrices(acc_data, gyro_data, or_data, grouped_indices, biasPlot_flag=False, interpPlot_flag=False, 
                    grid=grid)
        
        for acc_noise in acc_noise_percent:
            for gyro_noise in gyro_noise_percent:
                for angle in angles:
                    for warp_min_val in warp_min:
                        for warp_max_val in warp_max:
                            for n_fict in number_of_fict:  # New loop
                                if warp_min_val >= warp_max_val:
                                    total_combinations -= 1
                                    continue
                                
                                current_combination += 1
                                progress_var.set(current_combination)
                                status_text = f"Processing: {current_combination}/{total_combinations}\n"
                                status_text += f"acc: {acc_noise:.1f}, gyro: {gyro_noise:.1f}\n"
                                status_text += f"angle: {angle}, warp: {warp_min_val:.1f}-{warp_max_val:.1f}\n"
                                status_text += f"fict: {n_fict}"
                                status_label.config(text=status_text)
                                window.update_idletasks()
                                print(f"Running combination {current_combination}/{total_combinations} ({(current_combination/total_combinations)*100:.1f}%)")
                                print(f"Parameters: acc_noise: {acc_noise}, gyro_noise: {gyro_noise}, angle: {angle}, warp_min: {warp_min_val}, warp_max: {warp_max_val}, number_of_fict: {n_fict}")
                                
                                
                                # Modified matrix creation call
                                X, Y, timestamps, segment_lengths, feature_names, frequencies = MatrixCreator.fictitious_trials(X_train, Y_train, sorted_timestamps_train, segment_lengths_train, 
                                                                                                                    feature_names_train, frequencies, grid, acc_grid=acc_noise, 
                                                                                                                    gyro_grid=gyro_noise, angle_grid=angle, warp_min_grid=warp_min_val, 
                                                                                                                    warp_max_grid=warp_max_val, num_fict_grid=n_fict)
                                
                                # Clear y_pred_list for this iteration
                                globals()["y_pred_list"] = []

                                try:    
                                    # Perform selected regression(s)
                                    if choice == '1':
                                        ridge_result, y_ridge = RidgeRegressionCV.enhanced_ridge_regression(
                                            X, Y, feature_names, alpha_range=(-7, 7, 40), cv=None, plot=False, frequencies=frequencies
                                        )
                                        model = ridge_result['model']
                                        print(f"Ridge Regression MSE: {ridge_result['mse']}")
                                    elif choice == '2':
                                        lasso_result, y_lasso = LassoRegressionCV.enhanced_lasso_regression(
                                            X, Y, feature_names, alpha_range=(-7, 7, 40), cv=None, plot=False, frequencies=frequencies
                                        )
                                        model = lasso_result['model']
                                    elif choice == '3':
                                        linear_model, y_linear, _, mse_linear = Linear_Reg.linear_regression(
                                            X, Y, frequencies, feature_names=feature_names, plot=False
                                        )
                                        model = linear_model
                                    
                                    mse_vector = []  # Initialize mse_vector as an empty list
                                    
                                    if choice == '1':
                                        for ts, matrix in timestamp_matrices.items():
                                            y_pred, mse = RidgeRegressionCV.test_ridge(model, matrix, frequencies, Plot_flag=False)
                                            # Store the predicted y for this test in a list
                                            y_pred_list = globals().get("y_pred_list", [])
                                            y_pred_list.append((ts, y_pred))
                                            globals()["y_pred_list"] = y_pred_list
                                            # Stores mse in a vector to store the results every iteration:
                                            mse_vector.append(mse)
                                        # compute average of mse_vector:
                                        average_mse = np.mean(mse_vector)
                                        print(f"Average MSE for Ridge Regression: {average_mse}")
                                    
                                    elif choice == '2':
                                        for ts, matrix in timestamp_matrices.items():
                                            y_pred, mse = LassoRegressionCV.test_lasso(model, matrix, frequencies, Plot_flag=False)
                                            # Store the predicted y for this test in a list
                                            y_pred_list = globals().get("y_pred_list", [])
                                            y_pred_list.append((ts, y_pred))
                                            globals()["y_pred_list"] = y_pred_list
                                            # Stores mse in a vector to store the results every iteration:
                                            mse_vector.append(mse)
                                        # compute average of mse_vector:
                                        average_mse = np.mean(mse_vector)
                                        print(f"Average MSE for Lasso Regression: {average_mse}")
                                    
                                    elif choice == '3':
                                        for ts, matrix in timestamp_matrices.items():
                                            y_pred, _, mse = Linear_Reg.test_regression(model, matrix, frequencies, Plot_flag=False)
                                            # Store the predicted y for this test in a list
                                            y_pred_list = globals().get("y_pred_list", [])
                                            y_pred_list.append((ts, y_pred))
                                            globals()["y_pred_list"] = y_pred_list
                                            # Stores mse in a vector to store the results every iteration:
                                            mse_vector.append(mse)
                                        # compute average of mse_vector:
                                        average_mse = np.mean(mse_vector)
                                        print(f"Average MSE for Linear Regression: {average_mse}")
                                    
                                    # Store results with new parameter
                                    all_results.append({
                                        'acc_noise': acc_noise,
                                        'gyro_noise': gyro_noise,
                                        'angle': angle,
                                        'warp_min': warp_min_val,
                                        'warp_max': warp_max_val,
                                        'number_of_fict': n_fict,  # Added
                                        'mse': average_mse,
                                        'model': model,
                                        'y_pred_list': globals()["y_pred_list"].copy()
                                    })
                                    
                                    
                                    # Check if current parameter set is the best so far
                                    if average_mse < lowest_mse:
                                        lowest_mse = average_mse
                                        best_params['acc_noise'] = acc_noise
                                        best_params['gyro_noise'] = gyro_noise
                                        best_params['angle'] = angle
                                        best_params['warp_min'] = warp_min_val
                                        best_params['warp_max'] = warp_max_val
                                        best_params['number_of_fict'] = n_fict
                                        best_model = model
                                        best_y_pred_list = globals()["y_pred_list"].copy()  # Save the y_pred_list for best parameters
                                        
                                        # Print update whenever a better parameter set is found
                                        print("\n>>> New best parameters found!")
                                        print(f"Best MSE so far: {lowest_mse}")
                                        print(f"Best parameters: acc_noise={acc_noise}, gyro_noise={gyro_noise}, "
                                            f"angle={angle}, warp_min={warp_min_val}, warp_max={warp_max_val}\n")
                                        

                                
                                except Exception as e:
                                    print(f"Error with combination: acc_noise={acc_noise}, gyro_noise={gyro_noise}, "
                                        f"angle={angle}, warp_min={warp_min_val}, warp_max={warp_max_val}")
                                    print(f"Error details: {str(e)}")
                                    continue
    
        # Sort all results by MSE for comprehensive reporting
        all_results.sort(key=lambda x: x['mse'])
        
        # After grid search is complete, print the optimal parameters and MSE
        print("\n" + "="*50)
        print("Grid Search Complete!")
        print("="*50)
        print(f"Optimal Parameters:")
        print(f"Accelerometer noise: {best_params['acc_noise']}")
        print(f"Gyroscope noise: {best_params['gyro_noise']}")
        print(f"Angle: {best_params['angle']}")
        print(f"Warp min: {best_params['warp_min']}")
        print(f"Warp max: {best_params['warp_max']}")
        print(f"Lowest Average MSE: {lowest_mse}")
        
        # Print top 5 parameter combinations
        print("\nTop 5 Parameter Combinations:")
        for i, result in enumerate(all_results[:5]):
            print(f"{i+1}. MSE: {result['mse']}, acc_noise: {result['acc_noise']}, gyro_noise: {result['gyro_noise']}, "
                f"angle: {result['angle']}, warp_min: {result['warp_min']}, warp_max: {result['warp_max']}, number_of_fict: {result['number_of_fict']}")
        
        # Set the global y_pred_list to the one from the best parameters
        globals()["y_pred_list"] = best_y_pred_list
        
        # Ask if user wants to save the best model
        save_model = input("Do you want to save the best model and results? (yes/no): ").strip().lower()
        if save_model == 'yes':
            import pickle
            import pandas as pd
            from datetime import datetime
            
            # Create a timestamp for the files
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save the model
            model_filename = f"best_model_{timestamp}.pkl"
            model_info = {
                'model': best_model,
                'parameters': best_params,
                'mse': lowest_mse,
                'regression_type': 'Ridge' if choice == '1' else 'Lasso' if choice == '2' else 'Linear',
                'y_pred_list': best_y_pred_list
            }
            with open(model_filename, 'wb') as f:
                pickle.dump(model_info, f)
            print(f"Best model saved as {model_filename}")
            
            # Save all results as CSV
            results_df = pd.DataFrame([{
                'acc_noise': r['acc_noise'],
                'gyro_noise': r['gyro_noise'],
                'angle': r['angle'],
                'warp_min': r['warp_min'],
                'warp_max': r['warp_max'],
                'mse': r['mse']
            } for r in all_results])
            
            csv_filename = f"grid_search_results_{timestamp}.csv"
            results_df.to_csv(csv_filename, index=False)
            print(f"All results saved as {csv_filename}")
        
        # Optionally, rerun the best model with plotting enabled to visualize results
        visualize_best = input("Do you want to visualize the results with the best parameters? (yes/no): ").strip().lower()
        if visualize_best == 'yes':
            print("\nRerunning with best parameters and plotting enabled...")
            X, Y, timestamps, segment_lengths, feature_names, frequencies = MatrixCreator.create_matrices(
                acc_data, gyro_data, or_data, grouped_indices, 
                biasPlot_flag=True, interpPlot_flag=True, 
                grid=grid, acc_noise=best_params['acc_noise'], gyro_noise=best_params['gyro_noise'], 
                angle=best_params['angle'], warp_min=best_params['warp_min'], warp_max=best_params['warp_max']
            )
            
            if choice == '1':
                ridge_result, y_ridge = RidgeRegressionCV.enhanced_ridge_regression(
                    X, Y, feature_names, alpha_range=(-7, 7, 40), cv=None, plot=True, frequencies=frequencies
                )
                print("\nTesting with best model:")
                for ts, matrix in timestamp_matrices.items():
                    y_pred, mse = RidgeRegressionCV.test_ridge(best_model, matrix, frequencies, Plot_flag=True)
                    print(f"Test MSE for timestamp {ts}: {mse}")
            
            elif choice == '2':
                lasso_result, y_lasso = LassoRegressionCV.enhanced_lasso_regression(
                    X, Y, feature_names, alpha_range=(-7, 7, 40), cv=None, plot=True, frequencies=frequencies
                )
                print("\nTesting with best model:")
                for ts, matrix in timestamp_matrices.items():
                    y_pred, mse = LassoRegressionCV.test_lasso(best_model, matrix, frequencies, Plot_flag=True)
                    print(f"Test MSE for timestamp {ts}: {mse}")
            
            elif choice == '3':
                linear_model, y_linear, _, mse_linear = Linear_Reg.linear_regression(
                    X, Y, frequencies, feature_names=feature_names, plot=True
                )
                print("\nTesting with best model:")
                for ts, matrix in timestamp_matrices.items():
                    y_pred, _, mse = Linear_Reg.test_regression(best_model, matrix, frequencies, Plot_flag=True)
                    print(f"Test MSE for timestamp {ts}: {mse}")
            
            print(f"Final Average MSE with best parameters: {lowest_mse}")





# NO GRID SEARCH:


    else:
        # Create X and Y matrices
        X, Y, timestamps, segment_lengths, feature_names, frequencies = MatrixCreator.create_matrices(acc_data, gyro_data, or_data, grouped_indices, biasPlot_flag=training_segmentation_flag, interpPlot_flag=training_interpolation_flag)
        print(f"Created X matrix with shape {X.shape} and Y matrix with length {len(Y)}")
        
        # Print column information
        print("\nColumn information:")
        for i, name in enumerate(feature_names):
            print(f"Column {i}: {name}")
        
        # Visualize matrices
        MatrixCreator.visualize_matrices(X, Y, timestamps, segment_lengths, feature_names)
        
        # User interaction
        print("\nRegression Options:")
        print("1. Ridge Regression\n2. Lasso Regression\n3. Linear Regression\n4. SVR Regression\n5. All regression")
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
        elif choice == '5':
            # parameter grid for SVR, reduce the number of parameters for faster computation
            # Note: The grid search will take longer with more parameters
            fast_comput = input("Do you want to reduce the number of parameters for faster SVR computation? (yes/no): ").strip().lower()
            if fast_comput == 'yes':
                # Reduced grid (3x3x3 = 27 combinations)
                param_grid = {
                    'svr__C': np.logspace(1, 3, 3),       # [10, 100, 1000]
                    'svr__epsilon': np.logspace(-3, -1, 3), # [0.001, 0.01, 0.1]
                    'svr__gamma': np.logspace(-3, -1, 3)   # [0.001, 0.01, 0.1]
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
        elif choice == '4':
            # parameter grid for SVR, reduce the number of parameters for faster computation
            # Note: The grid search will take longer with more parameters
            fast_comput = input("Do you want to reduce the number of parameters for faster SVR computation? (yes/no): ").strip().lower()
            if fast_comput == 'yes':
                # Reduced grid (3x3x3 = 27 combinations)
                param_grid = {
                    'svr__C': np.logspace(1, 3, 3),       # [10, 100, 1000]
                    'svr__epsilon': np.logspace(-3, -1, 3), # [0.001, 0.01, 0.1]
                    'svr__gamma': np.logspace(-3, -1, 3)   # [0.001, 0.01, 0.1]
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

            print("\nPlotting comparison...")
            plt.figure(figsize=(12, 6))
            plt.plot(y_linear, label='Linear', color='blue')
            plt.plot(y_ridge, label='Ridge', color='orange')
            plt.plot(y_lasso, label='Lasso', color='green')
            plt.plot(y_svr, label='SVR', color='purple')
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
                    f"SVR MSE: {format_mse(mse_svr)}")
            
            # Add MSE values as text in the plot, keeping it within the grid
            plt.text(0.95, 0.05, mse_text, fontsize=10, ha='right', va='bottom', 
                transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
            
            plt.show()
            # Store all models for testing
            current_model = {
                'ridge': ridge_result['model'] if 'ridge_result' in locals() else None,
                'lasso': lasso_result['model'] if 'lasso_result' in locals() else None,
                'linear': linear_model if 'linear_model' in locals() else None,
                'svr': svr_model['model'] if 'svr_model' in locals() else None
            }

            # Testing:
            TestManager.handle_test_decision(choice, current_model, frequencies, plot_flag_segment=tests_segment_flag, plot_flag_interp=tests_interp_flag)
        else:
            current_model = ridge_result['model'] if choice == '1' else lasso_result['model'] if choice == '2' else svr_model['model'] if choice == '4' else linear_model
            TestManager.handle_test_decision(choice, current_model, frequencies, plot_flag_segment=tests_segment_flag, plot_flag_interp=tests_interp_flag)
    
    
        plt.show(block=True)
        
except Exception as e:
    print(f"An error occurred: {e}")
