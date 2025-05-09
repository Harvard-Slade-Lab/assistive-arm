# TimestampMatrixCreator.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askdirectory
import BiasAndSegmentation
from Interpolation import interpolate_and_visualize
from Regression_Methods import RidgeRegressionCV
from Regression_Methods import LassoRegressionCV        
from Regression_Methods import Linear_Reg
from Regression_Methods import SVR_Reg
from Regression_Methods import RandomForest
import DataLoader
import DataLoaderYinkai
import CyclicSegmentationManager
from sklearn.metrics import mean_squared_error
import Interpolation
from EulerTransform import quaternion_to_euler

def handle_test_decision(choice, model, frequencies, segment_choice, load_choice, plot_flag_segment, plot_flag_interp):
    """Handle user decision about testing"""
    test_decision = input("\nDo you want to perform the test? (yes/no): ").lower()
    if test_decision == 'yes':
        execute_test(choice, model, frequencies, segment_choice, load_choice, plot_flag_segment, plot_flag_interp)


def execute_test(choice, model, frequencies, segment_choice, load_choice, plot_flag_segment, plot_flag_interp):
    # Select folder
    folder_path = DataLoader.select_folder()
    if not folder_path:
        print("No folder selected. Exiting.")
        return
    
    if load_choice == '2':
        # Load and process files
        acc_data, gyro_data, or_data, acc_files, gyro_files, or_files = DataLoaderYinkai.load_and_process_files(folder_path)
        
        # Group files by timestamp
        grouped_indices = DataLoaderYinkai.group_files_by_timestamp(acc_files, gyro_files, or_files)
    else:
        # Load and process files
        acc_data, gyro_data, or_data, acc_files, gyro_files, or_files = DataLoader.load_and_process_files(folder_path)
        
        # Group files by timestamp
        grouped_indices = DataLoader.group_files_by_timestamp(acc_files, gyro_files, or_files)
    
    if not grouped_indices:
        print("No complete groups of files found. Exiting.")
        return
    
    # Create matrices for each timestamp
    timestamp_matrices, feature_names = create_timestamp_matrices(
        acc_data, gyro_data, or_data, grouped_indices, segment_choice, frequencies,
        biasPlot_flag=plot_flag_segment, interpPlot_flag=plot_flag_interp
    )
    
    # Print information about created matrices
    print(f"\nCreated {len(timestamp_matrices)} matrices for different timestamps:")
    for ts, matrix in timestamp_matrices.items():
        print(f"Timestamp: {ts}, Matrix shape: {matrix.shape}")

    mse_vector = []  # Initialize mse_vector as an empty list
    if choice == '1':
        for ts, matrix in timestamp_matrices.items():
            y_pred, y_target, mse = RidgeRegressionCV.test_ridge(model, matrix, frequencies, Plot_flag=False)
            # Store the predicted y for this test in a list
            y_pred_list = globals().get("y_pred_list", [])
            y_pred_list.append((ts, y_pred))
            globals()["y_pred_list"] = y_pred_list
            
            # Store the target y for this test in a list
            y_target_list = globals().get("y_target_list", [])
            y_target_list.append((ts, y_target))
            globals()["y_target_list"] = y_target_list
            # Stores mse in a vector to store the results every iteration:
            mse_vector.append(mse)
        # compute average of mse_vector:
        average_mse = np.mean(mse_vector)
        y_target_combined = np.concatenate([y for _, y in y_target_list])
        y_pred_combined = np.concatenate([y for _, y in y_pred_list])
        average_mse_2 = mean_squared_error(y_target_combined, y_pred_combined)

        standard_deviation_2 = np.std(y_target_combined - y_pred_combined)
        standard_deviation_mse = np.std(mse_vector)
        
        print(f"Average MSE for Ridge Regression: {average_mse}")
        print(f"Standard Deviation of errors for Ridge Regression (using target): {standard_deviation_2:.4f}")
        print(f"Standard Deviation of MSE for Ridge Regression: {standard_deviation_mse:.4f}")
    elif choice == '2':
        for ts, matrix in timestamp_matrices.items():
            y_pred, y_target, mse = LassoRegressionCV.test_lasso(model, matrix, frequencies, Plot_flag=False)
            # Store the predicted y for this test in a list
            y_pred_list = globals().get("y_pred_list", [])
            y_pred_list.append((ts, y_pred))
            globals()["y_pred_list"] = y_pred_list
            
            # Store the target y for this test in a list
            y_target_list = globals().get("y_target_list", [])
            y_target_list.append((ts, y_target))
            globals()["y_target_list"] = y_target_list
            # Stores mse in a vector to store the results every iteration:
            mse_vector.append(mse)
        # compute average of mse_vector:
        average_mse = np.mean(mse_vector)
        y_target_combined = np.concatenate([y for _, y in y_target_list])
        y_pred_combined = np.concatenate([y for _, y in y_pred_list])
        average_mse_2 = mean_squared_error(y_target_combined, y_pred_combined)

        standard_deviation_2 = np.std(y_target_combined - y_pred_combined)
        standard_deviation_mse = np.std(mse_vector)
        
        print(f"Average MSE for Lasso Regression: {average_mse}")
        print(f"Standard Deviation of errors for Lasso Regression (using target): {standard_deviation_2:.4f}")
        print(f"Standard Deviation of MSE for Lasso Regression: {standard_deviation_mse:.4f}")

    elif choice == '3':
        for ts, matrix in timestamp_matrices.items():
            y_pred, y_target, mse = Linear_Reg.test_regression(model, matrix, frequencies, Plot_flag=False)
            # Store the predicted y for this test in a list
            y_pred_list = globals().get("y_pred_list", [])
            y_pred_list.append((ts, y_pred))
            globals()["y_pred_list"] = y_pred_list
            
            # Store the target y for this test in a list
            y_target_list = globals().get("y_target_list", [])
            y_target_list.append((ts, y_target))
            globals()["y_target_list"] = y_target_list
            # Stores mse in a vector to store the results every iteration:
            mse_vector.append(mse)
        # compute average of mse_vector:
        average_mse = np.mean(mse_vector)
        y_target_combined = np.concatenate([y for _, y in y_target_list])
        y_pred_combined = np.concatenate([y for _, y in y_pred_list])
        average_mse_2 = mean_squared_error(y_target_combined, y_pred_combined)

        standard_deviation_2 = np.std(y_target_combined - y_pred_combined)
        standard_deviation_mse = np.std(mse_vector)
        
        print(f"Average MSE for Linear Regression: {average_mse}")
        print(f"Standard Deviation of errors for Linear Regression (using target): {standard_deviation_2:.4f}")
        print(f"Standard Deviation of MSE for Linear Regression: {standard_deviation_mse:.4f}")

    elif choice == '4':
        for ts, matrix in timestamp_matrices.items():
            y_pred, y_target, mse = SVR_Reg.test_svr(model, matrix, frequencies, Plot_flag=False)
            # Store the predicted y for this test in a list
            y_pred_list = globals().get("y_pred_list", [])
            y_pred_list.append((ts, y_pred))
            globals()["y_pred_list"] = y_pred_list
            
            # Store the target y for this test in a list
            y_target_list = globals().get("y_target_list", [])
            y_target_list.append((ts, y_target))
            globals()["y_target_list"] = y_target_list
            # Stores mse in a vector to store the results every iteration:
            mse_vector.append(mse)
        # compute average of mse_vector:
        average_mse = np.mean(mse_vector)
        y_target_combined = np.concatenate([y for _, y in y_target_list])
        y_pred_combined = np.concatenate([y for _, y in y_pred_list])

        average_mse_2 = mean_squared_error(y_target_combined, y_pred_combined)

        standard_deviation_2 = np.std(y_target_combined - y_pred_combined)
        standard_deviation_mse = np.std(mse_vector)
        
        print(f"Average MSE for SVR: {average_mse}")
        print(f"Standard Deviation of errors for SVR (using target): {standard_deviation_2:.4f}")
        print(f"Standard Deviation of MSE for SVR: {standard_deviation_mse:.4f}")
        
    elif choice == '6':
        for ts, matrix in timestamp_matrices.items():
            y_pred, y_target, mse = RandomForest.test_random_forest(model, matrix, frequencies, Plot_flag=False)
            # Store the predicted y for this test in a list
            y_pred_list = globals().get("y_pred_list", [])
            y_pred_list.append((ts, y_pred))
            globals()["y_pred_list"] = y_pred_list
            
            # Store the target y for this test in a list
            y_target_list = globals().get("y_target_list", [])
            y_target_list.append((ts, y_target))
            globals()["y_target_list"] = y_target_list
            # Stores mse in a vector to store the results every iteration:
            mse_vector.append(mse)
        # compute average of mse_vector:
        average_mse = np.mean(mse_vector)
        y_target_combined = np.concatenate([y for _, y in y_target_list])
        y_pred_combined = np.concatenate([y for _, y in y_pred_list])
        average_mse_2 = mean_squared_error(y_target_combined, y_pred_combined)

        standard_deviation_2 = np.std(y_target_combined - y_pred_combined)
        standard_deviation_mse = np.std(mse_vector)
        
        print(f"Average MSE for Random Forest: {average_mse}")
        print(f"Standard Deviation of errors for Random Forest (using target): {standard_deviation_2:.4f}")
        print(f"Standard Deviation of MSE for Random Forest: {standard_deviation_mse:.4f}")

    if choice != '5':
        # Interpolate all y_pred to align them for plotting
        if "y_pred_list" in globals():
            plt.figure(figsize=(12, 12))

            # Subplot 1: Plot all interpolated_y_pred with target from 0 to 1
            plt.subplot(2, 1, 1)
            target_length = min(len(y_pred) for _, y_pred in globals()["y_pred_list"])
            for i, (ts, y_pred) in enumerate(globals()["y_pred_list"], start=1):
                interpolated_y_pred = np.interp(
                    np.linspace(0, len(y_pred) - 1, target_length),
                    np.arange(len(y_pred)),
                    y_pred
                )
                plt.plot(interpolated_y_pred, alpha=0.7)
                # Store y_pred in a matrix for later use
                if i == 1:
                    y_pred_matrix = np.zeros((len(globals()["y_pred_list"]), target_length))
                y_pred_matrix[i - 1, :] = interpolated_y_pred
            # Plot the target as a reference
            target = np.linspace(0, 1, target_length)
            plt.plot(target, label='Target', color='black', linestyle='--', linewidth=2)
            plt.xlabel('Task Completion (%)', fontsize=6)
            plt.ylabel('Estimated Phase (0-1)', fontsize=6)
            plt.xticks(fontsize=6)
            plt.yticks(fontsize=6)
            plt.title('Test Predictions (Normalized)', fontsize=6)
            plt.legend(loc='best', fontsize='x-small', markerscale=0.7)
            plt.grid(True)

            # Subplot 2: Plot mean, std, and target from 0 to 100
            plt.subplot(2, 1, 2)
            mean_y_pred = np.mean(y_pred_matrix, axis=0) * 100
            std_y_pred = np.std(y_pred_matrix, axis=0) * 100
            plt.plot(mean_y_pred, label='Mean Prediction', color='red', linewidth=2)
            plt.fill_between(
                np.arange(target_length),
                mean_y_pred - std_y_pred,
                mean_y_pred + std_y_pred,
                color='red',
                alpha=0.2,
                label='Std Dev'
            )
            # Plot the target as a reference
            target = np.linspace(0, 100, target_length)
            plt.plot(target, label='Target', color='black', linestyle='--', linewidth=2)
            plt.xlabel('Task Completion (%)', fontsize=6)
            plt.ylabel('Estimated Phase (%)', fontsize=6)
            plt.xticks(fontsize=6)
            plt.yticks(fontsize=6)
            plt.title('Mean and Std Dev of Predictions', fontsize=6)
            plt.legend(loc='best', fontsize='x-small', markerscale=0.7)
            plt.grid(True)

            plt.tight_layout()
            plt.show()
            # Save it as SVG file:
            plt.savefig('test_predictions.svg', format='svg', bbox_inches='tight')

        else:
            print("No predictions available to plot.")

        # Plot the MSE for each test
        plt.figure(figsize=(10, 6))
        test_numbers = list(range(1, len(mse_vector) + 1))  # Test numbers starting from 1
        plt.bar(test_numbers, mse_vector, color='skyblue', label='MSE per Test')
        # Plot the mean as a dashed line
        plt.axhline(y=average_mse, color='red', linestyle='--', label=f'Mean MSE: {average_mse:.4f}')
        # Add labels and legend
        plt.xlabel('Test Number')
        plt.ylabel('Mean Squared Error (MSE)')
        plt.title('MSE for Regression Tests')
        plt.xticks(test_numbers)  # Ensure all test numbers are shown on the x-axis
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        # Show the plot
        plt.tight_layout()
        plt.show()
    
    else:
        model_ridge = model['ridge']
        model_lasso = model['lasso']
        model_linear = model['linear']
        model_svr = model['svr']
        model_randomforest = model['randomforest']
        mse_vector_ridge = []  # Initialize mse_vector as an empty list
        mse_vector_lasso = []  # Initialize mse_vector as an empty list
        mse_vector_linear = []  # Initialize mse_vector as an empty list
        mse_vector_svr = []  # Initialize mse_vector as an empty list
        mse_vector_randomforest = []  # Initialize mse_vector as an empty list

        # Perform the test for all the methods taking just the mse
        # Ridge
        mse_vector_ridge = []  # Ensure this is initialized before the loop
        for ts, matrix in timestamp_matrices.items():
            _, mse_ridge = RidgeRegressionCV.test_ridge(model_ridge, matrix, frequencies, Plot_flag=False)
            mse_vector_ridge.append(mse_ridge)
        # Compute average of mse_vector for Ridge Regression
        average_mse_ridge = np.mean(mse_vector_ridge)
        print(f"Average MSE for Ridge Regression: {average_mse_ridge}")

        mse_vector_lasso = []  # Ensure this is initialized before the loop
        for ts, matrix in timestamp_matrices.items():
            _, mse_lasso = LassoRegressionCV.test_lasso(model_lasso, matrix, frequencies, Plot_flag=False)
            mse_vector_lasso.append(mse_lasso)
        # Compute average of mse_vector for Lasso Regression
        average_mse_lasso = np.mean(mse_vector_lasso)
        print(f"Average MSE for Lasso Regression: {average_mse_lasso}")

        mse_vector_linear = []  # Ensure this is initialized before the loop
        for ts, matrix in timestamp_matrices.items():
            _, _, mse_linear = Linear_Reg.test_regression(model_linear, matrix, frequencies, Plot_flag=False)
            mse_vector_linear.append(mse_linear)
        # Compute average of mse_vector for Linear Regression
        average_mse_linear = np.mean(mse_vector_linear)
        print(f"Average MSE for Linear Regression: {average_mse_linear}")

        mse_vector_svr = []  # Ensure this is initialized before the loop
        for ts, matrix in timestamp_matrices.items():
            _, mse_svr = SVR_Reg.test_svr(model_svr, matrix, frequencies, Plot_flag=False)
            mse_vector_svr.append(mse_svr)
        # Compute average of mse_vector for SVR
        average_mse_svr = np.mean(mse_vector_svr)
        print(f"Average MSE for SVR: {average_mse_svr}")

        mse_vector_randomforest = []  # Ensure this is initialized before the loop
        for ts, matrix in timestamp_matrices.items():
            _, mse_randomforest = RandomForest.test_random_forest(model_randomforest, matrix, frequencies, Plot_flag=False)
            mse_vector_randomforest.append(mse_randomforest)
        # Compute average of mse_vector for Random Forest
        average_mse_randomforest = np.mean(mse_vector_randomforest)
        print(f"Average MSE for Random Forest: {average_mse_randomforest}")

        # Prepare data for plots
        methods = ['Ridge', 'Lasso', 'Linear', 'SVR', 'Random Forest']
        average_mses = [average_mse_ridge, average_mse_lasso, average_mse_linear, average_mse_svr, average_mse_randomforest]
        mse_data = [mse_vector_ridge, mse_vector_lasso, mse_vector_linear, mse_vector_svr, mse_vector_randomforest]
        colors = ['blue', 'green', 'orange', 'purple', 'brown']

        # Option 1: Original bar plot (keep this if you want both visualizations)
        plt.figure(figsize=(8, 6))
        plt.bar(methods, average_mses, color=colors)
        plt.xlabel('Regression Methods')
        plt.ylabel('Average MSE')
        plt.title('Comparison of Average MSE Across Regression Methods')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

        # Option 2: Box plot (this is the transformation you requested)
        plt.figure(figsize=(8, 6))
        box = plt.boxplot(mse_data, patch_artist=True, labels=methods, showmeans=True)

        # Customize the box plot to match your reference image
        # Color the boxes
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)

        # Set mean markers as red triangles
        for mean in box['means']:
            mean.set_marker('^')
            mean.set_markerfacecolor('red')
            mean.set_markeredgecolor('red')

        # Set median lines to be white/light colored for better visibility
        for median in box['medians']:
            median.set_color('white')
            median.set_linewidth(1.5)

        plt.xlabel('Regression Methods')
        plt.ylabel('MSE')
        plt.title('Distribution of MSE Across Regression Methods')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()




def create_timestamp_matrices(acc_data, gyro_data, or_data, grouped_indices, segment_choice, frequencies, biasPlot_flag=False, interpPlot_flag=False):
    """
    Create separate matrices for each timestamp, concatenating sensor data along columns.
    
    Parameters:
    -----------
    acc_data : list of DataFrame
        List of accelerometer data frames
    gyro_data : list of DataFrame
        List of gyroscope data frames
    or_data : list of DataFrame
        List of orientation data frames
    grouped_indices : dict
        Dictionary mapping timestamps to indices in the data lists
    biasPlot_flag : bool, optional
        Flag to plot bias correction results
    interpPlot_flag : bool, optional
        Flag to plot interpolation results
        
    Returns:
    --------
    timestamp_matrices : dict
        Dictionary mapping timestamps to their respective matrices
    feature_names : list
        List of feature names in the matrices
    frequencies : dict
        Dictionary of sensor frequencies
    """
    timestamp_matrices = {}
    feature_names = []
    
    # Sort timestamps to ensure chronological order
    sorted_timestamps = sorted(grouped_indices.keys())
    
    for timestamp in sorted_timestamps:
        indices = grouped_indices[timestamp]
        
        # Get the data for this timestamp
        acc = acc_data[indices["acc"]]
        gyro = gyro_data[indices["gyro"]]
        or_data_item = or_data[indices["or"]]

        euler_data_item = quaternion_to_euler(or_data_item, frequencies[2])
        or_data_item = euler_data_item
        
        print(f"Processing data set from timestamp: {timestamp}")
        
        if segment_choice != '5':
            # Apply the segmentation and bias correction
            gyro_processed, acc_processed, or_processed, *_ = BiasAndSegmentation.segmentation_and_bias(
                gyro, acc, or_data_item, segment_choice=segment_choice, timestamp=timestamp, frequencies=frequencies, plot_flag=biasPlot_flag
            )
            
            # Apply interpolation
            gyro_interp, acc_interp, or_interp = interpolate_and_visualize(
                gyro_processed, acc_processed, or_processed, 
                frequencies, plot_flag=interpPlot_flag
            )
            
            # Concatenate features horizontally for this timestamp's matrix
            features = np.concatenate([acc_interp.values, gyro_interp.values, or_interp.values[:, [0]]], axis=1)
            timestamp_matrices[timestamp] = features
            acc_cols = [f"ACC_{col}" for col in acc_interp.columns]
            gyro_cols = [f"GYRO_{col}" for col in gyro_interp.columns]
            or_cols = [f"OR_{col}" for col in or_interp.columns]
            feature_names = acc_cols + gyro_cols + [or_cols[0]]

        else:
            step_data = CyclicSegmentationManager.motion_segmenter(
                gyro, acc, or_data_item, timestamp=timestamp, frequencies=frequencies, plot_flag=biasPlot_flag
            )
            timestamp_matrices = {}
            for step in step_data:
                # Apply interpolation
                gyro_processed = step['gyro']
                acc_processed = step['acc']
                or_processed = step['orientation']
                abs_filtered_gyro_derivative = step['absgyro']

                gyro_processed = pd.concat([gyro_processed, abs_filtered_gyro_derivative], axis=1)

                gyro_interp, acc_interp, or_interp = Interpolation.interpolate_and_visualize(
                    gyro_processed, acc_processed, or_processed, 
                    frequencies, plot_flag=False
                )

                abs_filtered_gyro_derivative_interp = gyro_interp.iloc[:, -1]
                gyro_interp = gyro_interp.iloc[:, :3]

                
                # Concatenate features for X matrix
                features = np.concatenate([acc_interp.values, gyro_interp.values, abs_filtered_gyro_derivative_interp.values.reshape(-1, 1)], axis=1)

                # # Concatenate features for X matrix
                # features = np.concatenate([gyro_interp.values, abs_filtered_gyro_derivative.values], axis=1)

                timestamp_matrices[step['step_number']] = features
            print(f"Detected {len(step_data)} steps")
            acc_cols = [f"ACC_{col}" for col in acc_interp.columns]
            gyro_cols = [f"GYRO_{col}" for col in gyro_interp.columns]
            abs_filtered_gyro_cols = [f"ABSGYRO_{col}" for col in pd.DataFrame(abs_filtered_gyro_derivative_interp).columns]
            or_cols = [f"OR_{col}" for col in or_interp.columns]
            feature_names = acc_cols + gyro_cols + abs_filtered_gyro_cols
            # feature_names = gyro_cols + abs_filtered_gyro_cols
        
        
            
    
    return timestamp_matrices, feature_names

def visualize_timestamp_matrix(matrix, timestamp, feature_names):
    """
    Visualize a single timestamp matrix.
    
    Parameters:
    -----------
    matrix : ndarray
        Matrix of sensor data
    timestamp : str
        Timestamp for this matrix
    feature_names : list
        List of feature names
    """
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Create separate axes for different sensor types
    ax_acc = ax.twinx()
    ax_or = ax.twinx()
    
    # Offset the right spine of ax_or
    ax_or.spines['right'].set_position(('outward', 60))
    
    # Set different colors for different sensor types
    acc_color = 'red'
    gyro_color = 'blue'
    or_color = 'green'
    
    time_steps = np.arange(matrix.shape[0])
    
    # Plot signals with different scales
    for i, name in enumerate(feature_names):
        if 'ACC' in name:
            ax_acc.plot(time_steps, matrix[:, i], color=acc_color, alpha=0.7, linewidth=0.8)
        elif 'GYRO' in name:
            ax.plot(time_steps, matrix[:, i], color=gyro_color, alpha=0.7, linewidth=0.8)
        elif 'OR' in name:
            ax_or.plot(time_steps, matrix[:, i], color=or_color, alpha=0.7, linewidth=0.8)
    
    # Set labels and legend
    ax.set_title(f"Sensor Signals for Timestamp: {timestamp}")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Gyroscope Values", color=gyro_color)
    ax_acc.set_ylabel("Accelerometer Values", color=acc_color)
    ax_or.set_ylabel("Orientation Values", color=or_color)
    
    # Set tick colors
    ax.tick_params(axis='y', labelcolor=gyro_color)
    ax_acc.tick_params(axis='y', labelcolor=acc_color)
    ax_or.tick_params(axis='y', labelcolor=or_color)
    
    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=gyro_color, lw=2, label='Gyroscope'),
        Line2D([0], [0], color=acc_color, lw=2, label='Accelerometer'),
        Line2D([0], [0], color=or_color, lw=2, label='Orientation')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Add grid
    ax.grid(True)
    
    plt.tight_layout()
    plt.show()

