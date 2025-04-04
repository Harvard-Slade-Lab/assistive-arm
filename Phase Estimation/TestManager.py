# TimestampMatrixCreator.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askdirectory
import BiasAndSegmentation
from Interpolation import interpolate_and_visualize
import RidgeRegressionCV
import LassoRegressionCV
import Linear_Reg
import SVR_Reg

def handle_test_decision(choice, model, frequencies):
    """Handle user decision about testing"""
    if choice != '5':
        test_decision = input("\nDo you want to perform the test? (yes/no): ").lower()
        if test_decision == 'yes':
            execute_test(choice, model, frequencies)


def execute_test(choice, model, frequencies):
    # Select folder
    folder_path = select_folder()
    if not folder_path:
        print("No folder selected. Exiting.")
        return
    
    # Load and process files
    acc_data, gyro_data, or_data, acc_files, gyro_files, or_files = load_and_process_files(folder_path)
    
    # Group files by timestamp
    grouped_indices = group_files_by_timestamp(acc_files, gyro_files, or_files)
    
    if not grouped_indices:
        print("No complete groups of files found. Exiting.")
        return
    
    # Create matrices for each timestamp
    timestamp_matrices, feature_names, frequencies = create_timestamp_matrices(
        acc_data, gyro_data, or_data, grouped_indices, 
        biasPlot_flag=False, interpPlot_flag=False
    )
    
    # Print information about created matrices
    print(f"\nCreated {len(timestamp_matrices)} matrices for different timestamps:")
    for ts, matrix in timestamp_matrices.items():
        print(f"Timestamp: {ts}, Matrix shape: {matrix.shape}")


    if choice == '1':
        mse_vector_ridge = []  # Initialize mse_vector as an empty list
        for ts, matrix in timestamp_matrices.items():
            _, mse = RidgeRegressionCV.test_ridge(model, matrix, frequencies)
            # Stores mse in a vector to store the results every iteration:
            mse_vector_ridge.append(mse)
        # compute average of mse_vector:
        average_mse_ridge = np.mean(mse_vector_ridge)
        print(f"Average MSE for Ridge Regression: {average_mse_ridge}")

    elif choice == '2':
        mse_vector_lasso = []  # Initialize mse_vector as an empty list
        for ts, matrix in timestamp_matrices.items():
            _, mse = LassoRegressionCV.test_lasso(model, matrix, frequencies)
            # Stores mse in a vector to store the results every iteration:
            mse_vector_lasso.append(mse)
        # compute average of mse_vector:
        average_mse_lasso = np.mean(mse_vector_lasso)
        print(f"Average MSE for Lasso Regression: {average_mse_lasso}")
    elif choice == '3':
        mse_vector_linear = []  # Initialize mse_vector as an empty list
        for ts, matrix in timestamp_matrices.items():
            _, _, mse = Linear_Reg.test_regression(model, matrix, frequencies)
            # Stores mse in a vector to store the results every iteration:
            mse_vector_linear.append(mse)
        # compute average of mse_vector:
        average_mse_linear = np.mean(mse_vector_linear)
        print(f"Average MSE for Linear Regression: {average_mse_linear}")
    elif choice == '4':
        mse_vector_svr = []  # Initialize mse_vector as an empty list
        for ts, matrix in timestamp_matrices.items():
            _, mse = SVR_Reg.test_svr(model, matrix, frequencies)
            # Stores mse in a vector to store the results every iteration:
            mse_vector_svr.append(mse)
        # compute average of mse_vector:
        average_mse_svr = np.mean(mse_vector_svr)
        print(f"Average MSE for SVR: {average_mse_svr}")


# Function to select folder
def select_folder():
    Tk().withdraw()  # Close the root window
    folder_path = askdirectory(title="Select Folder")
    return folder_path

# Function to load and process files
def load_and_process_files(folder_path):
    acc_files = []
    gyro_files = []
    or_files = []

    # Iterate through files in the folder
    for file_name in os.listdir(folder_path):
        if "ACC_Profile" in file_name:
            acc_files.append(file_name)
        elif "GYRO_Profile" in file_name:
            gyro_files.append(file_name)
        elif "OR_Profile" in file_name and "OR_Debug_Profile" not in file_name:
            or_files.append(file_name)
    
    # Sort files
    acc_files.sort()
    gyro_files.sort()
    or_files.sort()
    
    # Load and process ACC files (and remove last 3 columns)
    acc_data = []
    for acc_file in acc_files:
        df = pd.read_csv(os.path.join(folder_path, acc_file))
        df = df.iloc[:, :-3]  # Remove last 3 columns
        acc_data.append(df)
    
    # Load GYRO files
    gyro_data = []
    for gyro_file in gyro_files:
        df = pd.read_csv(os.path.join(folder_path, gyro_file))
        gyro_data.append(df)
    
    # Load OR files
    or_data = []
    for or_file in or_files:
        df = pd.read_csv(os.path.join(folder_path, or_file))
        or_data.append(df)
    
    return acc_data, gyro_data, or_data, acc_files, gyro_files, or_files

# Function to extract timestamp from filename
def extract_timestamp(filename):
    parts = filename.split("_")
    for part in parts:
        if len(part) == 14 and part.isdigit():  # YYYYMMDDHHMMSS format
            return part
    return None

# Function to group files by timestamp
def group_files_by_timestamp(acc_files, gyro_files, or_files):
    grouped_indices = {}
    
    # Process ACC files
    for i, file_name in enumerate(acc_files):
        timestamp = extract_timestamp(file_name)
        if timestamp not in grouped_indices:
            grouped_indices[timestamp] = {"acc": None, "gyro": None, "or": None}
        grouped_indices[timestamp]["acc"] = i
    
    # Process GYRO files
    for i, file_name in enumerate(gyro_files):
        timestamp = extract_timestamp(file_name)
        if timestamp not in grouped_indices:
            grouped_indices[timestamp] = {"acc": None, "gyro": None, "or": None}
        grouped_indices[timestamp]["gyro"] = i
    
    # Process OR files
    for i, file_name in enumerate(or_files):
        timestamp = extract_timestamp(file_name)
        if timestamp not in grouped_indices:
            grouped_indices[timestamp] = {"acc": None, "gyro": None, "or": None}
        grouped_indices[timestamp]["or"] = i
    
    # Filter out incomplete groups
    complete_groups = {ts: indices for ts, indices in grouped_indices.items() 
                      if indices["acc"] is not None and indices["gyro"] is not None and indices["or"] is not None}
    
    return complete_groups

def create_timestamp_matrices(acc_data, gyro_data, or_data, grouped_indices, biasPlot_flag=False, interpPlot_flag=False):
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
    
    # Get frequencies (only need to do this once)
    frequencies = BiasAndSegmentation.sensors_frequencies()
    
    for timestamp in sorted_timestamps:
        indices = grouped_indices[timestamp]
        
        # Get the data for this timestamp
        acc = acc_data[indices["acc"]]
        gyro = gyro_data[indices["gyro"]]
        or_data_item = or_data[indices["or"]]
        
        print(f"Processing data set from timestamp: {timestamp}")
        
        # Apply the segmentation and bias correction
        gyro_processed, acc_processed, or_processed, *_ = BiasAndSegmentation.segmentation_and_bias(
            gyro, acc, or_data_item, frequencies, plot_flag=biasPlot_flag
        )
        
        # Apply interpolation
        gyro_interp, acc_interp, or_interp = interpolate_and_visualize(
            gyro_processed, acc_processed, or_processed, 
            frequencies, plot_flag=interpPlot_flag
        )
        
        # Concatenate features horizontally for this timestamp's matrix
        features = np.concatenate([acc_interp.values, gyro_interp.values, or_interp.values], axis=1)
        timestamp_matrices[timestamp] = features
        
        # Store feature names (first time only)
        if not feature_names:
            acc_cols = [f"ACC_{col}" for col in acc_interp.columns]
            gyro_cols = [f"GYRO_{col}" for col in gyro_interp.columns]
            or_cols = [f"OR_{col}" for col in or_interp.columns]
            feature_names = acc_cols + gyro_cols + or_cols
    
    return timestamp_matrices, feature_names, frequencies

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

def main():
    """
    Main function to execute the script.
    """
    # Select folder
    folder_path = select_folder()
    if not folder_path:
        print("No folder selected. Exiting.")
        return
    
    # Load and process files
    acc_data, gyro_data, or_data, acc_files, gyro_files, or_files = load_and_process_files(folder_path)
    
    # Group files by timestamp
    grouped_indices = group_files_by_timestamp(acc_files, gyro_files, or_files)
    
    if not grouped_indices:
        print("No complete groups of files found. Exiting.")
        return
    
    # Create matrices for each timestamp
    timestamp_matrices, feature_names, frequencies = create_timestamp_matrices(
        acc_data, gyro_data, or_data, grouped_indices, 
        biasPlot_flag=False, interpPlot_flag=False
    )
    
    # Print information about created matrices
    print(f"\nCreated {len(timestamp_matrices)} matrices for different timestamps:")
    for ts, matrix in timestamp_matrices.items():
        print(f"Timestamp: {ts}, Matrix shape: {matrix.shape}")
    
    # Example: Visualize a matrix (user can choose which one to visualize)
    visualize_option = input("\nWould you like to visualize a matrix? (yes/no): ")
    if visualize_option.lower() == 'yes':
        print("\nAvailable timestamps:")
        for i, ts in enumerate(sorted(timestamp_matrices.keys())):
            print(f"{i+1}. {ts}")
        
        try:
            idx = int(input("\nEnter the number of the timestamp to visualize: ")) - 1
            timestamps = sorted(timestamp_matrices.keys())
            if 0 <= idx < len(timestamps):
                selected_ts = timestamps[idx]
                visualize_timestamp_matrix(timestamp_matrices[selected_ts], selected_ts, feature_names)
            else:
                print("Invalid selection.")
        except ValueError:
            print("Invalid input.")
    
    # Save matrices to files
    save_option = input("\nWould you like to save matrices to files? (yes/no): ")
    if save_option.lower() == 'yes':
        # Create output directory
        output_dir = os.path.join(folder_path, "timestamp_matrices")
        os.makedirs(output_dir, exist_ok=True)
        
        for ts, matrix in timestamp_matrices.items():
            # Create DataFrame with feature names
            df = pd.DataFrame(matrix, columns=feature_names)
            
            # Save to CSV
            output_file = os.path.join(output_dir, f"matrix_{ts}.csv")
            df.to_csv(output_file, index=False)
            
            # Also save as numpy array for convenience
            np_output_file = os.path.join(output_dir, f"matrix_{ts}.npy")
            np.save(np_output_file, matrix)
            
        print(f"\nMatrices saved to directory: {output_dir}")

if __name__ == "__main__":
    main()
