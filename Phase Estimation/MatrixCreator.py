import BiasAndSegmentation
import Interpolation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def create_matrices(acc_data, gyro_data, or_data, grouped_indices, biasPlot_flag=True, interpPlot_flag=True):
    X = []
    Y = []
    segment_lengths = []
    feature_names = []
    
    # Sort timestamps to ensure chronological order
    sorted_timestamps = sorted(grouped_indices.keys())
    executed = False
    for timestamp in sorted_timestamps:
        indices = grouped_indices[timestamp]
        
        # Get the data for this timestamp
        acc = acc_data[indices["acc"]]
        gyro = gyro_data[indices["gyro"]]
        or_data_item = or_data[indices["or"]]
        
        print(f"Processing data set from timestamp: {timestamp}")
        
        if not executed:
            frequencies = BiasAndSegmentation.sensors_frequencies()
        executed = True

        # Apply the segmentation and bias correction
        gyro_processed, acc_processed, or_processed, *_ = BiasAndSegmentation.segmentation_and_bias(
            gyro, acc, or_data_item, frequencies, plot_flag=biasPlot_flag
        )
        
        # Apply interpolation
        gyro_interp, acc_interp, or_interp = Interpolation.interpolate_and_visualize(
            gyro_processed, acc_processed, or_processed, 
            frequencies, plot_flag=interpPlot_flag
        )
        
        # Concatenate features for X matrix
        features = np.concatenate([acc_interp.values, gyro_interp.values, or_interp.values], axis=1)
        X.append(features)
        
        # Create Y matrix segment
        dataset_length = len(features)
        y = np.linspace(0, 1, dataset_length)
        Y.append(y)
        
        segment_lengths.append(dataset_length)
        
        # Store feature names (first time only)
        if not feature_names:
            acc_cols = [f"ACC_{col}" for col in acc_interp.columns]
            gyro_cols = [f"GYRO_{col}" for col in gyro_interp.columns]
            or_cols = [f"OR_{col}" for col in or_interp.columns]
            feature_names = acc_cols + gyro_cols + or_cols

    # Fictitious trials generation
    add_fictitious = input("Do you want to add fictitious trials? (yes/no): ").strip().lower()
    if add_fictitious == 'yes':
        try:
            num_fictitious = int(input("Enter the number of fictitious trials to add: "))
        except ValueError:
            print("Invalid input. No fictitious trials added.")
            num_fictitious = 0
        
        # Modified noise generation section with sensor-specific scaling
        if num_fictitious > 0:
            original_X = X.copy()
            original_segment_lengths = segment_lengths.copy()
            
            # Get sensor indices from feature names
            acc_cols = [i for i,name in enumerate(feature_names) if 'ACC' in name]
            gyro_cols = [i for i,name in enumerate(feature_names) if 'GYRO' in name]
            or_cols = [i for i,name in enumerate(feature_names) if 'OR' in name]

            # Get noise scaling factors from user (example values shown)
            acc_noise_percent = float(input("Enter accelerometer noise percentage (e.g., 1.0): ")) / 100
            gyro_noise_percent = float(input("Enter gyroscope noise percentage (e.g., 0.5): ")) / 100
            or_noise_percent = float(input("Enter orientation noise percentage (e.g., 0.1): ")) / 100

            for i in range(num_fictitious):
                seg_idx = np.random.randint(0, len(original_X))
                original_segment = original_X[seg_idx].copy()
                
                # Split into sensor components
                acc_data = original_segment[:, acc_cols]
                gyro_data = original_segment[:, gyro_cols]
                or_data = original_segment[:, or_cols]

                # Calculate sensor-specific noise magnitudes [2][5][8]
                acc_scale = np.std(acc_data) * acc_noise_percent
                gyro_scale = np.std(gyro_data) * gyro_noise_percent
                or_scale = np.std(or_data) * or_noise_percent

                # Generate colored noise [3][7]
                acc_noise = np.random.normal(0, acc_scale, acc_data.shape)
                gyro_noise = np.random.normal(0, gyro_scale, gyro_data.shape)
                or_noise = np.random.normal(0, or_scale, or_data.shape)

                # Apply noise to components
                acc_data += acc_noise
                gyro_data += gyro_noise
                or_data += or_noise

                # Recombine noisy components [6]
                fictitious_segment = np.hstack([acc_data, gyro_data, or_data])
                
                # Append to matrices
                X.append(fictitious_segment)
                segment_lengths.append(original_segment.shape[0])
                sorted_timestamps.append(f"fictitious_{i+1}")
                
                # Create matching Y segment
                y_fictitious = np.linspace(0, 1, original_segment.shape[0])
                Y.append(y_fictitious)

    # Stack matrices vertically
    X_matrix = np.vstack(X)
    Y_matrix = np.concatenate(Y)
    
    return X_matrix, Y_matrix, sorted_timestamps, segment_lengths, feature_names, frequencies

# Visualization function remains unchanged
def visualize_matrices(X, Y, timestamps, segment_lengths, feature_names):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    # Create separate axes for different sensor types
    ax1_acc = ax1.twinx()
    ax1_or = ax1.twinx()
    
    # Offset the right spine of ax1_or
    ax1_or.spines['right'].set_position(('outward', 60))
    
    # Set different colors for different sensor types
    acc_color = 'red'
    gyro_color = 'blue'
    or_color = 'green'
    
    time_steps = np.arange(X.shape[0])
    
    # Plot concatenated signals with different scales
    for i, name in enumerate(feature_names):
        if 'ACC' in name:
            ax1_acc.plot(time_steps, X[:, i], color=acc_color, alpha=0.7, linewidth=0.8)
        elif 'GYRO' in name:
            ax1.plot(time_steps, X[:, i], color=gyro_color, alpha=0.7, linewidth=0.8)
        elif 'OR' in name:
            ax1_or.plot(time_steps, X[:, i], color=or_color, alpha=0.7, linewidth=0.8)
    
    # Set labels and legend
    ax1.set_title("Concatenated Sensor Signals")
    ax1.set_ylabel("Gyroscope Values", color=gyro_color)
    ax1_acc.set_ylabel("Accelerometer Values", color=acc_color)
    ax1_or.set_ylabel("Orientation Values", color=or_color)
    
    # Set tick colors
    ax1.tick_params(axis='y', labelcolor=gyro_color)
    ax1_acc.tick_params(axis='y', labelcolor=acc_color)
    ax1_or.tick_params(axis='y', labelcolor=or_color)
    
    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=gyro_color, lw=2, label='Gyroscope'),
        Line2D([0], [0], color=acc_color, lw=2, label='Accelerometer'),
        Line2D([0], [0], color=or_color, lw=2, label='Orientation')
    ]
    ax1.legend(handles=legend_elements, loc='upper right')
    
    # Add grid
    ax1.grid(True)
    
    # Plot progress indicators (Y matrix)
    current_pos = 0
    for ts, length in zip(timestamps, segment_lengths):
        segment_range = np.arange(current_pos, current_pos + length)
        segment_values = np.linspace(0, 1, length)
        ax2.plot(segment_range, segment_values, label=f"Segment {ts}")
        current_pos += length
    
    ax2.set_title("Dataset Progress Indicators (0 to 1)")
    ax2.set_xlabel("Time Steps")
    ax2.set_ylabel("Progress")
    ax2.grid(True)
    
    # Add vertical lines to separate different datasets
    current_pos = 0
    for ts, length in zip(timestamps, segment_lengths):
        current_pos += length
        
        if current_pos < X.shape[0]:  # Don't draw a line after the last dataset
            ax1.axvline(x=current_pos, color='r', linestyle='--', alpha=0.5)
            ax2.axvline(x=current_pos, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()