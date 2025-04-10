import BiasAndSegmentation
import Interpolation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools

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
        # try:
        #     num_fictitious = int(input("Enter the number of fictitious trials to add: "))
        # except ValueError:
        #     print("Invalid input. No fictitious trials added.")
        #     num_fictitious = 0
        num_fictitious = 1  # For testing purposes, set to 1
        # Modified noise generation section with sensor-specific scaling
        if num_fictitious > 0:
            original_X = X.copy()
            
            # Get sensor indices from feature names
            acc_cols = [i for i,name in enumerate(feature_names) if 'ACC' in name]
            gyro_cols = [i for i,name in enumerate(feature_names) if 'GYRO' in name]
            or_cols = [i for i,name in enumerate(feature_names) if 'OR' in name]

            # # Get noise scaling factors from user (example values shown)
            # acc_noise_percent = float(input("Enter accelerometer noise percentage (e.g., 1.0): ")) / 100
            # gyro_noise_percent = float(input("Enter gyroscope noise percentage (e.g., 0.5): ")) / 100
            # or_noise_percent = float(input("Enter orientation noise percentage (e.g., 0.1): ")) / 100

            # Generate angle combinations (-5°, -3°, -1°, 1°, 3°, 5° for each axis)
            angles = np.arange(-5, 5, 2)  # [-5, -3, -1, 1, 3, 5]
            angle_combinations = list(itertools.product(angles, repeat=3))
            print(f"Number of angle combinations: {len(angle_combinations)}")
            # Modified data augmentation cycle
            for rot_angles in angle_combinations:  # Each combination contains (x_angle, y_angle, z_angle)
                # Generate rotation matrices for current angles
                R_x = rotation_matrix_x(rot_angles[0])
                R_y = rotation_matrix_y(rot_angles[1])
                R_z = rotation_matrix_z(rot_angles[2])
                
                # Combine rotations: R = R_z @ R_y @ R_x (applied in x->y->z order)
                R = R_z @ R_y @ R_x
                
                # Generate rotation quaternion for orientations
                qx = rotation_quaternion_x(rot_angles[0])
                qy = rotation_quaternion_y(rot_angles[1])
                qz = rotation_quaternion_z(rot_angles[2])
                q_rot = quaternion_multiply(qz, quaternion_multiply(qy, qx))
                q_rot /= np.linalg.norm(q_rot)  # Ensure unit quaternion

                for seg_idx in range(len(original_X)):
                    original_segment = original_X[seg_idx].copy()
                    
                    # Split into sensor components
                    acc_data = original_segment[:, acc_cols]
                    gyro_data = original_segment[:, gyro_cols]
                    or_data = original_segment[:, or_cols]

                    # Apply rotation to linear signals
                    rotated_acc = acc_data @ R.T  # Rotate accelerometer data
                    rotated_gyro = gyro_data.copy()
                    rotated_gyro[:, :3] = gyro_data[:, :3] @ R.T  # Rotate only the first three columns of gyroscope data
                    rotated_gyro[:, 3] = np.sqrt(
                        rotated_gyro[:, 0] ** 2 + 
                        rotated_gyro[:, 1] ** 2 + 
                        rotated_gyro[:, 2] ** 2
                    )
                    
                    # Apply rotation to quaternions (Hamilton product)
                    rotated_or = np.array([quaternion_multiply(q_rot, q) for q in or_data])
                    rotated_or /= np.linalg.norm(rotated_or, axis=1, keepdims=True)  # Normalize

                    # Recombine rotated components
                    fictitious_segment = np.hstack([rotated_acc, rotated_gyro, rotated_or])
                    
                    # Append to matrices
                    X.append(fictitious_segment)
                    segment_lengths.append(original_segment.shape[0])
                    sorted_timestamps.append(f"rotated_{rot_angles}_{seg_idx}")

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


# Functions to create rotation matrices for 3D transformations
def rotation_matrix_x(angle_degrees):
    """Rotation matrix for X-axis rotation"""
    angle_radians = np.radians(angle_degrees)
    cos_angle = np.cos(angle_radians)
    sin_angle = np.sin(angle_radians)
    return np.array([[1, 0, 0],
                     [0, cos_angle, -sin_angle],
                     [0, sin_angle, cos_angle]])

def rotation_matrix_y(angle_degrees):
    """Rotation matrix for Y-axis rotation"""
    angle_radians = np.radians(angle_degrees)
    cos_angle = np.cos(angle_radians)
    sin_angle = np.sin(angle_radians)
    return np.array([[cos_angle, 0, sin_angle],
                     [0, 1, 0],
                     [-sin_angle, 0, cos_angle]])

def rotation_matrix_z(angle_degrees):
    """Rotation matrix for Z-axis rotation"""
    angle_radians = np.radians(angle_degrees)
    cos_angle = np.cos(angle_radians)
    sin_angle = np.sin(angle_radians)
    return np.array([[cos_angle, -sin_angle, 0],
                     [sin_angle, cos_angle, 0],
                     [0, 0, 1]])

def rotation_quaternion_x(angle_degrees):
    """Create a unit quaternion for rotation around X-axis"""
    angle_radians = np.radians(angle_degrees) / 2  # Half-angle for quaternions
    w = np.cos(angle_radians)
    x = np.sin(angle_radians)
    return np.array([w, x, 0.0, 0.0])

def rotation_quaternion_y(angle_degrees):
    """Create a unit quaternion for rotation around Y-axis"""
    angle_radians = np.radians(angle_degrees) / 2
    w = np.cos(angle_radians)
    y = np.sin(angle_radians)
    return np.array([w, 0.0, y, 0.0])

def rotation_quaternion_z(angle_degrees):
    """Create a unit quaternion for rotation around Z-axis"""
    angle_radians = np.radians(angle_degrees) / 2
    w = np.cos(angle_radians)
    z = np.sin(angle_radians)
    return np.array([w, 0.0, 0.0, z])

def quaternion_multiply(q1, q2):
    """
    Hamilton product of two quaternions (q1 * q2)
    Input format: [w, x, y, z]
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return np.array([w, x, y, z])