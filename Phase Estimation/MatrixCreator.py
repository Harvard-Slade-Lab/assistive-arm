import BiasAndSegmentation
import Interpolation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
from scipy.interpolate import interp1d
import CyclicSegmentationManager
from Segmentation_Methods import GyroSaggitalSegm
from EulerTransform import quaternion_to_euler

def create_matrices(acc_data, gyro_data, or_data, grouped_indices, segment_choice, frequencies, biasPlot_flag=True, interpPlot_flag=True):
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

        # euler_data_item = quaternion_to_euler(or_data_item, frequencies[2])
        # or_data_item = euler_data_item
        
        
        print(f"Processing data set from timestamp: {timestamp}")

        if segment_choice != '5':
            # Apply the segmentation and bias correction
            gyro_processed, acc_processed, or_processed, *_ = BiasAndSegmentation.segmentation_and_bias(
                gyro, acc, or_data_item, segment_choice=segment_choice, timestamp=timestamp, frequencies=frequencies, plot_flag=biasPlot_flag
            )

            # Apply interpolation
            gyro_interp, acc_interp, or_interp = Interpolation.interpolate_and_visualize(
                gyro_processed, acc_processed, or_processed, 
                frequencies, plot_flag=interpPlot_flag
            )
            
            # Concatenate features for X matrix
            features = np.concatenate([acc_interp.values, gyro_interp.values, or_interp.values[:, :2]], axis=1)
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
                feature_names = acc_cols + gyro_cols + or_cols[:2]
            
        else:
            step_data = CyclicSegmentationManager.motion_segmenter(
                gyro, acc, or_data_item, timestamp=timestamp, frequencies=frequencies, plot_flag=biasPlot_flag
            )
            X1 = []
            Y1 = []
            segment_lengths1 = []

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
                features = np.concatenate([
                    acc_interp.values,
                    gyro_interp.values,
                    abs_filtered_gyro_derivative_interp.values.reshape(-1, 1),
                    or_interp.values[:, :2]
                ], axis=1)


                # # Concatenate features for X matrix
                # features = np.concatenate([or_interp.values], axis=1)

                X1.append(features)
                dataset_length = len(features)
                segment_lengths1.append(dataset_length)

                # Create Y matrix segment
                y = np.linspace(0, 1, dataset_length)
                Y1.append(y)
                
                print(f"Step {step['step_number']}:\n", step['gyro'].head())
                # print(f"Step {step['step_number']} Acc:\n", step['acc'].head())
                # print(f"Step {step['step_number']} Orientation:\n", step['orientation'].head())

            print(f"Detected {len(step_data)} steps")
            acc_cols = [f"ACC_{col}" for col in acc_interp.columns]
            gyro_cols = [f"GYRO_{col}" for col in gyro_interp.columns]
            abs_filtered_gyro_cols = [f"ABSGYRO_{col}" for col in pd.DataFrame(abs_filtered_gyro_derivative_interp).columns]
            or_cols = [f"OR_{col}" for col in or_interp.columns]
            feature_names = acc_cols + gyro_cols + abs_filtered_gyro_cols + or_cols[:2]
            # feature_names = acc_cols + gyro_cols + abs_filtered_gyro_cols
            # Print number of columns in X
            print(f"Number of features in X: {len(X1[0][0])}")
            X.extend(X1)
            Y.extend(Y1)
            segment_lengths.append(segment_lengths1)


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
            
            # Get sensor indices from feature names
            acc_cols = [i for i,name in enumerate(feature_names) if 'ACC' in name]
            gyro_cols = [i for i,name in enumerate(feature_names) if 'GYRO' in name]
            or_cols = [i for i,name in enumerate(feature_names) if 'OR' in name]

            # Get noise scaling factors from user (example values shown)
            acc_noise_percent = float(input("Enter accelerometer noise percentage (e.g., 1.0): ")) / 100
            gyro_noise_percent = float(input("Enter gyroscope noise percentage (e.g., 0.5): ")) / 100
            # or_noise_percent = float(input("Enter orientation noise percentage (e.g., 0.1): ")) / 100

            angle = int(input("Enter rotation angle (in degrees): "))
            warp_min = float(input("  Minimum warp factor (e.g., 0.8 for 80perc speed): "))
            warp_max = float(input("  Maximum warp factor (e.g., 1.2 for 120perc speed): "))
            # Generate angle combinations 
            angles = np.arange(-angle, angle, 2) 
            angle_combinations = list(itertools.product(angles, repeat=3))
            print(f"Number of angle combinations: {len(angle_combinations)}")
            # Modified data augmentation cycle
            for _ in range(num_fictitious):  # Randomly select angle combinations
                rot_angles = angle_combinations[np.random.randint(len(angle_combinations))]  # Random selection

                seg_idx = np.random.randint(0, len(original_X))
                original_segment = original_X[seg_idx].copy()
                
                #print the iteration number
                print(f"Fictitious trial {_ + 1}/{num_fictitious} with angles: {rot_angles}")

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
                
                # Split into sensor components
                acc_data = original_segment[:, acc_cols]
                gyro_data = original_segment[:, gyro_cols]
                or_data = original_segment[:, or_cols]

                # Apply rotation to linear signals
                rotated_acc = acc_data @ R.T  # Rotate accelerometer data
                rotated_gyro = gyro_data.copy()
                rotated_gyro[:, :3] = gyro_data[:, :3] @ R.T  # Rotate only the first three columns of gyroscope data


                # Calculate sensor-specific noise magnitudes [2][5][8]
                acc_scale = np.std(rotated_acc) * acc_noise_percent
                gyro_scale = np.std(rotated_gyro) * gyro_noise_percent

                # Generate colored noise [3][7]
                acc_noise = np.random.normal(0, acc_scale, rotated_acc.shape)
                gyro_noise = np.random.normal(0, gyro_scale, rotated_gyro.shape)

                # Apply noise to components
                rotated_acc += acc_noise
                rotated_gyro += gyro_noise
                
                # Apply rotation to quaternions (Hamilton product)
                rotated_or = np.array([quaternion_multiply(q_rot, q) for q in or_data])
                rotated_or /= np.linalg.norm(rotated_or, axis=1, keepdims=True)  # Normalize


                time_warp_range = (warp_min, warp_max)
                warp_factor = np.random.uniform(time_warp_range[0], time_warp_range[1])

                # Apply sensor-specific warping
                rotated_acc = apply_time_warping(rotated_acc, warp_factor, 'acc')
                rotated_gyro = apply_time_warping(rotated_gyro, warp_factor, 'gyro') 
                rotated_or = apply_time_warping(rotated_or, warp_factor, 'or')
                # rotated_gyro[:, 3] = np.sqrt(
                #     rotated_gyro[:, 0] ** 2 + 
                #     rotated_gyro[:, 1] ** 2 + 
                #     rotated_gyro[:, 2] ** 2
                # )
                # rotated_gyro[:, 3] = np.sqrt(
                #     rotated_gyro[:, 0] ** 2 + 
                #     rotated_gyro[:, 1] ** 2 + 
                #     rotated_gyro[:, 2] ** 2
                # )

                # Recombine rotated components
                fictitious_segment = np.hstack([rotated_acc, rotated_gyro, rotated_or])

                print(f"Fictitious segment shape: {fictitious_segment.shape}")
                
                # Append to matrices
                X.append(fictitious_segment)
                sorted_timestamps.append(f"rotated_{rot_angles}_{seg_idx}")

                # Correct implementation:
                new_length = rotated_acc.shape[0]
                segment_lengths.append(new_length)
                y_fictitious = np.linspace(0, 1, new_length)
                Y.append(y_fictitious)


    # Stack matrices vertically
    X_matrix = np.vstack(X)
    Y_matrix = np.concatenate(Y)
    
    
    return X_matrix, Y_matrix, sorted_timestamps, segment_lengths, feature_names

# Visualization function remains unchanged
def visualize_matrices(X, Y, timestamps, segment_choice, segment_lengths, feature_names, frequencies):
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
    
    # Calculate time steps using the minimum frequency
    min_frequency = min(frequencies)
    time_steps = np.arange(X.shape[0]) / min_frequency
    
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
    ax2.plot(time_steps, Y, label="Progress Indicators")
    
    ax2.set_title("Dataset Progress Indicators (0 to 1)")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Progress")
    ax2.grid(True)
    
    # Add vertical lines to separate different datasets
    current_pos = np.cumsum(segment_lengths)[:-1]  # Exclude the last cumulative position
    for pos in current_pos:
        ax1.axvline(x=pos / min_frequency, color='r', linestyle='--', alpha=0.5)
        ax2.axvline(x=pos / min_frequency, color='r', linestyle='--', alpha=0.5)
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


def apply_time_warping(original_segment, warp_factor, sensor_type):
    """Apply physics-aware time warping with sensor-specific scaling."""
    original_length = original_segment.shape[0]
    original_time = np.linspace(0, 1, original_length)
    
    # Calculate new length based on warp factor
    warped_length = int(original_length * warp_factor)
    warped_time = np.linspace(0, 1, warped_length)
    
    # Create interpolation function
    interp_fn = interp1d(original_time, original_segment, axis=0, 
                        kind='quadratic', fill_value='extrapolate')
    
    # Apply temporal interpolation
    warped_segment = interp_fn(warped_time)
    
    # Physics-based scaling
    if sensor_type == 'acc':
        # Acceleration scales with 1/warp_factor² (2nd derivative of position)
        warped_segment *= (1 / warp_factor)**2
    elif sensor_type == 'gyro':
        # Angular velocity scales with 1/warp_factor (1st derivative of orientation)
        warped_segment[:, :3] *= (1 / warp_factor)  # XYZ components
        if warped_segment.shape[1] > 3:  # Preserve magnitude if present
            warped_segment[:, 3] = np.linalg.norm(warped_segment[:, :3], axis=1)
    elif sensor_type == 'or':
        # Maintain unit quaternions with Slerp interpolation
        warped_segment = np.array([slerp(original_time, original_segment, t) 
                                  for t in warped_time])
    
    return warped_segment

def slerp(times, quats, target_time):
    """Spherical linear interpolation for quaternions."""
    idx = np.searchsorted(times, target_time, side='right') - 1
    if idx == len(times) - 1:
        return quats[-1]
    
    t0, t1 = times[idx], times[idx+1]
    q0, q1 = quats[idx], quats[idx+1]
    
    # Ensure quaternions are normalized
    q0 /= np.linalg.norm(q0)
    q1 /= np.linalg.norm(q1)
    
    # Compute cosine of angle between quaternions
    cos_omega = np.dot(q0, q1)
    
    # If negative, flip to take shorter path
    if cos_omega < 0:
        q1 = -q1
        cos_omega = -cos_omega
    
    # Linear interpolation for small angles
    if cos_omega > 0.9995:
        result = q0 + (target_time - t0) * (q1 - q0)/(t1 - t0)
        return result / np.linalg.norm(result)
    
    # Slerp interpolation
    omega = np.arccos(cos_omega)
    so = np.sin(omega)
    alpha = (target_time - t0)/(t1 - t0)
    return (np.sin((1-alpha)*omega)/so)*q0 + (np.sin(alpha*omega)/so)*q1
