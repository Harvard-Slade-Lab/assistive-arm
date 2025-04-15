import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tkinter.filedialog import askdirectory
from Interpolation import interpolate_and_visualize
import DataLoader
import MatrixCreator
from scipy.interpolate import CubicSpline, interp1d
from scipy.signal import correlate

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, interp1d
from scipy.signal import correlate

def synchronize_multi_sensor_data(gyro_data_trimmed, acc_data_trimmed, or_data_trimmed, 
                               frequencies, start_idx, end_idx,
                               target_freq=None, interpolation_method='cubic',
                               check_alignment=True):
    """
    Synchronize data from multiple sensors with different sampling frequencies.
    
    Parameters:
    gyro_data_trimmed (DataFrame): Trimmed gyroscope data
    acc_data_trimmed (DataFrame): Trimmed accelerometer data
    or_data_trimmed (DataFrame): Trimmed orientation data
    frequencies (list): Sampling frequencies [gyro, acc, orientation]
    start_idx, end_idx (int): Indices for the segmented motion in gyro data
    target_freq (float, optional): Target frequency for synchronization (default: max of input frequencies)
    interpolation_method (str): 'cubic' or 'linear' interpolation
    check_alignment (bool): Whether to check and adjust for sensor misalignment
    
    Returns:
    tuple: (gyro_sync, acc_sync, or_sync, common_time)
    """
    # Calculate indices for other sensors based on time scaling
    start_idx_acc = int(start_idx * frequencies[1] / frequencies[0])
    end_idx_acc = int(end_idx * frequencies[1] / frequencies[0])
    
    start_idx_or = int(start_idx * frequencies[2] / frequencies[0])
    end_idx_or = int(end_idx * frequencies[2] / frequencies[0])
    
    # Safety checks to prevent index errors
    start_idx_acc = max(0, start_idx_acc)
    end_idx_acc = min(len(acc_data_trimmed), end_idx_acc)
    start_idx_or = max(0, start_idx_or)
    end_idx_or = min(len(or_data_trimmed), end_idx_or)
    
    # Extract segmented data
    gyro_data_segment = gyro_data_trimmed.iloc[start_idx:end_idx].reset_index(drop=True)
    acc_data_segment = acc_data_trimmed.iloc[start_idx_acc:end_idx_acc].reset_index(drop=True)
    or_data_segment = or_data_trimmed.iloc[start_idx_or:end_idx_or].reset_index(drop=True)
    
    # Create time arrays for each sensor's segment
    time_gyro = np.arange(len(gyro_data_segment)) / frequencies[0]
    time_acc = np.arange(len(acc_data_segment)) / frequencies[1]
    time_or = np.arange(len(or_data_segment)) / frequencies[2]
    
    # Check for temporal misalignment if requested
    time_shifts = [0, 0, 0]  # [gyro, acc, or]
    
    if check_alignment and len(gyro_data_segment) > 0 and len(acc_data_segment) > 0 and len(or_data_segment) > 0:
        print("Checking sensor alignment...")
        try:
            # Compute magnitude for all sensors
            gyro_mag = np.sqrt(np.sum(gyro_data_segment.iloc[:, :3]**2, axis=1))
            acc_mag = np.sqrt(np.sum(acc_data_segment**2, axis=1))
            or_mag = np.sqrt(np.sum(or_data_segment**2, axis=1))
            
            # Resample to common frequency for correlation
            common_freq = min(frequencies)
            if len(time_gyro) > 0 and time_gyro[-1] > 0:
                t_gyro_resampled = np.arange(0, time_gyro[-1], 1/common_freq)
                gyro_mag_interp = interp1d(time_gyro, gyro_mag, bounds_error=False, fill_value=0)(t_gyro_resampled)
            else:
                t_gyro_resampled = np.array([])
                gyro_mag_interp = np.array([])
                
            if len(time_acc) > 0 and time_acc[-1] > 0:
                t_acc_resampled = np.arange(0, time_acc[-1], 1/common_freq)
                acc_mag_interp = interp1d(time_acc, acc_mag, bounds_error=False, fill_value=0)(t_acc_resampled)
            else:
                t_acc_resampled = np.array([])
                acc_mag_interp = np.array([])
                
            if len(time_or) > 0 and time_or[-1] > 0:
                t_or_resampled = np.arange(0, time_or[-1], 1/common_freq)
                or_mag_interp = interp1d(time_or, or_mag, bounds_error=False, fill_value=0)(t_or_resampled)
            else:
                t_or_resampled = np.array([])
                or_mag_interp = np.array([])
            
            # Compute cross-correlation to find time shift (using gyro as reference)
            if len(gyro_mag_interp) > 0 and len(acc_mag_interp) > 0:
                corr_acc = correlate(gyro_mag_interp, acc_mag_interp, mode='full')
                shift_acc = np.argmax(corr_acc) - (len(acc_mag_interp) - 1)
                time_shifts[1] = shift_acc / common_freq
                print(f"Detected time shift between gyro and acc: {time_shifts[1]:.4f} seconds")
            
            if len(gyro_mag_interp) > 0 and len(or_mag_interp) > 0:
                corr_or = correlate(gyro_mag_interp, or_mag_interp, mode='full')
                shift_or = np.argmax(corr_or) - (len(or_mag_interp) - 1)
                time_shifts[2] = shift_or / common_freq
                print(f"Detected time shift between gyro and orientation: {time_shifts[2]:.4f} seconds")
        except Exception as e:
            print(f"Alignment check failed: {e}. Proceeding without alignment correction.")
    
    # Adjust time arrays with detected shifts
    time_acc = time_acc + time_shifts[1]
    time_or = time_or + time_shifts[2]
    
    # Determine target frequency and create common time base
    if target_freq is None:
        target_freq = max(frequencies)
    
    # Ensure we're using the longest time duration
    t_end = max(time_gyro[-1] if len(time_gyro) > 0 else 0, 
                time_acc[-1] if len(time_acc) > 0 else 0, 
                time_or[-1] if len(time_or) > 0 else 0)
    
    if t_end > 0:
        common_time = np.arange(0, t_end, 1/target_freq)
    else:
        # Fallback if we have no valid time data
        common_time = np.array([0])
    
    # Initialize synchronized dataframes
    gyro_sync = pd.DataFrame(index=range(len(common_time)), columns=gyro_data_segment.columns)
    acc_sync = pd.DataFrame(index=range(len(common_time)), columns=acc_data_segment.columns)
    or_sync = pd.DataFrame(index=range(len(common_time)), columns=or_data_segment.columns)
    
    # Choose interpolation method
    if interpolation_method == 'cubic':
        interp_func = lambda x, y: CubicSpline(x, y) if len(x) > 3 else interp1d(x, y, bounds_error=False, fill_value="extrapolate")
    else:  # Default to linear
        interp_func = lambda x, y: interp1d(x, y, bounds_error=False, fill_value="extrapolate")
    
    # Interpolate each column with error handling
    try:
        # Gyroscope data interpolation
        for column in gyro_data_segment.columns:
            if len(time_gyro) > 1:  # Need at least 2 points for interpolation
                interp = interp_func(time_gyro, gyro_data_segment[column])
                gyro_sync[column] = interp(common_time)
        
        # Accelerometer data interpolation
        for column in acc_data_segment.columns:
            if len(time_acc) > 1:
                interp = interp_func(time_acc, acc_data_segment[column])
                acc_sync[column] = interp(common_time)
        
        # Orientation data interpolation
        for column in or_data_segment.columns:
            if len(time_or) > 1:
                interp = interp_func(time_or, or_data_segment[column])
                or_sync[column] = interp(common_time)
    except Exception as e:
        print(f"Interpolation error: {e}. Returning original segmented data.")
        return gyro_data_segment, acc_data_segment, or_data_segment, None
    
    return gyro_sync, acc_sync, or_sync, common_time


def segment_arm_motion_with_hysteresis(raw_magnitude, acc_data, orientation_data, frequency=519, 
                                      visualization=True, hysteresis_ratio=0.7, min_duration_ratio=1.0):
    """
    Robust arm motion segmentation using hysteresis thresholding and neural networks
    
    Parameters:
    raw_magnitude (array-like): Magnitude of bias-removed gyroscope data
    acc_data (pandas.DataFrame): Acceleration data (x, y, z)
    orientation_data (pandas.DataFrame): Orientation/quaternion data
    frequency (int): Sampling frequency in Hz
    visualization (bool): Whether to visualize the segmentation results
    hysteresis_ratio (float): Ratio between end threshold and start threshold (0-1)
    min_duration_ratio (float): Minimum duration multiplier for post-motion stability check
    
    Returns:
    tuple: (start_idx, end_idx) indices of the detected motion segment
    """
    import numpy as np
    import pandas as pd
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from scipy.signal import medfilt, savgol_filter
    import matplotlib.pyplot as plt
    
    # Convert inputs to numpy arrays
    raw_magnitude = np.array(raw_magnitude)
    
    # Step 1: Signal preprocessing
    window_length = min(int(frequency * 0.1), 51)  # 100ms window, max 51 samples
    if window_length % 2 == 0:  # Must be odd
        window_length += 1
    
    filtered_magnitude = savgol_filter(raw_magnitude, window_length=window_length, polyorder=3)
    
    # Step 2: Feature extraction using sliding windows
    window_size = int(frequency * 0.2)  # 200ms window
    features = []
    
    # Safety check - ensure we have enough data
    if len(raw_magnitude) <= window_size:
        print("Error: Not enough magnitude data for analysis")
        return None, None
    
    # Get shapes/dimensions safely
    acc_columns = acc_data.shape[1] if isinstance(acc_data, pd.DataFrame) else acc_data.shape[1]
    ori_columns = orientation_data.shape[1] if isinstance(orientation_data, pd.DataFrame) else orientation_data.shape[1]
    
    for i in range(len(raw_magnitude) - window_size):
        feature_vector = []
        
        # Extract window slices
        mag_window = raw_magnitude[i:i+window_size]
        filt_mag_window = filtered_magnitude[i:i+window_size]
        
        # Process acceleration data - with bounds checking
        if isinstance(acc_data, pd.DataFrame):
            if i+window_size <= len(acc_data):
                acc_window = acc_data.iloc[i:i+window_size].values
            else:
                acc_window = acc_data.iloc[i:].values
        else:
            end_idx = min(i+window_size, len(acc_data))
            acc_window = acc_data[i:end_idx]
            
        # Process orientation data - with bounds checking
        if isinstance(orientation_data, pd.DataFrame):
            if i+window_size <= len(orientation_data):
                ori_window = orientation_data.iloc[i:i+window_size].values
            else:
                ori_window = orientation_data.iloc[i:].values
        else:
            end_idx = min(i+window_size, len(orientation_data))
            ori_window = orientation_data[i:end_idx]
        
        # Calculate features from gyroscope magnitude
        feature_vector.extend([
            np.mean(mag_window),
            np.std(mag_window) if len(mag_window) > 1 else 0,
            np.max(mag_window) if len(mag_window) > 0 else 0,
            np.median(mag_window) if len(mag_window) > 0 else 0
        ])
        
        # Calculate features from filtered magnitude
        feature_vector.extend([
            np.mean(filt_mag_window),
            np.std(filt_mag_window) if len(filt_mag_window) > 1 else 0
        ])
        
        # Derivative features to capture rate of change
        if len(filt_mag_window) > 1:
            deriv = np.diff(filt_mag_window)
            feature_vector.extend([
                np.mean(np.abs(deriv)) if len(deriv) > 0 else 0,
                np.std(deriv) if len(deriv) > 1 else 0,
                np.max(np.abs(deriv)) if len(deriv) > 0 else 0
            ])
        else:
            feature_vector.extend([0, 0, 0])  # Default values if too few samples
        
        # Features from acceleration - with empty array checks
        if acc_window.size > 0 and acc_window.shape[1] > 0:
            for axis in range(min(acc_columns, acc_window.shape[1])):
                acc_axis = acc_window[:, axis]
                
                if len(acc_axis) > 0:
                    feature_vector.extend([
                        np.mean(acc_axis),
                        np.std(acc_axis) if len(acc_axis) > 1 else 0,
                        (np.max(acc_axis) - np.min(acc_axis)) if len(acc_axis) > 0 else 0  # Range
                    ])
                    
                    # Acceleration derivative
                    if len(acc_axis) > 1:
                        acc_deriv = np.diff(acc_axis)
                        feature_vector.extend([
                            np.mean(np.abs(acc_deriv)) if len(acc_deriv) > 0 else 0,
                            np.max(np.abs(acc_deriv)) if len(acc_deriv) > 0 else 0
                        ])
                    else:
                        feature_vector.extend([0, 0])
                else:
                    # Default values for empty arrays
                    feature_vector.extend([0, 0, 0, 0, 0])
        else:
            # Default features if acceleration data is empty
            feature_vector.extend([0, 0, 0, 0, 0] * acc_columns)
        
        # Features from orientation (quaternion) - with empty array checks
        if ori_window.size > 0 and ori_window.shape[1] > 0:
            for axis in range(min(ori_columns, ori_window.shape[1])):
                if axis < ori_window.shape[1]:  # Extra safety check
                    ori_axis = ori_window[:, axis]
                    
                    if len(ori_axis) > 0:
                        feature_vector.extend([
                            np.std(ori_axis) if len(ori_axis) > 1 else 0,
                            (np.max(ori_axis) - np.min(ori_axis)) if len(ori_axis) > 0 else 0  # Range - FIXED
                        ])
                        
                        # Orientation derivative
                        if len(ori_axis) > 1:
                            ori_deriv = np.diff(ori_axis)
                            feature_vector.append(np.mean(np.abs(ori_deriv)) if len(ori_deriv) > 0 else 0)
                        else:
                            feature_vector.append(0)
                    else:
                        feature_vector.extend([0, 0, 0])
                else:
                    feature_vector.extend([0, 0, 0])
        else:
            # Default features if orientation data is empty
            feature_vector.extend([0, 0, 0] * ori_columns)
        
        features.append(feature_vector)
    
    # Step 3: Generate synthetic labels for training based on magnitude
    baseline = np.median(filtered_magnitude)
    peak = np.max(filtered_magnitude)
    
    # Calculate start and end thresholds with hysteresis
    start_threshold = baseline + 0.1 * (peak - baseline)  # 10% above baseline
    end_threshold = baseline + hysteresis_ratio * 0.1 * (peak - baseline)  # Lower threshold for end detection
    
    # Create initial labels based on simple thresholding (for training)
    labels = (filtered_magnitude > start_threshold).astype(int)
    
    # Smooth labels to remove noise
    med_window = int(frequency * 0.2)  # 200ms window
    if med_window % 2 == 0:
        med_window += 1
    smoothed_labels = medfilt(labels, kernel_size=med_window)
    
    # Extract labels for each feature window
    window_labels = []
    for i in range(len(features)):
        idx = i + window_size // 2
        if idx < len(smoothed_labels):
            window_labels.append(smoothed_labels[idx])
        else:
            window_labels.append(0)
    
    # Step 4: Train neural network classifier
    X = np.array(features)
    y = np.array(window_labels)
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Configure and train MLP
    mlp = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size='auto',
        learning_rate='adaptive',
        max_iter=500,
        random_state=42
    )
    
    try:
        mlp.fit(X_scaled, y)
        
        # Step 5: Predict on the entire sequence
        predictions = mlp.predict(X_scaled)
        
        # Apply median filter to smooth predictions
        predictions_smoothed = medfilt(predictions, kernel_size=med_window)
        
        # Step 6: Find motion segments using hysteresis
        # Implement hysteresis-based state machine
        state = 0  # 0: no motion, 1: in motion
        start_idx = None
        end_idx = None
        potential_end = None
        stability_counter = 0
        required_stability = int(frequency * 0.2 * min_duration_ratio)  # 200ms of stability
        
        motion_segments = []
        
        # Apply hysteresis-based segmentation
        for i in range(len(filtered_magnitude)):
            current_value = filtered_magnitude[i]
            
            if state == 0:  # Currently in no-motion state
                if current_value > start_threshold:
                    # Motion start detected
                    state = 1
                    start_idx = i
                    potential_end = None
                    stability_counter = 0
            else:  # Currently in motion state
                if current_value < end_threshold:
                    # Potential end of motion
                    if potential_end is None:
                        potential_end = i
                    
                    stability_counter += 1
                    
                    # Check if stability has been maintained long enough
                    if stability_counter >= required_stability:
                        # Confirm end of motion
                        end_idx = potential_end
                        motion_segments.append((start_idx, end_idx))
                        state = 0  # Return to no-motion state
                else:
                    # Reset potential end if magnitude goes back above threshold
                    potential_end = None
                    stability_counter = 0
        
        # If still in motion at the end of the data, close the segment
        if state == 1 and start_idx is not None:
            if potential_end is not None:
                end_idx = potential_end
            else:
                end_idx = len(filtered_magnitude) - 1
            motion_segments.append((start_idx, end_idx))
        
        # Step 7: Select the most significant motion segment
        if motion_segments:
            # Sort segments by duration and intensity
            segments = []
            for start, end in motion_segments:
                duration = end - start
                segment_magnitude = np.mean(filtered_magnitude[start:end])
                segments.append((start, end, duration, segment_magnitude))
            
            # Sort by intensity (highest average magnitude)
            segments.sort(key=lambda x: x[3], reverse=True)
            
            # Select most intense segment
            start_idx, end_idx, _, _ = segments[0]
            
            # Use additional criteria to refine end detection
            try:
                # Look for significant deceleration phase at the end of motion
                if start_idx < len(acc_data) and end_idx + 50 < len(acc_data):
                    # Get acceleration data for the segment plus a bit more
                    acc_segment = acc_data.iloc[start_idx:end_idx+50].values
                    
                    # Only proceed if we have enough data
                    if acc_segment.size > 0 and acc_segment.shape[0] > 1:
                        # Calculate acceleration magnitude 
                        acc_magnitude = np.sqrt(np.sum(np.diff(acc_segment, axis=0)**2, axis=1))
                        
                        if len(acc_magnitude) > 0:
                            # Find significant deceleration points
                            acc_threshold = np.mean(acc_magnitude) * 0.3  # 30% of mean acceleration
                            decel_points = np.where(acc_magnitude < acc_threshold)[0]
                            
                            # Find the first stable section after strong motion
                            if len(decel_points) > 0 and decel_points[-1] > 0:
                                # Check if there's a stable deceleration region
                                stable_window = int(frequency * 0.1)  # 100ms window for stability check
                                
                                # Make sure we have enough points to check
                                if len(decel_points) > stable_window:
                                    for i in range(len(decel_points) - stable_window):
                                        if np.all(decel_points[i:i+stable_window] == np.arange(decel_points[i], decel_points[i]+stable_window)):
                                            # Found stable deceleration region
                                            refined_end = start_idx + decel_points[i] + stable_window//2
                                            if refined_end > end_idx - int(frequency * 0.5) and refined_end < end_idx + int(frequency * 0.5):
                                                # Only update if reasonable (within 0.5s of original end)
                                                end_idx = refined_end
                                                break
            except Exception as e:
                print(f"Warning: Error during end refinement: {e}")
                # Continue with the existing end_idx if refinement fails
            
            # Visualization
            if visualization:
                plt.figure(figsize=(12, 6))
                plt.plot(raw_magnitude, label='Raw Magnitude', alpha=0.5)
                plt.plot(filtered_magnitude, label='Filtered Magnitude', alpha=0.8)
                plt.axhline(y=start_threshold, color='r', linestyle='--', label='Start Threshold')
                plt.axhline(y=end_threshold, color='r', linestyle=':', label='End Threshold')
                plt.axvspan(start_idx, end_idx, alpha=0.2, color='green', label='Detected Motion')
                plt.title('Arm Motion Segmentation')
                plt.xlabel('Samples')
                plt.ylabel('Magnitude')
                plt.legend()
                plt.grid(True)
                plt.show()
            
            return start_idx, end_idx
            
    except Exception as e:
        print(f"Error during classification or segmentation: {e}")
    
    # If no significant segment found
    print("No significant motion segment detected.")
    return None, None

# ----------- HYPERPARAMETERS -----------------
# Hyperparameters for bias removal
bias_average_window = 1000 # Number of samples to average for bias removal
frequency = 519

# Select folder
folder_path = DataLoader.select_folder()

if not folder_path:
    print("No folder selected. Exiting...")
    
print(f"Selected folder: {folder_path}")

# Load and process files
acceleration_data, gyro_data, or_data, acc_files, gyro_files, or_files = DataLoader.load_and_process_files(folder_path)
print(f"Loaded {len(acc_files)} ACC files, {len(gyro_files)} GYRO files, and {len(or_files)} OR files")
# Group files by timestamp
grouped_indices = DataLoader.group_files_by_timestamp(acc_files, gyro_files, or_files)
    
# Sort timestamps to ensure chronological order
sorted_timestamps = sorted(grouped_indices.keys())
for timestamp in sorted_timestamps:
    indices = grouped_indices[timestamp]
    
    # Get the data for this timestamp
    gyro = gyro_data[indices["gyro"]]
    acc_data = acceleration_data[indices["acc"]]
    orientation_data = or_data[indices["or"]]
    
    print(f"Processing data set from timestamp: {timestamp}")

    # ------------------------- BIAS REMOVAL ----------------------------
    print("Removing bias...")
    non_zero_index = (gyro != 0).any(axis=1).idxmax()
    sample_size = bias_average_window
    
    if non_zero_index + sample_size <= len(gyro):
        means = gyro.iloc[non_zero_index:non_zero_index + sample_size].mean()
        print("Initial mean values:", means)
    else:
        print("Not enough data after first non-zero value")

    gyro_data_centered = gyro - means
    gyro_data_trimmed = gyro_data_centered.iloc[non_zero_index:].reset_index(drop=True)
    raw_magnitude = np.sqrt(gyro_data_trimmed.iloc[:,0]**2 + 
                    gyro_data_trimmed.iloc[:,1]**2 + 
                    gyro_data_trimmed.iloc[:,2]**2)

    plt.figure(figsize=(15, 5))
    plt.plot(raw_magnitude, label='Filtered Magnitude', color='blue')
    plt.title(f'Filtered Magnitude for {timestamp}')
    plt.xlabel('Samples')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.grid()

    # New improved code with hysteresis-based end detection
    start_idx, end_idx = segment_arm_motion_with_hysteresis(
        raw_magnitude, 
        acc_data, 
        orientation_data, 
        frequency=frequency,
        visualization=True,
        hysteresis_ratio=0.6,  # 60% of start threshold for end detection
        min_duration_ratio=1.0  # Stability duration multiplier
    )
    
    if start_idx is None or end_idx is None:
        print(f"No motion segment detected for {timestamp}. Skipping...")
        continue

    # Define sensor frequencies
    frequencies = [519, 519, 222]  # Gyro, Acc, Orientation
    
    # Calculate indices for other sensors based on time scaling
    start_idx_acc = int(start_idx * frequencies[1] / frequencies[0])
    end_idx_acc = int(end_idx * frequencies[1] / frequencies[0])
    non_zero_index_acc = int(non_zero_index * frequencies[1] / frequencies[0])
    
    start_idx_or = int(start_idx * frequencies[2] / frequencies[0])
    end_idx_or = int(end_idx * frequencies[2] / frequencies[0])
    non_zero_index_or = int(non_zero_index * frequencies[2] / frequencies[0])
    
    # Create time arrays for each sensor
    time_gyro = np.arange(len(gyro)) / frequencies[0]
    time_acc = np.arange(len(acc_data)) / frequencies[1]
    time_orientation = np.arange(len(orientation_data)) / frequencies[2]
    
    # Extract trimmed data
    time_gyro_segmented = time_gyro[non_zero_index + start_idx:non_zero_index + end_idx]
    
    # Process acceleration data
    acc_data_trimmed = acc_data.iloc[non_zero_index_acc:].reset_index(drop=True)
    acc_data_segmented = acc_data_trimmed.iloc[start_idx_acc:end_idx_acc].reset_index(drop=True)
    time_acc_segmented = time_acc[non_zero_index_acc + start_idx_acc:non_zero_index_acc + end_idx_acc]
    
    # Process orientation data
    or_data_trimmed = orientation_data.iloc[non_zero_index_or:].reset_index(drop=True)
    or_data_segmented = or_data_trimmed.iloc[start_idx_or:end_idx_or].reset_index(drop=True)
    time_orientation_segmented = time_orientation[non_zero_index_or + start_idx_or:non_zero_index_or + end_idx_or]
    
    # ------------------------- SENSOR SYNCHRONIZATION ----------------------------
    print("Synchronizing sensor data...")
    
    # Synchronize the segmented data
    gyro_sync, acc_sync, or_sync, common_time = synchronize_multi_sensor_data(
        gyro_data_trimmed, 
        acc_data_trimmed, 
        or_data_trimmed, 
        frequencies, 
        start_idx, 
        end_idx,
        target_freq=frequencies[0],  # Use gyro frequency as target
        check_alignment=True         # Check for temporal misalignment
    )
    
    # FIGURE 2: Motion Region Highlights (3x1 grid)
    fig2, axs2 = plt.subplots(3, 1, figsize=(15, 12))
    
    # Gyro Highlight
    for column in gyro_data_trimmed.columns:
        axs2[0].plot(time_gyro[non_zero_index:], gyro_data_trimmed[column], label=column)
    axs2[0].axvline(time_gyro[non_zero_index + start_idx], color='lime', linewidth=2, label='Motion Start')
    axs2[0].axvline(time_gyro[non_zero_index + end_idx], color='red', linewidth=2, label='Motion End')
    axs2[0].set_title("Gyro Data with Motion Region")
    axs2[0].set(xlabel="Time (s)", ylabel="Angular Velocity (rad/s)")
    axs2[0].legend()
    axs2[0].grid(True)
    
    # Acceleration Highlight
    for column in acc_data_trimmed.columns:
        axs2[1].plot(time_acc[non_zero_index_acc:], acc_data_trimmed[column], label=column)
    axs2[1].axvline(time_acc[non_zero_index_acc + start_idx_acc], color='lime', linewidth=2)
    axs2[1].axvline(time_acc[non_zero_index_acc + end_idx_acc], color='red', linewidth=2)
    axs2[1].set_title("Acceleration Data with Motion Region")
    axs2[1].set(xlabel="Time (s)", ylabel="Acceleration (m/s²)")
    axs2[1].legend()
    axs2[1].grid(True)
    
    # Orientation Highlight
    for column in or_data_trimmed.columns:
        axs2[2].plot(time_orientation[non_zero_index_or:], or_data_trimmed[column], label=column)
    axs2[2].axvline(time_orientation[non_zero_index_or + start_idx_or], color='lime', linewidth=2)
    axs2[2].axvline(time_orientation[non_zero_index_or + end_idx_or], color='red', linewidth=2)
    axs2[2].set_title("Orientation Data with Motion Region")
    axs2[2].set(xlabel="Time (s)", ylabel="Orientation (rad)")
    axs2[2].legend()
    axs2[2].grid(True)
    
    # FIGURE 3: Synchronized Data Visualization
    if common_time is not None:
        fig3, axs3 = plt.subplots(3, 1, figsize=(15, 12))
        
        # Plot synchronized gyro data
        for column in gyro_sync.columns:
            axs3[0].plot(common_time, gyro_sync[column], label=column)
        axs3[0].set_title("Synchronized Gyro Data")
        axs3[0].set(xlabel="Time (s)", ylabel="Angular Velocity (rad/s)")
        axs3[0].legend()
        axs3[0].grid(True)
        
        # Plot synchronized acceleration data
        for column in acc_sync.columns:
            axs3[1].plot(common_time, acc_sync[column], label=column)
        axs3[1].set_title("Synchronized Acceleration Data")
        axs3[1].set(xlabel="Time (s)", ylabel="Acceleration (m/s²)")
        axs3[1].legend()
        axs3[1].grid(True)
        
        # Plot synchronized orientation data
        for column in or_sync.columns:
            axs3[2].plot(common_time, or_sync[column], label=column)
        axs3[2].set_title("Synchronized Orientation Data")
        axs3[2].set(xlabel="Time (s)", ylabel="Orientation (rad)")
        axs3[2].legend()
        axs3[2].grid(True)
        
        plt.tight_layout()
    
plt.show(block=True)
