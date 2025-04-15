import numpy as np

def shoe_detector(accel, gyro, window_size, gamma):
    """
    Implements the SHOE (Stance Hypothesis Optimal Estimation) detector for motion segmentation.
    
    Parameters:
        accel (np.ndarray): 3D linear acceleration data (Nx3 array) with gravity removed.
        gyro (np.ndarray): 3D angular velocity data (Nx3 array).
        window_size (int): Size of the sliding window.
        gamma (float): Threshold for motion detection.
        
    Returns:
        start_idx (int): Index where motion starts.
        end_idx (int): Index where motion ends.
    """
    # Compute variances of acceleration and gyroscope data
    accel_var = np.var(accel)  # Overall variance of acceleration
    gyro_var = np.var(gyro)    # Overall variance of gyroscope
    
    # Initialize array to store motion metric
    motion_metric = np.zeros(len(accel))
    
    # For each point, compute motion metric using a window of data
    half_window = window_size // 2
    valid_start = half_window
    valid_end = len(accel) - half_window
    
    for k in range(valid_start, valid_end):
        # Extract window of data centered at point k
        window_start = k - half_window
        window_end = k + half_window
        
        # Ensure window_end is correct for odd window sizes
        if window_size % 2 != 0:
            window_end += 1
        
        accel_window = accel[window_start:window_end]
        gyro_window = gyro[window_start:window_end]
        
        # Compute normalized squared norms
        accel_norms = np.linalg.norm(accel_window, axis=1) ** 2
        gyro_norms = np.linalg.norm(gyro_window, axis=1) ** 2
        
        # Compute the motion metric according to the formula
        accel_term = np.sum(accel_norms) / accel_var
        gyro_term = np.sum(gyro_norms) / gyro_var
        
        motion_metric[k] = (1.0 / window_size) * (accel_term + gyro_term)
    
    # Extend the motion metric to points at the boundaries
    motion_metric[:valid_start] = motion_metric[valid_start]
    motion_metric[valid_end:] = motion_metric[valid_end-1]
    
    # Determine motion status for each point
    # According to the formula:
    # yk = 1 if metric < gamma (stationary), yk = 0 if metric >= gamma (motion)
    motion_status = (motion_metric < gamma).astype(int)
    
    # Define in_motion as points where motion_status = 0
    in_motion = (motion_status == 0)
    
    # Edge case: if no motion detected
    if not np.any(in_motion):
        return 0, 0  # No motion detected
    
    # Find transitions between stationary and motion
    transitions = np.diff(in_motion.astype(int))
    
    # Find start of motion
    if in_motion[0]:
        # Motion starts at the beginning
        start_idx = 0
    else:
        # Find first transition from stationary to motion
        start_indices = np.where(transitions == 1)[0] + 1
        if len(start_indices) > 0:
            start_idx = start_indices[0]
        else:
            # No transitions from stationary to motion
            start_idx = 0
    
    # Find end of motion
    if in_motion[-1]:
        # Motion continues to the end
        end_idx = len(accel) - 1
    else:
        # Find last transition from motion to stationary
        end_indices = np.where(transitions == -1)[0]
        if len(end_indices) > 0:
            end_idx = end_indices[-1]
        else:
            # No transitions from motion to stationary
            end_idx = len(accel) - 1
    
    return start_idx, end_idx
