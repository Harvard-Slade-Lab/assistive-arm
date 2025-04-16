import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def shoe_detector_corrected(accel, gyro, frequencies, visualize=False):
    """
    Corrected SHOE detector implementation
    
    Parameters:
        accel (np.ndarray): Accelerometer data (Nx3)
        gyro (np.ndarray): Gyroscope data (Nx3)
        frequency (int): Sampling frequency in Hz
        visualize (bool): Whether to show visualization
        
    Returns:
        start_idx, end_idx: Indices of detected motion
    """
    frequency = frequencies[0]  # Assuming first frequency is for accelerometer
    # Ensure proper shape
    accel = np.asarray(accel)
    gyro = np.asarray(gyro)
    
    # 1. Remove gravity from acceleration
    accel = remove_gravity(accel, frequency)
    
    # 2. Compute correct SHOE metric
    window_size = int(0.2 * frequency)  # 0.2 seconds window
    shoe_metric = improved_shoe_metric(accel, gyro, window_size)
    
    # 3. Calculate proper adaptive threshold
    gamma = calculate_adaptive_threshold(shoe_metric)
    
    # 4. Detect motion with hysteresis
    start_idx, end_idx = detect_motion_with_hysteresis(shoe_metric, gamma)
    
    # 5. Refine boundaries
    refined_start = refine_boundary(shoe_metric, start_idx, direction='backward')
    refined_end = refine_boundary(shoe_metric, end_idx, direction='forward')
    
    if visualize:
        plot_results(accel, gyro, shoe_metric, gamma, refined_start, refined_end, frequency)
    
    return refined_start, refined_end

def remove_gravity(accel, frequency):
    """Remove gravity component using low-pass filter"""
    cutoff = 0.1  # 0.1 Hz cutoff (standard for gravity)
    b, a = signal.butter(4, cutoff, 'low', fs=frequency)
    gravity = signal.filtfilt(b, a, accel, axis=0)
    return accel - gravity

def improved_shoe_metric(accel, gyro, window_size):
    """Compute correct SHOE metric with proper normalization"""
    n_samples = len(accel)
    combined_metric = np.zeros(n_samples)
    
    # Get baseline from first 0.5 seconds
    baseline_len = min(int(0.5 * 519), n_samples // 10)
    
    # Process each axis
    for axis in range(3):
        # Get variance (add epsilon to avoid division by zero)
        accel_var = np.var(accel[:baseline_len, axis]) + 1e-10
        gyro_var = np.var(gyro[:baseline_len, axis]) + 1e-10
        
        # Normalized squared values
        accel_norm = (accel[:, axis]**2) / accel_var
        gyro_norm = (gyro[:, axis]**2) / gyro_var
        
        # Add to combined metric
        combined_metric += accel_norm + gyro_norm
    
    # Apply smoothing window
    window = np.ones(window_size) / window_size
    return np.convolve(combined_metric, window, mode='same')

def calculate_adaptive_threshold(metric):
    """Calculate appropriate threshold based on signal statistics"""
    # Use first 10% of data as baseline (assumed stationary)
    baseline_len = int(len(metric) * 0.1)
    baseline = metric[:baseline_len]
    
    # Calculate statistics
    mean_val = np.mean(baseline)
    std_val = np.std(baseline)
    
    # Set threshold using standard robust statistical methods
    # 3-sigma rule is common for outlier detection
    return mean_val + 3 * std_val

def detect_motion_with_hysteresis(metric, gamma, hysteresis_factor=0.2):
    """Detect motion with hysteresis to avoid rapid switching"""
    n_samples = len(metric)
    in_motion = np.zeros(n_samples, dtype=bool)
    
    # Apply hysteresis thresholding
    upper_threshold = gamma * (1 + hysteresis_factor)
    lower_threshold = gamma * (1 - hysteresis_factor)
    
    # State machine for hysteresis
    state = False  # Start in stationary state
    for i in range(n_samples):
        if not state and metric[i] > upper_threshold:
            state = True  # Enter motion state
        elif state and metric[i] < lower_threshold:
            state = False  # Exit motion state
        in_motion[i] = state
    
    # Find transitions
    transitions = np.diff(in_motion.astype(int))
    starts = np.where(transitions == 1)[0]
    ends = np.where(transitions == -1)[0]
    
    # Handle edge cases
    if len(starts) == 0:
        return 0, n_samples - 1
    if len(ends) == 0 or ends[-1] < starts[-1]:
        ends = np.append(ends, n_samples - 1)
    
    # Get primary motion segment (largest)
    if len(starts) > 1 and len(ends) > 1:
        durations = ends - starts
        longest_idx = np.argmax(durations)
        return starts[longest_idx], ends[longest_idx]
    
    return starts[0], ends[0]

def refine_boundary(metric, idx, direction='backward', window_size=50):
    """Refine boundary using gradient analysis"""
    if direction == 'backward':
        window = metric[max(0, idx-window_size):idx+1]
        if len(window) < 3:
            return idx
        
        # Calculate gradient
        grad = np.gradient(window)
        
        # Find where gradient becomes significant
        threshold = np.std(grad) * 1.5
        for i in range(len(grad)-1, -1, -1):
            if abs(grad[i]) > threshold:
                return max(0, idx-window_size+i)
        
        return idx
    else:  # forward
        window = metric[idx:min(len(metric), idx+window_size)]
        if len(window) < 3:
            return idx
        
        # Calculate gradient
        grad = np.gradient(window)
        
        # Find where gradient becomes significant
        threshold = np.std(grad) * 1.5
        for i in range(len(grad)):
            if abs(grad[i]) > threshold:
                return min(len(metric)-1, idx+i)
        
        return idx

def plot_results(accel, gyro, metric, threshold, start_idx, end_idx, freq):
    """Visualize results with correct scaling"""
    time = np.arange(len(metric)) / freq
    
    fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    # Accelerometer plot
    ax[0].plot(time, accel)
    ax[0].axvspan(time[start_idx], time[end_idx], color='red', alpha=0.3)
    ax[0].set_ylabel('Linear Acceleration (m/s²)')
    ax[0].set_title('Linear Acceleration with Detected Motion')
    
    # Gyroscope plot
    ax[1].plot(time, gyro)
    ax[1].axvspan(time[start_idx], time[end_idx], color='red', alpha=0.3)
    ax[1].set_ylabel('Angular Velocity (rad/s)')
    ax[1].set_title('Angular Velocity with Detected Motion')
    
    # SHOE metric plot (with proper scaling)
    ax[2].plot(time, metric, label='SHOE Metric')
    ax[2].axhline(threshold, color='red', linestyle='--', label=f'Threshold: {threshold:.2f}')
    ax[2].axvline(time[start_idx], color='green', linewidth=2, label='Motion Start')
    ax[2].axvline(time[end_idx], color='blue', linewidth=2, label='Motion End')
    ax[2].set_xlabel('Time (s)')
    ax[2].set_ylabel('SHOE Metric')
    ax[2].set_title('SHOE Metric with Corrected Threshold')
    ax[2].legend()
    
    plt.tight_layout()
    plt.show()
