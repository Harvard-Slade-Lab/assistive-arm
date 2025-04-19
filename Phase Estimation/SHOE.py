import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt

# Hyperparameters:
interval_size = 500
derivative_threshold = 0.15
threshold_offset = 5

def motion_segmenter(acc_data, gyro_data, frequency=519, visualize=False):
    """
    Robust motion segmentation using SHOE detector with dual-threshold refinement.

    Parameters:
    acc_data (pd.DataFrame): Raw accelerometer data (x, y, z)
    gyro_data (pd.DataFrame): Raw gyroscope data (x, y, z)
    frequency (int): Sampling frequency in Hz
    visualize (bool): Whether to show debug plots

    Returns:
    tuple: (start_idx, end_idx) of refined motion segment
    """
    # Convert to numpy arrays and ensure proper shape
    acc_array = acc_data.values.astype(float)
    gyro_array = gyro_data.values.astype(float)

    # Remove gravity component from acceleration
    linear_acc = remove_gravity(acc_array, frequency)

    # Compute optimal parameters
    window_size = int(0.25 * frequency) # 0.25s window
    gamma = calculate_optimal_gamma(linear_acc, gyro_array)

    # Detect motion using SHOE
    motion_metric = compute_shoe_metric(linear_acc, gyro_array, window_size)

    # Find initial transitions with hysteresis
    initial_start, initial_end = find_initial_transitions(motion_metric, gamma, frequency)
    
    # Refine motion boundaries with fixed thresholds and rate of change analysis
    refined_start, refined_end = refine_motion_boundaries(motion_metric, initial_start, initial_end)

    if visualize:
        plot_segmentation_with_thresholds(acc_array, gyro_array, motion_metric,
                               gamma, initial_start, initial_end, 
                               refined_start, refined_end, frequency)

    return refined_start, refined_end

def remove_gravity(acc_data, frequency):
    """Remove gravity using Butterworth low-pass filter"""
    b, a = signal.butter(4, 0.1, 'low', fs=frequency)
    gravity = signal.filtfilt(b, a, acc_data, axis=0)
    return acc_data - gravity

def calculate_optimal_gamma(linear_acc, gyro_data):
    """Automatically determine gamma threshold"""
    acc_var = np.var(linear_acc, axis=0)
    gyro_var = np.var(gyro_data, axis=0)

    # Compute baseline noise levels
    acc_baseline = np.mean(np.linalg.norm(linear_acc[:100], axis=1))
    gyro_baseline = np.mean(np.linalg.norm(gyro_data[:100], axis=1))

    return 5 * (acc_baseline/np.mean(acc_var) + gyro_baseline/np.mean(gyro_var))

def compute_shoe_metric(linear_acc, gyro_data, window_size):
    """Compute SHOE detection metric"""
    # Compute squared norms
    acc_sq = np.linalg.norm(linear_acc, axis=1)**2
    gyro_sq = np.linalg.norm(gyro_data, axis=1)**2

    # Compute variances
    acc_var = np.var(linear_acc, axis=0).mean()
    gyro_var = np.var(gyro_data, axis=0).mean()

    # Normalize and combine
    combined = (acc_sq/acc_var) + (gyro_sq/gyro_var)

    # Apply moving average
    window = np.ones(window_size)/window_size
    return np.convolve(combined, window, mode='same')

def find_initial_transitions(metric, gamma, frequency):
    """Find initial motion start/end indices with hysteresis"""
    # Apply threshold with hysteresis
    in_motion = np.zeros_like(metric)
    #Compute threshold as median of last 500 samples of metric:
    threshold = np.median(metric[-500:]) + 5
    state = 0
    for i in range(len(metric)):
        if state == 0 and metric[i] > threshold:
            state = 1
        elif state == 1 and metric[i] < threshold:
            state = 0
        in_motion[i] = state

    # Find transitions
    transitions = np.diff(in_motion)
    starts = np.where(transitions == 1)[0]
    ends = np.where(transitions == -1)[0]
    
    print(f"Starts: {starts}, Ends: {ends}")
    # Handle edge cases
    if len(starts) == 0 or len(ends) == 0:
        return 0, 0
    
    start_idx = starts[0]
    end_idx = ends[-1]

    # print start and end in seconds:
    print(f"Initial motion segment: {start_idx/frequency:.2f}s to {end_idx/frequency:.2f}s")
    return start_idx, end_idx

def refine_motion_boundaries(metric, rough_start, rough_end):
    """
    Refine the motion boundaries using fixed thresholds and rate of change analysis.
    
    Parameters:
    -----------
    metric : array-like
        The SHOE metric signal
    rough_start : int
        Initial estimate of motion start index
    rough_end : int
        Initial estimate of motion end index
        
    Returns:
    --------
    refined_start : int
        Refined motion start index
    refined_end : int
        Refined motion end index
    """
    refined_start = rough_start
    refined_end = rough_end


    # Verify refinement with rate of change to avoid noise artifacts
    # Calculate first derivative of metric signal
    metric_diff_smoothed = np.diff(metric)
    
    # # Smooth the derivative using Savitzky-Golay filter
    # metric_diff_smoothed = signal.savgol_filter(
    #     np.concatenate(([0], metric_diff)),
    #     window_length=11, polyorder=2
    # )
    
    # For start point: look for significant positive rate of change
    start_search_range = (max(0, refined_start - interval_size), 
                         min(len(metric_diff_smoothed)-1, refined_start))
    
    print(f"Start search range: {start_search_range}")
    for i in range(start_search_range[1], start_search_range[0], -1):
        if metric_diff_smoothed[i] < metric_diff_smoothed[refined_start] * derivative_threshold:
            refined_start = i
            break
    
    # For end point: look for significant negative rate of change
    end_search_range = (max(0, refined_end), 
                       min(len(metric_diff_smoothed)-1, refined_end + interval_size))
    print(f"End search range: {end_search_range}")

    for i in range(end_search_range[0], end_search_range[1]):
        if metric_diff_smoothed[i] > metric_diff_smoothed[refined_end] * derivative_threshold:
            refined_end = i
            break
    print(metric_diff_smoothed[refined_end] * derivative_threshold)
    return refined_start, refined_end

def plot_segmentation_with_thresholds(acc, gyro, metric, gamma, initial_start, initial_end, 
                                     refined_start, refined_end, freq):
    """Enhanced visualization function showing dual thresholds"""
    time = np.arange(len(metric))/freq
    
    threshold = np.median(metric[-500:]) + threshold_offset
    # Fixed thresholds for refinement
    start_threshold = threshold
    end_threshold = threshold

    fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # Accelerometer plot
    ax[0].plot(time, acc)
    ax[0].set_ylabel('Linear Acceleration (m/s²)')
    ax[0].axvspan(time[initial_start], time[initial_end], color='red', alpha=0.2, label='Original Segment')
    ax[0].axvspan(time[refined_start], time[refined_end], color='green', alpha=0.3, label='Refined Segment')
    ax[0].legend(loc='upper right')

    # Gyroscope plot
    ax[1].plot(time, gyro)
    ax[1].set_ylabel('Angular Velocity (rad/s)')
    ax[1].axvspan(time[initial_start], time[initial_end], color='red', alpha=0.2)
    ax[1].axvspan(time[refined_start], time[refined_end], color='green', alpha=0.3)

    # SHOE metric plot with all thresholds
    ax[2].plot(time, metric, label='SHOE Metric')
    ax[2].axhline(gamma, color='r', linestyle='--', label='Original Threshold')
    ax[2].axhline(start_threshold, color='g', linestyle='--', label='Start Threshold (5)')
    ax[2].axhline(end_threshold, color='b', linestyle='--', label='End Threshold (10)')
    
    # Mark the refined points
    ax[2].plot(time[refined_start], metric[refined_start], 'go', markersize=8, label='Refined Start')
    ax[2].plot(time[refined_end], metric[refined_end], 'bo', markersize=8, label='Refined End')
    
    ax[2].axvspan(time[initial_start], time[initial_end], color='red', alpha=0.2)
    ax[2].axvspan(time[refined_start], time[refined_end], color='green', alpha=0.3)
    ax[2].set_xlabel('Time (s)')
    ax[2].legend()

    plt.suptitle(f'Motion Segmentation with Dual-Threshold Refinement:\nOriginal: {initial_start/freq:.2f}s to {initial_end/freq:.2f}s\nRefined: {refined_start/freq:.2f}s to {refined_end/freq:.2f}s')
    plt.tight_layout()
    plt.show()
