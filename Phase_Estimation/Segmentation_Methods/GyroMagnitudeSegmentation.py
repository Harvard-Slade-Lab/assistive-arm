# Hyperparameters for segmentation
offset = 7.0 # Offset for threshold calculation
before_count = 200 # Number of samples to check before a potential transition
after_count = 200 # Number of samples to check after a potential transition
min_below_ratio = 0.8 # Minimum ratio of samples that must be below threshold in the before region
min_above_ratio = 0.8 # Minimum ratio of samples that must be above threshold in the after region



import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def robust_find_peak_start(signal, threshold_idx, window_size=5, slope_threshold=0.5):
    """Find the true start of peak using robust slope analysis"""
    # Calculate smoothed derivatives using a sliding window
    derivatives = []
    for i in range(window_size, threshold_idx):
        # Compute average slope in window
        avg_slope = (signal[i] - signal[i-window_size]) / window_size
        derivatives.append((i, avg_slope))
    
    # Work backwards from threshold crossing
    for i in range(threshold_idx-window_size, window_size, -1):
        cur_slope = (signal[i] - signal[i-window_size]) / window_size
        # Detect significant slope change
        if cur_slope < slope_threshold:
            return i
    
    return max(0, threshold_idx - 100)  # Fallback if no suitable point found

def comprehensive_peak_start(signal, threshold_idx, window_size=5, noise_level=None):
    """Robust peak start detection using multiple features
    
    Args:
        signal: array-like, input signal
        threshold_idx: int, index where signal crosses threshold
        window_size: int, window size for slope calculation
        noise_level: float, estimated noise level (auto-calculated if None)
        
    Returns:
        int: index of detected peak start
    """
    if noise_level is None:
        # Estimate noise level from signal baseline
        baseline = signal[:min(100, threshold_idx//2)]
        noise_level = np.std(baseline) * 2
    
    # Methods array to hold candidate points
    candidates = []
    
    # Method 1: Slope-based detection
    # Start from threshold crossing and move backwards
    for i in range(threshold_idx-window_size, window_size, -1):
        # Calculate slope over window
        avg_slope = (signal[i] - signal[i-window_size]) / window_size
        if abs(avg_slope) < 0.5:  # Detect significant slope change
            candidates.append(i)
            break
    
    # Method 2: Noise-threshold based detection
    for i in range(threshold_idx-1, 0, -1):
        if signal[i] < noise_level:
            # Confirm by checking previous few points
            if np.mean(signal[max(0, i-5):i+1]) < noise_level:
                candidates.append(i)
                break
    
    # Method 3: Curvature analysis (detect inflection point)
    if threshold_idx > 10:
        derivatives = np.diff(signal[:threshold_idx])
        if len(derivatives) > 2:
            second_derivatives = np.diff(derivatives)
            if len(second_derivatives) > 2:
                # Find where second derivative changes sign (inflection points)
                inflection_indices = np.where(np.diff(np.sign(second_derivatives)) != 0)[0]
                if len(inflection_indices) > 0:
                    # Get the last inflection point before threshold (closest to peak)
                    for idx in reversed(inflection_indices):
                        if idx < threshold_idx - 5:  # Ensure it's not too close to threshold
                            candidates.append(idx)
                            break
    
    # Failsafe: If no candidates found, use a fixed offset
    if not candidates:
        candidates.append(max(0, threshold_idx - 20))
    
    # Return median of candidate points for robustness
    return int(np.median(candidates))
    
    # Return median of candidate points for robustness
    return int(np.median(candidates))

def comprehensive_peak_end(signal, threshold_idx, window_size=20, noise_level=None):
    """Robust peak end detection using multiple features
    
    Args:
        signal: array-like, input signal
        threshold_idx: int, index where signal crosses threshold
        window_size: int, window size for slope calculation
        noise_level: float, estimated noise level (auto-calculated if None)
        
    Returns:
        int: index of detected peak end
    """
    if noise_level is None:
        # Estimate noise level from signal baseline (using data after the peak)
        end_idx = min(len(signal), threshold_idx + 100)
        baseline = signal[end_idx-50:end_idx] if end_idx > 50 else signal[-50:]
        noise_level = np.std(baseline) * 2
    
    # Methods array to hold candidate points
    candidates = []
    
    # Method 1: Slope-based detection
    for i in range(threshold_idx, len(signal)-window_size):
        # Calculate slope over window
        avg_slope = (signal[i+window_size] - signal[i]) / window_size
        if abs(avg_slope) < 0.5:  # Detect when slope becomes flat
            candidates.append(i)
            break
    
    # Method 2: Noise-threshold based detection
    for i in range(threshold_idx, len(signal)):
        if signal[i] < noise_level:
            # Confirm by checking next few points to avoid false detection
            end_check = min(i+5, len(signal))
            if np.mean(signal[i:end_check]) < noise_level:
                candidates.append(i)
                break
    
    # Method 3: Curvature analysis (detect inflection point)
    if len(signal) - threshold_idx > 10:
        derivatives = np.diff(signal[threshold_idx:])
        if len(derivatives) > 2:
            second_derivatives = np.diff(derivatives)
            if len(second_derivatives) > 2:
                # Find where second derivative changes sign (inflection points)
                raw_indices = np.where(np.diff(np.sign(second_derivatives)) != 0)[0]
                if len(raw_indices) > 0:
                    # Convert back to original signal indices
                    inflection_indices = raw_indices + threshold_idx + 2
                    # Get the first inflection point after threshold (closest to peak)
                    for idx in inflection_indices:
                        if idx > threshold_idx + 5:  # Ensure it's not too close to threshold
                            candidates.append(idx)
                            break
    
    # Failsafe: If no candidates found, use a fixed offset
    if not candidates:
        candidates.append(min(len(signal)-1, threshold_idx + 20))
    
    # Return median of candidate points for robustness
    return int(np.median(candidates))

def detect_peak_boundaries(signal, s_idx, e_idx, threshold, visualize=True):

    
    initial_start = s_idx
    initial_end = e_idx
    
    # Apply robust start detection
    refined_start = comprehensive_peak_start(signal, initial_start)
    
    # Apply robust end detection (similar to start, but forward direction)
    refined_end = comprehensive_peak_end(signal, initial_end)
    
    
    # Generate visualization if requested
    if visualize:
        plt.figure(figsize=(12, 7))
        time = np.arange(len(signal))
        
        # Plot original signal
        plt.plot(time, signal, 'purple', label='Magnitude')
        
        # Plot threshold and crossings
        plt.axhline(y=threshold, color='green', linestyle='--', 
                    label=f'Threshold ({threshold:.2f})')
        plt.axvline(x=initial_start, color='orange', linestyle='-', 
                    label='Initial Motion Start')
        plt.axvline(x=initial_end, color='orange', linestyle='-', 
                    label='Initial Motion End')
        
        # Plot refined boundaries
        plt.axvline(x=refined_start, color='blue', linestyle='--', 
                    label='Refined Start')
        plt.axvline(x=refined_end, color='red', linestyle='--', 
                    label='Refined End')
        
        
        plt.title('Signal Magnitude Analysis with Advanced Peak Detection')
        plt.xlabel('Time (s)')
        plt.ylabel('Composite Magnitude')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    return refined_start, refined_end


def check_real_motion(magnitude, threshold_indices, threshold=None, 
                     before_count=before_count, after_count=after_count,
                     min_below_ratio=min_below_ratio, min_above_ratio=min_above_ratio):
    """
    Check for real motion by identifying clean transitions in the magnitude signal.
    
    Args:
        magnitude: The magnitude signal array
        threshold_indices: Indices where magnitude exceeds threshold (used as fallback)
        threshold: The threshold value for magnitude comparison (if None, will be estimated)
        before_count: Number of samples to check before a potential transition
        after_count: Number of samples to check after a potential transition
        min_below_ratio: Minimum ratio of samples that must be below threshold in the before region
        min_above_ratio: Minimum ratio of samples that must be above threshold in the after region
    
    Returns:
        start_idx, end_idx: The start and end indices of the real motion
    """
    # Convert threshold_indices to a list to avoid boolean evaluation errors
    threshold_indices_list = list(threshold_indices)
    
    if len(threshold_indices_list) == 0:
        return None, None
    
    # If threshold not provided, estimate it from the threshold crossings
    if threshold is None:
        threshold = min(magnitude[i] for i in threshold_indices_list)
    
    # Parameter validation
    before_count = max(1, min(before_count, len(magnitude) - 1))
    after_count = max(1, min(after_count, len(magnitude) - 1))
    
    # Calculate minimum counts to satisfy ratio requirements
    min_below_count = int(before_count * min_below_ratio)
    min_above_count = int(after_count * min_above_ratio)
    
    # Find start index - looking for transition from below threshold to above threshold
    start_idx = None
    # Scan through the signal, not just the threshold indices
    for idx in range(before_count, len(magnitude) - after_count):
        # Skip if not near a threshold crossing to speed up processing
        if idx not in threshold_indices_list and idx-1 not in threshold_indices_list and idx+1 not in threshold_indices_list:
            continue
            
        # Count samples below threshold before this point
        below_count_before = sum(1 for i in range(idx - before_count, idx) if magnitude[i] <= threshold)
        
        # Count samples above threshold after this point (including current)
        above_count_after = sum(1 for i in range(idx, idx + after_count) if magnitude[i] > threshold)
        
        # Check if this point represents a clean transition
        if below_count_before >= min_below_count and above_count_after >= min_above_count:
            start_idx = idx
            break
    
    # Find end index - looking for transition from above threshold to below threshold
    end_idx = None
    # Scan backward through the signal
    for idx in range(len(magnitude) - after_count - 1, before_count - 1, -1):
        # Skip if not near a threshold crossing
        if idx not in threshold_indices_list and idx-1 not in threshold_indices_list and idx+1 not in threshold_indices_list:
            continue
            
        # Count samples above threshold before this point (including current)
        above_count_before = sum(1 for i in range(idx - before_count + 1, idx + 1) if magnitude[i] > threshold)
        
        # Count samples below threshold after this point
        below_count_after = sum(1 for i in range(idx + 1, idx + after_count + 1) if magnitude[i] <= threshold)
        
        # Check if this point represents a clean transition
        if above_count_before >= min_below_count and below_count_after >= min_above_count:
            end_idx = idx
            break
    
    # If no suitable points found, use first/last threshold indices as fallback
    if start_idx is None and len(threshold_indices_list) > 0:
        start_idx = threshold_indices_list[0]
    if end_idx is None and len(threshold_indices_list) > 0:
        end_idx = threshold_indices_list[-1]
    
    return start_idx, end_idx

def GyroMagnitudeSegmentation(frequencies, raw_magnitude, gyro_data_trimmed, time_gyro, non_zero_index, threshold, plot_flag=False):
    from scipy.signal import butter
    from scipy.signal import filtfilt

    # Magitude Filtering:
    sampling_rate = frequencies[0]  # Hz
    nyquist = 0.5 * sampling_rate
    cutoff = 5  # Hz
    normal_cutoff = cutoff / nyquist
    b, a = butter(4, normal_cutoff, btype='low', analog=False)
    magnitude = filtfilt(b, a, raw_magnitude)

    # Storing the magnitude in the gyro_data dataframe:
    gyro_data_trimmed['magnitude'] = magnitude
    

    threshold = threshold
    threshold_indices = np.where(magnitude > threshold)[0]
    s_idx, e_idx = check_real_motion(magnitude, threshold_indices, threshold=threshold)
    start_idx, end_idx = detect_peak_boundaries(magnitude, s_idx, e_idx, threshold, visualize=plot_flag)
                                        
    if plot_flag:
        # Magnitude Analysis Plot
        plt.figure(figsize=(12, 8))  # Create a single plot
        plt.plot(time_gyro[non_zero_index:], magnitude, color='purple', label='Magnitude')
        plt.axhline(threshold, color='green', linestyle='--', label=f'Threshold ({threshold:.2f})')
        if start_idx is not None:
            plt.axvline(time_gyro[start_idx + non_zero_index], color='orange', label='Motion Start')
            plt.axvline(time_gyro[end_idx + non_zero_index], color='orange', label='Motion End')
        plt.title("Signal Magnitude Analysis")
        plt.xlabel("Time (s)")
        plt.ylabel("Composite Magnitude")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()
        
    return start_idx, end_idx