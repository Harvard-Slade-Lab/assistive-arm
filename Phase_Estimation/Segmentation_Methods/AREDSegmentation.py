import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal


# ----------- HYPERPARAMETERS -----------------
# Hyperparameters for bias removal
bias_average_window = 1000 # Number of samples to average for bias removal
frequency = 519

# Hyperparameters for motion segmentation
window_size = 20           # Size of the sliding window 
threshold_factor = 0.1     # Factor to scale down the threshold (30% of Otsu's threshold)
min_duration = 1          # Minimum duration of a motion segment in samples
refinement_threshold = 0.005 # Threshold for edge refinement (5% of peak value)
merge_threshold = 50 # Threshold for merging segments (in samples)

class RefinedMotionSegmenter:
    def __init__(self, window_size=20, threshold_factor=0.003, min_duration=1000, refinement_threshold=0.005):
        """
        Initialize the motion segmenter with refinement capabilities.
        
        Parameters:
        -----------
        window_size : int
            Size of the sliding window for calculating motion energy
        threshold_factor : float
            Factor to scale down the automatically determined threshold (0.3 means 30% of optimal)
        min_duration : int
            Minimum duration of a motion segment in samples
        refinement_threshold : float
            Threshold for edge refinement as a fraction of peak value (0.05 = 5% of peak)
        """
        self.window_size = window_size
        self.threshold_factor = threshold_factor
        self.min_duration = min_duration
        self.refinement_threshold = refinement_threshold
    
    def compute_ared(self, magnitude):
        """
        Compute the Angular Rate Energy Detector signal.
        """
        ared_signal = np.zeros_like(magnitude)
        half_window = self.window_size // 2
        
        for i in range(len(magnitude) - self.window_size + 1):
            window = magnitude[i:i+self.window_size]
            ared_signal[i + half_window] = np.mean(window**2)
        
        # Fill the boundary values
        for i in range(half_window):
            ared_signal[i] = ared_signal[half_window]
            ared_signal[-(i+1)] = ared_signal[-(half_window+1)]
            
        return ared_signal
    
    def find_optimal_threshold(self, ared_signal):
        """
        Find a threshold using Otsu's method and scale it down.
        """
        # Use histogram to get distribution
        hist, bin_edges = np.histogram(ared_signal, bins=100)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Calculate cumulative sums
        cum_sum = hist.cumsum()
        if cum_sum[-1] == 0:
            return 0
        
        cum_sum_norm = cum_sum / cum_sum[-1]
        
        # Find threshold that maximizes between-class variance (Otsu's method)
        max_variance = 0
        optimal_threshold = 0
        
        for i in range(1, len(hist)):
            # Weight of the classes
            w0 = cum_sum_norm[i-1]
            w1 = 1 - w0
            
            if w0 == 0 or w1 == 0:
                continue
            
            # Mean of the classes
            mu0 = np.sum(hist[:i] * bin_centers[:i]) / np.sum(hist[:i]) if np.sum(hist[:i]) > 0 else 0
            mu1 = np.sum(hist[i:] * bin_centers[i:]) / np.sum(hist[i:]) if np.sum(hist[i:]) > 0 else 0
            
            # Between-class variance
            variance = w0 * w1 * (mu0 - mu1)**2
            
            if variance > max_variance:
                max_variance = variance
                optimal_threshold = bin_centers[i]
        
        # Scale down the threshold to capture more of the motion
        return optimal_threshold * self.threshold_factor
    
    def estimate_noise_floor(self, ared_signal):
        """
        Estimate the noise floor from the ARED signal.
        """
        # Sort the ARED signal values
        sorted_values = np.sort(ared_signal)
        
        # Use the 10th percentile as an estimate of the noise floor
        noise_floor = np.percentile(sorted_values, 10)
        
        return noise_floor
    
    def refine_motion_boundaries(self, ared_signal, rough_start, rough_end):
        """
        Refine the motion boundaries by searching for true start and end points.
        
        Parameters:
        -----------
        ared_signal : array-like
            The ARED signal
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
        # Calculate peak value in the motion segment
        peak_value = np.max(ared_signal[rough_start:rough_end+1])
        
        # Calculate the edge detection threshold as a percentage of peak value
        # Ensure it's above the noise floor
        noise_floor = self.estimate_noise_floor(ared_signal)
        edge_threshold = max(self.refinement_threshold * peak_value, noise_floor * 2)
        

        # Verify refinement with rate of change to avoid noise artifacts
        # Calculate first derivative of ARED signal
        ared_diff = np.diff(ared_signal)
        
        # Smooth the derivative
        ared_diff_smoothed = signal.savgol_filter(np.concatenate(([0], ared_diff)), 
                                                 window_length=11, polyorder=2)
        refined_start = rough_start
        refined_end = rough_end

        # For start point: look for significant positive rate of change
        start_search_range = max(0, rough_start - 100), min(len(ared_diff_smoothed)-1, rough_start)
        for i in range(start_search_range[1], start_search_range[0], -1):
            if ared_diff_smoothed[i] < ared_diff_smoothed[rough_start] * 0.2:
                refined_start = i
                break
        
        # For end point: look for significant negative rate of change
        end_search_range = max(0, rough_end), min(len(ared_diff_smoothed)-1, rough_end + 100)
        for i in range(end_search_range[0], end_search_range[1]):
            if ared_diff_smoothed[i] > ared_diff_smoothed[rough_end] * 0.2:
                refined_end = i
                break


        return refined_start, refined_end
    
    def segment_motion(self, magnitude):
        """
        Segment motion from gyroscope magnitude data with boundary refinement.
        """
        # Step 1: Apply signal smoothing to reduce noise
        smoothed_magnitude = signal.savgol_filter(magnitude, window_length=15, polyorder=3)
        
        # Step 2: Apply ARED detector
        ared_signal = self.compute_ared(smoothed_magnitude)
        
        # Step 3: Determine a lower threshold
        threshold = self.find_optimal_threshold(ared_signal)
        
        # Ensure threshold is above noise floor but low enough to capture motion
        noise_floor = self.estimate_noise_floor(ared_signal)
        threshold = max(threshold, noise_floor * 1.5)
        
        # Step 4: Apply threshold to get binary mask
        binary_mask = (ared_signal > threshold).astype(int)
        
        # # Step 5: Apply median filter to remove noise
        # filtered_mask = signal.medfilt(binary_mask, kernel_size=5)
        
        # Step 6: Find contiguous segments
        diff = np.diff(np.concatenate(([0], binary_mask, [0])))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0] - 1  # Adjust to get the last 1 in each segment
        
        # Step 7: Filter segments by duration
        valid_segments = [(start, end) for start, end in zip(starts, ends) 
                         if (end - start + 1) >= self.min_duration]
        
        if not valid_segments:
            # No valid segments found
            return 0, len(magnitude) - 1, ared_signal, threshold
        
        # Step 7.1: Merge segments if the distance between them is below a threshold
        merge_threshold = merge_threshold  # Define a threshold for merging segments (in samples)
        merged_segments = []
        current_start, current_end = valid_segments[0]

        for start, end in valid_segments[1:]:
            if start - current_end <= merge_threshold:
                # Merge the segments
                current_end = end
            else:
                # Save the current segment and start a new one
                merged_segments.append((current_start, current_end))
                current_start, current_end = start, end

        # Append the last segment
        merged_segments.append((current_start, current_end))

        valid_segments = merged_segments
        # Step 8: Find the segment with the highest energy
        segment_energies = [np.sum(magnitude[start:end+1]**2) / (end-start+1) 
                          for start, end in valid_segments]
        main_segment_idx = np.argmax(segment_energies)
        
        rough_start, rough_end = valid_segments[main_segment_idx]

        
        
        # Step 9: Refine the motion boundaries
        refined_start, refined_end = self.refine_motion_boundaries(ared_signal, rough_start, rough_end)
        
        return refined_start, refined_end, ared_signal, threshold, rough_start, rough_end
    
    def plot_segmentation(self, magnitude, motion_start, motion_end, rough_start, rough_end, ared_signal=None, threshold=None, frequency=None):
        """
        Plot gyroscope data with motion segmentation.
        """
        if frequency is None:
            time = np.arange(len(magnitude))
            xlabel = 'Samples'
        else:
            time = np.arange(len(magnitude)) / frequency
            xlabel = 'Time (s)'
        
        plt.figure(figsize=(12, 6))
        
        plt.plot(time, magnitude, 'purple', label='Magnitude')
        
        if ared_signal is not None and threshold is not None:
            plt.plot(time, ared_signal, 'blue', alpha=0.5, label='ARED Signal')
            plt.axhline(y=threshold, color='g', linestyle='--', label=f'Threshold ({threshold:.2f})')
        
        plt.axvline(x=time[motion_start], color='orange', linestyle='-', label='Motion Start')
        plt.axvline(x=time[motion_end], color='orange', linestyle='-', label='Motion End')
        
        # Plot rough_start and rough_end as stars
        plt.plot(time[rough_start], ared_signal[rough_start], 'r*', markersize=10, label='Rough Start')
        plt.plot(time[rough_end], ared_signal[rough_end], 'b*', markersize=10, label='Rough End')
        
        plt.xlabel(xlabel)
        plt.ylabel('Magnitude')
        plt.title('Signal Magnitude with Motion Segmentation')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

def AREDSegmentation(raw_magnitude, timestamp, plot_flag=False):

    # Initialize motion segmenter
    segmenter = RefinedMotionSegmenter(
        window_size=window_size,
        threshold_factor=threshold_factor,
        min_duration=min_duration,
        refinement_threshold=refinement_threshold
    )
    # Dictionary to store motion indices for each timestamp
    motion_indices = {}

    print("Segmenting motion...")
    
    # Convert raw_magnitude to numpy array for compatibility with RefinedMotionSegmenter
    magnitude_np = raw_magnitude.to_numpy()
    
    # Segment motion
    motion_start, motion_end, ared_signal, threshold, rough_start, rough_end = segmenter.segment_motion(magnitude_np)
    
    # Store motion indices
    # motion_indices[timestamp] = (motion_start, motion_end)
    
    print(f"Motion detected: Start={motion_start}, End={motion_end}")
    
    # Visualize the segmentation
    if plot_flag:
        segmenter.plot_segmentation(magnitude_np, motion_start, motion_end, rough_start, rough_end, ared_signal, threshold, frequency)

    start_idx = motion_start
    end_idx = motion_end

    return start_idx, end_idx

    