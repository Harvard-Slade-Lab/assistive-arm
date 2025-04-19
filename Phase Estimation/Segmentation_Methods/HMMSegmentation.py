import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import warnings
from sklearn.cluster import KMeans
from scipy.signal import find_peaks

class GVMAREDSegmenter:
    """
    Motion segmentation using Gyroscope Vector Magnitude (GVM) with
    Angular Rate Energy Detection (ARED)
    """
    def __init__(self, window_size=30, min_duration=500, ared_window=20):
        # Configuration parameters
        self.window_size = window_size  # Smoothing window
        self.min_duration = min_duration  # Minimum segment duration
        self.ared_window = ared_window  # ARED calculation window
        
    def compute_gvm(self, gyro_magnitude):
        """
        Process gyroscope magnitude data
        (In 3-axis case, this would compute sqrt(gx^2 + gy^2 + gz^2))
        """
        # For single-axis gyro data, just return the absolute value
        return np.abs(gyro_magnitude)
    
    def smooth_signal(self, signal_data):
        """Apply robust smoothing to the signal"""
        # Remove outliers
        q75, q25 = np.percentile(signal_data, [75, 25])
        iqr = q75 - q25
        filtered = np.clip(signal_data, q25 - 1.5*iqr, q75 + 1.5*iqr)
        
        # Median filtering for spike removal
        filtered = signal.medfilt(filtered, kernel_size=5)
        
        # Savitzky-Golay smoothing preserves peaks better than moving average
        smoothed = signal.savgol_filter(filtered, self.window_size+1, 3)
        
        return smoothed
    
    def compute_ared(self, gvm_data):
        """
        Compute Angular Rate Energy Detector (ARED)
        """
        length = len(gvm_data)
        ared = np.zeros(length)
        half_window = self.ared_window // 2
        
        # Calculate energy in sliding windows
        for i in range(length):
            start = max(0, i - half_window)
            end = min(length, i + half_window + 1)
            window = gvm_data[start:end]
            ared[i] = np.mean(window**2)  # Use mean for window size invariance
            
        return ared
    
    def find_otsu_threshold(self, ared_signal):
        """
        Find optimal threshold using Otsu's method
        """
        # Normalize to 0-255 range for Otsu
        data_range = np.max(ared_signal) - np.min(ared_signal)
        if data_range == 0:
            return 0
            
        normalized = ((ared_signal - np.min(ared_signal)) / data_range * 255).astype(np.uint8)
        
        # Use K-means to find optimal threshold (similar to Otsu but more robust)
        data = normalized.reshape(-1, 1)
        kmeans = KMeans(n_clusters=2, random_state=0).fit(data)
        centers = kmeans.cluster_centers_.flatten()
        
        # Get threshold between cluster centers
        threshold = (centers[0] + centers[1]) / 2
        
        # Map back to original range
        return threshold / 255 * data_range + np.min(ared_signal)
    
    def segment(self, gyro_magnitude, plot=False):
        """Main segmentation method"""
        try:
            # 1. Preprocess the magnitude data
            gvm = self.compute_gvm(gyro_magnitude)
            gvm_smooth = self.smooth_signal(gvm)
            
            # 2. Compute ARED signal
            ared_signal = self.compute_ared(gvm_smooth)
            
            # 3. Find threshold using Otsu's method
            threshold = self.find_otsu_threshold(ared_signal)
            
            # 4. Create binary mask using thresholding
            motion_mask = (ared_signal > threshold).astype(int)
            
            # 5. Apply median filter to remove noise in mask
            motion_mask = signal.medfilt(motion_mask, kernel_size=15)
            
            # 6. Find contiguous segments
            diff = np.diff(np.concatenate(([0], motion_mask, [0])))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0] - 1
            
            # 7. Filter segments by minimum duration
            valid_segments = [(s, e) for s, e in zip(starts, ends) 
                             if (e - s + 1) >= self.min_duration]
            
            if not valid_segments:
                return 0, len(gyro_magnitude)-1
                
            # 8. Select segment with highest energy
            segment_energies = [np.sum(gvm_smooth[s:e+1]**2) / (e-s+1) for s, e in valid_segments]
            main_segment_idx = np.argmax(segment_energies)
            rough_start, rough_end = valid_segments[main_segment_idx]
            
            # 9. Refine boundaries by looking for near-zero velocity
            refined_start = rough_start
            refined_end = rough_end
            
            # Define dynamic threshold as a fraction of peak value
            peak_value = np.max(gvm_smooth[rough_start:rough_end+1])
            base_value = np.median(gvm_smooth[:min(100, rough_start)] if rough_start > 0 else gvm_smooth[:100])
            dynamic_threshold = base_value + 0.05 * (peak_value - base_value)
            
            # Refine start - look for point where signal first exceeds threshold
            for i in range(rough_start, max(0, rough_start-200), -1):
                if gvm_smooth[i] <= dynamic_threshold:
                    refined_start = i + 1
                    break
            
            # Refine end - look for point where signal first goes below threshold
            for i in range(rough_end, min(len(gvm_smooth), rough_end+200)):
                if gvm_smooth[i] <= dynamic_threshold:
                    refined_end = i - 1
                    break
            
            if plot:
                self.plot_segmentation(gyro_magnitude, gvm_smooth, ared_signal, 
                                      threshold, rough_start, rough_end, 
                                      refined_start, refined_end)
                
            return refined_start, refined_end
            
        except Exception as e:
            warnings.warn(f"Segmentation failed: {str(e)}")
            return 0, len(gyro_magnitude)-1
    
    def plot_segmentation(self, gyro_magnitude, gvm_smooth, ared_signal, 
                         threshold, rough_start, rough_end, 
                         refined_start, refined_end):
        """Visualize segmentation with all components"""
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Original signal with boundaries
        plt.subplot(3, 1, 1)
        plt.plot(gyro_magnitude, 'b', label='Gyro Magnitude')
        plt.axvline(refined_start, color='g', linestyle='-', label='Refined Start')
        plt.axvline(refined_end, color='r', linestyle='-', label='Refined End')
        plt.axvline(rough_start, color='g', linestyle='--', label='Rough Start')
        plt.axvline(rough_end, color='r', linestyle='--', label='Rough End')
        plt.legend()
        plt.title('Gyro Magnitude Signal')
        
        # Plot 2: Smoothed GVM
        plt.subplot(3, 1, 2)
        plt.plot(gvm_smooth, 'purple', label='Smoothed GVM')
        plt.axvline(refined_start, color='g', linestyle='-')
        plt.axvline(refined_end, color='r', linestyle='-')
        
        # Show dynamic threshold
        peak_value = np.max(gvm_smooth[rough_start:rough_end+1])
        base_value = np.median(gvm_smooth[:min(100, rough_start)] if rough_start > 0 else gvm_smooth[:100])
        dynamic_threshold = base_value + 0.05 * (peak_value - base_value)
        plt.axhline(dynamic_threshold, color='k', linestyle='--', label='Dynamic Threshold')
        
        plt.legend()
        plt.title('Smoothed Gyroscope Vector Magnitude (GVM)')
        
        # Plot 3: ARED signal
        plt.subplot(3, 1, 3)
        plt.plot(ared_signal, 'orange', label='ARED Signal')
        plt.axhline(threshold, color='k', linestyle='--', label='ARED Threshold')
        plt.axvline(refined_start, color='g', linestyle='-')
        plt.axvline(refined_end, color='r', linestyle='-')
        plt.legend()
        plt.title('Angular Rate Energy Detector (ARED)')
        
        plt.tight_layout()
        plt.show()


def HMMSegmentation(raw_magnitude, timestamp, plot_flag=False):
    """Interface function for GVM-ARED segmentation"""
    try:
        # Convert to numpy array if needed
        if not isinstance(raw_magnitude, np.ndarray):
            magnitude = np.array(raw_magnitude).flatten()
        else:
            magnitude = raw_magnitude.flatten() if raw_magnitude.ndim > 1 else raw_magnitude
            
        # Create segmenter with adjusted parameters for better sensitivity
        segmenter = GVMAREDSegmenter(
            window_size=25,   # For smoothing
            min_duration=500,  # Minimum segment length
            ared_window=20    # ARED calculation window - smaller window for better response
        )
        
        start_idx, end_idx = segmenter.segment(magnitude, plot=plot_flag)
        print(f"Motion detected: Start={start_idx}, End={end_idx}")
        return start_idx, end_idx
        
    except Exception as e:
        print(f"Segmentation failed: {str(e)}")
        return 0, len(raw_magnitude)-1
