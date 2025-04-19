import numpy as np
from scipy import signal
from hmmlearn import hmm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings

class EnhancedHMMSegmenter:
    """
    Enhanced HMM motion segmenter with specific focus on improved start point detection
    """
    def __init__(self, min_duration=20, refinement_threshold=0.005, n_states=3, prob_threshold=0.6):
        self.min_duration = min_duration
        self.refinement_threshold = refinement_threshold
        self.n_states = n_states
        self.prob_threshold = prob_threshold
        self.model = None
        self._means = None
        self.noise_floor = None
        self.scaler = StandardScaler()
        
    def _preprocess(self, magnitude):
        """
        Preprocess the magnitude data for better feature extraction
        """
        # Hard filtering of extreme outliers
        q95 = np.percentile(magnitude, 95)
        q05 = np.percentile(magnitude, 5)
        threshold = 3 * (q95 - q05)
        
        filtered = np.copy(magnitude)
        outlier_mask = np.abs(filtered - np.median(filtered)) > threshold
        for i in np.where(outlier_mask)[0]:
            start = max(0, i-5)
            end = min(len(filtered), i+6)
            filtered[i] = np.median(magnitude[start:end])
        
        # Median filter to remove remaining spikes
        filtered = signal.medfilt(filtered, kernel_size=5)
        
        # Smoothing with Savitzky-Golay filter
        smoothed = signal.savgol_filter(filtered, 15, 3)
        
        # Store noise floor for later use
        self.noise_floor = np.percentile(np.abs(smoothed - magnitude), 10)
        
        return smoothed.reshape(-1, 1)
    
    def fit(self, magnitude):
        """
        Train HMM on magnitude data with multiple initializations
        """
        X = self._preprocess(magnitude)
        scaled_X = self.scaler.fit_transform(X)
        
        # Train with multiple restarts to avoid local optima
        best_score = -np.inf
        best_model = None
        
        for i in range(5):  # Try 5 different initializations
            try:
                model = hmm.GaussianHMM(
                    n_components=self.n_states, 
                    covariance_type="diag", 
                    n_iter=200,
                    random_state=i*42
                )
                
                model.fit(scaled_X)
                score = model.score(scaled_X)
                if score > best_score:
                    best_score = score
                    best_model = model
            except Exception as e:
                warnings.warn(f"HMM training attempt {i} failed: {str(e)}")
        
        if best_model is None:
            # Fallback to simple initialization if all restarts failed
            best_model = hmm.GaussianHMM(
                n_components=self.n_states,
                covariance_type="diag",
                n_iter=100,
                random_state=42
            ).fit(scaled_X)
        
        self.model = best_model
        self._means = self.model.means_.flatten()
        return self
    
    def _get_motion_state(self):
        """
        Identify which state corresponds to motion
        """
        return np.argmax(self._means)
    
    def _find_continuous_segments(self, motion_probs):
        """
        Find continuous segments using smoothed probabilities
        """
        # Apply smoothing to probabilities for more robust segmentation
        smoothed_probs = signal.savgol_filter(motion_probs, 15, 3)
        
        # Create mask based on threshold
        motion_mask = (smoothed_probs >= self.prob_threshold).astype(int)
        
        # Apply median filter to remove short spurious detections
        motion_mask = signal.medfilt(motion_mask, kernel_size=15)
        
        # Find contiguous segments
        diff = np.diff(np.concatenate(([0], motion_mask, [0])))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0] - 1
        
        # Filter segments by minimum duration
        valid_segments = [(s, e) for s, e in zip(starts, ends) if (e - s + 1) >= self.min_duration]
        
        return valid_segments
    
    def _compute_signal_gradient(self, magnitude, window_size=10):
        """
        Compute gradient of signal for edge detection
        """
        # Compute gradient using Savitzky-Golay to get smoother derivatives
        return signal.savgol_filter(magnitude, window_size*2+1, 2, deriv=1)
    
    def _compute_energy_profile(self, magnitude, window_size=20):
        """
        Compute energy profile for more robust motion detection
        """
        energy = np.zeros_like(magnitude)
        padded = np.pad(magnitude, (window_size//2, window_size//2), mode='edge')
        
        for i in range(len(magnitude)):
            window = padded[i:i+window_size]
            energy[i] = np.sum(window**2) / window_size
            
        return energy
    
    def _enhanced_start_detection(self, magnitude, rough_start, search_window=200):
        """
        Advanced start point detection using multiple strategies
        """
        min_idx = max(0, rough_start - search_window)
        
        # Strategy 1: Look for significant gradient increase
        gradient = self._compute_signal_gradient(magnitude)
        smoothed_gradient = signal.savgol_filter(gradient, 15, 3)
        
        # Scale gradient to 0-1 for easier thresholding
        if np.max(smoothed_gradient) > np.min(smoothed_gradient):
            scaled_gradient = (smoothed_gradient - np.min(smoothed_gradient)) / (np.max(smoothed_gradient) - np.min(smoothed_gradient))
        else:
            scaled_gradient = np.zeros_like(smoothed_gradient)
        
        # Strategy 2: Compute energy profile
        energy = self._compute_energy_profile(magnitude)
        smoothed_energy = signal.savgol_filter(energy, 15, 3)
        
        # Scale energy to 0-1
        if np.max(smoothed_energy) > np.min(smoothed_energy):
            scaled_energy = (smoothed_energy - np.min(smoothed_energy)) / (np.max(smoothed_energy) - np.min(smoothed_energy))
        else:
            scaled_energy = np.zeros_like(smoothed_energy)
        
        # Strategy 3: Threshold-based detection using noise level
        noise_level = np.median(magnitude[:min(rough_start, 100)])
        peak_value = np.max(magnitude[rough_start:rough_start+50])
        threshold = noise_level + 0.05 * (peak_value - noise_level)  # Very sensitive threshold
        
        # Look for the start point going backwards from rough_start
        gradient_start = rough_start
        for i in range(rough_start, min_idx, -1):
            if scaled_gradient[i] < 0.1 and scaled_gradient[i-1] < 0.1:
                gradient_start = i
                break
        
        energy_start = rough_start
        for i in range(rough_start, min_idx, -1):
            if scaled_energy[i] < 0.1 and scaled_energy[i-1] < 0.1:
                energy_start = i
                break
        
        threshold_start = rough_start
        for i in range(rough_start, min_idx, -1):
            if magnitude[i] < threshold and magnitude[i-1] < threshold:
                threshold_start = i
                break
        
        # Strategy 4: Look for consistent trend change
        trend_start = rough_start
        for i in range(rough_start, min_idx, -1):
            if i < 5:
                break
                
            # Check if 5 consecutive points are near baseline
            window = magnitude[i-5:i]
            if np.max(window) < threshold * 1.5:
                trend_start = i
                break
                
        # Combine results - use median for robustness
        all_starts = [gradient_start, energy_start, threshold_start, trend_start]
        refined_start = int(np.median(all_starts))
        
        # Make sure we didn't detect too early
        if refined_start < min_idx:
            refined_start = min_idx
            
        return refined_start
    
    def _refine_end_boundary(self, magnitude, rough_end, motion_probs):
        """
        End boundary refinement (keeping the successful implementation from before)
        """
        # Calculate peak value in the motion segment
        peak_value = np.max(magnitude[:rough_end+1])
        
        # Calculate adaptive thresholds 
        primary_thresh = max(self.refinement_threshold * peak_value, self.noise_floor * 2)
        
        # Enhanced end point detection with stability check
        refined_end = rough_end
        stability_window = 15
        
        # First search for consistent drop below threshold
        for i in range(rough_end, min(len(magnitude)-stability_window, rough_end+100)):
            window = magnitude[i:i+stability_window]
            if np.mean(window) < primary_thresh and np.max(window) < primary_thresh * 1.2:
                # Found a potential stable endpoint
                
                # Check if motion probability has also dropped
                if np.mean(motion_probs[i:i+stability_window]) < 0.3:
                    refined_end = i
                    break
        
        # Secondary gradient-based refinement for end point
        if refined_end == rough_end:
            # Use gradient information to find where signal flattens
            gradients = np.abs(np.gradient(magnitude[rough_end-50:rough_end+50]))
            smoothed_gradients = signal.savgol_filter(gradients, 11, 3)
            
            # Find where gradient becomes stable (approaches zero)
            stable_points = np.where(smoothed_gradients < np.mean(smoothed_gradients) * 0.2)[0]
            if len(stable_points) > 0:
                # Find first stable point after rough_end
                offset = stable_points[stable_points > 50][0] if any(stable_points > 50) else 50
                refined_end = min(rough_end - 50 + offset, len(magnitude)-1)
        
        return refined_end
    
    def segment(self, magnitude, plot=False):
        """
        Main segmentation method
        """
        if self.model is None:
            self.fit(magnitude)
        
        # Preprocess data
        X = self._preprocess(magnitude)
        scaled_X = self.scaler.transform(X)
        
        # Get state probabilities
        _, posteriors = self.model.score_samples(scaled_X)
        
        # Identify motion state
        motion_state = self._get_motion_state()
        motion_probs = posteriors[:, motion_state]
        
        # Find segments
        segments = self._find_continuous_segments(motion_probs)
        
        if not segments:
            return 0, len(magnitude)-1
        
        # Select segment with highest energy
        max_energy = 0
        best_segment = (0, 0)
        
        for start, end in segments:
            segment_energy = np.sum(magnitude[start:end+1]**2)
            if segment_energy > max_energy:
                max_energy = segment_energy
                best_segment = (start, end)
        
        rough_start, rough_end = best_segment
        
        # Enhanced start point detection - focus area for improvement
        refined_start = self._enhanced_start_detection(magnitude, rough_start)
        
        # End point detection (keeping the successful implementation from before)
        refined_end = self._refine_end_boundary(magnitude, rough_end, motion_probs)
        
        if plot:
            plt.figure(figsize=(14, 8))
            
            plt.subplot(2, 1, 1)
            plt.plot(magnitude, 'b', label='Signal')
            plt.axvline(refined_start, color='g', linestyle='--', label='Start')
            plt.axvline(refined_end, color='r', linestyle='--', label='End')
            plt.legend()
            plt.title('Signal with Motion Boundaries')
            
            plt.subplot(2, 1, 2)
            plt.plot(motion_probs, 'purple', label='Motion Probability')
            plt.axvline(refined_start, color='g', linestyle='--')
            plt.axvline(refined_end, color='r', linestyle='--')
            plt.axhline(self.prob_threshold, color='orange', linestyle='-', label='Threshold')
            plt.legend()
            plt.title('HMM Motion State Posterior Probabilities')
            
            plt.tight_layout()
            plt.show()
        
        return refined_start, refined_end


def HMMSegmentation(raw_magnitude, timestamp, plot_flag=False):
    """
    Interface function for HMM-based motion segmentation
    """
    # Convert to numpy array if needed
    mag = raw_magnitude if isinstance(raw_magnitude, np.ndarray) else raw_magnitude.to_numpy()
    
    # Create and use enhanced segmenter
    segmenter = EnhancedHMMSegmenter(
        min_duration=20, 
        refinement_threshold=0.005,
        n_states=3, 
        prob_threshold=0.6
    )
    
    start_idx, end_idx = segmenter.segment(mag, plot=plot_flag)
    
    print(f"Motion detected: Start={start_idx}, End={end_idx}")
    return start_idx, end_idx
