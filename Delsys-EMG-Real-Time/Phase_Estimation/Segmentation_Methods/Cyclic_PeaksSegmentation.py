import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

def segment_gait_cycles(magnitude_signal, time_vector, plot_results=True):

    # Find peaks with constraints
    peaks, _ = find_peaks(magnitude_signal, 
                          height=150,          # Minimum peak height (adjust based on your data)
                          distance=100,         # Minimum samples between peaks
                          prominence=50)       # Minimum peak prominence
    
    # Find valleys (local minima) preceding the peaks
    valleys, _ = find_peaks(-magnitude_signal, 
                            distance=100,
                            prominence=18)       # Minimum samples between valleys

    # Filter valleys to ensure they precede the detected peaks
    filtered_valleys = []
    for peak in peaks:
        preceding_valleys = [valley for valley in valleys if valley < peak]
        if preceding_valleys:
            filtered_valleys.append(preceding_valleys[-1])  # Take the closest preceding valley

    valleys = np.array(filtered_valleys)
    
    # Create segments based on valleys
    segments = []
    for i in range(len(valleys)-1):
        start_idx = valleys[i]
        end_idx = valleys[i+1]
        segments.append((start_idx, end_idx))
    
    # Plot results if requested
    if plot_results:
        plt.figure(figsize=(12, 6))
        plt.plot(time_vector, magnitude_signal, 'b-', label='Magnitude')
        plt.plot(time_vector[peaks], magnitude_signal[peaks], 'ro', label='Detected Peaks')
        plt.plot(time_vector[valleys], magnitude_signal[valleys], 'go', label='Detected Valleys')
        
        # Highlight segments
        for i, (start, end) in enumerate(segments):
            plt.axvspan(time_vector[start], time_vector[end], 
                      alpha=0.2, color='g', label='Segment' if i==0 else None)
        
        plt.legend()
        plt.title('Gait Cycle Segmentation')
        plt.xlabel('Time (s)')
        plt.ylabel('Magnitude')
        plt.grid(True)
        plt.show()

    return segments