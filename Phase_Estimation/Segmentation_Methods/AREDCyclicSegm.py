import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from tkinter.filedialog import askdirectory
from Interpolation import interpolate_and_visualize
import DataLoader
import MatrixCreator

# ----------- HYPERPARAMETERS -----------------
# Hyperparameters for bias removal
bias_average_window = 1000 # Number of samples to average for bias removal
frequency = 519

# Hyperparameters for motion segmentation
window_size = 20           # Size of the sliding window 
threshold_factor = 0.1     # Factor to scale down the threshold (30% of Otsu's threshold)
min_duration = 1          # Minimum duration of a motion segment in samples
refinement_threshold = 0.005 # Threshold for edge refinement (5% of peak value)

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

def AREDSegmCycl(magnitude):    
    # Compute ARED signal
    ared = RefinedMotionSegmenter(window_size, threshold_factor, min_duration, refinement_threshold).compute_ared(magnitude)

    # plot ared:
    plt.figure(figsize=(10, 5))
    plt.plot(ared, label='ARED Signal')
    plt.title('Angular Rate Energy Detector Signal')
    plt.xlabel('Sample Index')
    plt.ylabel('ARED Value')
    plt.legend()
    plt.grid()
    plt.show(block=True)