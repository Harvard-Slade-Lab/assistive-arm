import BiasAndSegmentation
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog, Tk
import Interpolation
from scipy import interpolate
from Interpolation import interpolate_and_visualize
from RidgeRegressionCV import ridge_regression
from RidgeRegressionCV import enhanced_ridge_regression

# Input the sensor frequencies
frequencies = BiasAndSegmentation.sensors_frequencies()

# Run the segmentation and bias correction
gyro, acc, orientation = BiasAndSegmentation.segmentation_and_bias(frequencies=frequencies)

# Interpolate and visualize
gyro_interp, acc_interp, orientation_interp = interpolate_and_visualize(gyro, acc, orientation)

# Perform Ridge regression
result = enhanced_ridge_regression(
    gyro_interp,
    acc_interp,
    orientation_interp,
    alpha_range=(-7, 7, 40),  # Custom range: 10^-7 to 10^7
    cv=None                 
)


plt.show(block=True)