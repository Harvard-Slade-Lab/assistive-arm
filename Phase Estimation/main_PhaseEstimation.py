import BiasAndSegmentation
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog, Tk
import Interpolation
from scipy import interpolate
from Interpolation import interpolate_and_visualize

# Input the sensor frequencies
frequencies = BiasAndSegmentation.sensors_frequencies()

# Run the segmentation and bias correction
gyro, acc, orientation = BiasAndSegmentation.segmentation_and_bias(frequencies=frequencies)
# Interpolate and visualize
gyro_interp, acc_interp, orientation_interp = interpolate_and_visualize(gyro, acc, orientation)




plt.show(block=True)