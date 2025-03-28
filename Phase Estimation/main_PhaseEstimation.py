import BiasAndSegmentation
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog, Tk
import Interpolation
from scipy import interpolate

# Input the sensor frequencies
frequencies = BiasAndSegmentation.sensors_frequencies()

# Run the segmentation and bias correction
gyro, acc, orientation = BiasAndSegmentation.segmentation_and_bias(frequencies=frequencies)
gyro, acc, orientation = Interpolation.process_signals(gyro, acc, orientation)