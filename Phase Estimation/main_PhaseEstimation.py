import BiasAndSegmentation
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Interpolation import interpolate_and_visualize
from RidgeRegressionCV import enhanced_ridge_regression
from RidgeRegressionCV import test_ridge
from LassoRegressionCV import enhanced_lasso_regression, test_lasso



# Input the sensor frequencies
frequencies = BiasAndSegmentation.sensors_frequencies()
# Run the segmentation and bias correction
gyro, acc, orientation, time_gyro_segmented, time_acc_segmented, time_orientation_segmented = BiasAndSegmentation.segmentation_and_bias(frequencies=frequencies)
# Interpolate the data
gyro_interp, acc_interp, or_interp = interpolate_and_visualize(gyro, acc, orientation, frequencies=frequencies)

# ------------------------ RIDGE REGRESSION ------------------------
# Perform Ridge regression
result = enhanced_ridge_regression(
    gyro_interp,
    acc_interp,
    or_interp,
    alpha_range=(-7, 7, 40),  # Custom range: 10^-7 to 10^7
    cv=None,
    frequencies=frequencies,                 
)
# Extract the Ridge model:
model = result['model']
# TEST RIDGE
# Run the segmentation and bias correction
gyro, acc, orientation, time_gyro_segmented, time_acc_segmented, time_orientation_segmented = BiasAndSegmentation.segmentation_and_bias(frequencies=frequencies)
# Interpolate the data
gyro_interp, acc_interp, or_interp = interpolate_and_visualize(gyro, acc, orientation, frequencies=frequencies)
# Perform the test 
test_ridge(
    model,
    gyro_interp,
    acc_interp,
    or_interp,
    frequencies=frequencies,
)

# ------------------------ LASSO REGRESSION ------------------------
# Perform Lasso regression
lasso_result = enhanced_lasso_regression(
    gyro_interp,
    acc_interp,
    or_interp,
    alpha_range=(-7, 7, 40),  # Custom range: 10^-7 to 10^7
    cv=None,
    frequencies=frequencies,                 
)
# Extract the Lasso model:
lasso_model = lasso_result['model']
# TEST LASSO
# Run the segmentation and bias correction
gyro, acc, orientation, time_gyro_segmented, time_acc_segmented, time_orientation_segmented = BiasAndSegmentation.segmentation_and_bias(frequencies=frequencies)
# Interpolate the data
gyro_interp, acc_interp, or_interp = interpolate_and_visualize(gyro, acc, orientation, frequencies=frequencies)
# Perform the test
test_lasso(
    lasso_model,
    gyro_interp,
    acc_interp,
    or_interp,
    frequencies=frequencies,
)   





plt.show(block=True)