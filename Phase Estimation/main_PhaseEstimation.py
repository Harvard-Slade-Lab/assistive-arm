import BiasAndSegmentation
import matplotlib.pyplot as plt

# Input the sensor frequencies
frequencies = BiasAndSegmentation.sensors_frequencies()

# Run the segmentation and bias correction
isolated_movement = BiasAndSegmentation.segmentation_and_bias(frequencies=frequencies)

