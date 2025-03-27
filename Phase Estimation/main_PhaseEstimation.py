import BiasAndSegmentation
import matplotlib.pyplot as plt

# Select the file you want to take the Data from:
file_path = BiasAndSegmentation.select_file()

# Load the data from the selected file
data = BiasAndSegmentation.load_csv(file_path)

# Input the sensor frequencies
frequencies = BiasAndSegmentation.sensors_frequencies()

# Run the segmentation and bias correction
isolated_movement = BiasAndSegmentation.segmentation_and_bias(file_path=file_path)

