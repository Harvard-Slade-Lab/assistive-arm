import numpy as np  # For numerical operations (if needed)
from threading import Lock  # For thread safety
from sklearn.base import BaseEstimator  # Assuming the model is a scikit-learn model

def estimated_phase(self):
    """
    Create a feature vector by concatenating ACC, GYRO, and OR data for all sensors
    using a single window size, predict phase using a trained model, and plot in real time.
    """
    # Get copies of the current data with thread safety
    with self.parent.plot_data_lock:
        plot_data_acc_copy = {k: {ax: v[ax][:] for ax in v} for k, v in self.parent.plot_data_acc.items()}
        plot_data_gyro_copy = {k: {ax: v[ax][:] for ax in v} for k, v in self.parent.plot_data_gyro.items()}
        plot_data_or_copy = {k: {ax: v[ax][:] for ax in v} for k, v in self.parent.plot_data_or.items()}
    
    # Create a feature vector by concatenating the most recent data from all sensors
    feature_vector = []
    
    # Get the list of sensor labels
    sensor_labels = list(self.parent.sensor_names.keys())
    do_it_once = 0
    # Concatenate ACC, GYRO, and OR data for each sensor
    for sensor_label in sensor_labels:
        # ACC data (X, Y, Z)
        if do_it_once == 0:
            do_it_once = 1
            if sensor_label in plot_data_acc_copy:
                for axis in ['X', 'Y', 'Z']:
                    data = plot_data_acc_copy[sensor_label].get(axis, [])
                    feature_vector.append(data[-1] if data else 0.0)
                print("Acc_FeatureVect", feature_vector)
        
        # GYRO data (X, Y, Z)
        if sensor_label in plot_data_gyro_copy:
            for axis in ['X', 'Y', 'Z']:
                data = plot_data_gyro_copy[sensor_label].get(axis, [])
                feature_vector.append(data[-1] if data else 0.0)
        
        # Orientation data (W, X, Y, Z)
        if sensor_label in plot_data_or_copy:
            for axis in ['W', 'X', 'Y', 'Z']:
                data = plot_data_or_copy[sensor_label].get(axis, [])
                feature_vector.append(data[-1] if data else 0.0)
    
    # Check if we have any features
    if not feature_vector:
        return

    # Compute the predicted phase using the trained model
    predicted_phase = self.parent.current_model.predict([feature_vector])[0]

    # saturate the predicted phase between 0 and 1
    predicted_phase = max(0, min(predicted_phase, 1))

    print(f"Predicted phase: {predicted_phase}")
    print(f"Feature vector: {feature_vector}")

    return predicted_phase
