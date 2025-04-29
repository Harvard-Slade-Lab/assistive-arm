import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore
import numpy as np
import threading
import os
import joblib

class Plotter:
    def __init__(self, parent):
        self.parent = parent
        self.plot_timer = QtCore.QTimer()
        self.plot_timer.timeout.connect(self.update_plot)
        self.plot_timer.start(50)  # Update plot every 50 ms

    def initialize_plot(self):
        """Initialize the plot for EMG, ACC, GYRO, OR, and Phase Estimation."""
        self.parent.setWindowTitle(f'Subject {self.parent.subject_number} Trial {self.parent.trial_number}')
        sensor_labels = list(self.parent.sensor_names.keys())
        total_rows = len(sensor_labels)
        self.parent.plot_layout = QtWidgets.QGridLayout(self.parent.plot_widget)

        self.emg_plots = {}
        self.acc_plots = {}
        self.gyro_plots = {}
        self.or_plots = {}

        # Phase Estimation Plot
        self.phase_plot = pg.PlotWidget(title='Estimated Phase')
        self.phase_plot.setLabel('left', 'Phase', units='(0-1)')
        self.phase_plot.setLabel('bottom', 'Time', units='s')
        self.phase_plot.showGrid(x=True, y=True)
        self.parent.plot_layout.addWidget(self.phase_plot, 0, 4, total_rows, 1)
        self.phase_history = []
        self.phase_time = []

        for i, sensor_label in enumerate(sensor_labels):
            sensor_name = self.parent.sensor_names.get(sensor_label, f"Sensor {sensor_label}")

            # EMG plot
            if sensor_label in self.parent.plot_data_emg:
                pw_emg = pg.PlotWidget(title=f'EMG {sensor_name}')
                pw_emg.setLabel('left', 'Amplitude', units='V')
                pw_emg.setLabel('bottom', 'Time', units='s')
                pw_emg.showGrid(x=True, y=True)
                self.parent.plot_layout.addWidget(pw_emg, i, 0)
                self.emg_plots[sensor_label] = pw_emg
            else:
                self.parent.plot_layout.addWidget(QtWidgets.QWidget(), i, 0)

            # ACC plot
            if sensor_label in self.parent.acc_channels_per_sensor:
                pw_acc = pg.PlotWidget(title=f'ACC {sensor_name}')
                pw_acc.setLabel('left', 'Acceleration', units='g')
                pw_acc.setLabel('bottom', 'Time', units='s')
                pw_acc.showGrid(x=True, y=True)
                self.parent.plot_layout.addWidget(pw_acc, i, 1)
                self.acc_plots[sensor_label] = pw_acc
            else:
                self.parent.plot_layout.addWidget(QtWidgets.QWidget(), i, 1)

            # GYRO plot
            if sensor_label in self.parent.gyro_channels_per_sensor:
                pw_gyro = pg.PlotWidget(title=f'GYRO {sensor_name}')
                pw_gyro.setLabel('left', 'Angular Velocity', units='dps')
                pw_gyro.setLabel('bottom', 'Time', units='s')
                pw_gyro.showGrid(x=True, y=True)
                self.parent.plot_layout.addWidget(pw_gyro, i, 2)
                self.gyro_plots[sensor_label] = pw_gyro
            else:
                self.parent.plot_layout.addWidget(QtWidgets.QWidget(), i, 2)

            # Orientation plot
            if sensor_label in self.parent.or_channels_per_sensor:
                pw_or = pg.PlotWidget(title=f'OR {sensor_name}')
                pw_or.setLabel('left', 'Orientation', units='deg')
                pw_or.setLabel('bottom', 'Time', units='s')
                pw_or.showGrid(x=True, y=True)
                self.parent.plot_layout.addWidget(pw_or, i, 3)
                self.or_plots[sensor_label] = pw_or

    def update_plot(self):
        """Update the plot with new data and estimate phase."""
        with self.parent.plot_data_lock:
            plot_data_emg_copy = {k: v.copy() for k, v in self.parent.plot_data_emg.items()}
            plot_data_acc_copy = {k: {ax: v[ax][:] for ax in v} for k, v in self.parent.plot_data_acc.items()}
            plot_data_gyro_copy = {k: {ax: v[ax][:] for ax in v} for k, v in self.parent.plot_data_gyro.items()}
            plot_data_or_copy = {k: {ax: v[ax][:] for ax in v} for k, v in self.parent.plot_data_or.items()}

        # Calculate elapsed time based on EMG data
        elapsed_times = []
        for sensor_label, data in plot_data_emg_copy.items():
            if data:
                sample_rate = self.parent.emg_sampling_frequencies[sensor_label]
                num_samples = len(data)
                elapsed_time = num_samples / sample_rate
                elapsed_times.append(elapsed_time)
        if elapsed_times:
            elapsed_time = max(elapsed_times)
        else:
            elapsed_time = 0

        self.parent.time_label.setText(f'Elapsed Time: {self.parent.total_elapsed_time + elapsed_time:.2f} s')

        # EMG Plots
        for sensor_label, data in plot_data_emg_copy.items():
            if len(data) == 0:
                continue
            y = np.array(data)
            sample_rate = self.parent.emg_sampling_frequencies[sensor_label]
            time_array = np.arange(len(y)) / sample_rate + self.parent.total_elapsed_time
            self.emg_plots[sensor_label].plot(time_array, y, clear=True, pen='g')
            self.emg_plots[sensor_label].setXRange(self.parent.total_elapsed_time, self.parent.total_elapsed_time + self.parent.window_duration)

        # ACC Plots
        for sensor_label, data_dict in plot_data_acc_copy.items():
            pw_acc = self.acc_plots[sensor_label]
            pw_acc.clear()
            sample_rate = self.parent.acc_sample_rates[sensor_label]
            for axis in ['X', 'Y', 'Z']:
                data = data_dict[axis]
                if len(data) == 0:
                    continue
                y = np.array(data)
                time_array = np.arange(len(y)) / sample_rate + self.parent.total_elapsed_time
                pw_acc.plot(time_array, y, pen=pg.mkPen({'X': 'r', 'Y': 'g', 'Z': 'b'}[axis]), name=axis)
            pw_acc.setXRange(self.parent.total_elapsed_time, self.parent.total_elapsed_time + self.parent.window_duration)
            pw_acc.addLegend()

        # GYRO Plots
        for sensor_label, data_dict in plot_data_gyro_copy.items():
            pw_gyro = self.gyro_plots[sensor_label]
            pw_gyro.clear()
            sample_rate = self.parent.gyro_sample_rates[sensor_label]
            for axis in ['X', 'Y', 'Z']:
                data = data_dict[axis]
                if len(data) == 0:
                    continue
                y = np.array(data)
                time_array = np.arange(len(y)) / sample_rate + self.parent.total_elapsed_time
                pw_gyro.plot(time_array, y, pen=pg.mkPen({'X': 'r', 'Y': 'g', 'Z': 'b'}[axis]), name=axis)
            pw_gyro.setXRange(self.parent.total_elapsed_time, self.parent.total_elapsed_time + self.parent.window_duration)
            pw_gyro.addLegend()

        # Orientation Plots
        for sensor_label, data_dict in plot_data_or_copy.items():
            pw_or = self.or_plots[sensor_label]
            pw_or.clear()
            sample_rate = self.parent.or_sample_rates[sensor_label]
            for axis in ['W', 'X', 'Y', 'Z']:
                data = data_dict[axis]
                if len(data) == 0:
                    continue
                y = np.array(data)
                time_array = np.arange(len(y)) / sample_rate + self.parent.total_elapsed_time
                pw_or.plot(time_array, y, pen=pg.mkPen({'W': 'r', 'X': 'g', 'Y': 'b', 'Z': 'm'}[axis]), name=axis)
            pw_or.setXRange(self.parent.total_elapsed_time, self.parent.total_elapsed_time + self.parent.window_duration)
            pw_or.addLegend()

        if self.parent.current_model is not None:
            # Phase estimation and plot
            predicted_phase = self.estimated_phase()

            # Plot Predicted Phase
            if predicted_phase is not None:
                self.phase_history.append(predicted_phase)
                self.phase_time.append(self.parent.total_elapsed_time + elapsed_time)
                self.phase_plot.clear()
                self.phase_plot.plot(self.phase_time, self.phase_history, pen='y', name='Predicted Phase')
                self.phase_plot.setXRange(self.parent.total_elapsed_time, self.parent.total_elapsed_time + self.parent.window_duration)
                self.phase_plot.setYRange(0, 1)

        # Check if it reached the window size, increment the cumulative time
        window_reached = any(len(data) >= self.parent.emg_window_sizes[sensor_label] for sensor_label, data in self.parent.plot_data_emg.items())
        if window_reached:
            self.parent.total_elapsed_time += self.parent.window_duration
            self.parent.data_processor.reset_plot_data()

    def autoscale_plots(self):
        """Autoscale the y-axis of all plots."""
        with self.parent.plot_data_lock:
            # EMG Plots
            for sensor_label, data in self.parent.plot_data_emg.items():
                if data:
                    y = np.array(data)
                    y_min = y.min()
                    y_max = y.max()
                    self.emg_plots[sensor_label].setYRange(y_min, y_max)
            # ACC Plots
            for sensor_label, data_dict in self.parent.plot_data_acc.items():
                data = []
                for axis in ['X', 'Y', 'Z']:
                    data.extend(data_dict[axis])
                if data:
                    y = np.array(data)
                    y_min = y.min()
                    y_max = y.max()
                    self.acc_plots[sensor_label].setYRange(y_min, y_max)
            # GYRO Plots
            for sensor_label, data_dict in self.parent.plot_data_gyro.items():
                data = []
                for axis in ['X', 'Y', 'Z']:
                    data.extend(data_dict[axis])
                if data:
                    y = np.array(data)
                    y_min = y.min()
                    y_max = y.max()
                    self.gyro_plots[sensor_label].setYRange(y_min, y_max)
            # Orientation Plots
            for sensor_label, data_dict in self.parent.plot_data_or.items():
                data = []
                for axis in ['W', 'X', 'Y', 'Z']:
                    data.extend(data_dict[axis])
                if data:
                    y = np.array(data)
                    y_min = y.min()
                    y_max = y.max()
                    self.or_plots[sensor_label].setYRange(y_min, y_max)

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
        miao = 0
        # Concatenate ACC, GYRO, and OR data for each sensor
        for sensor_label in sensor_labels:
            # ACC data (X, Y, Z)
            if miao == 0:
                miao = 1
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
 
        print(f"Feature vector: {feature_vector}")

        predicted_phase = self.parent.current_model.predict([feature_vector])[0]

        print(f"Predicted phase: {predicted_phase}")

        return predicted_phase

