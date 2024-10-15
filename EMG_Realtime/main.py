import sys
import time
import threading
import numpy as np
import os
import datetime
import queue
import scipy as sp
from scipy.signal import find_peaks, argrelextrema
import pandas as pd

from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg

from AeroPy.TrignoBase import TrignoBase
from AeroPy.DataManager import DataKernel


class EMGDataCollector(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.base = TrignoBase(self)
        self.data_handler = DataKernel(self.base)
        self.collection_data_handler = self

        # Variables for data processing
        self.analysed_segments = 0
        self.segment_start_idx = 0
        self.segment_end_idx = 0
        self.count_peak = 0
        self.peak = False

        # Variables to calculate reference scores
        self.unassisted = False
        self.unassisted_mean = 0

        # Connect to the server
        # self.data_handler.connect_to_server()

        # Initialize attributes expected by TrignoBase
        self.EMGplot = None
        self.streamYTData = False
        self.pauseFlag = True
        self.outData = []
        self.pair_number = 0
        self.complete_emg_data = {}  # Key: sensor label, value: data list
        self.complete_acc_data = {}  # Key: sensor label, value: dict with 'X', 'Y', 'Z' data lists
        self.complete_gyro_data = {}  # Similar structure
        self.is_collecting = False
        self.sensor_labels = {}
        self.sensor_names = {}

        # Initialize thread-safe queue for inter-thread communication
        self.data_queue = queue.Queue()

        # Initialize plotting variables
        self.plot_data_emg = {}  # Key: sensor label, value: data list
        self.plot_data_acc = {}  # Key: sensor label, value: dict with 'X', 'Y', 'Z' data lists
        self.plot_data_gyro = {}  # Similar structure
        self.window_duration = 5  # Duration in seconds
        self.emg_window_sizes = {}  # Key: sensor label, value: window size

        self.total_elapsed_time = 0  # Initialize total elapsed time

        # Initialize trial variables
        self.current_date = datetime.datetime.now().strftime('%Y%m%d')
        self.subject_number = ''
        self.data_directory = "Data"
        self.trial_number = 1

        # Ensure data directory exists
        if not os.path.exists(self.data_directory):
            os.makedirs(self.data_directory)

        # Lock for synchronizing access to plot_data
        self.plot_data_lock = threading.Lock()

        # Set up the GUI
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('EMG Data Collection')
        self.resize(1200, 800)

        # Central widget
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        # Layouts
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        button_layout = QtWidgets.QHBoxLayout()

        # Time display label
        self.time_label = QtWidgets.QLabel('Elapsed Time: 0.00 s')
        self.time_label.hide()  # Hide the time_label initially
        main_layout.addWidget(self.time_label)

        # Buttons
        self.start_unassisted_button = QtWidgets.QPushButton('Start Unassisted Trial')
        self.start_button = QtWidgets.QPushButton('Start Trial')
        self.stop_button = QtWidgets.QPushButton('Stop Trial')
        self.autoscale_button = QtWidgets.QPushButton('Autoscale')
        self.quit_button = QtWidgets.QPushButton('Quit')

        self.start_unassisted_button.clicked.connect(self.start_unassisted_trial)
        self.start_button.clicked.connect(self.start_trial)
        self.stop_button.clicked.connect(self.stop_trial)
        self.autoscale_button.clicked.connect(self.autoscale_plots)
        self.quit_button.clicked.connect(self.on_quit)

        button_layout.addWidget(self.start_unassisted_button)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.autoscale_button)
        button_layout.addWidget(self.quit_button)

        # PyQtGraph Plot Widget
        self.plot_widget = QtWidgets.QWidget()
        main_layout.addWidget(self.plot_widget)
        main_layout.addLayout(button_layout)

        self.show()

    def connect_base(self):
        print("Connecting to Trigno base...")
        self.base.Connect_Callback()
        print("Base connected.")

    def scan_and_pair_sensors(self):
        while True:
            self.base.Scan_Callback()
            user_input = input("Press y/n to continue scanning or stop: ").strip().lower()
            if user_input == 'y':
                continue  # Continue scanning
            elif user_input == 'n':
                break  # Stop scanning
            else:
                print("Invalid input. Please enter 'y', 'n'.")

        rename_input = input("Rename sensors? (y/n): ").strip().lower()
        if rename_input == 'y':
            for sensor in self.base.all_scanned_sensors:
                label = sensor.PairNumber
                new_name = input(f"Type name for sensor '{label}': ").strip()
                self.sensor_names[label] = new_name
        else:
            for sensor in self.base.all_scanned_sensors:
                label = sensor.PairNumber
                self.sensor_names[label] = f"Sensor {label}"
        print("Sensor renaming complete.")

        # Build a mapping from sensor label to index
        self.sensor_label_to_index = {}
        for idx, sensor in enumerate(self.base.all_scanned_sensors):
            label = sensor.PairNumber
            self.sensor_label_to_index[label] = idx

        # Mode selection for each sensor
        default_mode = None  # Set your default mode here as a string, or keep None to prompt
        if default_mode is None:
            for label, sensor_index in self.sensor_label_to_index.items():
                modes = self.base.getSampleModes(sensor_index)
                print(f"Available modes for sensor {label}:")
                for idx, mode in enumerate(modes):
                    print(f"{idx}: {mode}")
                mode_index = int(input(f"Select mode index for sensor {label}: "))
                selected_mode = modes[mode_index]
                self.base.setSampleMode(sensor_index, selected_mode)
                print(f"Set mode '{selected_mode}' for sensor {label}")
        else:
            for label, sensor_index in self.sensor_label_to_index.items():
                self.base.setSampleMode(sensor_index, default_mode)
                print(f"Set mode '{default_mode}' for sensor {label}")

        # Get subject number
        self.subject_number = input("Enter subject number: ").strip()

        # Determine the next trial number
        subject_folder = os.path.join(self.data_directory, f"Subject {self.subject_number}")
        if os.path.exists(subject_folder):
            existing_trials = [fname for fname in os.listdir(subject_folder) if 'Trial' in fname]
            trial_numbers = [int(fname.split('Trial ')[-1].split('.')[0]) for fname in existing_trials if 'Trial' in fname]
            if trial_numbers:
                self.trial_number = max(trial_numbers) + 1
            else:
                self.trial_number = 1
        else:
            os.makedirs(subject_folder)
            self.trial_number = 1

        # Show the time label once the subject number is provided
        self.time_label.show()

        # Configure sensors and initialize plot
        self.configure_sensors()
        self.initialize_plot()

    def configure_sensors(self):
        print("Configuring sensors...")
        self.base.start_trigger = False
        self.base.stop_trigger = False
        configured = self.base.ConfigureCollectionOutput()
        if not configured:
            print("Failed to configure sensors.")
            return

        # Initialize data structures
        self.complete_emg_data = {}  # Key: sensor label, value: data list
        self.plot_data_emg = {}      # Key: sensor label, value: data list
        self.emg_sampling_frequencies = {}  # Key: sensor label, value: sampling frequency

        self.complete_acc_data = {}  # Key: sensor label, value: dict with 'X', 'Y', 'Z' data lists
        self.plot_data_acc = {}
        self.acc_sample_rates = {}   # Key: sensor label, value: sampling rate

        self.complete_gyro_data = {}
        self.plot_data_gyro = {}
        self.gyro_sample_rates = {}  # Key: sensor label, value: sampling rate

        # Build a mapping from data channel indices to sensor labels
        self.channel_index_to_label = {}
        self.acc_channels_per_sensor = {}
        self.gyro_channels_per_sensor = {}

        # EMG channels
        for idx, channel_idx in enumerate(self.base.emgChannelsIdx):
            sensor_label = self.base.emgChannelSensors[idx]  # Access sensor label for EMG channel
            if sensor_label not in self.complete_emg_data:
                self.complete_emg_data[sensor_label] = []
                self.plot_data_emg[sensor_label] = []
                self.emg_sampling_frequencies[sensor_label] = self.base.emgSampleRates[sensor_label]
                self.emg_window_sizes[sensor_label] = int(self.window_duration * self.emg_sampling_frequencies[sensor_label])

        # ACC and GYRO sample rates per sensor
        for sensor_label in self.base.accSampleRates:
            self.acc_sample_rates[sensor_label] = self.base.accSampleRates[sensor_label]
        for sensor_label in self.base.gyroSampleRates:
            self.gyro_sample_rates[sensor_label] = self.base.gyroSampleRates[sensor_label]

        for label in self.base.sensor_label_to_channels:
            sensor_name = self.sensor_names.get(label, f"Sensor {label}")
            # EMG channels
            for ch in self.base.sensor_label_to_channels[label]['EMG']:
                idx = ch['index']
                self.channel_index_to_label[idx] = f"EMG {sensor_name} {ch['label']}"
            # ACC channels
            acc_channels = self.base.sensor_label_to_channels[label]['ACC']
            if acc_channels:
                self.acc_channels_per_sensor[label] = {'indices': [], 'labels': []}
                self.plot_data_acc[label] = {'X': [], 'Y': [], 'Z': []}
                self.complete_acc_data[label] = {'X': [], 'Y': [], 'Z': []}
                for ch in acc_channels:
                    idx = ch['index']
                    ch_label = ch['label']
                    self.acc_channels_per_sensor[label]['indices'].append(idx)
                    self.acc_channels_per_sensor[label]['labels'].append(ch_label)
            # GYRO channels
            gyro_channels = self.base.sensor_label_to_channels[label]['GYRO']
            if gyro_channels:
                self.gyro_channels_per_sensor[label] = {'indices': [], 'labels': []}
                self.plot_data_gyro[label] = {'X': [], 'Y': [], 'Z': []}
                self.complete_gyro_data[label] = {'X': [], 'Y': [], 'Z': []}
                for ch in gyro_channels:
                    idx = ch['index']
                    ch_label = ch['label']
                    self.gyro_channels_per_sensor[label]['indices'].append(idx)
                    self.gyro_channels_per_sensor[label]['labels'].append(ch_label)

        print("Sensors configured.")

    def initialize_plot(self):
        """Initialize the plot for EMG, ACC, and GYRO data streaming using PyQtGraph."""
        # Set window title
        self.setWindowTitle(f'Subject {self.subject_number} Trial {self.trial_number}')

        # Number of sensors
        sensor_labels = list(self.sensor_names.keys())
        total_rows = len(sensor_labels)

        # Create a grid layout for plots
        self.plot_layout = QtWidgets.QGridLayout(self.plot_widget)

        self.emg_plots = {}
        self.acc_plots = {}
        self.gyro_plots = {}

        for i, sensor_label in enumerate(sensor_labels):
            sensor_name = self.sensor_names.get(sensor_label, f"Sensor {sensor_label}")

            # EMG plot
            if sensor_label in self.plot_data_emg:
                pw_emg = pg.PlotWidget(title=f'EMG {sensor_name}')
                pw_emg.setLabel('left', 'Amplitude', units='V')
                pw_emg.setLabel('bottom', 'Time', units='s')
                pw_emg.showGrid(x=True, y=True)
                self.plot_layout.addWidget(pw_emg, i, 0)
                self.emg_plots[sensor_label] = pw_emg
            else:
                self.plot_layout.addWidget(QtWidgets.QWidget(), i, 0)

            # ACC plot
            if sensor_label in self.acc_channels_per_sensor:
                pw_acc = pg.PlotWidget(title=f'ACC {sensor_name}')
                pw_acc.setLabel('left', 'Acceleration', units='g')
                pw_acc.setLabel('bottom', 'Time', units='s')
                pw_acc.showGrid(x=True, y=True)
                self.plot_layout.addWidget(pw_acc, i, 1)
                self.acc_plots[sensor_label] = pw_acc
            else:
                self.plot_layout.addWidget(QtWidgets.QWidget(), i, 1)

            # GYRO plot
            if sensor_label in self.gyro_channels_per_sensor:
                pw_gyro = pg.PlotWidget(title=f'GYRO {sensor_name}')
                pw_gyro.setLabel('left', 'Angular Velocity', units='dps')
                pw_gyro.setLabel('bottom', 'Time', units='s')
                pw_gyro.showGrid(x=True, y=True)
                self.plot_layout.addWidget(pw_gyro, i, 2)
                self.gyro_plots[sensor_label] = pw_gyro
            else:
                self.plot_layout.addWidget(QtWidgets.QWidget(), i, 2)

        # Start a timer to update the plot periodically
        self.plot_timer = QtCore.QTimer()
        self.plot_timer.timeout.connect(self.update_plot)
        self.plot_timer.start(50)  # Update plot every 50 ms (20 FPS)

    def reset_plot_data(self):
        """Reset plot data to start over."""
        with self.plot_data_lock:
            for sensor_label in self.plot_data_emg:
                self.plot_data_emg[sensor_label] = []
            for sensor_label in self.plot_data_acc:
                self.plot_data_acc[sensor_label] = {'X': [], 'Y': [], 'Z': []}
            for sensor_label in self.plot_data_gyro:
                self.plot_data_gyro[sensor_label] = {'X': [], 'Y': [], 'Z': []}

    def autoscale_plots(self):
        """Autoscale the y-axis of all plots."""
        with self.plot_data_lock:
            # EMG Plots
            for sensor_label, data in self.plot_data_emg.items():
                if data:
                    y = np.array(data)
                    y_min = y.min()
                    y_max = y.max()
                    self.emg_plots[sensor_label].setYRange(y_min, y_max)
            # ACC Plots
            for sensor_label, data_dict in self.plot_data_acc.items():
                data = []
                for axis in ['X', 'Y', 'Z']:
                    data.extend(data_dict[axis])
                if data:
                    y = np.array(data)
                    y_min = y.min()
                    y_max = y.max()
                    self.acc_plots[sensor_label].setYRange(y_min, y_max)
            # GYRO Plots
            for sensor_label, data_dict in self.plot_data_gyro.items():
                data = []
                for axis in ['X', 'Y', 'Z']:
                    data.extend(data_dict[axis])
                if data:
                    y = np.array(data)
                    y_min = y.min()
                    y_max = y.max()
                    self.gyro_plots[sensor_label].setYRange(y_min, y_max)

    def update_plot(self):
        """Update the plot with new data."""
        with self.plot_data_lock:
            plot_data_emg_copy = self.plot_data_emg.copy()
            plot_data_acc_copy = {k: v.copy() for k, v in self.plot_data_acc.items()}
            plot_data_gyro_copy = {k: v.copy() for k, v in self.plot_data_gyro.items()}

        # Calculate elapsed time based on EMG data
        elapsed_times = []
        for sensor_label, data in plot_data_emg_copy.items():
            if data:
                sample_rate = self.emg_sampling_frequencies[sensor_label]
                num_samples = len(data)
                elapsed_time = num_samples / sample_rate
                elapsed_times.append(elapsed_time)
        if elapsed_times:
            elapsed_time = max(elapsed_times)
        else:
            elapsed_time = 0

        self.time_label.setText(f'Elapsed Time: {self.total_elapsed_time + elapsed_time:.2f} s')  # Display cumulative elapsed time

        # EMG Plots
        for sensor_label, data in plot_data_emg_copy.items():
            if len(data) == 0:
                continue  # Skip if no data
            y = np.array(data)
            sample_rate = self.emg_sampling_frequencies[sensor_label]
            time_array = np.arange(len(y)) / sample_rate + self.total_elapsed_time
            self.emg_plots[sensor_label].plot(time_array, y, clear=True, pen='g')
            self.emg_plots[sensor_label].setXRange(self.total_elapsed_time, self.total_elapsed_time + self.window_duration)

        # ACC Plots
        for sensor_label, data_dict in plot_data_acc_copy.items():
            pw_acc = self.acc_plots[sensor_label]
            pw_acc.clear()
            sample_rate = self.acc_sample_rates[sensor_label]
            for axis in ['X', 'Y', 'Z']:
                data = data_dict[axis]
                if len(data) == 0:
                    continue
                y = np.array(data)
                time_array = np.arange(len(y)) / sample_rate + self.total_elapsed_time
                pw_acc.plot(time_array, y, pen=pg.mkPen({'X': 'r', 'Y': 'g', 'Z': 'b'}[axis]), name=axis)
            pw_acc.setXRange(self.total_elapsed_time, self.total_elapsed_time + self.window_duration)
            pw_acc.addLegend()

        # GYRO Plots
        for sensor_label, data_dict in plot_data_gyro_copy.items():
            pw_gyro = self.gyro_plots[sensor_label]
            pw_gyro.clear()
            sample_rate = self.gyro_sample_rates[sensor_label]
            for axis in ['X', 'Y', 'Z']:
                data = data_dict[axis]
                if len(data) == 0:
                    continue
                y = np.array(data)
                time_array = np.arange(len(y)) / sample_rate + self.total_elapsed_time
                pw_gyro.plot(time_array, y, pen=pg.mkPen({'X': 'r', 'Y': 'g', 'Z': 'b'}[axis]), name=axis)
            pw_gyro.setXRange(self.total_elapsed_time, self.total_elapsed_time + self.window_duration)
            pw_gyro.addLegend()

        # Check if it reached the window size, increment the cumulative time
        window_reached = any(len(data) >= self.emg_window_sizes[sensor_label] for sensor_label, data in self.plot_data_emg.items())
        if window_reached:
            self.total_elapsed_time += self.window_duration  # Increment total elapsed time by the window duration
            self.reset_plot_data()

    def start_trial(self):
        self.unassisted = False
        if not self.is_collecting:
            print(f"Starting Trial {self.trial_number}...")
            # Reset data structures
            self.reset_plot_data()
            for sensor_label in self.complete_emg_data:
                self.complete_emg_data[sensor_label] = []
            for sensor_label in self.complete_acc_data:
                self.complete_acc_data[sensor_label] = {'X': [], 'Y': [], 'Z': []}
            for sensor_label in self.complete_gyro_data:
                self.complete_gyro_data[sensor_label] = {'X': [], 'Y': [], 'Z': []}
            self.data_queue = queue.Queue()
            self.is_collecting = True
            self.pauseFlag = False  # Ensure pauseFlag is False
            self.total_elapsed_time = 0  # Reset total elapsed time
            self.start_collection()
        else:
            print("A trial is already running.")

    def start_unassisted_trial(self):
        self.unassisted = True
        if not self.is_collecting:
            print(f"Starting Trial {self.trial_number}...")
            # Reset data structures
            self.reset_plot_data()
            for sensor_label in self.complete_emg_data:
                self.complete_emg_data[sensor_label] = []
            for sensor_label in self.complete_acc_data:
                self.complete_acc_data[sensor_label] = {'X': [], 'Y': [], 'Z': []}
            for sensor_label in self.complete_gyro_data:
                self.complete_gyro_data[sensor_label] = {'X': [], 'Y': [], 'Z': []}
            self.data_queue = queue.Queue()
            self.is_collecting = True
            self.pauseFlag = False  # Ensure pauseFlag is False
            self.total_elapsed_time = 0  # Reset total elapsed time
            self.start_collection()
        else:
            print("A trial is already running.")

    def stop_trial(self):
        if self.is_collecting:
            print("Stopping trial...")
            self.stop_collection()
            time.sleep(10)  # Wait for last batch data

            filename_emg = f"{self.current_date} EMG Trial {self.trial_number}.csv"
            filename_acc = f"{self.current_date} ACC Trial {self.trial_number}.csv"
            filename_gyro = f"{self.current_date} GYRO Trial {self.trial_number}.csv"
            subject_folder = os.path.join(self.data_directory, f"Subject {self.subject_number}")
            if not os.path.exists(subject_folder):
                os.makedirs(subject_folder)
            filepath_emg = os.path.join(subject_folder, filename_emg)
            filepath_acc = os.path.join(subject_folder, filename_acc)
            filepath_gyro = os.path.join(subject_folder, filename_gyro)

            # Export collected data
            self.export_data_to_csv(filepath_emg, filepath_acc, filepath_gyro)
            print(f"Trial {self.trial_number} data saved as: {filename_emg}, {filename_acc}, and {filename_gyro}")
            self.trial_number += 1

        else:
            print("No trial is currently running.")

    def start_collection(self):
        print("Starting data collection...")
        
        # Connect to the server here
        # try:
        #     self.data_handler.connect_to_server()
        #     print("Connected to the server.")
        # except Exception as e:
        #     print(f"Failed to connect to the server: {e}")
        #     return  # Exit if the connection fails

        self.base.Start_Callback(start_trigger=False, stop_trigger=False)
        self.threadManager(start_trigger=False, stop_trigger=False)

    def stop_collection(self):
        print("Stopping data collection...")
        self.pauseFlag = True
        self.base.Stop_Callback()
        
        # Wait for threads to finish
        time.sleep(3)
        if hasattr(self, "streaming_thread"):
            self.streaming_thread.join()
        if hasattr(self, "processing_thread"):
            self.processing_thread.join()

        print("Data collection stopped.")
        self.is_collecting = False  # Reset collecting flag

        # Close the connection when stopping collection
        # self.data_handler.close_connection()

    def threadManager(self, start_trigger, stop_trigger):
        self.start_trigger = start_trigger
        self.stop_trigger = stop_trigger

        # Start new threads for streaming and processing data
        self.streaming_thread = threading.Thread(target=self.stream_data)
        self.processing_thread = threading.Thread(target=self.process_data)

        self.streaming_thread.start()
        self.processing_thread.start()

    def extract_time(self, df):
        """
        Detect the start and end of a sit-to-stand motion based on angular acceleration in z-axis.
        
        Parameters:
        - df: pandas.DataFrame with the data
        Returns:
        start_time (float): Time when the motion starts.
        end_time (float): Time when the motion ends.
        """
        # First derivative (rate of change) to detect where the acceleration starts decreasing
        acc_z_diff = np.diff(df['ACC Z (G)'])

        # Find all acceleration-change extremas
        maxima = argrelextrema(acc_z_diff, np.greater)[0]

        # Find the global minimum
        minimum = np.argmin(df['ACC Z (G)'])

        # Get the two maxima closest to the global minimum
        maxima.sort()
        start_idx = maxima[np.searchsorted(maxima, minimum) - 1]
        end_idx = maxima[np.searchsorted(maxima, minimum)]

        return start_idx, end_idx



    def apply_lowpass_filter(self, df: pd.DataFrame, cutoff_freq: float, sampling_freq: float, filter_order=4) -> pd.DataFrame:
        """
        Apply a low-pass filter to the first 7 columns of the given DataFrame.

        Parameters:
        - df: pandas.DataFrame with the data to be filtered.
        - cutoff_freq: The cutoff frequency for the low-pass filter (in Hz).
        - sampling_freq: The sampling frequency of the data (in Hz).
        - filter_order: The order of the Butterworth filter (default is 4).

        Returns:
        - df_filtered: DataFrame with the low-pass filtered values.
        """
        # Normalize the cutoff frequency with respect to Nyquist frequency
        nyquist_freq = sampling_freq / 2
        normalized_cutoff = cutoff_freq / nyquist_freq

        # Design a Butterworth low-pass filter
        b, a = sp.signal.butter(filter_order, normalized_cutoff, btype='low')

        # Create a copy of the DataFrame to hold the filtered data
        df_filtered = df.copy()

        # Apply the filter to data
        for col in df.columns:
            df_filtered[col] = sp.signal.filtfilt(b, a, df[col])

        return df_filtered
    
    def filter_emg(unfiltered_df: pd.DataFrame, low_pass=4, sfreq=1259.2593, high_band=20, low_band=450) -> pd.DataFrame:
        """ Filter EMG signals


        Args:
            unfiltered_df (pd.DataFrame): DataFrame containing the EMG data and time
            low_pass (int, optional): Low-pass cut off frequency. Defaults to 4.
            sfreq (int, optional): Sampling frequency. Defaults to 1259.2593.
            high_band (int, optional): High-band frequency for bandpass filter. Defaults to 20.
            low_band (int, optional): Low-band frequency for bandpass filter. Defaults to 450.


        Returns:
            pd.DataFrame: filtered dataframe
        """
        emg_data = unfiltered_df.copy()

        # Normalize cut-off frequencies to sampling frequency
        high_band_normalized = high_band / (sfreq / 2)
        low_band_normalized = low_band / (sfreq / 2)
        low_pass_normalized = low_pass / (sfreq / 2)

        # Bandpass filter coefficients
        b1, a1 = sp.signal.butter(4, [high_band_normalized, low_band_normalized], btype='bandpass')

        # Lowpass filter coefficients
        b2, a2 = sp.signal.butter(4, low_pass_normalized, btype='lowpass')

        def process_emg(emg):
            # Handle NaNs: skip filtering for NaN segments
            if emg.isna().all():
                return emg  # Returns as is if all are NaNs

            # Correct mean for non-NaN values
            non_nan_emg = emg.dropna()
            emg_correctmean = non_nan_emg - non_nan_emg.mean()

            # Filter EMG: bandpass, rectify, lowpass for non-NaN values
            emg_filtered = sp.signal.filtfilt(b1, a1, emg_correctmean)
            emg_rectified = np.abs(emg_filtered)
            emg_envelope = sp.signal.filtfilt(b2, a2, emg_rectified)

            # Construct the resulting series, placing NaNs back in their original positions
            result = pd.Series(index=emg.index, data=np.nan)
            result[emg.notna()] = emg_envelope

            return result

        # Apply processing to each column
        envelopes = emg_data.apply(process_emg, axis=0)
        env_freq = int(low_pass_normalized * sfreq)

        return envelopes, env_freq


    def calculate_stuff(self):
        imu_sensor_label = 0
        gyro_axis = 'X'
        acc_axis = 'Z'

        # Hyperparameters
        peak_threshold = 50

        relevant_emg = []

        if len(self.complete_gyro_data[imu_sensor_label][gyro_axis]) > 100:
            # Get current Gyro X value
            # gyro_x = self.complete_gyro_data[sensor_label][gyro_axis][-1]

            # Get the mean of the past 10 Gyro X values
            gyro_x = np.mean(self.complete_gyro_data[imu_sensor_label][gyro_axis][-100:])

            if gyro_x > peak_threshold and not self.peak:
                self.peak = True
                self.segment_end_idx = len(self.complete_gyro_data[imu_sensor_label][gyro_axis])
                self.count_peak += 1

            if self.count_peak > self.analysed_segments:
                self.analysed_segments += 1
                z_acc_segment = self.complete_acc_data[imu_sensor_label][acc_axis][self.segment_start_idx:self.segment_end_idx]
                # Convert to DataFrame
                z_acc_segment = pd.DataFrame(z_acc_segment, columns=['ACC Z (G)'])
                z_acc_segment_filtered = self.apply_lowpass_filter(z_acc_segment, 1, 148.1481)

                # Extract time (start,end)
                start_time, end_time = self.extract_time(z_acc_segment_filtered)

                # Extract relevant_emg data
                for sensor_label in self.complete_emg_data:
                    if relevant_emg:
                        relevant_emg.append(self.complete_emg_data[sensor_label][start_time:end_time])
                    else:
                        relevant_emg = self.complete_emg_data[sensor_label][start_time:end_time]

                # Filter relevant_emg data
                relevant_emg_filtered, env_freq = self.filter_emg(relevant_emg)

                # Get score
                # Calculated reference score if unassisted
                if self.unassisted:
                    self.unassisted_mean = np.mean(relevant_emg)

                # Compare score of assisted vs unassisted

                # Send comparison score to Raspi

            if self.peak:
                if gyro_x < 50:
                    self.peak = False
                    self.segment_start_idx = len(self.complete_gyro_data[imu_sensor_label][gyro_axis])





    def stream_data(self):
        """Collect data from sensors and put it into the queue."""
        while not self.pauseFlag:
            self.data_handler.processData(self.data_queue)
            self.calculate_stuff()

        # Process any remaining data after stopping
        self.process_remaining_data()

    def process_remaining_data(self):
        """Process any remaining data in the queue after stopping the collection."""
        while not self.data_queue.empty():
            try:
                data_batch = self.data_queue.get(timeout=1)
                self.process_data_batch(data_batch)
                self.data_queue.task_done()
            except queue.Empty:
                break

    def process_data(self):
        """Process data from the queue."""
        while not self.pauseFlag or not self.data_queue.empty():
            try:
                data_batch = self.data_queue.get(timeout=1)
                self.process_data_batch(data_batch)
                self.data_queue.task_done()
            except queue.Empty:
                continue

    def process_data_batch(self, data_batch):
        """Process a single data batch."""
        if not isinstance(data_batch, list) or not data_batch:
            print("Received empty or invalid data_batch")
            return

        # Update plot data buffer and collect data for exporting
        with self.plot_data_lock:
            # EMG Data
            for idx, channel_idx in enumerate(self.base.emgChannelsIdx):
                sensor_label = self.base.emgChannelSensors[idx]
                if channel_idx < len(data_batch):
                    data = data_batch[channel_idx]
                    self.plot_data_emg[sensor_label].extend(data)
                    self.complete_emg_data[sensor_label].extend(data)
                else:
                    print(f"EMG Channel index {channel_idx} out of range in data_batch.")

            # ACC Data
            for sensor_label, sensor_info in self.acc_channels_per_sensor.items():
                indices = sensor_info['indices']
                labels = sensor_info['labels']
                for idx, ch_label in zip(indices, labels):
                    axis = ch_label[-1]  # Assuming labels end with 'X', 'Y', 'Z'
                    if idx < len(data_batch):
                        data = data_batch[idx]
                        self.plot_data_acc[sensor_label][axis].extend(data)
                        self.complete_acc_data[sensor_label][axis].extend(data)

                        #print("\n Acceleration_data")
                        #print(data)
                    else:
                        print(f"ACC Channel index {idx} out of range in data_batch.")

            # GYRO Data
            for sensor_label, sensor_info in self.gyro_channels_per_sensor.items():
                indices = sensor_info['indices']
                labels = sensor_info['labels']
                for idx, ch_label in zip(indices, labels):
                    axis = ch_label[-1]  # Assuming labels end with 'X', 'Y', 'Z'
                    if idx < len(data_batch):
                        data = data_batch[idx]
                        self.plot_data_gyro[sensor_label][axis].extend(data)
                        self.complete_gyro_data[sensor_label][axis].extend(data)

                        #print("\n Gyro_data")
                        #print(data)

                    else:
                        print(f"GYRO Channel index {idx} out of range in data_batch.")


    def export_data_to_csv(self, filename_emg="EMG_data.csv", filename_acc="ACC_data.csv", filename_gyro="GYRO_data.csv"):
        print("Exporting collected data...")

        # Export EMG data
        with open(filename_emg, 'w') as f_emg:
            headers = []
            for sensor_label in self.complete_emg_data:
                headers.append(f'EMG {self.sensor_names.get(sensor_label, sensor_label)}')
            f_emg.write(','.join(headers) + '\n')

            max_length = max(len(data) for data in self.complete_emg_data.values()) if self.complete_emg_data else 0
            for row_idx in range(max_length):
                row = []
                for data in self.complete_emg_data.values():
                    if row_idx < len(data):
                        row.append(str(data[row_idx]))
                    else:
                        row.append("")
                f_emg.write(",".join(row) + '\n')

        # Export ACC data
        with open(filename_acc, 'w') as f_acc:
            # Build headers
            headers = []
            for sensor_label in self.complete_acc_data:
                for axis in ['X', 'Y', 'Z']:
                    headers.append(f'ACC {self.sensor_names.get(sensor_label, sensor_label)} {axis}')
            f_acc.write(','.join(headers) + '\n')

            # Find max length
            max_length = 0
            for sensor_data in self.complete_acc_data.values():
                max_length = max(max_length, max(len(axis_data) for axis_data in sensor_data.values()))
            # Write data
            for row_idx in range(max_length):
                row = []
                for sensor_data in self.complete_acc_data.values():
                    for axis in ['X', 'Y', 'Z']:
                        data = sensor_data[axis]
                        if row_idx < len(data):
                            row.append(str(data[row_idx]))
                        else:
                            row.append("")
                f_acc.write(",".join(row) + '\n')

        # Export GYRO data
        with open(filename_gyro, 'w') as f_gyro:
            # Build headers
            headers = []
            for sensor_label in self.complete_gyro_data:
                for axis in ['X', 'Y', 'Z']:
                    headers.append(f'GYRO {self.sensor_names.get(sensor_label, sensor_label)} {axis}')
            f_gyro.write(','.join(headers) + '\n')

            # Find max length
            max_length = 0
            for sensor_data in self.complete_gyro_data.values():
                max_length = max(max_length, max(len(axis_data) for axis_data in sensor_data.values()))
            # Write data
            for row_idx in range(max_length):
                row = []
                for sensor_data in self.complete_gyro_data.values():
                    for axis in ['X', 'Y', 'Z']:
                        data = sensor_data[axis]
                        if row_idx < len(data):
                            row.append(str(data[row_idx]))
                        else:
                            row.append("")
                f_gyro.write(",".join(row) + '\n')

    def on_quit(self):
        print("Quitting application.")
        self.stop_collection()
        self.close()

if __name__ == "__main__":
    appQt = QtWidgets.QApplication(sys.argv)
    collector = EMGDataCollector()
    collector.connect_base()
    collector.scan_and_pair_sensors()
    sys.exit(appQt.exec_())
