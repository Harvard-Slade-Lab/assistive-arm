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

from PyQt5 import QtWidgets

from AeroPy.TrignoBase import TrignoBase
from AeroPy.DataManager import DataKernel

from UI.UISetup import UISetup
from Plotting.Plotter import Plotter
from DataProcessing.DataProcessor import DataProcessor
from DataExport.DataExporter import DataExporter

class EMGDataCollector(QtWidgets.QMainWindow):
    def __init__(self, window_duration=5, data_directory="Data"):
        super().__init__()
        self.window_duration = window_duration
        self.data_directory = data_directory

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
        self.complete_emg_data = {}
        self.complete_acc_data = {}
        self.complete_gyro_data = {}
        self.is_collecting = False
        self.sensor_labels = {}
        self.sensor_names = {}

        # Initialize thread-safe queue for inter-thread communication
        self.data_queue = queue.Queue()

        # Initialize plotting variables
        self.plot_data_emg = {}
        self.plot_data_acc = {}
        self.plot_data_gyro = {}
        self.emg_window_sizes = {}

        self.total_elapsed_time = 0

        # Initialize trial variables
        self.current_date = datetime.datetime.now().strftime('%Y%m%d')
        self.subject_number = ''
        self.trial_number = 1

        # Ensure data directory exists
        if not os.path.exists(self.data_directory):
            os.makedirs(self.data_directory)

        # Lock for synchronizing access to plot_data
        self.plot_data_lock = threading.Lock()

        # **Initialize the Plotter BEFORE setting up the UI**
        self.plotter = Plotter(self)

        # Set up the UI
        self.ui = UISetup(self)
        self.ui.init_ui()

        # Set up the DataProcessor
        self.data_processor = DataProcessor(self)

        # Set up the DataExporter
        self.data_exporter = DataExporter(self)


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
        self.plotter.initialize_plot()

    def configure_sensors(self):
        print("Configuring sensors...")
        self.base.start_trigger = False
        self.base.stop_trigger = False
        configured = self.base.ConfigureCollectionOutput()
        if not configured:
            print("Failed to configure sensors.")
            return

        # Initialize data structures
        self.complete_emg_data = {}
        self.plot_data_emg = {}
        self.emg_sampling_frequencies = {}
        self.emg_window_sizes = {}

        self.complete_acc_data = {}
        self.plot_data_acc = {}
        self.acc_sample_rates = {}

        self.complete_gyro_data = {}
        self.plot_data_gyro = {}
        self.gyro_sample_rates = {}

        # Build a mapping from data channel indices to sensor labels
        self.channel_index_to_label = {}
        self.acc_channels_per_sensor = {}
        self.gyro_channels_per_sensor = {}

        # EMG channels
        for idx, channel_idx in enumerate(self.base.emgChannelsIdx):
            sensor_label = self.base.emgChannelSensors[idx]
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

    def start_trial(self):
        self.unassisted = False
        if not self.is_collecting:
            print(f"Starting Trial {self.trial_number}...")
            # Reset data structures
            self.data_processor.reset_plot_data()
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
            self.data_exporter.export_data_to_csv(filepath_emg, filepath_acc, filepath_gyro)
            print(f"Trial {self.trial_number} data saved as: {filename_emg}, {filename_acc}, and {filename_gyro}")
            self.trial_number += 1

        else:
            print("No trial is currently running.")

    def start_collection(self):
        print("Starting data collection...")
        # Connect to the server (socket connection)
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

    def threadManager(self, start_trigger, stop_trigger):
        self.start_trigger = start_trigger
        self.stop_trigger = stop_trigger

        # Start new threads for streaming and processing data
        self.streaming_thread = threading.Thread(target=self.stream_data)
        self.processing_thread = threading.Thread(target=self.data_processor.process_data)

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
        self.data_processor.process_remaining_data()

    def on_quit(self):
        print("Quitting application.")
        self.stop_collection()
        self.close()
