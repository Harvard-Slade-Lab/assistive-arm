import sys
import time
import threading
import numpy as np
import os
import datetime
import queue
import re
import scipy as sp
from scipy.signal import find_peaks, argrelextrema
import pandas as pd

from PyQt5 import QtWidgets

from AeroPy.TrignoBase import TrignoBase
from AeroPy.DataManager import DataKernel

from UI.UISetup import UISetup
from UI.NoPlotUISetup import NoPlotUISetup
from Plotting.Plotter import Plotter
from DataProcessing.DataProcessor import DataProcessor
from DataExport.DataExporter import DataExporter

class EMGDataCollector(QtWidgets.QMainWindow):
    def __init__(self, plot=False, socket=False, window_duration=5, data_directory="Data"):
        super().__init__()
        self.window_duration = window_duration
        self.data_directory = data_directory

        self.base = TrignoBase(self)
        self.data_handler = DataKernel(self.base)
        self.collection_data_handler = self

        # Variables for data processing
        self.analysed_segments = 0
        self.segment_start_idx_imu = 0
        self.segment_end_idx_imu = 0
        self.sts_start_idx_imu = 0
        self.sts_end_idx_imu = 0
        self.count_peak = 0
        self.peak = False

        self.log_entries = []

        # Variables to calculate reference scores
        self.unassisted = False
        self.unassisted_counter = 0
        self.unassisted_mean = 0

        # Flag to have socket connection or not
        self.socket = socket
        if self.socket:
            # Connect to the server
            self.data_handler.connect_to_server()

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
        self.assistive_profile_name = ''
        self.trial_number = 1

        # Ensure data directory exists
        if not os.path.exists(self.data_directory):
            os.makedirs(self.data_directory)

        # Lock for synchronizing access to plot_data
        self.plot_data_lock = threading.Lock()

        # Flag to select if plots should be shown or not
        self.plot = plot
        if self.plot:
            # **Initialize the Plotter BEFORE setting up the UI**
            self.plotter = Plotter(self)
            # Set up the UI
            self.ui = UISetup(self)
            self.ui.init_ui()
        else:
            self.ui = NoPlotUISetup(self)
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

        project_name = input("Enter project name (any key if no project defined): ").strip()

        if project_name == '0':
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

        if project_name == 'sts':
            sensor_names = {1: "IMU", 2: "RF_R", 3: "VM_R", 4: "BF_R", 5: "G_R", 6: "RF_L", 7: "VM_L", 8: "BF_L", 9: "G_L"}

            self.sensor_label_to_index = {}
            for idx, sensor in enumerate(self.base.all_scanned_sensors):
                label = sensor.PairNumber
                self.sensor_label_to_index[label] = idx

            for label, sensor_index in self.sensor_label_to_index.items():
                self.sensor_names[label] = sensor_names[sensor_index+1]
                modes = self.base.getSampleModes(sensor_index)
                if self.sensor_names[label] == 'IMU':
                    self.base.setSampleMode(sensor_index, modes[110])
                else:
                    self.base.setSampleMode(sensor_index, modes[2])

        if project_name == 'sts_2':
            sensor_names = {1: "IMU", 2: "RF_R", 3: "VM_R", 4: "BF_R", 5: "G_R", 6: "RF_L", 7: "VM_L", 8: "BF_L", 9: "G_L", 10: "GL_R", 11: "SO_R", 12: "TA_R"}

            self.sensor_label_to_index = {}
            for idx, sensor in enumerate(self.base.all_scanned_sensors):
                label = sensor.PairNumber
                self.sensor_label_to_index[label] = idx

            for label, sensor_index in self.sensor_label_to_index.items():
                self.sensor_names[label] = sensor_names[sensor_index+1]
                modes = self.base.getSampleModes(sensor_index)
                if self.sensor_names[label] == 'IMU':
                    self.base.setSampleMode(sensor_index, modes[110])
                else:
                    self.base.setSampleMode(sensor_index, modes[2])

        # Get subject number
        self.subject_number = input("Enter subject number: ").strip()
        self.assistive_profile_name = input("Enter assistive profile name: ").strip()

        # Determine the next trial number
        subject_folder = os.path.join(self.data_directory, f"subject_{self.subject_number}")
        if os.path.exists(subject_folder):
            existing_trials = [fname for fname in os.listdir(subject_folder) if 'Trial' in fname]
            trial_numbers = [int(fname.split('Trial_')[-1].split('.')[0]) for fname in existing_trials if 'Trial' in fname]
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

        if self.plot:
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
        if not self.is_collecting:
            # Set unassisted flag to false to trigger score calculation
            self.unassisted = False
            # Reset counter and mean for unassisted runs
            self.count_unassisted = 0
            self.unassisted_mean = 0

            # Send singal to socket server to start data collection
            if self.socket:
                self.data_handler.send_data("Start")

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
        if not self.is_collecting:
            # Set unassisted flag to true to trigger unassisted data reference collection
            self.unassisted = True
            # Send singal to socket server to start data collection
            if self.socket:
                self.data_handler.send_data("Start")

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

    def stop_trial(self):
        if self.is_collecting:
            # Send signal to socket server to stop data collection
            if self.socket:
                self.data_handler.send_data("Stop")

            print("Stopping trial...")
            self.stop_collection()
            time.sleep(10)  # Wait for last batch data
            filename_emg = f"{self.current_date}_EMG_Profile_{self.assistive_profile_name}_Trial_{self.trial_number}.csv"
            filename_acc = f"{self.current_date}_ACC_Profile_{self.assistive_profile_name}_Trial_{self.trial_number}.csv"
            filename_gyro = f"{self.current_date}_GYRO_Profile_{self.assistive_profile_name}_Trial_{self.trial_number}.csv"
            subject_folder = os.path.join(self.data_directory, f"subject_{self.subject_number}")
            if not os.path.exists(subject_folder):
                os.makedirs(subject_folder)
            filepath_emg = os.path.join(subject_folder, filename_emg)
            filepath_acc = os.path.join(subject_folder, filename_acc)
            filepath_gyro = os.path.join(subject_folder, filename_gyro)

            # Export collected data
            self.data_exporter.export_data_to_csv(filepath_emg, filepath_acc, filepath_gyro)
            print(f"Trial {self.trial_number} data saved as: {filename_emg}, {filename_acc}, and {filename_gyro}")

            # Reset data logger
            self.log_entries = []
            self.trial_number += 1

        assistive_profile_name, ok = QtWidgets.QInputDialog.getText(self, 'Assistive Profile Name', 'Enter new assistive profile name or cancel:')
        if ok:
            self.trial_number = 1
            self.assistive_profile_name = assistive_profile_name

        else:
            print("No trial is currently running.")

    def start_collection(self):
        print("Starting data collection...")
        self.base.Start_Callback(start_trigger=False, stop_trigger=False)
        self.threadManager(start_trigger=False, stop_trigger=False)

    def stop_collection(self):
        print("Stopping data collection1...")
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
        # Find the global minimum in X velocity
        minimum = np.argmin(df['GYRO X (deg/s)'])

        # First derivative (rate of change) to detect where the acceleration starts decreasing
        acc_z_diff = np.diff(df['ACC Z (G)'])
        # Find all acceleration-change extremas
        maxima = argrelextrema(acc_z_diff, np.greater)[0]
        # Get the two maxima closest to the global minimum
        maxima.sort()

        try:
            # Maximum in acc z diff before global minimum is a good way to detect the start of the motion
            start_idx = maxima[np.searchsorted(maxima, minimum) - 1]

            # More conservative (stops later)
            # # We need to have at least three maxima to be able to detect the end of the motion, as sometimes people might sit down to quickly
            # if len(maxima) > 2:
            #     end_idx = maxima[np.searchsorted(maxima, minimum)]
            # # If there is no maxima, there will still always be a change of signs in the jerk
            # else:
            #     acc_z_diff_diff = np.diff(acc_z_diff)
            #     acc_z_diff_diff_minima = argrelextrema(acc_z_diff_diff, np.less)[0]
            #     # Select the minimum closest to the global minimum (higher than the global minimum)
            #     end_idx = acc_z_diff_diff_minima[np.searchsorted(acc_z_diff_diff_minima, minimum)]

            # Less conservative (stops earlier)
            gyro_x_diff = np.diff(df['GYRO X (deg/s)'])
            gyro_x_diff_diff = np.diff(gyro_x_diff)
            minima = argrelextrema(gyro_x_diff_diff, np.less)[0]
            end_idx = minima[np.searchsorted(minima, minimum)]
        except Exception as e:
            print(e)
            start_idx = None
            end_idx = None

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
    
    def filter_emg(self, unfiltered_df: pd.DataFrame, low_pass=4, sfreq=1259.2593, high_band=20, low_band=450) -> pd.DataFrame:
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
        # IMU sensor label, using the sensor on the Rectus Femoris (RIGHT)
        # The arrow of the sensor should be pointing towards the torso
        imu_sensor_label = 0
        gyro_axis = 'X'
        acc_axis = 'Z'

        # Hyperparameters
        peak_threshold = 50

        relevant_emg = []
        assisted_mean = 0

        if len(self.complete_gyro_data[imu_sensor_label][gyro_axis]) > 100:
            # Get the mean of the past 100 Gyro X values
            gyro_x_mean = np.mean(self.complete_gyro_data[imu_sensor_label][gyro_axis][-100:])

            if gyro_x_mean > peak_threshold and not self.peak:
                self.peak = True
                self.segment_end_idx_imu = len(self.complete_gyro_data[imu_sensor_label][gyro_axis])
                self.count_peak += 1

            if self.count_peak > self.analysed_segments:
                self.analysed_segments += 1

                #Extract the segment of the x-axis gyro data
                gyro_x_segment = self.complete_gyro_data[imu_sensor_label][gyro_axis][self.segment_start_idx_imu:self.segment_end_idx_imu]
                gyro_x_segment = pd.DataFrame(gyro_x_segment, columns=['GYRO X (deg/s)'])

                # Extract the segment of the z-axis acceleration data
                z_acc_segment = self.complete_acc_data[imu_sensor_label][acc_axis][self.segment_start_idx_imu:self.segment_end_idx_imu]
                z_acc_segment = pd.DataFrame(z_acc_segment, columns=['ACC Z (G)'])

                # Concate the two segments
                imu_data = pd.concat([gyro_x_segment, z_acc_segment], axis=1)
                print(self.gyro_sample_rates[imu_sensor_label])
                imu_filtered = self.apply_lowpass_filter(imu_data, 1, self.gyro_sample_rates[imu_sensor_label])   

                # Extract time (start,end)
                self.sts_start_idx_imu, self.sts_end_idx_imu = self.extract_time(imu_filtered)

                if self.sts_start_idx_imu is None or self.sts_end_idx_imu is None or self.sts_start_idx_imu >= self.sts_end_idx_imu:
                    print("Failed to extract start and end indices.")
                    # TODO: Write a function, that takes a larger segment of data into account and tries to extract the start and end indices again and compares them with the previously found indeces to avoid taking the same segment twice

                else:
                    # Convert start and end indices from imu to emg indices
                    emg_start_idx = int(np.round(self.sts_start_idx_imu * self.emg_sampling_frequencies[imu_sensor_label] / self.acc_sample_rates[imu_sensor_label]))
                    emg_end_idx = int(np.round(self.sts_end_idx_imu * self.emg_sampling_frequencies[imu_sensor_label] / self.acc_sample_rates[imu_sensor_label]))

                    # Extract relevant_emg data
                    for sensor_label in self.complete_emg_data.keys():
                        # Log the extracted emg data
                        self.data_exporter.export_sts_data_to_csv(self.complete_emg_data[sensor_label][emg_start_idx:emg_end_idx], sensor_label)
                        if relevant_emg:
                            # Append to existing data
                            relevant_emg += self.complete_emg_data[sensor_label][emg_start_idx:emg_end_idx]
                        else:
                            relevant_emg = self.complete_emg_data[sensor_label][emg_start_idx:emg_end_idx]

                    # Convert to DataFrame
                    relevant_emg = pd.DataFrame(relevant_emg)
                    # Filter relevant_emg data
                    relevant_emg_filtered, env_freq = self.filter_emg(relevant_emg)

                    # Get score
                    # Calculated reference score if unassisted
                    if self.unassisted:
                        self.unassisted_counter += 1
                        if self.unassisted_counter == 1:
                            self.unassisted_mean = np.mean(relevant_emg_filtered)
                        else:
                            self.unassisted_mean = (self.unassisted_mean * (self.unassisted_counter - 1) + np.mean(relevant_emg_filtered)) / self.unassisted_counter

                        # Log the extracted variables
                        log_entry = {
                            'Segment Start Index': self.segment_start_idx_imu,
                            'Segment End Index': self.segment_end_idx_imu,
                            'Start Time': self.sts_start_idx_imu,
                            'End Time': self.sts_end_idx_imu,
                            'Unassisted Mean': self.unassisted_mean
                        }
                        self.log_entries.append(log_entry)

                    else: 
                        # Compare score of assisted vs unassisted
                        assisted_mean = np.mean(relevant_emg_filtered)
                        # Send comparison score to socket server
                        if self.socket:
                            self.data_handler.send_data(str(self.unassisted_mean - assisted_mean))

                        # Log the extracted variables
                        log_entry = {
                            'Segment Start Index': self.segment_start_idx_imu,
                            'Segment End Index': self.segment_end_idx_imu,
                            'Start Time': self.sts_start_idx_imu,
                            'End Time': self.sts_end_idx_imu,
                            'Assisted Mean': assisted_mean,
                        }
                        self.log_entries.append(log_entry)


            if self.peak:
                if gyro_x_mean < peak_threshold:
                    self.peak = False
                    self.segment_start_idx_imu = len(self.complete_gyro_data[imu_sensor_label][gyro_axis])


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
        # Close the socket connection
        if self.socket:
            self.data_handler.send_data("Kill")
            self.data_handler.close_connection()
        self.close()
