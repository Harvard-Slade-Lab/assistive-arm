import sys
import time
import threading
import numpy as np
import os
import datetime
import queue
import json
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
from EMGutils.EMGfilter import apply_lowpass_filter, filter_emg

from concurrent.futures import ThreadPoolExecutor

class EMGDataCollector(QtWidgets.QMainWindow):
    def __init__(self, plot=False, socket=False, real_time=False, window_duration=5, data_directory="Data"):
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
        self.unassisted_mean = None

        # Flag to see if motor is running
        self.motor_running = False

        # Flag to select if data should be processed in real-time or not
        self.real_time = real_time

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

        # New thread for segmentation
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.last_submission_time = None

        # Simulation activation means
        self.simulation_means = None
        self.simulation_max = None

        self.load_activation_means()

    def load_activation_means(self):
        # means = pd.read_csv("EMGutils/means.csv")
        with open('C:/Users/patty/Desktop/Nate_3rd_arm/code/assistive-arm/Delsys-EMG-Real-Time/EMGutils/means.json', 'r') as f:
            self.simulation_means = json.load(f)
        with open('C:/Users/patty/Desktop/Nate_3rd_arm/code/assistive-arm/Delsys-EMG-Real-Time/EMGutils/max.json', 'r') as f:
            self.simulation_max = json.load(f)

    def reconnect_to_raspi(self):
        if self.socket:
            self.data_handler.connect_to_server()
            # Resend profile name
            self.data_handler.send_data(f"Profile:{self.assistive_profile_name}")

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

        while True:
            project_name = input("Enter project name (0 if no project defined): ").strip()
            if project_name in ['0', 'sts', 'sts_2']:
                break  # Exit the loop if a valid name is entered
            else:
                print("Invalid project name. Please enter a valid project name.")


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
            id_to_name = {"000140e7": "IMU", "00014173": "RF_R", "000140e6": "VM_R", "00014163": "BF_R", "00013f5b": "G_R", "000140dd": "RF_L", "000140e9": "VM_L", "00014174": "BF_L", "00013f2a": "G_L"}
            # sensor_names = {1: "IMU", 2: "RF_R", 3: "VM_R", 4: "BF_R", 5: "G_R", 6: "RF_L", 7: "VM_L", 8: "BF_L", 9: "G_L"}

            self.sensor_label_to_index = {}
            for idx, sensor in enumerate(self.base.all_scanned_sensors):
                label = sensor.PairNumber
                self.sensor_label_to_index[label] = idx
                # Extract first 8 characters of the sensor ID
                sensor_id = str(sensor.Id)[:8]
                if sensor_id in id_to_name:
                    self.sensor_names[label] = id_to_name[sensor_id]

            for label, sensor_index in self.sensor_label_to_index.items():
                # self.sensor_names[label] = sensor_names[sensor_index+1]
                modes = self.base.getSampleModes(sensor_index)
                if self.sensor_names[label] == 'IMU':
                    self.base.setSampleMode(sensor_index, modes[110])
                else:
                    self.base.setSampleMode(sensor_index, modes[4])

        if project_name == 'sts_2':
            id_to_name = {"000140e7": "IMU", "00014173": "RF_R", "000140e6": "VM_R", "00014163": "BF_R", "00013f5b": "G_R", "000140dd": "TA_R", "000140e9": "SO_R", "00014174": "RF_L", "00013f2a": "VM_L", "00014178": "BF_L", "00014111": "G_L", "00013f2d": "TA_L", "0001416d": "SO_L"}

            self.sensor_label_to_index = {}
            for idx, sensor in enumerate(self.base.all_scanned_sensors):
                label = sensor.PairNumber
                self.sensor_label_to_index[label] = idx
                # Extract first 8 characters of the sensor ID
                sensor_id = str(sensor.Id)[:8]
                if sensor_id in id_to_name:
                    self.sensor_names[label] = id_to_name[sensor_id]

            for label, sensor_index in self.sensor_label_to_index.items():
                # self.sensor_names[label] = sensor_names[sensor_index+1]
                modes = self.base.getSampleModes(sensor_index)
                if self.sensor_names[label] == 'IMU':
                    self.base.setSampleMode(sensor_index, modes[110])
                else:
                    self.base.setSampleMode(sensor_index, modes[4])

        # Get subject number
        self.subject_number = input("Enter subject number: ").strip()
        # self.assistive_profile_name = input("Enter assistive profile name: ").strip()

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

        self.assistive_profile_name = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        # Send profile label to socket server
        if self.socket:
            self.data_handler.send_data(f"Profile:{self.assistive_profile_name}")

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
            # Reset the start index
            self.segment_start_idx_imu = 0

            # Reinitialize the executor if it was shutdown
            if self.executor is None or self.executor._shutdown:
                self.executor = ThreadPoolExecutor(max_workers=4)

            # Set unassisted flag to false to trigger score calculation
            self.unassisted = False
            # Reset counter and for unassisted runs
            self.count_unassisted = 0
            self.unassisted_mean = self.data_exporter.load_unassisted_mean_from_csv()

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

            # Reset the start index
            self.segment_start_idx_imu = 0

            # Reinitialize the executor if it was shutdown
            if self.executor is None or self.executor._shutdown:
                self.executor = ThreadPoolExecutor(max_workers=4)

            # Set unassisted flag to true to trigger unassisted data reference collection
            self.unassisted = True

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

    def toggle_motor(self):
        if self.motor_running:
            if self.socket:
                self.data_handler.send_data("Stop")
                self.assistive_profile_name = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                # Send profile label to socket server for next profile
                self.data_handler.send_data(f"Profile:{self.assistive_profile_name}")
        else:
            if self.socket:
                self.data_handler.send_data("Start")
        self.motor_running = not self.motor_running

    def select_raspi_mode(self):
        if self.socket:
            self.data_handler.send_data("Mode")

    def stop_trial(self):
        if self.is_collecting:
            # This is used if single profiles are run and only one sts is performed
            if self.count_peak == 0 and self.real_time:
                self.calculate_score(0, len(self.complete_gyro_data[0]["X"]))

            while not self.data_queue.empty():
                time.sleep(0.1)

            print("Stopping trial...")
            self.stop_collection()
            # time.sleep(2)  # Wait for last batch data
            filename_emg = f"{self.current_date}_EMG_Profile_{self.assistive_profile_name}_Trial_{self.trial_number}.csv"
            filename_acc = f"{self.current_date}_ACC_Profile_{self.assistive_profile_name}_Trial_{self.trial_number}.csv"
            filename_gyro = f"{self.current_date}_GYRO_Profile_{self.assistive_profile_name}_Trial_{self.trial_number}.csv"
            subject_folder = os.path.join(self.data_directory, f"subject_{self.subject_number}")
            emg_folder = os.path.join(subject_folder, "Raw")
            if not os.path.exists(emg_folder):
                os.makedirs(emg_folder)
            filepath_emg = os.path.join(emg_folder, filename_emg)
            filepath_acc = os.path.join(emg_folder, filename_acc)
            filepath_gyro = os.path.join(emg_folder, filename_gyro)

            # Export collected data
            self.data_exporter.export_data_to_csv(filepath_emg, filepath_acc, filepath_gyro)
            print(f"Trial {self.trial_number} data saved as: {filename_emg}, {filename_acc}, and {filename_gyro}")

            # Reset data logger
            self.log_entries = []
            self.trial_number += 1

        # Ask for new assistive profile name
        # assistive_profile_name, ok = QtWidgets.QInputDialog.getText(self, 'Assistive Profile Name', 'Enter new assistive profile name or cancel:')
        # if ok:
        #     self.trial_number = 1
        #     self.assistive_profile_name = assistive_profile_name

    def start_collection(self):
        print("Starting data collection...")
        self.base.Start_Callback(start_trigger=False, stop_trigger=False)
        self.threadManager(start_trigger=False, stop_trigger=False)

    def stop_collection(self):
        print("Stopping data collection...")
        self.pauseFlag = True
        self.base.Stop_Callback()

        try:
            if hasattr(self, "executor") and self.executor:
                self.executor.shutdown(wait=True)  # Gracefully wait for all tasks to complete
                self.executor = None
        except Exception as e:
            print(f"Error while shutting down executor: {e}")

        self.is_collecting = False  # Reset collecting flag
    
    def export_to_host(self):
        self.data_exporter.export_to_host()

    def threadManager(self, start_trigger, stop_trigger):
        self.start_trigger = start_trigger
        self.stop_trigger = stop_trigger

        # Submit tasks to the executor
        self.executor.submit(self.stream_data)
        self.executor.submit(self.data_processor.process_data)

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

    def calculate_score(self, current_segment_start_idx_imu, current_segment_end_idx_imu, current_assistive_profile_name):
        # IMU sensor label, using the sensor on the Rectus Femoris (RIGHT)
        # The arrow of the sensor should be pointing towards the torso
        imu_sensor_label = 0
        gyro_axis = 'X'
        acc_axis = 'Z'
        assisted_mean = 0
        self.analysed_segments += 1

        # Extract the segment of the x-axis gyro data
        gyro_x_segment = self.complete_gyro_data[imu_sensor_label][gyro_axis][current_segment_start_idx_imu:current_segment_end_idx_imu]
        gyro_x_segment = pd.DataFrame(gyro_x_segment, columns=['GYRO X (deg/s)'])
        # Filter gyro x first, as sometimes the buffer for the acc data is not filled
        try:
            gyro_x_filtered = apply_lowpass_filter(gyro_x_segment, 1, self.gyro_sample_rates[imu_sensor_label])
        except Exception as e:
            print(e)
            print(f"Failed to filter gyro data, length was {len(gyro_x_segment)}.")
            # Send message to raspi to repeat iteration and add identification tag
            self.data_handler.send_data(f"Repeat_{current_assistive_profile_name}")
            return

        # Extract the segment of the z-axis acceleration data
        z_acc_segment = self.complete_acc_data[imu_sensor_label][acc_axis][current_segment_start_idx_imu:current_segment_end_idx_imu]
        z_acc_segment = pd.DataFrame(z_acc_segment, columns=['ACC Z (G)'])
        z_acc_filtered = apply_lowpass_filter(z_acc_segment, 1, self.acc_sample_rates[imu_sensor_label])

        # Concatenate the two segments maybe better to input directly without concatenation to avoid nans
        imu_filtered = pd.concat([gyro_x_filtered, z_acc_filtered], axis=1)

        # Extract time (start,end)
        self.sts_start_idx_imu, self.sts_end_idx_imu = self.extract_time(imu_filtered)

        if self.sts_start_idx_imu is None or self.sts_end_idx_imu is None or self.sts_start_idx_imu >= self.sts_end_idx_imu:
            print("Failed to extract start and end indices, iteration will be repeated.")
            # Send message to raspi to repeat iteration and add identification tag
            self.data_handler.send_data(f"Repeat_{current_assistive_profile_name}")
        else:
            # Convert start and end indices from imu to emg indices
            ratio = self.emg_sampling_frequencies[imu_sensor_label] / self.acc_sample_rates[imu_sensor_label]
            current_segment_start_idx_emg = int(np.round(current_segment_start_idx_imu * ratio))
            current_segment_end_idx_emg = int(np.round(current_segment_end_idx_imu * ratio))
            emg_start_idx = int(np.round(self.sts_start_idx_imu * ratio))
            emg_end_idx = int(np.round(self.sts_end_idx_imu * ratio))

            # Get an extra buffer at both ends (larger than it has to be, as I want to save some extra data as well)
            # I don't see when the checks for accessing a non existent index is needed, but it is a potential failure case
            buffer_size = 0.25 * self.emg_sampling_frequencies[imu_sensor_label]
            if emg_start_idx - buffer_size < 0:
                start_buffer_idx = current_segment_start_idx_emg
            else:
                start_buffer_idx = int(np.round(emg_start_idx - buffer_size))
            if emg_end_idx + buffer_size > len(self.complete_emg_data[imu_sensor_label]):
                end_buffer_idx = current_segment_end_idx_emg
            else:
                end_buffer_idx = int(np.round(emg_end_idx + buffer_size))
            
            imu_start_buffer_idx = int(np.round(start_buffer_idx / ratio))
            imu_end_buffer_idx = int(np.round(end_buffer_idx / ratio))

            complete_emg_data = pd.DataFrame()
            # complete_imu_data = pd.DataFrame()
            # Extract relevant_emg data, one sensor at a time
            for sensor_label in self.complete_emg_data.keys():
                if self.sensor_names[sensor_label] == 'IMU':
                    gyro_data = {key: value[imu_start_buffer_idx:imu_end_buffer_idx] for key, value in self.complete_gyro_data[sensor_label].items()}
                    acc_data = {key: value[imu_start_buffer_idx:imu_end_buffer_idx] for key, value in self.complete_acc_data[sensor_label].items()}
                    sensor_data = pd.concat([pd.DataFrame(gyro_data), pd.DataFrame(acc_data)], axis=1)
                    sensor_data.columns = ['GYRO X', 'GYRO Y', 'GYRO Z', 'ACC X', 'ACC Y', 'ACC Z']
                    # Save IMU data
                    self.data_exporter.export_sts_data_to_csv(sensor_data, self.sensor_names[sensor_label])
                else:
                    sensor_data = self.complete_emg_data[sensor_label][start_buffer_idx:end_buffer_idx]
                    complete_emg_data[self.sensor_names[sensor_label]] = sensor_data
                
            # Log the extracted emg data as a single file
            self.data_exporter.export_all_sts_data_to_csv(complete_emg_data)

            # Filter relevant_emg data
            relevant_emg_filtered, env_freq = filter_emg(complete_emg_data, sfreq=self.emg_sampling_frequencies[2])

            # Cut the relevant emg data to the start and end of the sts motion
            # Check if the start and end indices are correctm -> checked
            relevant_emg_filtered = relevant_emg_filtered[emg_start_idx-start_buffer_idx:emg_end_idx-start_buffer_idx]

            # Calculated reference score if unassisted
            if self.unassisted:
                self.unassisted_counter += 1
                if self.unassisted_counter == 1:
                    # self.unassisted_mean = np.mean(relevant_emg_filtered, axis=0)
                    # Convert to max (TODO change naming if this works)
                    self.unassisted_mean = np.max(relevant_emg_filtered, axis=0)
                else:
                    self.unassisted_mean = (self.unassisted_mean * (self.unassisted_counter - 1) + np.mean(relevant_emg_filtered, axis=0)) / self.unassisted_counter

                # Save the unassisted mean to a file
                self.data_exporter.export_unassisted_mean_to_csv(self.unassisted_mean)

                # Log the extracted variables
                log_entry = {
                    'Tag': current_assistive_profile_name,
                    'Segment Start Index': current_segment_start_idx_imu,
                    'Segment End Index': current_segment_end_idx_imu,
                    'Start Time': self.sts_start_idx_imu,
                    'End Time': self.sts_end_idx_imu,
                    'Score': 0,
                }
                self.log_entries.append(log_entry)

            else: 
                # Compare score of assisted vs unassisted
                assisted_mean = np.mean(relevant_emg_filtered, axis=0)

                score = 0

                # Scale by how much we expect the muscle to change
                # rf_sim = self.simulation_means['rf_un']
                # vm_sim = self.simulation_means['vm_un']
                rf_sim = self.simulation_max['rf_un']
                vm_sim = self.simulation_max['vm_un']

                for i, sensor_label in enumerate(relevant_emg_filtered.keys()):
                    if 'RF' in sensor_label:
                        score += rf_sim*(1-assisted_mean[sensor_label]/self.unassisted_mean[i])
                    elif 'VM' in sensor_label:
                        score += vm_sim*(1-assisted_mean[sensor_label]/self.unassisted_mean[i])

                # Send comparison score to socket server
                if self.socket:
                    self.data_handler.send_data(f"Score_{score}_Tag_{current_assistive_profile_name}")

                # Log the extracted variables
                log_entry = {
                    'Tag': current_assistive_profile_name,
                    'Segment Start Index': current_segment_start_idx_imu,
                    'Segment End Index': current_segment_end_idx_imu,
                    'Start Time': self.sts_start_idx_imu,
                    'End Time': self.sts_end_idx_imu,
                    'Score': score,
                }
                self.log_entries.append(log_entry)

    
    def detect_peak_and_calculate(self):
        # IMU sensor label, using the sensor on the Rectus Femoris (RIGHT)
        # The arrow of the sensor should be pointing towards the torso
        imu_sensor_label = 0
        gyro_axis = 'X'

        # Hyperparameters
        peak_threshold = 50

        if len(self.complete_gyro_data[imu_sensor_label][gyro_axis]) > 100:
            # Get the mean of the past 100 Gyro X values
            gyro_x_mean = np.mean(self.complete_gyro_data[imu_sensor_label][gyro_axis][-100:])

            # Detect peak to see if a sts has ended
            if gyro_x_mean > peak_threshold and not self.peak:
                self.peak = True
                self.segment_end_idx_imu = len(self.complete_gyro_data[imu_sensor_label][gyro_axis])
                self.count_peak += 1

            if self.count_peak > self.analysed_segments:
                if self.last_submission_time is None or time.time() - self.last_submission_time > 1:
                    self.last_submission_time = time.time()
                    # Get current segment indices, so they cannot get overwritten
                    current_segment_start_idx_imu = self.segment_start_idx_imu
                    current_segment_end_idx_imu = self.segment_end_idx_imu
                    current_profile_name = self.assistive_profile_name
                    # Process the data in a separate thread
                    self.executor.submit(self.calculate_score(current_segment_start_idx_imu, current_segment_end_idx_imu, current_profile_name))

            if self.peak:
                if gyro_x_mean < peak_threshold:
                    self.peak = False
                    self.segment_start_idx_imu = len(self.complete_gyro_data[imu_sensor_label][gyro_axis])

    def stream_data(self):
        """Collect data from sensors and put it into the queue."""
        while not self.pauseFlag:
            self.data_handler.processData(self.data_queue)
            if self.real_time:
                if self.executor is None or self.executor._shutdown:
                    continue
                else:
                    self.detect_peak_and_calculate()

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
