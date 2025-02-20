import sys
from PyQt5 import QtWidgets
from EMGDataCollector import EMGDataCollector

def main():
    # Set main parameters here
    window_duration = 5  # Duration in seconds

    data_directory = "C:/Users/patty/Desktop/Nate_3rd_arm/code/assistive-arm/Data/"
    # Flag for real time plots
    plot = False
    # Flag for Socket connection
    socket = False
    # Flag for imu score calcualtion (default is OR, if real_time_processing is True)
    imu_processing = True
    # Flag for mixed processing (start through IMU, stop through OR)
    mixed_processing = False
    # Flag for EMG control
    emg_control = False
    # Flag for real time processing (if it is off, the data will be segmented by the start and stop buttons)
    # This is a backup if the segemntation fails due to lag or other issues
    real_time_processing = True

    appQt = QtWidgets.QApplication(sys.argv)
    collector = EMGDataCollector(plot, socket, imu_processing, mixed_processing, emg_control, real_time_processing, window_duration=window_duration, data_directory=data_directory)
    collector.connect_base()
    collector.scan_and_pair_sensors()
    sys.exit(appQt.exec_())

if __name__ == "__main__":
    main()
