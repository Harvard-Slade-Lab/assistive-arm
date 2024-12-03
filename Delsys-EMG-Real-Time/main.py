import sys
from PyQt5 import QtWidgets
from EMGDataCollector import EMGDataCollector

def main():
    # Set main parameters here
    window_duration = 5  # Duration in seconds

    data_directory = "Data"
    # Flag for real time plots
    plot = False
    # Flag for Socket connection
    socket = False
    # Flag for realtime processing
    real_time = True

    appQt = QtWidgets.QApplication(sys.argv)
    collector = EMGDataCollector(plot, socket, real_time, window_duration=window_duration, data_directory=data_directory)
    collector.connect_base()
    collector.scan_and_pair_sensors()
    sys.exit(appQt.exec_())

if __name__ == "__main__":
    main()
