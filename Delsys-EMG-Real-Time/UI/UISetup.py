from PyQt5 import QtWidgets, QtCore
import os

class UISetup:
    def __init__(self, parent):
        self.parent = parent

    def init_ui(self):
        self.parent.setWindowTitle("EMG Data Collection")
        self.parent.resize(1200, 800)

        # Central widget
        central_widget = QtWidgets.QWidget()
        self.parent.setCentralWidget(central_widget)

        # Main layout
        main_layout = QtWidgets.QVBoxLayout(central_widget)

        # Time display label
        self.parent.time_label = QtWidgets.QLabel("Elapsed Time: 0.00 s")
        self.parent.time_label.hide()
        main_layout.addWidget(self.parent.time_label)

        # Button layout (first page buttons)
        first_page_buttons = QtWidgets.QWidget()
        first_page_layout = QtWidgets.QHBoxLayout(first_page_buttons)

        self.parent.start_unassisted_button = QtWidgets.QPushButton("Start Unassisted Trial")
        self.parent.start_button = QtWidgets.QPushButton("Start Trial")
        self.parent.select_mode_button = QtWidgets.QPushButton("Select Mode on Raspi")
        self.parent.reconnect_button = QtWidgets.QPushButton("Reconnect to Raspi")
        self.parent.scp_button = QtWidgets.QPushButton("Export Data to Host")
        self.parent.autoscale_button = QtWidgets.QPushButton("Autoscale")
        self.parent.quit_button = QtWidgets.QPushButton("Quit")

        first_page_layout.addWidget(self.parent.start_unassisted_button)
        first_page_layout.addWidget(self.parent.start_button)
        first_page_layout.addWidget(self.parent.select_mode_button)
        first_page_layout.addWidget(self.parent.reconnect_button)
        first_page_layout.addWidget(self.parent.scp_button)
        first_page_layout.addWidget(self.parent.autoscale_button)
        first_page_layout.addWidget(self.parent.quit_button)

        # Button layout (second page buttons)
        second_page_buttons = QtWidgets.QWidget()
        second_page_layout = QtWidgets.QHBoxLayout(second_page_buttons)

        self.parent.motor_button = QtWidgets.QPushButton("Stop Motor")
        self.parent.stop_emg_button = QtWidgets.QPushButton("Stop EMG Trial")
        self.parent.second_reconnect_button = QtWidgets.QPushButton("Reconnect to Raspi")
        self.parent.quit_second_page_button = QtWidgets.QPushButton("Quit")

        second_page_layout.addWidget(self.parent.motor_button)
        second_page_layout.addWidget(self.parent.stop_emg_button)
        second_page_layout.addWidget(self.parent.second_reconnect_button)
        second_page_layout.addWidget(self.parent.quit_second_page_button)

        # Plot widget
        self.parent.plot_widget = QtWidgets.QWidget()
        self.parent.plot_widget.setMinimumHeight(500)  # Ensure plots are always visible

        # Add elements to the main layout
        main_layout.addWidget(self.parent.plot_widget)
        main_layout.addWidget(first_page_buttons)
        main_layout.addWidget(second_page_buttons)

        # Show the first page buttons by default
        second_page_buttons.hide()

        # Button actions
        self.parent.start_unassisted_button.clicked.connect(self.start_unassisted_trial)
        self.parent.start_button.clicked.connect(self.start_trial)
        self.parent.select_mode_button.clicked.connect(self.select_mode)
        self.parent.reconnect_button.clicked.connect(self.parent.reconnect_to_raspi)
        self.parent.scp_button.clicked.connect(self.parent.export_to_host)
        self.parent.autoscale_button.clicked.connect(self.parent.plotter.autoscale_plots)
        self.parent.quit_button.clicked.connect(self.parent.on_quit)

        self.parent.motor_button.clicked.connect(self.toggle_motor)
        self.parent.stop_emg_button.clicked.connect(self.stop_trial)
        self.parent.second_reconnect_button.clicked.connect(self.parent.reconnect_to_raspi)
        self.parent.quit_second_page_button.clicked.connect(self.parent.on_quit)

        self.first_page_buttons = first_page_buttons
        self.second_page_buttons = second_page_buttons

        self.parent.show()

    def start_unassisted_trial(self):
        self.parent.unassisted = True
        self.toggle_motor()
        self.switch_to_second_page()
        self.parent.start_trial()

    def start_trial(self):
        self.parent.unassisted = False
        filename_unassisted_mean = "most_recent_unassisted_mean.npy"
        filepath_unassisted = os.path.join(self.parent.data_directory, f"subject_{self.parent.subject_number}", filename_unassisted_mean)
        # If the file doesn't exist, don't start the trial
        if not os.path.exists(filepath_unassisted):
            QtWidgets.QMessageBox.information(self.parent, "Error", "Please run an unassisted trial first.")
        else:
            self.toggle_motor()
            self.switch_to_second_page()
            self.parent.start_trial()

    def select_mode(self):
        self.parent.select_raspi_mode()

    def toggle_motor(self):
        if self.parent.motor_running:
            self.parent.motor_button.setText("Start Motor")
            print("Motor stopped")
        else:
            self.parent.motor_button.setText("Stop Motor")
            print("Motor started")
        self.parent.toggle_motor()

    def stop_trial(self):
        if self.parent.motor_running:
            self.toggle_motor()
        self.parent.stop_trial()
        self.switch_to_first_page()

    def switch_to_first_page(self):
        self.first_page_buttons.show()
        self.second_page_buttons.hide()

    def switch_to_second_page(self):
        self.first_page_buttons.hide()
        self.second_page_buttons.show()
