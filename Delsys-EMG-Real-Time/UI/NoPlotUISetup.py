from PyQt5 import QtWidgets
import os
class NoPlotUISetup:
    def __init__(self, parent):
        self.parent = parent
        # self.motor_running = False  # Tracks motor state

    def init_ui(self):
        self.parent.setWindowTitle("EMG Data Collection")
        self.parent.resize(400, 200)

        # Central widget
        self.central_widget = QtWidgets.QWidget()
        self.parent.setCentralWidget(self.central_widget)

        # Main layout
        self.main_layout = QtWidgets.QVBoxLayout(self.central_widget)

        # Time display label
        self.parent.time_label = QtWidgets.QLabel('Elapsed Time: 0.00 s')
        self.parent.time_label.hide()
        self.main_layout.addWidget(self.parent.time_label)

        # First Page Buttons
        self.first_page_buttons = QtWidgets.QWidget()
        first_page_layout = QtWidgets.QVBoxLayout(self.first_page_buttons)

        self.start_unassisted_button = QtWidgets.QPushButton("Start Unassisted Trial")
        self.start_button = QtWidgets.QPushButton("Start Trial")
        self.select_mode_button = QtWidgets.QPushButton("Select Mode on Raspi")
        self.reconnect_button = QtWidgets.QPushButton("Reconnect to Raspi")
        self.scp_button = QtWidgets.QPushButton("Export collected data to Host")
        self.quit_button = QtWidgets.QPushButton("Quit")

        self.start_unassisted_button.clicked.connect(self.start_unassisted_trial)
        self.start_button.clicked.connect(self.start_trial)
        self.select_mode_button.clicked.connect(self.select_mode)
        self.reconnect_button.clicked.connect(self.parent.reconnect_to_raspi)
        self.scp_button.clicked.connect(self.parent.export_to_host)
        self.quit_button.clicked.connect(self.parent.on_quit)

        first_page_layout.addWidget(self.start_unassisted_button)
        first_page_layout.addWidget(self.start_button)
        first_page_layout.addWidget(self.select_mode_button)
        first_page_layout.addWidget(self.reconnect_button)
        first_page_layout.addWidget(self.scp_button)
        first_page_layout.addWidget(self.quit_button)

        # Second Page Buttons
        self.second_page_buttons = QtWidgets.QWidget()
        second_page_layout = QtWidgets.QVBoxLayout(self.second_page_buttons)

        self.motor_button = QtWidgets.QPushButton("Stop Motor")
        self.stop_emg_button = QtWidgets.QPushButton("Stop EMG Trial")
        self.quit_second_page_button = QtWidgets.QPushButton("Quit")

        self.motor_button.clicked.connect(self.toggle_motor)
        self.stop_emg_button.clicked.connect(self.stop_trial)
        self.quit_second_page_button.clicked.connect(self.parent.on_quit)

        second_page_layout.addWidget(self.motor_button)
        second_page_layout.addWidget(self.stop_emg_button)
        second_page_layout.addWidget(self.quit_second_page_button)

        # Add first page buttons to the main layout
        self.main_layout.addWidget(self.first_page_buttons)
        self.main_layout.addWidget(self.second_page_buttons)

        # Show the first page by default
        self.second_page_buttons.hide()

        self.parent.show()

    def start_unassisted_trial(self):
        self.toggle_motor()
        self.switch_to_second_page()
        self.parent.start_unassisted_trial()

    def start_trial(self):
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
        QtWidgets.QMessageBox.information(self.parent, "Mode Selection", "Mode selection on Raspi initiated.")
        self.parent.select_raspi_mode()

    def toggle_motor(self):
        if self.parent.motor_running:
            self.motor_button.setText("Start Motor")
            print("Motor stopped")
        else:
            self.motor_button.setText("Stop Motor")
            print("Motor started")
        self.parent.toggle_motor()


    def stop_trial(self):
        # Only stop the motor if it is running
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

