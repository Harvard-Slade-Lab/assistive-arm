from PyQt5 import QtWidgets
class UISetup:
    def __init__(self, parent, plot_flag):
        self.parent = parent
        self.plot_flag = plot_flag

    def init_ui(self):
        self.parent.setWindowTitle("EMG Data Collection")
        if self.plot_flag:
            self.parent.resize(1200, 800)
        else:
            self.parent.resize(200, 400)

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
        if self.plot_flag:
            first_page_layout = QtWidgets.QHBoxLayout(first_page_buttons)
        else:
            first_page_layout = QtWidgets.QVBoxLayout(first_page_buttons)

        self.parent.calibration_button = QtWidgets.QPushButton("Calibrate Height")
        self.parent.start_unassisted_button = QtWidgets.QPushButton("Start Unassisted Trial")
        self.parent.start_button = QtWidgets.QPushButton("Start Trial")
        self.parent.select_mode_button = QtWidgets.QPushButton("Select Mode on Raspi")
        self.parent.reconnect_button = QtWidgets.QPushButton("Reconnect to Raspi")
        self.parent.scp_button = QtWidgets.QPushButton("Export Data to Host")
        self.parent.test_button = QtWidgets.QPushButton("Test")
        if self.plot_flag:
            self.parent.autoscale_button = QtWidgets.QPushButton("Autoscale")
        self.parent.quit_button = QtWidgets.QPushButton("Quit")

        first_page_layout.addWidget(self.parent.calibration_button)
        first_page_layout.addWidget(self.parent.start_unassisted_button)
        first_page_layout.addWidget(self.parent.start_button)
        first_page_layout.addWidget(self.parent.select_mode_button)
        first_page_layout.addWidget(self.parent.reconnect_button)
        first_page_layout.addWidget(self.parent.scp_button)
        first_page_layout.addWidget(self.parent.test_button)
        if self.plot_flag:
            first_page_layout.addWidget(self.parent.autoscale_button)
        first_page_layout.addWidget(self.parent.quit_button)

        # Button layout (second page buttons)
        second_page_buttons = QtWidgets.QWidget()
        if self.plot_flag:
            second_page_layout = QtWidgets.QHBoxLayout(second_page_buttons)
        else:
            second_page_layout = QtWidgets.QVBoxLayout(second_page_buttons)

        self.parent.motor_button = QtWidgets.QPushButton("Stop Motor")
        self.parent.stop_emg_button = QtWidgets.QPushButton("Stop EMG Trial")
        self.parent.second_reconnect_button = QtWidgets.QPushButton("Reconnect to Raspi")
        self.parent.quit_second_page_button = QtWidgets.QPushButton("Quit")

        second_page_layout.addWidget(self.parent.motor_button)
        second_page_layout.addWidget(self.parent.stop_emg_button)
        second_page_layout.addWidget(self.parent.second_reconnect_button)
        second_page_layout.addWidget(self.parent.quit_second_page_button)

        if self.plot_flag:
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
        self.parent.calibration_button.clicked.connect(self.calibrate_height)
        self.parent.start_unassisted_button.clicked.connect(self.start_unassisted_trial)
        self.parent.start_button.clicked.connect(self.start_trial)
        self.parent.select_mode_button.clicked.connect(self.select_mode)
        self.parent.reconnect_button.clicked.connect(self.parent.reconnect_to_raspi)
        self.parent.scp_button.clicked.connect(self.parent.export_to_host)
        self.parent.test_button.clicked.connect(self.toggle_test_flag)
        if self.plot_flag:
            self.parent.autoscale_button.clicked.connect(self.parent.plotter.autoscale_plots)
        self.parent.quit_button.clicked.connect(self.parent.on_quit)

        self.parent.motor_button.clicked.connect(self.toggle_motor)
        self.parent.stop_emg_button.clicked.connect(self.stop_trial)
        self.parent.second_reconnect_button.clicked.connect(self.parent.reconnect_to_raspi)
        self.parent.quit_second_page_button.clicked.connect(self.parent.on_quit)

        self.first_page_buttons = first_page_buttons
        self.second_page_buttons = second_page_buttons

        self.parent.show()

    def toggle_test_flag(self):
        self.parent.test_flag = not self.parent.test_flag
        if self.parent.test_flag:
            self.test_button.setText("Test (ON)")
        else:
            self.test_button.setText("Test (OFF)")

    def calibrate_height(self):
        self.parent.calibration = True
        self.parent.unassisted = False
        self.toggle_motor()
        self.switch_to_second_page()
        self.parent.start_trial()

    def start_unassisted_trial(self):
        self.parent.check_calibration()

        if self.parent.max_roll_angle is None:
            QtWidgets.QMessageBox.information(self.parent, "Error", "Please calibrate the height first.")
            return 
        else:
            self.parent.unassisted = True
            self.parent.calibration = False
            self.toggle_motor()
            self.switch_to_second_page()
            self.parent.start_trial()

    def start_trial(self):
        self.parent.check_calibration()
        self.parent.check_unassisted()

        if self.parent.max_roll_angle is None:
            QtWidgets.QMessageBox.information(self.parent, "Error", "Please calibrate the height first.")
            return
        elif self.parent.unassisted_mean is None:
            QtWidgets.QMessageBox.information(self.parent, "Error", "Please collect unpowered data first.")
            return
        else:
            self.parent.unassisted = False
            self.parent.calibration = False
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
