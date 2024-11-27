from PyQt5 import QtWidgets

class NoPlotUISetup:
    def __init__(self, parent):
        self.parent = parent
        self.motor_running = False  # Tracks motor state

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
        self.quit_button = QtWidgets.QPushButton("Quit")

        self.start_unassisted_button.clicked.connect(self.start_unassisted_trial)
        self.start_button.clicked.connect(self.start_trial)
        self.select_mode_button.clicked.connect(self.select_mode)
        self.quit_button.clicked.connect(self.on_quit)

        first_page_layout.addWidget(self.start_unassisted_button)
        first_page_layout.addWidget(self.start_button)
        first_page_layout.addWidget(self.select_mode_button)
        first_page_layout.addWidget(self.quit_button)

        # Second Page Buttons
        self.second_page_buttons = QtWidgets.QWidget()
        second_page_layout = QtWidgets.QVBoxLayout(self.second_page_buttons)

        self.motor_button = QtWidgets.QPushButton("Stop Motor")
        self.stop_emg_button = QtWidgets.QPushButton("Stop EMG Trial")
        self.quit_second_page_button = QtWidgets.QPushButton("Quit")

        self.motor_button.clicked.connect(self.toggle_motor)
        self.stop_emg_button.clicked.connect(self.stop_emg_trial)
        self.quit_second_page_button.clicked.connect(self.on_quit)

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
        self.switch_to_second_page()

    def start_trial(self):
        self.switch_to_second_page()

    def select_mode(self):
        QtWidgets.QMessageBox.information(self.parent, "Mode Selection", "Mode selection on Raspi initiated.")

    def on_quit(self):
        self.parent.close()

    def toggle_motor(self):
        if self.motor_running:
            self.motor_button.setText("Start Motor")
            # Add logic to stop the motor
            print("Motor stopped")
        else:
            self.motor_button.setText("Stop Motor")
            # Add logic to start the motor
            print("Motor started")
        self.motor_running = not self.motor_running

    def stop_emg_trial(self):
        self.switch_to_first_page()

    def switch_to_first_page(self):
        self.first_page_buttons.show()
        self.second_page_buttons.hide()

    def switch_to_second_page(self):
        self.first_page_buttons.hide()
        self.second_page_buttons.show()



# class NoPlotUISetup:
#     def __init__(self, parent):
#         self.parent = parent

#     def init_ui(self):
#         self.parent.setWindowTitle('EMG Data Collection')
#         self.parent.resize(1200, 200)

#         # Central widget
#         central_widget = QtWidgets.QWidget()
#         self.parent.setCentralWidget(central_widget)

#         # Layouts
#         main_layout = QtWidgets.QVBoxLayout(central_widget)
#         button_layout = QtWidgets.QHBoxLayout()

#         # Time display label
#         self.parent.time_label = QtWidgets.QLabel('Elapsed Time: 0.00 s')
#         self.parent.time_label.hide()
#         main_layout.addWidget(self.parent.time_label)

#         # Buttons
#         self.parent.start_unassisted_button = QtWidgets.QPushButton('Start Unassisted Trial')
#         self.parent.start_button = QtWidgets.QPushButton('Start Trial')
#         self.parent.stop_button = QtWidgets.QPushButton('Stop Trial')
#         self.parent.quit_button = QtWidgets.QPushButton('Quit')

#         self.parent.start_unassisted_button.clicked.connect(self.parent.start_unassisted_trial)
#         self.parent.start_button.clicked.connect(self.parent.start_trial)
#         self.parent.stop_button.clicked.connect(self.parent.stop_trial)
#         self.parent.quit_button.clicked.connect(self.parent.on_quit)

#         button_layout.addWidget(self.parent.start_unassisted_button)
#         button_layout.addWidget(self.parent.start_button)
#         button_layout.addWidget(self.parent.stop_button)
#         button_layout.addWidget(self.parent.quit_button)

#         # PyQtGraph Plot Widget
#         main_layout.addLayout(button_layout)

#         self.parent.show()

