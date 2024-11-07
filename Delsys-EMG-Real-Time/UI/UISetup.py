from PyQt5 import QtWidgets

class UISetup:
    def __init__(self, parent):
        self.parent = parent

    def init_ui(self):
        self.parent.setWindowTitle('EMG Data Collection')
        self.parent.resize(1200, 800)

        # Central widget
        central_widget = QtWidgets.QWidget()
        self.parent.setCentralWidget(central_widget)

        # Layouts
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        button_layout = QtWidgets.QHBoxLayout()

        # Time display label
        self.parent.time_label = QtWidgets.QLabel('Elapsed Time: 0.00 s')
        self.parent.time_label.hide()
        main_layout.addWidget(self.parent.time_label)

        # Buttons
        self.parent.start_unassisted_button = QtWidgets.QPushButton('Start Unassisted Trial')
        self.parent.start_button = QtWidgets.QPushButton('Start Trial')
        self.parent.stop_button = QtWidgets.QPushButton('Stop Trial')
        self.parent.autoscale_button = QtWidgets.QPushButton('Autoscale')
        self.parent.quit_button = QtWidgets.QPushButton('Quit')

        self.parent.start_unassisted_button.clicked.connect(self.parent.start_unassisted_trial)
        self.parent.start_button.clicked.connect(self.parent.start_trial)
        self.parent.stop_button.clicked.connect(self.parent.stop_trial)
        # **Corrected connection: Connect to plotter's autoscale_plots method**
        self.parent.autoscale_button.clicked.connect(self.parent.plotter.autoscale_plots)
        self.parent.quit_button.clicked.connect(self.parent.on_quit)

        button_layout.addWidget(self.parent.start_unassisted_button)
        button_layout.addWidget(self.parent.start_button)
        button_layout.addWidget(self.parent.stop_button)
        button_layout.addWidget(self.parent.autoscale_button)
        button_layout.addWidget(self.parent.quit_button)

        # PyQtGraph Plot Widget
        self.parent.plot_widget = QtWidgets.QWidget()
        main_layout.addWidget(self.parent.plot_widget)
        main_layout.addLayout(button_layout)

        self.parent.show()
