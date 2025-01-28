import numpy as np
import pandas as pd
import time
import os
import can

from pathlib import Path
from enum import Enum

import RPi.GPIO as GPIO

from assistive_arm.motor_control import CubemarsMotor, setup_can_and_motors, shutdown_can_and_motors
from bayesian_optimization import ForceProfileOptimizer
from session_manager import SessionManager
from socket_server import SocketServer
from read_imu import IMUReader
from sts_control import calibrate_height, collect_unpowered_data, apply_simulation_profile


# Set options
np.set_printoptions(precision=3, suppress=True)


class States(Enum):
    CALIBRATING = 1
    UNPOWERED_COLLECTION = 2
    RUN = 3
    EXIT = 0

    def __eq__(self, other):
        if isinstance(other, int):
            return self.value == other
        return False


if __name__ == "__main__":

    subject_id = "M"
    subject_folder = Path(f"./subject_logs/subject_{subject_id}")
    session_manager = SessionManager(subject_id=subject_id)

    trigger_mode = "ENTER" # TRIGGER, ENTER or SOCKET


    # Chose how many calibration iterations
    iterations_calibration = 10
    # Chose how many repetitions for each condition
    iterations_per_parameter_set = 3

    # Flag to decide wether the emg IMU (True) or the wired IMU should be used for control
    emg_control = False

    # Initialize the IMU reader
    if not emg_control:
        imu_reader = IMUReader()
    else:
        imu_reader = None

    # Use Broadcom SOC Pin numbers
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(17, GPIO.IN)

    logging = True
    freq = 200

    if trigger_mode == "SOCKET":
        socket_server = SocketServer()
    else:
        socket_server = None

    # High force
    # profile_path = "subject_logs/subject_Nathan/January_28/Motor/profiles/profile_t11_1922_f11_23_t21_388_t22_1387_t23_1633_f21_45_Profile_20250128104000.csv"
    # Low Force
    profile_path = "subject_logs/subject_Nathan/January_28/Motor/profiles/profile_t11_1895_f11_33_t21_435_t22_1419_t23_1687_f21_11_Profile_20250128112409.csv"

    profile_name = "try"

    base_profile = pd.read_csv(profile_path)

    # Start the main interaction loop
    try:
        while True:
            # Reset the mode flag for the socket server
            if trigger_mode == "SOCKET":
                socket_server.mode_flag = False

            # Display menu for user input
            print("\nOptions:")
            print("1 - Calibrate Height")
            print("2 - Collect unpowered data")
            print("3 - Run Profile")
            print("0 - Exit")

            # Get user's choice and handle it based on selection
            try:
                choice = int(input("Enter your choice: "))

                if choice == States.CALIBRATING:
                    can_bus, motor_1, motor_2 = setup_can_and_motors()
                    calibrate_height(
                        freq=freq,
                        session_manager=session_manager,
                        mode=trigger_mode,
                        socket_server=socket_server,
                        imu_reader=imu_reader
                    )
                    shutdown_can_and_motors(can_bus, motor_1, motor_2)
                    time.sleep(1)

                # If the device is not calibrated for the user's height, the user will not be able to collect data
                elif choice == States.UNPOWERED_COLLECTION and session_manager.load_device_height_calibration() is not None:
                    can_bus, motor_1, motor_2 = setup_can_and_motors()
                    try:
                        collect_unpowered_data(
                            motor_1=motor_1,
                            motor_2=motor_2,
                            freq=freq,
                            iterations=iterations_calibration,
                            session_manager=session_manager,
                            mode=trigger_mode,
                            socket_server=socket_server,
                            imu_reader=imu_reader
                        )
                    except FileNotFoundError as e:
                        print(e)
                        print("Returning to the main menu...")
                    shutdown_can_and_motors(can_bus, motor_1, motor_2)
                    time.sleep(1)

                # elif choice == States.RUN and session_manager.load_device_height_calibration() is not None:
                elif choice == States.RUN and session_manager.load_device_height_calibration() is not None: 
                    can_bus, motor_1, motor_2 = setup_can_and_motors()
                    apply_simulation_profile(
                        motor_1=motor_1,
                        motor_2=motor_2,
                        freq=freq,
                        session_manager=session_manager,
                        profile = base_profile,
                        profile_name = profile_name,
                        mode=trigger_mode,
                        socket_server=socket_server,
                        imu_reader=imu_reader
                    )
                            

                    shutdown_can_and_motors(can_bus, motor_1, motor_2)

                elif choice == States.EXIT:
                    print("Exiting...")
                    if trigger_mode == "SOCKET":
                        socket_server.stop()
                    break

                if session_manager.roll_angles is None:
                    print("Please calibrate the device height before proceeding.")

            except ValueError:
                print("Invalid input. Please enter a number.")

    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected. Exiting...")

    finally:
        # Ensure cleanup of GPIO and socket server resources
        GPIO.cleanup()
        if trigger_mode == "SOCKET":
            socket_server.stop()
