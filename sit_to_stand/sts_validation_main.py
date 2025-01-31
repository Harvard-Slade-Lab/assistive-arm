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

# This script is to validate the performance of single profiles
# Ideally the person doing the validation did the optimization


# Set options
np.set_printoptions(precision=3, suppress=True)


class States(Enum):
    CALIBRATING = 1
    UNPOWERED_COLLECTION = 2
    RUNForwards = 3
    RUNBackwards = 4
    EXIT = 0

    def __eq__(self, other):
        if isinstance(other, int):
            return self.value == other
        return False


if __name__ == "__main__":

    subject_id = "Validation"
    subject_folder = Path(f"./subject_logs/subject_{subject_id}")
    session_manager = SessionManager(subject_id=subject_id)

    validation_profiles_path = "./sit_to_stand/validation_profiles"
    # Read all the profiles and add them to a list
    profiles = {}
    for file in os.listdir(validation_profiles_path):
        if file.endswith(".csv"):
            profile = pd.read_csv(os.path.join(validation_profiles_path, file))
            profile["percentage"] = np.linspace(0, 100, len(profile))
            profile.set_index("percentage", inplace=True)
            profiles[file] = profile

    trigger_mode = "SOCKET" # TRIGGER, ENTER or SOCKET

    # Chose how many unassisted iterations
    iterations_unassisted = 10
    # Chose how many repetitions for each condition
    iterations_per_condition = 10

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

    # Start the main interaction loopx
    try:
        while True:
            # Reset the mode flag for the socket server
            if trigger_mode == "SOCKET":
                socket_server.mode_flag = False

            # Display menu for user input
            print("\nOptions:")
            print("1 - Calibrate Height")
            print("2 - Collect unpowered data")
            print("3 - Run Forwards")
            print("4 - Run Backwards")
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
                    collect_unpowered_data(
                        motor_1=motor_1,
                        motor_2=motor_2,
                        freq=freq,
                        iterations=iterations_unassisted,
                        session_manager=session_manager,
                        mode=trigger_mode,
                        socket_server=socket_server,
                        imu_reader=imu_reader
                    )
                    shutdown_can_and_motors(can_bus, motor_1, motor_2)
                    time.sleep(1)

                elif choice == States.RUNForwards and session_manager.load_device_height_calibration() is not None: 
                    can_bus, motor_1, motor_2 = setup_can_and_motors()
                    for profile_name, profile in profiles.items():
                        for i in range(iterations_per_condition):
                            # Wait to get new name
                            time.sleep(0.5)
                            current_profile_name = socket_server.profile_name
                            if "Profile_" in profile_name:
                                parts = profile_name.split("Profile_")
                                if len(parts) == 2:
                                    # Insert the current profile name after 'Profile_'
                                    new_filename = f"{parts[0]}Tag_{current_profile_name}"
                            else:
                                new_filename = profile_name.replace(".csv", f"_Tag_{current_profile_name}")

                            print(f"Running profile {profile_name}")
                            apply_simulation_profile(
                                motor_1=motor_1,
                                motor_2=motor_2,
                                freq=freq,
                                session_manager=session_manager,
                                profile = profile,
                                profile_name = new_filename,
                                mode=trigger_mode,
                                socket_server=socket_server,
                                imu_reader=imu_reader
                            )
                    shutdown_can_and_motors(can_bus, motor_1, motor_2)
                    time.sleep(1)

                elif choice == States.RUNBackwards and session_manager.load_device_height_calibration() is not None:
                    can_bus, motor_1, motor_2 = setup_can_and_motors()
                    # Go through profiles in inverted order
                    for profile_name, profile in reversed(profiles.items()):
                        for i in range(iterations_per_condition):
                            # Wait to get new name
                            time.sleep(0.5)  
                            current_profile_name = socket_server.profile_name
                            if "Profile_" in profile_name:
                                parts = profile_name.split("Profile_")
                                if len(parts) == 2:
                                    # Insert the current profile name after 'Profile_'
                                    new_filename = f"{parts[0]}Tag_{current_profile_name}"
                            else:
                                new_filename = profile_name.replace(".csv", f"_Tag_{current_profile_name}")

                            print(f"Running profile {profile_name}")
                            apply_simulation_profile(
                                motor_1=motor_1,
                                motor_2=motor_2,
                                freq=freq,
                                session_manager=session_manager,
                                profile = profile,
                                profile_name = new_filename,
                                mode=trigger_mode,
                                socket_server=socket_server,
                                imu_reader=imu_reader
                            )
                    shutdown_can_and_motors(can_bus, motor_1, motor_2)
                    time.sleep(1)

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
