import numpy as np
import pandas as pd

from pathlib import Path
from enum import Enum

import RPi.GPIO as GPIO

from assistive_arm.motor_control import CubemarsMotor
from session_manager import SessionManager
from socket_server import SocketServer
from sts_control import calibrate_height, collect_unpowered_data, apply_simulation_profile


# Set options
np.set_printoptions(precision=3, suppress=True)

# Directory to directly save the logs to remote host, also need to add Host in ~/.ssh/config

class States(Enum):
    CALIBRATING = 1
    UNPOWERED_COLLECTION = 2
    HILO = 3
    EXIT = 0

    def __eq__(self, other):
        if isinstance(other, int):
            return self.value == other
        return False


if __name__ == "__main__":
    # Use Broadcom SOC Pin numbers
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(17, GPIO.IN)

    logging = True
    freq = 200

    subject_id = "P"
    subject_folder = Path(f"./subject_logs/subject_{subject_id}")
    session_manager = SessionManager(subject_id=subject_id)

    trigger_mode = "SOCKET" # TRIGGER, ENTER or SOCKET

    if trigger_mode == "SOCKET":
        socket_server = SocketServer()
    else:
        socket_server = None

    unadjusted_profile_dir = Path(f"./torque_profiles/simulation_profile_Camille_xy.csv")


    # Start the main interaction loop
    try:
        while True:
            # Display menu for user input
            print("\nOptions:")
            print("1 - Calibrate Height")
            print("2 - Collect unpowered data")
            print("3 - HILO")
            print("0 - Exit")

            # Get user's choice and handle it based on selection
            try:
                choice = int(input("Enter your choice: "))

                if choice == States.CALIBRATING:
                    with CubemarsMotor(motor_type="AK70-10", frequency=freq) as motor_1:
                        with CubemarsMotor(motor_type="AK60-6", frequency=freq) as motor_2:
                            calibrate_height(
                                motor_1=motor_1,
                                motor_2=motor_2,
                                freq=freq,
                                session_manager=session_manager,
                                profile_dir=unadjusted_profile_dir
                            )

                # If the device is not calibrated for the user's height, the user will not be able to collect data
                elif choice == States.UNPOWERED_COLLECTION and session_manager.theta_2_scaled is not None:
                    with CubemarsMotor(motor_type="AK70-10", frequency=freq) as motor_1:
                        with CubemarsMotor(motor_type="AK60-6", frequency=freq) as motor_2:
                            try:
                                collect_unpowered_data(
                                    motor_1=motor_1,
                                    motor_2=motor_2,
                                    freq=freq,
                                    session_manager=session_manager,
                                    profile_dir=unadjusted_profile_dir,
                                    mode=trigger_mode,
                                    server=socket_server
                                )
                            except FileNotFoundError as e:
                                print(e)
                                print("Returning to the main menu...")

                elif choice == States.HILO and session_manager.theta_2_scaled is not None:
                    with CubemarsMotor(motor_type="AK70-10", frequency=freq) as motor_1:
                        with CubemarsMotor(motor_type="AK60-6", frequency=freq) as motor_2:
                            # Get assistive profile from optimizer

                            # Adjust the profile to the height of the subject
                            try:
                                apply_simulation_profile(
                                    motor_1=motor_1,
                                    motor_2=motor_2,
                                    freq=freq,
                                    session_manager=session_manager,
                                    profile_dir=unadjusted_profile_dir,
                                    mode=trigger_mode,
                                    server=socket_server
                                )
                            except FileNotFoundError as e:
                                print(e)
                                print("Returning to the main menu...")

                elif choice == States.EXIT:
                    print("Exiting...")
                    if trigger_mode == "SOCKET":
                        socket_server.stop()
                    break

                if session_manager.theta_2_scaled is None:
                    print("Please calibrate the device height before proceeding.")

                # Optionally load a new profile if the user wants to switch profiles
                load_new_profile = input("Do you want to load a different profile? (y/any key to skip): ")
                if load_new_profile.lower() == "y":
                    profile_path = Path(input("Enter the path to the new profile: "))
                    try:
                        # Check if the profile path is valid
                        profile = pd.read_csv(profile_path, index_col="Percentage")
                        print("Valid profile directory.")
                        unadjusted_profile_dir = profile_path
                    except FileNotFoundError:
                        print("File not found. Please try again.")
                    except pd.errors.EmptyDataError:
                        print("Invalid CSV file. Please try again.")

            except ValueError:
                print("Invalid input. Please enter a number.")

    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected. Exiting...")

    finally:
        # Ensure cleanup of GPIO and socket server resources
        GPIO.cleanup()
        if trigger_mode == "SOCKET":
            socket_server.stop()
