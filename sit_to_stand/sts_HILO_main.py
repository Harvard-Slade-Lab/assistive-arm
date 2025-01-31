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
    HILO = 3
    EXIT = 0

    def __eq__(self, other):
        if isinstance(other, int):
            return self.value == other
        return False


if __name__ == "__main__":

    subject_id = "Nathan"
    subject_folder = Path(f"./subject_logs/subject_{subject_id}")
    session_manager = SessionManager(subject_id=subject_id)

    trigger_mode = "SOCKET" # TRIGGER, ENTER or SOCKET

    # HYPERPARMAETERS
    kappa = 2.5
    # Chose how much exploration is done
    exploration_iterations = 5
    # Chose how many unassisted iterations
    iterations_unassisted = 10
    # Chose how many repetitions for each condition
    iterations_per_parameter_set = 3
    # Peak force
    max_force = 55
    # Scale factor for force in x
    scale_factor_x = 2/3
    # Maximum length (currently not used)
    max_time = 360
    # Minimum width of the profile in percentage of the total time
    minimum_width_p = 0.2

    # Flag to select, if the Gaussian Process should be informed by a parameter set, based on the simulation
    informed = True

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
            print("3 - HILO")
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
                            iterations=iterations_unassisted,
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

                elif choice == States.HILO and session_manager.load_device_height_calibration() is not None:
                    can_bus, motor_1, motor_2 = setup_can_and_motors()
                    profile_optimizer = ForceProfileOptimizer(
                            motor_1=motor_1,
                            motor_2=motor_2,
                            kappa=kappa,
                            freq = freq, 
                            iterations = iterations_per_parameter_set,
                            session_manager = session_manager, 
                            trigger_mode = trigger_mode, 
                            socket_server = socket_server, 
                            imu_reader = imu_reader,
                            max_force=max_force,
                            scale_factor_x=scale_factor_x,
                            max_time=max_time,
                            minimum_width_p=minimum_width_p,
                        )
                        
                    # Explorate the space exploration iterations - iterations done, so they are not done again when reloading
                    if exploration_iterations > len(profile_optimizer.optimizer.space):
                        for exploration_iteration in range(exploration_iterations - len(profile_optimizer.optimizer.space)):
                            if socket_server.mode_flag or socket_server.kill_flag:
                                break
                            else:
                                profile_optimizer.explorate()
                    
                    # Add informed profiles to the optimizer
                    if informed:
                        profile_optimizer.informed_optimization()
                        
                    # Optimize until the server stops (Kill command is sent) or the user exits
                    while not socket_server.stop_server and not socket_server.kill_flag and not socket_server.mode_flag:
                        profile_optimizer.optimize()

                    profile_optimizer.log_to_remote()

                    shutdown_can_and_motors(can_bus, motor_1, motor_2)
                    time.sleep(1)

                elif choice == States.EXIT or socket_server.kill_flag:
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
