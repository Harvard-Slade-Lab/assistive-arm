import csv
import os
import sys
import time
import zipfile
import numpy as np
import pandas as pd
import yaml
import threading
from typing import Literal

from pathlib import Path

import RPi.GPIO as GPIO

from NeuroLocoMiddleware.SoftRealtimeLoop import SoftRealtimeLoop
from assistive_arm.motor_control import CubemarsMotor
from assistive_arm.robotic_arm import calculate_ee_pos, get_target_torques
from session_manager import SessionManager, get_logger
from socket_server import SocketServer
from read_imu import IMUReader
from sts_utils import await_trigger_signal, countdown


def calibrate_height(
        freq: int,
        session_manager: SessionManager,
        mode: Literal["TRIGGER", "ENTER", "SOCKET"],
        socket_server: SocketServer,
        imu_reader: IMUReader):
    """
    Perform height calibration by running the sit-to-stand motion.

    Args:
        freq (int): Control loop frequency (Hz).
        session_manager (SessionManager): Manages session and logging paths.
        mode (Literal["TRIGGER", "ENTER", "SOCKET"]): Start trigger mode.
        socket_server (SocketServer): Socket server instance for communication.
        imu_reader (IMUReader): IMU reader instance for roll angle data.
    """
    # Use session_manager to retrieve the YAML path for calibration data
    yaml_path = session_manager.yaml_path

    # Set up the real-time control loop
    loop = SoftRealtimeLoop(dt=1 / freq, report=False, fade=0)

    # Data for calibration
    calibration_data = dict()
    roll_angles = []
    start_time = 0

    try:
        await_trigger_signal(mode=mode, socket_server=socket_server)

        # Start reading from the IMU
        if imu_reader is not None:
            imu_reader.start_reading_imu_data()

        countdown(duration=3)

        print("Calibration started. Please perform the sit-to-stand motion.")

        sts_start = time.time()

        for t in loop:
            if socket_server.mode_flag or socket_server.kill_flag:
                print("Stopped recording, exiting...")
                break
            if mode == "TRIGGER" and GPIO.input(17):
                print("Stopped recording, exiting...")
                break
            elif mode == "SOCKET" and not socket_server.collect_flag:
                print("Stopped recording, exiting...")
                break
            
            with threading.Lock():
                if imu_reader is not None:
                    roll_angle = imu_reader.imu_data.pitch
                else:
                    roll_angle = socket_server.roll_angle

            if roll_angle is not None:
                roll_angles.append(roll_angle)

            if t - start_time >= 0.05:
                print(f"Roll angle: {roll_angle}", end="\r")
                start_time = t
            
            # Wait to avoid duplicates
            # time.sleep(0.01)

        # Stop the IMU reader
        if imu_reader is not None:
            imu_reader.stop_reading_imu_data()

        sts_duration = time.time() - sts_start

        print("\nRecording stopped. Processing data...\n")

        # Calibration calculations
        roll_angles = np.array(roll_angles)
        new_max = roll_angles.max()
        new_min = roll_angles.min()

        # Calculate the number of entries
        num_entries = int(sts_duration * freq)

        # Store calibration data
        calibration_data["new_range"] = {"min": float(new_min), "max": float(new_max)}
        # Generate the roll angles array
        calibration_data["roll_angles"] = np.linspace(new_min, new_max, num=num_entries).tolist()
        # Generate the percentage array
        calibration_data["Percentage"] = np.linspace(0, 100, num=num_entries).tolist()

        remote_path = session_manager.session_remote_dir

        # Save calibration data in YAML and sync with the remote directory
        with open(yaml_path, "w") as f:
            yaml.dump(calibration_data, f)
        try:
            os.system(f"scp {yaml_path} macbook:{remote_path}")
        except Exception as e:
            print(f"Error transferring YAML file: {e}")
        
    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Shutting down...")

def control_loop_and_log(
        motor_1: CubemarsMotor,
        motor_2: CubemarsMotor,
        logger: csv.writer,
        profile: pd.DataFrame,
        freq: int,
        mode: Literal["TRIGGER", "ENTER", "SOCKET"],
        apply_force: bool,
        log_path: Path,
        socket_server: SocketServer,
        imu_reader: IMUReader,
        session_manager: SessionManager):
    
    # Clear motor buffers
    # Check motor_1.position_buffer -> works
    motor_1.clear_buffers()
    motor_2.clear_buffers()

    print("Recording started. Please perform the sit-to-stand motion.")
    print("Press Ctrl + C or trigger to stop recording.\n")
    print_time = 0
    start_time = time.time()

    loop = SoftRealtimeLoop(dt=1 / freq, report=False, fade=0)
    success = True
    printed = False
    increment = 0.05
    scale_start_torque = increment

    for t in loop:
        if socket_server.mode_flag or socket_server.kill_flag:
            print("Stopped recording, exiting...")
            break
        if mode == "TRIGGER" and GPIO.input(17):
            print("Stopped recording, exiting...")
            break
        elif mode == "SOCKET" and not socket_server.collect_flag:
            print("Stopped recording, exiting...")
            break

        cur_time = time.time()

        if motor_1._emergency_stop or motor_2._emergency_stop:
            success = False
            break

        # Maybe want to move this into the function
        with threading.Lock():
            if imu_reader is not None:
                roll_angle = imu_reader.imu_data.pitch
            else:
                roll_angle = socket_server.roll_angle

        tau_1, tau_2, P_EE, index = get_target_torques(
            theta_1=motor_1.position,
            theta_2=motor_2.position,
            current_roll_angle=roll_angle,
            profiles=profile
        )

        if apply_force and t >= 0.1:
            if not printed:
                print("\nGO!")
                printed = True
            # Continue from the start torque
            # tau_1 = max(tau_1, 1)
            # tau_2 = max(tau_2, -1)
            motor_1.send_torque(desired_torque=tau_1, safety=False)
            motor_2.send_torque(desired_torque=tau_2, safety=False)
        # Start applying a small torque to get ready and avoid the initial jerk
        # elif apply_force and t < 0.2:
        #     # Cap start torque at 1 Nm
        #     start_torque = min(scale_start_torque, 1)
        #     motor_1.send_torque(desired_torque=start_torque, safety=False) 
        #     motor_2.send_torque(desired_torque=-start_torque, safety=False)
        #     scale_start_torque += increment
        else:
            if not printed:
                print("\nGO!")
                printed = True
            motor_1.send_torque(desired_torque=0, safety=False)
            motor_2.send_torque(desired_torque=0, safety=False)

        if t - print_time >= 0.05:
            print(f"{motor_1.type}: Angle: {np.rad2deg(motor_1.position):.3f} Torque: {motor_1.measured_torque:.3f} Sent Torque: {tau_1}")
            print(f"{motor_2.type}: Angle: {np.rad2deg(motor_2.position):.3f} Torque: {motor_2.measured_torque:.3f} Sent Torque: {tau_2}")
            print(f"Body height: {-P_EE[0]}")
            print(f"Movement: {index:.0f}%. tau_1: {tau_1}, tau_2: {tau_2}")
            sys.stdout.write(f"\x1b[4A\x1b[2K")
                    
            print_time = t
        logger.writerow([cur_time - start_time, index, roll_angle, tau_1, motor_1.measured_torque, motor_1.position, motor_1.velocity, tau_2, motor_2.measured_torque, motor_2.position, motor_2.velocity, P_EE[0], P_EE[1]])
    del loop

    motor_1.send_torque(desired_torque=0, safety=False)
    motor_2.send_torque(desired_torque=0, safety=False)

    if not success:
        motor_1._emergency_stop = False
        motor_2._emergency_stop = False
        print("\nSomething went wrong. Repeating the iteration...")

    session_manager.save_log_or_delete(log_path=log_path, successful=success)

    return success


def apply_simulation_profile(
        motor_1: CubemarsMotor,
        motor_2: CubemarsMotor,
        freq: int,
        session_manager: SessionManager,
        profile: pd.DataFrame,
        profile_name: str,
        mode: Literal["TRIGGER", "ENTER", "SOCKET"],
        socket_server: SocketServer,
        imu_reader: IMUReader):
    """
    Apply a calibrated profile to simulate an assistive arm motion.

    Args:
        motor_1 (CubemarsMotor): First motor in the assistive arm.
        motor_2 (CubemarsMotor): Second motor in the assistive arm.
        freq (int): Control loop frequency (Hz).
        session_manager (SessionManager): Manages session and logging paths.
        profile: Most recent profile.
        profile_name (str): Name of the profile to apply.
        mode (Literal["TRIGGER", "ENTER", "SOCKET"]): Start trigger mode.
        server (SocketServer): Socket server instance for communication.
        imu_reader (IMUReader): IMU reader instance for roll angle data.
    """

    # Wait for trigger signal and start recording based on mode
    await_trigger_signal(mode=mode, socket_server=socket_server)

    if socket_server is not None:
        if socket_server.mode_flag or socket_server.kill_flag:
            return
        
    # Start reading from the IMU
    if imu_reader is not None:
        imu_reader.start_reading_imu_data()

    # Set up logging for this iteration
    log_path, logger = get_logger(log_name=f"{profile_name}", session_manager=session_manager, socket_server=socket_server)
    
    # Countdown before starting
    countdown(duration=1)

    try:
        # Run the control loop, passing session_manager and server
        success = control_loop_and_log(
            motor_1=motor_1,
            motor_2=motor_2,
            logger=logger,
            profile=profile,
            freq=freq,
            mode=mode,
            apply_force=True,
            log_path=log_path,
            socket_server=socket_server,
            imu_reader=imu_reader,
            session_manager=session_manager
        )

        # Save or delete the log based on success
        session_manager.save_log_or_delete(log_path=log_path, successful=success)

        # Stop reading from the IMU
        if imu_reader is not None:
            imu_reader.stop_reading_imu_data()

    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Stopping...")


def collect_unpowered_data(
        motor_1: CubemarsMotor,
        motor_2: CubemarsMotor,
        freq: int,
        iterations: int,
        session_manager: SessionManager,
        mode: Literal["TRIGGER", "ENTER"],
        socket_server: SocketServer,
        imu_reader: IMUReader):
    """
    Collect unpowered data for EMG synchronization.

    Args:
        motor_1 (CubemarsMotor): The first motor for the assistive arm.
        motor_2 (CubemarsMotor): The second motor for the assistive arm.
        freq (int): Control loop frequency (Hz).
        session_manager (SessionManager): Manages session and logging paths.
        mode (Literal["TRIGGER", "ENTER"]): Start trigger mode.
        socket_server (SocketServer): Socket server instance for connection control.
        imu_reader (IMUReader): IMU reader instance for roll angle data.
    """ 
    profile = session_manager.roll_angles
    # Add columns with zero force
    profile["force_X"] = 0
    profile["force_Y"] = 0

    i = 1

    # Loop over the number of iterations for unpowered data collection
    while i <= iterations:  # Collect data across 5 iterations
        print(f"\nIteration number: {i}")
        success = False
        profile_name = socket_server.profile_name
        
        # Wait until successful data collection
        while not success:
            try:
                # Wait for trigger signal to start based on the selected mode, or kill the process/exit
                await_trigger_signal(mode=mode, socket_server=socket_server)
                # Check if the server is in a mode that requires stopping the process
                if socket_server is not None:
                    if socket_server.mode_flag or socket_server.kill_flag:
                        return
                    
                # Start reading from the IMU
                if imu_reader is not None:
                    imu_reader.start_reading_imu_data()
                
                # Prepare the logger for the current iteration
                log_path, logger = get_logger(
                    log_name=f"unpowered_device_tag_{profile_name}_iteration_{i}",
                    session_manager=session_manager,
                    socket_server=socket_server
                )

                # Start countdown before data collection
                countdown(duration=1)
                print(f"Recording to {log_path}")

                # Start the control loop for data collection without force application
                success = control_loop_and_log(
                    motor_1=motor_1,
                    motor_2=motor_2,
                    logger=logger,
                    profile=profile,
                    freq=freq,
                    mode=mode,
                    apply_force=False,
                    log_path=log_path,
                    socket_server=socket_server,
                    imu_reader=imu_reader,
                    session_manager=session_manager
                )

                # Save or delete the log based on the success of data collection
                session_manager.save_log_or_delete(log_path=log_path, successful=success)

                # Stop reading from the IMU
                if imu_reader is not None:
                    imu_reader.stop_reading_imu_data()

            except Exception as e:
                print(f"An error occurred: {e}. Repeating the iteration...")

            except KeyboardInterrupt:
                print("Keyboard interrupt detected. Exiting...")
                return
        # Will miss repetition if last iteration fails (maybe add getting the profile tag)
        if socket_server.repeat_flag:
            print("Iteration has to be repeated.")
            socket_server.repeat_flag = False
        else:
            i += 1
