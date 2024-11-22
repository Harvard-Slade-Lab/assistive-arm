import csv
import os
import sys
import time
import zipfile
import numpy as np
import pandas as pd
import yaml
from typing import Literal

from pathlib import Path

import RPi.GPIO as GPIO

from NeuroLocoMiddleware.SoftRealtimeLoop import SoftRealtimeLoop
from assistive_arm.motor_control import CubemarsMotor
from assistive_arm.robotic_arm import calculate_ee_pos, get_target_torques
from session_manager import SessionManager, get_logger
from socket_server import SocketServer
from sts_utils import await_trigger_signal, countdown


def calibrate_height(
        motor_1: CubemarsMotor,
        motor_2: CubemarsMotor,
        freq: int,
        session_manager: SessionManager,
        profile_dir: Path):
    """
    Perform height calibration by running the sit-to-stand motion.

    Args:
        motor_1 (CubemarsMotor): The first motor for the assistive arm.
        motor_2 (CubemarsMotor): The second motor for the assistive arm.
        freq (int): The frequency for control loop (Hz).
        session_manager (SessionManager): Instance managing session directories and logging.
        profile_dir (Path): Path to the unadjusted profile file.
    """
    # Use session_manager to retrieve the YAML path for calibration data
    yaml_path = session_manager.get_yaml_path(yaml_name="device_height_calibration")

    # Load unadjusted profile data
    unadjusted_profile = pd.read_csv(profile_dir, index_col="Percentage")

    # Set up the real-time control loop
    loop = SoftRealtimeLoop(dt=1 / freq, report=False, fade=0)

    # Data for calibration
    calibration_data = dict()
    P_EE_values = []
    theta_2 = []
    start_time = 0

    try:
        motor_1.send_torque(desired_torque=0, safety=True)
        motor_2.send_torque(desired_torque=0, safety=True)

        input("\nPress Enter to start calibrating...")
        countdown(duration=3)

        print("Calibration started. Please perform the sit-to-stand motion.")
        print("Press Ctrl + C to stop recording.\n")

        for t in loop:
            P_EE = calculate_ee_pos(theta_1=motor_1.position, theta_2=motor_2.position)
            
            if not t < 0.5:
                P_EE_values.append(P_EE[0])  # Assuming x is the first element
                theta_2.append(motor_2.position)

            motor_1.send_torque(desired_torque=0, safety=True)
            motor_2.send_torque(desired_torque=0, safety=True)

            if t - start_time >= 0.05:
                print(f"P_EE x: {P_EE[0]}, y: {P_EE[1]}", end="\r")
                start_time = t

        print("\nRecording stopped. Processing data...\n")

        # Calibration calculations
        theta_2 = np.array(theta_2)
        new_max = theta_2.max() - 0.01
        new_min = theta_2.min() + 0.1

        original_max = unadjusted_profile.theta_2.max()
        original_min = unadjusted_profile.theta_2.min()

        scale = (new_max - new_min) / (original_max - original_min)
        theta_2_scaled = unadjusted_profile['theta_2'].apply(lambda x: new_min + (x - original_min) * scale)

        # Store calibration data
        calibration_data["new_range"] = {"min": float(new_min), "max": float(new_max)}
        calibration_data["theta_2_values"] = [float(angle) for angle in theta_2_scaled]

        remote_path = session_manager.session_remote_dir / "calibrated"

        # Save calibration data in YAML and sync with the remote directory
        with open(yaml_path, "w") as f:
            yaml.dump(calibration_data, f)
        try:
            os.system(f"scp {yaml_path} macbook:{remote_path}")
        except Exception as e:
            print(f"Error transferring YAML file: {e}")
        
    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Shutting down...")


# def save_calibrated_profiles(theta_2_scaled, profile_dir, session_manager, yaml_path, calibration_data):
#     """
#     Save the calibrated profiles to both local and remote directories.

#     Args:
#         theta_2_scaled (pd.Series): Scaled theta_2 values.
#         profile_dir (Path): Directory where the profiles are stored.
#         session_manager (SessionManager): Instance for handling session directory and remote sync.
#         yaml_path (Path): Path to the YAML file for calibration data.
#         calibration_data (dict): Calibration data to save in YAML.
#     """
#     spline_path = Path("./torque_profiles")
#     calibrated_dir = spline_path / "calibrated"
#     calibrated_dir.mkdir(parents=True, exist_ok=True)

#     # Zip and prepare for transfer
#     zip_file_path = calibrated_dir / "calibrated_profiles.zip"
#     with zipfile.ZipFile(zip_file_path, "w") as zip_file:
#         for profile in spline_path.iterdir():
#             if profile.suffix == ".csv" and "calibrated" not in profile.stem:
#                 # Read and calibrate profile data
#                 spline_profile = pd.read_csv(profile, index_col="Percentage")
#                 spline_profile['theta_2'] = theta_2_scaled
                
#                 # Save calibrated profile in the target directory
#                 calibrated_profile_path = calibrated_dir / f"{profile.stem}_calibrated.csv"
#                 spline_profile.to_csv(calibrated_profile_path, index=True)

#                 # Add calibrated profile to zip
#                 zip_file.write(calibrated_profile_path, calibrated_profile_path.name)

#     # Transfer to remote host
#     remote_path = session_manager.session_remote_dir / "calibrated"
#     try:
#         os.system(f"ssh macbook 'mkdir -p {remote_path}'")
#     except Exception as e:
#         print(f"Error creating remote directory: {e}")
        
#     try:
#         os.system(f"scp {zip_file_path} macbook:{remote_path}")
#         os.system(f"ssh macbook 'unzip -oq {remote_path / zip_file_path.name} -d {remote_path}'")
#         os.remove(zip_file_path)
#     except Exception as e:
#         print(f"Error transferring file: {e}")

#     # Save calibration data in YAML and sync with the remote directory
#     with open(yaml_path, "w") as f:
#         yaml.dump(calibration_data, f)
#     try:
#         os.system(f"scp {yaml_path} macbook:{remote_path}")
#     except Exception as e:
#         print(f"Error transferring YAML file: {e}")


def control_loop_and_log(
        motor_1: CubemarsMotor,
        motor_2: CubemarsMotor,
        logger: csv.writer,
        profile: pd.DataFrame,
        freq: int,
        mode: Literal["TRIGGER", "ENTER", "SOCKET"],
        apply_force: bool,
        log_path: Path,
        server: SocketServer,
        session_manager: SessionManager):

    print("Recording started. Please perform the sit-to-stand motion.")
    print("Press Ctrl + C or trigger to stop recording.\n")
    print_time = 0
    start_time = time.time()

    loop = SoftRealtimeLoop(dt=1 / freq, report=False, fade=0)
    success = True

    for t in loop:
        if mode == "TRIGGER" and GPIO.input(17):
            print("Stopped recording, exiting...")
            break
        elif mode == "SOCKET" and not server.collect_flag:
            print("Stopped recording, exiting...")
            break

        cur_time = time.time()

        if motor_1._emergency_stop or motor_2._emergency_stop:
            success = False
            break

        tau_1, tau_2, P_EE, index = get_target_torques(
            theta_1=motor_1.position,
            theta_2=motor_2.position,
            profiles=profile
        )
        
        if apply_force and t >= 0.1:
            motor_1.send_torque(desired_torque=tau_1, safety=False)
            motor_2.send_torque(desired_torque=tau_2, safety=False)
        else:
            motor_1.send_torque(desired_torque=0, safety=False) 
            motor_2.send_torque(desired_torque=0, safety=False)

        if t - print_time >= 0.05:
            print(f"{motor_1.type}: Angle: {np.rad2deg(motor_1.position):.3f} Torque: {motor_1.measured_torque:.3f}")
            print(f"{motor_2.type}: Angle: {np.rad2deg(motor_2.position):.3f} Torque: {motor_2.measured_torque:.3f}")
            print(f"Body height: {-P_EE[0]}")
            print(f"Movement: {index:.0f}%. tau_1: {tau_1}, tau_2: {tau_2}")
            sys.stdout.write(f"\x1b[4A\x1b[2K")
                    
            print_time = t
        logger.writerow([cur_time - start_time, index, tau_1, motor_1.measured_torque, motor_1.position, motor_1.velocity, tau_2, motor_2.measured_torque, motor_2.position, motor_2.velocity, P_EE[0], P_EE[1]])
    
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
        profile_dir: Path,
        mode: Literal["TRIGGER", "ENTER", "SOCKET"],
        server: SocketServer):
    """
    Apply a calibrated profile to simulate an assistive arm motion.

    Args:
        motor_1 (CubemarsMotor): First motor in the assistive arm.
        motor_2 (CubemarsMotor): Second motor in the assistive arm.
        freq (int): Control loop frequency (Hz).
        session_manager (SessionManager): Manages session and logging paths.
        profile_dir (Path): Directory path to the calibrated profile.
        mode (Literal["TRIGGER", "ENTER", "SOCKET"]): Start trigger mode.
        server (SocketServer): Socket server instance for communication.
    """
    adjusted_profile_dir = profile_dir.parent / f"calibrated/{profile_dir.stem}_calibrated.csv"
    print(f"Using profile: {adjusted_profile_dir}")
    
    # Check if the calibrated profile exists before proceeding
    if not adjusted_profile_dir.exists():
        raise FileNotFoundError(f"File {adjusted_profile_dir} does not exist. Please calibrate the device first.")

    profile = pd.read_csv(adjusted_profile_dir, index_col="Percentage")

    # Loop through multiple iterations of profile application
    for i in range(5):
        # Wait for trigger signal and start recording based on mode
        await_trigger_signal(mode=mode, server=server)
        
        # Set up logging for this iteration
        log_path, logger = get_logger(log_name=f"{profile_dir.stem}", session_manager=session_manager, server=server)
        
        # Countdown before starting
        countdown(duration=3)

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
                server=server,
                session_manager=session_manager
            )

            # Save or delete the log based on success
            session_manager.save_log_or_delete(log_path=log_path, successful=success)

        except KeyboardInterrupt:
            print("Keyboard interrupt detected. Shutting down...")
            break


def collect_unpowered_data(
        motor_1: CubemarsMotor,
        motor_2: CubemarsMotor,
        freq: int,
        session_manager: SessionManager,
        profile_dir: Path,
        mode: Literal["TRIGGER", "ENTER"],
        server: SocketServer):
    """
    Collect unpowered data for EMG synchronization.

    Args:
        motor_1 (CubemarsMotor): The first motor for the assistive arm.
        motor_2 (CubemarsMotor): The second motor for the assistive arm.
        freq (int): Control loop frequency (Hz).
        session_manager (SessionManager): Manages session and logging paths.
        profile_dir (Path): Path to the calibrated profile.
        mode (Literal["TRIGGER", "ENTER"]): Start trigger mode.
        server (SocketServer): Socket server instance for connection control.
    """
    adjusted_profile_dir = profile_dir.parent / f"calibrated/{profile_dir.stem}_calibrated.csv"
    
    if not adjusted_profile_dir.exists():
        raise FileNotFoundError(f"File {adjusted_profile_dir} does not exist. Please calibrate the device first.")
    
    profile = pd.read_csv(adjusted_profile_dir, index_col="Percentage")

    # Loop over the number of iterations for unpowered data collection
    for i in range(1, 6):  # Collect data across 5 iterations
        print(f"\nIteration number: {i}")
        success = False
        
        # Wait until successful data collection
        while not success:
            try:
                # Wait for trigger signal to start based on the selected mode
                await_trigger_signal(mode=mode, server=server)
                
                # Prepare the logger for the current iteration
                log_path, logger = get_logger(
                    log_name=f"unpowered_device",
                    session_manager=session_manager,
                    server=server
                )

                # Start countdown before data collection
                countdown(duration=3)
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
                    server=server,
                    session_manager=session_manager
                )

                # Save or delete the log based on the success of data collection
                session_manager.save_log_or_delete(log_path=log_path, successful=success)

            except Exception as e:
                print(f"An error occurred: {e}. Repeating the iteration...")

            except KeyboardInterrupt:
                print("Keyboard interrupt detected. Exiting...")
                return
