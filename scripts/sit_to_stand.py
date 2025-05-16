import csv
from datetime import datetime
import os
import sys
import time
import zipfile
import numpy as np
import pandas as pd
import yaml
import re
from typing import Literal

from pathlib import Path
from enum import Enum

import RPi.GPIO as GPIO

from NeuroLocoMiddleware.SoftRealtimeLoop import SoftRealtimeLoop
from assistive_arm.motor_control import CubemarsMotor
from assistive_arm.robotic_arm import calculate_ee_pos, get_jacobian, get_target_torques

# Network socket server
import socket
import threading

# Configuration
HOST = '0.0.0.0'  # Listen on all available interfaces
PORT = 3003      # Arbitrary non-privileged port

# Set options
np.set_printoptions(precision=3, suppress=True)

# Directory to directly save the logs to remote host, also need to add Host in ~/.ssh/config
PROJECT_DIR_REMOTE = Path("/Users/filippo.mariani/Desktop/Universita/Harvard/Third_Arm_Data")

class States(Enum):
    CALIBRATING = 1
    ASSISTING = 2
    ASSIST_PROFILES = 3
    UNPOWERED_COLLECTION = 4
    EXIT = 0

    def __eq__(self, other):
        if isinstance(other, int):
            return self.value == other
        return False

class SocketServer:
    """Handles the socket connection and communication."""
    
    def __init__(self, host=HOST, port=PORT):
        self.host = host
        self.port = port
        self.collect_flag = False
        self.profile_name = None
        self.stop_server = False
        self.conn = None
        self.server_thread = threading.Thread(target=self.start_server)
        self.server_thread.start()

    def start_server(self):
        """Start the socket server and accept connections."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.bind((self.host, self.port))
            server_socket.listen()
            print(f"Server listening on {self.host}:{self.port}")

            while not self.stop_server:
                try:
                    server_socket.settimeout(1.0)  # Set a timeout to check stop_server periodically
                    self.conn, addr = server_socket.accept()
                    with self.conn:
                        print(f"Connected by {addr}")
                        while not self.stop_server:
                            data = self.conn.recv(1024)
                            if not data:
                                break
                            data_decoded = data.decode('utf-8', errors='replace')
                            self.process_data(data_decoded)
                except socket.timeout:
                    # Timeout reached, loop back to check stop_server
                    continue

    def process_data(self, data):
        """Process received data to control the session state."""
        if data == "Start":
            self.collect_flag = True
        elif data == "Stop":
            self.collect_flag = False
        elif data == "Kill":
            print("Closing connection...")
            if self.conn:
                self.conn.close()
            self.server_thread.join()
        elif "Profile" in data:
            self.profile_name = data.split(":")[1]

    def stop(self):
        """Stop the server."""
        self.stop_server = True
        self.collect_flag = False
        if self.conn:
            self.conn.close()
        if self.server_thread.is_alive():
            self.server_thread.join()



class SessionManager:
    """Handles session logging and data storage."""
    
    def __init__(self, subject_id):
        self.subject_folder = Path(f"./subject_logs/subject_{subject_id}")
        self.session_dir, self.session_remote_dir = self.set_up_logging_dir()

    def set_up_logging_dir(self):
        """Set up directories for logging."""
        current_date = datetime.now()
        session_dir = self.subject_folder / f"{current_date.strftime('%B_%d')}" / "Motor"
        session_dir.mkdir(parents=True, exist_ok=True)
        
        session_remote_dir = Path(f"{PROJECT_DIR_REMOTE}/subject_logs") / session_dir.relative_to("subject_logs")
        try:
            os.system(f"ssh macbook 'mkdir -p {session_remote_dir}'")
        except Exception as e:
            print(f"Error creating remote directory: {e}")
        
        return session_dir, session_remote_dir

    def save_log_or_delete(self, log_path: Path, successful: bool):
        """Save or delete log file based on success of the session."""
        if successful:
            print("\nSending logfile to Mac...")
            try:
                os.system(f"scp {log_path} macbook:{self.session_remote_dir}")
            except Exception as e:
                print(f"Error transferring file: {e}")
        else:
            print(f"Removing {log_path}")
            os.remove(log_path)

    def get_yaml_path(self, yaml_name: str) -> Path:
        yaml_file = f"{yaml_name}.yaml"
        return self.session_dir / yaml_file

            
def await_trigger_signal(mode: Literal["TRIGGER", "ENTER", "SOCKET"], server: SocketServer=None):
    """ Wait for trigger signal OR Enter to start recording """
    if mode == "ENTER": 
        input("\nPress Enter to start recording...")

    if mode == "TRIGGER":
        print("\nPress trigger to start recording P_EE...")
        while not GPIO.input(17):
            pass

    elif mode == "SOCKET" and server:
        print("\nWaiting for socket data to start recording")
        while not server.collect_flag:
            time.sleep(0.1)


def get_logger(log_name: str, session_manager: SessionManager, profile_details: list = None, server: SocketServer = None) -> tuple[Path, csv.writer]:
    """
    Set up a logger for various tasks in the script. Return the log file path and logger.

    Args:
        log_name (str): Name for the log file.
        session_manager (SessionManager): Instance managing the session directory.
        profile_details (list, optional): Details for logging [peak_time, peak_force]. Defaults to None.

    Returns:
        tuple[Path, csv.writer]: The path to the log file and the CSV writer instance.
    """
    
    # Variables to be logged
    logged_vars = ["Percentage", "target_tau_1", "measured_tau_1", "theta_1", "velocity_1", 
                   "target_tau_2", "measured_tau_2", "theta_2", "velocity_2", "EE_X", "EE_Y"]

    # Use session_manager to determine the next sample number
    if not server:
        sample_num = get_next_sample_number(session_manager.session_dir, log_name)
    else: 
        sample_num = get_next_sample_number(session_manager.session_dir, f"{log_name}_{server.profile_name}")
    
    # Format log file path
    if not server:
        log_file = f"{log_name}_{sample_num:02}.csv"
    else:
        log_file = f"{log_name}_{server.profile_name}_{sample_num:02}.csv"  # session_manager.profile_name
    log_path = session_manager.session_dir / log_file
    log_path.touch(exist_ok=True)  # Ensure the file is created

    # Set up the CSV writer
    with open(log_path, "w") as fd:
        writer = csv.writer(fd)
        
        # Optionally add profile details
        if profile_details:
            writer.writerow(["peak_time", profile_details[0]])
            writer.writerow(["peak_force", profile_details[1]])
        
        # Write headers
        writer.writerow(["time"] + logged_vars)

    # Open the CSV file for appending and set up the logger
    csv_file = open(log_path, "a").__enter__()
    task_logger = csv.writer(csv_file)

    return log_path, task_logger


def get_next_sample_number(session_dir: Path, log_name: str) -> int:
    pattern = re.compile(rf"{log_name}_(\d+).csv")
    max_sample = 0
    for file in session_dir.iterdir():
        match = pattern.match(file.name)
        if match:
            sample_number = int(match.group(1))
            max_sample = max(max_sample, sample_number)
    return max_sample + 1  # First sample will be 1 if no files match



def countdown(duration: int=3):
    for i in range(duration, 0, -1):
        print(f"Recording in {i} seconds...", end="\r")
        time.sleep(1)
    print("\nGO!")


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
        calibration_data["theta_2_values"] = [float(angle) for angle in theta_2]

        # Update and save profiles
        save_calibrated_profiles(
            theta_2_scaled=theta_2_scaled, profile_dir=profile_dir, 
            session_manager=session_manager, yaml_path=yaml_path, calibration_data=calibration_data
        )

    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Shutting down...")


def save_calibrated_profiles(theta_2_scaled, profile_dir, session_manager, yaml_path, calibration_data):
    """
    Save the calibrated profiles to both local and remote directories.

    Args:
        theta_2_scaled (pd.Series): Scaled theta_2 values.
        profile_dir (Path): Directory where the profiles are stored.
        session_manager (SessionManager): Instance for handling session directory and remote sync.
        yaml_path (Path): Path to the YAML file for calibration data.
        calibration_data (dict): Calibration data to save in YAML.
    """
    spline_path = Path("./torque_profiles")
    calibrated_dir = spline_path / "calibrated"
    calibrated_dir.mkdir(parents=True, exist_ok=True)

    # Zip and prepare for transfer
    zip_file_path = calibrated_dir / "calibrated_profiles.zip"
    with zipfile.ZipFile(zip_file_path, "w") as zip_file:
        for profile in spline_path.iterdir():
            if profile.suffix == ".csv" and "calibrated" not in profile.stem:
                # Read and calibrate profile data
                spline_profile = pd.read_csv(profile, index_col="Percentage")
                spline_profile['theta_2'] = theta_2_scaled
                
                # Save calibrated profile in the target directory
                calibrated_profile_path = calibrated_dir / f"{profile.stem}_calibrated.csv"
                spline_profile.to_csv(calibrated_profile_path, index=True)

                # Add calibrated profile to zip
                zip_file.write(calibrated_profile_path, calibrated_profile_path.name)

    # Transfer to remote host
    remote_path = session_manager.session_remote_dir / "calibrated"
    try:
        os.system(f"ssh macbook 'mkdir -p {remote_path}'")
    except Exception as e:
        print(f"Error creating remote directory: {e}")
        
    try:
        os.system(f"scp {zip_file_path} macbook:{remote_path}")
        os.system(f"ssh macbook 'unzip -oq {remote_path / zip_file_path.name} -d {remote_path}'")
        os.remove(zip_file_path)
    except Exception as e:
        print(f"Error transferring file: {e}")

    # Save calibration data in YAML and sync with the remote directory
    with open(yaml_path, "w") as f:
        yaml.dump(calibration_data, f)
    try:
        os.system(f"scp {yaml_path} macbook:{remote_path}")
    except Exception as e:
        print(f"Error transferring YAML file: {e}")


def assist_multiple_profiles(
        motor_1: CubemarsMotor,
        motor_2: CubemarsMotor,
        freq: int,
        session_manager: SessionManager,
        profile_dir: Path,
        mode: Literal["TRIGGER", "ENTER"],
        server: SocketServer):
    """
    Run assistance profiles across multiple configurations.

    Args:
        motor_1 (CubemarsMotor): First motor of the assistive arm.
        motor_2 (CubemarsMotor): Second motor of the assistive arm.
        freq (int): Control loop frequency (Hz).
        session_manager (SessionManager): Manages session and logging paths.
        profile_dir (Path): Directory containing profiles.
        mode (Literal["TRIGGER", "ENTER"]): Start trigger mode.
        server (SocketServer): Socket server instance for managing connection flags.
    """
    # Load spline profiles into a dictionary by peak time and peak force
    spline_profiles_path = profile_dir.parent / "calibrated"
    spline_dict = {}

    for path in spline_profiles_path.iterdir():
        if path.suffix == ".csv":
            peak_time = int(path.stem.split("_")[2])
            peak_force = int(path.stem.split("_")[5])

            if peak_time not in spline_dict:
                spline_dict[peak_time] = {}

            spline_dict[peak_time][peak_force] = pd.read_csv(path, index_col="Percentage")

    # Display the options for profile selection
    print("Available modes:\n")
    print("1 - Single profile")
    print("2 - All profiles for a given peak time")
    print("3 - All profiles for a given peak force")
    print("4 - All profiles")
    print("0 - Exit")

    chosen_mode = input("\nChoose mode: ")

    if chosen_mode == "1":
        # Run a single profile
        peak_times = sorted(spline_dict.keys())
        peak_forces = sorted(spline_dict[peak_times[0]].keys())

        print("Available peak times: [%]", peak_times)
        print("Available peak forces: [N]", peak_forces)

        peak_time = int(input("Enter peak time: "))
        peak_force = int(input("Enter peak force: "))
        profile = spline_dict[peak_time][peak_force]

        log_path, logger = get_logger(
            log_name=f"single_time_{peak_time}_force_{peak_force}",
            session_manager=session_manager,
            profile_details=[peak_time, peak_force],
            server=server
        )

        await_trigger_signal(mode=mode, server=server)
        control_loop_and_log(
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
        session_manager.save_log_or_delete(log_path=log_path)

    elif chosen_mode == "2":
        # Run all profiles for a given peak time
        peak_times = sorted(spline_dict.keys())
        peak_time = int(input("Enter peak time: "))
        profiles = spline_dict[peak_time]

        for peak_force, profile in profiles.items():
            log_path, logger = get_logger(
                log_name=f"peak_time_{peak_time}_force_{peak_force}",
                session_manager=session_manager,
                server=server
            )

            await_trigger_signal(mode=mode, server=server)
            control_loop_and_log(
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
            session_manager.save_log_or_delete(log_path=log_path)

    elif chosen_mode == "3":
        # Run all profiles for a given peak force
        peak_forces = sorted(spline_dict[sorted(spline_dict.keys())[0]].keys())
        peak_force = int(input("Enter peak force: "))

        for peak_time, profiles in spline_dict.items():
            profile = profiles[peak_force]
            log_path, logger = get_logger(
                log_name=f"assist_time_{peak_time}_force_{peak_force}",
                session_manager=session_manager,
                profile_details=[peak_time, peak_force],
                server=server
            )

            await_trigger_signal(mode=mode, server=server)
            control_loop_and_log(
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
            session_manager.save_log_or_delete(log_path=log_path)

    elif chosen_mode == "4":
        # Run all profiles for all peak times and forces
        for peak_time, profiles in spline_dict.items():
            for peak_force, profile in profiles.items():
                log_path, logger = get_logger(
                    log_name=f"assist_time_{peak_time}_force_{peak_force}",
                    session_manager=session_manager,
                    profile_details=[peak_time, peak_force],
                    server=server
                )

                await_trigger_signal(mode=mode, server=server)
                control_loop_and_log(
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
                session_manager.save_log_or_delete(log_path=log_path)

    elif chosen_mode == "0":
        print("Exiting assist multiple profiles mode...")


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
    # ./torque_profiles/simulation_profile_Camille.csv
    # ./torque_profiles/simulation_profile_Camille_scalex_scaley.csv
    # ./torque_profiles/simulation_profile_Camille_xy.csv
    # ./torque_profiles/simulation_profile_Camille_y.csv
    # ./torque_profiles/Camille_fitted.csv
    # ./torque_profiles/simulation_profile_Camille_scalex_scaley_fitted.csv
    # ./torque_profiles/simulation_profile_Camille_xy_fitted.csv
    # ./torque_profiles/Camille_y_fitted.csv

    # Start the main interaction loop
    try:
        while True:
            # Display menu for user input
            print("\nOptions:")
            print("1 - Calibrate Height")
            print("2 - Run Assistance")
            print("3 - Apply multiple assistance profiles")
            print("4 - Collect unpowered data")
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

                elif choice == States.ASSISTING:
                    with CubemarsMotor(motor_type="AK70-10", frequency=freq) as motor_1:
                        with CubemarsMotor(motor_type="AK60-6", frequency=freq) as motor_2:
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

                elif choice == States.ASSIST_PROFILES:
                    with CubemarsMotor(motor_type="AK70-10", frequency=freq) as motor_1:
                        with CubemarsMotor(motor_type="AK60-6", frequency=freq) as motor_2:
                            try:
                                assist_multiple_profiles(
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

                elif choice == States.UNPOWERED_COLLECTION:
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

                elif choice == States.EXIT:
                    print("Exiting...")
                    if trigger_mode == "SOCKET":
                        socket_server.stop()
                    break

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
