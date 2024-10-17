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
PORT = 3000      # Arbitrary non-privileged port

# Set options
np.set_printoptions(precision=3, suppress=True)

# Use Broadcom SOC Pin numbers
GPIO.setmode(GPIO.BCM)
GPIO.setup(17, GPIO.IN)

# Directory to directly save the logs to remote host, also need to add Host in ~/.ssh/config
PROJECT_DIR_REMOTE = Path("/Users/nathanirniger/Desktop/MA/Project/Code/assistive-arm")

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
    

def await_trigger_signal(mode: Literal["TRIGGER", "ENTER", "SOCKET"], conn: socket.socket=None):
    """ Wait for trigger signal OR Enter to start recording """
    if mode == "ENTER": 
        input("\nPress Enter to start recording...")

    if mode == "TRIGGER":
        print("\nPress trigger to start recording P_EE...")
        while not GPIO.input(17):
            pass

    elif mode == "SOCKET":
        print("\nWaiting for socket data to start recording")
        if collect_flag == True:
            print("Start recording")


def start_server():
    """ Start a TCP socket server and accept a connection. """
    global collect_flag
    collect_flag = False
    global score
    score = None

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((HOST, PORT))
        server_socket.listen()
        print(f"Server listening on {HOST}:{PORT}")

        conn, addr = server_socket.accept()
        with conn:
            print(f"Connected by {addr}")
            while True:
                data = conn.recv(1024)  # Buffer size of 1024 bytes

                data_decoded = data.decode('utf-8', errors='replace')

                if data_decoded == "Start":
                    print("Start recording")
                    collect_flag = True
                elif data_decoded == "Stop":
                    print("Stop recording")
                    collect_flag = False
                    break
                elif not data:
                    break
                else:
                    # Process the received data (print, save, etc.)
                    print("Received score:", data_decoded)
                    score = data_decoded

def save_log_or_delete(remote_dir: Path, log_path: Path, successful: bool=False):
    print("\n\n\n\n")
    # If next iteration starts to soon -> add time delay (due to trigger signal)
    if successful:
        print("\nSending logfile to Mac...")
        print("log file: ", log_path)
        os.system(f"scp {log_path} macbook:{remote_dir}")
    else:
        print(f"Removing {log_path}")
        os.remove(log_path)


def get_logger(log_name: str, session_dir: Path, profile_details: list=None) -> tuple[Path, csv.writer]:
    """ Set up logger for the various task in the script. Return the necessary paths

    Args:
        log_name (str): log name. If it exists, a number will be added in front of it.
        session_dir (Path): session directory where we store the logs
        profile_details (list, optional): [peak_time, peak_force]. Defaults to None.

    Returns:
        tuple[Path, csv.writer]: log_path, task_logger
    """
    
    logged_vars = ["Percentage", "target_tau_1", "measured_tau_1", "theta_1", "velocity_1", "target_tau_2", "measured_tau_2", "theta_2", "velocity_2", "EE_X", "EE_Y"]

    sample_num = get_next_sample_number(session_dir=session_dir, log_name=log_name)
    log_file = f"{log_name}_{sample_num:02}.csv"
    log_path = session_dir / log_file
    log_path.touch(exist_ok=True)

    with open(log_path, "w") as fd:
        writer = csv.writer(fd)
        if profile_details:
            writer.writerow(["peak_time", profile_details[0]])
            writer.writerow(["peak_force", profile_details[1]])
        writer.writerow(["time"] + logged_vars)

    csv_file = open(log_path, "a").__enter__()
    task_logger = csv.writer(csv_file)

    return log_path, task_logger


def get_yaml_path(yaml_name: str, session_dir: Path) -> Path:
    yaml_file = f"{yaml_name}.yaml"
    yaml_path = session_dir / yaml_file

    return yaml_path


def get_next_sample_number(session_dir: Path, log_name: str) -> int:
    """
    Get the next sample number for the log file.

    :param session_dir: The directory where log files are stored.
    :param log_name: The base name of the log files.
    :return: The next sample number.
    """
    # Regular expression to match files and extract sample number
    pattern = re.compile(rf"{log_name}_(\d+).csv")

    # Find the highest sample number in existing files
    max_sample = 0
    for file in session_dir.iterdir():
        match = pattern.match(file.name)
        if match:
            sample_number = int(match.group(1))
            max_sample = max(max_sample, sample_number)

    # Return the next sample number
    return max_sample + 1


def set_up_logging_dir(subject_folder: Path) -> tuple:
    """Set up logger for the various task in the script. Return the necessary paths
    for storing and manipulating log files.

    Args:
        logged_vars (list[str]): variables that we want to log

    Returns:
        tuple: task_logger, remote_path, log_path, log_stem
    """
    current_date = datetime.now()
    month_name = current_date.strftime("%B")
    day = current_date.strftime("%d")

    session_dir = subject_folder / f"{month_name}_{day}"
    session_dir.mkdir(parents=True, exist_ok=True)

    session_remote_dir = Path(f"{PROJECT_DIR_REMOTE}/subject_logs/") / session_dir.relative_to("subject_logs")
    os.system(f"ssh macbook 'mkdir -p {session_remote_dir}'")
    
    return session_dir, session_remote_dir


def countdown(duration: int=3):
    for i in range(duration, 0, -1):
        print(f"Recording in {i} seconds...", end="\r")
        time.sleep(1)
    print("\nGO!")


def calibrate_height(motor_1: CubemarsMotor, motor_2: CubemarsMotor, freq: int, session_dir: Path, remote_dir: Path, profile_dir: Path):
    yaml_path = get_yaml_path(yaml_name="device_height_calibration", session_dir=session_dir)

    unadjusted_profile = pd.read_csv(profile_dir, index_col="Percentage")

    loop = SoftRealtimeLoop(dt=1 / freq, report=False, fade=0)

    calibration_data = dict()

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
        del loop

        print("\n\n\n\n")
        print("Recording stopped. Processing data...\n")

        # Add offset to ensure that when the subject stands, 0 or 100% will be reached
        theta_2 = np.array(theta_2)
        new_max = theta_2.max() - 0.01
        new_min = theta_2.min() + 0.1

        original_max = unadjusted_profile.theta_2.max()
        original_min = unadjusted_profile.theta_2.min()

        scale = (new_max - new_min) / (original_max - original_min)

        theta_2_scaled = unadjusted_profile['theta_2'].apply(lambda x: new_min + (x - original_min) * scale)

        print(f"Estimated duration: {len(theta_2) / freq}s")
        
        print("Calibration completed. New range: ")
        print(f"Old 0% STS: {unadjusted_profile.theta_2.max()} 0%: {new_max}")
        print(f"STS 100%: {unadjusted_profile.theta_2.min()} 100%: {new_min}\n")

        calibration_data["new_range"] = {"min": float(new_min), "max": float(new_max)}
        calibration_data["theta_2_values"] = [float(angle) for angle in theta_2]

        # Apply calibrated theta to all spline profiles
        spline_path = Path("./torque_profiles/spline_profiles")
        # Create directory on remote host
        os.system(f"ssh macbook 'mkdir -p {PROJECT_DIR_REMOTE / spline_path}'")

        zip_file_path = Path("./torque_profiles/spline_profiles/spline_profiles.zip")
        
        with zipfile.ZipFile(zip_file_path, "w") as zip_file:
            for profile in Path(spline_path).iterdir():
                if profile.suffix == ".csv":
                    spline_profile = pd.read_csv(profile, index_col="Percentage")
                    spline_profile.theta_2 = theta_2_scaled
                    spline_profile.to_csv(profile)
                    zip_file.write(profile, os.path.basename(profile))
        # Send zip file to remote host
        os.system(f"scp {zip_file_path} macbook:{PROJECT_DIR_REMOTE / spline_path}")
        unzip_command = f"ssh macbook 'unzip -oq {PROJECT_DIR_REMOTE / zip_file_path} -d {PROJECT_DIR_REMOTE / spline_path}'"
        # Unzip the file on the remote host
        os.system(unzip_command)
        # Remove the zip file on the remote host
        os.system(f"ssh macbook 'rm {PROJECT_DIR_REMOTE / zip_file_path}'")
        
        # Unzip the file locally
        os.system(f"unzip -oq {zip_file_path} -d {spline_path}")
        # Remove the zip file locally
        os.remove(zip_file_path)

        # Apply scaled theta to original optimal profile
        scaled_optimal_profile = unadjusted_profile.copy()
        scaled_optimal_profile.theta_2 = theta_2_scaled
        scaled_profile_path = Path(profile_dir.parent) / f"{profile_dir.stem}_scaled.csv"
        scaled_optimal_profile.to_csv(scaled_profile_path)

        os.system(f"scp {scaled_profile_path} macbook:{PROJECT_DIR_REMOTE / scaled_profile_path.parent}")

        with open(yaml_path, "w") as f:
            yaml.dump(calibration_data, f)

        os.system(f"scp {yaml_path} macbook:{remote_dir}")
        

    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Shutting down...")

def assist_multiple_profiles(motor_1: CubemarsMotor, motor_2: CubemarsMotor, freq: int, session_dir: Path, remote_dir: Path, profile_dir: Path, mode: Literal["TRIGGER", "ENTER"]):
    # This function currently only makes sense for the naming convention peak_time_"time"_peak_force_"force".csv
    # Need to adapt, once I come up with a better naming convention
    spline_profiles_path = Path(profile_dir.parent) / f"spline_profiles"
    spline_dict = dict()

    # Load spline profiles
    for path in spline_profiles_path.iterdir():
        if path.suffix == ".csv":
            peak_time = int(path.stem.split("_")[2])
            peak_force = int(path.stem.split("_")[5])

            if peak_time not in spline_dict:
                spline_dict[peak_time] = dict()

            spline_dict[peak_time][peak_force] = pd.read_csv(path, index_col="Percentage")

    try:
        motor_1.send_torque(desired_torque=0, safety=True)
        motor_2.send_torque(desired_torque=0, safety=True)

        print('Choose mode:\n')
        print('1 - Single profile')
        print('2 - All profiles for a given peak time')
        print('3 - All profiles for a given peak force')
        print('4 - All profiles')
        print('0 - Exit')

        chosen_mode = input("\nChoose mode: ")

        if chosen_mode == "1":
            peak_times = sorted(list(spline_dict.keys()))
            peak_forces = sorted(list(spline_dict[peak_times[0]].keys()))

            print("Available peak times: [%]\n", peak_times)
            print("Peak forces: [N]\n", peak_forces)

            peak_time = int(input("Enter peak time: "))
            peak_force = int(input("Enter peak force: "))
            profile = spline_dict[peak_time][peak_force]

            log_path, logger = get_logger(log_name=f"single_time_{peak_time}_force_{peak_force}", session_dir=session_dir, profile_details=[peak_time, peak_force])

            print(f"Recording to {log_path}\n")
            print(f"Using profile with:")
            print(f"Peak time: {peak_time}%")
            print(f"Peak force: {peak_force}N")

            await_trigger_signal(mode=mode)

            control_loop_and_log(motor_1=motor_1, motor_2=motor_2, logger=logger, profile=profile, freq=freq, mode=mode)
            save_log_or_delete(remote_dir=remote_dir, log_path=log_path) 

        elif chosen_mode == "2":
            peak_time = None
            peak_times = sorted(list(spline_dict.keys()))

            print("Available peak times: [%]\n", )

            # Check for valid entry
            while not peak_time:
                peak_time = int(input("Enter peak time: "))
                if peak_time not in spline_dict:
                    peak_time = None
                    print("Invalid peak time. Try again.")
            
            profiles = spline_dict[peak_time]

            print(f"Using following profiles for a peak time of {peak_time}%")
            print(list(spline_dict[peak_force].keys()))
            
            for peak_force, profile in profiles.items():
                # peak_time because we select a specific peak time and iterate over the peak forces
                log_path, logger = get_logger(log_name=f"peak_time_time_{peak_time}_force_{peak_force}", session_dir=session_dir)

                print(f"\nCurrent profile: \nPeak time: {peak_time}% \nPeak force: {peak_force}N")
                print(f"Recording to {log_path}\n")
                await_trigger_signal()
                control_loop_and_log(motor_1=motor_1, motor_2=motor_2, logger=logger, profile=profile, freq=freq)
                save_log_or_delete(remote_dir=remote_dir, log_path=log_path) 

        elif chosen_mode == '3':
            peak_force = None
            peak_times = sorted(list(spline_dict.keys()))
            peak_forces = sorted(list(spline_dict[peak_times[0]].keys()))
            
            print("Available peak forces: [N]\n", peak_forces)

            # Check for valid entry
            while not peak_force:
                peak_force = int(input("Enter peak force: "))
                if peak_force not in peak_forces:
                    peak_force = None
                    print("Peak force not in profile. Try again.")

            print(f"Using following profiles for a peak force of {peak_force}N")
            print(f"Peak times: \n", peak_times)
            
            for i, peak_time in enumerate(peak_times):
                success = False
                print(f"\n\nIteration {i + 1} of {len(peak_times)}\n\n")

                while not success:
                    profile = spline_dict[peak_time][peak_force]
                    # peak_force because we select a specific peak force and iterate over the peak times
                    log_path, logger = get_logger(log_name=f"assist", session_dir=session_dir, profile_details=[peak_time, peak_force])

                    print(f"\n\nCurrent profile: \nPeak time: {peak_time}% \nPeak force: {peak_force}N")
                    print(f"\nRecording to {log_path}")

                    await_trigger_signal(mode=mode)
                    success = control_loop_and_log(motor_1=motor_1, motor_2=motor_2, logger=logger, profile=profile, freq=freq, apply_force=False, mode=mode)

                    save_log_or_delete(remote_dir=remote_dir, log_path=log_path, successful=success)

        elif chosen_mode == '0':
            print("Exiting...")
            return

    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Shutting down...")


def control_loop_and_log(
        motor_1: CubemarsMotor,
        motor_2: CubemarsMotor,
        logger: csv.writer,
        profile: pd.DataFrame,
        freq: int,
        mode: Literal["TRIGGER", "ENTER", "SOCKET"],
        apply_force: bool=True):

    print("Recording started. Please perform the sit-to-stand motion.")
    print("Press Ctrl + C or trigger to stop recording.\n")
    print_time = 0
    start_time = time.time()

    loop = SoftRealtimeLoop(dt=1 / freq, report=False, fade=0)

    success = True

    for t in loop:
        if mode == "TRIGGER":
            if GPIO.input(17):  # Detects if stop signal is triggered
                print("Stopped recording, exiting...")
                break
        
        if mode == "SOCKET":
            if not collect_flag:
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
        
        if apply_force:
            motor_1.send_torque(desired_torque=tau_1, safety=False)
            motor_2.send_torque(desired_torque=tau_2, safety=False)
        else:
            motor_1.send_torque(desired_torque=0, safety=False) 
            motor_2.send_torque(desired_torque=0, safety=False)

        if t - print_time >= 0.05:
            print(f"{motor_1.type}: Angle: {np.rad2deg(motor_1.position):.3f} Torque: {motor_1.measured_torque:.3f}")
            print(f"{motor_2.type}: Angle: {np.rad2deg(motor_2.position):.3f} Torque: {motor_2.measured_torque:.3f}")
            print(f"Body height: {-P_EE[0]}")
            print(f"Movement: {index: .0f}%. tau_1: {tau_1}, tau_2: {tau_2}")
            sys.stdout.write(f"\x1b[4A\x1b[2K")
                    
            print_time = t
        # ["Percentage", "target_tau_1", "measured_tau_1", "theta_1", "velocity_1", "target_tau_2", "measured_tau_2", "theta_2", "velocity_2", "EE_X", "EE_Y"]
        # print(f"tau_1: {motor_1.measured_torque}, tau_2: {motor_2.measured_torque}")
        logger.writerow([cur_time - start_time , index, tau_1, motor_1.measured_torque, motor_1.position, motor_1.velocity, tau_2, motor_2.measured_torque, motor_2.position, motor_2.velocity, P_EE[0], P_EE[1]])
    del loop

    motor_1.send_torque(desired_torque=0, safety=False)
    motor_2.send_torque(desired_torque=0, safety=False)

    if not success:
        motor_1._emergency_stop = False
        motor_2._emergency_stop = False

        print("\nSomething went wrong. Repeating the iteration...")

    return success


def apply_simulation_profile(motor_1: CubemarsMotor, motor_2: CubemarsMotor, freq: int, session_dir: Path, remote_dir: Path, profile_dir: Path, mode: Literal["TRIGGER", "ENTER"]):
    log_path, logger = get_logger(log_name=f"{profile_dir.stem}_scaled", session_dir=session_dir)

    adjusted_profile_dir = profile_dir.parent / f"{profile_dir.stem}_scaled.csv"
    print(f"Using profile: {adjusted_profile_dir}")
    
    # Throw an error if the file does not exist
    if not adjusted_profile_dir.exists():
        raise FileNotFoundError(f"File {adjusted_profile_dir} does not exist, please calibrate the device first.")
    
    profile = pd.read_csv(adjusted_profile_dir, index_col="Percentage")

    await_trigger_signal(mode=mode)
    countdown(duration=3)

    try:
        success = control_loop_and_log(motor_1=motor_1, motor_2=motor_2, logger=logger, profile=profile, freq=freq, mode=mode, apply_force=True)
        save_log_or_delete(remote_dir=remote_dir, log_path=log_path, successful=success) 

    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Shutting down...")


def collect_unpowered_data(motor_1: CubemarsMotor, motor_2: CubemarsMotor, freq: int, session_dir: Path, remote_dir: Path, profile_dir: Path, mode: Literal["TRIGGER", "ENTER"]):
    """ Collect unpowered data for EMG synchronization.

    Args:
        motor_1 (CubemarsMotor): motor_1
        motor_2 (CubemarsMotor): motor_2
        freq (int): target frequency (Hz)
        session_dir (Path): session directory
        remote_dir (Path): remote session directory
        mode (Literal["TRIGGER", "ENTER"]): record on trigger or Enter pressing
    """
    iterations = 5

    adjusted_profile_dir = profile_dir.parent / f"{profile_dir.stem}_scaled.csv"
    profile = pd.read_csv(adjusted_profile_dir, index_col="Percentage")

    # range from 1-5
    for i in range(1, iterations + 1):
        print("\nIteration number: ", i)
        success = False
        while not success:
            try:
                log_path, logger = get_logger(log_name=f"unpowered_device_{i}", session_dir=session_dir)
                print(f"Recording to {log_path}")
                await_trigger_signal(mode=mode)
                # wait for 3 seconds
                countdown(duration=3)
                success = control_loop_and_log(motor_1=motor_1, motor_2=motor_2, logger=logger, profile=profile, freq=freq, apply_force=False, mode=mode)
                save_log_or_delete(remote_dir=remote_dir, log_path=log_path, successful= success)

            except Exception as e:
                print(e.with_traceback)
                print(f"An error occurred: {e}")
                print("Repeating the iteration...")

if __name__ == "__main__":
    logging = True
    freq = 200

    subject_id = "Nathan"
    subject_folder = Path(f"./subject_logs/subject_{subject_id}")


    unadjusted_profile_dir = Path(f"./torque_profiles/spline_profiles/peak_time_57_peak_force_62.csv")
    # unadjusted_profile_dir = Path(f"./torque_profiles/spline_profiles/peak_time_57_peak_force_62.csv")
    # unadjusted_profile_dir = Path(f"./torque_profiles/simulation_profile_Camille.csv")
    # unadjusted_profile_dir = Path(f"./torque_profiles/simulation_profile_Camille_scalex_scaley.csv")
    # unadjusted_profile_dir = Path(f"./torque_profiles/simulation_profile_Camille_xy.csv")
    # unadjusted_profile_dir = Path(f"./torque_profiles/simulation_profile_Camille_y.csv")
    # Camille.csv
    # Camille_scalex_scaley.csv
    # Camille_xy.csv
    # Camille_y.csv

    trigger_mode = "SOCKET" # TRIGGER, ENTER or SOCKET

    if trigger_mode == "SOCKET":
        client_thread = threading.Thread(target=start_server)
        client_thread.start()

    session_dir, session_remote_dir = set_up_logging_dir(subject_folder=subject_folder)
    try:
        while True:
            # Display the menu
            print("\nOptions:")
            print("1 - Calibrate Height")
            print("2 - Run Assistance")
            print("3 - Apply multiple assistance profiles")
            print("4 - Collect unpowered data")
            print("0 - Exit")

            # Get user's choice
            choice = int(input("Enter your choice: "))

            if choice == States.CALIBRATING:
                with CubemarsMotor(motor_type="AK70-10", frequency=freq) as motor_1:
                    with CubemarsMotor(motor_type="AK60-6", frequency=freq) as motor_2:
                        calibrate_height(motor_1, motor_2, freq=freq, session_dir=session_dir, remote_dir=session_remote_dir, profile_dir=unadjusted_profile_dir)

            elif choice == States.UNPOWERED_COLLECTION:
                with CubemarsMotor(motor_type="AK70-10", frequency=freq) as motor_1:
                    with CubemarsMotor(motor_type="AK60-6", frequency=freq) as motor_2:
                        collect_unpowered_data(motor_1, motor_2, freq=freq, session_dir=session_dir, remote_dir=session_remote_dir, profile_dir=unadjusted_profile_dir, mode=trigger_mode)

            elif choice == States.ASSISTING:
                with CubemarsMotor(motor_type="AK70-10", frequency=freq) as motor_1:
                    with CubemarsMotor(motor_type="AK60-6", frequency=freq) as motor_2:
                        apply_simulation_profile(motor_1, motor_2, freq=freq, session_dir=session_dir, remote_dir=session_remote_dir, profile_dir=unadjusted_profile_dir, mode=trigger_mode)

            elif choice == States.ASSIST_PROFILES:
                with CubemarsMotor(motor_type="AK70-10", frequency=freq) as motor_1:
                    with CubemarsMotor(motor_type="AK60-6", frequency=freq) as motor_2:
                        assist_multiple_profiles(motor_1, motor_2, freq=freq, session_dir=session_dir, remote_dir=session_remote_dir, profile_dir=unadjusted_profile_dir, mode=trigger_mode)

            elif choice == States.EXIT:
                print("Exiting...")
                break
    finally:
        client_thread.join()
        GPIO.cleanup()
