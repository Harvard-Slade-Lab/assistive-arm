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

from pathlib import Path
from enum import Enum

import RPi.GPIO as GPIO

from NeuroLocoMiddleware.SoftRealtimeLoop import SoftRealtimeLoop
from assistive_arm.motor_control import CubemarsMotor
from assistive_arm.robotic_arm import calculate_ee_pos, get_jacobian, get_target_torques

# Set options
np.set_printoptions(precision=3, suppress=True)

# Use Broadcom SOC Pin numbers
GPIO.setmode(GPIO.BCM)
GPIO.setup(17, GPIO.IN)

PROJECT_DIR_REMOTE = Path("/Users/xabieririzar/uni-projects/Harvard/assistive-arm")

class States(Enum):
    CALIBRATING = 1
    ASSISTING = 2
    ASSIST_PROFILES = 3
    EXIT = 0

    def __eq__(self, other):
            if isinstance(other, int):
                return self.value == other
            return False

def get_next_sample_number(session_dir: Path, log_name: str) -> int:
    """
    Get the next sample number for the log file.

    :param session_dir: The directory where log files are stored.
    :param log_name: The base name of the log files.
    :return: The next sample number.
    """
    # Regular expression to match files and extract sample number
    pattern = re.compile(rf"(\d+)_{log_name}.csv")

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

    session_remote_dir = f"{PROJECT_DIR_REMOTE}/subject_logs/{subject_folder.name}/"
    os.system(f"ssh macbook 'mkdir -p {session_remote_dir}'")
    
    return session_dir, session_remote_dir


def countdown(duration: int=3):
    for i in range(duration, 0, -1):
        print(f"Recording in {i} seconds...", end="\r")
        time.sleep(1)
    print("\nGO!")


def calibrate_height(motor_1: CubemarsMotor, motor_2: CubemarsMotor, freq: int, session_dir: Path, remote_dir: Path):
    yaml_path = get_yaml_path(yaml_name="calibration_data", session_dir=session_dir)

    unadjusted_profile = pd.read_csv("./torque_profiles/simulation_profile.csv", index_col="Percentage")

    loop = SoftRealtimeLoop(dt=1 / freq, report=True, fade=0)

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
        os.system(f"ssh macbook 'mkdir -p {PROJECT_DIR_REMOTE / spline_path}'")

        zip_file_path = Path("./torque_profiles/spline_profiles/spline_profiles.zip")
        
        with zipfile.ZipFile(zip_file_path, "w") as zip_file:
            for profile in Path(spline_path).iterdir():
                if profile.suffix == ".csv":
                    spline_profile = pd.read_csv(profile, index_col="Percentage")
                    spline_profile.theta_2 = theta_2_scaled
                    spline_profile.to_csv(profile)
                    zip_file.write(profile, os.path.basename(profile))

        os.system(f"scp {zip_file_path} macbook:{PROJECT_DIR_REMOTE / spline_path}")
        unzip_command = f"ssh macbook 'unzip -oq {PROJECT_DIR_REMOTE / zip_file_path} -d {PROJECT_DIR_REMOTE / spline_path}'"
        os.system(unzip_command)
        os.system(f"ssh macbook 'rm {PROJECT_DIR_REMOTE / zip_file_path}'")
        
        # Apply scaled theta to original optimal profile
        scaled_optimal_profile = unadjusted_profile.copy()
        scaled_optimal_profile.theta_2 = theta_2_scaled
        scaled_profile_path = Path("./torque_profiles/scaled_simulation_profile.csv")
        scaled_optimal_profile.to_csv(scaled_profile_path)

        os.system(f"scp {scaled_profile_path} macbook:{PROJECT_DIR_REMOTE / scaled_profile_path.parent}")

        with open(yaml_path, "w") as f:
            yaml.dump(calibration_data, f)

        os.system(f"scp {yaml_path} macbook:{remote_dir}")
        

    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Shutting down...")

def assist_multiple_profiles(motor_1: CubemarsMotor, motor_2: CubemarsMotor, freq: int, session_dir: Path, remote_dir: Path):
    spline_profiles_path = Path("./torque_profiles/spline_profiles/")
    spline_dict = dict()

    for path in spline_profiles_path.iterdir():
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

        chosen_mode = input("\nChoose mode: ")

        if chosen_mode == "1":
            peak_times = sorted(list(spline_dict.keys()))
            peak_forces = sorted(list(spline_dict[peak_times[0]].keys()))
            print("Available peak times: [%]\n", peak_times)
            print("Peak forces: [N]\n", peak_forces)

            peak_time = int(input("Enter peak time: "))
            peak_force = int(input("Enter peak force: "))
            profile = spline_dict[peak_time][peak_force]

            log_path, logger = get_logger(log_name=f"single_time_{peak_time}_force_{peak_force}", session_dir=session_dir)
            print(f"Recording to {log_path}")
            print(f"\nUsing profile with:")
            print(f"Peak time: {peak_time}%")
            print(f"Peak force: {peak_force}N")

            print("\nPress trigger to start recording P_EE...")
            while not GPIO.input(17):
                pass
            print()
            control_loop_and_log(motor_1=motor_1, motor_2=motor_2, logger=logger, profile=profile, freq=freq)
            save_log_or_delete(remote_dir=remote_dir, log_path=log_path) 

        elif chosen_mode == "2":
            peak_time = None
            print("Available peak times: [%]\n", sorted(list(spline_dict.keys())))

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
                print(f"Recording to {log_path}")
                input("Press Enter to start recording...")
                countdown(duration=3)
                print()
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
            print(f"Peak times: \n",list(spline_dict.keys()))
            
            for peak_time in peak_times:
                profile = spline_dict[peak_time][peak_force]
                # peak_force because we select a specific peak force and iterate over the peak times
                log_path, logger = get_logger(log_name=f"fixed_force_time_{peak_time}_force_{peak_force}", session_dir=session_dir)

                print(f"Current profile: \nPeak time: {peak_time}% \nPeak force: {peak_force}N")
                print(f"Recording to {log_path}")
                print("\nPress trigger to start recording P_EE...")
                while not GPIO.input(17):
                    pass
                print()
                control_loop_and_log(motor_1=motor_1, motor_2=motor_2, logger=logger, profile=profile, freq=freq)
                save_log_or_delete(remote_dir=remote_dir, log_path=log_path)

    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Shutting down...")

def save_log_or_delete(remote_dir, log_path):
    ans = input("Keep log file? [Y/n] ")
    if ans == "n":
        try:
            os.remove(log_path)
        except FileNotFoundError:
            print("File doesn't exist or was already deleted.")
    else:
        print("Sending logfile to Mac...")
        print("log file: ", log_path)
        os.system(f"scp {log_path} macbook:{remote_dir}")


def control_loop_and_log(motor_1: CubemarsMotor, motor_2: CubemarsMotor, logger: csv.writer, profile: pd.DataFrame, freq: int):

    print("Recording started. Please perform the sit-to-stand motion.")
    print("Press Ctrl + C to stop recording.\n")
    print_time = 0
    start_time = time.time()

    loop = SoftRealtimeLoop(dt=1 / freq, report=True, fade=0)

    for t in loop:
        if not GPIO.input(17):  # Detects if signal turns off (low signal)
            print("Stopped recording, exiting...")
            break
        cur_time = time.time()
        if motor_1._emergency_stop or motor_2._emergency_stop:
            break
                    
        tau_1, tau_2, P_EE, index = get_target_torques(
                        theta_1=motor_1.position,
                        theta_2=motor_2.position,
                        profiles=profile
                    )

        motor_1.send_torque(desired_torque=tau_1, safety=False)
        motor_2.send_torque(desired_torque=tau_2, safety=False)

        if t - print_time >= 0.05:
            print(f"{motor_1.type}: Angle: {np.rad2deg(motor_1.position):.3f} Torque: {motor_1.torque:.3f}")
            print(f"{motor_2.type}: Angle: {np.rad2deg(motor_2.position):.3f} Torque: {motor_2.torque:.3f}")
            print(f"Body height: {-P_EE[0]}")
            print(f"Movement: {index: .0f}%. tau_1: {tau_1}, tau_2: {tau_2}")
            sys.stdout.write(f"\x1b[4A\x1b[2K")
                    
            print_time = t

        logger.writerow([cur_time - start_time, index, tau_1, motor_1.torque, motor_1.position, motor_1.velocity, tau_2, motor_2.torque, motor_2.position, motor_2.velocity, P_EE[0], P_EE[1]])
    del loop

    motor_1.send_torque(desired_torque=0, safety=False)
    motor_2.send_torque(desired_torque=0, safety=False)


def get_logger(log_name: str, session_dir: Path) -> tuple[Path, csv.writer]:
    logged_vars = ["index", "target_tau_1", "measured_tau_1", "theta_1", "velocity_1", "target_tau_2", "measured_tau_2", "theta_2", "velocity_2", "EE_X", "EE_Y"]

    sample_num = get_next_sample_number(session_dir=session_dir, log_name=log_name)
    log_file = f"{sample_num}_{log_name}.csv"
    log_path = session_dir / log_file
    log_path.touch(exist_ok=True)

    with open(log_path, "w") as fd:
        writer = csv.writer(fd)
        writer.writerow(["time"] + logged_vars)

    csv_file = open(log_path, "a").__enter__()
    task_logger = csv.writer(csv_file)

    return log_path, task_logger


def get_yaml_path(yaml_name: str, session_dir: Path) -> Path:
    yaml_file = f"{yaml_name}.yaml"
    yaml_path = session_dir / yaml_file

    return yaml_path


def apply_simulation_profile(motor_1: CubemarsMotor, motor_2: CubemarsMotor, freq: int, session_dir: Path):
    log_path, logger = get_logger(log_name="simulation_profile", session_dir=session_dir)

    profile = pd.read_csv("./torque_profiles/scaled_simulation_profile.csv", index_col="Percentage")

    input("\nPress Enter to start calibrating...")
    countdown(duration=3)

    try:
        control_loop_and_log(motor_1=motor_1, motor_2=motor_2, logger=logger, profile=profile, freq=freq)

    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Shutting down...")


if __name__ == "__main__":
    logging = True
    freq = 200

    subject_id = "Xabi"
    subject_folder = Path(f"./subject_logs/subject_{subject_id}")

    session_dir, session_remote_dir = set_up_logging_dir(subject_folder=subject_folder)
    try:
        while True:
            # Display the menu
            print("\nOptions:")
            print("1 - Calibrate Height")
            print("2 - Run Assistance")
            print("3 - Apply multiple assistance profiles")
            print("0 - Exit")

            # Get user's choice
            choice = int(input("Enter your choice: "))

            if choice == States.CALIBRATING:
                with CubemarsMotor(motor_type="AK70-10", frequency=freq) as motor_1:
                    with CubemarsMotor(motor_type="AK60-6", frequency=freq) as motor_2:
                        calibrate_height(motor_1, motor_2, freq=freq, session_dir=session_dir, remote_dir=session_remote_dir)

            elif choice == States.ASSISTING:
                with CubemarsMotor(motor_type="AK70-10", frequency=freq) as motor_1:
                    with CubemarsMotor(motor_type="AK60-6", frequency=freq) as motor_2:
                        apply_simulation_profile(motor_1, motor_2, freq=freq, session_dir=session_dir)

            elif choice == States.ASSIST_PROFILES:
                with CubemarsMotor(motor_type="AK70-10", frequency=freq) as motor_1:
                    with CubemarsMotor(motor_type="AK60-6", frequency=freq) as motor_2:
                        assist_multiple_profiles(motor_1, motor_2, freq=freq, session_dir=session_dir, remote_dir=session_remote_dir)

            elif choice == States.EXIT:
                print("Exiting...")
                break
    finally:
        GPIO.cleanup()
