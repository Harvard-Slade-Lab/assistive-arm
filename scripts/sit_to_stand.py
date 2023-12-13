import csv
from datetime import datetime
import os
import sys
import time
import numpy as np
import pandas as pd
import yaml

from pathlib import Path
from enum import Enum, auto

from NeuroLocoMiddleware.SoftRealtimeLoop import SoftRealtimeLoop
from assistive_arm.motor_control import CubemarsMotor
from assistive_arm.robotic_arm import calculate_ee_pos, get_jacobian

# Set options
np.set_printoptions(precision=3, suppress=True)


class States(Enum):
    CALIBRATING = 1
    ASSISTING = 2
    ASSIST_PROFILES = 3
    EXIT = 0

    def __eq__(self, other):
            if isinstance(other, int):
                return self.value == other
            return False

def set_up_logger(logged_vars: list[str]) -> tuple:
    """Set up logger for the various task in the script. Return the necessary paths
    for storing and manipulating log files.

    Args:
        logged_vars (list[str]): variables that we want to log

    Returns:
        tuple: task_logger, remote_path, log_path, log_stem
    """
    script_name = os.path.basename(sys.argv[0]).split(".")[0]
    current_date = datetime.now()
    month_name = current_date.strftime("%B")
    day = current_date.strftime("%d")

    log_dir = Path('./motor_logs/') / f"{month_name}_{day}"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_stem = f"{script_name}_{time.strftime('%m-%d-%H-%M-%S')}"
    log_file = f"{log_stem}.csv"
    log_path = log_dir.absolute() / log_file

    os.system(f"touch {log_path}")

    with open(log_path, "w") as fd:
        writer = csv.writer(fd)
        writer.writerow(["time"] + logged_vars)

    csv_file = open(log_path, "a").__enter__()
    task_logger = csv.writer(csv_file)

    remote_dir = f"/Users/xabieririzar/uni-projects/Harvard/assistive-arm/motor_logs/{log_dir.name}/"
    os.system(f"ssh macbook 'mkdir -p {remote_dir}'")
    
    return task_logger, remote_dir, log_path, log_dir
    

def get_target_torques(theta_1: float, theta_2: float, profiles: pd.DataFrame) -> tuple:
    """ Get target torques for a given configuration, based on optimal profile

    Args:
        theta_1 (float): motor_1 angle
        theta_2 (float): motor_2 angle
        profiles (pd.DataFrame): optimal profile dataframe

    Returns:
        tuple: torques (tau_1, tau_2), index (percentage of profile)
    """

    P_EE = calculate_ee_pos(theta_1=theta_1, theta_2=theta_2)
    jacobian = get_jacobian(theta_1, theta_2)

    closest_point = abs(profiles.theta_2 - theta_2).argmin()
    force_vector = profiles.iloc[closest_point][["force_X", "force_Y"]]

    tau_1, tau_2 = -jacobian.T @ force_vector

    index = profiles.index[closest_point]

    return tau_1, tau_2, P_EE, index


def countdown(duration: int=3):
    for i in range(duration, 0, -1):
        print(f"Recording in {i} seconds...", end="\r")
        time.sleep(1)
    print("GO!")


def calibrate_height(motor_1: CubemarsMotor, motor_2: CubemarsMotor, freq: int, yaml_path: Path=None):
    unadjusted_profile = pd.read_csv("./torque_profiles/simulation_profile.csv", index_col="Percentage")

    loop = SoftRealtimeLoop(dt=1 / freq, report=True, fade=0)

    calibration_data = dict()

    P_EE_values = []
    theta_2 = []
    start_time = 0

    try:
        motor_1.send_torque(desired_torque=0, safety=True)
        motor_2.send_torque(desired_torque=0, safety=True)

        input("\nPress Enter to start recording P_EE...")
        countdown(duration=2)  # 3-second countdown
        print("Recording started. Please perform the sit-to-stand motion.")
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

        print("Recording stopped. Processing data...\n")

        # Add offset to ensure that when the subject stands, 0 or 100% will be reached
        theta_2 = np.array(theta_2)
        new_max = theta_2.max() - 0.01
        new_min = theta_2.min() + 0.03

        original_max = unadjusted_profile.theta_2.max()
        original_min = unadjusted_profile.theta_2.min()

        scale = (new_max - new_min) / (original_max - original_min)

        theta_2_scaled = unadjusted_profile['theta_2'].apply(lambda x: new_min + (x - original_min) * scale)

        print(f"Estimated duration: {len(theta_2) / freq}s")
        
        print("Calibration completed. New range: ")
        print(f"Old 0% STS: {np.rad2deg(unadjusted_profile.theta_2.max())} 0%: {np.rad2deg(new_max)}")
        print(f"STS 100%: {np.rad2deg(unadjusted_profile.theta_2.min())} 100%: {np.rad2deg(new_min)}\n")

        calibration_data["new_range"] = {"min": float(new_min), "max": float(new_max)}
        calibration_data["theta_2_values"] = [float(angle) for angle in theta_2]

        scaled_optimal_profile = unadjusted_profile.copy()
        scaled_optimal_profile.theta_2 = theta_2_scaled
        scaled_profile_path = Path("./torque_profiles/scaled_simulation_profile.csv")
        scaled_optimal_profile.to_csv(scaled_profile_path)

        os.system(f"scp {scaled_profile_path} macbook:/Users/xabieririzar/uni-projects/Harvard/assistive-arm/torque_profiles/")

        with open(yaml_path, "w") as f:
            yaml.dump(calibration_data, f, default_flow_style=False)

    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Shutting down...")

def assist_multiple_profiles(motor_1: CubemarsMotor, motor_2: CubemarsMotor, freq: int, logger: csv.writer=None):
    spline_profiles_path = Path("./torque_profiles/spline_profiles/")
    spline_dict = dict()

    loop = SoftRealtimeLoop(dt=1 / freq, report=True, fade=0)

    for path in spline_profiles_path.iterdir():
        peak_time = int(path.stem.split("_")[2])
        peak_force = int(path.stem.split("_")[5])

        if peak_time not in spline_dict:
            spline_dict[peak_time] = dict()

        spline_dict[peak_time][peak_force] = pd.read_csv(path, index_col="Percentage")

    try:
        motor_1.send_torque(desired_torque=0, safety=True)
        motor_2.send_torque(desired_torque=0, safety=True)

        print('Choose mode:')
        print('1 - Single profile')
        print('2 - All profiles for 1 peak time')
        print('3 - All profiles for 1 peak force')
        print('4 - All profiles')

        chosen_mode = input("\nChoose mode: ")

        if chosen_mode == "1":
            peak_time = int(input("Enter peak time: "))
            peak_force = int(input("Enter peak force: "))
            profiles = spline_dict[peak_time][peak_force]

            print(f"Using profile: {peak_time}_{peak_force}")

            print("Press Enter to start recording...")
            countdown(duration=3)
        
        elif chosen_mode == "2":
            peak_time = None
            print("Available peak times: [%]\n", list(spline_dict.keys()))

            # Check for valid entry
            while not peak_time:
                peak_time = int(input("Enter peak time: "))
                if peak_time not in spline_dict:
                    peak_time = None
                    print("Invalid peak time. Try again.")
            
            profiles = spline_dict[peak_time]

            print(f"Using following profiles for peak time: {peak_time}")
            print(list(spline_dict[peak_time].keys()))
            
            for peak_force, profile in profiles.items():
                print(f"Current profile: \nPeak time: {peak_time}% \nPeak force: {peak_force}N")
                print("Press Enter to start recording...")
                countdown(duration=3)
                print()
                print_time = 0
                start_time = time.time()


                for t in loop:
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

    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Shutting down...")


def main(motor_1: CubemarsMotor, motor_2: CubemarsMotor, freq: int, logger: csv.writer=None):

    loop = SoftRealtimeLoop(dt=1 / freq, report=True, fade=0)

    profiles = pd.read_csv("./torque_profiles/scaled_simulation_profile.csv", index_col="Percentage")

    print_time = 0
    start_time = time.time()

    try:
        for t in loop:
            cur_time = time.time()
            if motor_1._emergency_stop or motor_2._emergency_stop:
                break
            
            tau_1, tau_2, P_EE, index = get_target_torques(
                theta_1=motor_1.position,
                theta_2=motor_2.position,
                profiles=profiles
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


    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Shutting down...")


if __name__ == "__main__":
    logging = True
    freq = 200

    task_logger, remote_path, log_path, log_dir = set_up_logger(logged_vars=["index", "target_tau_1", "measured_tau_1", "theta_1", "velocity_1", "target_tau_2", "measured_tau_2", "theta_2", "velocity_2", "EE_X", "EE_Y"])
    yaml_path = log_path.with_suffix(".yaml")

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
                    calibrate_height(motor_1, motor_2, freq=freq, yaml_path=yaml_path)
                    os.system(f"scp {yaml_path} macbook:{remote_path}")

        elif choice == States.ASSISTING:
            with CubemarsMotor(motor_type="AK70-10", frequency=freq) as motor_1:
                with CubemarsMotor(motor_type="AK60-6", frequency=freq) as motor_2:
                    main(motor_1, motor_2, freq=freq, logger=task_logger)

        elif choice == States.ASSIST_PROFILES:
            with CubemarsMotor(motor_type="AK70-10", frequency=freq) as motor_1:
                with CubemarsMotor(motor_type="AK60-6", frequency=freq) as motor_2:
                    assist_multiple_profiles(motor_1, motor_2, freq=freq, logger=task_logger)

        elif choice == States.EXIT:
            print("Exiting...")
            break

    ans = input("Keep log file? [Y/n] ")
    if ans == "n":
        try:
            os.remove(log_path)
        except FileNotFoundError:
            print("File doesn't exist or was already deleted.")
    else:
        print("Sending logfile to Mac...")
        print("log file: ", log_path)
        os.system(f"scp {log_path} macbook:{remote_path}") 
