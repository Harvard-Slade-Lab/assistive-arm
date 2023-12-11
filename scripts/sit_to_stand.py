import csv
from datetime import datetime
import os
import sys
import time
import numpy as np
import pandas as pd
import threading

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from NeuroLocoMiddleware.SoftRealtimeLoop import SoftRealtimeLoop
from assistive_arm.motor_control import CubemarsMotor
from assistive_arm.robotic_arm import calculate_ee_pos, get_jacobian

# Set options
np.set_printoptions(precision=3, suppress=True)


def set_up_logger(logged_vars: list[str]) -> tuple:
    """ Set up logger for motor data

    Args:
        logged_vars (list): list of variables to log

    Returns:
        tuple: task_logger, remote_path, log_path
    """
    filename = os.path.basename(sys.argv[0]).split(".")[0]
    current_date = datetime.now()
    month_name = current_date.strftime("%B")
    day = current_date.strftime("%d")

    log_dir = Path('./logs/') / f"{month_name}_{day}"
    log_file_name = f"{filename}_{time.strftime('%m-%d-%H-%M-%S')}.csv"
    log_path = log_dir.absolute() / log_file_name

    os.system(f"touch {log_path}")

    with open(log_path, "w") as fd:
        writer = csv.writer(fd)
        writer.writerow(["time"] + logged_vars)

    csv_file = open(log_path, "a").__enter__()
    task_logger = csv.writer(csv_file)

    remote_path = f"/Users/xabieririzar/uni-projects/Harvard/assistive-arm/motor_logs/{log_dir.name}/"
    
    return task_logger, remote_path, log_path
    

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

    closest_point = abs(profiles.EE_X - P_EE[0]).argmin()
    force_vector = profiles.iloc[closest_point][["force_X", "force_Y"]]
    tau_1, tau_2 = profiles.iloc[closest_point][["tau_1", "tau_2"]]

    jacobian = get_jacobian(theta_1, theta_2)

    # tau_1, tau_2 = jacobian.T @ force_vector
    index = profiles.index[closest_point]

    return tau_1, tau_2, P_EE, index


def countdown(duration: int=3):
    for i in range(duration, 0, -1):
        print(f"Recording in {i} seconds...", end="\r")
        time.sleep(1)
    print("GO!")


def calibrate_height(motor_1: CubemarsMotor, motor_2: CubemarsMotor):
    unadjusted_profile = pd.read_csv("./torque_profiles/optimal_profile.csv", index_col="Percentage")

    freq = 200  # Hz
    loop = SoftRealtimeLoop(dt=1 / freq, report=True, fade=0)

    P_EE_values = []
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

            motor_1.send_torque(desired_torque=0, safety=True)
            motor_2.send_torque(desired_torque=0, safety=True)

            if t - start_time >= 0.05:
                print(f"P_EE x: {P_EE[0]}, y: {P_EE[1]}", end="\r")
                start_time = t

        print("Recording stopped. Processing data...\n")

        # inverted because EE_X is negative
        print(f"Estimated duration: {len(P_EE_values) / freq}s")
        new_max = max(P_EE_values)
        new_min = P_EE_values[-1]

        print("Calibration completed. New range: ")
        print(f"Old 0% STS: {unadjusted_profile.EE_X.max()} 0%: {new_max}")
        print(f"STS 100%: {unadjusted_profile.EE_X.min()} 100%: {new_min}\n")

        original_max = unadjusted_profile.EE_X.max()
        original_min = unadjusted_profile.EE_X.min()

        scale = (new_max - new_min) / (original_max - original_min)

        scaled_optimal_profile = unadjusted_profile.copy()
        scaled_curve = unadjusted_profile['EE_X'].apply(lambda x: new_min + (x - original_min) * scale)
        for i, x in enumerate(scaled_curve):
            if i == 0:
                continue
            if x > scaled_curve.iloc[i-1]:
                scaled_curve.iloc[i] = scaled_curve.iloc[i-1]
        scaled_optimal_profile['EE_X'] = scaled_curve
        scaled_profile_path = Path("./torque_profiles/scaled_optimal_profile.csv")
        scaled_optimal_profile.to_csv(scaled_profile_path)

        # mac_path = Path("/Users/xabieririzar/uni-projects/Harvard/assistive-arm/profiles/scaled_optimal_profile.csv")
        # os.system(f"scp {scaled_profile_path.absolute()} macbook:{mac_path}")

    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Shutting down...")


def main(motor_1: CubemarsMotor, motor_2: CubemarsMotor):
    freq = 400  # Hz
    loop = SoftRealtimeLoop(dt=1 / freq, report=True, fade=0)


    profiles = pd.read_csv("./torque_profiles/scaled_optimal_profile.csv", index_col="Percentage")

    start_time = 0

    task_logger, remote_path, log_path = set_up_logger(logged_vars=["index", "target_tau_1", "target_tau_2", "measured_tau_1", "measured_tau_2", "EE_X", "EE_Y"])

    try:
        for t in loop:
            if motor_1._emergency_stop or motor_2._emergency_stop:
                break
            
            tau_1, tau_2, P_EE, index = get_target_torques(
                theta_1=motor_1.position,
                theta_2=motor_2.position,
                profiles=profiles
            )

            motor_1.send_torque(desired_torque=tau_1, safety=False)
            motor_2.send_torque(desired_torque=tau_2, safety=False)

            if t - start_time >= 0.05:
                print(
                    f"{motor_1.type}: Angle: {np.rad2deg(motor_1.position):.3f} Velocity: {motor_1.velocity:.3f} Torque: {motor_1.measured_torque:.3f}"
                )
                print(
                    f"{motor_2.type}: Angle: {np.rad2deg(motor_2.position):.3f} Velocity: {motor_2.velocity:.3f} Torque: {motor_2.measured_torque:.3f}"
                )
                print(
                    f"P_EE: x:{P_EE[0]} y:{P_EE[1]}"
                )
                print(f"Movement: {index: .0f}%. tau_1: {tau_1}, tau_2: {tau_2}")
                sys.stdout.write(f"\x1b[4A\x1b[2K")
            
                start_time = t

            if t > 0.1:
                task_logger.writerow([t, index, tau_1, tau_2, motor_1.measured_torque, motor_2.measured_torque, P_EE[0], P_EE[1]])
            
        del loop

        print("Sending logfile to Mac...")
        print("log path: ", log_path)
        os.system(f"ssh macbook 'mkdir -p {remote_path}'")
        os.system(f"scp {log_path} macbook:{remote_path}")

    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Shutting down...")


if __name__ == "__main__":
    logging = True

    while True:
        # Display the menu
        print("\nOptions:")
        print("1 - Calibrate Height")
        print("2 - Run Assistance")
        print("0 - Exit")

        # Get user's choice
        choice = input("Enter your choice: ")

        if choice == '1':
            with CubemarsMotor(motor_type="AK70-10", logging=False) as motor_1:
                with CubemarsMotor(motor_type="AK60-6", logging=False) as motor_2:
                    calibrate_height(motor_1, motor_2)
        elif choice == '2':
            with CubemarsMotor(motor_type="AK70-10", logging=logging) as motor_1:
                with CubemarsMotor(motor_type="AK60-6", logging=logging) as motor_2:
                    main(motor_1, motor_2)
        elif choice == '0':
            print("Exiting...")
            break

