import csv
import numpy as np
import os
import time
import sys
import can

from pathlib import Path

from NeuroLocoMiddleware.SoftRealtimeLoop import SoftRealtimeLoop
from assistive_arm.motor_control import CubemarsMotor
import os


# export PYTHONPATH=/home/xabier/Documents/assistive-arm:$PYTHONPATH
# sudo ip link set can1 down
# sudo ip link set can1 up type can bitrate 1000000

# CHANGE THESE TO MATCH YOUR DEVICE!
dt = 0.005
print_every = 0.05  # seconds


def set_dh_origin(motor: CubemarsMotor, origin: float):
    loop_3 = SoftRealtimeLoop(dt=dt, report=True, fade=0)

    print("Setting new origin...")
    start_time = 0

    for t in loop_3:
        # Print angle every print_every seconds
        if t < 2:
            motor.send_velocity(desired_vel=-1)
        else:
            motor.send_velocity(desired_vel=-0.5)

        target_diff = abs(motor.position - origin)

        if t - start_time >= print_every:
            sys.stdout.write("\x1b[1A\x1b[2K")
            print(
                f"Diff: {np.rad2deg(target_diff): .3f} Angle: {np.rad2deg(motor.position): .3f} Velocity: {motor.velocity: .3f} Torque: {motor.measured_torque: .3f}"
            )
            start_time = t

        if target_diff < 0.005:
            motor.send_zero_position()
            print("Origin reached! Setting zero position...")
            print(f"Target offset: {np.rad2deg(target_diff): .3f}º")
            break

    del loop_3


def limit_tracking(motor: CubemarsMotor, direction="right", velocity=1):
    if direction not in ["right", "left"]:
        raise ValueError("Direction must be 'right' or 'left'")

    loop = SoftRealtimeLoop(dt=dt, report=True, fade=0)
    action = (
        "Checking right limit..." if direction == "right" else "Checking left limit..."
    )
    print(action)

    prev_angle = 0
    start_time = 0
    velocity_sign = -1 if direction == "right" else 1

    for t in loop:
        # Adjust velocity based on time and direction
        cur_vel = velocity_sign * (velocity if t < 0.5 else velocity / 3)

        motor.send_velocity(desired_vel=cur_vel)

        # Print angle every print_every seconds
        if t - start_time >= print_every:
            sys.stdout.write("\x1b[1A\x1b[2K")
            print(
                f"Angle: {np.rad2deg(motor.position): .3f} Velocity: {motor.velocity: .3f} Torque: {motor.measured_torque: .3f}"
            )
            start_time = t

            if abs(motor.position - prev_angle) < 0.001:
                if direction == "right":
                    print("Right limit reached! Setting zero position...")
                    motor.send_zero_position()
                    break
                else:
                    print("Left limit reached! Setting zero position...")
                    return motor.position

            prev_angle = motor.position

    del loop


if __name__ == "__main__":
    freq = 200

    while True:
        # Display the menu
        print("\nOptions:")
        print("1 - Calibrate Motor 1 (AK70-10)")
        print("2 - Calibrate Motor 2 (AK60-6)")
        print("0 - Exit")

        # Get user's choice
        choice = input("Enter your choice: ")

        if choice == '1':
            os.system(f"sudo ip link set can0 up type can bitrate 1000000")
            can_bus = can.interface.Bus(channel="can0", bustype="socketcan")
            with CubemarsMotor(motor_type="AK70-10", frequency=freq, can_bus=can_bus) as motor_1:
                print(f"Calibrating {motor_1.type}... Do not touch.")
                time.sleep(1)
                limit_tracking(motor_1, direction="right", velocity=3)
                print(f"Setting origin at 0º...")
                print("Sleeping...")
                time.sleep(0.1)
                left_limit = limit_tracking(motor_1, direction="left", velocity=3)
                print("Angle range: ", [0, left_limit])
                time.sleep(1.5)
            os.system(f"sudo ip link set can0 down")

        elif choice == '2':
            os.system(f"sudo ip link set can0 up type can bitrate 1000000")
            can_bus = can.interface.Bus(channel="can0", bustype="socketcan")
            with CubemarsMotor(motor_type="AK60-6", frequency=freq, can_bus=can_bus) as motor_2:
                print("Calibrating motor... Do not touch.")
                time.sleep(1)
                limit_tracking(motor_2, direction='right', velocity=3)
                print("Sleeping...")
                time.sleep(2)
                left_limit = limit_tracking(motor_2, direction='left', velocity=3)
                print("Angle range: ", [0, np.rad2deg(left_limit)])
                print(f"Setting zero position to {np.rad2deg(left_limit/2): .2f}º...")
                time.sleep(1.5)
                set_dh_origin(motor_2, left_limit / 2)
            os.system(f"sudo ip link set can0 down")

        elif choice == '0':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 0.")


# Idea to simpify the calibration process:
# def limit_tracking_simplified(motor: CubemarsMotor, direction="right", velocity=1, total_range=np.pi):
#     if direction not in ["right", "left"]:
#         raise ValueError("Direction must be 'right' or 'left'")

#     loop = SoftRealtimeLoop(dt=dt, report=True, fade=0)
#     action = "Checking right limit..." if direction == "right" else "Checking left limit..."
#     print(action)

#     prev_angle = 0
#     start_time = 0
#     velocity_sign = -1 if direction == "right" else 1

#     for t in loop:
#         cur_vel = velocity_sign * (velocity if t < 0.5 else velocity / 3)

#         motor.send_velocity(desired_vel=cur_vel)

#         if t - start_time >= print_every:
#             sys.stdout.write("\x1b[1A\x1b[2K")
#             print(f"Angle: {np.rad2deg(motor.position): .3f} Velocity: {motor.velocity: .3f} Torque: {motor.measured_torque: .3f}")
#             start_time = t

#             if abs(motor.position - prev_angle) < 0.001:
#                 print(f"{direction.capitalize()} limit reached!")
#                 motor.send_zero_position()
#                 break

#             prev_angle = motor.position

#     # Calculate the other limit based on known total range
#     if direction == "right":
#         left_limit = motor.position - total_range
#     else:
#         right_limit = motor.position + total_range

#     print(f"Total angular range: {np.rad2deg(total_range):.2f}º")
#     print(f"Calculated opposite limit: {np.rad2deg(left_limit if direction == 'right' else right_limit):.2f}º")

#     # Optionally, set zero position at the midpoint
#     midpoint = (motor.position + (left_limit if direction == 'right' else right_limit)) / 2
#     motor.send_zero_position()

#     return motor.position

# # Use the simplified version in your main code
# if __name__ == "__main__":
#     freq = 200

#     while True:
#         print("\nOptions:")
#         print("1 - Calibrate Motor 1 (AK70-10)")
#         print("2 - Calibrate Motor 2 (AK60-6)")
#         print("0 - Exit")

#         choice = input("Enter your choice: ")

#         if choice == '1':
#             with CubemarsMotor(motor_type="AK70-10", frequency=freq) as motor_1:
#                 print(f"Calibrating {motor_1.type}... Do not touch.")
#                 time.sleep(1)
#                 limit_tracking_simplified(motor_1, direction="right", velocity=3)
#                 print(f"Setting origin at midpoint...")
#                 time.sleep(1.5)

#         elif choice == '2':
#             with CubemarsMotor(motor_type="AK60-6", frequency=freq) as motor_2:
#                 print("Calibrating motor... Do not touch.")
#                 time.sleep(1)
#                 limit_tracking_simplified(motor_2, direction="right", velocity=3)
#                 print("Setting origin at midpoint...")
#                 time.sleep(1.5)

#         elif choice == '0':
#             print("Exiting...")
#             break
#         else:
#             print("Invalid choice. Please enter 1, 2, or 0.")
