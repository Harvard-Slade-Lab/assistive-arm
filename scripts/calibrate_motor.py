import csv
import numpy as np
import os
import time
import sys

from NeuroLocoMiddleware.SoftRealtimeLoop import SoftRealtimeLoop
from assistive_arm.motor_control import CubemarsMotor
import os

# CHANGE THESE TO MATCH YOUR DEVICE!
dt = 0.005
print_every = 0.1  # seconds


def set_dh_origin(motor, origin):
    loop_3 = SoftRealtimeLoop(dt=dt, report=True, fade=0)

    print("Setting new origin...")
    start_time = 0

    for t in loop_3:
        # Print angle every print_every seconds
        if t < 2:
            cur_angle, vel, torque = motor.send_velocity(desired_vel=-1)
        else:
            cur_angle, vel, torque = motor.send_velocity(desired_vel=-0.5)

        if t - start_time >= print_every:
            sys.stdout.write('\x1b[1A\x1b[2K')
            print(f"Diff: {abs(cur_angle - origin): .3f} Angle: {cur_angle: .3f} Velocity: {vel: .3f} Torque: {torque: .3f}")
            start_time = t

        if abs(cur_angle - origin) < 0.1:
            motor.send_zero_position()
            print("Origin reached! Setting zero position...")
            print(f"Target offset: {abs(cur_angle - origin): .3f}º")
            break

    del loop_3


def limit_tracking(motor, direction='right', velocity=1):
    if direction not in ['right', 'left']:
        raise ValueError("Direction must be 'right' or 'left'")

    loop = SoftRealtimeLoop(dt=dt, report=True, fade=0)
    action = "Checking right limit..." if direction == 'right' else "Checking left limit..."
    print(action)

    prev_angle = 0
    start_time = 0
    velocity_sign = -1 if direction == 'right' else 1

    for t in loop:
        # Adjust velocity based on time and direction
        cur_vel = velocity_sign * (velocity if t < 0.5 else velocity / 3)
        cur_angle, vel, torque = motor.send_velocity(desired_vel=cur_vel)

        # Print angle every print_every seconds
        if t - start_time >= print_every:
            sys.stdout.write('\x1b[1A\x1b[2K')
            print(f"Angle: {cur_angle: .3f} Velocity: {vel: .3f} Torque: {torque: .3f}")
            start_time = t

            if abs(cur_angle - prev_angle) < 0.001:
                if direction == 'right':
                    print("Right limit reached! Setting zero position...")
                    motor.send_zero_position()
                    break
                else:
                    print("Left limit reached! Setting zero position...")
                    return cur_angle

            prev_angle = cur_angle

    del loop


if __name__ == "__main__":
    # Create CSV file for later analysis, naming it with current time

    filename = os.path.basename(__file__)
    log_file = f"../logs/{filename.split('.')[0]}_{time.strftime('%m-%d-%H-%M-%S')}.csv"
    os.system(f"touch {log_file}")

    # with CubemarsMotor(motor_type="AK70-10", csv_file=log_file) as motor_1:
    #     print(f"Calibrating {motor_1.type}... Do not touch.")

    #     limit_tracking(motor_1, direction='right', velocity=2)
    #     print(f"Setting origin at 0º...")
    #     print("Sleeping...")
    #     time.sleep(2)

    #     left_limit = limit_tracking(motor_1, direction='left', velocity=2)

    #     print("Angle range: ", [0, left_limit])
    #     time.sleep(1.5)

    with CubemarsMotor(motor_type="AK60-6", csv_file=log_file) as motor_2:
        print("Calibrating motor... Do not touch.")

        limit_tracking(motor_2, direction='right', velocity=3)
        print("Sleeping...")
        time.sleep(2)

        left_limit = limit_tracking(motor_2, direction='left', velocity=3)

        print("Angle range: ", [0, left_limit])
        print(f"Setting zero position to {left_limit/2: .2f}º...")
        time.sleep(1.5)
        set_dh_origin(motor_2, left_limit / 2)
