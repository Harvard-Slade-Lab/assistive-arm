import csv
import numpy as np
import os
import time
import sys

from NeuroLocoMiddleware.SoftRealtimeLoop import SoftRealtimeLoop
from assistive_arm.motor_control import CubemarsMotor

# CHANGE THESE TO MATCH YOUR DEVICE!
dt = 0.005
print_every = 0.5  # seconds


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
            print(f"Angle: {cur_angle: .3f} Velocity: {vel: .3f} Torque: {torque: .3f}")
            start_time = t
        if abs(cur_angle - origin) < 0.05:
            print("Origin reached! Setting zero position...")
            motor.send_zero_position()
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
        cur_vel = velocity_sign * (velocity if t < 0.5 else velocity / 2)
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
    log_file = f"../logs/{time.strftime('%Y-%m-%d-%H-%M-%S')}.csv"
    os.system(f"touch {log_file}")

    with CubemarsMotor(motor_type="AK70-10") as motor_1:
        print("Calibrating motor... Do not touch.")

        limit_tracking(motor_1, direction='right', velocity=3)
        print("Sleeping...")
        time.sleep(2)

        left_limit = limit_tracking(motor_1, direction='left', velocity=3)

        print("Angle range: ", [0, left_limit])
        print(f"Setting zero position to {left_limit/2: .2f}º...")
        time.sleep(1.5)
        set_dh_origin(motor_1, left_limit / 2)
