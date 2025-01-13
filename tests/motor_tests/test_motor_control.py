import os
import sys
import time
import can
import numpy as np

from pathlib import Path

from NeuroLocoMiddleware.SoftRealtimeLoop import SoftRealtimeLoop
from assistive_arm.motor_control import CubemarsMotor


def single_motor_control_loop(motor_1: CubemarsMotor):
    # Start control loop
    freq = 200  # Hz

    loop = SoftRealtimeLoop(dt=1 / freq, report=True, fade=0)
    start_time = 0

    # General control loop
    try:
        for t in loop:
            motor_1.send_torque(desired_torque=0.2, safety=False)

            if t - start_time > 0.1:
                print(
                    f"{motor_1.type}: Angle: {np.rad2deg(motor_1.position):.3f} Velocity: {motor_1.velocity:.3f}"
                )
                sys.stdout.write(f"\x1b[1A\x1b[2K")
                start_time = t

            if motor_1._emergency_stop:
                break
        del loop

    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Shutting down...")


def dual_motor_control_loop(motor_1: CubemarsMotor, motor_2: CubemarsMotor):
    """Control loop to operate both motors simultaneously."""
    freq = 200  # Hz
    loop = SoftRealtimeLoop(dt=1 / freq, report=True, fade=0)
    start_time = 0

    try:
        for t in loop:
            # Send commands to both motors
            motor_1.send_torque(desired_torque=0.1, safety=False)
            motor_2.send_torque(desired_torque=0.1, safety=False)  # Opposite torque for demonstration

            # Log data for both motors every 0.1 seconds
            if t - start_time > 0.1:
                print(
                    f"{motor_1.type}: Angle: {np.rad2deg(motor_1.position):.3f} "
                    f"Velocity: {motor_1.velocity:.3f}"
                )
                print(
                    f"{motor_2.type}: Angle: {np.rad2deg(motor_2.position):.3f} "
                    f"Velocity: {motor_2.velocity:.3f}"
                )
                sys.stdout.write(f"\x1b[1A\x1b[2K")  # Clear last line for clean output
                sys.stdout.write(f"\x1b[1A\x1b[2K")
                start_time = t

            # Emergency stop check
            if motor_1._emergency_stop or motor_2._emergency_stop:
                print("Emergency stop activated. Exiting...")
                break

    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Shutting down...")

    finally:
        print("Exiting control loop...")
        del loop


if __name__ == "__main__":
    os.system(f"sudo ip link set can0 up type can bitrate 1000000")
    can_bus = can.interface.Bus(channel="can0", bustype="socketcan")

    with CubemarsMotor("AK70-10", frequency=200, can_bus=can_bus) as motor_1, CubemarsMotor("AK60-6", frequency=200, can_bus=can_bus) as motor_2:
        # Control loop for each motor (one after the other)
        if False:
            single_motor_control_loop(motor_1)
            single_motor_control_loop(motor_2)

        # Control loop for both motors simultaneously
        if True:
            dual_motor_control_loop(motor_1, motor_2)


    os.system(f"sudo ip link set can0 down")
