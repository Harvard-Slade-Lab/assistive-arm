import os
import sys
import time
import can
import numpy as np

from pathlib import Path

from NeuroLocoMiddleware.SoftRealtimeLoop import SoftRealtimeLoop
from assistive_arm.motor_control import CubemarsMotor, setup_can_and_motors, shutdown_can_and_motors


def single_motor_control_loop(motor: CubemarsMotor):
    # Start control loop
    freq = 200  # Hz

    loop = SoftRealtimeLoop(dt=1 / freq, report=True, fade=0)
    start_time = 0

    # General control loop
    try:
        for t in loop:
            motor.send_torque(desired_torque=0.5, safety=False)

            if t - start_time > 0.1:
                print(
                    f"{motor.type}: Angle: {np.rad2deg(motor.position):.3f} Velocity: {motor.velocity:.3f}"
                )
                sys.stdout.write(f"\x1b[1A\x1b[2K")
                start_time = t

            if motor._emergency_stop:
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
            motor_1.send_torque(desired_torque=0.6, safety=False)
            motor_2.send_torque(desired_torque=-0.2, safety=False)  # Opposite torque for demonstration

            # Log data for both motors every 0.1 seconds
            if t - start_time > 0.1:
                if not motor_1.swapped_motors:
                    print(f"{motor_1.type}: Angle: {np.rad2deg(motor_1.position):.3f}, measured torque: {motor_1.measured_torque:.3f} " f"Velocity: {motor_1.velocity:.3f}")
                    print(f"{motor_2.type}: Angle: {np.rad2deg(motor_2.position):.3f}, measured torque: {motor_2.measured_torque:.3f} "f"Velocity: {motor_2.velocity:.3f}")
                    sys.stdout.write(f"\x1b[1A\x1b[2K")  # Clear last line for clean output
                    sys.stdout.write(f"\x1b[1A\x1b[2K")
                    start_time = t
                else:
                    print("SWAPPED!!!!")
                    print(f"{motor_1.type}: Angle: {np.rad2deg(motor_2.position):.3f}, measured torque: {motor_2.measured_torque:.3f} " f"Velocity: {motor_2.velocity:.3f}")
                    print(f"{motor_2.type}: Angle: {np.rad2deg(motor_1.position):.3f}, measured torque: {motor_1.measured_torque:.3f} "f"Velocity: {motor_1.velocity:.3f}")
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

import numpy as np


def dual_motor_control_loop_sinusoid(motor_1: CubemarsMotor, motor_2: CubemarsMotor):
    """Control loop to operate both motors with sinusoidal torque signals."""
    freq = 200  # Hz
    signal_freq = 1  # Frequency of the sinusoidal signal (Hz)
    amplitude_motor_1 = 0.6  # Amplitude of the sinusoidal signal for motor_1
    amplitude_motor_2 = 0.2 # Amplitude of the sinusoidal signal for motor_2

    loop = SoftRealtimeLoop(dt=1 / freq, report=True, fade=0)
    start_time = 0

    try:
        for t in loop:
            # Calculate sinusoidal torques
            torque_1 = amplitude_motor_1 * np.sin(2 * np.pi * signal_freq * t)
            torque_2 = amplitude_motor_2 * np.sin(2 * np.pi * signal_freq * t)

            # Send sinusoidal torque commands to both motors
            motor_1.send_torque(desired_torque=torque_1, safety=False)
            motor_2.send_torque(desired_torque=torque_2, safety=False)

            # Log data for both motors every 0.1 seconds
            if t - start_time > 0.1:
                if not motor_1.swapped_motors:
                    print(f"{motor_1.type}: Angle: {np.rad2deg(motor_1.position):.3f}, measured torque: {motor_1.measured_torque:.3f} " f"Velocity: {motor_1.velocity:.3f}")
                    print(f"{motor_2.type}: Angle: {np.rad2deg(motor_2.position):.3f}, measured torque: {motor_2.measured_torque:.3f} "f"Velocity: {motor_2.velocity:.3f}")
                    sys.stdout.write(f"\x1b[1A\x1b[2K")  # Clear last line for clean output
                    sys.stdout.write(f"\x1b[1A\x1b[2K")
                    start_time = t
                else:
                    print("SWAPPED!!!!")
                    print(f"{motor_1.type}: Angle: {np.rad2deg(motor_2.position):.3f}, measured torque: {motor_2.measured_torque:.3f} " f"Velocity: {motor_2.velocity:.3f}")
                    print(f"{motor_2.type}: Angle: {np.rad2deg(motor_1.position):.3f}, measured torque: {motor_1.measured_torque:.3f} "f"Velocity: {motor_1.velocity:.3f}")
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
    can_bus, motor_1, motor_2 = setup_can_and_motors()

    # Control loop for each motor (one after the other)
    # if True:
    #     single_motor_control_loop(motor_1)
    #     single_motor_control_loop(motor_2)

    # Control loop for both motors simultaneously
    # if True:
    for i in range(3):
        dual_motor_control_loop(motor_1, motor_2)
        dual_motor_control_loop_sinusoid(motor_1, motor_2)

    shutdown_can_and_motors(can_bus, motor_1, motor_2)
