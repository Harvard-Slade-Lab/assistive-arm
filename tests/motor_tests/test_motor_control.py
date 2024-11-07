import os
import sys
import time
import numpy as np

from pathlib import Path

from NeuroLocoMiddleware.SoftRealtimeLoop import SoftRealtimeLoop
from assistive_arm.motor_control import CubemarsMotor


def main(motor_1: CubemarsMotor):
    # Start control loop
    freq = 200  # Hz

    loop = SoftRealtimeLoop(dt=1 / freq, report=True, fade=0)
    start_time = 0

    # General control loop
    try:
        for t in loop:
            motor_1.send_torque(desired_torque=1, safety=False)

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


if __name__ == "__main__":
    with CubemarsMotor("AK70-10", frequency=200) as motor_1:
        main(motor_1)
