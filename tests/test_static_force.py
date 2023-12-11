import sys
import numpy as np
import pandas as pd
import os
import time

np.set_printoptions(precision=3, suppress=True)

from pathlib import Path

from NeuroLocoMiddleware.SoftRealtimeLoop import SoftRealtimeLoop
from assistive_arm.motor_control import CubemarsMotor


def main(motor_1: CubemarsMotor, motor_2: CubemarsMotor):
    freq = 200  # Hz
    loop = SoftRealtimeLoop(dt=1 / freq, report=True, fade=0)

    profile_path = Path(
        "~/ability-lab/assistive-arm/torque_profiles/scaled_torque_profile.csv"
    )

    start_time = 0

    max_tau_1 = 24
    max_tau_2 = 9

    try:
        for t in loop:
            tau_1 = np.sin(t / 5) * max_tau_1
            tau_2 = np.sin(t / 5) * max_tau_2

            motor_1.send_torque(desired_torque=tau_1, safety=False)
            motor_2.send_torque(desired_torque=tau_2, safety=False)

            if t - start_time > 0.05:
                print(
                    f"{motor_1.type}: Angle: {np.rad2deg(motor_1.position):.3f} Velocity: {motor_1.velocity:.3f} Torque: {motor_1.measured_torque:.3f}"
                )
                print(
                    f"{motor_2.type}: Angle: {np.rad2deg(motor_2.position):.3f} Velocity: {motor_2.velocity:.3f} Torque: {motor_2.measured_torque:.3f}"
                )

                print(f"targets: tau_1: {tau_1}, tau_2: {tau_2}")
                sys.stdout.write(f"\x1b[3A\x1b[2K")

                start_time = t
        del loop

    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Shutting down...")


if __name__ == "__main__":
    filename = os.path.basename(__file__).split(".")[0]
    log_file = Path(f"../logs/{filename}_{time.strftime('%m-%d-%H-%M-%S')}.csv")
    os.system(f"touch {log_file}")

    with CubemarsMotor(motor_type="AK70-10", csv_file=log_file) as motor_1:
        with CubemarsMotor(motor_type="AK60-6", csv_file=log_file) as motor_2:
            main(motor_1, motor_2)
