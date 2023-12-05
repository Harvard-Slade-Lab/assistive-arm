import sys
import numpy as np
import pandas as pd
import os
import time

np.set_printoptions(precision=3, suppress=True)

from pathlib import Path

from NeuroLocoMiddleware.SoftRealtimeLoop import SoftRealtimeLoop
from assistive_arm.motor_control import CubemarsMotor
from assistive_arm.robotic_arm import calculate_ee_pos


def read_profiles(profile_path: Path) -> pd.DataFrame:
    profiles = pd.read_csv(profile_path, index_col="Percentage")

    return profiles


def main(motor_1: CubemarsMotor, motor_2: CubemarsMotor):
    freq = 200  # Hz
    loop = SoftRealtimeLoop(dt=1 / freq, report=True, fade=0)

    profile_path = Path(
        "~/ability-lab/assistive-arm/torque_profiles/optimal_profile.csv"
    )

    profiles = read_profiles(profile_path=profile_path)

    start_time = 0

    L1 = 0.44
    L2 = 0.41

    try:
        for t in loop:
            P_EE = calculate_ee_pos(motor_1, motor_2)

            closest_point = abs(profiles.Y - P_EE[1]).argmin()
            tau_1, tau_2 = profiles.iloc[closest_point][["tau_1", "tau_2"]]

            index = profiles.index[closest_point]

            # motor_1.send_torque(desired_torque=0, safety=True)
            # motor_2.send_torque(desired_torque=0, safety=True)
            motor_1.send_torque(desired_torque=tau_1, safety=True)
            motor_2.send_torque(desired_torque=tau_2, safety=True)

            if t - start_time > 0.05:
                print(
                    f"{motor_1.type}: Angle: {np.rad2deg(motor_1.position):.3f} Velocity: {motor_1.velocity:.3f} Torque: {motor_1.torque:.3f}"
                )
                print(
                    f"{motor_2.type}: Angle: {np.rad2deg(motor_2.position):.3f} Velocity: {motor_2.velocity:.3f} Torque: {motor_2.torque:.3f}"
                )
                print(
                    f"P_EE: x:{P_EE[0]:.3f} y:{P_EE[1]:.3f} theta_1+theta_2:{np.rad2deg(P_EE[2]):.3f}"
                )
                print(
                    f"Movement %: {index: .0f}%. tau_1: {tau_1}, tau_2: {tau_2}"
                )
                sys.stdout.write(f"\x1b[4A\x1b[2K")

                start_time = t
        del loop

    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Shutting down...")


if __name__ == "__main__":
    with CubemarsMotor(motor_type="AK70-10", logging=True) as motor_1:
        with CubemarsMotor(motor_type="AK60-6", logging=True) as motor_2:
            main(motor_1, motor_2)
