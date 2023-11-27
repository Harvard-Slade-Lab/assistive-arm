import numpy as np
import os
import pandas as pd
import sys
import time

from pathlib import Path

from NeuroLocoMiddleware.SoftRealtimeLoop import SoftRealtimeLoop
from assistive_arm.motor_control import CubemarsMotor
from assistive_arm.robotic_arm import calculate_ee_pos

def read_profiles(profile_path: Path) -> pd.DataFrame:
    profiles = pd.read_csv(profile_path, index_col="Percentage")

    return profiles

class PIDController:
    def __init__(self, kp: float, kd: float, ki: float, dt: float) -> None:
        self.kp = kp
        self.kd = kd
        self.ki = ki
        self.dt = dt

        self.error_prev = 0
        self.error_sum = 0

    def update(self, error: float) -> float:
        self.error_sum += error * self.dt
        error_d = (error - self.error_prev) / self.dt
        self.error_prev = error

        return self.kp * error + self.kd * error_d + self.ki * self.error_sum


def provide_assistance(motor_1: CubemarsMotor, motor_2: CubemarsMotor):
    # 200Hz control loop
    freq = 200

    loop = SoftRealtimeLoop(dt = 1/freq, report=True, fade=0)

    torque_1_controller = PIDController(kp=10, kd=0.5, ki=0, dt=1/freq)
    torque_2_controller = PIDController(kp=10, kd=0.5, ki=0, dt=1/freq)


    profile_path = Path("~/ability-lab/assistive-arm/torque_profiles/scaled_torque_profile.csv")

    profiles = read_profiles(profile_path=profile_path)
    profile_EE = profiles[["X", "Y", "theta_1_2"]]

    start_time = 0

    L1 = 0.44
    L2 = 0.41

    try:
        for i, t in enumerate(loop):
            # Read height of the hip joint / get hip angle

            P_EE = calculate_ee_pos(motor_1=motor_1, motor_2=motor_2)

            closest_point = np.linalg.norm(profile_EE - P_EE, axis=1).argmin()
            target_torque = profiles.iloc[closest_point][['tau_1', 'tau_2']]
            index = profiles.index[closest_point]

            
            try:
                motor_1.send_torque(desired_torque=-profiles.iloc[i].tau_1)
                motor_2.send_torque(desired_torque=-profiles.iloc[i].tau_2)
                print(profiles.iloc[i].tau_1, profiles.iloc[i].tau_2)
            except IndexError:
                print("Out of bounds")
                break

            # error_1 = target_torque.tau_1 - motor_1.torque
            # error_2 = target_torque.tau_2 - motor_2.torque
            
            # motor_1.send_torque(torque_1_controller.update(error_1))
            # motor_2.send_torque(torque_2_controller.update(error_2))


            
            if t - start_time > 0.05:

                print(f"{motor_1.type}: Angle: {motor_1.position:.3f} Velocity: {motor_1.velocity:.3f} Torque: {motor_1.torque:.3f}")
                print(f"{motor_2.type}: Angle: {motor_2.position:.3f} Velocity: {motor_2.velocity:.3f} Torque: {motor_2.torque:.3f}")
                print(f"P_EE: x:{P_EE[0]:.3f} y:{P_EE[1]:.3f} theta_1+theta_2:{np.rad2deg(P_EE[2]):.3f}")
                print(f"Movement %: {index: .0f}%. tau_1: {target_torque.tau_1}, tau_2: {target_torque.tau_2}")
                sys.stdout.write(f'\x1b[4A\x1b[2K')

                start_time = t

        del loop

    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Shutting down...")

if __name__ == '__main__':
    # to use additional motors, simply add another with block
    # remember to give each motor a different log name!

    filename = os.path.basename(__file__).split('.')[0]
    log_file = Path(f"../logs/{filename}_{time.strftime('%m-%d-%H-%M-%S')}.csv")

    with CubemarsMotor(motor_type="AK70-10", csv_file=log_file) as base_motor:
        with CubemarsMotor(motor_type="AK60-6", csv_file=log_file) as elbow_motor:
            provide_assistance(motor_1=base_motor, motor_2=elbow_motor)