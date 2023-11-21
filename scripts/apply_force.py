import numpy as np
import time
import pandas as pd

from pathlib import Path

from NeuroLocoMiddleware.SoftRealtimeLoop import SoftRealtimeLoop
from TMotorCANControl.mit_can import TMotorManager_mit_can

# CHANGE THESE TO MATCH YOUR DEVICE!
ID_1 = 1
ID_2 = 2

Type_1 = 'AK70-10'
Type_2 = 'AK60-6'

def load_assistive_profile(control_solution: Path):
    df = pd.read_csv(control_solution)


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


def provide_assistance(joint_1, joint_2):

    # Set up motors for control
    joint_1.set_zero_position()
    joint_2.set_zero_position()
    time.sleep(1.5) # wait for the motors to zero (~1 second)
    joint_1.set_impedance_gains_real_unit(K=10.0,B=0.5)
    joint_2.set_impedance_gains_real_unit(K=10.0,B=0.5)
    
    print("Starting 2 DOF demo. Press ctrl+C to quit.")


    # 200Hz control loop
    dt = 1 / 200
    K_p = 10
    K_d = 0.5

    torque_1_controller = PIDController(kp=10, kd=0.5, ki=0, dt=dt)
    torque_2_controller = PIDController(kp=10, kd=0.5, ki=0, dt=dt)

    loop = SoftRealtimeLoop(dt = dt, report=True, fade=0)

    for t in loop:
        # Get motor states
        joint_1.update()
        joint_2.update()

        # Read height of the hip joint / get hip angle
        hip_position = get_hip_position()

        # Turn hip position into percentage
        motion_percent = hip_position / (hip_profile.max() - hip_profile.min())

        tau_1_des, tau_2_des = torque_profile.iloc[motion_percent]

        error_1 = tau_1_des - joint_1.get_motor_torque_newton_meters()
        error_2 = tau_2_des - joint_2.get_motor_torque_newton_meters()
        
        joint_1.torque = torque_1_controller.update(error_1)
        joint_2.torque = torque_2_controller.update(error_2)

        # implement torque control
        


    del loop

if __name__ == '__main__':
    # to use additional motors, simply add another with block
    # remember to give each motor a different log name!
    with TMotorManager_mit_can(motor_type=Type_1, motor_ID=ID_1) as base_motor:
        with TMotorManager_mit_can(motor_type=Type_2, motor_ID=ID_2) as elbow_motor:
            provide_assistance(base_motor,elbow_motor)