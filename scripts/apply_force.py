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


def provide_assistance(joint_1, joint_2):

    # Set up motors for control
    joint_1.set_zero_position()
    joint_2.set_zero_position()
    time.sleep(1.5) # wait for the motors to zero (~1 second)
    joint_1.set_impedance_gains_real_unit(K=10.0,B=0.5)
    joint_2.set_impedance_gains_real_unit(K=10.0,B=0.5)
    
    print("Starting 2 DOF demo. Press ctrl+C to quit.")

    # 200Hz control loop
    loop = SoftRealtimeLoop(dt = 0.005, report=True, fade=0)

    for t in loop:
        # Get motor states
        joint_1.update()
        joint_2.update()

        # Read height of the hip joint / get hip angle
        hip_position = get_hip_position()

        # Turn hip position into percentage
        motion_percent = hip_position / (hip_profile.max() - hip_profile.min())

        target_tau_1, target_tau_2 = torque_profile.iloc[motion_percent]

        joint_1.torque = target_tau_1
        joint_2.torque = target_tau_2

        # implement torque control
        


    del loop

if __name__ == '__main__':
    # to use additional motors, simply add another with block
    # remember to give each motor a different log name!
    with TMotorManager_mit_can(motor_type=Type_1, motor_ID=ID_1) as base_motor:
        with TMotorManager_mit_can(motor_type=Type_2, motor_ID=ID_2) as elbow_motor:
            provide_assistance(base_motor,elbow_motor)