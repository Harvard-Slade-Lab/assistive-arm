import sys
import numpy as np

from NeuroLocoMiddleware.SoftRealtimeLoop import SoftRealtimeLoop
from assistive_arm.motor_control import CubemarsMotor
from assistive_arm.base import AssistiveArm

np.set_printoptions(precision=3, suppress=True)

def main(motor_1: CubemarsMotor, motor_2: CubemarsMotor):
    freq = 200 # Hz
    loop = SoftRealtimeLoop(dt=1/freq, report=True, fade=0)

    start_time = 0

    L1 = 0.44
    L2 = 0.41

    try:
        for t in loop:
            cur_time = t
            motor_1.send_torque(desired_torque=0)
            motor_2.send_torque(desired_torque=0)

            P_EE = np.array([
                L1*np.cos(motor_1.position) + L2*np.cos(motor_1.position + motor_2.position),
                L1*np.sin(motor_1.position) + L2*np.sin(motor_1.position + motor_2.position),
                motor_1.position + motor_2.position])

            if cur_time - start_time > 0.05:
                motor_1.print_state(motor_2, P_EE)

                start_time = cur_time
        del loop


    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Shutting down...")

    

if __name__ == "__main__":
    with CubemarsMotor(motor_type="AK70-10", csv_file="motor_data.csv") as motor_1:
        with CubemarsMotor(motor_type="AK60-6", csv_file="motor_data.csv") as motor_2:
            main(motor_1, motor_2)