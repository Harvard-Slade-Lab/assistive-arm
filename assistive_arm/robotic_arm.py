import numpy as np

from assistive_arm.motor_control import CubemarsMotor

class AssistiveArm:
    def __init__(self):
        self.motors = [
            CubemarsMotor(motor_type="AK70-10", csv_file="motor_data.csv"),
            CubemarsMotor(motor_type="AK60-6", csv_file="motor_data.csv"),
        ]


def calculate_ee_pos(motor_1: CubemarsMotor, motor_2: CubemarsMotor):
    L1 = 0.44
    L2 = 0.41

    P_EE = np.array([
        L1*np.cos(motor_1.position) + L2*np.cos(motor_1.position + motor_2.position),
        L1*np.sin(motor_1.position) + L2*np.sin(motor_1.position + motor_2.position),
        motor_1.position + motor_2.position])

    return P_EE