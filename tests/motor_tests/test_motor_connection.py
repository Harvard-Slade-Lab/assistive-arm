import os
import time

from pathlib import Path


from assistive_arm.motor_control import CubemarsMotor

def main(motor: CubemarsMotor):
    motor.send_zero_position()
        

if __name__ == "__main__":
    with CubemarsMotor(motor_type="AK60-6", logging=False) as motor:
        main(motor)
