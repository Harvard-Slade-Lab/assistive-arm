import os
import time

from pathlib import Path


from assistive_arm.motor_control import CubemarsMotor

def main(motor: CubemarsMotor):
    motor.send_zero_position()
        

if __name__ == "__main__":
    filename = os.path.basename(__file__)
    log_file = Path(f"../logs/{filename.split('.')[0]}_{time.strftime('%m-%d-%H-%M-%S')}.csv")
    with CubemarsMotor(motor_type="AK60-6", csv_file=log_file) as motor:
        main(motor)
