import os
import time
import can

from pathlib import Path


from assistive_arm.motor_control import CubemarsMotor

def main(motor: CubemarsMotor):
    motor.send_zero_position()
        

if __name__ == "__main__":
    os.system(f"sudo ip link set can0 up type can bitrate 1000000")

    # Initialize the shared CAN bus
    can_bus = can.interface.Bus(channel="can0", bustype="socketcan")

    with CubemarsMotor(motor_type="AK70-10", frequency=200, can_bus=can_bus) as motor:
        main(motor)

    os.system(f"sudo ip link set can0 down")
