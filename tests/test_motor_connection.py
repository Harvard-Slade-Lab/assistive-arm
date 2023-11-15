import os
import can
import traceback
from time import sleep

from assistive_arm.motor_control import connect_motor, stop_motor

def main():
    os.system("sudo ip link set can0 type can bitrate 1000000")
    os.system("sudo ip link set can1 type can bitrate 1000000")
    os.system("sudo ifconfig can0 up")
    os.system("sudo ifconfig can1 up")

    base_motor = 1 # 0x01
    elbow_motor = 2 # 0x02

    can0 = can.Bus(channel="can0", bustype="socketcan")
    can1 = can.Bus(channel="can1", bustype="socketcan")

    connect_motor(can_bus=can0, motor_id=base_motor)
    connect_motor(can_bus=can1, motor_id=elbow_motor)

    sleep(1)

    stop_motor(can_bus=can0, motor_id=base_motor)
    stop_motor(can_bus=can1, motor_id=elbow_motor)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
    finally:
        os.system("sudo ifconfig can0 down")
        os.system("sudo ifconfig can1 down")
