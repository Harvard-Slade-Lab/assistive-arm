import os
import can
import numpy as np
import traceback
import time

from NeuroLocoMiddleware.SoftRealtimeLoop import SoftRealtimeLoop

from assistive_arm.motor_control import connect_motor, stop_motor, update_motor

def main():
    os.system("sudo ip link set can0 type can bitrate 1000000")
    os.system("sudo ifconfig can0 up")

    base_motor = 1 # 0x01

    can0 = can.Bus(channel="can0", bustype="socketcan")

    connect_motor(can_bus=can0, motor_id=base_motor)

    # Start control loop
    freq = 200 # Hz
    
    loop = SoftRealtimeLoop(dt = 1/freq, report=True, fade=0)

    # General control loop
    try:
        for t in loop:
            if t < 0.5:
                cmd = [0, 0, 0, 0, 0]
                update_motor(can_bus=can0, motor_id=base_motor, cmd=cmd)
            else:
                cmd = [np.sin(np.pi*t), 0, 5, 0.2, 0]
                update_motor(can_bus=can0, motor_id=base_motor, cmd=cmd)
        del loop


    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Shutting down...")
    
    stop_motor(can_bus=can0, motor_id=base_motor)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
    finally:
        os.system("sudo ifconfig can0 down")
        os.system("sudo ifconfig can1 down")
