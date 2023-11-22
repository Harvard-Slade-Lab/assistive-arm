import os
import sys
import time
import numpy as np

from NeuroLocoMiddleware.SoftRealtimeLoop import SoftRealtimeLoop
from assistive_arm.motor_control import CubemarsMotor

def main(motor_1: CubemarsMotor):
    # Start control loop
    freq = 200 # Hz
    
    loop = SoftRealtimeLoop(dt = 1/freq, report=True, fade=0)
    start_time = 0

    # General control loop
    try:
        for t in loop:
            motor_1.send_torque(desired_torque=0)
            
            if t - start_time > 0.1:
                motor_1.print_state()
                start_time = t
        del loop


    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Shutting down...")
    

if __name__ == "__main__":
    filename = os.path.basename(__file__)
    log_file = f"../logs/{filename.split('.')[0]}_{time.strftime('%m-%d-%H-%M-%S')}.csv"
    os.system(f"touch {log_file}")

    with CubemarsMotor('AK60-6', csv_file=log_file) as motor_1:
        main(motor_1)