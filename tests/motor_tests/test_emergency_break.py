import csv
import numpy as np
import os
import time

from NeuroLocoMiddleware.SoftRealtimeLoop import SoftRealtimeLoop
from TMotorCANControl.mit_can import TMotorManager_mit_can

# CHANGE THESE TO MATCH YOUR DEVICE!
Type = 'AK60-6'
ID = 2

def position_tracking(dev):
    dev.set_zero_position() # has a delay!
    time.sleep(1.5)
    dev.set_impedance_gains_real_unit(K=10,B=0.5)

    print("Starting emergency break demo. Press ctrl+C to trigger.")

    dt = 0.01

    loop = SoftRealtimeLoop(dt = dt, report=True, fade=0)

    prev_speed = 0.0

    for t in loop:
        dev.update()
        
        cur_speed = dev.get_motor_velocity_radians_per_second()

        if abs(prev_speed - cur_speed) / dt > 5:
            print("Aborting...")
            exit()


        if t < 1.0:
            dev.torque = 0.0
        else:
            dev.torque = 0.5

        prev_speed = cur_speed
    
    del loop

if __name__ == '__main__':
    # Create CSV file for later analysis, naming it with current time
    log_file = f"../logs/{time.strftime('%Y-%m-%d-%H-%M-%S')}.csv"
    os.system(f"touch {log_file}")

    with TMotorManager_mit_can(motor_type=Type, motor_ID=ID, CSV_file=log_file) as dev:
        position_tracking(dev)