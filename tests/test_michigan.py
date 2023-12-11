from NeuroLocoMiddleware.SoftRealtimeLoop import SoftRealtimeLoop
import numpy as np
import time
from TMotorCANControl.mit_can import TMotorManager_mit_can

# CHANGE THESE TO MATCH YOUR DEVICE!
Type = 'AK70-10'
ID = 1

def position_tracking(dev):
    dev.set_zero_position() # has a delay!
    time.sleep(1.5)
    dev.set_impedance_gains_real_unit(K=10,B=0.5)

    print("Starting position tracking demo. Press ctrl+C to quit.")

    loop = SoftRealtimeLoop(dt = 0.01, report=True, fade=0)
    for t in loop:
        dev.update()
        if t < 1.0:
            dev.position = 0.0
        else:
            print(dev.get_motor_torque_newton_meters())
    
    del loop

if __name__ == '__main__':
    with TMotorManager_mit_can(motor_type=Type, motor_ID=ID) as dev:
        position_tracking(dev)