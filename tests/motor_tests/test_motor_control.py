import sys

from NeuroLocoMiddleware.SoftRealtimeLoop import SoftRealtimeLoop
from assistive_arm.motor_control import CubemarsMotor

def main(motor_1: CubemarsMotor):
    # Start control loop
    freq = 200 # Hz
    
    loop = SoftRealtimeLoop(dt = 1/freq, report=True, fade=0)
    start_time = 0
    pos = 0
    # General control loop
    try:
        for t in loop:
            cur_time = t
            pos, vel, torque = motor_1.send_torque(desired_torque=0)

            if cur_time - start_time > 0.1:
                sys.stdout.write('\x1b[1A\x1b[2K')
                print(f"Angle: {pos: .3f} Velocity: {vel: .3f} Torque: {torque: .3f}")
        del loop


    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Shutting down...")
    

if __name__ == "__main__":
    with CubemarsMotor('AK70-10') as motor_1:
        main(motor_1)