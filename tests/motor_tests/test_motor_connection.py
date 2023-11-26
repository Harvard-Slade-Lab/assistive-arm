from NeuroLocoMiddleware.SoftRealtimeLoop import SoftRealtimeLoop
from assistive_arm.motor_control import CubemarsMotor

def main(motor: CubemarsMotor):
    motor.send_zero_position()
        

if __name__ == "__main__":
    with CubemarsMotor(motor_type="AK60-6") as motor:
        main(motor)
