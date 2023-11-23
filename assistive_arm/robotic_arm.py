from assistive_arm.motor_control import CubemarsMotor

class AssistiveArm:
    def __init__(self):
        self.motors = [
            CubemarsMotor(motor_type="AK70-10", csv_file="motor_data.csv"),
            CubemarsMotor(motor_type="AK60-6", csv_file="motor_data.csv"),
        ]