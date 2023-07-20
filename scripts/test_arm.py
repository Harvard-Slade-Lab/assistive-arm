from time import sleep

from assistive_arm.base import AssistiveArm

def main():
    arm = AssistiveArm()
    print("ports: ", arm.joints[0].ports)
    arm.forward(theta_1=45, theta_2=90)

    print(arm.get_joint_angles())
    print(arm.get_joint_positions())
    print(arm.get_pose())

    arm.joints[0]._motor.cleanup()

if __name__ == '__main__':
    main()