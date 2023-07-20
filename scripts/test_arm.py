from time import sleep

from assistive_arm.base import AssistiveArm

def main():
    sleep(1)
    arm = AssistiveArm()
    print("ports: ", arm.joints[0].ports)
    arm.forward(theta_1=45, theta_2=135)

if __name__ == '__main__':
    main()