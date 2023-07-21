from time import sleep

from assistive_arm.base import AssistiveArm

def main():
    arm = AssistiveArm()

    arm.forward(theta_1=0, theta_2=0)
    sleep(1)
    try:
        arm.forward(theta_1=-45, theta_2=60)
        sleep(1)
        arm.forward(theta_1=45, theta_2=-60)
        sleep(1)
    except KeyboardInterrupt:
        arm._cleanup_ports()
    sleep(1)
    arm._cleanup_ports()

if __name__ == "__main__":
    main()