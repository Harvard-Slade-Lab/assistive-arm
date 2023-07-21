from time import sleep

from assistive_arm.base import AssistiveArm

def main():
    arm = AssistiveArm()
    # arm.joints[0]._motor._gpio_pin = 15
    # arm.joints[1]._motor._gpio_pin = 16
    try:
        while True:
            arm.forward(theta_1=45, theta_2=90)
            sleep(1)
    except KeyboardInterrupt:
        arm._cleanup_ports()

if __name__ == "__main__":
    main()