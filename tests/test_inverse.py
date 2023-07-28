from assistive_arm.base import AssistiveArm

def main():
    arm = AssistiveArm()

    arm.inverse(x=0.3, y=0.3)

    arm._cleanup_ports()

if __name__ == '__main__':
    main()