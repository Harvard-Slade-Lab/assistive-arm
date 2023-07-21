import zmq

from time import sleep

from assistive_arm.base import AssistiveArm
from assistive_arm.network.client import setup_logger, request_data


def main():
    logger = setup_logger()
    context = zmq.Context()
    logger.info("Connecting to server...")
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://10.245.250.27:5555")

    arm = AssistiveArm()
    arm.forward(theta_1=45, theta_2=90)

    while True:
        markers, force_data = request_data(logger, socket)
        print("Joint positions (fw kin): ", arm.get_joint_positions())
        # Get marker 2D coordinates (x, y)
        print("Error: ")

    arm.joints[0]._motor.cleanup()


if __name__ == "__main__":
    main()
