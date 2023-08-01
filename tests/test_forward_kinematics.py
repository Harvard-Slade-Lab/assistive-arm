""" The goal of this test is to determine the error when computing 3D position using forward kinematics, and cross-checking with the position of the end effector in the robot frame obtained using inverse kinematics"""
import numpy as np

from time import sleep

from assistive_arm.base import AssistiveArm
from assistive_arm.network.client import (
    setup_client_logger,
    get_qrt_data,
    connect_to_server,
)


def main():
    client_logger = setup_client_logger()
    socket = connect_to_server(logger=client_logger)

    arm = AssistiveArm()
    arm.forward(theta_1=0, theta_2=0)

    poses = arm.get_joint_positions()

    try:
        while True:
            print()
            markers, _ = get_qrt_data(logger=client_logger, socket=socket)
            marker_array = np.vstack(list(markers.values()))
            # We assume that the first marker is the robot base
            for marker, coordinates in markers.items():
                print(f"{marker}: {coordinates}")

            print("Robot pose (robot frame): ", poses)
            error = np.linalg.norm(marker_array - poses, axis=1)
            print(f"Offset: {error} mm")


    except KeyboardInterrupt:
        arm._cleanup_ports()
        exit(0)


if __name__ == "__main__":
    main()
