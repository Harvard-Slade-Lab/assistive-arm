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
            markers, _ = get_qrt_data(logger=client_logger, socket=socket)
            marker_array = np.vstack(list(markers.values()))
            # We assume that the first marker is the robot base
            offset = np.linalg.norm(marker_array - poses, axis=1)
            
            for joint_position, coordinates in zip(poses, markers.values()):
                offset = np.linalg.norm(joint_position - coordinates)
                print("Joint Position (WF) | Marker Position (WF) | Offset (mm)")
                print(f"{joint_position} | {coordinates} | {offset}")
            print("\n")

    except KeyboardInterrupt:
        arm._cleanup_ports()


if __name__ == "__main__":
    main()
