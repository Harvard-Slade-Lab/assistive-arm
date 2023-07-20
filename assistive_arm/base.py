import numpy as np

from abc import ABC, abstractmethod
from typing import List

from assistive_arm.servo_control import ServoControl


class BaseArm(ABC):
    """Base class for the assistive 3rd arm"""

    @abstractmethod
    def get_pose() -> np.ndarray:
        """Get the pose of the end effector in robot frame

        Returns:
            np.ndarray: 3D pose of the end effector
        """
        pass

    @abstractmethod
    def get_velocity() -> np.ndarray:
        """Get the velocity of the end effector in robot frame

        Returns:
            np.ndarray: 3D velocity of the end effector
        """
        pass

    @abstractmethod
    def get_joint_positions(joint: List[int]) -> np.ndarray:
        """Get 3D position of a given joint

        Args:
            joint (int): joint number starting from 0
        """
        pass

    @abstractmethod
    def get_joint_angles() -> np.ndarray:
        """Get the joint angles

        Args:
            joint (int): joint number starting from 0
        """
        pass

    @abstractmethod
    def get_link_position(link: int):
        # TODO Decide if we want to take the CoG as the link position or the origin
        pass

    @abstractmethod
    def set_joint_positions(position: List[int]):
        pass


class AssistiveArm(BaseArm):
    def __init__(self):
        self.joints = [
            Joint(name="joint_1"),
            Joint(name="joint_2"),
        ]

        self.link_length = 0.3
        self.dist_links = 0.03

        self._T_0_1 = None
        self._T_1_2 = None
        self._T_2_3 = None

    def set_joint_positions(self, positions: np.ndarray):
        for joint, position in zip(self.joints, positions):
            if np.isnan(position):
                continue
            joint.set_qpos(position)

    def get_link_position(link: int):
        pass

    def get_velocity(self) -> np.ndarray:
        pass

    def forward(self, theta_1: int, theta_2: int):
        """Forward kinematics of the arm
        Args:
            theta_1 (int): angle of joint 1
            theta_2 (int): angle of joint 2
        Returns:
            np.ndarray: x, y, z position of the end effector
        """
        # TODO Implement forward kinematics
        self._T_0_1 = np.array(
            [
                [np.cos(theta_1), -np.sin(theta_1), 0, 0],
                [np.sin(theta_1), np.cos(theta_1), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        self._T_0_2 = np.array(
            [
                [
                    np.cos(theta_1 + theta_2),
                    -np.sin(theta_1 + theta_2),
                    0,
                    self.link_length * np.cos(theta_1),
                ],
                [
                    np.sin(theta_1 + theta_2),
                    np.cos(theta_1 + theta_2),
                    0,
                    self.link_length * np.sin(theta_1),
                ],
                [0, 0, 1, self.dist_links],
                [0, 0, 0, 1],
            ]
        )
        self._T_0_3 = np.array(
            [
                [
                    0,
                    -np.sin(theta_1 + theta_2),
                    0,
                    self.link_length * (np.cos(theta_1) + np.cos(theta_1 + theta_2)),
                ],
                [
                    0,
                    np.cos(theta_1 + theta_2),
                    0,
                    self.link_length * (np.sin(theta_1) + np.sin(theta_1 + theta_2)),
                ],
                [0, 0, 1, self.dist_links],
                [0, 0, 0, 1],
            ]
        )

        self.transformations = [self._T_0_1, self._T_0_2, self._T_0_3]
        self.set_joint_positions(np.array([theta_1, theta_2]))

        for joint, transf_matrix in zip(self.joints[:-1], self.transformations[:-1]):
            joint.pose = transf_matrix[:3, 3]

        return self._T_0_3[:3, 3]

    def get_pose(self) -> np.ndarray:
        """Return 3D pose of the end effector in robot frame
        Returns:
            np.ndarray: x, y, z position of the end effector
        """
        return self.T_0_3[:3, 3]


    def get_joint_positions(self) -> np.ndarray:
        """Get 3D position of a given joint

        Args:
            joint (int): joint number starting from 0
        """

        return np.array([transform[:3, 3] for transform in self.transformations])

    def get_joint_angles(self) -> np.ndarray:
        return np.array([joint.get_qpos() for joint in self.joints])


class Joint:
    ports = [11, 13, 15, 16, 18, 22] # List of ports that can be used for the servos
    # See https://www.raspberrypi.com/documentation/computers/images/GPIO-Pinout-Diagram-2.png

    def __init__(self, name: str) -> None:
        self.qpos = None
        self.name = name
        self.pose = None
        self._motor = ServoControl(pin=self._assign_port())
        self.limits = self._motor.angle_range

    def get_qpos(self) -> float:
        # TODO This returns last set angle, modify to read from servo using a control loop
        return self.qpos

    def get_pose(self) -> np.ndarray:
        return self.pose

    def set_qpos(self, angle: int) -> None:
        self._motor.set_angle(angle)
        self.qpos = angle

    def get_limits(self) -> np.ndarray:
        return self.limits

    def set_limits(self, limits: np.ndarray) -> None:
        self.limits = limits
        self._motor.angle_range.min = limits[0]
        self._motor.angle_range.max = limits[1]

    def _assign_port(self):
        try:
            return self.ports.pop(0)
        except IndexError:
            raise Exception("No more available GPIO ports")
