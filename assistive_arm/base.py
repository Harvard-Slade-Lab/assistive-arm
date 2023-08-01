import numpy as np

from abc import ABC, abstractmethod
from typing import List
from time import sleep
from threading import Thread

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
    def get_joint_positions() -> np.ndarray:
        """Get 3D position of the joints

        Returns:
            np.ndarray: 3D position of all joints
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
    def set_joint_angles(position: List[int]):
        pass


class AssistiveArm(BaseArm):
    def __init__(self):
        self.joints = [
            Joint(name="joint_1"),
            Joint(name="joint_2"),
        ]

        # Measurements in mm
        self.link_length = 250
        self.dist_links = 25

        # REMINDER: Offset z by dist_links
        self._T_W_0 = np.array([[0, -1, 0, 516.78],
                                [1, 0, 0, -1672.9],
                                [0, 0, 1, 732.7],
                                [0, 0, 0, 1]])

        self._T_W_1 = None
        self._T_W_2 = None
        self._T_W_3 = None

    def set_joint_angles(self, angles: np.ndarray):
        """Set the joint to a given angle and move motors
        Args:
            positions (np.ndarray): array of joint angles [theta_1, theta_2]
        """
        threads = []
        for joint, angle in zip(self.joints, angles):
            if np.isnan(angle):
                continue
            # joint.set_qpos(angle)
            thread = Thread(target=joint.set_qpos, args=(angle,))
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()

    def get_link_position(link: int):
        """Get 3D pose of a given link

        Args:
            link (int): link number starting from 0
        """
        # TODO
        pass

    def get_velocity(self) -> np.ndarray:
        """Get the velocity of the end effector in robot frame

        Returns:
            np.ndarray: [Vx, Vy, Vz]
        """
        # TODO Define which frame we want to use for the velocity
        pass

    def forward(self, theta_1: int, theta_2: int):
        """Forward kinematics of the arm
        Args:
            theta_1 (int): angle of joint 1
            theta_2 (int): angle of joint 2
        Returns:
            np.ndarray: x, y, z position of the end effector
        """
        # TODO Check if forward kinematics is correct
        rad_1 = self._get_radians(theta_1)
        rad_2 = self._get_radians(theta_2)

        # For all transformations, _T_W_0 * T_0_N
        self._T_W_1 = self._T_W_0 @ np.array(
            [
                [np.cos(rad_1), -np.sin(rad_1), 0, 0],
                [np.sin(rad_1), np.cos(rad_1), 0, 0],
                [0, 0, 1, self.z_offset],
                [0, 0, 0, 1],
            ]
        )
        self._T_W_2 = self._T_W_0 @ np.array(
            [
                [
                    np.cos(rad_1 + rad_2),
                    -np.sin(rad_1 + rad_2),
                    0,
                    self.link_length * np.cos(rad_1),
                ],
                [
                    np.sin(rad_1 + rad_2),
                    np.cos(rad_1 + rad_2),
                    0,
                    self.link_length * np.sin(rad_1),
                ],
                [0, 0, 1, 2 * self.z_offset],
                [0, 0, 0, 1],
            ]
        )
        self._T_W_3 = self._T_W_0 @ np.array(
            [
                [
                    np.cos(rad_1 + rad_2),
                    -np.sin(rad_1 + rad_2),
                    0,
                    self.link_length * (np.cos(rad_1) + np.cos(rad_1 + rad_2)),
                ],
                [
                    np.sin(rad_1 + rad_2),
                    np.cos(rad_1 + rad_2),
                    0,
                    self.link_length * (np.sin(rad_1) + np.sin(rad_1 + rad_2)),
                ],
                [0, 0, 1, 2 * self.z_offset],
                [0, 0, 0, 1],
            ]
        )

        self.transformations = [self._T_W_1, self._T_W_2, self._T_W_3]
        self.set_joint_angles(np.array([theta_1, theta_2]))

        for joint, transf_matrix in zip(self.joints[:-1], self.transformations[:-1]):
            joint.pose = transf_matrix[:3, 3]

        return self._T_W_3[:3, 3]

    def inverse(self, x: float, y: float) -> np.array:
        """Inverse kinematics of the arm

        Args:
            x (float): x position of the end effector
            y (float): y position of the end effector

        Returns:
            np.array: [theta_1, theta_2]
        """
        theta_2 = -np.arccos(
            (x**2 + y**2 - 2 * self.link_length**2) / (2 * self.link_length**2)
        )
        theta_1 = np.arctan(y / x) + np.arctan(
            self.link_length
            * np.sin(theta_2)
            / (self.link_length * (1 + np.cos(theta_2)))
        )
        # Save angles in degrees
        target_angles = np.degrees(np.array([theta_1, theta_2]))
        print("target: ", target_angles)

        self.set_joint_angles(target_angles)

        return target_angles

    def _get_radians(self, angle: int) -> float:
        """Convert angle from degrees to radians

        Args:
            angle (int): angle in degrees

        Returns:
            float: angle in radians
        """
        return angle * np.pi / 180

    def get_pose(self) -> np.ndarray:
        """Return 3D pose of the end effector in robot frame
        Returns:
            np.ndarray: x, y, z position of the end effector
        """
        return self._T_W_3[:3, 3]

    def get_joint_positions(self) -> np.ndarray:
        """Get 3D position of a given joint

        Args:
            joint (int): joint number starting from 0
        """

        return np.array([transform[:3, 3] for transform in self.transformations])

    def get_joint_angles(self) -> np.ndarray:
        """Read

        Returns:
            np.ndarray: _description_
        """
        return np.array([joint.get_qpos() for joint in self.joints])

    def _cleanup_ports(self) -> None:
        """Cleanup all the GPIO ports used by the servos"""
        self.joints[0]._motor.cleanup()


class Joint:
    ports = [11, 13, 15, 16, 18, 22]  # List of ports that can be used for the servos
    # See https://www.raspberrypi.com/documentation/computers/images/GPIO-Pinout-Diagram-2.png

    def __init__(self, name: str) -> None:
        self.qpos = 0
        self.name = name
        self.pose = None
        self._motor = ServoControl(pin=self._assign_port())
        self.limits = self._motor.angle_range

    def get_qpos(self) -> float:
        # TODO This returns last set angle, modify to read from servo using a control loop
        return self.qpos

    def get_pose(self) -> np.ndarray:
        """Return 3D pose of the joint in robot frame

        Returns:
            np.ndarray: np.array
        """
        return self.pose

    def set_qpos(self, angle: int) -> None:
        """Store the angle and move the motor

        Args:
            angle (int): angle in degrees
        """
        self._motor.set_angle(angle)
        self.qpos = angle
        sleep(0.05)
        # TODO Figure out what's the ideal sleep time for the motors
        # sleep(abs(angle)*0.14/(np.pi/3) + self.qpos)

    def get_limits(self) -> np.ndarray:
        """Get joint limits

        Returns:
            np.ndarray: [min, max]
        """
        return self.limits

    def set_limits(self, limits: np.ndarray) -> None:
        """Set new joint limits

        Args:
            limits (np.ndarray): new limits [min, max]
        """
        self.limits = limits
        self._motor.angle_range.min = limits[0]
        self._motor.angle_range.max = limits[1]

    def _assign_port(self):
        """Assign a GPIO port to the servo

        Raises:
            Exception: _description_

        Returns:
            _type_: _description_
        """
        try:
            return self.ports.pop(0)
        except IndexError:
            raise Exception("No more available GPIO ports")
