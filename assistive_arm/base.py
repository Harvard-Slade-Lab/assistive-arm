import numpy as np

from abc import ABC, abstractmethod


class BaseArm(ABC):
    """Base class for the assistive 3rd arm"""

    @abstractmethod
    def get_pose() -> np.ndarray:
        """ Get the pose of the end effector in robot frame

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
    def get_joint_position(joint: int) -> np.ndarray:
        """Get 3D position of a given joint

        Args:
            joint (int): joint number starting from 0
        """
        pass

    @abstractmethod
    def get_link_position(link: int):
        # TODO Decide if we want to take the CoG as the link position or the origin
        pass

    @abstractmethod
    def set_joint_positions(joint: int, position: np.ndarray):
        pass


class AssistiveArm(BaseArm):
    def __init__(self):
        self.joints = []


class Joint:
    def __init__(self, limits: np.ndarray, name: str) -> None:
        self.qpos = None
        self.limits = limits
        self.name = name
    
    def get_qpos(self) -> float:
        return self.qpos
    
    def get_limits(self) -> np.ndarray:
        return self.limits
    
    def set_limits(self, limits: np.ndarray) -> None:
        self.limits = limits