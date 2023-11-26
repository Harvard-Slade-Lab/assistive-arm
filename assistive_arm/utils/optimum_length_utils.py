import numpy as np
import pandas as pd


def get_rotation_matrix(degrees: float) -> np.array:
    return np.array(
        [
            [np.cos(np.deg2rad(degrees)), -np.sin(np.deg2rad(degrees)), 0],
            [np.sin(np.deg2rad(degrees)), np.cos(np.deg2rad(degrees)), 0],
            [0, 0, 1],
        ]
    )

def get_jacobian(l1: float, l2: float, N: int, theta_1: float, theta_2: float) -> np.array:
    jacobian = np.array(
        [
            [
                -l1 * np.sin(theta_1) - l2 * np.sin(theta_1 + theta_2),
                -l2 * np.sin(theta_1 + theta_2),
            ],
            [
                l1 * np.cos(theta_1) + l2 * np.cos(theta_1 + theta_2),
                l2 * np.cos(theta_1 + theta_2),
            ],
            np.ones((2, N)),
        ]
    )

    return np.transpose(jacobian, (2, 0, 1))  # Bring jacobian to correct shape


def compute_torque_profiles(
    l1: float, l2: float, F: pd.DataFrame, position: pd.DataFrame
) -> tuple:
    """Compute the torque at the shoulder joint given the position of the end effector and the force applied to it.

    Args:
        l1 (float): length of the first link
        l2 (float): length of the second link
        position (N, 3): dataframe with the position of the end effector over time
        elbow_up (int, optional): 1 if the elbow is up, -1 if the elbow is down. Defaults to 1.

    Returns:
        np.array: torque array
    """
    N = position.shape[0]

    # Rotate EE position to robot frame
    rotate_ee = get_rotation_matrix(-90)
    pos_rot = (
        rotate_ee @ position.T
    ).T  # We apply the transpose to get the correct shape
    pos_rot.columns = ["X", "Y", "Z"]

    arccos_argument = (pos_rot.X**2 + pos_rot.Y**2 - l1**2 - l2**2) / (
        2 * l1 * l2
    )
    if np.any(arccos_argument > 1) or np.any(arccos_argument < -1):
        return np.nan, np.nan, np.nan

    theta_2 = np.arccos(arccos_argument)  # *(1 if elbow_up else -1)
    theta_1 = np.arctan2(pos_rot.Y, pos_rot.X) - np.arctan2(
        l2 * np.sin(theta_2), l1 + l2 * np.cos(theta_2)
    )

    jacobian = get_jacobian(l1, l2, N, theta_1, theta_2)
    # Force vector in robot frame, negate because we want to push
    rotate_forces = get_rotation_matrix(90)
    F_rot = -(rotate_forces @ F.T).T.to_numpy()
    F_rot = F_rot[:, :-1]  # remove Z component

    # Build force and torque vector (N, 3, 1))
    torque = np.cross(pos_rot, F_rot)[:, -1].reshape(N, 1)
    F_tot = np.concatenate((F_rot, torque), axis=1).reshape(N, 3, 1)

    torque = (jacobian.transpose((0, 2, 1)) @ F_tot).squeeze()
    joint_angles = pd.concat((theta_1, theta_2), axis=1)
    joint_angles.columns = ["theta_1", "theta_2"]

    return torque, joint_angles, jacobian
