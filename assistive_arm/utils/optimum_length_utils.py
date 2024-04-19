import numpy as np
import pandas as pd


def get_rotation_matrix(degrees: float) -> np.array:
    """ Get 3x3 rotation matrix

    Args:
        degrees (float): rotate by degrees (rad)

    Returns:
        np.array: np array of 3x3 matrix
    """
    return np.array(
        [
            [np.cos(np.deg2rad(degrees)), -np.sin(np.deg2rad(degrees)), 0],
            [np.sin(np.deg2rad(degrees)), np.cos(np.deg2rad(degrees)), 0],
            [0, 0, 1],
        ]
    )

def check_theta(series: pd.Series, theta_lims: np.array) -> pd.Series:
    """ Check if angles are within allowed limits

    Args:
        series (pd.Series): target angles
        theta_lims (np.array): angle range

    Returns:
        pd.Series: filtered series
    """
    return series.apply(lambda x: theta_lims[0] <= x <= theta_lims[1]).all()

def get_jacobian(l1: float, l2: float, theta_1: pd.Series, theta_2: pd.Series) -> np.array:
    """ Get jacobian matrix

    Args:
        l1 (float): link 1
        l2 (float): link 2
        theta_1 (float): angle series
        theta_2 (float): angle series

    Returns:
        np.array: jacobian matrix
    """
    jacobian = np.array(
        [
            [
                -l1 * np.sin(theta_1) - l2 * np.sin(theta_1 + theta_2),
                -l2 * np.sin(theta_1 + theta_2),
            ],
            [
                l1 * np.cos(theta_1) + l2 * np.cos(theta_1 + theta_2),
                l2 * np.cos(theta_1 + theta_2),
            ]
        ]
    )
    return jacobian


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
        pd.DataFrame: torques at joints over time
        pd.DataFrame: joint angles over time
        np.array: jacobian over time (not transposed)
    """
    N = position.shape[0]

    # Get rotation matrices
    rotate_ee = get_rotation_matrix(-90) # Rotate EE position to robot frame
    rotate_forces = get_rotation_matrix(90) # Rotate forces to robot frame by 90 degrees

    pos_rot = (rotate_ee @ position.T).T  # Get correct shape
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

    thetas = pd.concat([theta_1, theta_2], axis=1, keys=["theta_1", "theta_2"])

    jacobian = get_jacobian(l1, l2, theta_1, theta_2)
    
    # Rotate force vector to robot frame
    F_rot = -(F @ rotate_forces).drop(2, axis=1)
    F_rot = F_rot.to_numpy().reshape(N, 2, 1)

    torques = (jacobian.T @ F_rot).squeeze()
    torques = pd.DataFrame(torques, columns=["tau_1", "tau_2"])



    return torques, thetas, jacobian


def interpolate_dataframe(df: pd.DataFrame, desired_frequency: int=200) -> pd.DataFrame:
    """ Interpolate dataframe to target frequency

    Args:
        df (pd.DataFrame): target dataframe
        desired_frequency (int, optional): target frequency (Hz). Defaults to 200.

    Returns:
        pd.DataFrame: interpolated dataframe
    """
    df_index = df.index
    df_index_new = pd.Index(np.arange(df_index.min(), df_index.max(), 1/desired_frequency), name="Time")

    df_interpolated = df.reindex(df_index_new, method="nearest").interpolate(method="polynomial", order=2)

    return df_interpolated

def smooth_dataframe(df: pd.DataFrame, window_size: int=30) -> pd.DataFrame:
    """ Smooth dataframe to filter out noise

    Args:
        df (pd.DataFrame): target dataframe
        window_size (int, optional): window size. Defaults to 30.

    Returns:
        pd.DataFrame: smoothed dataframe
    """
    df_smoothed = df.copy().rolling(window=window_size, min_periods=1, center=True).mean()
    return df_smoothed