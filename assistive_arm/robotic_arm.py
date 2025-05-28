import numpy as np
import pandas as pd

from assistive_arm.motor_control import CubemarsMotor


def calculate_ee_pos(theta_1: CubemarsMotor, theta_2: CubemarsMotor):
    L1 = 0.44
    L2 = 0.41

    P_EE = np.array([
        L1*np.cos(theta_1) + L2*np.cos(theta_1 + theta_2),
        L1*np.sin(theta_1) + L2*np.sin(theta_1 + theta_2),
        theta_1 + theta_2])

    return P_EE

def get_jacobian(theta_1: float, theta_2: float) -> np.array:
    L1 = 0.44
    L2 = 0.41

    jacobian = np.array(
        [
            [
                -L1 * np.sin(theta_1) - L2 * np.sin(theta_1 + theta_2),
                -L2 * np.sin(theta_1 + theta_2),
            ],
            [
                L1 * np.cos(theta_1) + L2 * np.cos(theta_1 + theta_2),
                L2 * np.cos(theta_1 + theta_2),
            ]
        ]
    )

    return jacobian


def get_target_torques(theta_1: float, theta_2: float, current_phase: float, profiles: pd.DataFrame) -> tuple:
    """ Get target torques for a given configuration, based on optimal profile

    Args:
        theta_1 (float): motor_1 angle
        theta_2 (float): motor_2 angle
        current_phase (float): current phase of the profile
        profiles (pd.DataFrame): optimal profile dataframe

    Returns:
        tuple: torques (tau_1, tau_2), index (percentage of profile)
    """

    P_EE = calculate_ee_pos(theta_1=theta_1, theta_2=theta_2)
    jacobian = get_jacobian(theta_1, theta_2)

    closest_point = abs(profiles.phase_baseline - current_phase).argmin() # it's an index
    force_vector = profiles.iloc[closest_point][["force_X", "force_Y"]]

    tau_1, tau_2 = -jacobian.T @ force_vector

    index = current_phase
    force_vector = np.array([force_vector.force_X, force_vector.force_Y]).reshape((2, 1))

    return tau_1, tau_2, P_EE, index, closest_point, force_vector



