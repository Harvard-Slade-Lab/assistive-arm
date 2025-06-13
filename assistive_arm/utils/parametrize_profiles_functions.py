import os
import pandas as pd
import numpy as np
from chspy import CubicHermiteSpline
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, argrelextrema
from scipy.interpolate import CubicSpline, PchipInterpolator
import matplotlib.pyplot as plt
import scienceplots



def cubic_hermite_spline(points):
    """
    Create a cubic Hermite spline given a list of points.

    Args:
        points (list): List of tuples (time, value, derivative).

    Returns:
        CubicHermiteSpline: The constructed spline object.
    """
    spline = CubicHermiteSpline(n=1)
    for t, value, derivative in points:
        spline.add((t, [value], [derivative]))
    return spline



def get_profile(force1_end_time, force1_peak_force, force2_start_time, force2_peak_time, force2_peak_force, force2_end_time, length):
    """
    Generate force profiles for X and Y forces based on given parameters and roll angles.

    Args:
        force1_end_time (float): End time for the X force profile.
        force1_peak_force (float): Peak force for the X force profile.
        force2_start_time (float): Start time for the Y force profile.
        force2_peak_time (float): Peak time for the Y force profile.
        force2_peak_force (float): Peak force for the Y force profile.
        force2_end_time (float): End time for the Y force profile.
        roll_angles (pd.DataFrame): DataFrame containing roll angles with time indices.

    Returns:
        pd.DataFrame: DataFrame containing time-aligned force profiles for X and Y.
    """
    base_profile = pd.DataFrame({"force_X": np.zeros(length), "force_Y": np.zeros(length)})

    # X Force Profile
    grf_x = cubic_hermite_spline([(0, 0, 0), (force1_end_time / 2, force1_peak_force, 0), (force1_end_time, 0, 0)])
    curve_x = [grf_x.get_state(i)[0] for i in range(int(np.round(force1_end_time)))]
    padded_curve_x = np.concatenate([curve_x, np.zeros(length - len(curve_x))])

    # Y Force Profile
    grf_y = cubic_hermite_spline([(0, 0, 0), (force2_peak_time - force2_start_time, force2_peak_force, 0), (force2_end_time - force2_start_time, 0, 0)])
    curve_y = [grf_y.get_state(i)[0] for i in range(int(np.round(force2_end_time - force2_start_time)))]
    padded_curve_y = np.concatenate([np.zeros(int(np.round(force2_start_time))), curve_y, np.zeros(length - len(curve_y) - int(np.round(force2_start_time)))])

    base_profile["force_X"] = padded_curve_x
    base_profile["force_Y"] = padded_curve_y

    return base_profile

def get_profile_2(force1_peak_time, force1_peak_force, force2_start_time, force2_peak_time, force2_peak_force, force2_end_time, length):
    """
    Generate force profiles for X and Y forces based on given parameters and roll angles.

    Args:
        force1_peak_time (float): Peak time for the X force profile.
        force1_peak_force (float): Peak force for the X force profile.
        force2_start_time (float): Start time for the Y force profile.
        force2_peak_time (float): Peak time for the Y force profile.
        force2_peak_force (float): Peak force for the Y force profile.
        force2_end_time (float): End time for the Y force profile.
        roll_angles (pd.DataFrame): DataFrame containing roll angles with time indices.

    Returns:
        pd.DataFrame: DataFrame containing time-aligned force profiles for X and Y.
    """
    base_profile = pd.DataFrame({"force_X": np.zeros(length), "force_Y": np.zeros(length)})

    # X Force Profile
    grf_x = cubic_hermite_spline([(0, 0, 0), (force1_peak_time, force1_peak_force, 0), (length, 0, 0)])
    curve_x = [grf_x.get_state(i)[0] for i in range(length)]
    padded_curve_x = np.concatenate([curve_x, np.zeros(length - len(curve_x))])

    # Y Force Profile
    grf_y = cubic_hermite_spline([(0, 0, 0), (force2_peak_time - force2_start_time, force2_peak_force, 0), (force2_end_time - force2_start_time, 0, 0)])
    curve_y = [grf_y.get_state(i)[0] for i in range(int(np.round(force2_end_time - force2_start_time)))]
    padded_curve_y = np.concatenate([np.zeros(int(np.round(force2_start_time))), curve_y, np.zeros(length - len(curve_y) - int(np.round(force2_start_time)))])

    base_profile["force_X"] = padded_curve_x
    base_profile["force_Y"] = padded_curve_y

    return base_profile



def percentage_to_actual(force1_end_time_p, force1_peak_force_p, force2_start_time_p, force2_peak_time_p, force2_peak_force_p, force2_end_time_p, max_time):
    max_force = 55
    minimum_width_p = 0.2
    minimum_distance = max_time * minimum_width_p / 2
    
    force1_end_time = minimum_width_p * max_time + force1_end_time_p * max_time * (1 - minimum_width_p)
    force1_peak_force = force1_peak_force_p * max_force * 2/3
    # Dynamic constraints
    force2_peak_time = force2_peak_time_p * max_time * 0.7 + 0.15 * max_time # 0.15 to 0.85
    force2_start_time = (force2_peak_time - minimum_distance) * force2_start_time_p # 0 to 0.05 of peak time
    force2_end_time = force2_peak_time + minimum_distance + force2_end_time_p * (max_time - force2_peak_time - minimum_distance) # minimum, distance off peak time
    force2_peak_force = force2_peak_force_p * max_force

    return force1_end_time, force1_peak_force, force2_start_time, force2_peak_time, force2_peak_force, force2_end_time


def percentage_to_actual_2(force1_peak_time_p, force1_peak_force_p, force2_start_time_p, force2_peak_time_p, force2_peak_force_p, force2_end_time_p, max_time):
    max_force = 47
    minimum_width_p = 0.2
    minimum_distance = max_time * minimum_width_p / 2
    
    force1_peak_time = force1_peak_time_p * max_time * (1-minimum_width_p) + minimum_distance
    force1_peak_force = force1_peak_force_p * max_force 
    # Dynamic constraints
    force2_peak_time = force2_peak_time_p * max_time * (1-minimum_width_p-0.1) + minimum_distance # 0.15 to 0.85
    force2_start_time = (force2_peak_time - minimum_distance) * force2_start_time_p # 0 to 0.05 of peak time
    force2_end_time = force2_peak_time + minimum_distance + force2_end_time_p * (max_time - force2_peak_time - minimum_distance) # minimum, distance off peak time
    force2_peak_force = force2_peak_force_p * max_force

    return force1_peak_time, force1_peak_force, force2_start_time, force2_peak_time, force2_peak_force, force2_end_time


def percentage_to_actual_3(force1_peak_time_p, force1_peak_force_p, force2_start_time_p, force2_peak_time_p, force2_peak_force_p, force2_end_time_p, max_time):
    max_force = 47
    minimum_width_p = 0.2
    minimum_distance = max_time * minimum_width_p / 2
    
    force1_peak_time = force1_peak_time_p * max_time * (1-minimum_width_p) + minimum_distance
    force1_peak_force = force1_peak_force_p * max_force 
    # Dynamic constraints
    force2_peak_time = force2_peak_time_p * max_time * (1-minimum_width_p-0.1) + minimum_distance # 0.15 to 0.85
    force2_start_time = (force2_peak_time - minimum_distance) * force2_start_time_p # 0 to 0.05 of peak time
    force2_end_time = force2_peak_time + minimum_distance + force2_end_time_p * (max_time - force2_peak_time - minimum_distance) # minimum, distance off peak time
    force2_peak_force = force2_peak_force_p * max_force

    return force1_peak_time, force1_peak_force, force2_start_time, force2_peak_time, force2_peak_force, force2_end_time



def plot_force_profile(profile, save_dir, name):
    # Add percentage column and set as index
    profile["STS"] = np.linspace(0, 100, len(profile))
    profile.set_index("STS", inplace=True)
    
    fig, ax = plt.subplots(figsize=(3, 2))
    # No LaTeX style or usetex anywhere

    ax.plot(profile["force_X"], label="F_X", color="limegreen", linewidth=2)
    ax.plot(profile["force_Y"], label="F_Y", color="darkgreen", linewidth=2)

    ax.set_ylabel("Forces [N]")
    ax.set_xlabel("STS [%]")
    ax.legend()

    if save_dir is not None:
        filepath = os.path.join(save_dir, name + ".pdf")
        fig.savefig(filepath, format="pdf")
        plt.close(fig)
    else:
        plt.show()
        
# def plot_force_profile(profile, save_dir, name):
#     # Add percentage to profile and set ti as index
#     profile["STS"] = np.linspace(0, 100, len(profile))
#     profile.set_index("STS", inplace=True)

#     fig, ax = plt.subplots(figsize=(3, 2))
#     plt.style.use('science')
#     ax.plot(profile["force_X"], label=r"$F_X$", color="limegreen", linewidth=2)
#     ax.plot(profile["force_Y"], label=r"$F_Y$", color="darkgreen", linewidth=2)

#     ax.set_ylabel("Forces $[N]$")  # Correct method for an Axes object
#     ax.set_xlabel("STS $[\%]$")  # Correct method for an Axes object
#     ax.legend()  # Show legend
    
#     if save_dir is not None:
#         fig.savefig(os.path.join(save_dir, name + ".pdf"), format="pdf")
#         plt.close()
#     else:
#         plt.show()