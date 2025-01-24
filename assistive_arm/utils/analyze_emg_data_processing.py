import yaml
from pathlib import Path
import pandas as pd
import numpy as np
import jsonlines
import os
import json
import re
from copy import deepcopy


from chspy import CubicHermiteSpline

from assistive_arm.utils.data_preprocessing import read_headers


def percentage_to_actual(force1_end_time_p, force1_peak_force_p, force2_start_time_p, force2_peak_time_p, force2_peak_force_p, force2_end_time_p, max_time):
    max_force = 65
    minimum_width_p = 0.1
    minimum_distance = max_time * minimum_width_p / 2
    
    force1_end_time = minimum_width_p * max_time + force1_end_time_p * max_time * (1 - minimum_width_p)
    force1_peak_force = force1_peak_force_p * max_force * 2/3
    # Dynamic constraints
    force2_peak_time = force2_peak_time_p * max_time * 0.8 + 0.1 * max_time # 0.1 to 0.9
    force2_start_time = (force2_peak_time - minimum_distance) * force2_start_time_p # 0 to 0.05 of peak time
    force2_end_time = force2_peak_time + minimum_distance + force2_end_time_p * (max_time - force2_peak_time - minimum_distance) # 0.05 of peak to max time
    force2_peak_force = force2_peak_force_p * max_force

    return force1_end_time, force1_peak_force, force2_start_time, force2_peak_time, force2_peak_force, force2_end_time


def sanity_check(session_data, name_tag_mapping, roll_angles):
    """
    Process a session to extract optimizer scores and compare them with calculated scores.
    
    Args:
        subject_name (str): Name of the subject.
        session_data (dict): Session data containing optimizer and assisted trial information.
        name_tag_mapping (dict): Mapping of tags to their display names.
    """
    optimizer = session_data["OPTIMIZER"]
    max_time = len(roll_angles)
    
    for tag, name in name_tag_mapping.items():
        if "Assisted" in name:
            pattern = r'\b(t\d{2}|f\d{2})_(\d+)\b'
            matches = re.findall(pattern, name)
            extracted_values = {key: int(value) for key, value in matches}

            for index, row in optimizer.iterrows():
                t11, f11, t21, t22, f21, t23 = percentage_to_actual(
                                                    row["params.force1_end_time_p"],
                                                    row["params.force1_peak_force_p"],
                                                    row["params.force2_start_time_p"],
                                                    row["params.force2_peak_time_p"],
                                                    row["params.force2_peak_force_p"],
                                                    row["params.force2_end_time_p"],
                                                    max_time
                                                )

                if (extracted_values.get("t11") == int(np.round(t11)) and
                    extracted_values.get("f11") == int(np.round(f11)) and
                    extracted_values.get("t21") == int(np.round(t21)) and
                    extracted_values.get("t22") == int(np.round(t22)) and
                    extracted_values.get("t23") == int(np.round(t23)) and
                    extracted_values.get("f21") == int(np.round(f21))):

                    optimizer_score = row["target"]
                    individual_info = session_data["ASSISTED"][tag]["LOG_INFO"]
                    total_score = sum(info.iloc[3] for info in individual_info)
                    calculated_score = total_score / len(individual_info)

                    # print(f"Optimizer score: {optimizer_score}, Calculated score: {calculated_score}")

                    difference = abs(optimizer_score-calculated_score)
                    if difference > 0.00001:
                        print(f"Check {name} with Optimizer score: {optimizer_score}, Calculated score: {calculated_score}")
                        
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


def get_profile(force1_end_time, force1_peak_force, force2_start_time, force2_peak_time, force2_peak_force, force2_end_time, roll_angles):
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
    length = len(roll_angles)
    base_profile = pd.DataFrame({"force_X": np.zeros(length), "force_Y": np.zeros(length)})
    base_profile.index = roll_angles.index
    base_profile = pd.concat([roll_angles, base_profile], axis=1)

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


def detect_peak_and_crop(df):
    df = df.iloc[2000:].reset_index(drop=True)

    # Detect the peak
    min_index = df["VM_L"].idxmin()
    # Crop the data
    df = df.loc[min_index-200:].reset_index(drop=True)
    df = df.loc[:2500].reset_index(drop=True)

    return df

# Function to process EMG data
def process_emg_data(session_data, mode):
    overall_accumulator = None
    total_dfs = 0  # Keep track of the total number of DataFrames processed
    
    for tag in session_data[mode]["FIRST_TAGS"]:
        filtered_emg_cond = session_data[mode][tag]["EMG"]["Filtered"]
        
        # Initialize an accumulator for the tag mean
        tag_accumulator = None
        tag_dfs_count = 0  # Count the number of DataFrames for the tag
        
        for filtered_emg in filtered_emg_cond:
            # Apply detect_peak_and_crop
            cropped_emg = detect_peak_and_crop(filtered_emg)
            
            # Accumulate DataFrames for the tag
            if tag_accumulator is None:
                tag_accumulator = cropped_emg.copy()
            else:
                tag_accumulator += cropped_emg
            
            tag_dfs_count += 1

        # Compute the mean for the current tag
        tag_mean = tag_accumulator / tag_dfs_count
        session_data[mode][tag]["EMG"]["Mean"] = tag_mean  # Optional: store the tag mean

        # Add the tag accumulator to the overall accumulator
        if overall_accumulator is None:
            overall_accumulator = tag_accumulator.copy()
        else:
            overall_accumulator += tag_accumulator
        
        total_dfs += tag_dfs_count

    # Compute the overall mean across all tags
    overall_mean = overall_accumulator / total_dfs
    session_data[mode]["Overall_Mean"] = overall_mean  # Optional: store the overall mean
    return overall_mean