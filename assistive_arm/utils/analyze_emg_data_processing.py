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
from assistive_arm.utils.parametrize_profiles_functions import *

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
        peak_accumulator = None
        tag_dfs_count = 0  # Count the number of DataFrames for the tag
        
        for filtered_emg in filtered_emg_cond:
            # Apply detect_peak_and_crop
            cropped_emg = detect_peak_and_crop(filtered_emg)

            # Cut 0.25*2148.148 off of each side
            # crop_size = int(0.25 * 2148.148)  # Convert to integer

            # # Crop the signal
            # cropped_emg = filtered_emg.iloc[crop_size:-crop_size]
            # cropped_emg = filtered_emg
            
            # Accumulate DataFrames for the tag
            if tag_accumulator is None:
                tag_accumulator = cropped_emg.copy()
                peak_accumulator = np.max(cropped_emg, axis=0)
            else:
                tag_accumulator += cropped_emg
                peak_accumulator += np.max(cropped_emg, axis=0)
            
            tag_dfs_count += 1

        # Compute the mean for the current tag
        tag_mean = tag_accumulator / tag_dfs_count
        session_data[mode][tag]["EMG"]["Mean"] = tag_mean  # Store the tag mean

        # Compute the mean peak value for the current tag
        peak_mean = peak_accumulator / tag_dfs_count
        session_data[mode][tag]["EMG"]["Peak"] = peak_mean 

        # Add the tag accumulator to the overall accumulator
        if overall_accumulator is None:
            overall_accumulator = tag_accumulator.copy()
        else:
            overall_accumulator += tag_accumulator
        
        total_dfs += tag_dfs_count

    # Compute the overall mean across all tags
    overall_mean = overall_accumulator / total_dfs
    session_data[mode]["Overall_Mean"] = overall_mean